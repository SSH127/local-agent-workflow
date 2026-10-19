"""Regression tests for write protection: snapshots, version checks, rollback.

Reproduces the two data-loss incidents (a 640-line file replaced with
``PLACEHOLDER``, a 506-line file truncated) and verifies that every modifying
tool now has a recoverable snapshot, a stale-write guard and an explicit
rollback that never clobbers newer external edits.
"""

import json
import os
from pathlib import Path

import pytest

from app import config, write_guard
from app.project_context import ProjectContext
from app.tools.base import ReadTracker, ToolContext
from app.tools.registry import get_tool


@pytest.fixture
def ws(tmp_path, monkeypatch):
    backup_dir = tmp_path.parent / f"{tmp_path.name}-backups"
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "WRITE_BACKUP_DIR", backup_dir)
    return tmp_path


def _call(name, args, ws, *, project=None, tracker=None, approval=None, run_id="run-w"):
    tool = get_tool(name)
    assert tool is not None
    ctx = ToolContext(
        run_id=run_id,
        workspace=ws,
        project=project,
        read_tracker=tracker,
        approval_check=approval,
    )
    return tool.handler(args, ctx)


def _manifest_of(result: dict) -> dict:
    manifest_path = result["data"].get("manifest")
    assert manifest_path, result
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The reported data-loss case
# ---------------------------------------------------------------------------


def test_placeholder_overwrite_is_refused_and_nothing_is_recorded(ws):
    original = "".join(f"line_{i} = {i}\n" for i in range(640))
    target = ws / "registry.py"
    target.write_text(original, encoding="utf-8")

    result = _call("write_file", {"path": "registry.py", "content": "PLACEHOLDER"}, ws)

    assert result["ok"] is False
    assert "拒绝覆盖" in result["output"]
    assert target.read_text(encoding="utf-8") == original
    # No transaction is created for a write that never started.
    assert not (write_guard.backup_root(ws)).exists() or not any(
        (write_guard.backup_root(ws)).rglob("manifest.json")
    )


# ---------------------------------------------------------------------------
# Snapshot + explicit rollback
# ---------------------------------------------------------------------------


def test_edit_snapshots_original_and_rollback_restores_exact_content(ws):
    original = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    target = ws / "mod.py"
    target.write_text(original, encoding="utf-8")

    edited = _call(
        "edit_file",
        {"path": "mod.py", "old_string": "    return 1", "new_string": "    return 99"},
        ws,
    )
    assert edited["ok"] is True
    tx_id = edited["data"]["tx_id"]
    manifest = _manifest_of(edited)
    assert manifest["run_id"] == "run-w"
    assert manifest["status"] == "applied"
    assert manifest["before_sha256"] != manifest["after_sha256"]
    assert Path(manifest["snapshot"]).exists()

    restored = _call(
        "rollback_file", {"tx_id": tx_id}, ws, approval=lambda n, a: True
    )
    assert restored["ok"] is True
    assert restored["data"]["rollback_status"] == "restored_original"
    assert target.read_text(encoding="utf-8") == original
    manifest = json.loads(
        Path(edited["data"]["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "rolled_back"
    assert manifest["rolled_back_at"]


def test_rollback_removes_a_file_that_write_created(ws):
    created = _call("write_file", {"path": "new.py", "content": "x = 1\n"}, ws)
    assert created["ok"] is True
    assert (ws / "new.py").exists()

    result = _call(
        "rollback_file", {"tx_id": created["data"]["tx_id"]}, ws,
        approval=lambda n, a: True,
    )
    assert result["ok"] is True
    assert result["data"]["rollback_status"] == "deleted_created"
    assert not (ws / "new.py").exists()


def test_rollback_is_idempotent(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    first = _call("write_file", {"path": "a.txt", "content": "two\n"}, ws)
    tx_id = first["data"]["tx_id"]

    assert _call("rollback_file", {"tx_id": tx_id}, ws, approval=lambda n, a: True)["ok"]
    again = _call("rollback_file", {"tx_id": tx_id}, ws, approval=lambda n, a: True)
    assert again["ok"] is True
    assert again["data"]["already"] is True
    assert target.read_text(encoding="utf-8") == "one\n"


def test_rollback_refuses_to_clobber_newer_external_edit(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    written = _call("write_file", {"path": "a.txt", "content": "two\n"}, ws)
    tx_id = written["data"]["tx_id"]

    target.write_text("three (external)\n", encoding="utf-8")
    refused = _call("rollback_file", {"tx_id": tx_id}, ws, approval=lambda n, a: True)

    assert refused["ok"] is False
    assert "修改" in refused["output"]
    assert target.read_text(encoding="utf-8") == "three (external)\n"


def test_forced_rollback_keeps_the_conflicting_content_as_evidence(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    written = _call("write_file", {"path": "a.txt", "content": "two\n"}, ws)
    tx_id = written["data"]["tx_id"]
    target.write_text("three\n", encoding="utf-8")

    forced = _call(
        "rollback_file", {"tx_id": tx_id, "force": True}, ws,
        approval=lambda n, a: True,
    )
    assert forced["ok"] is True
    assert forced["data"]["forced"] is True
    assert target.read_text(encoding="utf-8") == "one\n"
    tx_dir = Path(written["data"]["manifest"]).parent
    assert (tx_dir / "conflict.bin").read_text(encoding="utf-8") == "three\n"
    assert (tx_dir / "superseded.bin").exists()


def test_rollback_rejects_a_transaction_from_another_run(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    written = _call("write_file", {"path": "a.txt", "content": "two\n"}, ws, run_id="run-a")
    denied = _call(
        "rollback_file", {"tx_id": written["data"]["tx_id"]}, ws,
        approval=lambda n, a: True, run_id="run-b",
    )
    assert denied["ok"] is False
    assert target.read_text(encoding="utf-8") == "two\n"


def test_rollback_requires_approval_even_at_handler_level(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    written = _call("write_file", {"path": "a.txt", "content": "two\n"}, ws)

    denied = _call("rollback_file", {"tx_id": written["data"]["tx_id"]}, ws)
    assert denied["ok"] is False
    assert "批准" in denied["output"]
    assert target.read_text(encoding="utf-8") == "two\n"


# ---------------------------------------------------------------------------
# Version / concurrency checks
# ---------------------------------------------------------------------------


def test_stale_write_after_external_edit_is_refused(ws):
    target = ws / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")
    tracker = ReadTracker()

    read = _call("read_file", {"path": "mod.py"}, ws, tracker=tracker)
    assert read["ok"] is True
    target.write_text("value = 2  # external\n", encoding="utf-8")

    refused = _call(
        "write_file", {"path": "mod.py", "content": "value = 3\n"}, ws, tracker=tracker
    )
    assert refused["ok"] is False
    assert "外部修改" in refused["output"]
    assert target.read_text(encoding="utf-8") == "value = 2  # external\n"


def test_external_edit_during_prepare_aborts_the_commit(ws, monkeypatch):
    target = ws / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")

    real_prepare = write_guard.prepare

    def racing_prepare(state, **kwargs):
        tx = real_prepare(state, **kwargs)
        state.abs_path.write_text("value = 999  # raced\n", encoding="utf-8")
        return tx

    monkeypatch.setattr(write_guard, "prepare", racing_prepare)
    result = _call("write_file", {"path": "mod.py", "content": "value = 3\n"}, ws)

    assert result["ok"] is False
    assert target.read_text(encoding="utf-8") == "value = 999  # raced\n"


def test_explicit_expected_sha256_is_enforced(ws):
    target = ws / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")
    good = write_guard.content_hash("value = 1\n")

    ok = _call(
        "write_file",
        {"path": "mod.py", "content": "value = 2\n", "expected_sha256": good},
        ws,
    )
    assert ok["ok"] is True

    stale = _call(
        "write_file",
        {"path": "mod.py", "content": "value = 3\n", "expected_sha256": good},
        ws,
    )
    assert stale["ok"] is False
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_multiple_edits_to_one_file_all_succeed(ws):
    target = ws / "mod.py"
    target.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    tracker = ReadTracker()
    _call("read_file", {"path": "mod.py"}, ws, tracker=tracker)

    for old, new in (("a = 1", "a = 10"), ("b = 2", "b = 20"), ("c = 3", "c = 30")):
        result = _call(
            "edit_file",
            {"path": "mod.py", "old_string": old, "new_string": new},
            ws,
            tracker=tracker,
        )
        assert result["ok"] is True, result

    assert target.read_text(encoding="utf-8") == "a = 10\nb = 20\nc = 30\n"


# ---------------------------------------------------------------------------
# Failure / verification semantics
# ---------------------------------------------------------------------------


def test_write_failure_leaves_file_intact_and_records_the_transaction(ws, monkeypatch):
    target = ws / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(write_guard, "_atomic_write", boom)
    result = _call("write_file", {"path": "mod.py", "content": "value = 2\n"}, ws)

    assert result["ok"] is False
    assert "原文件未被修改" in result["output"]
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    manifests = list(write_guard.backup_root(ws).rglob("manifest.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "disk full" in data["error"]


def test_syntax_error_is_reported_but_not_auto_rolled_back(ws):
    result = _call(
        "write_file", {"path": "broken.py", "content": "def f(:\n    pass\n"}, ws
    )
    assert result["ok"] is True
    assert result["data"]["syntax_ok"] is False
    assert "语法检查未通过" in result["output"]
    # The agent must be able to see and repair its own change.
    assert (ws / "broken.py").read_text(encoding="utf-8") == "def f(:\n    pass\n"


def test_failing_test_does_not_roll_back_the_change(ws):
    (ws / "test_fail.py").write_text("def test_f():\n    assert False\n", encoding="utf-8")
    written = _call("write_file", {"path": "app.py", "content": "x = 1\n"}, ws)
    assert written["ok"] is True

    failed = _call(
        "run_project_command",
        {"program": "python", "args": ["-m", "pytest", "test_fail.py", "-q"], "cwd": "."},
        ws,
    )
    assert failed["ok"] is False

    # A failing test is normal work, not a write failure: the file stays changed
    # and the transaction remains `applied` for an explicit, audited rollback.
    assert (ws / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    manifest = _manifest_of(written)
    assert manifest["status"] == "applied"


# ---------------------------------------------------------------------------
# Snapshot location / secrets
# ---------------------------------------------------------------------------


def test_snapshot_directory_is_outside_the_workspace_and_unreadable(ws, monkeypatch):
    monkeypatch.setattr(config, "WRITE_BACKUP_DIR", ws / "backups")
    (ws / "a.txt").write_text("zero\n", encoding="utf-8")
    result = _call("write_file", {"path": "a.txt", "content": "one\n"}, ws)
    assert result["ok"] is True
    snapshot = Path(result["data"]["snapshot"])
    assert not snapshot.is_relative_to(ws.resolve())

    # The backup root must not be reachable by the agent's own file tools.
    read = _call("read_file", {"path": str(snapshot)}, ws)
    assert read["ok"] is False


def test_protected_file_snapshot_is_outside_workspace_and_not_exposed(ws):
    target = ws / ".env"
    target.write_text("API_KEY=sk-super-secret\n", encoding="utf-8")
    args = {"path": ".env", "content": "API_KEY=sk-rotated\n"}
    result = _call("write_file", args, ws, approval=lambda n, a: True)

    assert result["ok"] is True
    manifest = _manifest_of(result)
    assert manifest["protected"] is True
    assert "sk-super-secret" not in json.dumps(result, ensure_ascii=False)
    snapshot = Path(manifest["snapshot"])
    assert snapshot.read_text(encoding="utf-8") == "API_KEY=sk-super-secret\n"
    assert not snapshot.is_relative_to(ws.resolve())
    assert _call("read_file", {"path": str(snapshot)}, ws)["ok"] is False


def test_protected_file_write_without_approval_is_refused(ws):
    target = ws / ".env"
    target.write_text("API_KEY=1\n", encoding="utf-8")
    result = _call(
        "write_file", {"path": ".env", "content": "API_KEY=2\n"}, ws,
        approval=lambda n, a: False,
    )
    assert result["ok"] is False
    assert target.read_text(encoding="utf-8") == "API_KEY=1\n"


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def _backdate(manifest_path: Path, minutes: int = 10) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    from datetime import UTC, datetime, timedelta

    data["started_at"] = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
    manifest_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_scan_reports_a_write_that_never_landed_as_abandoned(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    state = write_guard.inspect(
        target, "a.txt", ws.resolve(), protected=False
    )
    tx = write_guard.prepare(state, run_id="run-crash", tool="write_file",
                             workspace=ws.resolve())
    _backdate(tx.manifest_path)

    findings = write_guard.scan_incomplete(ws, min_age_seconds=0)
    assert len(findings) == 1
    assert findings[0]["verdict"] == write_guard.STATUS_ABANDONED
    # Nothing is auto-restored: the file is unchanged and the tx is still there.
    assert target.read_text(encoding="utf-8") == "one\n"


def test_scan_flags_an_unrecorded_commit_for_review(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    state = write_guard.inspect(target, "a.txt", ws.resolve(), protected=False)
    tx = write_guard.prepare(state, run_id="run-crash", tool="write_file",
                             workspace=ws.resolve())
    target.write_text("two\n", encoding="utf-8")  # landed, manifest not updated
    _backdate(tx.manifest_path)

    findings = write_guard.scan_incomplete(ws, min_age_seconds=0)
    assert len(findings) == 1
    assert findings[0]["verdict"] == write_guard.STATUS_NEEDS_REVIEW


def test_fresh_transactions_are_not_reported_as_incomplete(ws):
    target = ws / "a.txt"
    target.write_text("one\n", encoding="utf-8")
    state = write_guard.inspect(target, "a.txt", ws.resolve(), protected=False)
    write_guard.prepare(state, run_id="run-live", tool="write_file", workspace=ws.resolve())
    assert write_guard.scan_incomplete(ws, min_age_seconds=60) == []


# ---------------------------------------------------------------------------
# Cache / symbol index synchronisation
# ---------------------------------------------------------------------------


def test_cache_and_symbol_index_follow_edit_and_rollback(ws):
    target = ws / "mod.py"
    original = "def alpha():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    project = ProjectContext(ws)

    edited = _call(
        "edit_file",
        {
            "path": "mod.py",
            "old_string": "def alpha():\n    return 1",
            "new_string": "def beta():\n    return 2",
        },
        ws,
        project=project,
    )
    assert edited["ok"] is True
    assert "beta" in project.get_file("mod.py").content
    assert any(s["name"] == "beta" for s in project.query_symbols("beta"))
    assert not project.query_symbols("alpha")

    rolled = _call(
        "rollback_file", {"tx_id": edited["data"]["tx_id"]}, ws,
        project=project, approval=lambda n, a: True,
    )
    assert rolled["ok"] is True
    assert project.get_file("mod.py").content == original
    assert any(s["name"] == "alpha" for s in project.query_symbols("alpha"))
    assert not project.query_symbols("beta")
    assert target.read_text(encoding="utf-8") == original


def test_newline_style_is_preserved_by_edits(ws):
    target = ws / "crlf.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\n")
    result = _call(
        "edit_file", {"path": "crlf.py", "old_string": "a = 1", "new_string": "a = 9"}, ws
    )
    assert result["ok"] is True
    assert target.read_bytes() == b"a = 9\r\nb = 2\r\n"


def test_write_file_rejects_symlink_escape(ws):
    outside = ws.parent / "outside-secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = ws / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not available on this platform")
    result = _call("write_file", {"path": "link.txt", "content": "overwritten\n"}, ws)
    assert result["ok"] is False
    assert outside.read_text(encoding="utf-8") == "secret\n"
