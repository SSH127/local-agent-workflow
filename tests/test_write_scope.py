"""Contract write scope: matching, tool-layer enforcement, evidence, defaults."""
from __future__ import annotations

import pytest

from app import config
from app.agents.agent_executor import run_agent
from app.engine.approvals import ApprovalDecision
from app.tools import tool_schemas
from app.tools.base import ToolContext
from app.tools.registry import get_tool
from app.tools.security import classify_command
from app.write_scope import (
    ScopeError,
    WriteScope,
    normalize_scope,
    path_in_scope,
    scopes_overlap,
)
from tests.conftest import FakeLLM, final_answer, tool_call

# ---------------------------------------------------------------------------
# Pattern semantics
# ---------------------------------------------------------------------------


def test_literal_pattern_covers_path_and_subtree():
    assert path_in_scope(["app"], "app/main.py")
    assert path_in_scope(["app/main.py"], "app/main.py")
    assert not path_in_scope(["app"], "apps/x.py")
    assert path_in_scope(["src/**/*.py"], "src/a/b/c.py")
    assert path_in_scope(["tests/test_*.py"], "tests/test_x.py")
    assert not path_in_scope(["tests/test_*.py"], "tests/sub/test_x.py")
    assert not path_in_scope([], "a.py")  # empty scope = read-only


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:/x", "../up", "a/../../b", ""])
def test_scope_rejects_absolute_and_parent_paths(bad):
    with pytest.raises(ScopeError):
        normalize_scope([bad])


def test_normalize_scope_unifies_separators():
    assert normalize_scope(["app\\engine\\", "./tests"]) == ["app/engine", "tests"]


def test_overlap_is_conservative():
    assert scopes_overlap(["app/**"], ["app/main.py"])
    assert scopes_overlap(["**"], ["x"])
    assert scopes_overlap(["src/*.py"], ["src/a.py"])
    assert not scopes_overlap(["src/*.py"], ["src/*.ts"])
    assert not scopes_overlap(["app"], ["tests"])
    assert not scopes_overlap([], ["**"])  # read-only never overlaps


# ---------------------------------------------------------------------------
# Tool layer enforcement
# ---------------------------------------------------------------------------


def _ctx(tmp_path, scope, sid="2"):
    return ToolContext(run_id="run-scope", workspace=tmp_path,
                       approval_check=lambda name, args: True,
                       write_scope=WriteScope(scope, subtask_id=sid))


def test_write_inside_scope_succeeds_outside_rejected_with_evidence(tmp_path):
    ctx = _ctx(tmp_path, ["src/**"])
    write = get_tool("write_file").handler
    ok = write({"path": "src/a.py", "content": "x = 1\n"}, ctx)
    assert ok.get("ok", True) and not ok.get("error"), ok
    assert (tmp_path / "src/a.py").read_text(encoding="utf-8") == "x = 1\n"
    bad = write({"path": "other.py", "content": "y = 2\n"}, ctx)
    assert "write_scope" in str(bad) or "写入范围" in str(bad)
    assert not (tmp_path / "other.py").exists()
    violations = ctx.write_scope.snapshot_violations()
    assert len(violations) == 1
    v = violations[0]
    assert v["type"] == "scope_violation" and v["path"] == "other.py"
    assert v["tool"] == "write_file" and v["subtask_id"] == "2" and v["scope"] == ["src/**"]


def test_edit_outside_scope_rejected(tmp_path):
    (tmp_path / "keep.py").write_text("a = 1\n", encoding="utf-8")
    ctx = _ctx(tmp_path, ["src"])
    res = get_tool("edit_file").handler(
        {"path": "keep.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
    assert "写入范围" in str(res) or "write_scope" in str(res)
    assert (tmp_path / "keep.py").read_text(encoding="utf-8") == "a = 1\n"
    assert ctx.write_scope.snapshot_violations()[0]["tool"] == "edit_file"


def test_empty_scope_is_read_only(tmp_path):
    ctx = _ctx(tmp_path, [])
    assert ctx.write_scope.read_only
    res = get_tool("write_file").handler({"path": "a.py", "content": "x"}, ctx)
    assert "只读" in str(res) or "read" in str(res).lower()
    assert not (tmp_path / "a.py").exists()


def test_rollback_outside_scope_rejected(tmp_path):
    wide = _ctx(tmp_path, ["**"])
    res = get_tool("write_file").handler({"path": "lib/x.py", "content": "v = 1\n"}, wide)
    tx_id = res.get("tx_id") or (res.get("data") or {}).get("tx_id")
    assert tx_id, res
    narrow = _ctx(tmp_path, ["src/**"])
    out = get_tool("rollback_file").handler({"tx_id": tx_id}, narrow)
    assert "写入范围" in str(out) or "write_scope" in str(out)
    assert (tmp_path / "lib/x.py").exists()
    assert narrow.write_scope.snapshot_violations()[0]["path"] == "lib/x.py"


def test_commands_declare_their_writes(tmp_path):
    d = classify_command("ruff", ["check", "--fix", "tests"], ".", tmp_path)
    assert "tests" in [w.rstrip("/") for w in d.writes]
    d = classify_command("ruff", ["check", "app"], ".", tmp_path)
    assert d.writes == ()
    d = classify_command("npm", ["install"], "frontend", tmp_path)
    assert "frontend/package.json" in d.writes
    assert any(w.startswith("frontend/node_modules") for w in d.writes)


def test_ruff_fix_outside_scope_rejected_before_running(tmp_path):
    ctx = _ctx(tmp_path, ["src/**"])
    res = get_tool("run_project_command").handler(
        {"program": "ruff", "args": ["check", "--fix", "tests"], "cwd": "."}, ctx)
    assert "写入范围" in str(res) or "write_scope" in str(res)
    v = ctx.write_scope.snapshot_violations()
    assert v and v[0]["tool"] == "run_project_command"


def test_ruff_fix_inside_scope_passes_scope_check(tmp_path):
    scope = WriteScope(["src/**"], subtask_id="1")
    assert scope.check_command_writes("run_project_command", ["src/"]) is None
    assert scope.check_command_writes("run_project_command", ["src/a.py"]) is None
    # A directory write is only inside the scope if the whole directory is.
    assert scope.check_command_writes("run_project_command", ["./"]) is not None


# ---------------------------------------------------------------------------
# Executor: scope precheck happens before approval, rejection is a tool error
# ---------------------------------------------------------------------------


class _Rec:
    def __init__(self):
        self.steps, self.approvals = [], []

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, *a, **k):
        return None

    def set_state(self, state):
        return None

    def request_approval(self, run_id, tool_name, args, risk_reason):
        self.approvals.append(tool_name)
        return ApprovalDecision(approval_id="a", approved=True)


def test_executor_rejects_out_of_scope_write_without_asking(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    rec = _Rec()
    scope = WriteScope(["src/**"], subtask_id="1")
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    run_agent("run-x", "测试任务", llm, tool_schemas(), rec, workspace=str(tmp_path),
              write_scope=scope)
    assert not (tmp_path / ".env").exists()
    assert rec.approvals == []  # never worth asking the user about
    failed = [s for s in rec.steps if s["kind"] == "tool" and s["status"] == "failed"]
    assert failed and "write scope" in (failed[0].get("output") or "") + str(failed[0].get("error"))
    assert scope.snapshot_violations()[0]["path"] == ".env"
