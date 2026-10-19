"""Regression tests for surgical editing and the destructive-overwrite guard."""

import pytest

from app import config
from app.project_context import ProjectContext
from app.tools.base import ToolContext
from app.tools.registry import get_tool


def _call(name: str, args: dict, workspace, project=None, approval=None):
    tool = get_tool(name)
    assert tool is not None
    ctx = ToolContext(
        run_id="run-edit",
        workspace=workspace,
        project=project,
        approval_check=approval,
    )
    return tool.handler(args, ctx)


def test_edit_file_replaces_only_the_target_snippet(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "mod.py"
    original = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    target.write_text(original, encoding="utf-8")

    result = _call(
        "edit_file",
        {"path": "mod.py", "old_string": "    return 1", "new_string": "    return 100"},
        tmp_path,
    )

    assert result["ok"] is True
    assert result["data"]["replacements"] == 1
    assert target.read_text(encoding="utf-8") == "def a():\n    return 100\n\ndef b():\n    return 2\n"


def test_edit_file_refuses_missing_old_string_and_leaves_file_intact(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    result = _call(
        "edit_file",
        {"path": "mod.py", "old_string": "x = 2", "new_string": "x = 3"},
        tmp_path,
    )

    assert result["ok"] is False
    assert "未找到" in result["output"]
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_file_refuses_ambiguous_match_unless_replace_all(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "mod.py"
    target.write_text("a = 1\nb = 1\n", encoding="utf-8")

    ambiguous = _call(
        "edit_file", {"path": "mod.py", "old_string": "= 1", "new_string": "= 9"}, tmp_path,
    )
    assert ambiguous["ok"] is False
    assert "2 次" in ambiguous["output"]

    replaced = _call(
        "edit_file",
        {"path": "mod.py", "old_string": "= 1", "new_string": "= 9", "replace_all": True},
        tmp_path,
    )
    assert replaced["ok"] is True
    assert replaced["data"]["replacements"] == 2
    assert target.read_text(encoding="utf-8") == "a = 9\nb = 9\n"


def test_edit_file_rejects_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    result = _call(
        "edit_file",
        {"path": "../outside.py", "old_string": "a", "new_string": "b"},
        tmp_path,
    )
    assert result["ok"] is False
    assert "路径越界" in result["output"]


def test_edit_file_protected_file_requires_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / ".env").write_text("KEY=old\n", encoding="utf-8")

    tool = get_tool("edit_file")
    assert tool is not None
    assert tool.requires_approval({"path": ".env"}, str(tmp_path)) is True

    denied = _call(
        "edit_file", {"path": ".env", "old_string": "old", "new_string": "new"}, tmp_path,
    )
    assert denied["ok"] is False
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "KEY=old\n"

    allowed = _call(
        "edit_file",
        {"path": ".env", "old_string": "old", "new_string": "new"},
        tmp_path,
        approval=lambda name, args: True,
    )
    assert allowed["ok"] is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "KEY=new\n"


def test_edit_file_updates_project_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "mod.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONTEXT_CACHE_DB", tmp_path / "cache.sqlite3")
    project = ProjectContext(tmp_path, db_path=config.CONTEXT_CACHE_DB)

    before = project.get_file("mod.py")
    assert "return 1" in before.content

    result = _call(
        "edit_file",
        {"path": "mod.py", "old_string": "return 1", "new_string": "return 2"},
        tmp_path,
        project=project,
    )
    assert result["ok"] is True

    after = project.get_file("mod.py")
    assert "return 2" in after.content
    assert "return 1" not in after.content
    assert any(s["name"] == "a" for s in project.query_symbols("a"))


def test_write_file_refuses_catastrophic_shrink(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "big.py"
    original = "".join(f"line_{i} = {i}\n" for i in range(200))
    target.write_text(original, encoding="utf-8")

    refused = _call("write_file", {"path": "big.py", "content": "PLACEHOLDER"}, tmp_path)

    assert refused["ok"] is False
    assert "拒绝覆盖" in refused["output"]
    assert "edit_file" in refused["output"]
    assert target.read_text(encoding="utf-8") == original


def test_write_file_allows_shrink_when_explicitly_confirmed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "big.py"
    target.write_text("x = 1\n" * 200, encoding="utf-8")

    # A deliberate whole-file shrink is allowed only with an explicit approval
    # bound to this exact call (the executor provides it via the approval gate).
    args = {"path": "big.py", "content": "x = 2\n", "allow_truncate": True}
    result = _call("write_file", args, tmp_path, approval=lambda name, a: a == args)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "x = 2\n"
    # Without that approval the same call must be refused and the file kept.
    result = _call(
        "write_file",
        {"path": "big.py", "content": "x = 3\n", "allow_truncate": True},
        tmp_path,
        approval=lambda name, a: False,
    )
    assert result["ok"] is False
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_write_file_guard_does_not_block_normal_rewrites(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "small.py"
    target.write_text("a = 1\n", encoding="utf-8")

    result = _call("write_file", {"path": "small.py", "content": "b = 2\n"}, tmp_path)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "b = 2\n"


@pytest.mark.parametrize("name", ["write_file", "edit_file"])
def test_edit_and_write_are_modify_phase(name):
    from app.agents.agent_executor import classify_phase

    assert classify_phase([name]) == "modify"


def test_edit_file_preserves_lf_line_endings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "lf.py"
    target.write_bytes(b"a = 1\nb = 2\nc = 3\n")

    result = _call(
        "edit_file", {"path": "lf.py", "old_string": "b = 2", "new_string": "b = 20"}, tmp_path,
    )

    assert result["ok"] is True
    assert target.read_bytes() == b"a = 1\nb = 20\nc = 3\n"


def test_edit_file_preserves_crlf_line_endings(monkeypatch, tmp_path):
    """A one-line edit on a CRLF file must not rewrite every line ending."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "crlf.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")

    result = _call(
        "edit_file", {"path": "crlf.py", "old_string": "b = 2", "new_string": "b = 20"}, tmp_path,
    )

    assert result["ok"] is True
    assert target.read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"


def test_edit_file_accepts_lf_old_string_in_crlf_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    target = tmp_path / "crlf.py"
    target.write_bytes(b"def a():\r\n    return 1\r\n")

    result = _call(
        "edit_file",
        {"path": "crlf.py", "old_string": "def a():\n    return 1", "new_string": "def a():\n    return 2"},
        tmp_path,
    )

    assert result["ok"] is True
    assert target.read_bytes() == b"def a():\r\n    return 2\r\n"
