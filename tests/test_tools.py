import json

import pytest

from app import config
from app.tools import security
from app.tools.base import ToolContext
from app.tools.registry import TOOLS, get_tool
from app.tools.security import SecurityError


def _ctx() -> ToolContext:
    return ToolContext(run_id="run-test")


def _call(name: str, args: dict):
    tool = get_tool(name)
    assert tool is not None
    return tool.handler(args, _ctx())


def test_path_escape_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    with pytest.raises(SecurityError):
        security.resolve_workspace_path("../outside.txt")
    with pytest.raises(SecurityError):
        security.resolve_workspace_path("/etc/passwd")


def test_path_inside_workspace_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    resolved = security.resolve_workspace_path("src/main.py")
    assert resolved.is_relative_to(tmp_path)


def test_redact_secrets():
    text = "DEEPSEEK_API_KEY=sk-abcdef1234567890 and password=secret123"
    out = security.redact_text(text)
    assert "sk-abcdef1234567890" not in out
    assert "secret123" not in out

    struct = {"headers": {"Authorization": "Bearer abcdef1234567890"}, "ok": True}
    out = security.redact_kv(struct)
    assert "abcdef1234567890" not in json.dumps(out)


@pytest.mark.parametrize(
    "program,args,expect_error",
    [
        ("bash", [], True),                      # not in allowlist
        ("python", ["-c", "print(1)"], True),    # arbitrary code
        ("python", ["--version"], False),        # harmless
        ("python", ["-m", "pytest", "-q"], False),
        ("python", ["-m", "os"], True),          # module not allowed
        ("npm", ["run", "build"], False),
        ("npm", ["install", "x"], False),        # allowed but requires approval
        ("pytest", ["-q"], False),
        ("rm", ["-rf", "/"], True),              # not in allowlist
        ("pytest", [";", "evil"], True),         # metacharacter
        ("npm", ["run", "build", ">", "out.txt"], True),  # redirection
    ],
)
def test_command_validation(program, args, expect_error):
    if expect_error:
        with pytest.raises(SecurityError):
            security.validate_command(program, args, ".")
    else:
        _, requires = security.validate_command(program, args, ".")
        assert isinstance(requires, bool)


def test_approval_classification():
    assert security.requires_approval("pip", ["install", "requests"], ".") is True
    assert security.requires_approval("npm", ["install", "x"], ".") is True
    assert security.requires_approval("npm", ["run", "build"], ".") is False
    # Bare ``pytest`` performs implicit whole-repo collection (config, conftest,
    # every test module), so its execution scope cannot be statically bounded:
    # it needs approval, while an explicit test-file target does not.
    assert security.requires_approval("pytest", ["-q"], ".") is True
    assert security.requires_approval("pytest", ["tests/test_tools.py", "-q"], ".") is False


def test_list_files(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b").mkdir()
    result = _call("list_files", {})
    assert result["ok"] is True
    names = {e["path"] for e in result["data"]["entries"]}
    assert "a.py" in names and "b" in names


def test_read_file_and_protected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.py").write_text("line1\nline2\n", encoding="utf-8")
    ok = _call("read_file", {"path": "a.py"})
    assert ok["ok"] is True and "line1" in ok["data"]["content"]

    # .env is protected.
    (tmp_path / ".env").write_text("KEY=value", encoding="utf-8")
    denied = _call("read_file", {"path": ".env"})
    assert denied["ok"] is False


def test_write_file_and_protected_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    result = _call("write_file", {"path": "new.py", "content": "print(1)"})
    assert result["ok"] is True
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)"

    # Overwriting a protected file without approval is refused.
    tool = get_tool("write_file")
    assert tool is not None
    assert tool.requires_approval({"path": ".env"}) is True
    denied = _call("write_file", {"path": ".env", "content": "x"})
    assert denied["ok"] is False


def test_search_code(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "x.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    result = _call("search_code", {"query": "hello", "glob": "*.py"})
    assert result["ok"] is True
    assert any("x.py" in m["path"] for m in result["data"]["matches"])


def test_run_project_command_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    result = _call("run_project_command", {"program": "python", "args": ["--version"], "cwd": "."})
    assert result["ok"] is True
    assert result["data"]["exit_code"] == 0


def test_check_port(monkeypatch):
    # An unused high port reports not-listening. The global network blocker
    # forbids real sockets, so mock the connection attempt with an explicit
    # "connection refused" result (which _check_port maps to "not listening").
    import socket

    def _refused(*args, **kwargs):
        raise ConnectionRefusedError("mock: no listener")

    monkeypatch.setattr(socket, "create_connection", _refused)
    result = _call("check_port", {"port": 59999, "host": "127.0.0.1"})
    assert result["ok"] is True
    assert result["data"]["listening"] is False


def test_check_http_local_only():
    result = _call("check_http", {"url": "https://example.com"})
    assert result["ok"] is False


def test_all_tools_have_schema():
    for tool in TOOLS:
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert "properties" in schema["function"]["parameters"]


def test_symlink_escape_rejected(monkeypatch, tmp_path):
    """A symlink inside the workspace pointing outside must be rejected."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not available on this platform")
    with pytest.raises(SecurityError):
        security.resolve_workspace_path("link.txt")


def test_search_code_excludes_protected_files(monkeypatch, tmp_path):
    """Search must never leak the contents of protected files (e.g. .env)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / ".env").write_text("API_KEY=sk-supersecret", encoding="utf-8")
    (tmp_path / "x.py").write_text("API_KEY=sk-supersecret", encoding="utf-8")
    result = _call("search_code", {"query": "sk-supersecret"})
    assert result["ok"] is True
    matches = result["data"]["matches"]
    assert all(m["path"] != ".env" for m in matches)
    assert any("x.py" in m["path"] for m in matches)


def test_run_project_command_timeout(monkeypatch, tmp_path):
    """A command exceeding its timeout is killed and reported as failed."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "test_slow.py").write_text(
        "import time\n"
        "def test_slow():\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )
    result = _call("run_project_command", {
        "program": "python", "args": ["-m", "pytest", "test_slow.py", "-q"],
        "cwd": ".", "timeout_seconds": 2,
    })
    assert result["ok"] is False
    assert "超时" in result["output"]


def test_run_project_command_nonzero_exit(monkeypatch, tmp_path):
    """A non-zero exit code is reported with the real exit code."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "test_fail.py").write_text("def test_f():\n    assert False\n", encoding="utf-8")
    result = _call("run_project_command", {
        "program": "python", "args": ["-m", "pytest", "test_fail.py", "-q"],
        "cwd": ".",
    })
    assert result["ok"] is False
    assert result["data"]["exit_code"] != 0


# ---------------------------------------------------------------------------
# Task 1 regression tests: read-only verification vs file-mutating commands.
# ---------------------------------------------------------------------------

def test_ruff_check_is_verification_not_mutation():
    """``ruff check`` (without --fix) is a read-only verification command."""
    _, requires = security.validate_command("ruff", ["check", "."], ".")
    assert requires is False
    assert security.is_verification_command("ruff", ["check", "."]) is True
    assert security.is_mutating_command("ruff", ["check", "."]) is False


def test_ruff_format_requires_approval_and_is_mutation():
    """``ruff format`` mutates files: it requires approval and is never a
    verification."""
    _, requires = security.validate_command("ruff", ["format", "."], ".")
    assert requires is True
    assert security.is_mutating_command("ruff", ["format", "."]) is True
    assert security.is_verification_command("ruff", ["format", "."]) is False


def test_ruff_check_fix_is_mutation():
    """``ruff check --fix`` applies fixes (mutates) and is not a verification."""
    _, requires = security.validate_command("ruff", ["check", "--fix", "."], ".")
    assert requires is True
    assert security.is_mutating_command("ruff", ["check", "--fix", "."]) is True
    assert security.is_verification_command("ruff", ["check", "--fix", "."]) is False


def test_wrapped_mutating_commands_cannot_bypass():
    """``uv run ruff format`` and ``python -m ruff format`` cannot bypass the
    mutation policy: both require approval and are not verifications."""
    for program, args in [
        ("uv", ["run", "ruff", "format", "."]),
        ("python", ["-m", "ruff", "format", "."]),
    ]:
        _, requires = security.validate_command(program, args, ".")
        assert requires is True, (program, args)
        assert security.is_mutating_command(program, args) is True
        assert security.is_verification_command(program, args) is False


def test_uv_run_unknown_target_rejected():
    """``uv run <unlisted>`` is rejected rather than widened to arbitrary shell."""
    with pytest.raises(SecurityError):
        security.validate_command("uv", ["run", "bash", "-c", "id"], ".")


def test_run_project_command_requires_approval_classification():
    """The real tool's requires_approval reflects the mutation classification."""
    tool = get_tool("run_project_command")
    assert tool is not None
    assert tool.requires_approval(
        {"program": "ruff", "args": ["format", "."], "cwd": "."}
    ) is True
    assert tool.requires_approval(
        {"program": "ruff", "args": ["check", "."], "cwd": "."}
    ) is False


def test_run_project_command_unapproved_ruff_format_blocked(monkeypatch, tmp_path):
    """Without approval, ``ruff format`` is refused and files are unchanged."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    result = _call("run_project_command", {
        "program": "ruff", "args": ["format", "."], "cwd": ".",
    })
    assert result["ok"] is False
    assert "批准" in result["output"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x=1\n"


# ---------------------------------------------------------------------------
# Task 4 regression tests: sensitive-file protection at any depth.
# ---------------------------------------------------------------------------

def test_protected_env_files_at_any_depth(monkeypatch, tmp_path):
    """.env and friends are protected by filename at any directory depth."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    sub = tmp_path / "subdir"
    sub.mkdir()
    for name in (".env", ".env.local", ".env.production"):
        (sub / name).write_text("SECRET=1", encoding="utf-8")
        assert security.is_protected_file(sub / name) is True
        # read_file refuses to expose the content.
        denied = _call("read_file", {"path": f"subdir/{name}"})
        assert denied["ok"] is False


def test_protected_files_case_insensitive(monkeypatch, tmp_path):
    """Case variants are protected on case-insensitive (Windows) filesystems."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / ".ENV").write_text("SECRET=1", encoding="utf-8")
    assert security.is_protected_file(tmp_path / ".ENV") is True
    assert security.is_protected_file(".env") is True  # resolves to .ENV
    denied = _call("read_file", {"path": ".ENV"})
    assert denied["ok"] is False


def test_protected_database_file(monkeypatch, tmp_path):
    """The configured database file is protected by filename."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    db = tmp_path / "workflow.sqlite3"
    db.write_bytes(b"SQLite format 3\x00")
    assert security.is_protected_file(db) is True
    denied = _call("read_file", {"path": "workflow.sqlite3"})
    assert denied["ok"] is False


def test_environment_py_not_falsely_protected(monkeypatch, tmp_path):
    """A file merely containing "env" in its name (environment.py) is NOT
    protected."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "environment.py").write_text("DEBUG = True\n", encoding="utf-8")
    assert security.is_protected_file(tmp_path / "environment.py") is False
    result = _call("read_file", {"path": "environment.py"})
    assert result["ok"] is True


def test_write_file_and_search_share_protection_rules(monkeypatch, tmp_path):
    """write_file and search_code apply the same protection as read_file."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / ".env").write_text("KEY=sk-secret", encoding="utf-8")
    (sub / "x.py").write_text("KEY=sk-secret", encoding="utf-8")

    # write_file requires approval to overwrite the protected file.
    tool = get_tool("write_file")
    assert tool.requires_approval({"path": "nested/.env"}) is True
    denied = _call("write_file", {"path": "nested/.env", "content": "X=1"})
    assert denied["ok"] is False

    # search_code never leaks the protected file, but still finds the normal one.
    result = _call("search_code", {"query": "sk-secret"})
    matches = result["data"]["matches"]
    assert all(m["path"] != "nested/.env" for m in matches)
    assert any("x.py" in m["path"] for m in matches)
