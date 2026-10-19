"""Regression tests for argument-level command policy.

The core regression is ``python -m pytest _recover.py -q -s``: an allowlisted
program name (``pytest``) used to run an arbitrary script. These tests pin the
behaviour of the replacement analysis: what is a controlled test run, what must
be approved, and what is a hard bypass rejection.
"""

import pytest

from app import config
from app.agents.agent_executor import ResumeState, run_agent
from app.engine.approvals import ApprovalDecision
from app.tools import security, tool_schemas
from app.tools.base import ToolContext
from app.tools.command_policy import (
    filter_model_env,
    sanitize_inherited_env,
)
from app.tools.errors import SecurityError
from app.tools.registry import get_tool
from tests.conftest import FakeLLM, final_answer, tool_call


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "_recover.py").write_text(
        "from pathlib import Path\nPath('side_effect.txt').write_text('ran')\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        "from pathlib import Path\nPath('conftest_ran.txt').write_text('ran')\n",
        encoding="utf-8",
    )
    return tmp_path


def _classify(program, args, ws, cwd="."):
    return security.classify_command(program, args, cwd, ws)


# ---------------------------------------------------------------------------
# The reported bypass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "program,args",
    [
        ("python", ["-m", "pytest", "_recover.py", "-q", "-s"]),
        ("python", ["-m", "pytest", "./_recover.py", "-q"]),
        ("pytest", ["_recover.py", "-q", "-s"]),
        ("python", ["-m", "pytest", "--", "_recover.py"]),
        ("python", ["-m", "pytest", "conftest.py", "-q"]),
    ],
)
def test_pytest_cannot_execute_non_test_scripts(program, args, ws):
    """An explicit .py target that is not a test module is imported+executed by
    pytest, so it must be rejected outright (no approval path)."""
    with pytest.raises(SecurityError, match="不是测试文件"):
        security.validate_command(program, args, ".", ws)


def test_pytest_script_bypass_does_not_execute(ws):
    """End-to-end: the handler refuses before spawning any subprocess."""
    tool = get_tool("run_project_command")
    result = tool.handler(
        {
            "program": "python",
            "args": ["-m", "pytest", "_recover.py", "-q", "-s"],
            "cwd": ".",
        },
        ToolContext(run_id="r", workspace=ws, approval_check=lambda n, a: True),
    )
    assert result["ok"] is False
    assert not (ws / "side_effect.txt").exists()


# ---------------------------------------------------------------------------
# Controlled test runs stay usable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "program,args",
    [
        ("python", ["-m", "pytest", "tests/test_ok.py", "-q"]),
        ("pytest", ["tests/test_ok.py", "-q"]),
        ("python", ["-m", "pytest", "tests/", "-q", "-x"]),
        ("python", ["-m", "pytest", "tests/test_ok.py::test_ok", "-q"]),
        ("python", ["-m", "pytest", "tests/test_ok.py", "-k", "ok or not slow"]),
        ("python", ["-m", "pytest", "tests/test_ok.py", "-p", "no:cacheprovider"]),
        ("python", ["-m", "pytest", "tests/test_ok.py", "--tb=short", "-q"]),
        ("python", ["-m", "pytest", "tests/test_ok.py", "--maxfail=1"]),
    ],
)
def test_controlled_test_commands_are_allowed(program, args, ws):
    decision = _classify(program, args, ws)
    assert decision.requires_approval is False
    assert decision.effective_program == "pytest"


def test_cwd_relative_targets_are_validated_against_cwd(ws):
    (ws / "tests" / "sub").mkdir()
    (ws / "tests" / "sub" / "test_inner.py").write_text("def test_i():\n    pass\n")
    decision = _classify(
        "python", ["-m", "pytest", "sub/test_inner.py", "-q"], ws, cwd="tests"
    )
    assert decision.requires_approval is False


def test_implicit_whole_repo_collection_requires_approval(ws):
    decision = _classify("python", ["-m", "pytest", "-q"], ws)
    assert decision.requires_approval is True
    assert "未指定测试目标" in (decision.risk or "")


def test_unknown_pytest_option_escalates_to_approval(ws):
    decision = _classify("python", ["-m", "pytest", "tests/test_ok.py", "--brand-new"], ws)
    assert decision.requires_approval is True
    assert "--brand-new" in (decision.risk or "")


def test_doctest_modules_escalates_to_approval(ws):
    decision = _classify("python", ["-m", "pytest", "tests/", "--doctest-modules"], ws)
    assert decision.requires_approval is True


# ---------------------------------------------------------------------------
# Plugin / config / module-name indirection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["tests/test_ok.py", "-p", "evil_plugin"],
        ["tests/test_ok.py", "-pevil_plugin"],
        ["tests/test_ok.py", "--plugin=evil_plugin"],
        ["tests/test_ok.py", "-c", "evil.ini"],
        ["tests/test_ok.py", "--config-file=evil.ini"],
        ["tests/test_ok.py", "-o", "addopts=-p evil"],
        ["tests/test_ok.py", "--override-ini", "python_files=*"],
        ["tests/test_ok.py", "--pyargs", "evil_module"],
    ],
)
def test_plugin_and_config_indirection_is_rejected(args, ws):
    with pytest.raises(SecurityError):
        security.validate_command("python", ["-m", "pytest", *args], ".", ws)


@pytest.mark.parametrize(
    "args",
    [
        ["../outside.py"],
        ["../../etc/passwd"],
        ["tests/test_ok.py", "--junitxml=../../out.xml"],
        ["tests/test_ok.py", "--basetemp=.."],
        ["tests/test_ok.py", "--rootdir=.."],
        ["tests/test_ok.py", "--ignore", "../outside"],
    ],
)
def test_paths_cannot_escape_the_workspace(args, ws):
    with pytest.raises(SecurityError):
        security.validate_command("python", ["-m", "pytest", *args], ".", ws)


def test_absolute_outside_target_is_rejected(ws):
    outside = ws.parent / "outside_test_file.py"
    outside.write_text("print('nope')\n", encoding="utf-8")
    with pytest.raises(SecurityError, match="越界"):
        security.validate_command("python", ["-m", "pytest", str(outside)], ".", ws)


def test_module_shadowing_is_rejected(ws):
    (ws / "pytest.py").write_text("print('hijacked')\n", encoding="utf-8")
    with pytest.raises(SecurityError, match="遮蔽"):
        security.validate_command(
            "python", ["-m", "pytest", "tests/test_ok.py", "-q"], ".", ws
        )


def test_shell_composition_is_rejected(ws):
    for args in (
        ["tests/test_ok.py", ";", "rm", "-rf", "."],
        ["tests/test_ok.py", "&&", "del", "/s", "x"],
        ["tests/test_ok.py", "$(rm -rf /)"],
        ["tests/test_ok.py", "|", "sh"],
    ):
        with pytest.raises(SecurityError):
            security.validate_command("python", ["-m", "pytest", *args], ".", ws)


# ---------------------------------------------------------------------------
# Other interpreter indirection
# ---------------------------------------------------------------------------


def test_coverage_cannot_run_arbitrary_script(ws):
    with pytest.raises(SecurityError, match="coverage run"):
        security.validate_command(
            "python", ["-m", "coverage", "run", "_recover.py"], ".", ws
        )


def test_coverage_wrapping_pytest_inherits_pytest_policy(ws):
    decision = _classify(
        "python",
        ["-m", "coverage", "run", "-m", "pytest", "tests/test_ok.py", "-q"],
        ws,
    )
    assert decision.requires_approval is False
    with pytest.raises(SecurityError, match="不是测试文件"):
        security.validate_command(
            "python",
            ["-m", "coverage", "run", "-m", "pytest", "_recover.py"],
            ".",
            ws,
        )


def test_unittest_module_import_is_rejected_but_discover_needs_approval(ws):
    with pytest.raises(SecurityError, match="unittest"):
        security.validate_command("python", ["-m", "unittest", "evil"], ".", ws)
    decision = _classify("python", ["-m", "unittest", "discover", "-s", "tests"], ws)
    assert decision.requires_approval is True
    with pytest.raises(SecurityError):
        security.validate_command(
            "python", ["-m", "unittest", "discover", "-s", ".."], ".", ws
        )


def test_compileall_cannot_write_outside_workspace(ws):
    with pytest.raises(SecurityError):
        security.validate_command("python", ["-m", "compileall", ".."], ".", ws)
    decision = _classify("python", ["-m", "compileall", "."], ws)
    assert decision.requires_approval is True


# ---------------------------------------------------------------------------
# Environment injection
# ---------------------------------------------------------------------------


def test_model_env_cannot_inject_pytest_plugins_or_path():
    kept, dropped = filter_model_env(
        {
            "PYTEST_ADDOPTS": "-p evil",
            "PYTEST_PLUGINS": "evil",
            "PYTHONPATH": "C:/outside",
            "PATH": ".",
            "SAFE_FLAG": "1",
        }
    )
    assert kept == {"SAFE_FLAG": "1"}
    assert set(dropped) == {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "PATH"}


def test_inherited_env_is_sanitized_but_path_survives():
    env = {
        "PATH": "C:/Windows",
        "PYTEST_ADDOPTS": "-p evil",
        "PYTHONPATH": "C:/outside",
        "NODE_OPTIONS": "--require evil",
        "HOME": "C:/Users/x",
    }
    sanitize_inherited_env(env)
    assert env["PATH"] == "C:/Windows"
    assert env["HOME"] == "C:/Users/x"
    assert "PYTEST_ADDOPTS" not in env
    assert "PYTHONPATH" not in env
    assert "NODE_OPTIONS" not in env


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, approve: bool):
        self.approve = approve
        self.steps: list[dict] = []
        self.status = None
        self.state: dict = {}

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        self.status = status

    def set_state(self, state):
        self.state.update(state)

    def request_approval(self, run_id, tool_name, args, risk_reason):
        return ApprovalDecision(approval_id="appr-1", approved=self.approve)


def test_denied_approval_does_not_execute(ws):
    recorder = _Recorder(approve=False)
    llm = FakeLLM([
        tool_call(
            "c1",
            "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "-q"], "cwd": "."},
        ),
        final_answer("无法执行未获批准的命令。"),
    ])
    run_agent("run-deny", "运行测试", llm, tool_schemas(), recorder)
    # Bare pytest is approval-gated; the conftest side effect must not happen.
    assert not (ws / "conftest_ran.txt").exists()
    assert any(s.get("kind") == "approval" and s["status"] == "denied" for s in recorder.steps)


def test_resumed_approval_is_bound_to_the_same_run(ws, monkeypatch):
    """A stored decision for another run must not authorize this one."""
    recorder = _Recorder(approve=False)
    llm = FakeLLM([final_answer("未执行。")])
    args = {"program": "python", "args": ["-m", "pytest", "-q"], "cwd": "."}
    resume = ResumeState(
        pending_tools=[
            {"name": "run_project_command", "args": args, "call_id": "c1"},
        ],
        preapproved={
            "approval_id": "stale",
            "run_id": "some-other-run",
            "tool_name": "run_project_command",
            "args": args,
            "approved": True,
        },
    )
    run_agent("run-resume", "运行测试", llm, tool_schemas(), recorder, resume=resume)
    assert not (ws / "conftest_ran.txt").exists()
    assert recorder.steps, "a fresh approval request should have been recorded"


def test_approval_reason_reports_test_scope(ws):
    reason = security.approval_reason(
        "run_project_command",
        {"program": "python", "args": ["-m", "pytest", "-q"], "cwd": "."},
        ws,
    )
    assert "未指定测试目标" in reason


def test_allowlisted_programs_still_rejected():
    for program, args in (
        ("bash", []),
        ("python", ["-c", "print(1)"]),
        ("python", ["-m", "os"]),
        ("rm", ["-rf", "/"]),
    ):
        with pytest.raises(SecurityError):
            security.validate_command(program, args, ".")


# ---------------------------------------------------------------------------
# Hidden code loading: pytest config addopts and conftest plugins
# ---------------------------------------------------------------------------


def test_addopts_cannot_load_plugins_invisibly(ws):
    """``addopts`` never appears in argv, so it must be validated separately."""
    (ws / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-p evil_plugin"\n', encoding="utf-8"
    )
    with pytest.raises(SecurityError, match="addopts"):
        security.validate_command(
            "python", ["-m", "pytest", "tests/test_ok.py", "-q"], ".", ws
        )


def test_pytest_ini_addopts_cannot_load_external_config(ws):
    (ws / "pytest.ini").write_text("[pytest]\naddopts = -q -c evil.ini\n", encoding="utf-8")
    with pytest.raises(SecurityError, match="addopts"):
        security.validate_command(
            "python", ["-m", "pytest", "tests/test_ok.py"], ".", ws
        )


def test_conftest_pytest_plugins_requires_approval(ws):
    (ws / "conftest.py").write_text('pytest_plugins = ["evil_plugin"]\n', encoding="utf-8")
    decision = _classify("python", ["-m", "pytest", "tests/test_ok.py", "-q"], ws)
    assert decision.requires_approval is True
    assert "pytest_plugins" in (decision.risk or "")


def test_nested_conftest_plugin_is_also_detected(ws):
    (ws / "tests" / "conftest.py").write_text(
        "pytest_plugins = (\n    'evil_plugin',\n)\n", encoding="utf-8"
    )
    decision = _classify("python", ["-m", "pytest", "tests/test_ok.py", "-q"], ws)
    assert decision.requires_approval is True


def test_plain_pytest_config_stays_allowed(ws):
    (ws / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "-q"\n',
        encoding="utf-8",
    )
    decision = _classify("python", ["-m", "pytest", "tests/test_ok.py"], ws)
    assert decision.requires_approval is False


def test_unparsable_pytest_config_escalates_to_approval(ws):
    (ws / "pyproject.toml").write_text("[tool.pytest.ini_options\naddopts = ", encoding="utf-8")
    decision = _classify("python", ["-m", "pytest", "tests/test_ok.py"], ws)
    assert decision.requires_approval is True
    assert "无法解析" in (decision.risk or "")


def test_rejected_command_never_asks_for_approval_and_never_runs(ws):
    """A hard policy rejection is not approvable: no approval step, no execution."""
    recorder = _Recorder(approve=True)  # even a permissive user cannot allow it
    llm = FakeLLM([
        tool_call(
            "c1",
            "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "_recover.py", "-q", "-s"]},
        ),
        final_answer("命令被策略拒绝。"),
    ])
    run_agent("run-reject", "运行脚本", llm, tool_schemas(), recorder)
    assert not (ws / "side_effect.txt").exists()
    assert not any(s.get("kind") == "approval" for s in recorder.steps)
    assert any(
        s.get("kind") == "tool" and s["status"] == "failed" and "不是测试文件" in (s["output"] or "")
        for s in recorder.steps
    )


# ---------------------------------------------------------------------------
# npm scripts are hidden code too
# ---------------------------------------------------------------------------


def test_npm_script_body_with_shell_composition_requires_approval(ws):
    (ws / "package.json").write_text(
        '{"scripts": {"build": "tsc && node evil.js"}}', encoding="utf-8"
    )
    decision = _classify("npm", ["run", "build"], ws)
    assert decision.requires_approval is True
    assert "package.json" in (decision.risk or "")


def test_npm_script_body_with_interpreter_call_requires_approval(ws):
    (ws / "package.json").write_text(
        '{"scripts": {"test": "node -e \\"require(\'child_process\')\\" "}}', encoding="utf-8"
    )
    assert _classify("npm", ["test"], ws).requires_approval is True


def test_plain_npm_build_script_stays_allowed(ws):
    (ws / "package.json").write_text('{"scripts": {"build": "tsc -p ."}}', encoding="utf-8")
    assert _classify("npm", ["run", "build"], ws).requires_approval is False


def test_unparsable_package_json_requires_approval(ws):
    (ws / "package.json").write_text('{"scripts": ', encoding="utf-8")
    assert _classify("npm", ["run", "build"], ws).requires_approval is True

