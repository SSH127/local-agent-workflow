"""Per-run workspace isolation tests.

Verifies that a task can be bound to an arbitrary existing project directory,
that the workspace is validated once at creation, persisted with the run, and
used by every file/command tool through the run's own context (never a shared
global). Path-escape and sensitive-file protection must keep holding, and two
concurrent runs must not share or mix workspaces.

No real DeepSeek API, no real network and no real project edits: these use
temporary directories and the autouse isolation fixtures.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import config, main, storage
from app.agents.agent_executor import run_agent
from app.engine import runner as runner_module
from app.engine.approvals import ApprovalDecision
from app.engine.runner import Runner
from app.tools import tool_schemas
from app.tools.base import ToolContext
from app.tools.registry import get_tool
from app.tools.security import SecurityError, detect_workspace, resolve_workspace_root
from tests.conftest import FakeLLM, final_answer, tool_call


class _Recorder:
    """In-memory recorder that approves high-risk operations (for control-flow
    tests) and captures approval requests so tests can assert binding."""

    def __init__(self, approve: bool = True):
        self.steps: list[dict] = []
        self.approvals: list[tuple[str, dict]] = []
        self.approve = approve

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        return None

    def request_approval(self, run_id, tool_name, args, risk_reason):
        self.approvals.append((tool_name, args))
        return ApprovalDecision(approval_id=f"a{len(self.approvals)}", approved=self.approve)


# ---------------------------------------------------------------------------
# Workspace root validation (API-facing)
# ---------------------------------------------------------------------------

def test_resolve_workspace_root_accepts_existing_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    assert resolve_workspace_root(proj) == proj.resolve()


def test_resolve_workspace_root_rejects_missing(tmp_path):
    with pytest.raises(SecurityError, match="不存在"):
        resolve_workspace_root(tmp_path / "does-not-exist")


def test_resolve_workspace_root_rejects_file(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(SecurityError, match="不是目录"):
        resolve_workspace_root(f)


def test_resolve_workspace_root_rejects_forbidden_system_dir(monkeypatch, tmp_path):
    # Deterministically mark a temp dir "forbidden" without depending on the OS.
    forbidden = tmp_path / "sysroot"
    forbidden.mkdir()
    monkeypatch.setattr(config, "FORBIDDEN_WORKSPACE_ROOTS", {str(forbidden)})
    with pytest.raises(SecurityError, match="系统目录"):
        resolve_workspace_root(forbidden)


# ---------------------------------------------------------------------------
# Natural-language goal -> workspace detection
# ---------------------------------------------------------------------------

def test_detect_workspace_from_goal_directory(tmp_path):
    proj = tmp_path / "an"
    proj.mkdir()
    goal = f"帮我检查 {proj} 项目，修复 FAL 自动化脚本的三个稳定性 BUG。"
    assert detect_workspace(goal) == str(proj.resolve())


def test_detect_workspace_no_path_returns_none():
    assert detect_workspace("检查项目结构，修复测试失败的用例。") is None
    assert detect_workspace("") is None


def test_detect_workspace_nonexistent_path_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SecurityError, match="不存在"):
        detect_workspace(f"请修复 {missing} 的构建问题。")


def test_detect_workspace_file_uses_parent_dir(tmp_path):
    proj = tmp_path / "an"
    proj.mkdir()
    script = proj / "fal.py"
    script.write_text("x = 1", encoding="utf-8")
    assert detect_workspace(f"修复 {script} 的 bug。") == str(proj.resolve())


def test_detect_workspace_quoted_path_with_spaces(tmp_path):
    proj = tmp_path / "my project"
    proj.mkdir()
    assert detect_workspace(f"检查 `{proj}` 项目。") == str(proj.resolve())


def test_detect_workspace_uses_first_mentioned_path(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    goal = f"对比 {first} 与 {second} 的差异。"
    assert detect_workspace(goal) == str(first.resolve())


# ---------------------------------------------------------------------------
# Tool-level: per-run workspace targeting
# ---------------------------------------------------------------------------

def test_tool_uses_per_run_workspace(tmp_path):
    ws_a = tmp_path / "proj_a"
    ws_b = tmp_path / "proj_b"
    ws_a.mkdir()
    ws_b.mkdir()
    (ws_a / "info.txt").write_text("from-a", encoding="utf-8")
    (ws_b / "info.txt").write_text("from-b", encoding="utf-8")

    read = get_tool("read_file")
    res_a = read.handler({"path": "info.txt"}, ToolContext(run_id="r", workspace=ws_a))
    res_b = read.handler({"path": "info.txt"}, ToolContext(run_id="r", workspace=ws_b))
    assert res_a["ok"] is True and "from-a" in res_a["data"]["content"]
    assert res_b["ok"] is True and "from-b" in res_b["data"]["content"]


def test_path_escape_rejected_within_per_run_workspace(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    read = get_tool("read_file")
    result = read.handler({"path": "../outside.txt"}, ToolContext(run_id="r", workspace=ws))
    assert result["ok"] is False
    assert "越界" in result["output"]


def test_protected_env_still_protected_per_run(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".env").write_text("TOKEN=1", encoding="utf-8")

    read = get_tool("read_file")
    denied = read.handler({"path": ".env"}, ToolContext(run_id="r", workspace=ws))
    assert denied["ok"] is False

    write = get_tool("write_file")
    assert write.requires_approval({"path": ".env"}, ws) is True
    denied_write = write.handler(
        {"path": ".env", "content": "X=1"}, ToolContext(run_id="r", workspace=ws)
    )
    assert denied_write["ok"] is False


def test_run_project_command_uses_per_run_workspace(monkeypatch, tmp_path):
    """A command's cwd is resolved inside the run workspace, not the default."""
    default_ws = tmp_path / "default"
    default_ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_PATH", default_ws)
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    tool = get_tool("run_project_command")
    result = tool.handler(
        {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ToolContext(run_id="r", workspace=ws),
    )
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Executor-level: the run's workspace is threaded through to real tools
# ---------------------------------------------------------------------------

def test_run_agent_without_workspace_uses_global(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "hello.txt").write_text("global", encoding="utf-8")
    recorder = _Recorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "hello.txt"}),
        final_answer("读取成功。"),
    ])
    outcome = run_agent("run-1", "read", llm, tool_schemas(), recorder)
    assert outcome.status == "succeeded"


def test_run_agent_reads_from_specified_workspace(monkeypatch, tmp_path):
    default_ws = tmp_path / "default"
    default_ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_PATH", default_ws)
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "hello.txt").write_text("hello-ws", encoding="utf-8")

    recorder = _Recorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "hello.txt"}),
        final_answer("读取成功。"),
    ])
    outcome = run_agent("run-1", "read", llm, tool_schemas(), recorder, workspace=ws)
    assert outcome.status == "succeeded"


def test_run_agent_write_targets_workspace_not_default(monkeypatch, tmp_path):
    default_ws = tmp_path / "default"
    default_ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_PATH", default_ws)
    ws = tmp_path / "proj"
    ws.mkdir()

    recorder = _Recorder()
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": "out.txt", "content": "ws-write"}),
        final_answer("完成。"),
    ])
    run_agent("run-1", "write", llm, tool_schemas(), recorder, workspace=ws)
    assert (ws / "out.txt").read_text(encoding="utf-8") == "ws-write"
    # The process default workspace was never touched.
    assert list(default_ws.iterdir()) == []


def test_approval_gated_protected_write_lands_in_workspace(monkeypatch, tmp_path):
    default_ws = tmp_path / "default"
    default_ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_PATH", default_ws)
    ws = tmp_path / "proj"
    ws.mkdir()

    recorder = _Recorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    run_agent("run-1", "write env", llm, tool_schemas(), recorder, workspace=ws)
    # Approval was requested exactly once, for the exact args, and the protected
    # write landed inside the run's workspace (not the default workspace).
    assert recorder.approvals == [("write_file", {"path": ".env", "content": "TOKEN=1"})]
    assert (ws / ".env").read_text(encoding="utf-8") == "TOKEN=1"
    assert not (default_ws / ".env").exists()


# ---------------------------------------------------------------------------
# Runner-level: concurrency isolation + resume keeps the workspace
# ---------------------------------------------------------------------------

_RESULT = {
    "status": "succeeded", "attempts": 1, "summary": "ok", "error": None,
    "iterations": 1, "tool_calls": 1, "tool_successes": 1, "tool_failures": 0,
}


def _save_run(run_id: str, status: str, workspace: str | None = None) -> None:
    storage.init_db()
    storage.save_run({
        "run_id": run_id, "goal": "g", "status": status,
        "attempts": 0, "summary": "", "error": None,
        "workspace_path": workspace,
        "state": {}, "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })


def _wait_inactive(r: Runner, *run_ids: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while any(r.is_active(rid) for rid in run_ids) and time.monotonic() < deadline:
        time.sleep(0.001)


def test_concurrent_runs_use_distinct_workspaces(monkeypatch, tmp_path):
    _save_run("r1", "queued")
    _save_run("r2", "queued")
    r = Runner()
    seen: dict[str, str | None] = {}
    lock = threading.Lock()

    def fake_run_workflow(goal, max_attempts=2, *, run_id=None, llm=None,
                          recorder=None, workspace=None, resume=None):
        with lock:
            seen[run_id] = workspace
        return _RESULT

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)

    ws1 = str(tmp_path / "ws1")
    ws2 = str(tmp_path / "ws2")
    assert r.submit("r1", "g", workspace=ws1) is True
    assert r.submit("r2", "g", workspace=ws2) is True
    _wait_inactive(r, "r1", "r2")
    assert seen == {"r1": ws1, "r2": ws2}


def test_resume_uses_persisted_workspace(monkeypatch, tmp_path):
    ws = str(tmp_path / "ws-resume")
    _save_run("r1", "waiting_for_approval", workspace=ws)
    r = Runner()
    seen: list[str | None] = []

    def fake_run_workflow(goal, max_attempts=2, *, run_id=None, llm=None,
                          recorder=None, workspace=None, resume=None):
        seen.append(workspace)
        return _RESULT

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)
    assert r.resume("r1", "g") is True
    _wait_inactive(r, "r1")
    assert seen == [ws]


# ---------------------------------------------------------------------------
# API-level: validation, persistence, backward compatibility
# ---------------------------------------------------------------------------

def _api_client(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "api-ws.sqlite3")
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", "")
    return TestClient(main.app)


def test_api_valid_workspace_persisted_and_passed_to_submit(monkeypatch, tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    submitted: list[tuple] = []

    def fake_submit(run_id, goal, max_attempts=2, workspace=None):
        submitted.append((run_id, goal, max_attempts, workspace))
        return True

    monkeypatch.setattr(main.runner, "submit", fake_submit)
    with _api_client(monkeypatch, tmp_path) as client:
        resp = client.post(
            "/runs",
            json={"goal": "分析目标项目", "max_attempts": 1, "workspace_path": str(ws)},
        )
        assert resp.status_code == 201
        run = resp.json()
        assert run["workspace_path"] == str(ws)
    assert submitted and submitted[0][3] == str(ws)
    assert storage.get_run(run["run_id"])["workspace_path"] == str(ws)


def test_api_invalid_workspace_rejected_400(monkeypatch, tmp_path):
    submitted: list = []
    monkeypatch.setattr(main.runner, "submit", lambda *a, **k: submitted.append(1) or True)
    with _api_client(monkeypatch, tmp_path) as client:
        resp = client.post(
            "/runs",
            json={"goal": "分析目标项目", "max_attempts": 1,
                  "workspace_path": str(tmp_path / "nope")},
        )
        assert resp.status_code == 400
        assert "不存在" in str(resp.json()["detail"])
    assert submitted == []
    assert storage.list_runs(limit=10) == []


def test_api_missing_workspace_uses_default(monkeypatch, tmp_path):
    submitted: list[tuple] = []

    def fake_submit(run_id, goal, max_attempts=2, workspace=None):
        submitted.append((run_id, goal, max_attempts, workspace))
        return True

    monkeypatch.setattr(main.runner, "submit", fake_submit)
    with _api_client(monkeypatch, tmp_path) as client:
        resp = client.post("/runs", json={"goal": "默认工作区任务", "max_attempts": 1})
        assert resp.status_code == 201
        run = resp.json()
        assert run["workspace_path"] is None
    # No workspace supplied -> executor falls back to config.WORKSPACE_PATH.
    assert submitted and submitted[0][3] is None


def test_api_goal_path_detected_as_workspace(monkeypatch, tmp_path):
    ws = tmp_path / "auto-proj"
    ws.mkdir()
    submitted: list[tuple] = []

    def fake_submit(run_id, goal, max_attempts=2, workspace=None):
        submitted.append((run_id, goal, max_attempts, workspace))
        return True

    monkeypatch.setattr(main.runner, "submit", fake_submit)
    with _api_client(monkeypatch, tmp_path) as client:
        resp = client.post(
            "/runs",
            json={"goal": f"检查 {ws} 项目并修复 BUG", "max_attempts": 1},
        )
        assert resp.status_code == 201
        run = resp.json()
        assert run["workspace_path"] == str(ws)
        assert run["state"]["workspace_source"] == "goal"
    assert submitted and submitted[0][3] == str(ws)


def test_api_goal_invalid_path_rejected_400_without_submit(monkeypatch, tmp_path):
    submitted: list = []
    monkeypatch.setattr(main.runner, "submit", lambda *a, **k: submitted.append(1) or True)
    missing = tmp_path / "ghost-proj"
    with _api_client(monkeypatch, tmp_path) as client:
        resp = client.post(
            "/runs",
            json={"goal": f"检查 {missing} 项目", "max_attempts": 1},
        )
        assert resp.status_code == 400
        assert "不存在" in str(resp.json()["detail"])
    # A clearly-invalid named path must not fall back to the default workspace
    # and must not start a run (no model iterations consumed).
    assert submitted == []
    assert storage.list_runs(limit=10) == []
