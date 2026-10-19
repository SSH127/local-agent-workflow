"""REQUIRE_APPROVAL switch: when false, approval-requiring tools run without
blocking; hard security rejections are never relaxed."""
from __future__ import annotations

from app import config
from app.agents.agent_executor import run_agent
from app.engine.approvals import ApprovalDecision
from app.tools import tool_schemas
from tests.conftest import FakeLLM, final_answer, tool_call


class MemoryRecorder:
    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.approvals: list[tuple[str, dict]] = []

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        return None

    def request_approval(self, run_id, tool_name, args, risk_reason):
        self.approvals.append((tool_name, args))
        return ApprovalDecision(approval_id="should-not-be-used", approved=False)


def _run(llm, recorder):
    return run_agent("run-test", "测试任务", llm, tool_schemas(), recorder)


def test_disabled_approval_runs_protected_write_without_blocking(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "REQUIRE_APPROVAL", False)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    _run(llm, recorder)
    # Executed without ever asking the approval channel.
    assert recorder.approvals == []
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "TOKEN=1"
    auto = [s for s in recorder.steps if s["kind"] == "approval"]
    assert len(auto) == 1 and auto[0]["status"] == "auto_granted"


def test_disabled_approval_still_denies_rejected_command(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "REQUIRE_APPROVAL", False)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-c", "print(1)"], "cwd": "."}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert recorder.approvals == []
    execs = [s for s in recorder.steps
             if s["kind"] == "tool" and s["tool_name"] == "run_project_command"]
    assert len(execs) == 1
    assert execs[0]["status"] == "failed"
    assert execs[0]["error"] == "rejected by security policy"
    assert not [s for s in recorder.steps if s.get("status") == "auto_granted"]


def test_enabled_approval_still_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()  # denies
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    _run(llm, recorder)
    assert len(recorder.approvals) == 1
    assert not (tmp_path / ".env").exists()
