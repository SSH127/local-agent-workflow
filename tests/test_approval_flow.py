"""End-to-end approval lifecycle tests: the executor + real tools.

These use a scripted fake LLM for control flow and the real tool handlers for
execution results, in an isolated temporary workspace.
"""
from __future__ import annotations

from app import config
from app.agents.agent_executor import ResumeState, run_agent
from app.engine.approvals import ApprovalDecision
from app.tools import tool_schemas
from tests.conftest import FakeLLM, final_answer, multi_tool_call, tool_call


class MemoryRecorder:
    def __init__(self, approve: bool = True):
        self.steps: list[dict] = []
        self.approvals: list[tuple[str, dict]] = []
        self.state_calls: list[dict] = []
        self.approve = approve

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        self.state_calls.append(dict(state))

    def request_approval(self, run_id, tool_name, args, risk_reason):
        self.approvals.append((tool_name, args))
        return ApprovalDecision(
            approval_id=f"appr-{len(self.approvals)}", approved=self.approve,
        )


def _run(llm, recorder, **kwargs):
    return run_agent("run-test", "测试任务", llm, tool_schemas(), recorder, **kwargs)


def test_approved_protected_write_executes_once(monkeypatch, tmp_path):
    """Approval grants execution of the exact protected operation, once."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    _run(llm, recorder)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "TOKEN=1"
    writes = [s for s in recorder.steps if s["kind"] == "tool" and s["tool_name"] == "write_file"]
    assert len(writes) == 1
    # Approval was requested exactly once.
    assert len(recorder.approvals) == 1


def test_unapproved_protected_write_does_not_execute(monkeypatch, tmp_path):
    """Without approval the protected write is refused and has no side effect."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=False)
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "SECRET=1"}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert not (tmp_path / ".env").exists()
    rejected = [s for s in recorder.steps if s["kind"] == "tool" and s.get("error") == "rejected by user"]
    assert len(rejected) == 1


def test_approved_command_executes_and_succeeds(monkeypatch, tmp_path):
    """An approved, approval-requiring command actually runs (exit code 0)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pip", "--version"], "cwd": "."}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    execs = [s for s in recorder.steps if s["kind"] == "tool" and s["tool_name"] == "run_project_command"]
    assert len(execs) == 1 and execs[0]["status"] == "succeeded"


def test_rejected_command_does_not_execute(monkeypatch, tmp_path):
    """A rejected approval-requiring command is not executed and fails cleanly."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=False)
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pip", "--version"], "cwd": "."}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    execs = [s for s in recorder.steps if s["kind"] == "tool" and s["tool_name"] == "run_project_command"]
    assert len(execs) == 1 and execs[0]["status"] == "failed"
    assert execs[0]["error"] == "rejected by user"


def test_resume_state_excludes_completed_tools(monkeypatch, tmp_path):
    """The persisted pending set contains only not-yet-executed tools, so a
    restart never re-runs already-completed work."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    recorder = MemoryRecorder(approve=True)
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "read_file", {"path": "a.txt"}),
            ("c2", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        ]),
        final_answer("完成。"),
    ])
    _run(llm, recorder)
    # The pending_tools persisted at approval time is just the write (the read
    # had already executed and its tool message is in agent_messages).
    pending_states = [
        s for s in recorder.state_calls
        if isinstance(s.get("pending_tools"), list) and s["pending_tools"]
    ]
    assert pending_states, "expected a persisted pending_tools state"
    last = pending_states[-1]
    assert [p["name"] for p in last["pending_tools"]] == ["write_file"]
    msgs = last.get("agent_messages", [])
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in msgs)


def test_resume_executes_only_pending_tools(monkeypatch, tmp_path):
    """Resuming with a pre-approved decision re-runs only the pending tool, and
    only once; the already-completed tool is not re-executed."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=True)
    assistant_msg = {
        "role": "assistant", "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "write_file", "arguments": '{"path": ".env", "content": "TOKEN=1"}'}},
        ],
    }
    resume = ResumeState(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
            assistant_msg,
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        ],
        pending_tools=[
            {"name": "write_file", "args": {"path": ".env", "content": "TOKEN=1"}, "call_id": "c2"},
        ],
        preapproved={
            "approval_id": "a1", "tool_name": "write_file",
            "args": {"path": ".env", "content": "TOKEN=1"},
            "approved": True, "reason": None,
        },
    )
    llm = FakeLLM([final_answer("完成。")])
    run_agent("run-test", "测试任务", llm, tool_schemas(), recorder, resume=resume)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "TOKEN=1"
    tool_steps = [s for s in recorder.steps if s["kind"] == "tool"]
    # Only the pending write executed; the completed read was not re-run.
    assert [s["tool_name"] for s in tool_steps] == ["write_file"]


def test_preapproved_does_not_authorize_changed_args(monkeypatch, tmp_path):
    """A pre-approved decision carried across a restart only authorizes a
    matching tool + args; changed args are re-checked (and rejected here)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=False)
    assistant_msg = {
        "role": "assistant", "content": "",
        "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "write_file", "arguments": '{"path": ".env", "content": "TOKEN=1"}'}},
        ],
    }
    resume = ResumeState(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
            assistant_msg,
        ],
        pending_tools=[
            # args changed vs the pre-approved args
            {"name": "write_file", "args": {"path": ".env", "content": "TOKEN=2"}, "call_id": "c2"},
        ],
        preapproved={
            "approval_id": "a1", "tool_name": "write_file",
            "args": {"path": ".env", "content": "TOKEN=1"},
            "approved": True, "reason": None,
        },
    )
    llm = FakeLLM([final_answer("完成。")])
    run_agent("run-test", "测试任务", llm, tool_schemas(), recorder, resume=resume)
    # The pre-approved decision did not match the changed args, so a fresh
    # approval was requested (denied by the recorder), and nothing was written.
    assert not (tmp_path / ".env").exists()
