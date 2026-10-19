import threading
import time
from typing import Any

import pytest

from app import config
from app.agents import agent_executor
from app.agents.agent_executor import ResumeState, run_agent
from app.engine.approvals import APPROVAL_GRANTED, ApprovalDecision
from app.engine.runner import ApprovalGate
from app.tools import tool_schemas
from app.tools.base import Tool, tool_ok
from tests.conftest import FakeLLM, final_answer, multi_tool_call, tool_call


class MemoryRecorder:
    def __init__(self, approve: bool = True):
        self.steps: list[dict[str, Any]] = []
        self.status: str | None = None
        self.summary: str | None = None
        self.error: str | None = None
        self.state: dict[str, Any] = {}
        self.approve = approve
        self.approvals: list[tuple[str, dict[str, Any]]] = []

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        self.status = status
        self.summary = summary
        self.error = error

    def set_state(self, state):
        self.state.update(state)

    def request_approval(self, run_id, tool_name, args, risk_reason):
        self.approvals.append((tool_name, args))
        return ApprovalDecision(
            approval_id=f"appr-{len(self.approvals)}", approved=self.approve,
        )


def _run(llm, recorder, **kwargs):
    return run_agent("run-test", "测试任务", llm, tool_schemas(), recorder, **kwargs)


def test_agent_calls_tool_then_finishes(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "hello.txt"}),
        final_answer("已读取文件并验证完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_calls == 1
    assert outcome.tool_successes == 1
    # The tool result was fed back to the model.
    assert any(m.get("role") == "tool" for m in llm.calls[-1][0])


def test_agent_recovers_from_failed_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "missing.txt"}),
        tool_call("c2", "read_file", {"path": "ok.txt"}),
        final_answer("第一次读取失败，改用正确路径后成功。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_failures == 1
    assert outcome.tool_successes == 1


def test_agent_max_iterations_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([tool_call("c1", "list_files", {})])  # repeats forever
    outcome = run_agent(
        "run-test", "测试任务", llm, tool_schemas(), recorder, max_iterations=2,
    )
    assert outcome.status == "failed"
    assert "最大迭代次数" in outcome.error


def test_agent_unknown_tool_recovers(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "no_such_tool", {}),
        tool_call("c2", "list_files", {}),
        final_answer("未知工具被记录，随后使用正确工具完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert any(s["tool_name"] == "no_such_tool" and s["status"] == "failed" for s in recorder.steps)


def test_agent_approval_denied(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder(approve=False)
    llm = FakeLLM([
        tool_call("c1", "run_project_command", {"program": "pip", "args": ["install", "requests"]}),
        final_answer("安装被拒绝。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert recorder.approvals and recorder.approvals[0][0] == "run_project_command"
    assert any(s["kind"] == "approval" for s in recorder.steps)
    assert any(s["status"] == "failed" and s.get("error") == "rejected by user" for s in recorder.steps)


def test_agent_task_timeout():
    recorder = MemoryRecorder()
    llm = FakeLLM([final_answer("不会被执行。")])
    outcome = run_agent(
        "run-test", "测试任务", llm, tool_schemas(), recorder, task_timeout=0,
    )
    assert outcome.status == "failed"
    assert "超时" in outcome.summary


def test_agent_final_answer_without_tool_fails():
    # The model never calls a tool — must not be marked succeeded.
    recorder = MemoryRecorder()
    llm = FakeLLM([final_answer("任务已完成，一切正常。")])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert outcome.tool_calls == 0
    assert "未调用任何工具" in outcome.error


def test_agent_failure_diagnosis_and_repair_loop(monkeypatch, tmp_path):
    """Verify the full execute -> fail -> diagnose -> fix -> re-verify loop.

    A scripted LLM first runs a test that fails, reads the error, fixes the
    file, re-runs the test successfully, then concludes. This exercises the
    real tool execution path end-to-end (only the model is simulated).
    """
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    # A pytest module that fails while a marker file is absent.
    (tmp_path / "test_check.py").write_text(
        "from pathlib import Path\n"
        "def test_marker_exists():\n"
        "    assert Path('marker.txt').exists(), 'marker missing'\n",
        encoding="utf-8",
    )
    recorder = MemoryRecorder()
    llm = FakeLLM([
        # 1) Run the test -> fails.
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_check.py", "-q"], "cwd": "."}),
        # 2) Read the source to diagnose.
        tool_call("c2", "read_file", {"path": "test_check.py"}),
        # 3) Create the missing marker (the "fix").
        tool_call("c3", "write_file", {"path": "marker.txt", "content": "fixed"}),
        # 4) Re-run the test -> passes.
        tool_call("c4", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_check.py", "-q"], "cwd": "."}),
        final_answer("已定位缺失的 marker.txt，创建后复验通过。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_calls == 4
    assert outcome.tool_failures == 1
    assert outcome.tool_successes == 3
    # The failed run really returned a non-zero exit code, and that real error
    # (exit_code 1 + stderr) was handed back to the model as a tool message.
    tool_msgs = [m for m in llm.calls[-1][0] if m.get("role") == "tool"]
    assert any('"exit_code": 1' in m["content"] for m in tool_msgs)
    assert (tmp_path / "marker.txt").exists()


def test_agent_failure_without_recovery_fails(monkeypatch, tmp_path):
    """A failing final step without a subsequent passing step must fail."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "missing.txt"}),
        final_answer("没有读取到文件，但任务已完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert "最后一步工具执行失败" in outcome.error


def test_agent_multi_tool_call_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "read_file", {"path": "a.txt"}),
            ("c2", "read_file", {"path": "b.txt"}),
        ]),
        final_answer("已读取两个文件。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_calls == 2
    assert outcome.tool_successes == 2


def test_agent_multi_tool_call_approval_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    recorder = MemoryRecorder(approve=False)
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "read_file", {"path": "a.txt"}),
            ("c2", "run_project_command", {"program": "pip", "args": ["install", "requests"]}),
        ]),
        final_answer("安装被拒绝。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    # Approval is per-call: only the approval-requiring call is rejected; the
    # safe read still runs and succeeds. Every assistant tool call is answered.
    assert recorder.approvals and recorder.approvals[0][0] == "run_project_command"
    tool_steps = [s for s in recorder.steps if s["kind"] == "tool"]
    read_steps = [s for s in tool_steps if s["tool_name"] == "read_file"]
    pip_steps = [s for s in tool_steps if s["tool_name"] == "run_project_command"]
    assert len(read_steps) == 1 and read_steps[0]["status"] == "succeeded"
    assert len(pip_steps) == 1 and pip_steps[0]["status"] == "failed"
    assert pip_steps[0]["error"] == "rejected by user"


def test_resume_executes_pending_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    recorder = MemoryRecorder()
    # Simulate a restart: the conversation already contains the assistant
    # message with two tool calls, and both calls are pending.
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
        ],
    }
    resume = ResumeState(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
            assistant_msg,
        ],
        pending_tools=[
            {"name": "read_file", "args": {"path": "a.txt"}, "call_id": "c1"},
            {"name": "read_file", "args": {"path": "b.txt"}, "call_id": "c2"},
        ],
    )
    llm = FakeLLM([final_answer("已读取两个文件。")])
    outcome = run_agent("run-test", "测试任务", llm, tool_schemas(), recorder, resume=resume)
    assert outcome.status == "succeeded"
    assert outcome.tool_calls == 2
    # Both tool messages are present, so the assistant message is fully answered.
    tool_msgs = [m for m in llm.calls[-1][0] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


def test_approval_gate_resolve():
    gate = ApprovalGate()
    record = gate.request("r1", "run_project_command", {"program": "pip"}, "风险")
    assert gate.get_pending_for_run("r1") is not None

    result: dict[str, str] = {}

    def waiter():
        result["status"] = gate.wait(record.approval_id, timeout=5)

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)  # let the waiter register
    assert gate.resolve_if_pending(record.approval_id, True, None) == "resolved"
    thread.join(timeout=2)
    assert result.get("status") == APPROVAL_GRANTED
    # A duplicate resolve is an idempotent no-op, never a second side effect.
    assert gate.resolve_if_pending(record.approval_id, False, None) == "already_resolved"


# ---------------------------------------------------------------------------
# Phase 3 regression tests (P0-1 / P0-2 evidence-based success & retry)
# ---------------------------------------------------------------------------

def test_verification_failure_not_masked_by_unrelated_success(monkeypatch, tmp_path):
    """Test A: a failed verification must not be masked by a later unrelated
    successful command, even if the model claims the task is done."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "test_x.py").write_text("def test_fail():\n    assert False\n", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        # 1) Real verification command fails (non-zero exit).
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_x.py", "-q"], "cwd": "."}),
        # 2) Unrelated, non-verifying command succeeds.
        tool_call("c2", "run_project_command",
                  {"program": "python", "args": ["--version"], "cwd": "."}),
        # 3) Model claims done anyway.
        final_answer("测试已通过。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert outcome.tool_successes == 1  # the unrelated command succeeded
    assert outcome.tool_failures == 1   # the verification failed
    assert "验证" in outcome.error       # failed verification is the reason


def test_mutation_requires_verification(monkeypatch, tmp_path):
    """Test B (complement): modifying a file without a later passing verification
    must not be marked succeeded."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": "a.py", "content": "print(1)"}),
        final_answer("已修改文件，任务完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert "验证" in outcome.error
    assert (tmp_path / "a.py").exists()  # the write really happened


def test_readonly_task_no_pytest_required(monkeypatch, tmp_path):
    """Test C: read-only tasks are verified by their own tool success; pytest is
    never forced on them."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "info.txt").write_text("hello", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "info.txt"}),
        final_answer("文件内容为 hello。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_successes == 1


def test_model_api_error_fails(monkeypatch, tmp_path):
    """Test F: a model API error terminates the run as failed (never succeeded)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()

    class RaisingLLM:
        def chat(self, messages, tools):
            raise RuntimeError("API down")

    outcome = _run(RaisingLLM(), recorder)
    assert outcome.status == "failed"
    # The exception is preserved as the error; the summary prefixes the cause.
    assert outcome.error == "API down"
    assert "模型调用失败" in outcome.summary


def test_tool_timeout_fails(monkeypatch):
    """Test F: a tool exceeding its timeout is reported as failed, not success."""
    from app.agents import agent_executor
    from app.tools.base import Tool, ToolContext

    def slow_handler(args, ctx):
        time.sleep(0.3)
        return {"ok": True, "output": "too late", "data": None}

    monkeypatch.setattr(
        agent_executor, "get_tool",
        lambda name: Tool(
            name=name, description="slow", parameters={},
            handler=slow_handler, requires_approval=lambda a, ws=None: False,
        ),
    )
    result, ok = agent_executor._run_tool_handler(
        "slow", {}, ToolContext(run_id="r1"), tool_timeout=0.05,
    )
    assert ok is False
    assert "超时" in result["output"]


def test_tool_nonzero_exit_records_failure(monkeypatch, tmp_path):
    """Test F: a non-zero command exit is recorded as a failed tool result with
    the real exit code handed back to the model."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "test_fail.py").write_text("def test_f():\n    assert False\n", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_fail.py", "-q"], "cwd": "."}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    tool_msgs = [m for m in llm.calls[-1][0] if m.get("role") == "tool"]
    assert any('"exit_code": 1' in m["content"] for m in tool_msgs)


# ---------------------------------------------------------------------------
# Task 1 regression tests: mutating commands record mutation, not verification.
# ---------------------------------------------------------------------------

def test_approved_mutating_command_records_mutation_not_verification(monkeypatch, tmp_path):
    """An approved file-mutating command (like ``ruff format``) whose handler
    succeeds must be recorded as a mutation, and its clean exit code must NOT
    satisfy verification: a final answer without a later passing verification
    fails."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)

    def mutating_handler(args, ctx):
        # Simulate the real side effect of a formatter: it rewrites a file.
        (tmp_path / "formatted.txt").write_text("mutated", encoding="utf-8")
        return tool_ok("已格式化。", data={"exit_code": 0})

    fake_tool = Tool(
        name="run_project_command",
        description="mutating",
        parameters={},
        handler=mutating_handler,
        requires_approval=lambda a, ws=None: True,
    )
    monkeypatch.setattr(agent_executor, "get_tool", lambda name: fake_tool)

    recorder = MemoryRecorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "ruff", "args": ["format", "."], "cwd": "."}),
        final_answer("已完成格式化。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert "验证" in outcome.error
    assert (tmp_path / "formatted.txt").exists()


# ---------------------------------------------------------------------------
# Task 2 regression tests: security rejections are tool failures, not crashes.
# ---------------------------------------------------------------------------

def test_security_rejection_recorded_as_failed_step(monkeypatch, tmp_path):
    """A disallowed command raises SecurityError; the executor records it as a
    failed tool step and hands the error back to the model (no crash)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "bash", "args": ["-c", "id"], "cwd": "."}),
        final_answer("无法执行，任务未完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    failed = [s for s in recorder.steps
              if s["kind"] == "tool" and s["tool_name"] == "run_project_command"]
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    assert "白名单" in failed[0]["output"]
    # The model received a tool message describing the rejection.
    assert any(m.get("role") == "tool" and "白名单" in m["content"]
               for m in llm.calls[-1][0])


def test_security_rejection_no_side_effect(monkeypatch, tmp_path):
    """``python -c <code>`` is rejected and its payload never executes."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python",
                   "args": ["-c", "import pathlib; pathlib.Path('pwned.txt').write_text('x')"],
                   "cwd": "."}),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    assert not (tmp_path / "pwned.txt").exists()


def test_security_rejection_then_model_recovers(monkeypatch, tmp_path):
    """After a security rejection, the model can continue and complete a safe
    task (the rejected command is not executed)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "info.txt").write_text("hello", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "bash", "args": ["-c", "x"], "cwd": "."}),
        tool_call("c2", "read_file", {"path": "info.txt"}),
        final_answer("改用安全方案读取文件成功。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_failures == 1
    assert outcome.tool_successes == 1


def test_unexpected_exception_in_requires_approval_propagates(monkeypatch, tmp_path):
    """Only SecurityError is caught; a real (non-security) exception in
    requires_approval still propagates as a system error, never a tool error."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)

    def boom(args, workspace=None):
        raise RuntimeError("unexpected bug")

    fake_tool = Tool(
        name="run_project_command",
        description="x",
        parameters={},
        handler=lambda args, ctx: tool_ok("x"),
        requires_approval=boom,
    )
    monkeypatch.setattr(agent_executor, "get_tool", lambda name: fake_tool)
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "ruff", "args": ["check"], "cwd": "."}),
    ])
    with pytest.raises(RuntimeError, match="unexpected bug"):
        _run(llm, recorder)


def test_batch_security_rejection_ordering(monkeypatch, tmp_path):
    """In a batch, a rejected command is not executed, while other safe calls
    still run and their results are recorded independently."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "read_file", {"path": "a.txt"}),
            ("c2", "run_project_command",
             {"program": "bash", "args": ["-c", "x"], "cwd": "."}),
        ]),
        final_answer("完成。"),
    ])
    outcome = _run(llm, recorder)
    assert outcome.status == "failed"
    steps = [s for s in recorder.steps if s["kind"] == "tool"]
    read = [s for s in steps if s["tool_name"] == "read_file"]
    cmd = [s for s in steps if s["tool_name"] == "run_project_command"]
    assert len(read) == 1 and read[0]["status"] == "succeeded"
    assert len(cmd) == 1 and cmd[0]["status"] == "failed"
