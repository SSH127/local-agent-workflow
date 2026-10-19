"""Regression tests for the "analysis-until-budget-exhausted" repair.

The engine used to let a run keep reading/listing/searching until the whole
iteration (or model-call) budget was gone, and then report only a summary with
no recoverable state. These tests pin the repaired behaviour:

* a read-only loop with no new information is detected and stopped *before* the
  total budget is exhausted, with a ``no_progress`` stop reason;
* a model-announced plan does not block the run from advancing to a real
  modification and verification;
* exhausting the model-call budget records a failed status plus a resumable
  checkpoint (goal/workspace/steps/files/tests/pending steps/stop reason);
* resuming from that checkpoint does not replay an already-applied write;
* an invalid target workspace is rejected *before* any model/tool activity and
  never falls back to the engine's own directory;
* a normal task still performs a real modification and passes verification;
* a change-request goal with no write is not reported as success.

No real DeepSeek API and no real network are used (autouse fixtures).
"""
from __future__ import annotations

import json
import time
from typing import Any

from app import config, storage
from app.agents.agent_executor import (
    STOP_COMPLETED,
    STOP_MODEL_CALL_BUDGET,
    STOP_MUTATION_REQUIRED,
    STOP_NO_PROGRESS,
    STOP_WORKSPACE_REJECTED,
    ResumeState,
    run_agent,
)
from app.agents.llm import LLMResponse
from app.engine import runner as runner_module
from app.engine.approvals import ApprovalDecision
from app.engine.workflow import run_workflow
from app.tools import tool_schemas
from tests.conftest import FakeLLM, final_answer, tool_call


class StatefulRecorder:
    """Capture steps plus the merged run state (checkpoints, stats)."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self.last_status: tuple[str, str | None, str | None] | None = None

    def save_step(self, step: dict[str, Any]) -> None:
        self.steps.append(step)

    def set_status(
        self, status: str, summary: str | None = None, error: str | None = None
    ) -> None:
        self.last_status = (status, summary, error)

    def set_state(self, state: dict[str, Any]) -> None:
        # ``set_state`` is called with partial updates; merge like storage does.
        self.state.update(state)

    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision:
        return ApprovalDecision(approval_id="appr-test", approved=True)


def _progress_steps(recorder: StatefulRecorder) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in recorder.steps:
        if step.get("kind") != "progress":
            continue
        try:
            out.append(json.loads(step.get("output") or "{}"))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# 1. No-progress detection: stop a duplicate read-only loop
# ---------------------------------------------------------------------------

def test_repeated_read_only_calls_are_detected_as_no_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    # Total budget is deliberately much larger than the no-progress limit: the
    # run must converge instead of consuming the whole budget.
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 40)
    monkeypatch.setattr(config, "AGENT_MAX_MODEL_CALLS", 40)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 40)
    monkeypatch.setattr(config, "AGENT_NO_PROGRESS_LIMIT", 2)
    (tmp_path / "f.txt").write_text("same content", encoding="utf-8")

    recorder = StatefulRecorder()
    # The fake repeats this identical read forever.
    llm = FakeLLM([tool_call("c1", "read_file", {"path": "f.txt"})])

    outcome = run_agent("run-test", "分析 f.txt", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert outcome.stop_reason == STOP_NO_PROGRESS
    assert "未产生进展" in (outcome.error or "")
    # It stopped well before the total budget: no endless analysis.
    assert outcome.iterations < config.AGENT_MAX_ITERATIONS
    assert len(llm.calls) < config.AGENT_MAX_ITERATIONS
    # Metrics prove *why* it stopped: repeated calls produced no new information.
    progress = _progress_steps(recorder)
    assert any(p.get("stalled_streak", 0) >= 1 for p in progress)
    assert any(p.get("duplicate_calls", 0) >= 1 for p in progress)
    assert any(s.get("status") == "hint" for s in recorder.steps)
    # A recoverable checkpoint is saved, not just a prose summary.
    checkpoint = outcome.checkpoint
    assert checkpoint["stop_reason"] == STOP_NO_PROGRESS
    assert checkpoint["status"] == "failed"
    assert checkpoint["pending_steps"]
    assert recorder.state.get("checkpoint", {}).get("stop_reason") == STOP_NO_PROGRESS


def test_duplicate_tool_limit_triggers_convergence(monkeypatch, tmp_path):
    """A repeated identical call is detected even when the read-only streak
    limit itself has not been reached yet."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 40)
    monkeypatch.setattr(config, "AGENT_MAX_MODEL_CALLS", 40)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 40)
    monkeypatch.setattr(config, "AGENT_NO_PROGRESS_LIMIT", 10)
    monkeypatch.setattr(config, "AGENT_DUPLICATE_TOOL_LIMIT", 2)
    (tmp_path / "f.txt").write_text("same content", encoding="utf-8")

    recorder = StatefulRecorder()
    llm = FakeLLM([tool_call("c1", "read_file", {"path": "f.txt"})])

    outcome = run_agent("run-test", "检查 f.txt", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert outcome.stop_reason == STOP_NO_PROGRESS
    # Duplicate-limit hint was sent before the abort.
    assert any(s.get("status") == "hint" for s in recorder.steps)
    progress = _progress_steps(recorder)
    assert any(p.get("duplicate_calls", 0) >= 1 for p in progress)
    assert outcome.iterations == 3  # two reads + the rejected third read


# ---------------------------------------------------------------------------
# 2. A model-emitted plan must not stall the run in the analyze phase
# ---------------------------------------------------------------------------

def test_plan_after_analysis_advances_to_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "src.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )

    # Turn 1: the model announces a plan *and* performs one read (analyze).
    plan_with_read = LLMResponse(
        content="实施计划：1. 写入 out.txt；2. 运行测试验证。",
        tool_calls=[{
            "id": "c1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "src.txt"}),
            },
        }],
    )
    recorder = StatefulRecorder()
    llm = FakeLLM([
        plan_with_read,
        tool_call("c2", "write_file", {"path": "out.txt", "content": "done"}),
        tool_call(
            "c3", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("已按计划完成修改并通过测试。"),
    ])

    # Run through the workflow so the plan node (plan -> execute) is exercised.
    result = run_workflow(
        "修复 out.txt", llm=llm, recorder=recorder, workspace=str(tmp_path),
    )

    assert result["status"] == "succeeded", result["error"]
    assert result["stop_reason"] == STOP_COMPLETED
    assert result["plan"], "the workflow must produce an executable plan"
    assert result["phase_used"].get("analyze", 0) >= 1
    assert result["phase_used"].get("modify", 0) >= 1
    assert result["phase_used"].get("verify", 0) >= 1
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done"
    # The plan is recorded as a real workflow step, before execution.
    plan_steps = [s for s in recorder.steps if s.get("kind") == "plan"]
    assert plan_steps
    assert "执行" in plan_steps[0]["output"]


# ---------------------------------------------------------------------------
# 3. Model-call budget exhaustion saves an incomplete, resumable checkpoint
# ---------------------------------------------------------------------------

def test_model_budget_exhaustion_saves_incomplete_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_MODEL_CALLS", 2)
    monkeypatch.setattr(config, "AGENT_NO_PROGRESS_LIMIT", 10)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 10)
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")

    recorder = StatefulRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "a.txt"}),
        tool_call("c2", "read_file", {"path": "b.txt"}),
    ])

    outcome = run_agent("run-test", "分析这两个文件", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert outcome.stop_reason == STOP_MODEL_CALL_BUDGET
    assert "预算耗尽" in (outcome.error or "")
    assert outcome.iterations == 2

    checkpoint = outcome.checkpoint
    assert checkpoint["status"] == "failed"
    assert checkpoint["stop_reason"] == STOP_MODEL_CALL_BUDGET
    assert checkpoint["workspace"] == str(tmp_path.resolve())
    assert checkpoint["goal"] == "分析这两个文件"
    assert checkpoint["model_calls"] == 2
    assert checkpoint["pending_steps"], "must tell the resume what remains"
    # Persisted through the recorder (run state), not just returned in memory.
    assert recorder.state["stop_reason"] == STOP_MODEL_CALL_BUDGET
    assert recorder.state["checkpoint"]["stop_reason"] == STOP_MODEL_CALL_BUDGET


# ---------------------------------------------------------------------------
# 4. Resume verifies files and does not replay an already-applied modification
# ---------------------------------------------------------------------------

def test_resume_does_not_reapply_already_completed_write(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "out.txt").write_text("old", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )

    # First attempt: one model call is enough to apply the write, then the
    # model-call budget is exhausted before verification.
    monkeypatch.setattr(config, "AGENT_MAX_MODEL_CALLS", 1)
    first_recorder = StatefulRecorder()
    first_llm = FakeLLM([
        tool_call("c1", "write_file", {"path": "out.txt", "content": "done"}),
    ])
    first = run_agent(
        "run-first", "修复 out.txt", first_llm, tool_schemas(), first_recorder,
    )

    assert first.status == "failed"
    assert first.stop_reason == STOP_MODEL_CALL_BUDGET
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done"
    checkpoint = first.checkpoint
    assert any(item["path"] == "out.txt" for item in checkpoint["modified_files"])

    # Resume: the model re-issues the *same* write (a common model behaviour).
    # It must be recognised as already applied and skipped, then verified.
    resume_llm = FakeLLM([
        tool_call("c2", "write_file", {"path": "out.txt", "content": "done"}),
        tool_call(
            "c3", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("确认修改已生效并通过测试。"),
    ])
    # A resume starts a fresh model-call budget (prior usage stays in stats).
    monkeypatch.setattr(config, "AGENT_MAX_MODEL_CALLS", 10)
    resume = ResumeState(checkpoint=checkpoint, memory=checkpoint.get("memory"))
    recorder = StatefulRecorder()

    outcome = run_agent(
        "run-resume", "修复 out.txt", resume_llm, tool_schemas(), recorder,
        resume=resume,
    )

    assert outcome.status == "succeeded", outcome.error
    assert outcome.stop_reason == STOP_COMPLETED
    # The write was acknowledged from the checkpoint, never executed again.
    skipped = [
        s for s in recorder.steps
        if s.get("kind") == "checkpoint" and s.get("status") == "skipped"
    ]
    assert len(skipped) == 1
    assert "跳过重复执行" in skipped[0]["output"]
    assert not [
        s for s in recorder.steps
        if s.get("kind") == "tool" and s.get("tool_name") == "write_file"
    ]
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done"
    # The resume prompt carries the checkpoint's pending steps, so the model is
    # told what remains instead of re-analyzing from scratch.
    resume_prompt = "\n".join(
        str(m.get("content", "")) for m in resume_llm.calls[0][0]
    )
    assert "这是中断后恢复的任务" in resume_prompt
    assert "待执行步骤" in resume_prompt


def test_runner_resume_passes_checkpoint_and_uses_its_workspace(monkeypatch, tmp_path):
    """The background runner must wire the persisted checkpoint into the resume
    and fall back to its actual workspace when the run row has none."""
    storage.init_db()
    workspace = tmp_path / "proj"
    workspace.mkdir()
    checkpoint = {
        "goal": "修复 x",
        "workspace": str(workspace),
        "status": "failed",
        "stop_reason": STOP_MODEL_CALL_BUDGET,
        "pending_steps": ["运行测试验证"],
    }
    storage.save_run({
        "run_id": "run-wiring", "goal": "修复 x", "status": "failed",
        "attempts": 1, "summary": "", "error": "budget",
        "workspace_path": None,
        "state": {"checkpoint": checkpoint},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })

    seen: dict[str, Any] = {}

    def fake_run_workflow(goal, max_attempts=2, *, run_id=None, llm=None,
                          recorder=None, workspace=None, resume=None):
        seen["workspace"] = workspace
        seen["checkpoint"] = resume.checkpoint if resume is not None else None
        return {
            "status": "failed", "attempts": 1, "summary": "", "error": "budget",
            "iterations": 0, "tool_calls": 0, "tool_successes": 0,
            "tool_failures": 0, "phase_used": {}, "tool_counts": {},
            "stop_reason": STOP_MODEL_CALL_BUDGET, "checkpoint": checkpoint,
        }

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)
    runner = runner_module.Runner()
    assert runner.resume("run-wiring", "修复 x") is True
    deadline = time.monotonic() + 5
    while runner.is_active("run-wiring") and time.monotonic() < deadline:
        time.sleep(0.001)

    assert seen["workspace"] == str(workspace)
    assert seen["checkpoint"]["stop_reason"] == STOP_MODEL_CALL_BUDGET
    persisted = storage.get_run("run-wiring")
    assert persisted["state"]["stop_reason"] == STOP_MODEL_CALL_BUDGET
    assert persisted["state"]["checkpoint"]["stop_reason"] == STOP_MODEL_CALL_BUDGET


# ---------------------------------------------------------------------------
# 5. Invalid workspace is rejected before any model/tool call, no fallback
# ---------------------------------------------------------------------------

def test_invalid_workspace_is_rejected_without_fallback(monkeypatch, tmp_path):
    engine_workspace = tmp_path / "engine"
    engine_workspace.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_PATH", engine_workspace)
    missing = tmp_path / "does-not-exist"

    recorder = StatefulRecorder()
    llm = FakeLLM([final_answer("完成。")])

    outcome = run_agent(
        "run-invalid", "修复项目", llm, tool_schemas(), recorder,
        workspace=str(missing),
    )

    assert outcome.status == "failed"
    assert outcome.stop_reason == STOP_WORKSPACE_REJECTED
    assert "工作区" in (outcome.error or "")
    # No model call and no tool execution happened at all.
    assert llm.calls == []
    assert outcome.tool_calls == 0
    assert not missing.exists()
    # The rejected path is recorded; the engine's own directory was never used.
    checkpoint = outcome.checkpoint
    assert checkpoint["workspace"] == str(missing.resolve())
    workspace_steps = [s for s in recorder.steps if s.get("kind") == "workspace"]
    assert workspace_steps
    assert str(engine_workspace) not in workspace_steps[0]["output"]
    assert recorder.state["stop_reason"] == STOP_WORKSPACE_REJECTED


# ---------------------------------------------------------------------------
# 6. Normal task: real modification + passing verification, real success status
# ---------------------------------------------------------------------------

def test_normal_task_modifies_and_verifies(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "src.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )

    recorder = StatefulRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "src.txt"}),
        tool_call("c2", "write_file", {"path": "out.txt", "content": "done"}),
        tool_call(
            "c3", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("已修改并通过验证。"),
    ])

    outcome = run_agent("run-test", "修复 out.txt", llm, tool_schemas(), recorder)

    assert outcome.status == "succeeded", outcome.error
    assert outcome.stop_reason == STOP_COMPLETED
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done"
    checkpoint = outcome.checkpoint
    assert checkpoint["status"] == "succeeded"
    assert checkpoint["stop_reason"] == STOP_COMPLETED
    assert any(
        item["path"] == "out.txt" and item["sha256"]
        for item in checkpoint["modified_files"]
    )
    assert checkpoint["test_results"]
    assert checkpoint["test_results"][-1]["ok"] is True


# ---------------------------------------------------------------------------
# 7. A change request with no real write is not success
# ---------------------------------------------------------------------------

def test_mutation_goal_without_write_is_not_reported_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "src.txt").write_text("hello", encoding="utf-8")

    recorder = StatefulRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "src.txt"}),
        final_answer("我已经分析完毕，问题应该修复了。"),
    ])

    outcome = run_agent("run-test", "修复登录 bug", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert outcome.stop_reason == STOP_MUTATION_REQUIRED
    assert "未实际修改任何文件" in (outcome.error or "")
    assert outcome.checkpoint["status"] == "failed"
    # The pending steps point the resume at a minimal executable change.
    assert any("修改" in step for step in outcome.checkpoint["pending_steps"])
