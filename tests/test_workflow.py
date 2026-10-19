from app import config
from app.agents.llm import DeepSeekClient
from app.engine.approvals import ApprovalDecision
from app.engine.workflow import run_workflow
from tests.conftest import FakeLLM, final_answer, multi_tool_call, tool_call


class RecordingRecorder:
    """Captures steps/approvals so tests can assert on side effects."""

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
        return ApprovalDecision(
            approval_id=f"appr-{len(self.approvals)}", approved=self.approve,
        )


def test_workflow_finishes_successfully_with_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    llm = FakeLLM([
        tool_call("c1", "list_files", {}),
        final_answer("任务完成，已通过验证。"),
    ])
    # A read-only goal is allowed to succeed with analysis evidence only. A
    # change request ("add/fix/...") must produce a real modification instead
    # (covered by tests/test_execution_engine_repair.py).
    result = run_workflow("Analyze the login endpoint", max_attempts=2, llm=llm)
    assert result["status"] == "succeeded"
    assert result["attempts"] == 1
    assert result["summary"]


def test_workflow_fails_when_no_tool_executed():
    # A model that only talks without invoking any tool must not be marked
    # succeeded — prose is not evidence of execution.
    llm = FakeLLM([final_answer("任务已完成，一切正常。")])
    result = run_workflow("Add a login endpoint", max_attempts=1, llm=llm)
    assert result["status"] == "failed"
    assert result["error"]
    assert "未调用任何工具" in result["error"]


def test_workflow_fails_without_api_key():
    # A real DeepSeek client with no key must fail with a clear error, not
    # silently "succeed".
    llm = DeepSeekClient(api_key="")
    result = run_workflow("Analyze the repo", max_attempts=1, llm=llm)
    assert result["status"] == "failed"
    assert result["error"]


# ---------------------------------------------------------------------------
# Phase 3 regression tests (P0-2 unified retry)
# ---------------------------------------------------------------------------

def test_workflow_stops_at_iteration_budget(monkeypatch, tmp_path):
    """Test D: the run must stop at the inner iteration budget and fail cleanly,
    without infinite retries and without falsely reporting success."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 2)
    llm = FakeLLM([tool_call("c1", "list_files", {})])  # loops forever
    result = run_workflow("loop forever", max_attempts=5, llm=llm)
    assert result["status"] == "failed"
    assert result["iterations"] == 2
    assert "最大迭代次数" in (result["error"] or "")
    # No outer retry: a single attempt even though max_attempts=5.
    assert result["attempts"] == 1


def test_workflow_single_pass_no_duplicate_side_effects(monkeypatch, tmp_path):
    """Test E: a failing run must not be re-executed from scratch by the outer
    workflow — writes must happen exactly once."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "test_fail.py").write_text("def test_f():\n    assert False\n", encoding="utf-8")
    recorder = RecordingRecorder()
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": "out.txt", "content": "v1"}),
        tool_call("c2", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_fail.py", "-q"], "cwd": "."}),
        final_answer("完成。"),
    ])
    result = run_workflow("fix", max_attempts=3, llm=llm, recorder=recorder)
    assert result["status"] == "failed"
    assert result["attempts"] == 1  # single pass — no outer restart
    writes = [s for s in recorder.steps if s.get("tool_name") == "write_file"]
    assert len(writes) == 1
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "v1"


def test_workflow_no_duplicate_approval(monkeypatch, tmp_path):
    """Test E: an approval-requiring tool is only approved once (single pass)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = RecordingRecorder(approve=True)
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pip", "--version"], "cwd": "."}),
        final_answer("完成。"),
    ])
    result = run_workflow("install", max_attempts=2, llm=llm, recorder=recorder)
    assert result["attempts"] == 1
    assert len(recorder.approvals) == 1  # requested exactly once, never re-requested


def test_counting_semantics_are_distinct(monkeypatch, tmp_path):
    """Model iterations, tool calls and task attempts are distinct fields."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "read_file", {"path": "a.txt"}),
            ("c2", "list_files", {}),
        ]),
        final_answer("完成。"),
    ])
    result = run_workflow("goal", max_attempts=1, llm=llm)
    assert result["attempts"] == 1      # task attempts (single pass)
    assert result["iterations"] == 2    # model turns: 1 tool-call turn + final
    assert result["tool_calls"] == 2    # tool invocations
    assert result["tool_successes"] == 2


# ---------------------------------------------------------------------------
# Task 3 regression tests: default recorder protocol & safe-default deny.
# ---------------------------------------------------------------------------

def test_default_recorder_denies_directly():
    """The default recorder denies approval-requiring ops (never auto-approves)
    and returns the formal ApprovalDecision type."""
    from app.engine.workflow import _DefaultRecorder

    decision = _DefaultRecorder().request_approval(
        "r1", "write_file", {"path": ".env"}, "覆盖受保护文件",
    )
    assert isinstance(decision, ApprovalDecision)
    assert decision.approved is False


def test_default_recorder_no_type_error_on_approval_tool(monkeypatch, tmp_path):
    """Running a workflow without an explicit recorder that hits an
    approval-requiring tool fails closed (no TypeError)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "ruff", "args": ["format", "."], "cwd": "."}),
        final_answer("完成。"),
    ])
    result = run_workflow("format", max_attempts=1, llm=llm)  # no recorder
    assert result["status"] == "failed"
    assert result["error"]


def test_default_recorder_does_not_silently_approve(monkeypatch, tmp_path):
    """Without an approval channel the high-risk write must NOT execute."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    llm = FakeLLM([
        tool_call("c1", "write_file", {"path": ".env", "content": "TOKEN=1"}),
        final_answer("完成。"),
    ])
    result = run_workflow("write", max_attempts=1, llm=llm)  # no recorder
    assert result["status"] == "failed"
    assert not (tmp_path / ".env").exists()


def test_explicit_recorder_still_used_over_default(monkeypatch, tmp_path):
    """Passing an explicit recorder (as the real Runner does) keeps the normal
    approval flow; the safe-default recorder is only a fallback."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    decisions: list[tuple[str, dict]] = []

    class Rec:
        def save_step(self, step): return None
        def set_status(self, *a, **k): return None
        def set_state(self, state): return None
        def request_approval(self, run_id, tool_name, args, risk_reason):
            decisions.append((tool_name, args))
            return ApprovalDecision(approval_id="a1", approved=True)

    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pip", "--version"], "cwd": "."}),
        final_answer("完成。"),
    ])
    result = run_workflow("w", max_attempts=1, llm=llm, recorder=Rec())
    assert result["status"] == "succeeded"
    assert decisions and decisions[0][0] == "run_project_command"
