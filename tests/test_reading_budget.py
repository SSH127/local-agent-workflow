"""Reading-efficiency and phase-budget tests.

Verifies the fixes for "the agent exhausts its iteration budget while still
reading files":

- ``read_file`` chunks a large file, returns total/actual range, and *explicitly*
  reports truncation with a continuation offset (never a silent cut).
- Duplicate reads of the same range are detected and reported via the run's
  shared ``ReadTracker``.
- Non-overlapping chunked reads can cover a whole file without re-reading.
- ``search_code`` can return optional context lines (but not whole files).
- A phase budget (analyze/modify/verify) is enforced independently, with a clear
  phase-specific failure reason, and is always bounded by the total budget.
- A normal read -> modify -> verify sequence still completes and records its
  phase/tool accounting.

No real DeepSeek API, no real network, temp directories only (autouse fixtures).
"""
from __future__ import annotations

from typing import Any

from app import config
from app.agents.agent_executor import run_agent
from app.tools import tool_schemas
from app.tools.base import ReadTracker, ToolContext
from app.tools.registry import get_tool
from tests.conftest import FakeLLM, final_answer, tool_call


class MemoryRecorder:
    def __init__(self, approve: bool = True):
        self.steps: list[dict[str, Any]] = []
        self.approve = approve

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        return None

    def set_state(self, state):
        return None

    def request_approval(self, run_id, tool_name, args, risk_reason):
        from app.engine.approvals import ApprovalDecision

        return ApprovalDecision(approval_id="a1", approved=self.approve)


# ---------------------------------------------------------------------------
# read_file: chunking, truncation hint, duplicate detection
# ---------------------------------------------------------------------------

def test_read_file_chunks_large_file_with_truncation_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "READ_FILE_MAX_CHARS", 200)
    (tmp_path / "big.py").write_text(
        "\n".join(f"line {i}" for i in range(60)), encoding="utf-8"
    )
    read = get_tool("read_file")
    result = read.handler(
        {"path": "big.py"}, ToolContext(run_id="r", workspace=tmp_path)
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["total_lines"] == 60
    # The requested range exceeded the char budget, so the read was truncated
    # *explicitly* (with a continuation offset), never silently.
    assert data["truncated"] is True
    assert data["next_offset"] is not None
    assert data["next_offset"] > data["start_line"]
    assert "截断" in result["output"]


def test_read_file_detects_duplicate_read(monkeypatch, tmp_path):
    (tmp_path / "f.txt").write_text(
        "\n".join(f"l{i}" for i in range(30)), encoding="utf-8"
    )
    read = get_tool("read_file")
    tracker = ReadTracker()
    ctx = ToolContext(run_id="r", workspace=tmp_path, read_tracker=tracker)
    first = read.handler({"path": "f.txt"}, ctx)
    assert first["ok"] is True and first["data"]["overlap_lines"] == 0
    second = read.handler({"path": "f.txt"}, ctx)
    assert second["ok"] is True
    assert second["data"]["overlap_lines"] > 0
    assert "此前已读取" in second["output"]
    assert second["data"]["already_read_ranges"] == [(1, 30)]


def test_read_file_chunks_cover_whole_file_without_rereading(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "READ_FILE_MAX_CHARS", 100)
    (tmp_path / "f.txt").write_text(
        "\n".join(f"l{i}" for i in range(40)), encoding="utf-8"
    )
    read = get_tool("read_file")
    tracker = ReadTracker()
    ctx = ToolContext(run_id="r", workspace=tmp_path, read_tracker=tracker)

    offset = 1
    covered = 0
    for _ in range(100):  # safety bound; should finish in ~3 chunks
        result = read.handler({"path": "f.txt", "offset": offset}, ctx)
        assert result["ok"] is True
        data = result["data"]
        covered += data["end_line"] - data["start_line"] + 1
        if not data["truncated"] and not data["has_more"]:
            break
        offset = data["next_offset"]
        assert offset is not None

    assert covered == 40
    # All chunks merged into one continuous range — nothing re-read twice.
    assert tracker.covered("f.txt") == [(1, 40)]


def test_search_code_returns_context_not_whole_file(monkeypatch, tmp_path):
    (tmp_path / "m.py").write_text(
        "alpha\nneedle = 1\nomega\n", encoding="utf-8"
    )
    search = get_tool("search_code")
    result = search.handler(
        {"query": "needle", "context_lines": 1},
        ToolContext(run_id="r", workspace=tmp_path),
    )
    assert result["ok"] is True
    match = result["data"]["matches"][0]
    assert match["line"] == 2
    # Context includes the surrounding lines, not the whole file.
    context_lines = [c["line"] for c in match["context"]]
    assert 1 in context_lines and 3 in context_lines


# ---------------------------------------------------------------------------
# Phase budgets in the ReAct loop
# ---------------------------------------------------------------------------

def test_analyze_phase_budget_exhaustion(monkeypatch, tmp_path):
    """A run stuck reading exhausts its analyze budget and fails with a clear
    phase-specific reason instead of a generic "max iterations" message."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 20)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 2)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    recorder = MemoryRecorder()
    llm = FakeLLM([tool_call("c1", "read_file", {"path": "f.txt"})])  # repeats forever
    outcome = run_agent("run-test", "read", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert "阅读与分析阶段预算耗尽" in outcome.error
    assert outcome.phase_used.get("analyze", 0) == 2
    # A phase step records the steering hint.
    assert any(s["kind"] == "phase" for s in recorder.steps)


def test_total_budget_bounds_phase_budgets(monkeypatch, tmp_path):
    """Phase budgets are clamped to the total, so a huge phase budget never lets
    the run overshoot ``max_iterations``."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 3)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 100)
    recorder = MemoryRecorder()
    llm = FakeLLM([tool_call("c1", "list_files", {})])  # repeats forever
    outcome = run_agent("run-test", "list", llm, tool_schemas(), recorder)

    assert outcome.status == "failed"
    assert outcome.iterations == 3
    assert "最大迭代次数" in outcome.error
    # analyze was clamped to the total (3), so it never "exhausted" early.
    assert outcome.phase_used.get("analyze", 0) == 3


def test_read_then_modify_then_verify_proceeds(monkeypatch, tmp_path):
    """A normal fix flow moves through analyze -> modify -> verify and succeeds,
    recording per-phase and per-tool accounting (not blocked by phase budgets)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "src.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "src.txt"}),
        tool_call("c2", "write_file", {"path": "out.txt", "content": "done"}),
        tool_call(
            "c3", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("已修改并通过验证。"),
    ])
    outcome = run_agent("run-test", "fix", llm, tool_schemas(), recorder)

    assert outcome.status == "succeeded"
    assert outcome.phase_used["analyze"] >= 1
    assert outcome.phase_used["modify"] >= 1
    assert outcome.phase_used["verify"] >= 1
    assert outcome.tool_counts.get("read_file") == 1
    assert outcome.tool_counts.get("write_file") == 1
    assert outcome.tool_counts.get("run_project_command") == 1
    # Workspace + phase steps are present in the log.
    assert any(s["kind"] == "workspace" for s in recorder.steps)
    assert any(s["kind"] == "phase" for s in recorder.steps)


def test_reads_after_mutation_do_not_abort_on_analyze_budget(monkeypatch, tmp_path):
    """Once real code has been changed, re-reading is normal work.

    Regression: an exhausted analyze budget used to abort the run on the next
    read even though the modification had already happened, with the misleading
    reason "仍未进入修改或验证阶段".
    """
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "AGENT_MAX_ITERATIONS", 20)
    monkeypatch.setattr(config, "AGENT_PHASE_BUDGET_ANALYZE", 2)
    (tmp_path / "src.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    recorder = MemoryRecorder()
    llm = FakeLLM([
        tool_call("c1", "read_file", {"path": "src.txt"}),
        tool_call("c2", "read_file", {"path": "src.txt"}),  # budget hit, hint sent
        tool_call("c3", "write_file", {"path": "out.txt", "content": "done"}),
        tool_call("c4", "read_file", {"path": "src.txt"}),  # would previously abort
        tool_call(
            "c5", "run_project_command",
            {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."},
        ),
        final_answer("已修改并通过验证。"),
    ])
    outcome = run_agent("run-test", "fix", llm, tool_schemas(), recorder)

    assert outcome.status == "succeeded", outcome.error
    assert outcome.phase_used["analyze"] > 2
    assert outcome.tool_counts.get("read_file") == 3
