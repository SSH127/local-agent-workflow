"""Backend reliability regression tests.

Covers the run-level concurrency guard, orphan ``running`` run handling and
unsafe tool replay prevention, all without touching the real DeepSeek API, the
real filesystem (beyond a per-test temporary workspace) or the development
database.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from app import config, storage
from app.agents.agent_executor import run_agent
from app.engine import runner as runner_module
from app.engine.approvals import ApprovalDecision
from app.engine.runner import Runner
from app.schemas import ApprovalRequest
from app.tools import tool_schemas
from tests.conftest import FakeLLM, final_answer, multi_tool_call


def _save_run(run_id: str, status: str, state: dict[str, Any] | None = None) -> None:
    storage.init_db()
    storage.save_run({
        "run_id": run_id, "goal": "测试任务", "status": status,
        "attempts": 0, "summary": "", "error": None,
        "state": state or {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })


_RESULT = {
    "status": "succeeded", "attempts": 1, "summary": "ok", "error": None,
    "iterations": 1, "tool_calls": 1, "tool_successes": 1, "tool_failures": 0,
}


def _wait_until_inactive(runner: Runner, run_id: str, timeout: float = 5.0) -> None:
    """Deterministically wait for a run's thread to finish (not a race delay)."""
    deadline = time.monotonic() + timeout
    while runner.is_active(run_id) and time.monotonic() < deadline:
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Task 1: run-level concurrency guard
# ---------------------------------------------------------------------------

def test_submit_rejects_duplicate_active_run(monkeypatch, tmp_path):
    _save_run("r1", "queued")
    r = Runner()
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def fake_run_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return _RESULT

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)

    assert r.submit("r1", "测试任务") is True
    assert entered.wait(timeout=5)
    # While active, neither submit nor resume may start a second thread.
    assert r.submit("r1", "测试任务") is False
    assert r.resume("r1", "测试任务") is False
    release.set()
    _wait_until_inactive(r, "r1")
    assert len(calls) == 1
    assert not r.is_active("r1")


def test_concurrent_resume_starts_single_thread(monkeypatch, tmp_path):
    _save_run("r1", "waiting_for_approval", {"pending_tools": [], "agent_messages": []})
    r = Runner()
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def fake_run_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return _RESULT

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)

    barrier = threading.Barrier(2)
    results: list[bool] = []

    def do_resume() -> None:
        barrier.wait()
        results.append(r.resume("r1", "测试任务"))

    threads = [threading.Thread(target=do_resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one resume wins the atomic registration; one thread runs.
    assert results.count(True) == 1
    assert results.count(False) == 1
    assert entered.wait(timeout=5)
    assert len(calls) == 1
    release.set()
    _wait_until_inactive(r, "r1")


def test_submit_and_resume_race_single_thread(monkeypatch, tmp_path):
    _save_run("r1", "queued")
    r = Runner()
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def fake_run_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return _RESULT

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)

    barrier = threading.Barrier(2)
    results: list[bool] = []

    def do_submit() -> None:
        barrier.wait()
        results.append(r.submit("r1", "测试任务"))

    def do_resume() -> None:
        barrier.wait()
        results.append(r.resume("r1", "测试任务"))

    threads = [threading.Thread(target=do_submit), threading.Thread(target=do_resume)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert entered.wait(timeout=5)
    assert len(calls) == 1
    release.set()
    _wait_until_inactive(r, "r1")


def test_exception_releases_active_slot(monkeypatch, tmp_path):
    _save_run("r1", "queued")
    r = Runner()

    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "run_workflow", boom)

    assert r.submit("r1", "测试任务") is True
    _wait_until_inactive(r, "r1")
    # The exception exit path released the slot and marked the run failed.
    assert not r.is_active("r1")
    assert storage.get_run("r1")["status"] == "failed"
    # The same run can be submitted again (slot was freed).
    assert r.submit("r1", "测试任务") is True
    _wait_until_inactive(r, "r1")


# ---------------------------------------------------------------------------
# Task 1 (cont.): concurrent approval requests resume exactly once
# ---------------------------------------------------------------------------

def test_concurrent_approve_resumes_once(monkeypatch, tmp_path):
    from app import main

    _save_run(
        "r1", "waiting_for_approval",
        {"pending_approval": {
            "approval_id": "lost-id", "tool_name": "write_file",
            "args": {"path": ".env"}, "risk_reason": "覆盖受保护文件",
            "status": "pending",
        }},
    )
    # A fresh Runner has an empty in-memory gate, so approve() returns
    # "not_found" (simulating a restart that dropped the in-memory approval).
    monkeypatch.setattr(main, "runner", Runner())

    resume_calls: list[str] = []
    lock = threading.Lock()

    def fake_resume(run_id: str, goal: str) -> bool:
        with lock:
            resume_calls.append(run_id)
        return True

    monkeypatch.setattr(main.runner, "resume", fake_resume)

    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []

    def do_approve() -> None:
        barrier.wait()
        results.append(main.approve_run("r1", ApprovalRequest(approved=True)))

    threads = [threading.Thread(target=do_approve) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The atomic status transition lets only one request trigger a resume.
    assert len(resume_calls) == 1
    assert resume_calls == ["r1"]
    # Both callers observe the same, consistent resulting status.
    assert {r["status"] for r in results} == {"running"}


def test_approve_deny_transitions_to_failed(monkeypatch, tmp_path):
    from app import main

    _save_run(
        "r1", "waiting_for_approval",
        {"pending_approval": {
            "approval_id": "lost-id", "tool_name": "write_file",
            "args": {"path": ".env"}, "risk_reason": "覆盖受保护文件",
            "status": "pending",
        }},
    )
    monkeypatch.setattr(main, "runner", Runner())
    monkeypatch.setattr(main.runner, "resume", lambda run_id, goal: True)

    run = main.approve_run("r1", ApprovalRequest(approved=False, reason="不批准"))
    assert run["status"] == "failed"
    assert run["error"] == "不批准"


# ---------------------------------------------------------------------------
# Task 2: orphaned ``running`` runs across a restart
# ---------------------------------------------------------------------------

def test_mark_interrupted_runs_only_affects_running(monkeypatch, tmp_path):
    _save_run("r-running", "running")
    _save_run("r-succeeded", "succeeded")
    _save_run("r-failed", "failed")
    _save_run("r-approval", "waiting_for_approval")
    _save_run("r-input", "waiting_for_input")

    interrupted = storage.mark_interrupted_runs()
    assert interrupted == 1
    run = storage.get_run("r-running")
    assert run["status"] == "interrupted"
    assert "重启" in run["error"]
    assert "无法确认" in run["error"]
    # Other statuses are untouched.
    assert storage.get_run("r-succeeded")["status"] == "succeeded"
    assert storage.get_run("r-failed")["status"] == "failed"
    assert storage.get_run("r-approval")["status"] == "waiting_for_approval"
    assert storage.get_run("r-input")["status"] == "waiting_for_input"


# ---------------------------------------------------------------------------
# Task 3: unsafe tool replay prevention
# ---------------------------------------------------------------------------

def test_orphan_pending_tools_not_replayed(monkeypatch, tmp_path):
    """A side-effecting tool left in ``pending_tools`` after a crash must not be
    re-executed by the interrupt-recovery path."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "out.txt").write_text("already-written", encoding="utf-8")

    # Simulate a crash: the run is ``running`` and ``pending_tools`` still holds
    # a write that (in the worst case) may already have run.
    _save_run("r1", "running", {
        "pending_tools": [
            {"name": "write_file", "args": {"path": "out.txt", "content": "REPLAYED"}, "call_id": "c1"},
        ],
        "agent_messages": [],
    })

    interrupted = storage.mark_interrupted_runs()
    assert interrupted == 1
    run = storage.get_run("r1")
    assert run["status"] == "interrupted"
    # The pending write was NOT replayed.
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "already-written"


class _StateRecorder:
    """Captures persisted state snapshots to assert pending-tool cursor moves."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.state_calls: list[dict[str, Any]] = []
        self.approvals: list[tuple[str, dict[str, Any]]] = []

    def save_step(self, step: dict[str, Any]) -> None:
        return None

    def set_status(self, status: str, summary: str | None = None, error: str | None = None) -> None:
        return None

    def set_state(self, state: dict[str, Any]) -> None:
        self.state_calls.append(dict(state))

    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision:
        self.approvals.append((tool_name, args))
        return ApprovalDecision(approval_id=f"a{len(self.approvals)}", approved=self.approve)


def test_pending_cursor_advances_after_each_tool(monkeypatch, tmp_path):
    """After each approved tool completes, it is removed from the persisted
    pending set immediately, shrinking the crash-replay window."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    recorder = _StateRecorder(approve=True)
    # ``python -m pip --version`` requires approval (pip is always gated) but is
    # not a mutating command, so three of them complete cleanly without needing
    # a follow-up verification step.
    pip_call = {"program": "python", "args": ["-m", "pip", "--version"], "cwd": "."}
    llm = FakeLLM([
        multi_tool_call([
            ("c1", "run_project_command", pip_call),
            ("c2", "run_project_command", pip_call),
            ("c3", "run_project_command", pip_call),
        ]),
        final_answer("完成。"),
    ])
    outcome = run_agent("run-test", "测试任务", llm, tool_schemas(), recorder)
    assert outcome.status == "succeeded"
    assert outcome.tool_calls == 3

    pending_snapshots = [
        [p["call_id"] for p in s["pending_tools"]]
        for s in recorder.state_calls
        if isinstance(s.get("pending_tools"), list) and s["pending_tools"]
    ]
    # After c1 completes, the cursor advances to [c2, c3]; after c2, to [c3].
    assert ["c2", "c3"] in pending_snapshots
    assert ["c3"] in pending_snapshots
    # The batch is finally cleared.
    assert any(s.get("pending_tools") is None for s in recorder.state_calls)
