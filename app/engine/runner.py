"""Background run execution, approval gating and persistence wiring."""
from __future__ import annotations

import contextlib
import threading
from typing import Any

from app import config, storage
from app.agents import agent_executor
from app.agents.agent_executor import ResumeState, utc_now
from app.agents.llm import DeepSeekClient
from app.engine.approvals import APPROVAL_GRANTED, ApprovalDecision, ApprovalGate
from app.engine.workflow import StoragePlanStore, run_workflow


class StorageRecorder:
    """Persists steps/status/state to SQLite and routes approvals to the gate."""

    def __init__(self, run_id: str, gate: ApprovalGate) -> None:
        self.run_id = run_id
        self.gate = gate
        # Task graph + execution lease persistence for the control loop.
        self.plan_store = StoragePlanStore(run_id)

    def save_step(self, step: dict[str, Any]) -> None:
        storage.save_step(step)

    def set_status(self, status: str, summary: str | None = None, error: str | None = None) -> None:
        storage.update_run_status(self.run_id, status, summary=summary, error=error)

    def set_state(self, state: dict[str, Any]) -> None:
        storage.update_run_state(self.run_id, state)

    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision:
        record = self.gate.request(run_id, tool_name, args, risk_reason)
        # Persist the pending approval so a restart can discover and (re-)resolve it.
        storage.update_run_state(self.run_id, {"pending_approval": record.to_dict()})
        with contextlib.suppress(Exception):
            from app.events import publish_run_event
            publish_run_event(self.run_id, "approval",
                              approval_id=record.approval_id, tool_name=tool_name)
        storage.update_run_status(self.run_id, agent_executor.STATUS_WAITING_APPROVAL)
        final_status = self.gate.wait(record.approval_id, timeout=config.APPROVAL_WAIT_SECONDS)
        storage.update_run_status(self.run_id, agent_executor.STATUS_RUNNING)
        storage.update_run_state(self.run_id, {"pending_approval": None})
        return ApprovalDecision(
            approval_id=record.approval_id,
            approved=(final_status == APPROVAL_GRANTED),
            reason=record.reason,
        )

    def request_input(self, run_id: str, question: str) -> str | None:
        """Ask the user a question and block until answered (or timed out).

        Reuses the approval gate: the answer travels in the record's ``reason``.
        Returns the answer text, or ``None`` when no answer arrived in time.
        """
        record = self.gate.request(
            run_id, "ask_user", {"question": question}, "等待用户补充信息",
        )
        storage.update_run_state(self.run_id, {"pending_input": {
            "input_id": record.approval_id,
            "question": question,
            "created_at": record.created_at,
        }})
        with contextlib.suppress(Exception):
            from app.events import publish_run_event
            publish_run_event(self.run_id, "input", input_id=record.approval_id,
                              question=question[:200])
        storage.update_run_status(self.run_id, agent_executor.STATUS_WAITING_INPUT)
        final_status = self.gate.wait(record.approval_id, timeout=config.INPUT_WAIT_SECONDS)
        storage.update_run_status(self.run_id, agent_executor.STATUS_RUNNING)
        storage.update_run_state(self.run_id, {"pending_input": None})
        if final_status != APPROVAL_GRANTED:
            return None
        answer = (record.reason or "").strip()
        return answer or None


class Runner:
    """Starts runs on background threads and coordinates approvals.

    A per-run in-process registry (``_active_runs``) guarantees that a given
    ``run_id`` has at most one live execution thread at a time. ``submit`` and
    ``resume`` both funnel through the same atomic check-and-register step, so
    the check and the registration cannot be raced. The registration is released
    in ``finally`` so an exception anywhere in the run path always frees the
    slot. This protects a single process; it is *not* a cross-process mutex.
    """

    def __init__(self) -> None:
        self.gate = ApprovalGate()
        self._lock = threading.Lock()
        self._active_runs: set[str] = set()

    def _try_acquire(self, run_id: str) -> bool:
        """Atomically register ``run_id`` as active. Returns ``False`` if it is
        already running (the check and the add happen under one lock)."""
        with self._lock:
            if run_id in self._active_runs:
                return False
            self._active_runs.add(run_id)
            return True

    def _release(self, run_id: str) -> None:
        with self._lock:
            self._active_runs.discard(run_id)

    def is_active(self, run_id: str) -> bool:
        """Expose active-run state for deterministic concurrency tests."""
        with self._lock:
            return run_id in self._active_runs

    def submit(
        self, run_id: str, goal: str, max_attempts: int = 2, workspace: str | None = None
    ) -> bool:
        """Start a new run on a background thread. Returns ``True`` if a thread
        was started, or ``False`` if ``run_id`` is already executing (no second
        thread is started)."""
        if not self._try_acquire(run_id):
            return False
        thread = threading.Thread(
            target=self._run, args=(run_id, goal, max_attempts),
            kwargs={"workspace": workspace},
            daemon=True, name=f"run-{run_id}",
        )
        thread.start()
        return True

    def resume(self, run_id: str, goal: str) -> bool:
        """Resume a paused run from persisted conversation + pending tool.

        Returns ``True`` if a resume thread was started, or ``False`` if the run
        is already executing (no second thread is started) or does not exist.
        """
        if not self._try_acquire(run_id):
            return False
        run = storage.get_run(run_id)
        if run is None:
            self._release(run_id)
            return False
        state = run.get("state", {}) or {}
        messages = state.get("agent_messages") or []
        pending_tools = state.get("pending_tools") or []
        preapproved = state.get("approval_decision")
        memory = state.get("memory")
        token_stats = state.get("token_stats")
        cache_stats = state.get("cache_stats")
        checkpoint = state.get("checkpoint")
        user_input = state.get("user_input")
        # Prefer the run's validated workspace; for a legacy/partial record fall
        # back to the checkpoint's actual directory (never to the engine's own).
        checkpoint_workspace = (
            checkpoint.get("workspace") if isinstance(checkpoint, dict) else None
        )
        workspace = run.get("workspace_path") or checkpoint_workspace
        resume = ResumeState(
            messages=messages, pending_tools=pending_tools, preapproved=preapproved,
            memory=memory, token_stats=token_stats, cache_stats=cache_stats,
            checkpoint=checkpoint,
            user_input=user_input if isinstance(user_input, str) and user_input else None,
        )
        thread = threading.Thread(
            target=self._run, args=(run_id, goal, 2),
            kwargs={"resume": resume, "workspace": workspace},
            daemon=True, name=f"resume-{run_id}",
        )
        thread.start()
        return True

    def _run(
        self,
        run_id: str,
        goal: str,
        max_attempts: int,
        resume: ResumeState | None = None,
        workspace: str | None = None,
    ) -> None:
        # The active-run registration covers the *entire* thread lifetime,
        # including the final persistence write, so a caller can never observe
        # the slot as free while this thread is still writing its result.
        try:
            recorder = StorageRecorder(run_id, self.gate)
            llm = DeepSeekClient()
            storage.update_run_status(run_id, agent_executor.STATUS_RUNNING, summary="执行中")
            try:
                result = run_workflow(
                    goal, max_attempts,
                    run_id=run_id, llm=llm, recorder=recorder,
                    workspace=workspace, resume=resume,
                )
            except Exception as exc:  # noqa: BLE001 - unexpected crash in the executor
                storage.update_run_status(
                    run_id, agent_executor.STATUS_FAILED,
                    summary="执行异常", error=f"{type(exc).__name__}: {exc}",
                )
                return

            # Persist final status/attempts/summary/error in one write.
            run = storage.get_run(run_id)
            if run is not None:
                run["status"] = result.get("status", agent_executor.STATUS_FAILED)
                run["attempts"] = result.get("attempts", 0)
                run["summary"] = result.get("summary", "")
                run["error"] = result.get("error")
                run["updated_at"] = utc_now()
                state = dict(run.get("state") or {})
                state["iterations"] = result.get("iterations", 0)
                state["tool_calls"] = result.get("tool_calls", 0)
                state["tool_successes"] = result.get("tool_successes", 0)
                state["tool_failures"] = result.get("tool_failures", 0)
                state["phase_used"] = result.get("phase_used", {})
                state["tool_counts"] = result.get("tool_counts", {})
                state["stop_reason"] = result.get("stop_reason", "")
                state["plan"] = result.get("plan", [])
                # The executor persists a checkpoint on every terminal path
                # (including budget exhaustion); keep it in the run record so a
                # later resume can verify files and continue from unfinished
                # steps instead of replaying the task.
                checkpoint = result.get("checkpoint") or {}
                if checkpoint:
                    state["checkpoint"] = checkpoint
                run["state"] = state
                storage.save_run(run)
        finally:
            # Always release the run slot, including exception exit paths.
            self._release(run_id)

    def approve(self, approval_id: str, approved: bool, reason: str | None = None) -> str:
        """Resolve a pending approval by id. Returns
        ``"resolved"`` | ``"already_resolved"`` | ``"not_found"``."""
        return self.gate.resolve_if_pending(approval_id, approved, reason)

    def provide_input(self, input_id: str, message: str) -> str:
        """Deliver the user's answer to a pending ``ask_user`` question. Returns
        ``"resolved"`` | ``"already_resolved"`` | ``"not_found"``."""
        return self.gate.resolve_if_pending(input_id, True, message)

    def is_waiting(self, run_id: str) -> bool:
        return self.gate.get_pending_for_run(run_id) is not None
