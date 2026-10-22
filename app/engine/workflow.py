"""Fixed control loop (LangGraph) driving a per-goal task graph.

Layer 1 — the control loop below is FIXED and never rewritten per goal::

    decompose ─▶ await_plan_approval ─(approve)▶ pick_next ─▶ execute_subtask
        ▲              │(reject+feedback)            │  ▲          │
        └──────────────┘                             │  └─(wait)───┤
                                                     │             ▼
                                                     │       verify_subtask
                                                     │   accept │   │ reject
                                                     ◀──────────┘   ▼
                                     transient: retry same subtask (attempt+1)
                                     rewrite:   rewrite remaining (same version, new revision)
                                     structural: bump plan_version → approval
    pick_next ─(all done)▶ final_verify ─▶ finish | fail     (separate exit)
    any breaker / blocked deps ─▶ needs_attention;  cancel ─▶ cancelled

Layer 2 — the task graph (``app/engine/planning.py``) is generated per goal
and is the only thing the planner / replanner / user may change.

Parallel execution (``MAX_PARALLEL``): the graph itself stays single-threaded
— every node (scheduling, verification, replanning, persistence of the plan)
runs on the run's own thread. Only the ReAct executor calls
(``agent_executor.run_agent``) are submitted to a per-run thread pool owned by
:class:`RunContext`. ``pick_next`` launches up to ``MAX_PARALLEL`` ready
subtasks whose ``write_scope`` overlap neither each other nor a subtask in
flight; ``execute_subtask`` submits them and waits for the *first* one to
finish, which is then verified while the others keep running. Each subtask
holds its own execution lease with a background heartbeat. A failure that
needs a replan is deferred until nothing is in flight and no unrelated ready
subtask is left, so it never blocks independent branches; pauses (plan
approval, high-risk confirmation, attention) wait for in-flight subtasks to
drain first. Cancel sets a cooperative CancelToken so in-flight agent loops
and run_project_command process trees stop promptly, then drains leftovers.

Pauses end the graph run; the API resumes it later from the persisted plan, so
nothing blocks a thread and the state survives a restart. Tool approvals /
ask_user inside a subtask keep the executor's blocking-gate behaviour (one
prompt at a time per run).
"""
from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app import config
from app.agents import agent_executor
from app.agents.agent_executor import AgentOutcome, ResumeState, goal_requires_mutation
from app.agents.llm import DeepSeekClient
from app.cancel import CancelToken
from app.engine import planning as P
from app.engine.approvals import ApprovalDecision
from app.tools import tool_schemas
from app.write_scope import WriteScope

# Identity of this process for execution leases.
RUNNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"

STATUS_PAUSED_RESULTS = {
    agent_executor.STATUS_WAITING_PLAN_APPROVAL,
    agent_executor.STATUS_WAITING_SUBTASK_APPROVAL,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Plan stores (task-graph persistence)
# ---------------------------------------------------------------------------

class MemoryPlanStore:
    """In-process store used when no persistent recorder is supplied."""

    def __init__(self) -> None:
        self.plan: dict[str, Any] | None = None
        self.leases: dict[str, dict[str, Any]] = {}
        self.exec_states: dict[str, dict[str, Any]] = {}
        self.versions: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any] | None:
        import copy
        with self._lock:
            return copy.deepcopy(self.plan) if self.plan is not None else None

    def save(self, plan: dict[str, Any]) -> None:
        import copy
        with self._lock:
            old = self.plan or {}
            if old.get("goal_clauses") is not None and plan.get("goal_clauses") is not None \
                    and (old.get("goal_clauses"), old.get("goal")) != \
                    (plan.get("goal_clauses"), plan.get("goal")):
                raise ValueError("任务图的原始目标与目标条款已冻结，不能修改。")
            self.plan = copy.deepcopy(plan)

    def get_lease(self, subtask_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if subtask_id is not None:
                return dict(self.leases[subtask_id]) if subtask_id in self.leases else None
            if not self.leases:
                return None
            return dict(max(self.leases.values(), key=lambda x: str(x.get("heartbeat_at"))))

    def all_leases(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self.leases.items()}

    def set_lease(self, lease: dict[str, Any] | None) -> None:
        if lease is None or not lease.get("subtask_id"):
            return
        with self._lock:
            self.leases[lease["subtask_id"]] = dict(lease)

    def acquire(self, lease: dict[str, Any], expected_id: str | None) -> bool:
        with self._lock:
            current = self.leases.get(lease["subtask_id"])
            current_id = current.get("lease_id") if current else None
            if current_id != expected_id:
                return False
            self.leases[lease["subtask_id"]] = dict(lease)
            return True

    def get_exec_state(self, subtask_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self.exec_states.get(subtask_id) or {})

    def save_exec_state(self, subtask_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self.exec_states[subtask_id] = dict(state)

    def append_version(self, entry: dict[str, Any]) -> bool:
        with self._lock:
            key = (entry["version"], entry.get("revision"))
            if any((v["version"], v.get("revision")) == key for v in self.versions):
                return False
            self.versions.append(dict(entry))
            return True

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self.versions]


class StoragePlanStore:
    """SQLite-backed store (``run_plans`` / ``subtask_leases`` / ``plan_versions``)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def load(self) -> dict[str, Any] | None:
        from app import storage
        return storage.get_plan(self.run_id)

    def save(self, plan: dict[str, Any]) -> None:
        from app import storage
        storage.save_plan(self.run_id, plan)

    def get_lease(self, subtask_id: str | None = None) -> dict[str, Any] | None:
        from app import storage
        return storage.get_lease(self.run_id, subtask_id)

    def all_leases(self) -> dict[str, dict[str, Any]]:
        from app import storage
        return storage.get_leases(self.run_id)

    def set_lease(self, lease: dict[str, Any] | None) -> None:
        from app import storage
        storage.set_lease(self.run_id, lease)

    def acquire(self, lease: dict[str, Any], expected_id: str | None) -> bool:
        from app import storage
        return storage.acquire_lease(self.run_id, lease, expected_lease_id=expected_id)

    def get_exec_state(self, subtask_id: str) -> dict[str, Any]:
        from app import storage
        return storage.get_exec_state(self.run_id, subtask_id)

    def save_exec_state(self, subtask_id: str, state: dict[str, Any]) -> None:
        from app import storage
        storage.save_exec_state(self.run_id, subtask_id, state)

    def append_version(self, entry: dict[str, Any]) -> bool:
        from app import storage
        return storage.append_plan_version(self.run_id, entry)

    def list_versions(self) -> list[dict[str, Any]]:
        from app import storage
        return storage.list_plan_versions(self.run_id)


# ---------------------------------------------------------------------------
# Per-run concurrency context
# ---------------------------------------------------------------------------

@dataclass
class Launch:
    subtask_id: str
    attempt: int
    lease: dict[str, Any]
    recorder: Any
    scope: WriteScope | None
    budget_steps: int
    order: int
    future: Future | None = None
    started: float = 0.0


@dataclass
class RunContext:
    """Thread pool + thread-safe counters for one run_workflow invocation."""

    run_id: str
    store: Any
    max_parallel: int
    lock: threading.RLock = field(default_factory=threading.RLock)
    lease_lock: threading.RLock = field(default_factory=threading.RLock)
    prompt_lock: threading.Lock = field(default_factory=threading.Lock)
    waiting: int = 0
    launches: dict[str, Launch] = field(default_factory=dict)
    live_tokens: dict[str, int] = field(default_factory=dict)
    live_steps: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    busy_since: float | None = None
    max_concurrency_seen: int = 0
    cancel: CancelToken = field(default_factory=CancelToken)
    _seq: int = 0
    _pool: ThreadPoolExecutor | None = None
    _hb_stop: threading.Event = field(default_factory=threading.Event)
    _hb_thread: threading.Thread | None = None

    # -- pool / heartbeat ----------------------------------------------------
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, self.max_parallel),
                thread_name_prefix=f"subtask-{self.run_id[:8]}",
            )
        if self._hb_thread is None:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True,
                name=f"lease-heartbeat-{self.run_id[:8]}",
            )
            self._hb_thread.start()
        return self._pool

    def _heartbeat_loop(self) -> None:
        interval = max(0.05, float(config.LEASE_HEARTBEAT_SECONDS))
        while not self._hb_stop.wait(interval):
            for launch in list(self.launches.values()):
                with contextlib.suppress(Exception):  # a heartbeat must never crash
                    launch.recorder.heartbeat()

    def close(self) -> None:
        self._hb_stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    # -- bookkeeping ---------------------------------------------------------
    def in_flight(self) -> set[str]:
        return set(self.launches)

    def next_order(self) -> int:
        self._seq += 1
        return self._seq

    def reserved_steps(self) -> int:
        return sum(launch.budget_steps for launch in self.launches.values())

    def add_live_step(self, sid: str) -> None:
        with self.lock:
            self.live_steps[sid] = self.live_steps.get(sid, 0) + 1

    def set_live_tokens(self, sid: str, total: int) -> None:
        with self.lock:
            self.live_tokens[sid] = int(total)

    def mark_started(self) -> None:
        if self.busy_since is None:
            self.busy_since = time.monotonic()
        self.max_concurrency_seen = max(self.max_concurrency_seen, len(self.launches))

    def busy_elapsed(self) -> float:
        return time.monotonic() - self.busy_since if self.busy_since is not None else 0.0

    def finish(self, sid: str, totals: dict[str, Any]) -> None:
        """Called on the control-loop thread when a launch has been collected."""
        self.launches.pop(sid, None)
        with self.lock:
            self.live_tokens.pop(sid, None)
            self.live_steps.pop(sid, None)
        if not self.launches and self.busy_since is not None:
            totals["active_seconds"] = round(
                float(totals.get("active_seconds") or 0) + self.busy_elapsed(), 3)
            self.busy_since = None

    def live_totals(self, plan: dict[str, Any]) -> tuple[int, float, int]:
        totals = plan.get("totals") or {}
        with self.lock:
            steps = int(totals.get("steps") or 0) + sum(self.live_steps.values())
            tokens = int(totals.get("tokens") or 0) + sum(self.live_tokens.values())
        seconds = float(totals.get("active_seconds") or 0) + self.busy_elapsed()
        return steps, seconds, tokens


# ---------------------------------------------------------------------------
# Recorder wrapper: per-subtask attribution, heartbeat, tokens, interventions
# ---------------------------------------------------------------------------

class SubtaskRecorder:
    def __init__(
        self, inner: Any, store: Any, run_id: str, subtask: dict[str, Any],
        *, legacy: bool, on_intervention: Any, ctx: RunContext,
    ) -> None:
        self._inner = inner
        self._store = store
        self._run_id = run_id
        self._subtask = subtask
        self._sid = subtask["id"]
        self._attempt = int(subtask.get("attempt_number") or 0)
        self._legacy = legacy
        self._on_intervention = on_intervention
        self._ctx = ctx
        self._last_beat = 0.0
        self.token_stats: dict[str, Any] = {}
        self.exec_state: dict[str, Any] = {} if legacy else dict(store.get_exec_state(self._sid))
        if hasattr(inner, "request_input"):
            self.request_input = self._request_input  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _owns_current_attempt(self) -> bool:
        """True while this attempt still owns the subtask's execution lease.

        A takeover bumps the attempt number and installs a new lease. From that
        moment this recorder belongs to a superseded attempt: anything it writes
        (conversation state, run status) is a late result and must not touch the
        newer attempt's state. Read paths (tool steps) are still recorded, since
        the work really happened and the step carries its attempt number.
        """
        return P.lease_matches(self._store.get_lease(self._sid), self._sid, self._attempt)

    def heartbeat(self, state: str | None = None) -> None:
        with self._ctx.lease_lock:
            lease = self._store.get_lease(self._sid)
            if not P.lease_matches(lease, self._sid, self._attempt):
                return
            if lease.get("state") == "released":
                return
            lease = dict(lease)
            lease["heartbeat_at"] = utc_now()
            if state:
                lease["state"] = state
            self._store.set_lease(lease)

    def _maybe_beat(self) -> None:
        now = time.monotonic()
        if now - self._last_beat >= config.LEASE_HEARTBEAT_SECONDS:
            self._last_beat = now
            self.heartbeat()

    def save_step(self, step: dict[str, Any]) -> None:
        step = dict(step)
        step.setdefault("subtask_id", self._sid)
        step.setdefault("attempt_number", self._attempt)
        self._inner.save_step(step)
        if step.get("kind") == "model":
            self._ctx.add_live_step(self._sid)
        self._maybe_beat()

    def set_status(self, status: str, summary: str | None = None, error: str | None = None) -> None:
        if self._legacy:
            self._inner.set_status(status, summary=summary, error=error)
            return
        if not self._owns_current_attempt():
            # A superseded attempt ending must not overwrite the status the
            # newer attempt (or a terminal run state) already published.
            return
        if status in (agent_executor.STATUS_SUCCEEDED, agent_executor.STATUS_FAILED,
                      agent_executor.STATUS_CANCELLED, agent_executor.STATUS_RUNNING):
            # A subtask ending is not the run ending; and while a sibling waits
            # for the user, its "waiting_for_*" status must not be overwritten.
            if self._ctx.waiting:
                return
            label = "执行中" if status == agent_executor.STATUS_RUNNING else f"执行结束（{status}），等待验收"
            self._inner.set_status(agent_executor.STATUS_RUNNING,
                                   summary=f"子任务 [{self._sid}] {label}")
            return
        self._inner.set_status(status, summary=summary, error=error)

    def set_state(self, state: dict[str, Any]) -> None:
        stats = state.get("token_stats")
        if isinstance(stats, dict):
            # Token spend is real even for a superseded attempt, so keep
            # accounting for it before deciding whether the state may persist.
            self.token_stats = dict(stats)
            self._ctx.set_live_tokens(self._sid, _tokens_from(stats)["total"])
        if self._legacy:
            self._inner.set_state(state)
            return
        if not self._owns_current_attempt():
            # Never persist a superseded conversation: the next resume would
            # replay the old attempt's messages instead of the newer one's.
            return
        # runs.state keeps the latest writer (diagnostics / pre-0.3.1 resume);
        # the authoritative, parallel-safe resume state is kept per subtask so
        # two subtasks never replay each other's conversation.
        self._inner.set_state(state)
        self.exec_state.update(state)
        self._store.save_exec_state(self._sid, self.exec_state)

    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision:
        with self._ctx.prompt_lock:  # one user prompt at a time per run
            self._ctx.waiting += 1
            self.heartbeat("waiting")
            try:
                decision = self._inner.request_approval(run_id, tool_name, args, risk_reason)
            finally:
                self._ctx.waiting -= 1
                self.heartbeat("running")
        self._on_intervention(self._subtask, {
            "type": "tool_approval", "tool": tool_name, "risk": risk_reason,
            "approved": bool(getattr(decision, "approved", False)),
        })
        return decision

    def _request_input(self, run_id: str, question: str) -> str | None:
        with self._ctx.prompt_lock:
            self._ctx.waiting += 1
            self.heartbeat("waiting")
            try:
                answer = self._inner.request_input(run_id, question)
            finally:
                self._ctx.waiting -= 1
                self.heartbeat("running")
        self._on_intervention(self._subtask, {
            "type": "ask_user", "question": question[:500], "answered": bool(answer),
        })
        return answer


def _tokens_from(stats: dict[str, Any]) -> dict[str, Any]:
    real_p = int(stats.get("prompt_tokens") or 0)
    real_c = int(stats.get("completion_tokens") or 0)
    est_p = int(stats.get("prompt_tokens_estimated") or 0)
    est_c = int(stats.get("completion_tokens_estimated") or 0)
    return {
        "prompt": real_p + est_p,
        "completion": real_c + est_c,
        "total": real_p + real_c + est_p + est_c,
        "estimated": bool(est_p or est_c),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    run_id: str
    goal: str
    status: str
    max_attempts: int
    attempts: int
    plan: dict[str, Any]
    route: str
    current_id: str | None
    failure_branch: str
    launch: list[str]
    summary: str
    error: str | None
    iterations: int
    tool_calls: int
    tool_successes: int
    tool_failures: int
    phase_used: dict[str, int]
    tool_counts: dict[str, int]
    stop_reason: str
    checkpoint: dict[str, Any]
    workspace: str | None
    llm: Any
    recorder: Any
    store: Any
    ctx: RunContext
    resume: ResumeState | None
    last_outcome: Any


def _legacy(state: WorkflowState) -> bool:
    return P.is_legacy(state["plan"])


def _step(
    state: WorkflowState, kind: str, output: str, *, status: str = "succeeded",
    subtask_id: str | None = None, error: str | None = None,
    args: dict[str, Any] | None = None,
) -> None:
    recorder = state.get("recorder")
    if recorder is None:
        return
    sub = P.find_subtask(state.get("plan") or {}, subtask_id) if subtask_id else None
    recorder.save_step({
        "step_id": uuid4().hex,
        "run_id": state.get("run_id", ""),
        "ts": utc_now(),
        "kind": kind,
        "tool_name": None,
        "args": args or {},
        "status": status,
        "output": output[:4000],
        "error": error,
        "subtask_id": subtask_id,
        "attempt_number": sub.get("attempt_number") if sub else None,
    })


def _save(state: WorkflowState) -> None:
    plan = state["plan"]
    store = state["store"]
    ctx = state.get("ctx")
    # A cancel requested through the API while we were running must survive
    # our in-memory copy being written back.
    fresh = store.load()
    if isinstance(fresh, dict) and fresh.get("cancel_requested"):
        plan["cancel_requested"] = True
    plan["updated_at"] = utc_now()
    if ctx is not None:
        with ctx.lock:  # workers append interventions under the same lock
            store.save(plan)
    else:
        store.save(plan)


def _record_version(
    state: WorkflowState, parent: dict[str, Any] | None, reason: str,
    failure_class: str | None = None,
) -> None:
    """Append an immutable (version, revision) snapshot to the history."""
    entry = P.version_entry(state["plan"], parent, reason=reason, failure_class=failure_class)
    state["store"].append_version(entry)


def _intervention(ctx: RunContext):
    def record(subtask: dict[str, Any], item: dict[str, Any]) -> None:
        with ctx.lock:
            subtask.setdefault("interventions", []).append({**item, "at": utc_now()})
    return record


def _set_run_status(state: WorkflowState, status: str, summary: str, error: str | None = None) -> None:
    recorder = state.get("recorder")
    if recorder is not None:
        recorder.set_status(status, summary=summary, error=error)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _generate_specs(
    state: WorkflowState, feedback: list[str], clauses: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    goal = state["goal"]
    ids = [c["id"] for c in clauses]
    if not config.PLANNER_ENABLED:
        return P.single_subtask_spec(goal, ids), "single", []
    use_p2 = bool(getattr(config, "PLANNER_STRICT", False))
    explore_text = None
    registry: list[dict[str, Any]] | None = None
    if use_p2 and getattr(config, "PLANNER_EXPLORE", False):
        try:
            from app.planner.explore import explore_repo
            from app.planner.quality import max_subtasks_for
            # Seed+index explore only here (no tool-calling LLM): avoids breaking
            # fake LLMs in tests and keeps decompose fast. try_plan.py does the
            # full agentic explore loop.
            er = explore_repo(
                goal, state.get("workspace") or config.WORKSPACE_PATH,
                llm=None,
                budget=int(getattr(config, "PLANNER_EXPLORE_BUDGET", 6) or 6),
                clauses=clauses,
            )
            explore_text = er.text
            registry = er.registry or None
            state["explore_meta"] = {
                "budget_used": er.budget_used, "done_reason": er.done_reason,
                "key_files": er.key_files, "key_symbols": er.key_symbols,
            }
        except Exception as exc:  # noqa: BLE001
            state["explore_meta"] = {"error": str(exc)[:200]}
    # Do not apply task-type step caps here — try_plan.py does that. Live
    # workflow keeps MAX_SUBTASKS so callers/tests that emit larger graphs still
    # validate.
    messages, _pack = P.build_grounded_planner_messages(
        goal, state.get("workspace"), config.MAX_SUBTASKS, clauses,
        phase2=use_p2, explore_text=explore_text)
    if registry and getattr(config, "PLANNER_WORKFLOW_STRICT", True):
        from app.planner.evidence import EVIDENCE_BINDING_HINT
        messages[-1]["content"] += "\n" + EVIDENCE_BINDING_HINT
    if feedback:
        messages[-1]["content"] += (
            "\n\n用户拒绝了上一版计划，反馈如下，请据此重新拆解：\n"
            + "\n".join(f"- {f}" for f in feedback[-5:])
        )
    # Keep soft request_subtasks in the live workflow so existing runs/tests
    # stay compatible; try_plan.py uses request_subtasks_phase2 + critic.
    if use_p2 and getattr(config, "PLANNER_WORKFLOW_STRICT", True):
        pmeta: dict[str, Any] = {}
        try:
            specs, errors = P.request_subtasks_strict_first(
                state["llm"], messages, config.MAX_SUBTASKS, goal=goal,
                workspace=state.get("workspace") or config.WORKSPACE_PATH,
                clause_ids=ids, registry=registry, retries=1, meta_out=pmeta)
        except Exception as exc:  # noqa: BLE001  hard failure → old lenient path
            pmeta = {"mode": "lenient-fallback", "strict_errors": [f"strict planner crashed: {exc}"[:300]]}
            specs, errors = P.request_subtasks(state["llm"], messages, config.MAX_SUBTASKS,
                                               clause_ids=ids, retries=1)
        state["planner_meta"] = pmeta
    else:
        specs, errors = P.request_subtasks(state["llm"], messages, config.MAX_SUBTASKS,
                                           clause_ids=ids, retries=1)
    if specs is None:
        return P.single_subtask_spec(goal, ids), "fallback", errors
    state["grounding_warnings"] = P.check_plan_grounding(specs, state.get("workspace"))
    return specs, "llm", errors


def _attach_grounding_warnings(state: Any, plan: dict[str, Any], source: str) -> None:
    pmeta = state.pop("planner_meta", None) if hasattr(state, "pop") else None
    if source == "llm" and pmeta:
        plan["planner_meta"] = {k: pmeta.get(k) for k in
                                ("mode", "strict_errors", "critic", "coverage", "strict_retry")
                                if pmeta.get(k) is not None}
        if pmeta.get("handoff_packets"):
            plan["handoff_packets"] = pmeta["handoff_packets"]
        if pmeta.get("mode") == "lenient-fallback":
            plan.setdefault("notes", []).append(
                "严格规划未通过，已回退到宽松计划：" + "；".join(pmeta.get("strict_errors") or [])[:300])
    warnings = state.pop("grounding_warnings", None) if hasattr(state, "pop") else None
    if source != "llm" or not warnings:
        return
    plan["grounding_warnings"] = list(warnings)[:20]
    plan.setdefault("notes", []).append(
        f"规划落地检查发现 {len(warnings)} 条警告（路径可能不存在）：" + "；".join(warnings[:5]))


def decompose_node(state: WorkflowState) -> dict[str, Any]:
    store = state["store"]
    plan = store.load()
    resume = state.get("resume")
    if plan and plan.get("approval_status") == P.PLAN_REJECTED:
        # Rejected with feedback → decompose again (bounded). The goal text and
        # its clauses are frozen: the new version keeps exactly the same ones.
        if int(plan.get("plan_rejections") or 0) > config.MAX_PLAN_REJECTIONS:
            state["plan"] = plan
            return _attention(state, f"计划已被拒绝 {plan['plan_rejections']} 次，超过上限 "
                              f"{config.MAX_PLAN_REJECTIONS}，需要人工处理。")
        goal = plan.get("goal") if plan.get("goal") is not None else state["goal"]
        clauses = plan.get("goal_clauses") or P.split_goal_clauses(goal, llm=state.get("llm"))
        specs, source, errors = _generate_specs(state, list(plan.get("feedback") or []), clauses)
        new = P.new_plan_record(
            specs, source, requires_approval=config.PLAN_REQUIRES_APPROVAL,
            notes=[f"规划输出无效，已回退为单子任务：{'; '.join(errors)}"] if source == "fallback" else [],
            goal=goal, goal_clauses=clauses,
        )
        _attach_grounding_warnings(state, new, source)
        P.bump_version(new, plan)
        P.assign_lineage(plan.get("subtasks"), new["subtasks"], by_id=False)
        for key in ("plan_rejections", "feedback", "totals", "structural_replans", "rewrites", "final_verify_replans"):
            new[key] = plan.get(key, new[key])
        new["history"] = (list(plan.get("history") or []) + [P.snapshot_history(plan)])[-10:]
        state["plan"] = new
        _save(state)
        feedback = (plan.get("feedback") or [""])[-1]
        _record_version(state, plan, f"计划被拒绝后按反馈重新拆解：{feedback}"[:500], "rejected")
        _step(state, "plan", "根据用户反馈重新拆解。\n" + P.describe_plan(new))
    elif plan:
        state["plan"] = plan
        _step(state, "plan", "加载已保存的任务图。\n" + P.describe_plan(plan), status="loaded")
    elif resume is not None and (resume.messages or resume.pending_tools):
        # In-flight run created before the task graph existed: wrap it.
        clauses = P.split_goal_clauses(state["goal"], llm=state.get("llm"))
        plan = P.new_plan_record(P.single_subtask_spec(state["goal"], [c["id"] for c in clauses]),
                                 "legacy", requires_approval=False,
                                 goal=state["goal"], goal_clauses=clauses)
        plan["subtasks"][0]["status"] = P.SUBTASK_RUNNING
        plan["subtasks"][0]["attempt_number"] = 1
        state["plan"] = plan
        _save(state)
        _record_version(state, None, "包装升级前的进行中任务（单子任务）")
    else:
        clauses = P.split_goal_clauses(state["goal"], llm=state.get("llm"))
        specs, source, errors = _generate_specs(state, [], clauses)
        plan = P.new_plan_record(
            specs, source, requires_approval=config.PLAN_REQUIRES_APPROVAL,
            notes=[f"规划输出无效，已回退为单子任务：{'; '.join(errors)}"] if source == "fallback" else [],
            goal=state["goal"], goal_clauses=clauses,
        )
        _attach_grounding_warnings(state, plan, source)
        P.assign_lineage(None, plan["subtasks"])
        state["plan"] = plan
        _save(state)
        _record_version(state, None, "初始拆解" if source == "llm" else f"初始拆解（{source}）")
        _step(state, "plan", P.describe_plan(plan))
    return {"plan": state["plan"], "status": "running", "error": None,
            "route": _approval_route(state["plan"])}


def _approval_route(plan: dict[str, Any]) -> str:
    if plan.get("attention"):
        return "attention"
    if (plan.get("approval_status") == P.PLAN_APPROVED
            and int(plan.get("approved_version") or -1) == int(plan.get("version") or 1)):
        return "approved"
    return "await"


def await_plan_approval_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    summary = f"任务图 v{plan.get('version')} 等待批准"
    _set_run_status(state, agent_executor.STATUS_WAITING_PLAN_APPROVAL, summary)
    _step(state, "plan_approval", summary, status=agent_executor.STATUS_WAITING_PLAN_APPROVAL)
    return {"status": agent_executor.STATUS_WAITING_PLAN_APPROVAL, "summary": summary,
            "error": None, "stop_reason": "awaiting_plan_approval"}


def _breaker(plan: dict[str, Any], ctx: RunContext) -> str | None:
    steps, seconds, tokens = ctx.live_totals(plan)
    if steps >= config.MAX_RUN_STEPS:
        return f"熔断：运行步数 {steps} ≥ MAX_RUN_STEPS={config.MAX_RUN_STEPS}"
    if seconds >= config.MAX_RUN_SECONDS:
        return f"熔断：执行时长 {seconds:.0f}s ≥ MAX_RUN_SECONDS={config.MAX_RUN_SECONDS:g}"
    if tokens >= config.MAX_RUN_TOKENS:
        return f"熔断：token 用量 {tokens} ≥ MAX_RUN_TOKENS={config.MAX_RUN_TOKENS}"
    return None


def _attention(state: WorkflowState, reason: str) -> dict[str, Any]:
    plan = state["plan"]
    plan["attention"] = {"reason": reason, "at": utc_now()}
    _save(state)
    return {"route": "attention", "error": reason}


def _needs_confirmation(plan: dict[str, Any], sub: dict[str, Any]) -> bool:
    return (not P.is_legacy(plan) and sub.get("status") == P.SUBTASK_PENDING
            and P.needs_preexec_confirmation(sub) and not sub.get("confirmed"))


def pick_next_node(state: WorkflowState) -> dict[str, Any]:
    """Scheduler: decide what to launch, whether to wait, pause or exit.

    Pauses / exits are only taken when nothing is in flight (drain first)."""
    plan = state["plan"]
    ctx = state["ctx"]
    in_flight = ctx.in_flight()
    fresh = state["store"].load()
    if isinstance(fresh, dict) and fresh.get("cancel_requested"):
        plan["cancel_requested"] = True
    if plan.get("cancel_requested"):
        ctx.cancel.set("用户取消")
        return {"route": "wait", "launch": []} if in_flight else {"route": "cancel"}
    # Legacy plans: fill analyze/verify scopes; block unscoped modify when strict.
    mig_notes = P.migrate_legacy_write_scopes(plan)
    if mig_notes:
        _save(state)
        for note in mig_notes[:5]:
            _step(state, "scope_migrate", note, status="succeeded")
    blocked_scope = P.unscoped_modify_ids(plan)
    if blocked_scope and not in_flight:
        return _attention(
            state,
            "旧计划修改类子任务缺少 write_scope，无法启动："
            + ", ".join(blocked_scope)
            + "。请 PATCH /runs/{id}/subtasks/{sid} 设置 write_scope 后 resume"
            "（或设置 LEGACY_WRITE_SCOPE_STRICT=false 以自动迁移为 **）。",
        )
    if _approval_route(plan) == "await" and not in_flight:
        return {"route": "await"}
    reason = _breaker(plan, ctx)
    if reason:
        return {"route": "wait", "launch": []} if in_flight else _attention(state, reason)

    legacy = P.is_legacy(plan)
    capacity = max(1, int(config.MAX_PARALLEL)) - len(in_flight)
    remaining_steps = (config.MAX_RUN_STEPS - int((plan.get("totals") or {}).get("steps") or 0)
                       - ctx.reserved_steps())
    exclude = set(ctx.blocked) | {s["id"] for s in plan["subtasks"] if _needs_confirmation(plan, s)}
    launch: list[str] = []
    if capacity > 0 and (legacy or remaining_steps >= 1):
        launch = [s["id"] for s in P.launchable_subtasks(plan, in_flight, capacity, exclude=exclude)]
    if launch:
        return {"route": "execute", "launch": launch}
    if in_flight:
        return {"route": "wait", "launch": []}

    # Idle: nothing in flight and nothing launchable.
    pending_failures = plan.get("pending_failures") or []
    if pending_failures:
        fatal = [f for f in pending_failures if f.get("kind") == "fail"]
        if fatal:
            return {"route": "fail", "error": "；".join(f["reason"] for f in fatal)}
        first = pending_failures[0]
        branch = P.FAIL_STRUCTURAL if any(
            f.get("branch") == P.FAIL_STRUCTURAL for f in pending_failures) else P.FAIL_REWRITE
        return {"route": "replan", "current_id": first["subtask_id"], "failure_branch": branch}
    if ctx.blocked:
        return _attention(state, "；".join(ctx.blocked.values()))
    confirm = next((s for s in P.ready_subtasks(plan) if _needs_confirmation(plan, s)), None)
    if confirm is not None:
        plan["pending_confirmation"] = confirm["id"]
        _save(state)
        summary = f"高风险子任务 [{confirm['id']}] {confirm['title']} 执行前等待确认"
        _set_run_status(state, agent_executor.STATUS_WAITING_SUBTASK_APPROVAL, summary)
        _step(state, "subtask_confirmation", summary, subtask_id=confirm["id"],
              status=agent_executor.STATUS_WAITING_SUBTASK_APPROVAL)
        return {"route": "confirm", "current_id": confirm["id"],
                "status": agent_executor.STATUS_WAITING_SUBTASK_APPROVAL, "summary": summary}
    _sub, why = P.next_ready_subtask(plan)
    if why == "complete":
        return {"route": "final", "current_id": None}
    if remaining_steps < 1 and not legacy:
        return _attention(state, f"熔断：剩余步数预算不足（MAX_RUN_STEPS={config.MAX_RUN_STEPS}）")
    blocked = [s["id"] for s in plan["subtasks"] if s["status"] == P.SUBTASK_PENDING]
    return _attention(state, "剩余子任务的依赖未成功（被跳过/失败），无法继续："
                      + ", ".join(blocked))


def await_subtask_confirmation_node(state: WorkflowState) -> dict[str, Any]:
    return {"status": agent_executor.STATUS_WAITING_SUBTASK_APPROVAL,
            "stop_reason": "awaiting_subtask_confirmation"}


def _resume_from_exec(es: dict[str, Any], fallback: ResumeState | None) -> ResumeState | None:
    if not es and fallback is None:
        return None
    return ResumeState(
        messages=list(es.get("agent_messages") or []),
        pending_tools=list(es.get("pending_tools") or []),
        preapproved=es.get("approval_decision") or (fallback.preapproved if fallback else None),
        memory=es.get("memory"),
        token_stats=es.get("token_stats"),
        cache_stats=es.get("cache_stats"),
        checkpoint=es.get("checkpoint"),
        user_input=es.get("user_input") or (fallback.user_input if fallback else None),
    )


def _start_subtask(state: WorkflowState, sid: str) -> bool:
    """Lease + attempt bookkeeping on the control-loop thread, then submit the
    executor call to the pool. Returns False when the subtask cannot start."""
    plan = state["plan"]
    store = state["store"]
    ctx = state["ctx"]
    sub = P.find_subtask(plan, sid)
    assert sub is not None
    legacy = P.is_legacy(plan)
    lease = store.get_lease(sid)
    resume_obj: ResumeState | None = None
    record = _intervention(ctx)
    run_resume = state.get("resume")

    if sub["status"] == P.SUBTASK_RUNNING:
        same = P.lease_matches(lease, sid, sub.get("attempt_number") or 0)
        if same and P.lease_is_live(lease) and lease.get("holder") != RUNNER_ID:
            ctx.blocked[sid] = (f"子任务 [{sid}] 的执行租约由另一个执行器持有"
                                f"（{lease.get('holder')}），为避免重复执行已暂停。")
            return False
        continue_same = (
            same and (lease.get("state") == "waiting" or lease.get("holder") == RUNNER_ID)
        ) or (lease is None and legacy)
        composed = P.compose_subtask_goal(state["goal"], plan, sub)

        def _run_resume_is_mine() -> bool:
            # The run-level state only seeds THIS subtask if its checkpoint was
            # written by this subtask (parallel siblings share runs.state).
            ck = run_resume.checkpoint if run_resume is not None else None
            return run_resume is not None and (
                legacy or (isinstance(ck, dict) and ck.get("goal") == composed))

        if continue_same:
            # Continue the SAME attempt (paused approval / input / restart while waiting).
            if legacy:
                resume_obj = run_resume
            else:
                es = store.get_exec_state(sid)
                if es:
                    waiting = (lease or {}).get("state") == "waiting"
                    resume_obj = _resume_from_exec(es, run_resume if waiting else None)
                elif _run_resume_is_mine() or (run_resume is not None and len(ctx.launches) == 0
                                                and not any(s.get("status") == P.SUBTASK_RUNNING
                                                            and s["id"] != sid for s in plan["subtasks"])):
                    resume_obj = run_resume  # pre-0.3.1 state or the only running subtask
            if resume_obj is None and sub.get("last_checkpoint"):
                resume_obj = ResumeState(checkpoint=sub["last_checkpoint"])
        else:
            # Expired running lease = crashed attempt → idempotent recovery:
            # new attempt seeded with the last checkpoint so already-applied
            # writes are verified on disk and skipped, not redone.
            sub["attempt_number"] = int(sub.get("attempt_number") or 0) + 1
            record(sub, {"type": "crash_recovery",
                         "note": "上次执行租约已过期（进程崩溃/重启），按检查点幂等恢复"})
            ckpt = sub.get("last_checkpoint")
            if not ckpt and not legacy:
                ckpt = store.get_exec_state(sid).get("checkpoint")
            if not ckpt and _run_resume_is_mine():
                ckpt = run_resume.checkpoint  # type: ignore[union-attr]
            resume_obj = ResumeState(checkpoint=ckpt) if ckpt else None
    else:
        sub["attempt_number"] = int(sub.get("attempt_number") or 0) + 1
        if sub.get("last_checkpoint") and not legacy:
            # Transient retry: carry the checkpoint so completed writes are skipped.
            resume_obj = ResumeState(checkpoint=sub["last_checkpoint"])

    new_lease = P.new_lease(state["run_id"], sid, sub["attempt_number"], RUNNER_ID)
    expected = lease.get("lease_id") if isinstance(lease, dict) else None
    if not store.acquire(new_lease, expected):
        ctx.blocked[sid] = f"子任务 [{sid}] 获取执行租约失败（并发冲突），已暂停。"
        return False

    fresh_conversation = resume_obj is None or not resume_obj.messages
    if not legacy and fresh_conversation:
        # Fresh conversation for this subtask: never replay older executor state.
        store.save_exec_state(sid, {})
        if not ctx.launches:
            state["recorder"].set_state({"agent_messages": None, "pending_tools": None,
                                         "approval_decision": None, "user_input": None})
    sub["status"] = P.SUBTASK_RUNNING
    sub["started_at"] = sub.get("started_at") or utc_now()
    plan["current_subtask_id"] = sid
    totals = plan.setdefault("totals", {"steps": 0, "active_seconds": 0.0, "tokens": 0})
    budget_steps = min(config.SUBTASK_MAX_ITERATIONS,
                       config.MAX_RUN_STEPS - int(totals.get("steps") or 0) - ctx.reserved_steps())
    _, active_now, _ = ctx.live_totals(plan)
    budget_secs = min(config.SUBTASK_TIMEOUT_SECONDS, config.MAX_RUN_SECONDS - active_now)
    _save(state)
    _step(state, "subtask",
          f"开始子任务 [{sid}] {sub['title']}（第 {sub['attempt_number']} 次尝试；"
          f"并行 {len(ctx.launches) + 1}/{config.MAX_PARALLEL}）",
          status="running", subtask_id=sid)

    wrapped = SubtaskRecorder(state["recorder"], store, state["run_id"], sub,
                              legacy=legacy, on_intervention=record, ctx=ctx)
    # Legacy missing keys are migrated in pick_next. Strict-blocked modify
    # should never reach here; still refuse to start unscoped modify.
    if (not legacy) and sub.get("needs_write_scope"):
        return {"route": "attention", "error": (
            f"子任务 [{sid}] 缺少 write_scope，无法执行。"
        )}
    if not legacy:
        P.migrate_legacy_write_scopes(plan)
        sub = P.find_subtask(plan, sid) or sub
    scope = (None if legacy
             else WriteScope(sub.get("write_scope") or [], subtask_id=sid))
    launch = Launch(subtask_id=sid, attempt=sub["attempt_number"], lease=new_lease,
                    recorder=wrapped, scope=scope,
                    budget_steps=0 if legacy else max(1, budget_steps), order=ctx.next_order())
    kwargs: dict[str, Any] = {
        "workspace": state.get("workspace"),
        "resume": resume_obj,
        "plan": [s["title"] for s in plan["subtasks"]],
        "max_iterations": None if legacy else max(1, budget_steps),
        "task_timeout": None if legacy else max(1.0, budget_secs),
        "mutation_required": None if legacy else (sub["kind"] == "modify"),
    }
    if scope is not None:
        kwargs["write_scope"] = scope
    kwargs["cancel_token"] = ctx.cancel
    goal = P.compose_subtask_goal(state["goal"], plan, sub)
    llm = state["llm"]
    run_id = state["run_id"]

    def _work() -> tuple[Any, float]:
        started = time.monotonic()
        try:
            outcome = agent_executor.run_agent(run_id, goal, llm, tool_schemas(), wrapped, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one branch crashing must not kill the run
            outcome = AgentOutcome(
                status=agent_executor.STATUS_FAILED, summary="子任务执行异常",
                error=f"{type(exc).__name__}: {exc}", stop_reason="crashed",
                checkpoint={},
            )
        return outcome, time.monotonic() - started

    ctx.launches[sid] = launch
    ctx.mark_started()
    launch.started = time.monotonic()
    launch.future = ctx.pool().submit(_work)
    return True


def _collect(state: WorkflowState, launch: Launch, outcome: Any, elapsed: float) -> dict[str, Any]:
    """Per-attempt accounting for a finished launch (control-loop thread)."""
    plan = state["plan"]
    store = state["store"]
    ctx = state["ctx"]
    sub = P.find_subtask(plan, launch.subtask_id)
    assert sub is not None
    current_attempt = int(sub.get("attempt_number") or 0)
    if int(launch.attempt or 0) != current_attempt:
        # A late result from a *superseded* attempt (e.g. another runner took the
        # lease over and bumped the attempt while this one was still in flight)
        # must never overwrite the newer attempt's status, evidence, attempts log
        # or completion stamp. Discard the result, but still count the real work
        # it consumed against the run budget.
        totals = plan.setdefault("totals", {"steps": 0, "active_seconds": 0.0, "tokens": 0})
        totals["steps"] = int(totals.get("steps") or 0) + int(outcome.iterations or 0)
        ctx.finish(launch.subtask_id, totals)
        with ctx.lease_lock:
            # Only release OUR lease: if a newer attempt already installed its
            # own lease, that row must survive untouched.
            held = store.get_lease(launch.subtask_id)
            if (isinstance(held, dict)
                    and held.get("lease_id") == (launch.lease or {}).get("lease_id")):
                store.set_lease(dict(launch.lease, state="released", heartbeat_at=utc_now()))
        _step(state, "subtask",
              f"子任务 [{launch.subtask_id}] 第 {launch.attempt} 次尝试的迟到结果已丢弃"
              f"（当前已是第 {current_attempt} 次尝试，其状态/证据保持不变）",
              status="stale", subtask_id=launch.subtask_id,
              args={"attempt": launch.attempt, "superseded_by": current_attempt})
        # Deliberately NO full-plan save here: this process's in-memory plan may
        # predate the takeover, and a whole-plan write is last-writer-wins. The
        # takeover owner either is this process (whose live plan object is
        # already current) or another process (whose newer plan must survive).
        return {
            "route": "next",
            "launch": [],
            "attempts": int(state.get("attempts") or 0) + 1,
            "iterations": int(state.get("iterations") or 0) + outcome.iterations,
            "tool_calls": int(state.get("tool_calls") or 0) + outcome.tool_calls,
            "tool_successes": int(state.get("tool_successes") or 0) + outcome.tool_successes,
            "tool_failures": int(state.get("tool_failures") or 0) + outcome.tool_failures,
            "stop_reason": outcome.stop_reason,
        }
    wrapped = launch.recorder
    tokens = _tokens_from(wrapped.token_stats)
    log = [a for a in sub.get("attempts_log", []) if a.get("attempt") != sub["attempt_number"]]
    prev = next((a for a in sub.get("attempts_log", []) if a.get("attempt") == sub["attempt_number"]), None)
    prev_seconds = float(prev.get("seconds") or 0) if prev else 0.0
    prev_tokens = int((prev or {}).get("tokens", {}).get("total") or 0)
    log.append({"attempt": sub["attempt_number"], "status": outcome.status,
                "stop_reason": outcome.stop_reason, "seconds": round(prev_seconds + elapsed, 3),
                "iterations": outcome.iterations, "tokens": tokens})
    totals = plan.setdefault("totals", {"steps": 0, "active_seconds": 0.0, "tokens": 0})
    with ctx.lock:
        sub["attempts_log"] = log
        sub["duration_seconds"] = round(sum(float(a.get("seconds") or 0) for a in log), 3)
        sub["tokens"] = {
            "prompt": sum(int(a["tokens"].get("prompt") or 0) for a in log),
            "completion": sum(int(a["tokens"].get("completion") or 0) for a in log),
            "total": sum(int(a["tokens"].get("total") or 0) for a in log),
            "estimated": any(a["tokens"].get("estimated") for a in log),
        }
        totals["steps"] = int(totals.get("steps") or 0) + int(outcome.iterations or 0)
        totals["tokens"] = int(totals.get("tokens") or 0) + max(0, tokens["total"] - prev_tokens)
        violations = launch.scope.snapshot_violations() if launch.scope is not None else []
        sub["acceptance_evidence"] = [
            e for e in sub.get("acceptance_evidence", []) if e.get("type") == "manual"
        ] + P.extract_acceptance_evidence(outcome) + violations
        P.store_subtask_result(sub, outcome.summary or "")
        sub["error"] = outcome.error
        sub["stop_reason"] = outcome.stop_reason
        sub["last_checkpoint"] = outcome.checkpoint or sub.get("last_checkpoint")
    ctx.finish(launch.subtask_id, totals)
    with ctx.lease_lock:
        store.set_lease(dict(launch.lease, state="released", heartbeat_at=utc_now()))
    _save(state)
    return {
        "attempts": int(state.get("attempts") or 0) + 1,
        "iterations": int(state.get("iterations") or 0) + outcome.iterations,
        "tool_calls": int(state.get("tool_calls") or 0) + outcome.tool_calls,
        "tool_successes": int(state.get("tool_successes") or 0) + outcome.tool_successes,
        "tool_failures": int(state.get("tool_failures") or 0) + outcome.tool_failures,
        "phase_used": _merge_counts(state.get("phase_used"), outcome.phase_used),
        "tool_counts": _merge_counts(state.get("tool_counts"), outcome.tool_counts),
        "stop_reason": outcome.stop_reason,
        "checkpoint": outcome.checkpoint,
        "summary": outcome.summary,
        "error": outcome.error,
        "last_outcome": outcome,
        "current_id": launch.subtask_id,
        "resume": None,  # run-level resume state is consumed by the first launch
        "launch": [],
        "route": "verify",
    }


def execute_subtask_node(state: WorkflowState) -> dict[str, Any]:
    """Submit the scheduled launches, then wait for the FIRST in-flight subtask
    to finish (others keep running) and account for it."""
    ctx = state["ctx"]
    plan = state["plan"]
    fresh = state["store"].load()
    if isinstance(fresh, dict) and fresh.get("cancel_requested"):
        plan["cancel_requested"] = True
        ctx.cancel.set("用户取消")
    if not plan.get("cancel_requested"):
        for sid in state.get("launch") or []:
            _start_subtask(state, sid)
    if not ctx.launches:
        return {"route": "next", "launch": []}
    by_future = {launch.future: launch for launch in ctx.launches.values() if launch.future}
    done, _ = wait(list(by_future), return_when=FIRST_COMPLETED)
    launch = min((by_future[f] for f in done), key=lambda x: x.order)
    outcome, elapsed = launch.future.result()  # type: ignore[union-attr]
    return _collect(state, launch, outcome, elapsed)


def _merge_counts(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + int(v or 0)
    return out


def verify_subtask_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    sub = P.find_subtask(plan, state.get("current_id"))
    outcome = state["last_outcome"]
    assert sub is not None
    legacy = P.is_legacy(plan)
    if (outcome.status == agent_executor.STATUS_CANCELLED
            or outcome.stop_reason == getattr(agent_executor, "STOP_CANCELLED", "cancelled")):
        sub["status"] = P.SUBTASK_CANCELLED
        sub["verify_note"] = outcome.summary or "用户取消"
        sub["error"] = outcome.error
        sub["stop_reason"] = outcome.stop_reason
        sub["finished_at"] = utc_now()
        sub["finished_seq"] = P.next_finish_seq(plan)
        plan["current_subtask_id"] = None
        _save(state)
        _step(state, "subtask", f"子任务 [{sub['id']}] 已取消", subtask_id=sub["id"],
              status="cancelled")
        return {"route": "next"}
    if legacy:
        ok = outcome.status == agent_executor.STATUS_SUCCEEDED
        note = "单子任务模式：沿用执行器成功判定" if ok else (outcome.error or "执行失败")
    else:
        ok, note = P.accept_subtask(sub, outcome, evidence=sub["acceptance_evidence"])
        violations = [e for e in sub["acceptance_evidence"] if e.get("type") == "scope_violation"]
        if violations:
            note += f"（写入范围拦截 {len(violations)} 次越界写入，已记录为证据）"
    sub["verify_note"] = note
    sub["finished_at"] = utc_now()
    sub["finished_seq"] = P.next_finish_seq(plan)
    if ok:
        sub["status"] = P.SUBTASK_DONE
        plan["current_subtask_id"] = None
        _save(state)
        _step(state, "subtask", f"子任务 [{sub['id']}] 验收通过：{note}", subtask_id=sub["id"],
              status="done")
        return {"route": "next"}

    sub["status"] = P.SUBTASK_FAILED
    plan["current_subtask_id"] = None
    if legacy:
        _save(state)
        _step(state, "subtask", f"子任务 [{sub['id']}] 未通过：{note}", subtask_id=sub["id"],
              status="failed", error=outcome.error)
        return {"route": "fail", "error": outcome.error or note}

    branch = P.classify_failure(sub, outcome)
    _step(state, "subtask", f"子任务 [{sub['id']}] 未通过验收（{note}），失败分类：{branch}",
          subtask_id=sub["id"], status="failed", error=outcome.error,
          args={"branch": branch})
    if branch == P.FAIL_TRANSIENT:
        # Retry the SAME subtask; plan untouched; other branches keep running.
        sub["status"] = P.SUBTASK_PENDING
        _save(state)
        return {"route": "next", "failure_branch": branch}
    used = int(plan.get("rewrites") or 0) + int(plan.get("structural_replans") or 0)
    kind = "fail" if used >= config.MAX_REPLANS else "replan"
    reason = (f"子任务 [{sub['id']}] {sub['title']} 失败，且重规划次数已达上限"
              f"（MAX_REPLANS={config.MAX_REPLANS}）：{outcome.error or note}") if kind == "fail" \
        else f"子任务 [{sub['id']}] 失败（{branch}）：{outcome.error or note}"
    # Deferred: unrelated branches keep running; the replan / fail happens when
    # nothing is in flight and no independent ready subtask is left.
    plan.setdefault("pending_failures", []).append(
        {"subtask_id": sub["id"], "branch": branch, "kind": kind, "reason": reason[:1500]})
    _save(state)
    return {"route": "next", "failure_branch": branch}


def replan_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    fv_pending = plan.pop("pending_final_verify_replan", None)
    from_final = isinstance(fv_pending, dict)
    sub = P.find_subtask(plan, state.get("current_id"))
    if sub is None and plan.get("subtasks"):
        sub = plan["subtasks"][0]
    assert sub is not None
    branch = (
        (fv_pending or {}).get("branch")
        or state.get("failure_branch")
        or P.FAIL_REWRITE
    )
    done_nodes = [s for s in plan["subtasks"] if s.get("status") == P.SUBTASK_DONE]
    if from_final:
        # Count the attempt immediately so a broken planner cannot loop forever.
        plan["final_verify_replans"] = int(plan.get("final_verify_replans") or 0) + 1
        _save(state)
        messages = P.build_final_verify_replan_messages(
            state["goal"], plan,
            missing=list(fv_pending.get("missing") or []),
            inconsistent=list(fv_pending.get("inconsistent") or []),
            max_subtasks=config.MAX_SUBTASKS, mode=branch,
        )
    else:
        messages = P.build_replan_messages(
            state["goal"], plan, sub, sub.get("error") or sub.get("verify_note") or "",
            config.MAX_SUBTASKS, mode=branch,
        )
    specs, errors = P.request_subtasks(state["llm"], messages, config.MAX_SUBTASKS,
                                       known_nodes=done_nodes,
                                       clause_ids=P.clause_ids_of(plan), retries=1)
    if specs is None:
        reason = f"重规划失败（模型输出无效）：{'; '.join(errors)[:500]}"
        _step(state, "replan", reason, status="failed", subtask_id=sub["id"])
        return {"route": "fail", "error": reason}
    old_remaining = [s for s in plan["subtasks"] if s["status"] != P.SUBTASK_DONE]
    diff = P.graph_diff(old_remaining, specs)
    structural = branch == P.FAIL_STRUCTURAL or P.is_structural_graph_diff(diff)
    parent = plan
    if structural:
        new = P.apply_structural_replan(plan, specs)
        reason = (
            (fv_pending or {}).get("reason")
            or f"结构性重规划（子任务 {sub['id']} 失败）"
        )
        if from_final and not str(reason).startswith("最终验收"):
            reason = f"最终验收未通过后的结构性重规划：{reason}"
        new["history"][-1]["diff"] = diff
        new["history"][-1]["reason"] = reason
        new["pending_failures"] = []
        if from_final:
            new["final_verify_replans"] = plan["final_verify_replans"]
            fv = dict(new.get("final_verify") or {})
            fv["replan_triggered"] = True
            fv["replan_reason"] = reason
            new["final_verify"] = fv
        state["plan"] = new
        _save(state)
        _record_version(
            state, parent, reason,
            P.FAIL_FINAL_VERIFY if from_final else (
                P.FAIL_STRUCTURAL if branch == P.FAIL_STRUCTURAL else f"{branch}→structural"),
        )
        _step(state, "replan",
              f"结构性重规划 → 任务图 v{new['version']}，需重新批准。\n" + P.describe_plan(new),
              args={"diff": diff, "branch": "structural",
                    "from_final_verify": from_final})
        return {"plan": new, "route": _approval_route(new)}
    new = P.apply_rewrite_remaining(plan, specs, failed_id=sub["id"])
    reason = (
        (fv_pending or {}).get("reason")
        or f"重写剩余子任务（子任务 {sub['id']} 做法不对，范围不变）"
    )
    if from_final and not str(reason).startswith("最终验收"):
        reason = f"最终验收未通过后的重写剩余：{reason}"
    new["history"][-1]["diff"] = diff
    new["history"][-1]["reason"] = reason
    new["pending_failures"] = []
    if from_final:
        new["final_verify_replans"] = plan["final_verify_replans"]
        fv = dict(new.get("final_verify") or {})
        fv["replan_triggered"] = True
        fv["replan_reason"] = reason
        new["final_verify"] = fv
    state["plan"] = new
    _save(state)
    _record_version(
        state, parent, reason,
        P.FAIL_FINAL_VERIFY if from_final else P.FAIL_REWRITE,
    )
    _step(state, "replan",
          f"重写剩余子任务（任务图版本保持 v{new['version']}，修订 r{new['revision']}，"
          "无需重新批准）。\n" + P.describe_plan(new),
          args={"diff": diff, "branch": "rewrite_remaining",
                "from_final_verify": from_final})
    return {"plan": new, "route": "next"}



# ---------------------------------------------------------------------------
# Final verify: a separate exit of the control loop
# ---------------------------------------------------------------------------

_FINAL_PROMPT = """你是最终验收器。只判断“各子任务的合并结果是否达成用户的原始目标”：
- 找出原始目标中没有被任何子任务覆盖的要求（missing_clauses）；
- 找出子任务结果之间的矛盾/不一致（inconsistencies）。
不要重复检查单个子任务的验收标准（它们已经按证据逐一验收）。
只输出 JSON：{"achieved": true|false, "missing_clauses": [], "inconsistencies": []}"""


# Completion-ordering modes for the final "modification must be followed by a
# passing verification" rule. New plans always carry a monotonic
# ``finished_seq``; plans created before that field existed may only have
# wall-clock timestamps, or nothing at all.
ORDER_SEQUENCE = "finished_seq"      # authoritative: stamped at completion time
ORDER_TIMESTAMPS = "finished_at"     # degraded: legacy entries without a sequence
ORDER_PLAN = "plan_order"            # last resort: no ordering evidence at all

_ORDERING_NOTES = {
    ORDER_TIMESTAMPS: (
        "最终验收顺序：部分历史子任务缺少完成序号（finished_seq），"
        "已整体退化为按完成时间戳排序（时间戳相同的按计划顺序）。"
    ),
    ORDER_PLAN: (
        "最终验收顺序：历史子任务既没有完成序号也没有完成时间，"
        "已退化为计划顺序；因此“最后一次修改之后是否有通过的验证”只能按静态顺序判断。"
    ),
}


def _completion_order(subs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Order subtasks by real completion, plus the mode that was used.

    The mode is all-or-nothing on purpose: mixing ``finished_seq`` with legacy
    entries would sort every legacy entry as if it finished first and invert the
    real timeline. A partial timeline is no better, so the fallback chain is
    authoritative sequence → wall-clock timestamps → static plan order.
    """
    if subs and all(s.get("finished_seq") is not None for s in subs):
        return sorted(
            subs,
            key=lambda s: (int(s["finished_seq"]), s.get("finished_at") or ""),
        ), ORDER_SEQUENCE
    if subs and all(s.get("finished_at") for s in subs):
        return sorted(subs, key=lambda s: str(s["finished_at"])), ORDER_TIMESTAMPS
    return list(subs), ORDER_PLAN


def _final_rule_checks(
    state: WorkflowState, notes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    from app.write_guard import content_hash

    plan = state["plan"]
    missing: list[str] = []
    inconsistent: list[str] = []
    subs = plan["subtasks"]
    not_done = [s for s in subs if s["status"] != P.SUBTASK_DONE]
    if not_done:
        missing.append("以下子任务未成功完成：" + ", ".join(f"[{s['id']}] {s['status']}" for s in not_done))
    files = [(s, e) for s in subs for e in s.get("acceptance_evidence", []) if e.get("type") == "file"]
    # A manual takeover must NOT silently exempt the file-change requirement:
    # manual file declarations already arrive as ``file`` evidence above, so a
    # mutation goal with no file evidence is a gap regardless of ``manual``.
    if goal_requires_mutation(state["goal"]) and not files:
        missing.append("原始目标要求修改代码，但所有子任务都没有文件变更证据（人工标记本身不能豁免该项检查）。")
    # Cross-subtask consistency: the last recorded hash of each file must still
    # match the disk (a later subtask must not have silently undone it).
    root = Path(state.get("workspace") or config.WORKSPACE_PATH)
    last_hash: dict[str, tuple[str, str]] = {}
    for s, e in files:
        if e.get("path") and e.get("sha256"):
            last_hash[str(e["path"])] = (s["id"], str(e["sha256"]))
    for path, (sid, digest) in last_hash.items():
        target = root / path
        try:
            disk = content_hash(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            inconsistent.append(f"文件 {path}（子任务 {sid} 修改）在验收时无法读取。")
            continue
        if disk != digest:
            inconsistent.append(f"文件 {path} 的当前内容与子任务 {sid} 记录的证据不一致（之后被改动）。")
    # A modification must be followed by a passing verification somewhere later.
    # Order by actual completion order, NOT the static plan list order: parallel
    # subtasks finish in scheduler order, so the list order does not reflect when
    # a file was last modified relative to a verification.
    ordered_subs, ordering = _completion_order(subs)
    if notes is not None and ordering in _ORDERING_NOTES:
        notes.append(_ORDERING_NOTES[ordering])
    seen_mod = False
    verified_after = True
    for s in ordered_subs:
        ev = s.get("acceptance_evidence", [])
        if any(e.get("type") == "file" for e in ev):
            seen_mod, verified_after = True, False
        if any(e.get("type") == "command" and e.get("ok") for e in ev):
            verified_after = True
    if seen_mod and not verified_after:
        inconsistent.append("最后一次文件修改之后没有任何通过的验证命令。")
    return missing, inconsistent


def final_verify_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    if P.is_legacy(plan):
        sub = plan["subtasks"][0]
        return {"route": "finish", "status": agent_executor.STATUS_SUCCEEDED,
                "summary": state.get("summary") or sub.get("result_summary", ""), "error": None}
    notes: list[str] = []
    missing, inconsistent = _final_rule_checks(state, notes=notes)
    clauses = P.clause_report(plan)
    for c in clauses:
        if c["status"] == "missing":
            missing.append(f"目标条款 {c['id']}「{c['text'][:80]}」没有被任何子任务覆盖。")
        elif c["status"] == "planned":
            missing.append(f"目标条款 {c['id']}「{c['text'][:80]}」仅有计划覆盖，缺少验收证据。")
        elif c["status"] == "insufficient":
            missing.append(f"目标条款 {c['id']}「{c['text'][:80]}」已有子任务完成但证据不足（未验证）。")
    if config.FINAL_VERIFY_LLM and config.PLANNER_ENABLED and not missing:
        payload = []
        for s in plan["subtasks"]:
            briefs = [
                b for b in (P._evidence_brief(e) for e in s.get("acceptance_evidence", []))
                if b
            ]
            payload.append({
                "id": s["id"], "title": s["title"], "kind": s["kind"],
                "manual": bool(s.get("manual")),
                "serves_clauses": s.get("serves_clauses") or [],
                "result_summary": (s.get("result_summary") or "")[:400],
                # Evidence (command exit codes, file hashes, checks) so the
                # reviewer can weigh cross-subtask consistency, not just prose.
                "evidence": briefs[:10],
            })
        clause_text = "\n".join(f"{c['id']}: {c['text']}" for c in plan.get("goal_clauses") or [])
        try:
            import json as _json
            resp = state["llm"].chat([
                {"role": "system", "content": _FINAL_PROMPT},
                {"role": "user", "content": f"原始目标：\n{state['goal']}\n\n目标条款：\n{clause_text}\n\n子任务结果：\n"
                 + _json.dumps(payload, ensure_ascii=False, indent=1)},
            ], [])
            review = P.parse_plan_text(getattr(resp, "content", None))
            if isinstance(review, dict):
                missing += [f"（复核）{m}" for m in review.get("missing_clauses") or []][:10]
                inconsistent += [f"（复核）{m}" for m in review.get("inconsistencies") or []][:10]
                if review.get("achieved") is False and not (missing or inconsistent):
                    missing.append("（复核）模型判断整体目标未达成。")
        except Exception as exc:  # noqa: BLE001 - review is advisory
            notes.append(f"最终复核模型不可用，仅使用规则检查：{exc}")
    passed = not (missing or inconsistent)
    fv = {"missing_clauses": missing, "inconsistencies": inconsistent,
          "clauses": clauses, "notes": notes, "at": utc_now(),
          "passed": passed, "replan_triggered": False, "replan_reason": None,
          "suggested_next_steps": []}
    plan["final_verify"] = fv
    _save(state)
    if missing or inconsistent:
        reason = "最终验收未通过：" + "；".join(missing + inconsistent)
        decision = P.classify_final_verify_failure(missing, inconsistent, notes)
        used = int(plan.get("final_verify_replans") or 0)
        can_replan = (
            bool(config.FINAL_VERIFY_REPLAN)
            and decision != "fail"
            and used < int(config.FINAL_VERIFY_MAX_REPLANS)
        )
        if not can_replan:
            tips = []
            if missing:
                tips.append("补齐未覆盖的目标条款（调整 serves_clauses 或新增子任务）。")
            if inconsistent:
                tips.append("核对子任务间文件/证据一致性后再提交。")
            if used >= int(config.FINAL_VERIFY_MAX_REPLANS) and config.FINAL_VERIFY_REPLAN:
                tips.append(
                    f"最终验收重规划次数已达上限（FINAL_VERIFY_MAX_REPLANS="
                    f"{config.FINAL_VERIFY_MAX_REPLANS}）。"
                )
            fv["suggested_next_steps"] = tips
            fv["replan_triggered"] = False
            plan["final_verify"] = fv
            _save(state)
            _step(state, "final_verify", reason, status="failed")
            return {"route": "fail", "error": reason}
        # Trigger replan: prefer structural for missing clauses.
        branch = decision if decision in (P.FAIL_STRUCTURAL, P.FAIL_REWRITE) else P.FAIL_STRUCTURAL
        replan_reason = (
            f"最终验收未通过后触发重规划（第 {used + 1}/"
            f"{config.FINAL_VERIFY_MAX_REPLANS} 次，分支 {branch}）："
            + "；".join((missing + inconsistent)[:5])
        )
        fv["replan_triggered"] = True
        fv["replan_reason"] = replan_reason
        plan["final_verify"] = fv
        plan["pending_final_verify_replan"] = {
            "branch": branch, "missing": missing, "inconsistent": inconsistent,
            "reason": replan_reason,
        }
        # Anchor replan_node on a done subtask (or any) for messaging context.
        anchor = next(
            (s["id"] for s in plan["subtasks"] if s.get("status") == P.SUBTASK_DONE),
            (plan["subtasks"][0]["id"] if plan["subtasks"] else None),
        )
        _save(state)
        _step(state, "final_verify", replan_reason, status="failed",
              args={"replan": True, "branch": branch})
        return {"route": "replan", "current_id": anchor, "failure_branch": branch,
                "error": reason}
    summary = f"全部 {len(plan['subtasks'])} 个子任务通过验收，最终验收通过。\n" + "\n".join(
        f"条款 {c['id']} ← {', '.join(c['covered_by'])}" for c in clauses) + "\n" + "\n".join(
        f"[{s['id']}] {s['title']}：{P.brief_result(s)}" for s in plan["subtasks"])
    _step(state, "final_verify", summary)
    return {"route": "finish", "status": agent_executor.STATUS_SUCCEEDED, "summary": summary,
            "error": None, "stop_reason": agent_executor.STOP_COMPLETED}


def finish_node(state: WorkflowState) -> dict[str, Any]:
    return {"status": agent_executor.STATUS_SUCCEEDED}


def fail_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    error = state.get("error") or "Maximum attempts reached"
    # Persist a structured failure report so the failure is inspectable from
    # storage/API (``plan["failure_report"]``), not only from the in-memory
    # object. Never fabricate missing evidence; gaps are listed explicitly.
    plan["failure_report"] = P.build_failure_report(
        plan, state.get("run_id", ""), failure_reason=error,
    )
    _save(state)
    return {"status": agent_executor.STATUS_FAILED, "error": error}


def needs_attention_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    reason = (plan.get("attention") or {}).get("reason") or state.get("error") or "需要人工处理"
    _step(state, "attention", reason, status=agent_executor.STATUS_NEEDS_ATTENTION)
    return {"status": agent_executor.STATUS_NEEDS_ATTENTION, "summary": reason, "error": reason,
            "stop_reason": "needs_attention"}


def cancel_node(state: WorkflowState) -> dict[str, Any]:
    plan = state["plan"]
    ctx = state.get("ctx")
    if ctx is not None:
        ctx.cancel.set("用户取消")
    for s in plan["subtasks"]:
        if s["status"] in (P.SUBTASK_PENDING, P.SUBTASK_RUNNING, P.SUBTASK_FAILED):
            s["status"] = P.SUBTASK_CANCELLED
    plan["pending_failures"] = []
    plan["pending_confirmation"] = None
    _save(state)
    _step(state, "cancel", "用户取消了剩余子任务。", status=agent_executor.STATUS_CANCELLED)
    return {"status": agent_executor.STATUS_CANCELLED, "summary": "用户取消了剩余子任务",
            "error": None, "stop_reason": "cancelled"}


def _route(state: WorkflowState) -> str:
    return state.get("route") or "fail"


def build_workflow():
    g = StateGraph(WorkflowState)
    g.add_node("decompose", decompose_node)
    g.add_node("await_plan_approval", await_plan_approval_node)
    g.add_node("pick_next", pick_next_node)
    g.add_node("await_subtask_confirmation", await_subtask_confirmation_node)
    g.add_node("execute_subtask", execute_subtask_node)
    g.add_node("verify_subtask", verify_subtask_node)
    g.add_node("replan", replan_node)
    g.add_node("final_verify", final_verify_node)
    g.add_node("finish", finish_node)
    g.add_node("fail", fail_node)
    g.add_node("needs_attention", needs_attention_node)
    g.add_node("cancel", cancel_node)
    g.add_edge(START, "decompose")
    g.add_conditional_edges("decompose", _route, {
        "approved": "pick_next", "await": "await_plan_approval", "attention": "needs_attention",
    })
    g.add_edge("await_plan_approval", END)
    g.add_conditional_edges("pick_next", _route, {
        "execute": "execute_subtask", "wait": "execute_subtask",
        "final": "final_verify", "await": "await_plan_approval",
        "confirm": "await_subtask_confirmation", "attention": "needs_attention", "cancel": "cancel",
        "replan": "replan", "fail": "fail",
    })
    g.add_edge("await_subtask_confirmation", END)
    g.add_conditional_edges("execute_subtask", _route, {
        "verify": "verify_subtask", "next": "pick_next", "attention": "needs_attention",
    })
    g.add_conditional_edges("verify_subtask", _route, {
        "next": "pick_next", "replan": "replan", "fail": "fail",
    })
    g.add_conditional_edges("replan", _route, {
        "next": "pick_next", "approved": "pick_next", "await": "await_plan_approval",
        "fail": "fail", "attention": "needs_attention",
    })
    g.add_conditional_edges("final_verify", _route, {"finish": "finish", "fail": "fail", "replan": "replan"})
    g.add_edge("finish", END)
    g.add_edge("fail", END)
    g.add_edge("needs_attention", END)
    g.add_edge("cancel", END)
    return g.compile()


workflow = build_workflow()


class _DefaultRecorder:
    """Safe-default recorder for direct invocations without a real approval gate.

    It discards events (no persistence) and DENIES every approval-requiring
    operation: without a real approval channel the safe default is to refuse
    high-risk operations, never to silently auto-approve them.
    """

    def save_step(self, step: dict[str, Any]) -> None:
        return None

    def set_status(self, status: str, summary: str | None = None, error: str | None = None) -> None:
        return None

    def set_state(self, state: dict[str, Any]) -> None:
        return None

    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision:
        return ApprovalDecision(
            approval_id="",
            approved=False,
            reason="未配置审批通道：默认拒绝高风险操作。",
        )

    def request_input(self, run_id: str, question: str) -> str | None:
        # No user channel: the agent's question goes unanswered.
        return None


def run_workflow(
    goal: str,
    max_attempts: int = 2,
    *,
    run_id: str | None = None,
    llm: Any = None,
    recorder: Any = None,
    workspace: str | None = None,
    resume: ResumeState | None = None,
) -> dict[str, Any]:
    # ``max_attempts`` is kept for API compatibility only: retries are owned by
    # the control loop (transient retry / rewrite / structural replan) and the
    # inner ReAct loop, never by re-running the whole graph.
    run_id = run_id or str(uuid4())
    llm = llm or DeepSeekClient()
    recorder = recorder or _DefaultRecorder()
    store = getattr(recorder, "plan_store", None) or MemoryPlanStore()
    ctx = RunContext(run_id=run_id, store=store, max_parallel=max(1, int(config.MAX_PARALLEL)))
    initial: WorkflowState = {
        "run_id": run_id, "goal": goal, "status": "queued", "max_attempts": max_attempts,
        "attempts": 0, "summary": "", "error": None, "iterations": 0, "tool_calls": 0,
        "tool_successes": 0, "tool_failures": 0, "phase_used": {}, "tool_counts": {},
        "stop_reason": "", "checkpoint": {}, "workspace": workspace, "llm": llm,
        "recorder": recorder, "store": store, "resume": resume, "plan": {},
        "ctx": ctx, "launch": [],
    }
    try:
        final = workflow.invoke(initial, {"recursion_limit": 5000})
    finally:
        ctx.close()
    plan = final.get("plan") or {}
    return {
        "run_id": run_id,
        "goal": goal,
        "status": final.get("status", "failed"),
        "attempts": final.get("attempts", 0),
        "plan": P.plan_titles(plan),
        "plan_version": plan.get("version"),
        "plan_revision": plan.get("revision"),
        "max_concurrency": ctx.max_concurrency_seen,
        "summary": final.get("summary", ""),
        "error": final.get("error"),
        "iterations": final.get("iterations", 0),
        "tool_calls": final.get("tool_calls", 0),
        "tool_successes": final.get("tool_successes", 0),
        "tool_failures": final.get("tool_failures", 0),
        "phase_used": final.get("phase_used", {}),
        "tool_counts": final.get("tool_counts", {}),
        "stop_reason": final.get("stop_reason", ""),
        "checkpoint": final.get("checkpoint", {}),
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


# ---------------------------------------------------------------------------
# Restart recovery: truly-running vs crashed-stuck
# ---------------------------------------------------------------------------

def recover_runs_on_startup(active_run_ids: set[str] | None = None) -> dict[str, list[str]]:
    """Classify every ``running`` run by its per-subtask execution leases.

    * any live lease (fresh heartbeat) held by ANOTHER process → keep ``running``;
    * no lease / expired heartbeat / held by this process but not active here
      → crashed attempt: mark ``interrupted``; ``POST /runs/{id}/resume``
      later re-leases the subtask as a NEW attempt seeded with its checkpoint
      (already-applied writes are verified on disk and skipped).
    """
    from app import storage

    active = active_run_ids or set()
    keep: list[str] = []
    crashed: list[str] = []
    for run_id in storage.list_run_ids_with_status("running"):
        leases = storage.get_leases(run_id).values()
        foreign_live = any(P.lease_is_live(lease) and lease.get("holder") != RUNNER_ID
                           and lease.get("state") != "released" for lease in leases)
        if run_id in active or foreign_live:
            keep.append(run_id)
        else:
            crashed.append(run_id)
    storage.mark_interrupted_runs(exclude_run_ids=set(keep))
    return {"kept_running": keep, "interrupted": crashed}
