"""Parallel subtask execution (MAX_PARALLEL): scheduling by write_scope,
per-subtask leases, crash recovery with several leases, independent branches
and budget accounting under concurrency."""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from app import config, storage
from app.agents import agent_executor
from app.engine import planning as P
from app.engine import workflow as W
from tests.test_task_graph import (
    _SID,
    FakeExecutor,
    PlannerLLM,
    Recorder,
    failed,
    make_run_row,
    ok,
    plan_json,
    setup_planner,
    sub,
)


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)

GOAL = "分析项目结构"


class Tracker:
    """Fake executor measuring real concurrency."""

    def __init__(self, delay=0.15, barrier=None, tokens=0, on_peak=None):
        self.delay = delay
        self.barrier = barrier
        self.tokens = tokens
        self.on_peak = on_peak
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []
        self.kwargs: dict[str, dict] = {}

    def __call__(self, run_id, goal, llm, tools, recorder, **kw):
        sid = _SID.search(goal).group(1)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(sid)
            self.kwargs[sid] = kw
        try:
            if self.tokens:
                recorder.set_state({"token_stats": {"prompt_tokens": self.tokens,
                                                    "completion_tokens": 10}})
            recorder.save_step({"step_id": f"{sid}-{len(self.order)}", "run_id": run_id,
                                "ts": "t", "kind": "model", "status": "succeeded",
                                "output": "x", "error": None})
            if self.barrier is not None:
                self.barrier.wait(timeout=5)  # fails unless both run at once
                if self.on_peak:
                    self.on_peak(recorder)
            time.sleep(self.delay)
        finally:
            with self.lock:
                self.active -= 1
        return ok()(goal, kw)


def _run(monkeypatch, executor, recorder, llm=None, run_id="par-1"):
    monkeypatch.setattr(agent_executor, "run_agent", executor)
    return W.run_workflow(GOAL, run_id=run_id, llm=llm or PlannerLLM([]), recorder=recorder,
                          workspace=str(config.WORKSPACE_PATH))


def _approved_plan(specs, **kw):
    specs = P.validate_subtasks(specs, 8, clause_ids=["C1"])
    plan = P.new_plan_record(specs, "llm", requires_approval=False, goal=GOAL,
                             goal_clauses=P.split_goal_clauses(GOAL), **kw)
    return plan


@pytest.fixture
def par(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_PARALLEL", 2)
    monkeypatch.setattr(config, "LEASE_HEARTBEAT_SECONDS", 0.05)
    return planner_on


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_two_non_overlapping_subtasks_run_in_parallel(par, monkeypatch):
    seen: dict[str, dict] = {}

    def on_peak(recorder):
        seen.update(recorder._store.all_leases())

    llm = PlannerLLM([plan_json(sub("1", write_scope=["app/**"]),
                                sub("2", write_scope=["tests/**"]))])
    ex = Tracker(barrier=threading.Barrier(2), on_peak=on_peak)
    rec = Recorder()
    res = _run(monkeypatch, ex, rec, llm)
    assert res["status"] == "succeeded", res
    assert ex.max_active == 2 and res["max_concurrency"] == 2
    # Each subtask holds its own lease while running.
    assert set(seen) == {"1", "2"}
    assert seen["1"]["lease_id"] != seen["2"]["lease_id"]
    assert {lease["state"] for lease in seen.values()} <= {"running"}
    # Released afterwards; tool steps are attributed per subtask.
    assert {lease["state"] for lease in rec.plan_store.all_leases().values()} == {"released"}
    assert {s.get("subtask_id") for s in rec.steps if s["kind"] == "model"} == {"1", "2"}
    # Each launch got its own write scope.
    assert ex.kwargs["1"]["write_scope"].patterns == ["app/**"]


def test_overlapping_scopes_are_serialized(par, monkeypatch):
    plan = _approved_plan([sub("1", write_scope=["app/a.py"]),
                           sub("2", write_scope=["tests/**"])])
    # Bypass the static check to model e.g. a manual edit: both ready, overlapping.
    plan["subtasks"][1]["write_scope"] = ["app/**"]
    rec = Recorder()
    rec.plan_store.save(plan)
    ex = Tracker(delay=0.1)
    res = _run(monkeypatch, ex, rec)
    assert res["status"] == "succeeded", res
    assert ex.max_active == 1 and ex.order == ["1", "2"]


def test_launchable_respects_in_flight_and_unscoped_nodes():
    plan = _approved_plan([sub("1", write_scope=["app/**"]), sub("2", write_scope=["docs"]),
                           sub("3", write_scope=["app/x.py"], deps=["1"])])
    ids = [s["id"] for s in P.launchable_subtasks(plan, set(), 5)]
    assert ids == ["1", "2"]
    plan["subtasks"][0]["status"] = P.SUBTASK_RUNNING
    assert [s["id"] for s in P.launchable_subtasks(plan, {"1"}, 5)] == ["2"]
    # A pre-0.3.1 node without a write_scope key conflicts with everything.
    del plan["subtasks"][1]["write_scope"]
    assert P.launchable_subtasks(plan, {"1"}, 5) == []
    assert P.launchable_subtasks(plan, set(), 0) == []


def test_max_parallel_one_is_strictly_serial(par, monkeypatch):
    monkeypatch.setattr(config, "MAX_PARALLEL", 1)
    llm = PlannerLLM([plan_json(sub("1"), sub("2"), sub("3"))])
    ex = Tracker(delay=0.05)
    res = _run(monkeypatch, ex, Recorder(), llm)
    assert res["status"] == "succeeded"
    assert ex.max_active == 1 and ex.order == ["1", "2", "3"]


def test_failure_in_one_branch_does_not_block_unrelated_branch(par, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2"), sub("3", deps=["2"])),
                      plan_json(sub("1", title="换个做法"))])

    def slow_two(sid, recorder):
        if sid in ("2", "3"):
            time.sleep(0.15)

    ex = FakeExecutor({"1": [failed(), ok()]}, hook=slow_two)
    rec = Recorder()
    res = _run(monkeypatch, ex, rec, llm)
    assert res["status"] == "succeeded", res
    order = ex.order()
    # 2 and 3 ran to completion before the deferred replan of 1.
    assert order.index("3") < len(order) - 1 - order[::-1].index("1")
    assert order.count("1") == 2 and len(llm.calls) == 2
    plan = rec.plan_store.load()
    assert plan["rewrites"] == 1 and plan["pending_failures"] == []
    assert all(s["status"] == "done" for s in plan["subtasks"])


# ---------------------------------------------------------------------------
# Budgets / breakers under concurrency
# ---------------------------------------------------------------------------


def test_counters_are_exact_under_concurrency(par, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2"), sub("3"), sub("4"))])
    ex = Tracker(delay=0.05, tokens=1000)
    rec = Recorder()
    res = _run(monkeypatch, ex, rec, llm)
    assert res["status"] == "succeeded"
    totals = rec.plan_store.load()["totals"]
    assert totals["tokens"] == 4 * 1010
    assert totals["steps"] == 4 * 2  # ok() reports 2 iterations each
    assert totals["active_seconds"] > 0
    assert ex.max_active == 2


def test_token_breaker_counts_live_parallel_usage(par, monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_TOKENS", 1500)
    llm = PlannerLLM([plan_json(sub("1"), sub("2"), sub("3"), sub("4"))])
    ex = Tracker(delay=0.05, tokens=1000, barrier=threading.Barrier(2))
    rec = Recorder()
    res = _run(monkeypatch, ex, rec, llm)
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert "MAX_RUN_TOKENS" in res["error"]
    assert sorted(ex.order) == ["1", "2"]  # nothing new launched after the trip
    assert rec.plan_store.load()["totals"]["tokens"] == 2 * 1010


def test_step_budget_is_reserved_across_parallel_launches(par, monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_STEPS", 30)
    monkeypatch.setattr(config, "SUBTASK_MAX_ITERATIONS", 20)
    llm = PlannerLLM([plan_json(sub("1"), sub("2"))])
    ex = Tracker(delay=0.1)
    _run(monkeypatch, ex, Recorder(), llm)
    budgets = sorted(kw["max_iterations"] for kw in ex.kwargs.values())
    assert budgets == [10, 20]  # the second launch only gets what is left


# ---------------------------------------------------------------------------
# Crash recovery with two leases
# ---------------------------------------------------------------------------


def _iso(dt):
    return dt.astimezone(UTC).isoformat()


def _crashed_plan(run_id, live: set[str] = frozenset()):
    plan = _approved_plan([sub("1", write_scope=["app/**"]), sub("2", write_scope=["docs/**"])])
    old = _iso(datetime.now(UTC) - timedelta(hours=1))
    for s in plan["subtasks"]:
        s["status"] = P.SUBTASK_RUNNING
        s["attempt_number"] = 1
        s["last_checkpoint"] = {"goal": "x", "marker": f"ckpt-{s['id']}"}
    storage.save_plan(run_id, plan)
    for sid in ("1", "2"):
        lease = P.new_lease(run_id, sid, 1, "other-host:1:dead")
        if sid not in live:
            lease["heartbeat_at"] = old
        storage.set_lease(run_id, lease)
    return plan


def test_crash_recovery_with_two_expired_leases(par, monkeypatch):
    make_run_row("par-crash")
    _crashed_plan("par-crash")
    ex = Tracker(delay=0.05)
    res = _run(monkeypatch, ex, Recorder(W.StoragePlanStore("par-crash")), run_id="par-crash")
    assert res["status"] == "succeeded", res
    assert sorted(ex.order) == ["1", "2"]
    plan = storage.get_plan("par-crash")
    for s in plan["subtasks"]:
        assert s["attempt_number"] == 2
        assert [i["type"] for i in s["interventions"]].count("crash_recovery") == 1
    # Each recovery was seeded with ITS OWN checkpoint.
    assert ex.kwargs["1"]["resume"].checkpoint["marker"] == "ckpt-1"
    assert ex.kwargs["2"]["resume"].checkpoint["marker"] == "ckpt-2"
    leases = storage.get_leases("par-crash")
    assert {(k, v["attempt_number"], v["state"]) for k, v in leases.items()} == {
        ("1", 2, "released"), ("2", 2, "released")}


def test_live_foreign_lease_blocks_only_that_subtask(par, monkeypatch):
    make_run_row("par-live")
    _crashed_plan("par-live", live={"2"})
    ex = Tracker(delay=0.05)
    res = _run(monkeypatch, ex, Recorder(W.StoragePlanStore("par-live")), run_id="par-live")
    assert ex.order == ["1"]  # never duplicates the live attempt
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert "另一个执行器" in res["error"]
    plan = storage.get_plan("par-live")
    assert P.find_subtask(plan, "1")["status"] == "done"
    assert P.find_subtask(plan, "2")["attempt_number"] == 1


def test_startup_recovery_inspects_every_subtask_lease(par):
    make_run_row("par-alive")
    _crashed_plan("par-alive", live={"2"})
    make_run_row("par-dead")
    _crashed_plan("par-dead")
    out = W.recover_runs_on_startup()
    assert "par-alive" in out["kept_running"] and "par-dead" in out["interrupted"]
    assert storage.get_run("par-alive")["status"] == "running"
    assert storage.get_run("par-dead")["status"] == "interrupted"


def test_old_single_lease_column_is_migrated(tmp_path, monkeypatch):
    import json
    import sqlite3

    db = tmp_path / "mig.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db()
    lease = P.new_lease("old-run", "3", 2, "h")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO run_plans (run_id, plan_version, plan_json, "
                     "execution_lease_json, updated_at) VALUES (?, 1, '{}', ?, 't')",
                     ("old-run", json.dumps(lease)))
    storage.init_db()
    assert storage.get_leases("old-run")["3"]["lease_id"] == lease["lease_id"]
    assert storage.get_lease("old-run")["attempt_number"] == 2
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"subtask_leases", "plan_versions"} <= tables


# ---------------------------------------------------------------------------
# Prompts and cancel while several subtasks are in flight
# ---------------------------------------------------------------------------


class ApprovingRecorder(Recorder):
    def __init__(self, store=None):
        super().__init__(store)
        self.active_prompts = 0
        self.max_prompts = 0
        self.lock = threading.Lock()

    def request_approval(self, run_id, tool_name, args, risk_reason):
        from app.engine.approvals import ApprovalDecision
        with self.lock:
            self.active_prompts += 1
            self.max_prompts = max(self.max_prompts, self.active_prompts)
        time.sleep(0.05)
        with self.lock:
            self.active_prompts -= 1
        return ApprovalDecision(approval_id="a", approved=True)


def test_parallel_prompts_are_serialized(par, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2"))])

    class Asking(Tracker):
        def __call__(self, run_id, goal, llm, tools, recorder, **kw):
            recorder.request_approval(run_id, "run_project_command", {}, "测试")
            return super().__call__(run_id, goal, llm, tools, recorder, **kw)

    rec = ApprovingRecorder()
    res = _run(monkeypatch, Asking(delay=0.05), rec, llm)
    assert res["status"] == "succeeded"
    assert rec.max_prompts == 1  # one user prompt at a time per run
    plan = rec.plan_store.load()
    assert all(any(i["type"] == "tool_approval" for i in s["interventions"])
               for s in plan["subtasks"])


def test_cancel_with_two_in_flight_drains_then_stops(par, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2"), sub("3", deps=["1", "2"]))])
    rec = Recorder()

    def cancel(recorder):
        plan = rec.plan_store.load()
        plan["cancel_requested"] = True
        rec.plan_store.save(plan)

    ex = Tracker(delay=0.05, barrier=threading.Barrier(2), on_peak=cancel)
    res = _run(monkeypatch, ex, rec, llm)
    assert res["status"] == agent_executor.STATUS_CANCELLED
    assert sorted(ex.order) == ["1", "2"]
    plan = rec.plan_store.load()
    assert P.find_subtask(plan, "3")["status"] == P.SUBTASK_CANCELLED
