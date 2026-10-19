"""Task-graph control loop tests (fake planner LLM + fake executor).

The ReAct executor is replaced by a scripted fake so these tests exercise the
control loop only: decomposition, plan approval, DAG order, evidence-based
acceptance, the three failure branches, circuit breakers, high-risk
confirmation, cancel, manual takeover, lease-based crash recovery and
persistence across a simulated restart.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import config, storage
from app.agents import agent_executor
from app.agents.agent_executor import AgentOutcome
from app.agents.llm import LLMResponse
from app.engine import planning as P
from app.engine import workflow as W
from app.write_guard import content_hash

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def sub(sid, title=None, deps=(), kind="analyze", risk="low", reversible=True,
        criteria=None, description="", write_scope=None, serves=("C1",)):
    if write_scope is None:
        write_scope = ["**"] if kind == "modify" else []
    return {"id": sid, "title": title or f"子任务{sid}", "description": description,
            "depends_on": list(deps), "kind": kind,
            "acceptance_criteria": criteria or [f"{sid} 完成"],
            "risk": risk, "reversible": reversible,
            "write_scope": list(write_scope), "serves_clauses": list(serves)}


def plan_json(*subs) -> str:
    return json.dumps({"subtasks": list(subs)}, ensure_ascii=False)


class PlannerLLM:
    """Only answers tool-less calls (planner / replanner / final review)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools):
        assert tools == [], "executor is faked; only planner calls expected"
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("planner LLM called more often than scripted")
        return LLMResponse(content=self.responses.pop(0))


_SID = re.compile(r"当前子任务 \[([^\]]+)\]")


def ok(tool_calls=1, iterations=2, files=(), tests_ok=None, summary="完成"):
    def make(goal, kw):
        ckpt = {"goal": goal, "modified_files": list(files), "test_results": []}
        if tests_ok is not None:
            ckpt["test_results"].append({"program": "python", "args": ["-m", "pytest"],
                                         "ok": tests_ok, "summary": "exit_code=0" if tests_ok else "exit_code=1"})
        return AgentOutcome(status="succeeded", summary=summary, tool_calls=tool_calls,
                            tool_successes=tool_calls, iterations=iterations,
                            stop_reason="completed", checkpoint=ckpt)
    return make


def failed(stop_reason="no_progress", error="做法不对", iterations=2):
    def make(goal, kw):
        return AgentOutcome(status="failed", summary="失败", error=error, tool_calls=1,
                            iterations=iterations, stop_reason=stop_reason,
                            checkpoint={"goal": goal, "modified_files": [], "test_results": [],
                                        "marker": f"ckpt-{stop_reason}"})
    return make


class FakeExecutor:
    def __init__(self, script: dict[str, list], tokens_per_call: int = 0, hook=None):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[dict[str, Any]] = []
        self.tokens_per_call = tokens_per_call
        self.hook = hook

    def __call__(self, run_id, goal, llm, tools, recorder, **kw):
        m = _SID.search(goal)
        sid = m.group(1) if m else "1"
        self.calls.append({"sid": sid, "goal": goal, **kw})
        if self.hook:
            self.hook(sid, recorder)
        if self.tokens_per_call:
            recorder.set_state({"token_stats": {"prompt_tokens": self.tokens_per_call,
                                                "completion_tokens": 10}})
        recorder.save_step({"step_id": f"s-{len(self.calls)}", "run_id": run_id, "ts": "t",
                            "kind": "tool", "tool_name": "read_file", "args": {},
                            "status": "succeeded", "output": "x", "error": None})
        queue = self.script.get(sid) or [ok()]
        maker = queue.pop(0) if len(queue) > 1 else queue[0]
        return maker(goal, kw)

    def order(self):
        return [c["sid"] for c in self.calls]


class Recorder:
    def __init__(self, store=None):
        self.plan_store = store or W.MemoryPlanStore()
        self.steps: list[dict[str, Any]] = []
        self.statuses: list[str] = []
        self.state: dict[str, Any] = {}

    def save_step(self, step):
        self.steps.append(step)

    def set_status(self, status, summary=None, error=None):
        self.statuses.append(status)

    def set_state(self, state):
        self.state.update(state)

    def request_approval(self, run_id, tool_name, args, risk_reason):
        raise AssertionError("not used")


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


def setup_planner(monkeypatch, tmp_path):
    """Shared by other test modules (they cannot import the fixture itself)."""
    monkeypatch.setattr(config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", False)
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    # The v0.3.0 control-loop tests assert a strict execution order; parallel
    # scheduling is covered separately in tests/test_parallel.py.
    monkeypatch.setattr(config, "MAX_PARALLEL", 1)
    storage.init_db()
    return tmp_path


def run(goal, llm, recorder, executor, monkeypatch, workspace, run_id="run-1"):
    monkeypatch.setattr(agent_executor, "run_agent", executor)
    return W.run_workflow(goal, run_id=run_id, llm=llm, recorder=recorder,
                          workspace=str(workspace))


def make_run_row(run_id, goal="分析项目", status="running", workspace=None):
    storage.save_run({"run_id": run_id, "goal": goal, "status": status, "attempts": 0,
                      "summary": "", "error": None, "created_at": "t", "updated_at": "t",
                      "state": {}, "workspace_path": workspace})


def iso(dt):
    return dt.astimezone(UTC).isoformat()


# ---------------------------------------------------------------------------
# Planner parsing / static checks / DAG
# ---------------------------------------------------------------------------


def test_parse_plan_tolerates_fences_and_prose():
    text = "好的，计划如下：\n```json\n" + plan_json(sub("1"), sub("2", deps=["1"])) + "\n```"
    specs = P.validate_subtasks(P.parse_plan_text(text), 8)
    assert [s["id"] for s in specs] == ["1", "2"]


def test_invalid_planner_output_retries_once_then_falls_back(planner_on, monkeypatch):
    llm = PlannerLLM(["不是 JSON", "还是不是"])
    ex = FakeExecutor({})
    rec = Recorder()
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on)
    assert len(llm.calls) == 2  # one retry
    plan = rec.plan_store.load()
    assert plan["source"] == "fallback" and plan["notes"]
    assert res["status"] == "succeeded"
    assert len(res["plan"]) == 1 and ex.order() == ["1"]


@pytest.mark.parametrize("bad, msg", [
    ([sub("1", reversible=False, risk="low")], "reversible"),
    ([sub("1", title="删除旧的数据库", risk="low")], "risk"),
    ([sub("1", deps=["2"]), sub("2", deps=["1"])], "环"),
    ([sub("1", deps=["9"])], "9"),
    ([sub(str(i)) for i in range(10)], "8"),
])
def test_static_plan_validation_rejects(bad, msg):
    with pytest.raises(P.PlanError) as err:
        P.validate_subtasks(bad, 8)
    assert msg in str(err.value) or msg.lower() in str(err.value).lower()


def test_irreversible_high_risk_is_accepted_and_needs_confirmation():
    [spec] = P.validate_subtasks([sub("1", reversible=False, risk="high")], 8)
    assert P.needs_preexec_confirmation(spec)


def test_dag_executes_in_dependency_order_and_steps_carry_subtask_id(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("3", deps=["1", "2"]), sub("2", deps=["1"]), sub("1"))])
    rec = Recorder()
    ex = FakeExecutor({})
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == "succeeded"
    assert ex.order() == ["1", "2", "3"]
    tool_steps = [s for s in rec.steps if s["kind"] == "tool"]
    assert [s["subtask_id"] for s in tool_steps] == ["1", "2", "3"]
    assert all(s["attempt_number"] == 1 for s in tool_steps)
    # Subtask goals carry overall goal + completed summaries.
    assert "分析项目结构" in ex.calls[2]["goal"]


def test_skipped_dependency_does_not_count_as_done():
    plan = P.new_plan_record(P.validate_subtasks([sub("1"), sub("2", deps=["1"])], 8),
                             "llm", requires_approval=False)
    plan["subtasks"][0]["status"] = P.SUBTASK_SKIPPED
    assert P.next_ready_subtask(plan)[1] == "blocked"


# ---------------------------------------------------------------------------
# Failure classification + the three branches
# ---------------------------------------------------------------------------


def test_classify_failure_branches(monkeypatch):
    monkeypatch.setattr(config, "MAX_SUBTASK_ATTEMPTS", 3)
    s = {"attempt_number": 1}
    mk = lambda stop, err="": AgentOutcome(status="failed", summary="", error=err, stop_reason=stop)
    assert P.classify_failure(s, mk("task_timeout")) == P.FAIL_TRANSIENT
    assert P.classify_failure(s, mk("tool_failure", "database is locked")) == P.FAIL_TRANSIENT
    assert P.classify_failure(s, mk("no_progress")) == P.FAIL_REWRITE
    assert P.classify_failure(s, mk("blocked")) == P.FAIL_STRUCTURAL
    assert P.classify_failure({"attempt_number": 3}, mk("task_timeout")) == P.FAIL_STRUCTURAL


def test_transient_failure_retries_same_subtask_same_version(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    rec = Recorder()
    ex = FakeExecutor({"1": [failed("task_timeout", "超时"), ok()]})
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == "succeeded"
    assert ex.order() == ["1", "1", "2"]
    plan = rec.plan_store.load()
    s1 = P.find_subtask(plan, "1")
    assert plan["version"] == 1 and s1["attempt_number"] == 2
    assert [a["attempt"] for a in s1["attempts_log"]] == [1, 2]
    # The retry is seeded with the failed attempt's checkpoint (idempotent).
    assert ex.calls[1]["resume"].checkpoint["marker"] == "ckpt-task_timeout"
    assert len(llm.calls) == 1  # no replanning for a transient failure


def test_wrong_approach_rewrites_remaining_without_reapproval(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    first = plan_json(sub("1"), sub("2", deps=["1"]))
    rewrite = plan_json(sub("2", title="换个做法读取配置", deps=["1"]))
    llm = PlannerLLM([first, rewrite])
    store = W.StoragePlanStore("run-rw")
    make_run_row("run-rw")
    rec = Recorder(store)
    ex = FakeExecutor({"2": [failed(), ok()]})
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-rw")
    assert res["status"] == agent_executor.STATUS_WAITING_PLAN_APPROVAL
    _approve(store)
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-rw")
    assert res["status"] == "succeeded", res
    plan = store.load()
    assert plan["version"] == 1 and plan["rewrites"] == 1
    assert P.find_subtask(plan, "2")["title"] == "换个做法读取配置"
    assert plan["history"][-1]["diff"]["changed"]
    assert agent_executor.STATUS_WAITING_PLAN_APPROVAL not in rec.statuses[1:]


def _approve(store):
    plan = store.load()
    plan["approval_status"] = P.PLAN_APPROVED
    plan["approved_version"] = plan["version"]
    store.save(plan)


def test_structural_replan_bumps_version_and_requires_approval(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    first = plan_json(sub("1"), sub("2", deps=["1"]))
    structural = plan_json(sub("2", deps=["1"]), sub("2b", title="先安装依赖", deps=["1"]))
    llm = PlannerLLM([first, structural])
    make_run_row("run-st")
    store = W.StoragePlanStore("run-st")
    rec = Recorder(store)
    ex = FakeExecutor({"2": [failed()]})
    run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-st")
    _approve(store)
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-st")
    assert res["status"] == agent_executor.STATUS_WAITING_PLAN_APPROVAL
    plan = store.load()
    assert plan["version"] == 2 and plan["structural_replans"] == 1
    assert plan["approval_status"] == P.PLAN_PENDING
    assert "2b" in plan["history"][-1]["diff"]["added"]
    # Only the approved version runs: nothing from v2 executed yet.
    assert "2b" not in ex.order()


def test_max_replans_fails_with_clear_reason(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_REPLANS", 1)
    first = plan_json(sub("1"))
    rewrite = plan_json(sub("1", title="再试一次"))
    llm = PlannerLLM([first, rewrite])
    ex = FakeExecutor({"1": [failed()]})
    res = run("分析项目结构", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == "failed"
    assert "MAX_REPLANS=1" in res["error"]
    assert ex.order() == ["1", "1"]


# ---------------------------------------------------------------------------
# Plan approval pause/resume + persistence across restart + rejection
# ---------------------------------------------------------------------------


def test_plan_approval_pauses_and_resume_survives_restart(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    llm = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    make_run_row("run-ap")
    ex = FakeExecutor({})
    res = run("分析项目结构", llm, Recorder(W.StoragePlanStore("run-ap")), ex,
              monkeypatch, planner_on, run_id="run-ap")
    assert res["status"] == agent_executor.STATUS_WAITING_PLAN_APPROVAL
    assert ex.calls == []
    # "Restart": brand-new store/recorder objects read the persisted plan.
    store2 = W.StoragePlanStore("run-ap")
    assert store2.load()["approval_status"] == P.PLAN_PENDING
    _approve(store2)
    res = run("分析项目结构", PlannerLLM([]), Recorder(W.StoragePlanStore("run-ap")), ex,
              monkeypatch, planner_on, run_id="run-ap")
    assert res["status"] == "succeeded"
    assert ex.order() == ["1", "2"]
    assert all(s["status"] == "done" for s in storage.get_plan("run-ap")["subtasks"])


def test_reject_with_feedback_regenerates_then_limit_needs_attention(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    monkeypatch.setattr(config, "MAX_PLAN_REJECTIONS", 1)
    llm = PlannerLLM([plan_json(sub("1")), plan_json(sub("1"), sub("2", deps=["1"]))])
    make_run_row("run-rj")
    store = W.StoragePlanStore("run-rj")
    ex = FakeExecutor({})
    run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on, run_id="run-rj")

    def reject(text):
        plan = store.load()
        plan["approval_status"] = P.PLAN_REJECTED
        plan["plan_rejections"] = int(plan.get("plan_rejections") or 0) + 1
        plan["feedback"] = list(plan.get("feedback") or []) + [text]
        store.save(plan)

    reject("请拆得更细，先读 README")
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on, run_id="run-rj")
    assert res["status"] == agent_executor.STATUS_WAITING_PLAN_APPROVAL
    assert "请拆得更细，先读 README" in llm.calls[-1][-1]["content"]
    plan = store.load()
    assert plan["version"] == 2 and len(plan["subtasks"]) == 2
    reject("还是不行")
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on, run_id="run-rj")
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert "拒绝" in res["error"]
    assert ex.calls == []


# ---------------------------------------------------------------------------
# High-risk / irreversible pre-exec confirmation
# ---------------------------------------------------------------------------


def test_high_risk_subtask_pauses_before_start_even_when_plan_approved(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2", title="迁移数据库", deps=["1"],
                                               risk="high", reversible=False),
                                sub("3", deps=["2"]))])
    make_run_row("run-hr")
    store = W.StoragePlanStore("run-hr")
    ex = FakeExecutor({})
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on, run_id="run-hr")
    assert res["status"] == agent_executor.STATUS_WAITING_SUBTASK_APPROVAL
    assert ex.order() == ["1"]
    assert store.load()["pending_confirmation"] == "2"
    # Decline → skipped → dependent subtask blocked → needs_attention (not failed).
    plan = store.load()
    P.find_subtask(plan, "2")["status"] = P.SUBTASK_SKIPPED
    plan["pending_confirmation"] = None
    store.save(plan)
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on, run_id="run-hr")
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert "3" in res["error"]


def test_high_risk_confirmed_then_runs(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1", risk="high"))])
    store = W.MemoryPlanStore()
    ex = FakeExecutor({})
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on)
    assert res["status"] == agent_executor.STATUS_WAITING_SUBTASK_APPROVAL
    plan = store.load()
    plan["subtasks"][0]["confirmed"] = True
    store.save(plan)
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on)
    assert res["status"] == "succeeded" and ex.order() == ["1"]


# ---------------------------------------------------------------------------
# Circuit breakers → needs_attention (not a silent fail)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key, value, kwargs, needle", [
    ("MAX_RUN_STEPS", 5, {"iterations": 6}, "MAX_RUN_STEPS"),
    ("MAX_RUN_TOKENS", 100, {"tokens": 500}, "MAX_RUN_TOKENS"),
])
def test_circuit_breakers_route_to_needs_attention(planner_on, monkeypatch, key, value, kwargs, needle):
    monkeypatch.setattr(config, key, value)
    llm = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    rec = Recorder()
    ex = FakeExecutor({"1": [ok(iterations=kwargs.get("iterations", 1))]},
                      tokens_per_call=kwargs.get("tokens", 0))
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert needle in res["error"]
    assert ex.order() == ["1"]
    plan = rec.plan_store.load()
    assert plan["attention"]["reason"] == res["error"]
    if key == "MAX_RUN_TOKENS":
        assert P.find_subtask(plan, "1")["tokens"]["prompt"] == 500


def test_run_seconds_breaker(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_SECONDS", 0.05)
    llm = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    ex = FakeExecutor({}, hook=lambda sid, rec: time.sleep(0.1))
    res = run("分析项目结构", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert "MAX_RUN_SECONDS" in res["error"]


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_requested_during_run_stops_at_subtask_boundary(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    store = W.MemoryPlanStore()

    def hook(sid, recorder):
        plan = store.load()
        plan["cancel_requested"] = True
        store.save(plan)

    ex = FakeExecutor({}, hook=hook)
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on)
    assert res["status"] == agent_executor.STATUS_CANCELLED
    assert ex.order() == ["1"]
    assert P.find_subtask(store.load(), "2")["status"] == P.SUBTASK_CANCELLED


# ---------------------------------------------------------------------------
# Leases: running vs crashed, idempotent recovery, CAS
# ---------------------------------------------------------------------------


def _inflight_plan(run_id, holder, heartbeat, state="running"):
    specs = P.validate_subtasks([sub("1", kind="analyze"), sub("2", deps=["1"])], 8)
    plan = P.new_plan_record(specs, "llm", requires_approval=False)
    s1 = plan["subtasks"][0]
    s1.update(status=P.SUBTASK_RUNNING, attempt_number=1,
              last_checkpoint={"goal": "x", "completed_steps": ["写入 a.py"], "marker": "crash"})
    storage.save_plan(run_id, plan)
    lease = P.new_lease(run_id, "1", 1, holder)
    lease.update(state=state, heartbeat_at=iso(heartbeat))
    storage.set_lease(run_id, lease)


def test_expired_lease_is_crashed_and_recovers_with_new_attempt(planner_on, monkeypatch):
    make_run_row("run-cr")
    _inflight_plan("run-cr", "deadhost:1:abc",
                   datetime.now(UTC) - timedelta(seconds=config.LEASE_TTL_SECONDS * 3))
    ex = FakeExecutor({})
    res = run("分析项目结构", PlannerLLM([]), Recorder(W.StoragePlanStore("run-cr")), ex,
              monkeypatch, planner_on, run_id="run-cr")
    assert res["status"] == "succeeded"
    plan = storage.get_plan("run-cr")
    s1 = P.find_subtask(plan, "1")
    assert s1["attempt_number"] == 2
    assert any(i["type"] == "crash_recovery" for i in s1["interventions"])
    # Recovery is seeded with the crashed attempt's checkpoint, so the executor
    # audits applied writes on disk instead of redoing them.
    assert ex.calls[0]["resume"].checkpoint["marker"] == "crash"
    assert storage.get_lease("run-cr")["state"] == "released"


def test_live_foreign_lease_is_not_stolen(planner_on, monkeypatch):
    make_run_row("run-live")
    _inflight_plan("run-live", "otherhost:2:def", datetime.now(UTC))
    ex = FakeExecutor({})
    res = run("分析项目结构", PlannerLLM([]), Recorder(W.StoragePlanStore("run-live")), ex,
              monkeypatch, planner_on, run_id="run-live")
    assert res["status"] == agent_executor.STATUS_NEEDS_ATTENTION
    assert ex.calls == []
    assert storage.get_lease("run-live")["holder"] == "otherhost:2:def"


def test_startup_recovery_distinguishes_running_from_crashed(planner_on):
    make_run_row("alive")
    make_run_row("dead")
    make_run_row("nolease")
    _inflight_plan("alive", "otherhost:3:x", datetime.now(UTC))
    _inflight_plan("dead", "otherhost:4:y", datetime.now(UTC) - timedelta(hours=1))
    result = W.recover_runs_on_startup()
    assert result["kept_running"] == ["alive"]
    assert sorted(result["interrupted"]) == ["dead", "nolease"]
    assert storage.get_run("alive")["status"] == "running"
    assert storage.get_run("dead")["status"] == "interrupted"


def test_lease_acquire_is_compare_and_swap(planner_on):
    make_run_row("run-cas")
    storage.save_plan("run-cas", P.new_plan_record(P.single_subtask_spec("x"), "llm",
                                                   requires_approval=False))
    acq = storage.acquire_lease
    a = P.new_lease("run-cas", "1", 1, "h1")
    assert acq("run-cas", a, expected_lease_id=None) is True
    b = P.new_lease("run-cas", "1", 2, "h2")
    assert acq("run-cas", b, expected_lease_id=None) is False      # stale expectation
    assert acq("run-cas", b, expected_lease_id="wrong-id") is False
    assert acq("run-cas", b, expected_lease_id=a["lease_id"]) is True
    assert storage.get_lease("run-cas")["holder"] == "h2"


def test_waiting_lease_continues_same_attempt(planner_on, monkeypatch):
    """A restart while blocked on a tool approval continues the SAME attempt."""
    make_run_row("run-wait")
    _inflight_plan("run-wait", "deadhost:9:z",
                   datetime.now(UTC) - timedelta(hours=1), state="waiting")
    ex = FakeExecutor({})
    run("分析项目结构", PlannerLLM([]), Recorder(W.StoragePlanStore("run-wait")), ex,
        monkeypatch, planner_on, run_id="run-wait")
    s1 = P.find_subtask(storage.get_plan("run-wait"), "1")
    assert s1["attempt_number"] == 1
    assert not any(i["type"] == "crash_recovery" for i in s1.get("interventions", []))


# ---------------------------------------------------------------------------
# Evidence-based acceptance + final verify (separate exit)
# ---------------------------------------------------------------------------


def test_self_report_without_evidence_is_rejected(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_REPLANS", 0)
    llm = PlannerLLM([plan_json(sub("1", kind="modify"))])
    ex = FakeExecutor({"1": [ok(summary="我已经改好了")]})  # no modified_files
    res = run("修改 a.py 修复 bug", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == "failed"
    assert "MAX_REPLANS" in res["error"]


def test_final_verify_detects_cross_subtask_inconsistency(planner_on, monkeypatch):
    target = planner_on / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    digest = content_hash("x = 1\n")
    llm = PlannerLLM([plan_json(sub("1", kind="modify"), sub("2", kind="analyze", deps=["1"]))])

    def hook(sid, recorder):
        if sid == "2":  # a later subtask silently changes the file
            target.write_text("x = 2\n", encoding="utf-8")

    ex = FakeExecutor({"1": [ok(files=[{"path": "a.py", "sha256": digest}], tests_ok=True)]},
                      hook=hook)
    rec = Recorder()
    res = run("修改 a.py 修复 bug", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == "failed"
    assert "a.py" in res["error"]
    fv = rec.plan_store.load()["final_verify"]
    assert fv["passed"] is False and fv["inconsistencies"]


def test_final_llm_review_checks_original_goal_clauses(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    llm = PlannerLLM([
        plan_json(sub("1")),
        json.dumps({"achieved": False, "missing_clauses": ["没有输出部署说明"],
                    "inconsistencies": []}, ensure_ascii=False),
    ])
    res = run("分析项目结构并输出部署说明", llm, Recorder(), FakeExecutor({}), monkeypatch, planner_on)
    assert res["status"] == "failed"
    assert "没有输出部署说明" in res["error"]
    assert "原始目标" in llm.calls[-1][-1]["content"]


def test_final_verify_passes_when_evidence_consistent(planner_on, monkeypatch):
    (planner_on / "a.py").write_text("x = 1\n", encoding="utf-8")
    digest = content_hash("x = 1\n")
    llm = PlannerLLM([plan_json(sub("1", kind="modify"))])
    ex = FakeExecutor({"1": [ok(files=[{"path": "a.py", "sha256": digest}], tests_ok=True)]})
    res = run("修改 a.py 修复 bug", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == "succeeded", res


# ---------------------------------------------------------------------------
# Manual takeover helpers
# ---------------------------------------------------------------------------


def test_contract_diff_structural_vs_cosmetic():
    old = sub("1")
    assert not P.is_structural_diff(P.contract_diff(old, {**old, "title": "新标题"}))
    assert P.is_structural_diff(P.contract_diff(old, {**old, "acceptance_criteria": ["更宽松"]}))
    assert P.is_structural_diff(P.contract_diff(old, {**old, "depends_on": ["0"]}))


# ---------------------------------------------------------------------------
# API: plan endpoints, rejection, manual takeover, resume, cancel
# ---------------------------------------------------------------------------


@pytest.fixture
def api(planner_on, monkeypatch):
    from app.engine import runner as runner_mod
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", "")
    holder: dict[str, Any] = {}
    monkeypatch.setattr(runner_mod, "DeepSeekClient", lambda: holder["llm"])
    from app.main import app
    with TestClient(app) as client:
        yield client, holder


def _wait(client, run_id, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/runs/{run_id}").json()
        if r["status"] in statuses:
            return r
        time.sleep(0.05)
    raise AssertionError(f"run stuck in {r['status']}")


PAUSED = {"waiting_for_plan_approval", "waiting_for_subtask_approval", "needs_attention",
          "succeeded", "failed", "cancelled"}


def test_api_plan_approve_flow(api, monkeypatch):
    client, holder = api
    holder["llm"] = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    ex = FakeExecutor({}, tokens_per_call=1000)
    monkeypatch.setattr(agent_executor, "run_agent", ex)
    run_id = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    assert _wait(client, run_id, PAUSED)["status"] == "waiting_for_plan_approval"
    view = client.get(f"/runs/{run_id}/plan").json()
    assert view["plan_version"] == 1 and [s["id"] for s in view["plan"]["subtasks"]] == ["1", "2"]
    assert client.post(f"/runs/{run_id}/subtasks/1/confirm", json={"approved": True}).status_code == 409
    r = client.post(f"/runs/{run_id}/plan/approve", json={"decision": "approve"})
    assert r.status_code == 200
    assert _wait(client, run_id, {"succeeded", "failed", "needs_attention"})["status"] == "succeeded"
    view = client.get(f"/runs/{run_id}/plan").json()
    s1 = view["plan"]["subtasks"][0]
    assert s1["status"] == "done" and s1["tokens"]["prompt"] == 1000 and s1["cost"] > 0
    assert view["plan"]["totals"]["cost"] > 0
    steps = client.get(f"/runs/{run_id}/steps").json()
    assert {s.get("subtask_id") for s in steps if s["kind"] == "tool"} == {"1", "2"}
    assert client.post(f"/runs/{run_id}/plan/approve", json={"decision": "approve"}).status_code == 409


def test_api_reject_with_feedback_goes_back_to_decompose(api, monkeypatch):
    client, holder = api
    holder["llm"] = PlannerLLM([plan_json(sub("1")), plan_json(sub("1"), sub("2", deps=["1"]))])
    monkeypatch.setattr(agent_executor, "run_agent", FakeExecutor({}))
    run_id = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    _wait(client, run_id, PAUSED)
    r = client.post(f"/runs/{run_id}/plan/approve",
                    json={"decision": "reject", "feedback": "先读 README 再分析"})
    assert r.status_code == 200
    time.sleep(0.2)
    run = _wait(client, run_id, PAUSED)
    assert run["status"] == "waiting_for_plan_approval"
    view = client.get(f"/runs/{run_id}/plan").json()
    assert view["plan_version"] == 2 and len(view["plan"]["subtasks"]) == 2
    assert "先读 README 再分析" in holder["llm"].calls[-1][-1]["content"]


def test_api_manual_takeover_edit_and_manual_result(api, monkeypatch):
    client, holder = api
    holder["llm"] = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    ex = FakeExecutor({})
    monkeypatch.setattr(agent_executor, "run_agent", ex)
    run_id = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    _wait(client, run_id, PAUSED)
    # Cosmetic edit → same version.
    r = client.patch(f"/runs/{run_id}/subtasks/2", json={"title": "改个标题"})
    assert r.status_code == 200 and r.json()["plan_version"] == 1
    # Structural edit (criteria) → new version, diffed, back to plan approval.
    r = client.patch(f"/runs/{run_id}/subtasks/2",
                     json={"acceptance_criteria": ["输出模块清单"], "note": "收紧标准"})
    body = r.json()
    assert r.status_code == 200 and body["plan_version"] == 2
    assert body["plan"]["history"][-1]["diff"]["changed"]["2"]["acceptance_criteria"]
    assert body["run_status"] == "waiting_for_plan_approval"
    # Invalid structural edit (cycle) rejected.
    assert client.patch(f"/runs/{run_id}/subtasks/1", json={"depends_on": ["2"]}).status_code == 400
    # Manual result: flagged in evidence and graph.
    r = client.post(f"/runs/{run_id}/subtasks/1/manual-result",
                    json={"status": "done", "note": "我手动做完了", "evidence_text": "pytest 全绿"})
    s1 = r.json()["plan"]["subtasks"][0]
    assert s1["manual"] is True and s1["acceptance_evidence"][-1]["type"] == "manual"
    client.post(f"/runs/{run_id}/plan/approve", json={"decision": "approve"})
    assert _wait(client, run_id, {"succeeded", "failed", "needs_attention"})["status"] == "succeeded"
    assert ex.order() == ["2"]  # manual subtask is not re-executed


def test_api_resume_needs_attention_and_cancel(api, monkeypatch):
    client, holder = api
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    monkeypatch.setattr(config, "MAX_RUN_STEPS", 3)
    holder["llm"] = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    monkeypatch.setattr(agent_executor, "run_agent", FakeExecutor({"1": [ok(iterations=5)]}))
    run_id = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    run = _wait(client, run_id, PAUSED)
    assert run["status"] == "needs_attention" and "MAX_RUN_STEPS" in run["error"]
    monkeypatch.setattr(config, "MAX_RUN_STEPS", 100)
    assert client.post(f"/runs/{run_id}/resume").status_code == 200
    time.sleep(0.1)
    assert _wait(client, run_id, {"succeeded", "failed"})["status"] == "succeeded"
    assert client.post(f"/runs/{run_id}/resume").status_code == 409

    holder["llm"] = PlannerLLM([plan_json(sub("1"))])
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    run2 = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    _wait(client, run2, PAUSED)
    r = client.post(f"/runs/{run2}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.get(f"/runs/{run2}/plan").json()["plan"]["subtasks"][0]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_adds_columns_and_plan_table(tmp_path, monkeypatch):
    if sqlite3.sqlite_version_info < (3, 35, 0):
        pytest.skip("needs ALTER TABLE DROP COLUMN to build an old schema")
    db = tmp_path / "old.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db()
    with sqlite3.connect(db) as conn:  # rebuild the pre-task-graph schema
        conn.execute("ALTER TABLE run_steps DROP COLUMN subtask_id")
        conn.execute("ALTER TABLE run_steps DROP COLUMN attempt_number")
        conn.execute("DROP TABLE run_plans")
    storage.init_db()
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_steps)")}
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"subtask_id", "attempt_number"} <= cols and "run_plans" in tables
