"""Recovery, idempotency and state-consistency audit (target: commit ``04745ad``).

Each test states its input, trigger and assertions in its docstring. The suite
covers four areas the review asked to verify dynamically:

1. checkpoint / file-write idempotency (crash before evidence, restart before
   status update, hash match vs external modification, re-resume after success);
2. execution leases under concurrent recovery (CAS race, superseded attempt);
3. evidence/counter consistency after a resume (attempts_log, acceptance
   evidence, finished_seq/finished_at, manual file evidence);
4. final verification ordering (real completion order, legacy plans without
   ``finished_seq``, persisted order across a simulated restart).

Chinese message literals are written as ``\\u`` escapes so this file stays pure
ASCII and cannot be corrupted by an editor/tool encoding round-trip.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import config, storage
from app.agents import agent_executor
from app.agents.agent_executor import ResumeState
from app.engine import planning as P
from app.engine import runner as runner_module
from app.engine import workflow as W
from app.tools import tool_schemas
from app.write_guard import content_hash
from tests.conftest import FakeLLM, final_answer, tool_call
from tests.test_execution_engine_repair import StatefulRecorder
from tests.test_task_graph import (
    _SID,
    PAUSED,
    FakeExecutor,
    PlannerLLM,
    Recorder,
    _wait,
    iso,
    make_run_row,
    ok,
    plan_json,
    run,
    setup_planner,
    sub,
)

# "fix a.py bug" - an ASCII mutation goal (matches the engine's \bfix\b hint).
GOAL = "fix a.py bug"
MISMATCH = "\u4e0d\u4e00\u81f4"                    # bu yi zhi (inconsistent)
LAST_MODIFY = "\u6700\u540e\u4e00\u6b21\u6587\u4ef6\u4fee\u6539"  # last file change


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


@pytest.fixture
def api(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", "")
    holder: dict = {}
    monkeypatch.setattr(runner_module, "DeepSeekClient", lambda: holder["llm"])
    from app.main import app

    with TestClient(app) as client:
        yield client, holder


def _wait_paused(client, run_id, timeout=10.0):
    """Wait for the paused status AND for the runner thread to release its slot.

    The status is persisted just before the thread unwinds, and manual takeover
    (correctly) refuses to change a plan while a run thread is still active, so
    tests must wait for both to avoid a spurious 409.
    """
    from app import main as main_module

    row = _wait(client, run_id, PAUSED)
    deadline = time.monotonic() + timeout
    while main_module.runner.is_active(run_id) and time.monotonic() < deadline:
        time.sleep(0.01)
    return row


class _TokRecorder:
    """Minimal recorder stand-in for a directly-driven ``_collect`` call."""

    def __init__(self) -> None:
        self.token_stats: dict[str, Any] = {}


def _plan(*specs, goal=GOAL):
    return P.new_plan_record(
        P.validate_subtasks(list(specs), 8), "llm", requires_approval=False,
        goal=goal, goal_clauses=P.split_goal_clauses(goal),
    )


# ---------------------------------------------------------------------------
# 1. Checkpoint / file-write idempotency
# ---------------------------------------------------------------------------


def test_write_not_skipped_without_checkpoint_evidence(tmp_path):
    """Input: ``out.txt`` already holds the desired content, but the crash
    happened before any checkpoint evidence was persisted.
    Trigger: the write-idempotency gate. Assert: it does NOT claim "already
    applied", so the write still goes through the guard instead of being
    silently skipped on the strength of the file content alone."""
    (tmp_path / "out.txt").write_text("done", encoding="utf-8")
    call = {"path": "out.txt", "content": "done"}

    # (a) no checkpoint at all (process died before persisting one)
    assert agent_executor._mutation_already_applied("write_file", call, tmp_path, None) is False
    # (b) checkpoint knows the path but carries no post-write hash -> unprovable
    assert agent_executor._mutation_already_applied(
        "write_file", call, tmp_path,
        {"modified_files": [{"path": "out.txt", "sha256": ""}]},
    ) is False


def test_write_skipped_only_when_checkpoint_hash_and_disk_agree(tmp_path):
    """Input: checkpoint records the post-write hash and the disk still matches.
    Trigger: same gate. Assert: the replay is skipped (no duplicate apply) and
    the recorded hash is what the checkpoint holds."""
    (tmp_path / "out.txt").write_text("done", encoding="utf-8")
    digest = agent_executor._sha256_text("done")
    checkpoint = {"modified_files": [{"path": "out.txt", "sha256": digest}]}
    assert agent_executor._mutation_already_applied(
        "write_file", {"path": "out.txt", "content": "done"}, tmp_path, checkpoint,
    ) is True


def test_hash_mismatch_means_write_is_not_skipped_and_is_reported(tmp_path):
    """Input: checkpoint says ``out.txt`` was written as ``done``, but the file
    was modified externally afterwards.
    Trigger: resume preflight + idempotency gate. Assert: the stale checkpoint
    does not skip the necessary write, and the mismatch is surfaced so the model
    re-checks instead of trusting the checkpoint."""
    digest = agent_executor._sha256_text("done")
    checkpoint = {"modified_files": [{"path": "out.txt", "sha256": digest}]}
    (tmp_path / "out.txt").write_text("externally-changed", encoding="utf-8")

    assert agent_executor._mutation_already_applied(
        "write_file", {"path": "out.txt", "content": "done"}, tmp_path, checkpoint,
    ) is False
    notes = agent_executor._audit_checkpoint_files(checkpoint, tmp_path)
    assert any(MISMATCH in n for n in notes), notes


def test_resume_after_crash_before_status_update_keeps_evidence_and_counters(
    planner_on, monkeypatch
):
    """Input: subtask 1 wrote ``a.py`` and its checkpoint persisted, but the
    service died before the subtask status/evidence were updated (status still
    ``running`` with a dead lease).
    Trigger: one resume. Assert: exactly one ``crash_recovery``, attempt bumps
    once, the checkpoint is handed to the executor, evidence appears once (no
    duplicates), ``attempts_log`` has one entry for the new attempt, and the
    completion stamp is both sequence and timestamp."""
    run_id = "rec-status-gap"
    make_run_row(run_id, goal=GOAL)
    (planner_on / "a.py").write_text("x = 1\n", encoding="utf-8")
    digest = content_hash("x = 1\n")

    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]),
                 sub("2", kind="verify", deps=("1",)))
    s1 = plan["subtasks"][0]
    s1.update(
        status=P.SUBTASK_RUNNING, attempt_number=1,
        started_at="2026-01-01T00:00:00+00:00",
        last_checkpoint={
            "goal": "x", "marker": "crash",
            "modified_files": [{"path": "a.py", "sha256": digest}],
            "test_results": [],
        },
    )
    storage.save_plan(run_id, plan)
    dead = P.new_lease(run_id, "1", 1, "deadhost:1:dead")
    dead["heartbeat_at"] = iso(
        datetime.now(UTC) - timedelta(seconds=config.LEASE_TTL_SECONDS * 3)
    )
    storage.set_lease(run_id, dead)

    ex = FakeExecutor({
        "1": [ok(files=[{"path": "a.py", "sha256": digest}], tests_ok=True)],
        "2": [ok(tests_ok=True)],
    })
    res = run(GOAL, PlannerLLM([]), Recorder(W.StoragePlanStore(run_id)), ex,
              monkeypatch, planner_on, run_id=run_id)

    assert res["status"] == "succeeded", res
    s1 = P.find_subtask(storage.get_plan(run_id), "1")
    assert s1["attempt_number"] == 2
    assert [a["attempt"] for a in s1["attempts_log"]] == [2]  # crashed try had no result
    assert [i["type"] for i in s1["interventions"]].count("crash_recovery") == 1
    # The crashed attempt's checkpoint is reused, not dropped.
    assert ex.calls[0]["resume"].checkpoint["marker"] == "crash"
    # Evidence is recorded exactly once (the crash never produced any).
    assert len([e for e in s1["acceptance_evidence"] if e["type"] == "file"]) == 1
    assert len([e for e in s1["acceptance_evidence"] if e["type"] == "command"]) == 1
    # Completion ordering is stamped consistently.
    assert s1["finished_seq"] == 1
    assert s1["finished_at"]
    assert s1["status"] == P.SUBTASK_DONE


def test_recovered_write_is_kept_when_the_model_does_not_rewrite_it(monkeypatch, tmp_path):
    """Input: a write landed and its checkpoint was persisted; the resume starts
    from that checkpoint and the model never re-issues the write.
    Trigger: a real executor resume. Assert: no write_file call is made and the
    file change stays in the returned checkpoint (acceptance evidence is not
    lost just because the resumed model did not redo the edit)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    (tmp_path / "out.txt").write_text("done", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    digest = agent_executor._sha256_text("done")
    checkpoint = {
        "goal": GOAL, "workspace": str(tmp_path), "status": "running",
        "stop_reason": "task_timeout", "pending_steps": ["run the tests"],
        "modified_files": [{"path": "out.txt", "sha256": digest}],
        "test_results": [],
        "memory": {
            "goal": GOAL, "modified_files": ["out.txt"],
            "modified_file_states": {"out.txt": digest},
        },
    }
    llm = FakeLLM([
        tool_call("c1", "run_project_command",
                  {"program": "python", "args": ["-m", "pytest", "test_ok.py", "-q"], "cwd": "."}),
        final_answer("tests passed"),
    ])
    recorder = StatefulRecorder()
    outcome = agent_executor.run_agent(
        "rec-keep", GOAL, llm, tool_schemas(), recorder,
        resume=ResumeState(checkpoint=checkpoint, memory=checkpoint["memory"]),
        workspace=str(tmp_path),
    )

    assert outcome.status == "succeeded", outcome.error
    assert not [s for s in recorder.steps
                if s.get("kind") == "tool" and s.get("tool_name") == "write_file"]
    modified = {item["path"] for item in outcome.checkpoint["modified_files"]}
    assert "out.txt" in modified


def test_resume_after_success_is_rejected_and_does_not_restamp(api, monkeypatch, tmp_path):
    """Input: a run that already succeeded (all subtasks done + evidence).
    Trigger: ``POST /runs/{id}/resume`` again. Assert: 409, and no state is
    touched - attempt number, evidence and the completion ordinal are unchanged
    (completed work is never re-counted)."""
    client, holder = api
    holder["llm"] = PlannerLLM([plan_json(sub("1"))])
    monkeypatch.setattr(agent_executor, "run_agent", FakeExecutor({}))
    rid = client.post("/runs", json={"goal": "analyze project structure",
                                     "workspace_path": str(tmp_path)}).json()["run_id"]
    assert _wait_paused(client, rid)["status"] == "waiting_for_plan_approval"
    client.post(f"/runs/{rid}/plan/approve", json={"decision": "approve"})
    assert _wait(client, rid, {"succeeded", "failed", "needs_attention"})["status"] == "succeeded"
    before = client.get(f"/runs/{rid}/plan").json()["plan"]["subtasks"][0]

    r = client.post(f"/runs/{rid}/resume")
    assert r.status_code == 409
    after = client.get(f"/runs/{rid}/plan").json()["plan"]["subtasks"][0]
    assert after["status"] == before["status"] == P.SUBTASK_DONE
    assert after["attempt_number"] == before["attempt_number"]
    assert after["acceptance_evidence"] == before["acceptance_evidence"]
    assert after["finished_seq"] == before["finished_seq"] == 1


# ---------------------------------------------------------------------------
# 2. Execution leases and concurrent recovery
# ---------------------------------------------------------------------------


def test_concurrent_lease_acquisition_has_exactly_one_winner():
    """Input: six runners race for the same subtask lease with no prior lease.
    Trigger: parallel ``acquire_lease`` (CAS under ``BEGIN IMMEDIATE``).
    Assert: exactly one wins and the stored lease belongs to that winner."""
    storage.init_db()
    make_run_row("lease-race")
    barrier = threading.Barrier(6)
    results: list[tuple[int, bool]] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        lease = P.new_lease("lease-race", "1", 1, f"holder-{i}")
        barrier.wait(timeout=5)
        won = storage.acquire_lease("lease-race", lease, expected_lease_id=None)
        with lock:
            results.append((i, won))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [i for i, won in results if won]
    assert len(winners) == 1, results
    assert storage.get_lease("lease-race", "1")["holder"] == f"holder-{winners[0]}"


def test_late_result_from_superseded_attempt_is_discarded():
    """Input: attempt 12 finished and recorded its own evidence; a late outcome
    from attempt 11 (superseded after a takeover) is then collected.
    Trigger: ``_collect``. Assert: it is discarded - status, evidence, attempts
    log, completion stamp and the newer lease all stay untouched."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]))
    s1 = plan["subtasks"][0]
    s1.update(
        status=P.SUBTASK_DONE, attempt_number=12,
        acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": "new"}],
        attempts_log=[{"attempt": 12, "status": "succeeded", "tokens": {
            "prompt": 0, "completion": 0, "total": 0, "estimated": False}}],
        finished_seq=3, finished_at="2026-01-02T00:00:00+00:00",
        result_summary="new attempt result",
    )
    store = W.MemoryPlanStore()
    store.save(plan)
    rec = Recorder(store)
    ctx = W.RunContext(run_id="late", store=store, max_parallel=1)
    stale = W.Launch(subtask_id="1", attempt=11, lease=P.new_lease("late", "1", 11, "old"),
                     recorder=_TokRecorder(), scope=None, budget_steps=1, order=1)
    newer_lease = P.new_lease("late", "1", 12, "new")
    store.set_lease(newer_lease)
    ctx.launches["1"] = stale
    state = {"run_id": "late", "goal": GOAL, "plan": plan, "store": store,
             "recorder": rec, "ctx": ctx, "workspace": "."}

    outcome = ok(files=[{"path": "a.py", "sha256": "old"}])(GOAL, {})
    result = W._collect(state, stale, outcome, 0.1)

    assert result.get("route") == "next"
    s1 = P.find_subtask(store.load(), "1")
    assert s1["attempt_number"] == 12
    assert s1["status"] == P.SUBTASK_DONE
    assert s1["acceptance_evidence"] == [{"type": "file", "path": "a.py", "sha256": "new"}]
    assert [a["attempt"] for a in s1["attempts_log"]] == [12]
    assert s1["finished_seq"] == 3
    assert s1["result_summary"] == "new attempt result"
    # The newer attempt's lease must not be released/overwritten by the stale one.
    assert store.get_lease("1") == newer_lease
    assert ctx.in_flight() == set()


@pytest.mark.parametrize("status,stop_reason", [
    ("succeeded", "completed"),
    ("failed", "no_progress"),
    ("cancelled", "cancelled"),
])
def test_superseded_attempt_result_never_writes_any_outcome(status, stop_reason):
    """Input: attempt 4 already recorded a done result (evidence, log, stamp);
    a late result from attempt 3 arrives with one of the three terminal kinds
    (success / failure / cancellation).
    Trigger: ``_collect``. Assert: *no* outcome kind can overwrite the newer
    attempt - not its status (e.g. flipping done to failed/cancelled), not its
    error/summary, evidence, attempts log, completion stamp or checkpoint - and
    the graph continues instead of running verify on the stale outcome."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]))
    s1 = plan["subtasks"][0]
    s1.update(
        status=P.SUBTASK_DONE, attempt_number=4,
        acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": "new"}],
        attempts_log=[{"attempt": 4, "status": "succeeded", "seconds": 1.0, "tokens": {
            "prompt": 1, "completion": 1, "total": 2, "estimated": False}}],
        finished_seq=9, finished_at="2026-01-03T00:00:00+00:00",
        result_summary="newer attempt result", error=None,
        last_checkpoint={"goal": GOAL, "marker": "newer"},
    )
    store = W.MemoryPlanStore()
    store.save(plan)
    rec = Recorder(store)
    ctx = W.RunContext(run_id="stale-kind", store=store, max_parallel=1)
    stale = W.Launch(subtask_id="1", attempt=3,
                     lease=P.new_lease("stale-kind", "1", 3, "old"),
                     recorder=_TokRecorder(), scope=None, budget_steps=1, order=1)
    store.set_lease(P.new_lease("stale-kind", "1", 4, "new"))
    ctx.launches["1"] = stale
    state = {"run_id": "stale-kind", "goal": GOAL, "plan": plan, "store": store,
             "recorder": rec, "ctx": ctx, "workspace": "."}
    outcome = agent_executor.AgentOutcome(
        status=status, summary="late", error="late-error", stop_reason=stop_reason,
        tool_calls=2, tool_successes=1, tool_failures=1, iterations=5,
        checkpoint={"goal": GOAL, "marker": "stale"},
    )

    result = W._collect(state, stale, outcome, 0.5)

    # The stale outcome must not reach verify (which is what writes status).
    assert result["route"] == "next"
    assert "current_id" not in result and "last_outcome" not in result
    s1 = P.find_subtask(store.load(), "1")
    assert s1["status"] == P.SUBTASK_DONE
    assert s1["attempt_number"] == 4
    assert s1["error"] is None
    assert s1["result_summary"] == "newer attempt result"
    assert s1["finished_seq"] == 9
    assert s1["finished_at"] == "2026-01-03T00:00:00+00:00"
    assert s1["acceptance_evidence"] == [{"type": "file", "path": "a.py", "sha256": "new"}]
    assert [a["attempt"] for a in s1["attempts_log"]] == [4]
    assert s1["last_checkpoint"]["marker"] == "newer"
    # The work still consumed budget, and the stale attempt is visible in the log.
    assert plan["totals"]["steps"] == 5
    stale_steps = [s for s in rec.steps if s.get("status") == "stale"]
    assert len(stale_steps) == 1
    assert stale_steps[0]["subtask_id"] == "1"


def test_superseded_attempt_recorder_cannot_persist_state_or_status():
    """Input: a recorder bound to attempt 2 while the lease already belongs to
    attempt 3 (takeover happened while the old executor was still running).
    Trigger: the old executor publishes conversation state and a run status.
    Assert: neither the per-subtask resume state nor the run status is written;
    once the lease matches again, the same recorder writes normally."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]))
    stale_sub = plan["subtasks"][0]
    stale_sub["attempt_number"] = 2
    store = W.MemoryPlanStore()
    store.save(plan)
    store.set_lease(P.new_lease("rec-guard", "1", 3, "newer"))
    ctx = W.RunContext(run_id="rec-guard", store=store, max_parallel=1)

    class _Inner:
        def __init__(self) -> None:
            self.statuses: list[str] = []
            self.states: list[dict] = []

        def set_status(self, status, summary=None, error=None):
            self.statuses.append(status)

        def set_state(self, state):
            self.states.append(state)

        def save_step(self, step):
            pass

    inner = _Inner()
    rec = W.SubtaskRecorder(inner, store, "rec-guard", stale_sub, legacy=False,
                            on_intervention=lambda *a: None, ctx=ctx)

    rec.set_state({"agent_messages": [{"role": "user", "content": "old attempt"}]})
    rec.set_status(agent_executor.STATUS_FAILED)
    assert store.get_exec_state("1") == {}
    assert inner.states == [] and inner.statuses == []
    # Token accounting still happens for superseded work (real spend).
    rec.set_state({"token_stats": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert store.get_exec_state("1") == {}

    store.set_lease(P.new_lease("rec-guard", "1", 2, "mine"))
    rec.set_state({"agent_messages": [{"role": "user", "content": "current attempt"}]})
    rec.set_status(agent_executor.STATUS_RUNNING)
    assert store.get_exec_state("1")["agent_messages"][0]["content"] == "current attempt"
    assert inner.statuses


def test_takeover_replaces_crashed_lease_and_clears_exec_state(planner_on, monkeypatch):
    """Input: a crashed attempt left a lease + executor conversation state.
    Trigger: resume. Assert: a NEW lease id is installed for attempt 2, the old
    executor state is cleared (fresh conversation), and the lease ends released."""
    run_id = "lease-takeover"
    make_run_row(run_id, goal=GOAL)
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]))
    plan["subtasks"][0].update(status=P.SUBTASK_RUNNING, attempt_number=1,
                               last_checkpoint={"goal": "x", "marker": "crash"})
    storage.save_plan(run_id, plan)
    old = P.new_lease(run_id, "1", 1, "deadhost:7:old")
    old["heartbeat_at"] = iso(datetime.now(UTC) - timedelta(seconds=config.LEASE_TTL_SECONDS * 3))
    storage.set_lease(run_id, old)
    storage.save_exec_state(run_id, "1", {"agent_messages": [{"role": "user", "content": "old chat"}]})

    res = run(GOAL, PlannerLLM([]), Recorder(W.StoragePlanStore(run_id)),
              FakeExecutor({"1": [ok(files=[{"path": "a.py", "sha256": ""}], tests_ok=True)]}),
              monkeypatch, planner_on, run_id=run_id)

    assert res["status"] == "succeeded", res
    lease = storage.get_lease(run_id, "1")
    assert lease["attempt_number"] == 2
    assert lease["lease_id"] != old["lease_id"]
    assert lease["state"] == "released"
    assert storage.get_exec_state(run_id, "1") == {}


# ---------------------------------------------------------------------------
# 3. Evidence / counter consistency
# ---------------------------------------------------------------------------


def test_parallel_finished_seq_is_unique_and_follows_completion(planner_on, monkeypatch):
    """Input: two independent subtasks; the one listed SECOND in the plan
    finishes FIRST.
    Trigger: a parallel workflow, then a second ``run_workflow`` call (simulated
    resume of a completed run). Assert: ``finished_seq`` is unique and matches
    real completion order (not plan order), every completion also has
    ``finished_at``, and re-entering the graph re-stamps nothing."""
    monkeypatch.setattr(config, "MAX_PARALLEL", 2)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    (planner_on / "a.py").write_text("x = 1\n", encoding="utf-8")
    digest = content_hash("x = 1\n")
    order: list[str] = []

    class Staged:
        def __call__(self, run_id, goal, llm, tools, recorder, **kw):
            sid = _SID.search(goal).group(1)
            if sid == "2":  # verify is listed first -> starts first, finishes last
                time.sleep(0.2)
            order.append(sid)
            if sid == "1":
                return ok(files=[{"path": "a.py", "sha256": digest}], tests_ok=True)(goal, kw)
            return ok(tests_ok=True)(goal, kw)

    # Plan list order is [verify, modify]: a list-order implementation would
    # conclude the modification happened last.
    llm = PlannerLLM([plan_json(sub("2", kind="verify"),
                                sub("1", kind="modify", write_scope=["a.py"]))])
    store = W.StoragePlanStore("seq-par")
    make_run_row("seq-par", goal=GOAL)
    res = run(GOAL, llm, Recorder(store), Staged(), monkeypatch, planner_on, run_id="seq-par")
    assert order == ["1", "2"], order
    assert res["status"] == "succeeded", res

    plan = storage.get_plan("seq-par")
    seqs = {s["id"]: s["finished_seq"] for s in plan["subtasks"]}
    assert seqs == {"1": 1, "2": 2}, seqs
    assert len(set(seqs.values())) == len(seqs)
    assert all(s["finished_at"] for s in plan["subtasks"])

    # Re-entering the completed graph must not re-stamp completed work.
    res = run(GOAL, PlannerLLM([]), Recorder(W.StoragePlanStore("seq-par")),
              Staged(), monkeypatch, planner_on, run_id="seq-par")
    assert res["status"] == "succeeded", res
    assert {s["id"]: s["finished_seq"] for s in storage.get_plan("seq-par")["subtasks"]} == seqs


def test_next_finish_seq_continues_after_a_persisted_plan_reload(planner_on):
    """Input: a persisted plan whose max ordinal is 7 (e.g. written before a
    restart). Trigger: reload + allocate. Assert: the next ordinal is 8 - a
    restart can never reuse an earlier completion number."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    plan["subtasks"][0]["finished_seq"] = 7
    storage.save_plan("seq-reload", plan)
    reloaded = storage.get_plan("seq-reload")
    assert P.next_finish_seq(reloaded) == 8


def test_manual_result_records_server_verified_file_hash(api, monkeypatch, tmp_path):
    """Input: user declares ``files: ["a.py"]`` for a manual takeover and the
    file exists in the run workspace.
    Trigger: ``POST .../manual-result``. Assert: the stored file evidence carries
    the hash the SERVER computed from disk (not a client-supplied claim) and is
    marked as server-verified."""
    client, holder = api
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    holder["llm"] = PlannerLLM([plan_json(sub("1", kind="modify", write_scope=["a.py"]))])
    rid = client.post("/runs", json={"goal": GOAL,
                                     "workspace_path": str(tmp_path)}).json()["run_id"]
    assert _wait_paused(client, rid)["status"] == "waiting_for_plan_approval"

    r = client.post(f"/runs/{rid}/subtasks/1/manual-result",
                    json={"status": "done", "note": "done by hand", "files": ["a.py"]})
    assert r.status_code == 200, r.text
    s1 = r.json()["plan"]["subtasks"][0]
    file_ev = [e for e in s1["acceptance_evidence"] if e["type"] == "file"]
    assert len(file_ev) == 1
    assert file_ev[0]["path"] == "a.py"
    assert file_ev[0]["sha256"] == content_hash("x = 1\n")
    assert file_ev[0]["verified_by"] == "server"


@pytest.mark.parametrize("claimed", ["missing.py", "../escape.py"])
def test_manual_result_rejects_unverifiable_file_claims(api, monkeypatch, tmp_path, claimed):
    """Input: a manual takeover declares a file that does not exist (or escapes
    the workspace). Trigger: ``POST .../manual-result``. Assert: 400 and no file
    evidence is recorded - a manual claim must match real server-side state."""
    client, holder = api
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    holder["llm"] = PlannerLLM([plan_json(sub("1", kind="modify", write_scope=["a.py"]))])
    rid = client.post("/runs", json={"goal": GOAL,
                                     "workspace_path": str(tmp_path)}).json()["run_id"]
    assert _wait_paused(client, rid)["status"] == "waiting_for_plan_approval"

    r = client.post(f"/runs/{rid}/subtasks/1/manual-result",
                    json={"status": "done", "note": "done by hand", "files": [claimed]})
    assert r.status_code == 400, r.text
    view = client.get(f"/runs/{rid}/plan").json()["plan"]["subtasks"][0]
    assert [e for e in view["acceptance_evidence"] if e["type"] == "file"] == []


def test_manual_file_evidence_drift_fails_final_verify(api, monkeypatch, tmp_path):
    """Input: manual takeover declares ``a.py`` (server hash H1); the file is
    then changed on disk before final verification, and a separate verify
    subtask supplies the required passing check.
    Trigger: approve the plan and let the run reach its final exit. Assert: the
    drift is detected - the server-verified hash is actually enforced."""
    client, holder = api
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    holder["llm"] = PlannerLLM([plan_json(sub("1", kind="modify", write_scope=["a.py"]),
                                          sub("2", kind="verify", deps=["1"]))])
    monkeypatch.setattr(agent_executor, "run_agent",
                        FakeExecutor({"2": [ok(tests_ok=True)]}))
    rid = client.post("/runs", json={"goal": GOAL,
                                     "workspace_path": str(tmp_path)}).json()["run_id"]
    assert _wait_paused(client, rid)["status"] == "waiting_for_plan_approval"
    assert client.post(f"/runs/{rid}/subtasks/1/manual-result",
                       json={"status": "done", "note": "done by hand",
                             "files": ["a.py"]}).status_code == 200

    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")  # drift after the claim
    client.post(f"/runs/{rid}/plan/approve", json={"decision": "approve"})
    run_row = _wait(client, rid, {"succeeded", "failed", "needs_attention"})
    assert run_row["status"] == "failed", run_row
    assert MISMATCH in (run_row["error"] or "")


# ---------------------------------------------------------------------------
# 4. Final verification ordering
# ---------------------------------------------------------------------------


def test_final_verify_uses_timestamps_when_legacy_plan_lacks_seq(tmp_path):
    """Input: a plan started before the upgrade - the modify subtask has only
    ``finished_at`` (and it is the LATER one), while the verify subtask already
    has ``finished_seq=1`` but an EARLIER timestamp.
    Trigger: final rule checks. Assert: the mixed data is NOT sorted "legacy
    first"; the real timeline (modify last) is used, so the unverified final
    modification is reported."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    s1, s2 = plan["subtasks"]
    s1.update(status=P.SUBTASK_DONE, finished_at="2026-01-02T00:00:00+00:00",
              acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    s2.update(status=P.SUBTASK_DONE, finished_at="2026-01-01T00:00:00+00:00",
              finished_seq=1,
              acceptance_evidence=[{"type": "command", "program": "python",
                                    "args": ["-m", "pytest"], "ok": True}])
    assert s1["finished_seq"] is None  # legacy entry

    missing, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": plan, "workspace": str(tmp_path)}
    )
    assert all("a.py" not in m for m in missing)
    assert any(LAST_MODIFY in i for i in inconsistent), inconsistent


def test_final_verify_seq_beats_plan_order_after_persisted_restart(tmp_path):
    """Input: plan list order is [verify, modify] but the persisted ordinals say
    modify completed first (modify=1, verify=2); timestamps are skewed the other
    way. Trigger: rule checks on the reloaded plan. Assert: the persisted
    sequence decides and the modification is correctly seen as verified."""
    plan = _plan(sub("2", kind="verify"), sub("1", kind="modify", write_scope=["a.py"]))
    verify, modify = plan["subtasks"]
    modify.update(status=P.SUBTASK_DONE, finished_seq=1,
                  finished_at="2026-01-02T00:00:00+00:00",  # skewed: looks later
                  acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    verify.update(status=P.SUBTASK_DONE, finished_seq=2,
                  finished_at="2026-01-01T00:00:00+00:00",
                  acceptance_evidence=[{"type": "command", "program": "python",
                                        "args": ["-m", "pytest"], "ok": True}])
    storage.init_db()
    storage.save_plan("order-restart", plan)
    reloaded = storage.get_plan("order-restart")

    _, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": reloaded, "workspace": str(tmp_path)}
    )
    assert inconsistent == []


def test_final_verify_with_no_ordering_data_falls_back_to_plan_order(tmp_path):
    """Input: legacy data with neither ``finished_seq`` nor ``finished_at``.
    Trigger: rule checks. Assert: no crash, and the documented degradation is
    plan order (there is no real ordering evidence available to use)."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    s1, s2 = plan["subtasks"]
    s1.update(status=P.SUBTASK_DONE,
              acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    s2.update(status=P.SUBTASK_DONE,
              acceptance_evidence=[{"type": "command", "program": "python",
                                    "args": ["-m", "pytest"], "ok": True}])
    _, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": plan, "workspace": str(tmp_path)}
    )
    assert inconsistent == []


def test_legacy_ordering_strategy_is_reported_when_timestamps_are_used(tmp_path):
    """Input: a legacy plan (no ``finished_seq``) whose entries both carry
    ``finished_at``. Trigger: rule checks with a notes sink. Assert: the
    timestamp fallback is actually used AND the degraded strategy is stated
    explicitly, so an operator can tell which ordering rule decided the run."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    s1, s2 = plan["subtasks"]
    s1.update(status=P.SUBTASK_DONE, finished_at="2026-01-01T00:00:00+00:00",
              acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    s2.update(status=P.SUBTASK_DONE, finished_at="2026-01-02T00:00:00+00:00",
              acceptance_evidence=[{"type": "command", "program": "python",
                                    "args": ["-m", "pytest"], "ok": True}])
    notes: list[str] = []
    _, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": plan, "workspace": str(tmp_path)}, notes=notes,
    )
    assert inconsistent == []
    assert W._completion_order(plan["subtasks"])[1] == W.ORDER_TIMESTAMPS
    assert W._ORDERING_NOTES[W.ORDER_TIMESTAMPS] in notes, notes


def test_legacy_ordering_falls_back_to_plan_order_only_without_any_ordering(tmp_path):
    """Input: legacy entries where one has a timestamp and one has neither a
    sequence nor a timestamp. Trigger: rule checks. Assert: a partial timeline
    is not trusted, the static plan order is used, and the degradation to
    "static order only" is reported explicitly."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    s1, s2 = plan["subtasks"]
    s1.update(status=P.SUBTASK_DONE, finished_at="2026-01-02T00:00:00+00:00",
              acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    s2.update(status=P.SUBTASK_DONE,
              acceptance_evidence=[{"type": "command", "program": "python",
                                    "args": ["-m", "pytest"], "ok": True}])
    notes: list[str] = []
    _, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": plan, "workspace": str(tmp_path)}, notes=notes,
    )
    # Plan order is [modify, verify] -> the modification IS followed by a pass,
    # so the degraded rule still passes here; what matters is the reported mode.
    assert inconsistent == []
    assert W._completion_order(plan["subtasks"])[1] == W.ORDER_PLAN
    assert W._ORDERING_NOTES[W.ORDER_PLAN] in notes, notes


def test_modern_plan_does_not_emit_a_degraded_ordering_note(tmp_path):
    """Input: every subtask has ``finished_seq``. Trigger: rule checks with a
    notes sink. Assert: the authoritative sequence mode is used and no legacy
    degradation note is emitted (warnings stay meaningful)."""
    plan = _plan(sub("1", kind="modify", write_scope=["a.py"]), sub("2", kind="verify"))
    s1, s2 = plan["subtasks"]
    s1.update(status=P.SUBTASK_DONE, finished_seq=1,
              finished_at="2026-01-02T00:00:00+00:00",
              acceptance_evidence=[{"type": "file", "path": "a.py", "sha256": ""}])
    s2.update(status=P.SUBTASK_DONE, finished_seq=2,
              finished_at="2026-01-01T00:00:00+00:00",
              acceptance_evidence=[{"type": "command", "program": "python",
                                    "args": ["-m", "pytest"], "ok": True}])
    notes: list[str] = []
    _, inconsistent = W._final_rule_checks(
        {"goal": GOAL, "plan": plan, "workspace": str(tmp_path)}, notes=notes,
    )
    assert inconsistent == []
    assert W._completion_order(plan["subtasks"])[1] == W.ORDER_SEQUENCE
    assert notes == []
