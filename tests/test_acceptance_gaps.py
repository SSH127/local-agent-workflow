"""Regression tests for the acceptance-entry gaps found in review:

1. a ``verify`` subtask must not pass on tool calls alone (``read_file`` is
   preparation, not proof);
2. a manual takeover must not silently exempt the file-change requirement;
3. goal-clause coverage requires substantive evidence, not just a ``done`` flag;
4. final verification orders evidence by actual completion time, not plan order.
"""
from __future__ import annotations

import time

import pytest

from app import config
from app.agents.agent_executor import AgentOutcome
from app.engine import planning as P
from app.engine import workflow as W
from app.write_guard import content_hash
from tests.test_task_graph import (
    _SID,
    FakeExecutor,
    PlannerLLM,
    Recorder,
    ok,
    plan_json,
    run,
    setup_planner,
    sub,
)


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# 1. verify acceptance requires a passing verification command / check
# ---------------------------------------------------------------------------


def test_accept_subtask_verify_requires_passed_evidence():
    outcome = AgentOutcome(status="succeeded", summary="读过了，看起来没问题",
                           tool_calls=1, tool_successes=1)
    accepted, note = P.accept_subtask({"kind": "verify"}, outcome, evidence=[])
    assert accepted is False
    assert "证据" in note


def test_verify_subtask_with_only_reads_is_rejected(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_REPLANS", 0)
    llm = PlannerLLM([plan_json(sub("1", kind="verify"))])
    # ``ok()`` reports tool_calls>0 but no passing command → must be rejected.
    ex = FakeExecutor({"1": [ok(tool_calls=1)]})
    res = run("运行测试验证", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == "failed", res
    assert "证据" in (res["error"] or "")


def test_verify_subtask_with_passed_command_is_accepted(planner_on, monkeypatch):
    llm = PlannerLLM([plan_json(sub("1", kind="verify"))])
    ex = FakeExecutor({"1": [ok(tests_ok=True)]})
    res = run("运行测试验证", llm, Recorder(), ex, monkeypatch, planner_on)
    assert res["status"] == "succeeded", res


# ---------------------------------------------------------------------------
# 2. manual takeover does not exempt the file-change requirement
# ---------------------------------------------------------------------------


def _manual_plan(tmp_path, files=None):
    plan = P.new_plan_record(
        [sub("1", kind="modify", write_scope=["a.py"], serves=("C1",))],
        "llm", requires_approval=False,
        goal="修复 a.py 的 bug",
        goal_clauses=P.split_goal_clauses("修复 a.py 的 bug"),
    )
    s1 = plan["subtasks"][0]
    s1["status"] = P.SUBTASK_DONE
    s1["manual"] = True
    s1["finished_at"] = "2026-01-01T00:00:00+00:00"
    s1["acceptance_evidence"].extend(
        P.manual_evidence("我手动修好了", "已修复", "done", files=files))
    return plan


def test_manual_mark_without_files_still_requires_file_change(tmp_path):
    plan = _manual_plan(tmp_path)
    missing, _ = W._final_rule_checks({
        "goal": "修复 a.py 的 bug", "plan": plan, "workspace": str(tmp_path),
    })
    assert any("文件变更证据" in m for m in missing)


def test_manual_mark_with_declared_files_satisfies_file_change(tmp_path):
    plan = _manual_plan(tmp_path, files=["a.py"])
    missing, _ = W._final_rule_checks({
        "goal": "修复 a.py 的 bug", "plan": plan, "workspace": str(tmp_path),
    })
    assert not any("文件变更证据" in m for m in missing)


# ---------------------------------------------------------------------------
# 3. clause coverage: planned vs verified
# ---------------------------------------------------------------------------


def test_clause_report_distinguishes_planned_from_verified():
    plan = P.new_plan_record(
        [sub("1", kind="modify", write_scope=["a.py"], serves=("C1",)),
         sub("2", kind="verify", serves=("C2",))],
        "llm", requires_approval=False,
        goal="修复并验证",
        goal_clauses=[{"id": "C1", "text": "修复"}, {"id": "C2", "text": "验证"}],
    )
    # Subtask 1 done but NO file evidence → planned, not covered.
    plan["subtasks"][0]["status"] = P.SUBTASK_DONE
    # Subtask 2 done with a passing command → covered.
    plan["subtasks"][1]["status"] = P.SUBTASK_DONE
    plan["subtasks"][1]["acceptance_evidence"] = [{
        "type": "command", "program": "python", "args": ["-m", "pytest"],
        "ok": True, "summary": "exit_code=0",
    }]
    report = {c["id"]: c for c in P.clause_report(plan)}
    assert report["C1"]["status"] == "planned" and report["C1"]["covered_by"] == []
    assert report["C1"]["planned_by"] == ["1"]
    assert report["C2"]["status"] == "covered" and report["C2"]["covered_by"] == ["2"]


# ---------------------------------------------------------------------------
# 4. final verification orders evidence by actual completion, not plan order
# ---------------------------------------------------------------------------


def test_final_verify_orders_by_actual_completion_not_plan_order(planner_on, monkeypatch):
    monkeypatch.setattr(config, "MAX_PARALLEL", 2)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    target = planner_on / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    digest = content_hash("x = 1\n")

    class StagedExecutor:
        def __init__(self):
            self.order: list[str] = []

        def __call__(self, run_id, goal, llm, tools, recorder, **kw):
            sid = _SID.search(goal).group(1)
            recorder.save_step({"step_id": f"s-{sid}", "run_id": run_id, "ts": "t",
                                "kind": "model", "status": "succeeded", "output": "x",
                                "error": None})
            if sid == "1":  # modify finishes LAST
                time.sleep(0.15)
                self.order.append(sid)  # record COMPLETION order, not call order
                return ok(files=[{"path": "a.py", "sha256": digest}])(goal, kw)
            time.sleep(0.03)  # verify finishes FIRST
            self.order.append(sid)
            return ok(tests_ok=True)(goal, kw)

    llm = PlannerLLM([plan_json(sub("1", kind="modify", write_scope=["a.py"]),
                                sub("2", kind="verify"))])
    ex = StagedExecutor()
    rec = Recorder()
    res = run("修复 a.py 的 bug", llm, rec, ex, monkeypatch, planner_on)
    # verify (2) actually completed before modify (1).
    assert ex.order.index("2") < ex.order.index("1")
    # The last modification has no passing verification after it.
    assert res["status"] == "failed", res
    assert "最后一次文件修改" in (res["error"] or "")


def test_next_finish_seq_is_monotonic_and_survives_plan_reload():
    plan = P.new_plan_record(
        [sub("1", kind="modify", write_scope=["a.py"]),
         sub("2", kind="verify")],
        "llm", requires_approval=False, goal="g",
    )
    # Stamp completions in order: each call reads the persisted max and yields
    # the next strictly-increasing ordinal.
    first = P.next_finish_seq(plan)
    plan["subtasks"][0]["finished_seq"] = first
    second = P.next_finish_seq(plan)
    assert second == first + 1
    # Simulate a reload: a plan persisted with a higher ordinal must continue
    # past it, so a restart never reuses an earlier number.
    plan["subtasks"][1]["finished_seq"] = 7
    assert P.next_finish_seq(plan) == 8


def test_final_verify_orders_by_finished_seq_not_timestamp(tmp_path):
    # Clock skew: the modify subtask has an EARLIER wall-clock timestamp but a
    # LATER completion ordinal. Ordering must follow the monotonic sequence, so
    # the modification is seen as happening after the verification.
    plan = P.new_plan_record(
        [sub("1", kind="modify", write_scope=["a.py"]),
         sub("2", kind="verify")],
        "llm", requires_approval=False,
        goal="修复 a.py 的 bug",
        goal_clauses=P.split_goal_clauses("修复 a.py 的 bug"),
    )
    s1, s2 = plan["subtasks"]
    s1["status"] = P.SUBTASK_DONE
    s1["finished_at"] = "2026-01-01T00:00:00+00:00"  # earlier (skewed)
    s1["finished_seq"] = 2                            # but completed last
    s1["acceptance_evidence"] = [{"type": "file", "path": "a.py", "sha256": ""}]
    s2["status"] = P.SUBTASK_DONE
    s2["finished_at"] = "2026-01-02T00:00:00+00:00"   # later (skewed)
    s2["finished_seq"] = 1                            # but completed first
    s2["acceptance_evidence"] = [{
        "type": "command", "program": "python", "args": ["-m", "pytest"],
        "ok": True, "summary": "exit_code=0",
    }]
    missing, inconsistent = W._final_rule_checks({
        "goal": "修复 a.py 的 bug", "plan": plan, "workspace": str(tmp_path),
    })
    # A timestamp-only ordering would wrongly put verify AFTER modify and clear
    # the check; the monotonic seq correctly flags the unverified modification.
    assert not missing
    assert any("最后一次文件修改" in i for i in inconsistent)
