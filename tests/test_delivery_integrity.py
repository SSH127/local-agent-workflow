"""Regression tests for the three delivery-integrity defects:

1. ``done`` / ``covered`` must be backed by per-kind delivery evidence, never a
   bare ``status == done`` flag;
2. a final-acceptance failure (including hitting ``FINAL_VERIFY_MAX_REPLANS``)
   must persist a structured failure report readable from storage/API;
3. a long subtask result must be preserved in full while the model-input
   summary and the final delivery report are explicitly marked as summaries.
"""
from __future__ import annotations

import json

import pytest

from app import config, storage
from app.engine import planning as P
from app.engine import workflow as W
from tests.test_task_graph import (
    FakeExecutor,
    PlannerLLM,
    Recorder,
    make_run_row,
    ok,
    plan_json,
    run,
    setup_planner,
    sub,
)


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


def _analyze_evidence(tool_calls=1, tool_successes=1, tool_failures=0):
    return [{
        "type": "executor_status", "status": "succeeded",
        "tool_calls": tool_calls, "tool_successes": tool_successes,
        "tool_failures": tool_failures,
    }]


def _done_plan(kind="analyze", serves=("C1",), evidence=None, write_scope=None,
               goal="分析项目结构", clauses=None):
    plan = P.new_plan_record(
        [sub("1", kind=kind, write_scope=write_scope, serves=serves)],
        "llm", requires_approval=False,
        goal=goal,
        goal_clauses=clauses or [{"id": "C1", "text": "分析项目结构"}],
    )
    plan["subtasks"][0]["status"] = P.SUBTASK_DONE
    plan["subtasks"][0]["acceptance_evidence"] = list(evidence or [])
    return plan


# ---------------------------------------------------------------------------
# Defect 1: done / covered must be backed by per-kind delivery evidence
# ---------------------------------------------------------------------------


def test_analyze_with_only_failed_tool_calls_is_not_verified():
    plan = _done_plan(evidence=_analyze_evidence(tool_calls=3, tool_successes=0, tool_failures=3))
    sub1 = plan["subtasks"][0]
    assert P.subtask_is_verified(sub1) is False
    verified, gaps = P.subtask_delivery(sub1)
    assert verified is False and gaps
    report = {c["id"]: c for c in P.clause_report(plan)}
    assert report["C1"]["status"] != "covered"


def test_analyze_with_a_successful_tool_call_is_verified():
    plan = _done_plan(evidence=_analyze_evidence(tool_calls=2, tool_successes=1, tool_failures=1))
    assert P.subtask_is_verified(plan["subtasks"][0]) is True


def test_analyze_with_manual_confirmation_is_verified():
    plan = _done_plan(evidence=[{"type": "manual", "by": "user", "status": "done",
                                 "note": "已人工核对", "evidence_text": "", "at": "t"}])
    assert P.subtask_is_verified(plan["subtasks"][0]) is True


def test_modify_with_failed_verification_is_not_covered():
    plan = _done_plan(
        kind="modify", write_scope=["a.py"],
        evidence=[
            {"type": "file", "path": "a.py", "sha256": "abc"},
            {"type": "command", "program": "python", "args": ["-m", "pytest"],
             "ok": False, "summary": "exit_code=1"},
        ],
        goal="修复 a.py",
        clauses=[{"id": "C1", "text": "修复 a.py"}],
    )
    verified, gaps = P.subtask_delivery(plan["subtasks"][0])
    assert verified is False
    assert any("验证" in g for g in gaps)
    report = {c["id"]: c for c in P.clause_report(plan)}
    assert report["C1"]["status"] != "covered"


def test_verify_requires_passing_check_or_manual():
    plan = _done_plan(kind="verify", evidence=_analyze_evidence(tool_calls=1))
    assert P.subtask_is_verified(plan["subtasks"][0]) is False  # reads alone are not proof
    plan["subtasks"][0]["acceptance_evidence"] = [{
        "type": "command", "program": "python", "args": ["-m", "pytest"],
        "ok": True, "summary": "exit_code=0",
    }]
    assert P.subtask_is_verified(plan["subtasks"][0]) is True


def test_clause_report_marks_done_without_evidence_as_insufficient():
    plan = _done_plan(kind="modify", write_scope=["a.py"],
                      goal="修复 a.py", clauses=[{"id": "C1", "text": "修复 a.py"}])
    report = {c["id"]: c for c in P.clause_report(plan)}
    assert report["C1"]["status"] == "insufficient"
    assert report["C1"]["covered_by"] == []


# ---------------------------------------------------------------------------
# Defect 2: final-acceptance failure persists a structured failure report
# ---------------------------------------------------------------------------


def _review_fail(missing):
    return json.dumps({"achieved": False, "missing_clauses": missing,
                       "inconsistencies": []}, ensure_ascii=False)


def test_final_verify_failure_persists_structured_report(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    make_run_row("run-fail-report")
    store = W.StoragePlanStore("run-fail-report")
    llm = PlannerLLM([plan_json(sub("1")), _review_fail(["缺口"])])
    res = run("分析并部署", llm, Recorder(store), FakeExecutor({}), monkeypatch,
              planner_on, run_id="run-fail-report")
    assert res["status"] == "failed", res

    # The report must be readable from storage, not only from the in-memory plan.
    plan = storage.get_plan("run-fail-report")
    report = plan.get("failure_report")
    assert report is not None
    assert report["run_id"] == "run-fail-report"
    assert report["status"] in ("failed", "final_verify_failed")
    assert report["failure_reason"]
    assert "缺口" in report["failure_reason"]
    assert isinstance(report["unmet_clauses"], list)
    assert isinstance(report["inconsistencies"], list)
    assert isinstance(report["available_evidence"], list)
    assert isinstance(report["evidence_gaps"], list)
    assert report["evidence_gaps"]  # the missing requirement is a documented gap
    assert report["suggested_next_steps"]


def test_final_verify_replan_limit_persists_failure_report(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_MAX_REPLANS", 1)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    make_run_row("run-limit-report")
    store = W.StoragePlanStore("run-limit-report")
    llm = PlannerLLM([
        plan_json(sub("1", serves=("C1",))),
        _review_fail(["缺部署说明"]),
        plan_json(sub("2", title="补充说明文档", kind="analyze", serves=("C1",))),
        _review_fail(["仍然缺部署说明"]),
    ])
    res = run("分析项目结构并输出部署说明", llm, Recorder(store), FakeExecutor({}),
              monkeypatch, planner_on, run_id="run-limit-report")
    assert res["status"] == "failed", res

    plan = storage.get_plan("run-limit-report")
    assert int(plan.get("final_verify_replans") or 0) >= 1
    report = plan.get("failure_report")
    assert report is not None
    assert report["run_id"] == "run-limit-report"
    assert report["failure_reason"]
    assert any("上限" in t or "FINAL_VERIFY_MAX_REPLANS" in t
               for t in report["suggested_next_steps"])


def test_failure_report_does_not_fabricate_evidence(planner_on, monkeypatch):
    """A failed run with NO real evidence must not present unattained evidence
    as confirmed facts in the report."""
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    llm = PlannerLLM([plan_json(sub("1")), _review_fail(["缺口"])])
    # ``ok(tool_calls=0)``: the executor reports success text but made no tool
    # call, so acceptance must reject it (no fabricated evidence).
    ex = FakeExecutor({"1": [ok(tool_calls=0)]})
    rec = Recorder()
    res = run("分析并部署", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == "failed", res
    report = rec.plan_store.load().get("failure_report")
    assert report is not None
    # The report records only what was really observed; a zero-tool subtask has
    # no evidence entries and is instead listed as an evidence gap.
    assert report["available_evidence"] == [] or all(
        e.get("evidence") for e in report["available_evidence"])


# ---------------------------------------------------------------------------
# Defect 3: full result preserved; summaries and reports are clearly marked
# ---------------------------------------------------------------------------


def test_long_result_preserved_in_full_and_summary_is_marked(planner_on, monkeypatch):
    long_summary = "复" * 2500  # exceeds the model-input summary bound
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    make_run_row("run-long-result")
    store = W.StoragePlanStore("run-long-result")
    llm = PlannerLLM([plan_json(sub("1", kind="analyze"))])
    ex = FakeExecutor({"1": [ok(summary=long_summary)]})
    res = run("分析项目结构", llm, Recorder(store), ex, monkeypatch, planner_on,
              run_id="run-long-result")
    assert res["status"] == "succeeded", res

    plan = storage.get_plan("run-long-result")
    s1 = plan["subtasks"][0]
    # The complete raw result is preserved and persisted (not silently lost).
    assert s1["result_full"] == long_summary
    # The model-input summary is bounded AND clearly marked as a summary.
    assert len(s1["result_summary"]) < len(long_summary)
    assert ("截断" in s1["result_summary"] or "摘要" in s1["result_summary"])
    assert "result_full" in s1["result_summary"]  # points to where the full text lives


def test_failure_report_preserves_full_result_without_silent_truncation(
        planner_on, monkeypatch):
    long_summary = "结" * 2500
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    llm = PlannerLLM([plan_json(sub("1")), _review_fail(["缺口"])])
    ex = FakeExecutor({"1": [ok(summary=long_summary)]})
    rec = Recorder()
    res = run("分析并部署", llm, rec, ex, monkeypatch, planner_on)
    assert res["status"] == "failed", res

    report = rec.plan_store.load().get("failure_report")
    assert report is not None
    # The final delivery report must carry the full result, not a silent cut.
    full_results = [e.get("result_full") for e in report["available_evidence"]]
    assert long_summary in full_results
