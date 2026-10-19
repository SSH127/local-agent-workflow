"""v0.3.3 residual-limit fixes: artifact path policy + final-verify replan."""
from __future__ import annotations

import json

import pytest

from app import config
from app.engine import planning as P
from app.tools.command_sandbox import (
    begin_session,
    finish_session,
    is_artifact_path,
)
from tests.test_task_graph import (
    FakeExecutor,
    PlannerLLM,
    Recorder,
    plan_json,
    run,
    setup_planner,
    sub,
)


@pytest.fixture
def planner_on(monkeypatch, tmp_path):
    return setup_planner(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# A) Artifact dirs
# ---------------------------------------------------------------------------


def test_is_artifact_path_defaults():
    assert is_artifact_path("dist/index.js")
    assert is_artifact_path("build/out.js")
    assert is_artifact_path("app/__pycache__/x.pyc")
    assert is_artifact_path("pkg.egg-info/PKG-INFO")
    assert is_artifact_path(".next/cache/x")
    assert not is_artifact_path("app/main.py")
    assert not is_artifact_path("tests/test_a.py")


def test_dist_change_allow_not_scope_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "allow")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("a\n", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("old\n", encoding="utf-8")
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="npm",
    )
    (tmp_path / "dist" / "bundle.js").write_text("new\n", encoding="utf-8")
    finish = finish_session(session)
    assert finish.violations == []
    assert any(a.path == "dist/bundle.js" for a in finish.artifacts)
    assert (tmp_path / "dist" / "bundle.js").read_text(encoding="utf-8") == "new\n"


def test_source_outside_scope_still_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "allow")
    (tmp_path / "app").mkdir()
    victim = tmp_path / "app" / "main.py"
    victim.write_text("keep\n", encoding="utf-8")
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="pytest",
    )
    victim.write_text("tampered\n", encoding="utf-8")
    finish = finish_session(session)
    assert any(v.path == "app/main.py" for v in finish.violations)
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_artifact_policy_restore_puts_dist_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "restore")
    (tmp_path / "dist").mkdir()
    target = tmp_path / "dist" / "bundle.js"
    target.write_text("old\n", encoding="utf-8")
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="npm",
    )
    target.write_text("new\n", encoding="utf-8")
    finish = finish_session(session)
    assert finish.violations == []
    assert any(a.path == "dist/bundle.js" and a.action == "restored" for a in finish.artifacts)
    assert target.read_text(encoding="utf-8") == "old\n"


def test_artifact_policy_fail_treats_dist_as_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "fail")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("old\n", encoding="utf-8")
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="npm",
    )
    (tmp_path / "dist" / "bundle.js").write_text("new\n", encoding="utf-8")
    finish = finish_session(session)
    assert any(v.path == "dist/bundle.js" for v in finish.violations)
    assert (tmp_path / "dist" / "bundle.js").read_text(encoding="utf-8") == "old\n"


# ---------------------------------------------------------------------------
# B) Final-verify replan
# ---------------------------------------------------------------------------


def test_classify_final_verify_missing_is_structural():
    assert P.classify_final_verify_failure(["条款 C2 未覆盖"], []) == P.FAIL_STRUCTURAL


def test_classify_final_verify_inconsistency_fails():
    assert P.classify_final_verify_failure([], ["文件 a.py 摘要不一致"]) == "fail"


def test_final_verify_replan_on_missing_clauses(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_MAX_REPLANS", 1)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    llm = PlannerLLM([
        plan_json(sub("1", serves=("C1",))),
        json.dumps({"achieved": False, "missing_clauses": ["没有输出部署说明"],
                    "inconsistencies": []}, ensure_ascii=False),
        plan_json(sub("2", title="补充说明文档", kind="analyze", serves=("C1",))),
        json.dumps({"achieved": True, "missing_clauses": [],
                    "inconsistencies": []}, ensure_ascii=False),
    ])
    rec = Recorder()
    res = run("分析项目结构并输出部署说明", llm, rec, FakeExecutor({}), monkeypatch, planner_on)
    plan = rec.plan_store.load()
    assert plan.get("final_verify_replans") == 1
    assert plan.get("final_verify", {}).get("replan_triggered") is True or plan.get("version", 1) >= 2
    assert res["status"] == "succeeded", res
    # Version history / reason should mention final verify
    hist = " ".join(str(h.get("reason") or "") for h in (plan.get("history") or []))
    versions = getattr(rec.plan_store, "versions", []) or []
    reasons = hist + " " + " ".join(str(v.get("reason") or "") for v in versions)
    assert "最终验收" in reasons



def test_final_verify_replan_budget_exhausted(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_MAX_REPLANS", 1)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    llm = PlannerLLM([
        plan_json(sub("1", serves=("C1",))),
        json.dumps({"achieved": False, "missing_clauses": ["缺部署说明"],
                    "inconsistencies": []}, ensure_ascii=False),
        plan_json(sub("2", title="补充说明文档", kind="analyze", serves=("C1",))),
        json.dumps({"achieved": False, "missing_clauses": ["仍然缺部署说明"],
                    "inconsistencies": []}, ensure_ascii=False),
    ])
    rec = Recorder()
    res = run("分析项目结构并输出部署说明", llm, rec, FakeExecutor({}), monkeypatch, planner_on)
    plan = rec.plan_store.load()
    assert res["status"] == "failed", res
    assert int(plan.get("final_verify_replans") or 0) >= 1
    err = res.get("error") or ""
    tips = plan.get("final_verify", {}).get("suggested_next_steps") or []
    assert "FINAL_VERIFY_MAX_REPLANS" in err or any("FINAL_VERIFY_MAX_REPLANS" in t for t in tips) or any(
        "上限" in t for t in tips
    )



def test_final_verify_disabled_replan_fails_immediately(planner_on, monkeypatch):
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", True)
    monkeypatch.setattr(config, "FINAL_VERIFY_REPLAN", False)
    llm = PlannerLLM([
        plan_json(sub("1")),
        json.dumps({"achieved": False, "missing_clauses": ["缺口"],
                    "inconsistencies": []}, ensure_ascii=False),
    ])
    res = run("分析并部署", llm, Recorder(), FakeExecutor({}), monkeypatch, planner_on)
    assert res["status"] == "failed"
    assert "缺口" in res["error"]
