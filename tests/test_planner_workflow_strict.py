from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.llm import LLMResponse
from app.engine import planning as P
from app.engine import workflow as W


def _plan(targets: bool) -> dict:
    st = {"id": "1", "title": "分析", "description": "只读分析 new_plan_record", "kind": "analyze",
          "serves_clauses": ["C1"], "write_scope": [],
          "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "人工审查"}]}
    if targets:
        st["targets"] = [{"path": "app/plan.py", "symbol": "new_plan_record"}]
    return {"subtasks": [st], "coverage": []}


class FakeLLM:
    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        p = self.plans[min(self.calls - 1, len(self.plans) - 1)]
        return LLMResponse(content=json.dumps(p, ensure_ascii=False))


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "plan.py").write_text("def new_plan_record():\n    return {}\n", encoding="utf-8")
    return tmp_path


MSG = [{"role": "user", "content": "x"}]


def test_lenient_plan_falls_back_without_extra_calls(repo, monkeypatch):
    monkeypatch.setattr(P, "strict_check", lambda *a, **k: ["缺少 targets"])
    llm = FakeLLM([_plan(False)])
    meta: dict = {}
    specs, _ = P.request_subtasks_strict_first(llm, MSG, 3, goal="分析 new_plan_record",
                                               workspace=repo, clause_ids=["C1"], meta_out=meta)
    assert specs and meta["mode"] == "lenient-fallback" and llm.calls == 1
    assert meta["strict_errors"] == ["缺少 targets"]


def test_strict_attempt_gets_one_retry_then_passes(repo, monkeypatch):
    seen = []

    def fake_check(specs, **k):
        seen.append(1)
        return ["bad"] if len(seen) == 1 else []
    monkeypatch.setattr(P, "strict_check", fake_check)
    monkeypatch.setattr(P.config, "PLANNER_WORKFLOW_CRITIC", False)
    llm = FakeLLM([_plan(True), _plan(True)])
    meta: dict = {}
    specs, _ = P.request_subtasks_strict_first(llm, MSG, 3, goal="分析 new_plan_record",
                                               workspace=repo, clause_ids=["C1"], meta_out=meta)
    assert meta["mode"] == "strict" and llm.calls == 2 and meta["strict_retry"]
    assert meta["handoff_packets"] and len(meta["handoff_packets"]) == 1


def test_strict_pass_runs_critic_once(repo, monkeypatch):
    monkeypatch.setattr(P, "strict_check", lambda *a, **k: [])
    monkeypatch.setattr(P.config, "PLANNER_WORKFLOW_CRITIC", True)
    monkeypatch.setattr(P.config, "PLANNER_CRITIC", True)
    calls = []
    import app.planner.quality as Q
    monkeypatch.setattr(Q, "critic_revise", lambda *a, **k: (calls.append(1), (None, {"verdict": "pass"}))[1])
    meta: dict = {}
    specs, _ = P.request_subtasks_strict_first(FakeLLM([_plan(True)]), MSG, 3, goal="分析 x",
                                               workspace=repo, clause_ids=["C1"], meta_out=meta)
    assert meta["mode"] == "strict" and calls == [1] and meta["critic"]["verdict"] == "pass"


def test_real_strict_check_rejects_unbound_targets(repo):
    from app.planner.explore import explore_repo
    reg = explore_repo("分析 new_plan_record", repo, llm=None).registry
    specs = P.validate_subtasks(_plan(True), 3, clause_ids=["C1"])
    errs = P.strict_check(specs, goal="分析 new_plan_record", workspace=repo, registry=reg, coverage=[])
    assert errs and "证据绑定失败" in errs[0]


def test_workflow_hard_failure_uses_old_lenient_path(repo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(P, "request_subtasks_strict_first", boom)
    monkeypatch.setattr(W.config, "PLANNER_STRICT", True)
    monkeypatch.setattr(W.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(W.config, "PLANNER_WORKFLOW_STRICT", True)
    llm = FakeLLM([_plan(False)])
    state = {"goal": "分析 new_plan_record", "llm": llm, "workspace": str(repo)}
    specs, source, _ = W._generate_specs(state, [], [{"id": "C1", "text": "分析 new_plan_record"}])
    assert source == "llm" and specs and llm.calls == 1
    assert state["planner_meta"]["mode"] == "lenient-fallback"
    plan: dict = {}
    W._attach_grounding_warnings(state, plan, source)
    assert plan["planner_meta"]["mode"] == "lenient-fallback"
    assert any("回退" in n for n in plan["notes"])


def test_workflow_flag_off_skips_strict(repo, monkeypatch):
    called = []
    monkeypatch.setattr(P, "request_subtasks_strict_first", lambda *a, **k: called.append(1))
    monkeypatch.setattr(W.config, "PLANNER_ENABLED", True)
    monkeypatch.setattr(W.config, "PLANNER_WORKFLOW_STRICT", False)
    llm = FakeLLM([_plan(False)])
    state = {"goal": "分析 new_plan_record", "llm": llm, "workspace": str(repo)}
    specs, source, _ = W._generate_specs(state, [], [{"id": "C1", "text": "分析 new_plan_record"}])
    assert source == "llm" and not called and "planner_meta" not in state