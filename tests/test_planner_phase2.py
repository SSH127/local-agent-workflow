from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine import planning as P
from app.planner import quality as Q
from app.planner.repo_index import build_index


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "def target_fn(x):\n    return x\n\ndef helper():\n    return target_fn(1)\n",
        encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(
        "from pkg.mod import target_fn\n\ndef test_target_fn():\n    assert target_fn(1)==1\n",
        encoding="utf-8")
    return tmp_path


def test_classify_and_limits():
    assert Q.classify_task_type("分析 MAX_PLAN_REJECTIONS 在哪里生效") == "analysis"
    assert Q.classify_task_type("修复 check_write_scopes 的缺陷") == "bugfix"
    assert Q.classify_task_type("把租约函数拆到独立模块") == "refactor"
    assert Q.max_subtasks_for("分析流程") == 3


def test_enforce_rejects_broad_glob(repo: Path):
    specs = [{
        "id": "1", "title": "fix", "description": "fix target_fn", "kind": "modify",
        "depends_on": [], "write_scope": ["pkg/**"],
        "targets": [{"path": "pkg/mod.py", "symbol": "target_fn", "reason": "bug", "evidence_ref": "L1", "status": "confirmed"}],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest tests/test_mod.py"}],
    }, {
        "id": "2", "title": "v", "description": "v", "kind": "verify", "depends_on": ["1"], "write_scope": [],
        "targets": [], "acceptance_criteria": [{"precondition": "a", "expected_behavior": "pass", "test_target": "pytest -q"}],
    }]
    with pytest.raises(P.PlanError, match="禁止使用宽泛"):
        Q.enforce_plan_quality(specs, "修复 target_fn", repo, strict=True)


def test_enforce_rejects_fake_symbol(repo: Path):
    specs = [{
        "id": "1", "title": "fix", "description": "fix", "kind": "modify",
        "depends_on": [], "write_scope": ["pkg/mod.py"],
        "targets": [{"path": "pkg/mod.py", "symbol": "no_such", "reason": "x", "evidence_ref": "L1", "status": "confirmed"}],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest tests/test_mod.py"}],
    }, {
        "id": "2", "title": "v", "description": "v", "kind": "verify", "depends_on": ["1"], "write_scope": [],
        "targets": [], "acceptance_criteria": [{"precondition": "a", "expected_behavior": "pass", "test_target": "pytest -q"}],
    }]
    with pytest.raises(P.PlanError, match="不存在"):
        Q.enforce_plan_quality(specs, "修复 target_fn", repo, strict=True)


def test_to_verify_allows_unknown_symbol(repo: Path):
    specs = [{
        "id": "1", "title": "fix", "description": "fix", "kind": "modify",
        "depends_on": [], "write_scope": ["pkg/mod.py"],
        "targets": [{"path": "pkg/mod.py", "symbol": "maybe", "reason": "x", "evidence_ref": "", "status": "to-verify"}],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest tests/test_mod.py"}],
    }, {
        "id": "2", "title": "v", "description": "v", "kind": "verify", "depends_on": ["1"], "write_scope": [],
        "targets": [], "acceptance_criteria": [{"precondition": "a", "expected_behavior": "pass", "test_target": "pytest -q"}],
    }]
    assert Q.enforce_plan_quality(specs, "修复 target_fn", repo, strict=True) == []


def test_analysis_no_modify(repo: Path):
    specs = [{
        "id": "1", "title": "改", "description": "改", "kind": "modify",
        "depends_on": [], "write_scope": ["pkg/mod.py"],
        "targets": [{"path": "pkg/mod.py", "symbol": "target_fn", "status": "confirmed", "reason": "x", "evidence_ref": "L1"}],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest tests/test_mod.py"}],
    }]
    with pytest.raises(P.PlanError, match="禁止包含 modify"):
        Q.enforce_plan_quality(specs, "分析 target_fn 的调用关系", repo, strict=True)


def test_handoff_packets(repo: Path):
    specs = [{
        "id": "1", "title": "fix", "description": "fix", "kind": "modify",
        "depends_on": [], "write_scope": ["pkg/mod.py"],
        "targets": [{"path": "pkg/mod.py", "symbol": "target_fn", "reason": "bug", "evidence_ref": "L1", "status": "confirmed"}],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest tests/test_mod.py::test_target_fn"}],
    }]
    pk = Q.build_handoff_packets(specs, repo)
    assert pk[0]["symbols"][0]["line"] == 1
    assert pk[0]["snippets"] and "target_fn" in pk[0]["snippets"][0]["text"]
    assert "pytest" in pk[0]["test_command"]


def test_subtask_spec_accepts_structured_acceptance_and_targets():
    s = P.SubtaskSpec(
        id="1", title="t", kind="modify",
        write_scope=["app/config.py"],
        targets=[{"path": "app/config.py", "symbol": "X", "reason": "r", "evidence_ref": "L1"}],
        acceptance_criteria=[{"precondition": "a", "expected_behavior": "b", "test_target": "pytest t", "edge_case": "e"}],
    )
    assert s.targets[0]["symbol"] == "X"
    assert "前提:a" in s.acceptance_criteria[0] and "测试:pytest t" in s.acceptance_criteria[0]


def test_phase2_messages_use_new_schema():
    msgs, _ = P.build_grounded_planner_messages("修复 target_fn", ".", 4, phase2=True)
    assert "targets" in msgs[-1]["content"] and "task_type" in msgs[-1]["content"]
    assert "PHASE2" not in msgs[0]["content"]  # prompt text, not constant name
    assert "可被弱模型执行" in msgs[0]["content"]
