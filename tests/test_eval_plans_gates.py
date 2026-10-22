from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_plans as E  # noqa: E402


def test_broad_glob_blocks():
    goal = {"id": "T", "type": "bugfix", "goal": "fix x", "expected_files": [], "expected_symbols": [], "expected_kinds": ["modify", "verify"]}
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "title": "fix", "description": "x", "write_scope": ["app/engine/**"],
         "depends_on": [], "acceptance_criteria": ["前提:a | 期望:b | 测试:pytest t"]},
        {"id": "2", "kind": "verify", "title": "v", "description": "v", "write_scope": [], "depends_on": ["1"],
         "acceptance_criteria": ["前提:a | 期望:pass | 测试:pytest -q"]},
    ]}
    r = E.score_plan(goal, plan, ROOT)
    assert r["blocked"] and any(g.startswith("broad_glob") for g in r["failed_gates"])


def test_analysis_modify_blocks():
    goal = {"id": "T", "type": "analysis", "goal": "分析流程", "expected_files": [], "expected_symbols": []}
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "title": "改", "description": "改", "write_scope": ["app/config.py"],
         "depends_on": [], "acceptance_criteria": ["x"]},
    ]}
    r = E.score_plan(goal, plan, ROOT)
    assert r["blocked"] and "analysis_has_modify" in r["failed_gates"]


def test_verify_must_depend_modify():
    goal = {"id": "T", "type": "feature", "goal": "add", "expected_files": ["app/config.py"], "expected_symbols": []}
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "title": "m", "description": "m", "write_scope": ["app/config.py"],
         "depends_on": [], "acceptance_criteria": ["前提:a | 期望:b | 测试:pytest t"]},
        {"id": "2", "kind": "verify", "title": "v", "description": "v", "write_scope": [], "depends_on": [],
         "acceptance_criteria": ["前提:a | 期望:ok | 测试:pytest -q"]},
    ]}
    r = E.score_plan(goal, plan, ROOT)
    assert r["blocked"] and any("verify_no_modify_dep" in g for g in r["failed_gates"])


def test_file_precision_uses_scopes_not_prose(tmp_path: Path):
    goal = {"id": "T", "type": "bugfix", "expected_files": ["app/config.py"], "expected_symbols": [],
            "expected_kinds": ["modify", "verify"]}
    # prose mentions many files but scope is exact
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "title": "m",
         "description": "also mentions app/engine/planning.py and app/main.py",
         "write_scope": ["app/config.py"], "depends_on": [],
         "targets": [{"path": "app/config.py", "symbol": "PLANNER_ENABLED", "status": "to-verify"}],
         "acceptance_criteria": [{"precondition": "给定配置", "expected_behavior": "可读", "test_target": "pytest tests/test_task_graph.py"}]},
        {"id": "2", "kind": "verify", "title": "v", "description": "run pytest", "write_scope": [],
         "depends_on": ["1"], "acceptance_criteria": [{"precondition": "改完", "expected_behavior": "通过", "test_target": "pytest -q"}]},
    ]}
    r = E.score_plan(goal, plan, ROOT)
    assert not r["blocked"]
    assert r["file_recall"] == 1.0
    assert r["file_precision"] == 1.0


def test_structured_acceptance_scores_high():
    subs = [{"kind": "modify", "acceptance_criteria": [
        {"precondition": "给定失败用例", "expected_behavior": "应返回 True", "test_target": "pytest tests/test_x.py::test_y", "edge_case": "空输入"}
    ]}]
    assert E._acceptance_score(subs) == 1.0
    assert E._acceptance_score([{"acceptance_criteria": ["跑一下测试通过"]}]) < 0.5


def test_pearson_and_gpt_file():
    assert abs(E.pearson([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    review = json.loads((ROOT / "evals" / "gpt_review_20261021.json").read_text(encoding="utf-8"))
    assert review["grounded"]["G01"] == 4.4
    assert len(review["ungrounded"]) == 10
