from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.engine import planning as P
from app.planner.evidence import (
    augment_registry, build_registry, check_evidence_binding, coverage_files, render_registry,
)
from app.planner.explore import explore_repo
from app.planner.repo_index import build_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "security.py").write_text(
        "PROTECTED = {'.env'}\n\n\ndef is_protected_file(path, root):\n    return path.name in PROTECTED\n",
        encoding="utf-8")
    (tmp_path / "app" / "overview.py").write_text(
        "from app.security import is_protected_file\n\n\ndef build_project_overview(root):\n"
        "    return [p for p in root.iterdir() if not is_protected_file(p, root)]\n",
        encoding="utf-8")
    (tmp_path / "app" / "plan.py").write_text("def new_plan_record():\n    return {}\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_security.py").write_text(
        "from app.security import is_protected_file\n\n\ndef test_env(tmp_path):\n"
        "    assert is_protected_file(tmp_path / '.env', tmp_path)\n", encoding="utf-8")
    return tmp_path


def _er(repo: Path):
    return explore_repo(
        "评估受保护文件（如 .env）在规划概览中是否被排除", repo,
        mock_steps=[
            {"name": "search_symbol", "args": {"query": "is_protected_file"}},
            {"name": "read_symbol", "args": {"path": "app/overview.py", "symbol": "build_project_overview"}},
            {"name": "list_callers", "args": {"symbol": "is_protected_file"}},
        ],
        clauses=[{"id": "C1", "text": "受保护文件在规划概览中被排除"}],
    )


def test_registry_has_real_lines_and_kinds(repo: Path) -> None:
    er = _er(repo)
    reg = er.registry
    assert reg and all(e["evidence_id"].startswith("EV") for e in reg)
    d = next(e for e in reg if e["symbol"] == "is_protected_file" and e["kind"] == "definition")
    assert (d["line_start"], d["line_end"]) == (4, 5)
    assert any(e["kind"] == "caller" for e in reg)
    assert any(e["kind"] == "test" for e in reg)
    assert all(e["supports"] for e in reg)
    assert d["evidence_id"] in er.text and "证据登记表" in er.text


def test_registry_never_registers_undefined_symbol(repo: Path) -> None:
    class It:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    idx = build_index(repo)
    reg = build_registry([It(kind="read", path="app/plan.py", symbol="ghost_fn", detail="", source="tool")], idx)
    assert not any(e["symbol"] == "ghost_fn" for e in reg)


def _ids(reg, sym, kind="definition"):
    return next(e["evidence_id"] for e in reg if e["symbol"] == sym and e["kind"] == kind)


def test_binding_passes_with_cited_evidence(repo: Path) -> None:
    reg = _er(repo).registry
    specs = [{"id": "1", "kind": "analyze", "write_scope": [], "targets": [
        {"path": "app/security.py", "symbol": "is_protected_file", "evidence_id": _ids(reg, "is_protected_file")},
        {"path": "app/overview.py", "symbol": "build_project_overview", "evidence_id": _ids(reg, "build_project_overview")},
    ]}]
    cov = [{"path": f, "role": "read", "reason": "x"} for f in coverage_files(reg)]
    errs, missing = check_evidence_binding(specs, reg, task_type="analysis", coverage=cov)
    assert errs == [] and missing == []


def test_binding_rejects_wrong_target_and_missing_coverage(repo: Path) -> None:
    reg = _er(repo).registry
    eid = _ids(reg, "is_protected_file")
    specs = [{"id": "1", "kind": "analyze", "write_scope": [], "targets": [
        {"path": "app/plan.py", "symbol": "new_plan_record", "evidence_id": eid},   # mismatched
        {"path": "app/plan.py", "symbol": "new_plan_record"},                       # no evidence
    ]}]
    errs, missing = check_evidence_binding(specs, reg, task_type="analysis", coverage=[])
    text = "\n".join(errs)
    assert "不一致" in text and "缺少 evidence_id" in text and "coverage 缺少" in text
    assert "new_plan_record" in missing


def test_binding_modify_rules(repo: Path) -> None:
    reg = _er(repo).registry
    eid = _ids(reg, "is_protected_file")
    cov = [{"path": f, "role": "modify" if f == "app/security.py" else "read", "reason": "x"}
           for f in coverage_files(reg)]
    bad = [{"id": "1", "kind": "modify", "write_scope": ["app/security.py", "app/overview.py"], "targets": [
        {"path": "app/security.py", "symbol": "is_protected_file", "evidence_id": eid, "status": "to-verify"}]}]
    errs, _ = check_evidence_binding(bad, reg, task_type="bugfix", coverage=cov)
    text = "\n".join(errs)
    assert "不允许 to-verify" in text and "没有对应 target" in text
    good = [{"id": "1", "kind": "modify", "write_scope": ["app/security.py", "tests/test_new.py"], "targets": [
        {"path": "app/security.py", "symbol": "is_protected_file", "evidence_id": eid},
        {"path": "tests/test_new.py", "symbol": "test_x", "status": "new"}]}]
    errs, _ = check_evidence_binding(good, reg, task_type="bugfix", coverage=cov)
    assert errs == []


def test_not_needed_requires_evidence(repo: Path) -> None:
    reg = _er(repo).registry
    files = coverage_files(reg)
    cov = [{"path": f, "role": "not-needed", "reason": ""} for f in files]
    errs, _ = check_evidence_binding([], reg, task_type="analysis", coverage=cov)
    assert any("not-needed" in e for e in errs)


def test_augment_registry_extra_round(repo: Path) -> None:
    idx = build_index(repo)
    reg: list[dict] = []
    new = augment_registry(reg, idx, ["new_plan_record"])
    assert new and reg[0]["path"] == "app/plan.py" and reg[0]["line_start"] == 1


def test_phase2_request_rejects_unbound_plan(repo: Path, monkeypatch) -> None:
    import json
    from app.agents.llm import LLMResponse
    reg = _er(repo).registry
    plan = {"subtasks": [{"id": "1", "title": "看", "description": "只读分析", "kind": "analyze",
                          "serves_clauses": ["C1"], "write_scope": [],
                          "targets": [{"path": "app/plan.py", "symbol": "new_plan_record"}],
                          "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "人工审查"}]}],
            "coverage": []}

    class L:
        calls = 0
        def chat(self, m, t):
            L.calls += 1
            return LLMResponse(content=json.dumps(plan, ensure_ascii=False))
    monkeypatch.setattr(P.config, "PLANNER_STRICT", False)
    meta: dict = {}
    specs, errors, _ = P.request_subtasks_phase2(
        L(), [{"role": "user", "content": "x"}], 3, goal="分析 is_protected_file", workspace=repo,
        clause_ids=["C1"], retries=1, registry=reg, meta_out=meta)
    assert specs is None and L.calls == 2
    assert any("证据绑定失败" in e for e in errors)
    assert "new_plan_record" in meta["binding_missing"]


def test_eval_extra_columns_reporting_only(repo: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_plans as E
    reg = _er(repo).registry
    eid = _ids(reg, "is_protected_file")
    plan = {"subtasks": [{"id": "1", "kind": "analyze", "targets": [
        {"path": "app/security.py", "symbol": "is_protected_file", "evidence_id": eid},
        {"path": "app/plan.py", "symbol": "ghost"}]}], "evidence_registry": reg, "coverage": []}
    goal = {"expected_symbols": ["is_protected_file"], "expected_files": ["app/security.py"]}
    ex = E.extra_columns(goal, plan, build_index(repo))
    assert ex["evidence_coverage"] == 0.5 and ex["wrong_target_rate"] == 0.5
    assert ex["expected_symbol_as_target"] == 1.0
