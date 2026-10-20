"""Static checks (write scope / goal clauses), goal traceability and the
append-only plan version history (storage + API + lineage + diff)."""
from __future__ import annotations

import pytest

from app import config, storage
from app.agents import agent_executor
from app.engine import planning as P
from app.engine import workflow as W
from tests.test_task_graph import (
    PAUSED,
    FakeExecutor,
    PlannerLLM,
    Recorder,
    _approve,
    _wait,
    failed,
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


@pytest.fixture
def api(planner_on, monkeypatch):
    from fastapi.testclient import TestClient

    from app.engine import runner as runner_mod
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", "")
    holder: dict = {}
    monkeypatch.setattr(runner_mod, "DeepSeekClient", lambda: holder["llm"])
    from app.main import app
    with TestClient(app) as client:
        yield client, holder

TWO_CLAUSES = "分析配置解析。总结测试覆盖。"

# ---------------------------------------------------------------------------
# Static checks: write scope
# ---------------------------------------------------------------------------


def test_modify_with_empty_write_scope_rejected():
    with pytest.raises(P.PlanError, match="write_scope 为空"):
        P.validate_subtasks([sub("1", kind="modify", write_scope=[])], 8)


def test_overlapping_scopes_without_dependency_path_rejected():
    specs = [sub("1", kind="modify", write_scope=["app/**"]),
             sub("2", kind="modify", write_scope=["app/main.py"])]
    with pytest.raises(P.PlanError, match="重叠"):
        P.validate_subtasks(specs, 8)


@pytest.mark.parametrize("deps", [("1",), ])
def test_overlap_allowed_with_dependency_path_either_direction(deps):
    a = sub("1", kind="modify", write_scope=["app/**"])
    b = sub("2", kind="modify", write_scope=["app/main.py"], deps=deps)
    assert [s["id"] for s in P.validate_subtasks([a, b], 8)] == ["1", "2"]
    # Reverse direction and a transitive path also count.
    a2 = sub("1", kind="modify", write_scope=["app/**"], deps=["3"])
    m = sub("3", deps=["2"])
    b2 = sub("2", kind="modify", write_scope=["app/main.py"])
    assert {s["id"] for s in P.validate_subtasks([a2, m, b2], 8)} == {"1", "2", "3"}


def test_disjoint_scopes_need_no_dependency():
    specs = [sub("1", kind="modify", write_scope=["app/**"]),
             sub("2", kind="modify", write_scope=["tests/**"])]
    assert len(P.validate_subtasks(specs, 8)) == 2


def test_analyze_and_verify_default_to_read_only():
    specs = P.validate_subtasks([{"id": "1", "title": "看", "kind": "analyze",
                                  "acceptance_criteria": ["x"]},
                                 {"id": "2", "title": "测", "kind": "verify",
                                  "acceptance_criteria": ["y"], "depends_on": ["1"]}], 8)
    assert [s["write_scope"] for s in specs] == [[], []]


def test_scope_overlap_retries_planner_then_succeeds(planner_on, monkeypatch):
    bad = plan_json(sub("1", kind="modify", write_scope=["app"]),
                    sub("2", kind="modify", write_scope=["app/x.py"]))
    good = plan_json(sub("1", kind="modify", write_scope=["app"]),
                     sub("2", kind="modify", write_scope=["app/x.py"], deps=["1"]))
    llm = PlannerLLM([bad, good])
    rec = Recorder()
    ex = FakeExecutor({"1": [ok(files=["app/a.py"], tests_ok=True)],
                       "2": [ok(files=["app/x.py"], tests_ok=True)]})
    res = run("分析项目结构", llm, rec, ex, monkeypatch, planner_on)
    assert len(llm.calls) == 2
    assert "重叠" in llm.calls[1][-1]["content"]  # the error is fed back
    assert rec.plan_store.load()["source"] == "llm"
    assert res["status"] == "succeeded", res


def test_scope_errors_twice_fall_back_to_single_subtask(planner_on, monkeypatch):
    bad = plan_json(sub("1", kind="modify", write_scope=[]))
    llm = PlannerLLM([bad, bad])
    rec = Recorder()
    run("分析项目结构", llm, rec, FakeExecutor({}), monkeypatch, planner_on)
    plan = rec.plan_store.load()
    assert plan["source"] == "fallback" and len(plan["subtasks"]) == 1
    assert plan["subtasks"][0]["write_scope"] == ["**"]
    assert "write_scope" in " ".join(plan.get("notes") or [])


# ---------------------------------------------------------------------------
# Goal traceability
# ---------------------------------------------------------------------------


def test_goal_split_is_deterministic():
    clauses = P.split_goal_clauses("1. 修复解析器；2. 补充测试\n- 更新 README。")
    assert [c["id"] for c in clauses] == ["C1", "C2", "C3"]
    assert [c["text"] for c in clauses] == ["修复解析器", "补充测试", "更新 README"]
    assert P.split_goal_clauses("分析项目结构") == [{"id": "C1", "text": "分析项目结构"}]
    assert P.split_goal_clauses("x") == [{"id": "C1", "text": "x"}]


def test_subtask_without_clause_and_uncovered_clause_rejected():
    with pytest.raises(P.PlanError, match="没有引用任何目标条款"):
        P.validate_subtasks([sub("1", serves=())], 8, clause_ids=["C1"])
    with pytest.raises(P.PlanError, match="不存在的目标条款"):
        P.validate_subtasks([sub("1", serves=("C9",))], 8, clause_ids=["C1"])
    with pytest.raises(P.PlanError, match="C2 没有被任何子任务覆盖"):
        P.validate_subtasks([sub("1", serves=("C1",))], 8, clause_ids=["C1", "C2"])
    # Done subtasks kept across a replan count as coverage.
    done = [{"id": "0", "depends_on": [], "serves_clauses": ["C2"], "write_scope": []}]
    P.validate_subtasks([sub("1", serves=("C1",))], 8, clause_ids=["C1", "C2"],
                        known_nodes=done)


def test_uncovered_clause_retries_planner(planner_on, monkeypatch):
    partial = plan_json(sub("1", serves=("C1",)))
    full = plan_json(sub("1", serves=("C1",)), sub("2", deps=["1"], serves=("C2",)))
    llm = PlannerLLM([partial, full])
    rec = Recorder()
    res = run(TWO_CLAUSES, llm, rec, FakeExecutor({}), monkeypatch, planner_on)
    assert len(llm.calls) == 2 and "C2" in llm.calls[1][-1]["content"]
    assert "C1：分析配置解析" in llm.calls[0][-1]["content"]  # clauses shown to planner
    plan = rec.plan_store.load()
    assert [c["id"] for c in plan["goal_clauses"]] == ["C1", "C2"]
    assert res["status"] == "succeeded"
    report = {c["id"]: c for c in plan["final_verify"]["clauses"]}
    assert report["C1"]["status"] == "covered" and report["C1"]["covered_by"] == ["1"]
    assert report["C2"]["covered_by"] == ["2"]
    assert all(c["evidence"] for c in report.values())


def test_clause_report_marks_missing_when_serving_subtask_not_accepted():
    plan = P.new_plan_record(
        [sub("1", serves=("C1",)), sub("2", deps=["1"], serves=("C2",))], "llm",
        requires_approval=False, goal=TWO_CLAUSES, goal_clauses=P.split_goal_clauses(TWO_CLAUSES))
    plan["subtasks"][0]["status"] = P.SUBTASK_DONE
    # A done analyze subtask is only "covered" when it carries tool-call evidence.
    plan["subtasks"][0]["acceptance_evidence"] = [
        {"type": "executor_status", "status": "succeeded", "tool_calls": 1,
         "tool_successes": 1, "tool_failures": 0},
    ]
    plan["subtasks"][1]["status"] = P.SUBTASK_SKIPPED
    report = {c["id"]: c for c in P.clause_report(plan)}
    assert report["C1"]["status"] == "covered"
    # A skipped serving subtask declares the clause but cannot verify it → the
    # clause is "planned" (declared, not implemented), not "covered".
    assert report["C2"]["status"] == "planned" and report["C2"]["covered_by"] == []
    assert report["C2"]["planned_by"] == ["2"]


def test_goal_and_clauses_are_frozen(planner_on):
    make_run_row("run-frozen")
    plan = P.new_plan_record([sub("1")], "llm", requires_approval=False, goal="分析项目结构",
                             goal_clauses=P.split_goal_clauses("分析项目结构"))
    storage.save_plan("run-frozen", plan)
    changed = dict(plan, goal_clauses=[{"id": "C1", "text": "别的目标"}])
    with pytest.raises(storage.GoalImmutableError):
        storage.save_plan("run-frozen", changed)
    store = W.MemoryPlanStore()
    store.save(plan)
    with pytest.raises(ValueError):
        store.save(changed)
    assert storage.get_frozen_goal("run-frozen")["goal"] == "分析项目结构"


def test_replans_keep_the_same_clauses_and_record_versions(planner_on, monkeypatch):
    first = plan_json(sub("1", serves=("C1",)), sub("2", deps=["1"], serves=("C2",)))
    rewrite = plan_json(sub("2", title="换个做法", deps=["1"], serves=("C2",)))
    llm = PlannerLLM([first, rewrite])
    make_run_row("run-ver")
    store = W.StoragePlanStore("run-ver")
    res = run(TWO_CLAUSES, llm, Recorder(store), FakeExecutor({"2": [failed(), ok()]}),
              monkeypatch, planner_on, run_id="run-ver")
    assert res["status"] == "succeeded", (res["error"], llm.calls[-1][-1]["content"][-600:])
    plan = store.load()
    assert plan["goal_clauses"] == P.split_goal_clauses(TWO_CLAUSES)
    assert (plan["version"], plan["revision"]) == (1, 2)
    # The replanner was told the frozen clauses.
    assert "C2：总结测试覆盖" in llm.calls[1][-1]["content"]
    versions = storage.list_plan_versions("run-ver")
    assert [(v["version"], v["revision"]) for v in versions] == [(1, 1), (1, 2)]
    assert versions[1]["failure_class"] == "rewrite_remaining"
    assert (versions[1]["parent_version"], versions[1]["parent_revision"]) == (1, 1)
    full = storage.get_plan_version("run-ver", 1, 2)
    assert full["plan"]["goal_clauses"] == plan["goal_clauses"]
    assert "2" in full["diff"]["changed"]
    node = next(s for s in full["plan"]["subtasks"] if s["id"] == "2")
    assert node["derived_from"] == ["2"]
    # Append-only: re-inserting an existing (version, revision) never overwrites.
    assert storage.append_plan_version("run-ver", {**full, "reason": "覆盖?"}) is False
    assert storage.get_plan_version("run-ver", 1, 2)["reason"] == full["reason"]


def test_structural_replan_creates_new_version_with_parent(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    first = plan_json(sub("1"), sub("2", deps=["1"]))
    structural = plan_json(sub("2", deps=["1"]), sub("2b", title="先安装依赖", deps=["1"]))
    llm = PlannerLLM([first, structural])
    make_run_row("run-sv")
    store = W.StoragePlanStore("run-sv")
    rec = Recorder(store)
    ex = FakeExecutor({"2": [failed()]})
    run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-sv")
    _approve(store)
    run("分析项目结构", llm, rec, ex, monkeypatch, planner_on, run_id="run-sv")
    versions = storage.list_plan_versions("run-sv")
    assert [(v["version"], v["revision"]) for v in versions] == [(1, 1), (2, 1)]
    v2 = storage.get_plan_version("run-sv", 2)
    assert v2["parent_version"] == 1 and v2["diff"]["added"] == ["2b"]
    assert store.load()["goal_clauses"] == P.split_goal_clauses("分析项目结构")


def test_rejection_records_new_version_keeping_clauses(planner_on, monkeypatch):
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", True)
    llm = PlannerLLM([plan_json(sub("1")), plan_json(sub("1"), sub("2", deps=["1"]))])
    make_run_row("run-rv")
    store = W.StoragePlanStore("run-rv")
    run("分析项目结构", llm, Recorder(store), FakeExecutor({}), monkeypatch, planner_on,
        run_id="run-rv")
    plan = store.load()
    plan["approval_status"] = P.PLAN_REJECTED
    plan["plan_rejections"] = 1
    plan["feedback"] = ["拆细一点"]
    store.save(plan)
    run("分析项目结构", llm, Recorder(store), FakeExecutor({}), monkeypatch, planner_on,
        run_id="run-rv")
    versions = storage.list_plan_versions("run-rv")
    assert [(v["version"], v["failure_class"]) for v in versions] == [(1, None), (2, "rejected")]
    assert "拆细一点" in versions[1]["reason"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_versions_endpoints_and_edits_record_history(api, monkeypatch):
    client, holder = api
    holder["llm"] = PlannerLLM([plan_json(sub("1"), sub("2", deps=["1"]))])
    monkeypatch.setattr(agent_executor, "run_agent", FakeExecutor({}))
    run_id = client.post("/runs", json={"goal": "分析项目结构"}).json()["run_id"]
    assert _wait(client, run_id, PAUSED)["status"] == "waiting_for_plan_approval"
    view = client.get(f"/runs/{run_id}/plan").json()
    assert view["plan"]["goal_clauses"] == [{"id": "C1", "text": "分析项目结构"}]
    assert "leases" in view and view["max_parallel"] == config.MAX_PARALLEL
    # Cosmetic edit → same version, new revision, recorded.
    r = client.patch(f"/runs/{run_id}/subtasks/2", json={"title": "改个标题"})
    assert r.status_code == 200 and r.json()["plan_version"] == 1
    assert r.json()["plan_revision"] == 2
    # Structural edit of the scope → new version.
    r = client.patch(f"/runs/{run_id}/subtasks/2",
                     json={"kind": "modify", "write_scope": ["docs\\notes.md"]})
    assert r.status_code == 200, r.text
    assert r.json()["plan_version"] == 2
    assert r.json()["plan"]["subtasks"][1]["write_scope"] == ["docs/notes.md"]
    # Invalid edits are rejected by the same static checks.
    bad = client.patch(f"/runs/{run_id}/subtasks/2", json={"serves_clauses": ["C7"]})
    assert bad.status_code == 400 and "C7" in bad.json()["detail"]
    bad = client.patch(f"/runs/{run_id}/subtasks/2", json={"write_scope": []})
    assert bad.status_code == 400 and "write_scope" in bad.json()["detail"]

    listing = client.get(f"/runs/{run_id}/plan/versions").json()
    keys = [(v["version"], v["revision"]) for v in listing["versions"]]
    assert keys == [(1, 1), (1, 2), (2, 1)]
    assert listing["goal_clauses"][0]["id"] == "C1"
    v12 = client.get(f"/runs/{run_id}/plan/versions/1", params={"revision": 2}).json()
    assert v12["diff"]["changed"]["2"]["title"]["after"] == "改个标题"
    v2 = client.get(f"/runs/{run_id}/plan/versions/2").json()
    assert v2["parent_version"] == 1 and "write_scope" in v2["diff"]["changed"]["2"]
    assert client.get(f"/runs/{run_id}/plan/versions/9").status_code == 404
