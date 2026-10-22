from __future__ import annotations

from pathlib import Path

import pytest

from app.planner.explore import ExploreToolkit, explore_repo, format_evidence_pack, seed_explore
from app.planner import quality as Q
from app.engine import planning as P


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "workflow.py").write_text(
        "from app.planning_ctx import build_grounding\n\n"
        "def _generate_specs(state):\n"
        "    return build_grounding(state['goal'])\n\n"
        "def decompose_node(state):\n"
        "    return _generate_specs(state)\n",
        encoding="utf-8")
    (tmp_path / "app" / "planning_ctx.py").write_text(
        "def build_project_overview(ws):\n    return 'ov'\n\n"
        "def build_grounding(goal):\n    return 'g'\n\n"
        "def build_grounded_planner_messages(goal, ws, n):\n"
        "    return [{'role':'user','content': build_grounding(goal)}], None\n",
        encoding="utf-8")
    (tmp_path / "app" / "security.py").write_text(
        "PROTECTED = {'.env'}\n\ndef is_protected_file(path, root):\n"
        "    return path.name in PROTECTED\n",
        encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_sec.py").write_text(
        "from app.security import is_protected_file\n"
        "def test_env_protected(tmp_path):\n"
        "    assert is_protected_file(tmp_path/'.env', tmp_path)\n",
        encoding="utf-8")
    return tmp_path


def test_toolkit_search_read_callers(repo: Path) -> None:
    tk = ExploreToolkit(repo)
    assert "build_grounding" in tk.search_symbol("build_grounding")
    assert "app/workflow.py" in tk.search_path("workflow")
    assert "is_protected_file" in tk.search_text("is_protected_file")
    body = tk.read_symbol("app/planning_ctx.py", "build_grounding")
    assert "def build_grounding" in body
    callers = tk.list_callers("build_grounding", "app/planning_ctx.py")
    assert "workflow.py" in callers or "_generate_specs" in callers


def test_seed_and_mock_explore(repo: Path) -> None:
    tk = ExploreToolkit(repo)
    items = seed_explore("把 build_project_overview 和 build_grounding 合并，保持 workflow 行为", tk)
    assert any("planning_ctx" in i.path or "grounding" in i.symbol for i in items)
    er = explore_repo(
        "评估 is_protected_file 是否排除 .env；检查 build_project_overview",
        repo,
        mock_steps=[
            {"name": "search_symbol", "args": {"query": "is_protected_file"}},
            {"name": "read_symbol", "args": {"path": "app/security.py", "symbol": "is_protected_file"}},
            {"name": "search_symbol", "args": {"query": "build_project_overview"}},
            {"name": "list_callers", "args": {"symbol": "build_grounding", "path": "app/planning_ctx.py"}},
        ],
    )
    assert er.budget_used == 4
    assert "is_protected_file" in er.text
    assert "探索证据" in er.text
    assert format_evidence_pack(er.items[:2])


class _MockLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.i = 0

    def chat(self, messages, tools):
        from app.agents.llm import LLMResponse
        if self.i >= len(self.rounds):
            return LLMResponse(content='{"done": true, "summary": "ok", "key_files": [], "key_symbols": []}')
        round = self.rounds[self.i]
        self.i += 1
        return LLMResponse(
            content=round.get("content"),
            tool_calls=round.get("tool_calls") or [],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def test_llm_explore_loop_mock(repo: Path) -> None:
    llm = _MockLLM([
        {"tool_calls": [{
            "id": "1", "function": {
                "name": "search_symbol",
                "arguments": '{"query": "is_protected_file"}',
            }
        }]},
        {"content": '{"done": true, "summary": "found", "key_files": ["app/security.py"], '
                    '"key_symbols": ["app/security.py:is_protected_file"]}'},
    ])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    er = explore_repo("分析 is_protected_file 对 .env 的排除", repo, llm=llm, budget=4, usage_acc=usage)
    assert er.budget_used >= 1
    assert "is_protected_file" in er.text
    assert usage["total_tokens"] > 0
    assert er.done_reason in {"llm_done", "enough", "llm_no_tools", "budget"}


def test_verify_cannot_have_write_scope(repo: Path) -> None:
    specs = [{
        "id": "1", "title": "v", "description": "add test", "kind": "verify",
        "depends_on": [], "write_scope": ["tests/test_sec.py"], "targets": [],
        "acceptance_criteria": [{"precondition": "a", "expected_behavior": "b", "test_target": "pytest -q"}],
    }]
    with pytest.raises(P.PlanError, match="禁止带 write_scope"):
        Q.enforce_plan_quality(specs, "分析 is_protected_file", repo, strict=True)


def test_messages_include_explore_text(repo: Path) -> None:
    er = explore_repo("合并 build_grounding", repo, mock_steps=[
        {"name": "search_symbol", "args": {"query": "build_grounding"}},
    ])
    msgs, pack = P.build_grounded_planner_messages(
        "合并 build_grounding", repo, 4, phase2=True, explore_text=er.text)
    assert "探索证据" in msgs[-1]["content"]
    assert pack and "explore_text" in pack
