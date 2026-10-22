from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.planner.repo_index import build_index
from app.planner.context import build_grounding_pack, goal_terms, rank_files
from app.engine import planning as P

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "parser.py").write_text(
        '"""Parser module."""\nimport json\n\n\nclass Parser:\n    def parse_line(self, line: str) -> dict:\n'
        '        return json.loads(line)\n\n\ndef parse_file(path: str) -> list:\n'
        '    return [Parser().parse_line(l) for l in open(path)]\n', encoding="utf-8")
    (tmp_path / "pkg" / "cli.py").write_text(
        "from pkg.parser import parse_file\n\n\ndef main(argv=None):\n    return parse_file(argv[0])\n",
        encoding="utf-8")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "App.tsx").write_text(
        "import { api } from './api'\nexport function App() { return api() }\n"
        "export const useThing = (x: number) => x\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    for d in (".venv", "node_modules", "data"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "skip.py").write_text("def hidden(): pass\n", encoding="utf-8")
    return tmp_path


def test_index_python_and_js(repo: Path) -> None:
    idx = build_index(repo)
    assert set(idx.files) == {"pkg/__init__.py", "pkg/parser.py", "pkg/cli.py", "web/App.tsx"}
    p = idx.files["pkg/parser.py"]
    names = {s.qualname: s for s in p.symbols}
    assert names["Parser"].kind == "class" and names["Parser"].line == 5
    assert "def parse_line(self, line: str) -> dict" in names["Parser.parse_line"].signature
    assert "parse_line" in names["parse_file"].calls
    assert "json" in p.imports
    js = idx.files["web/App.tsx"]
    assert {s.name for s in js.symbols} >= {"App", "useThing"}
    assert any(s.kind == "component" for s in js.symbols)
    assert "./api" in js.imports
    assert idx.import_graph()["pkg/cli.py"] == {"pkg/parser.py"}
    assert [s.qualname for _, s in idx.callers_of("parse_file")] == ["main"]


def test_index_cache_reuses_unchanged(repo: Path) -> None:
    a = build_index(repo).files["pkg/cli.py"]
    b = build_index(repo).files["pkg/cli.py"]
    assert a is b


def test_goal_terms_chinese_mapping() -> None:
    idents, words, paths = goal_terms("修复规划器的 write_scope 校验，见 app/engine/planning.py")
    assert "write_scope" in idents
    assert {"plan", "validate", "scope"} <= words
    assert "app/engine/planning.py" in paths


def test_rank_and_pack(repo: Path) -> None:
    idx = build_index(repo)
    hits = rank_files(idx, "fix parse_file so the parser handles blank lines")
    assert hits[0].path == "pkg/parser.py"
    pack = build_grounding_pack("fix parse_file blank lines", repo, max_chars=4000, index=idx)
    assert "pkg/parser.py" in pack["text"]
    assert "pkg/cli.py" in pack["text"]          # one hop / impact
    assert "影响面" in pack["text"] and "main" in pack["text"]
    assert "SECRET" not in pack["text"]
    small = build_grounding_pack("fix parse_file", repo, max_chars=300, index=idx)
    assert len(small["text"]) <= 300 + 20


def test_planner_messages_include_grounding(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(P.config, "PLANNER_GROUNDING", True)
    msgs, pack = P.build_grounded_planner_messages("fix parse_file", repo, 5)
    assert pack and "pkg/parser.py" in msgs[-1]["content"]
    monkeypatch.setattr(P.config, "PLANNER_GROUNDING", False)
    msgs2, pack2 = P.build_grounded_planner_messages("fix parse_file", repo, 5)
    assert pack2 is None and "代码定位" not in msgs2[-1]["content"]


def test_check_plan_grounding(repo: Path) -> None:
    specs = [
        {"id": "1", "title": "改解析", "description": "修改 pkg/parser.py 的 parse_file", "kind": "modify",
         "write_scope": ["pkg/parser.py", "pkg/missing.py", "tests/test_*.py"]},
        {"id": "2", "title": "新建测试", "description": "新建 pkg/test_new.py", "kind": "modify",
         "write_scope": ["pkg/test_new.py"]},
        {"id": "3", "title": "看看", "description": "阅读 pkg/ghost.py", "kind": "analyze", "write_scope": []},
    ]
    w = P.check_plan_grounding(specs, repo)
    joined = "\n".join(w)
    assert "pkg/missing.py" in joined
    assert "tests/test_*.py" in joined
    assert "pkg/ghost.py" in joined
    assert "test_new.py" not in joined
    assert not any("pkg/parser.py」" in x for x in w)


def test_try_plan_dry_run(repo: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "try_plan.py"), "--workspace", str(repo),
         "--goal", "fix parse_file", "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr
    assert ("EXPLORE" in out.stdout or "GROUNDING" in out.stdout) and "PLANNER MESSAGES" in out.stdout
    assert "pkg/parser.py" in out.stdout and "[system]" in out.stdout


def test_eval_plans_scoring(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_plans
    goal = {"id": "X", "type": "bugfix", "expected_files": ["app/engine/planning.py"],
            "expected_symbols": ["check_write_scopes"], "expected_kinds": ["modify", "verify"]}
    plan = {"subtasks": [
        {"id": "1", "kind": "modify", "title": "fix", "description": "改 app/engine/planning.py 的 check_write_scopes",
         "write_scope": ["app/engine/planning.py"], "depends_on": [],
         "targets": [{"path": "app/engine/planning.py", "symbol": "check_write_scopes", "status": "confirmed"}],
         "acceptance_criteria": [{"precondition": "给定重叠 glob", "expected_behavior": "应报错", "test_target": "pytest tests/test_write_scope.py"}]},
        {"id": "2", "kind": "verify", "title": "test", "description": "run pytest tests/test_write_scope.py", "write_scope": [],
         "depends_on": ["1"],
         "acceptance_criteria": [{"precondition": "修复完成", "expected_behavior": "全部通过", "test_target": "pytest tests/test_write_scope.py -q"}]}]}
    r = eval_plans.score_plan(goal, plan, ROOT)
    assert not r["blocked"], r.get("failed_gates")
    assert r["file_recall"] == 1.0 and r["dependencies"] == 1.0
    assert r["overall"] > 0.7
    # broad glob must block under new scorer
    bad = {"subtasks": [{"id": "1", "kind": "modify", "title": "x", "description": "x",
                         "write_scope": ["app/**"], "depends_on": [], "acceptance_criteria": ["x"]}]}
    assert eval_plans.score_plan(goal, bad, ROOT)["blocked"]
    lines = (ROOT / "evals" / "planning_goals.jsonl").read_text(encoding="utf-8").splitlines()
    goals = [json.loads(l) for l in lines if l.strip()]
    assert 8 <= len(goals) <= 10


