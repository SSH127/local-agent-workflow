"""Score saved plan JSONs offline against evals/planning_goals.jsonl (no LLM).

Hard gates fail the whole plan (blocked=true); quality metrics only apply when
gates pass. Fatal errors are never averaged away.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

for _s in (sys.stdout,):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_NEW = re.compile(r"新建|新增|创建|新文件|create|new file|add (?:a )?new", re.I)
_FILE = re.compile(r"[\w./-]+\.(?:py|tsx?|jsx?|toml|json|md)")
_TEST_TARGET = re.compile(
    r"(?:pytest\b|python\s+-m\s+pytest|npm\s+test|cargo\s+test|"
    r"tests?/[\w./-]+\.py|\btest_[A-Za-z0-9_]+\b|"
    r"运行\s*[\w./-]+\.py|执行\s*pytest)",
    re.I,
)
_PRECOND = re.compile(r"给定|前提|当|若|if\b|when\b|given\b|precondition|输入", re.I)
_EXPECT = re.compile(
    r"应|返回|等于|通过|失败|抛出|包含|不?存在|expect|should|returns?|raises?|"
    r"==|!=|True|False|exit\s*code|状态码",
    re.I,
)


def _text(st: dict[str, Any]) -> str:
    parts = [str(st.get("title", "")), str(st.get("description", ""))]
    for c in st.get("acceptance_criteria") or []:
        if isinstance(c, dict):
            parts.append(" ".join(str(v) for v in c.values()))
        else:
            parts.append(str(c))
    for t in st.get("targets") or []:
        if isinstance(t, dict):
            parts.append(" ".join(str(v) for v in t.values()))
        else:
            parts.append(str(t))
    return " ".join(parts)


def _scope_paths(st: dict[str, Any]) -> list[str]:
    out = []
    for p in st.get("write_scope") or []:
        out.append(str(p).replace("\\", "/").lstrip("./"))
    for t in st.get("targets") or []:
        if isinstance(t, dict) and t.get("path"):
            out.append(str(t["path"]).replace("\\", "/").lstrip("./"))
    return out


def _ancestors(subs: list[dict[str, Any]], sid: str) -> set[str]:
    by = {s["id"]: s for s in subs}
    out: set[str] = set()
    stack = list(by.get(sid, {}).get("depends_on") or [])
    while stack:
        x = stack.pop()
        if x in out or x not in by:
            continue
        out.add(x)
        stack.extend(by[x].get("depends_on") or [])
    return out


def _has_cycle(subs: list[dict[str, Any]]) -> bool:
    by = {s["id"]: list(s.get("depends_on") or []) for s in subs}
    state: dict[str, int] = {}

    def dfs(n: str) -> bool:
        state[n] = 1
        for d in by.get(n, []):
            if d not in by:
                continue
            if state.get(d) == 1:
                return True
            if state.get(d) == 0 and dfs(d):
                return True
        state[n] = 2
        return False

    for n in by:
        state.setdefault(n, 0)
        if state[n] == 0 and dfs(n):
            return True
    return False


def _load_index(workspace: Path | None):
    if workspace is None:
        return None
    try:
        from app.planner.repo_index import build_index
        return build_index(workspace)
    except Exception:  # noqa: BLE001
        return None


def _symbol_in_file(index, path: str, symbol: str) -> bool:
    if index is None or not symbol:
        return False
    fi = index.files.get(path)
    if not fi:
        return False
    for s in fi.symbols:
        if s.name == symbol or s.qualname == symbol or s.qualname.endswith("." + symbol):
            return True
    return False


def run_gates(goal: dict[str, Any], plan: dict[str, Any],
              workspace: Path | None, index=None) -> list[str]:
    """Return list of failed gate names (empty = pass)."""
    failed: list[str] = []
    subs = plan.get("subtasks") or []
    if not subs:
        return ["empty_plan"]

    ids = {s.get("id") for s in subs}
    for s in subs:
        for d in s.get("depends_on") or []:
            if d not in ids:
                failed.append(f"depends_missing:{s.get('id')}->{d}")
    if _has_cycle(subs):
        failed.append("depends_cycle")

    gtype = goal.get("type") or _infer_type(goal.get("goal", ""))
    if gtype == "analysis" and any(s.get("kind") == "modify" for s in subs):
        failed.append("analysis_has_modify")

    for s in subs:
        if s.get("kind") != "modify":
            continue
        scopes = [str(p).replace("\\", "/").lstrip("./") for p in (s.get("write_scope") or [])]
        text = _text(s)
        allows_new = bool(_NEW.search(text))
        if not scopes:
            failed.append(f"modify_empty_scope:{s.get('id')}")
            continue
        for p in scopes:
            if any(ch in p for ch in "*?[") or p.endswith("/**") or "/**" in p or p.endswith("/"):
                failed.append(f"broad_glob:{s.get('id')}:{p}")
                continue
            if workspace is not None:
                exists = (workspace / p).is_file()
                if not exists and not (allows_new and (workspace / p).parent.exists()):
                    failed.append(f"scope_missing:{s.get('id')}:{p}")

        # Symbol evidence: target.symbol must exist in target.path when confirmed
        for t in s.get("targets") or []:
            if not isinstance(t, dict):
                continue
            path = str(t.get("path") or "").replace("\\", "/").lstrip("./")
            sym = str(t.get("symbol") or "").strip()
            status = str(t.get("status") or "confirmed").lower()
            if status in {"to-verify", "to_verify", "unconfirmed", "new"}:
                continue
            if path and sym and workspace is not None:
                if not _symbol_in_file(index, path, sym):
                    failed.append(f"symbol_missing:{s.get('id')}:{path}:{sym}")

    verifies = [s for s in subs if s.get("kind") == "verify"]
    modifies = {s["id"] for s in subs if s.get("kind") == "modify"}
    if modifies and verifies:
        for v in verifies:
            if not (_ancestors(subs, v["id"]) & modifies):
                failed.append(f"verify_no_modify_dep:{v.get('id')}")
    elif modifies and not verifies and gtype != "analysis":
        failed.append("modify_without_verify")

    # de-dup while preserving order
    seen: set[str] = set()
    out = []
    for f in failed:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _infer_type(goal: str) -> str:
    g = goal.lower()
    if any(k in goal for k in ("分析", "评估", "说明", "梳理")) or "analy" in g:
        return "analysis"
    if any(k in goal for k in ("修复", "bug", "fix", "缺陷")):
        return "bugfix"
    if any(k in goal for k in ("重构", "拆到", "迁移", "refactor")):
        return "refactor"
    return "feature"


def _acceptance_score(subs: list[dict[str, Any]]) -> float:
    items: list[Any] = []
    for s in subs:
        items.extend(s.get("acceptance_criteria") or [])
        if s.get("acceptance"):
            items.append(s["acceptance"])
    if not items:
        return 0.0
    ok = 0
    for c in items:
        if isinstance(c, dict):
            has_pre = bool(str(c.get("precondition") or "").strip())
            has_exp = bool(str(c.get("expected_behavior") or c.get("expected") or "").strip())
            has_tgt = bool(str(c.get("test_target") or c.get("test") or "").strip())
            if has_pre and has_exp and has_tgt:
                ok += 1
            elif has_exp and has_tgt:
                ok += 0.6
            continue
        text = str(c)
        score = 0.0
        if _PRECOND.search(text):
            score += 0.34
        if _EXPECT.search(text):
            score += 0.33
        if _TEST_TARGET.search(text):
            score += 0.33
        ok += score
    return ok / len(items)


def _scoped_files(subs: list[dict[str, Any]]) -> set[str]:
    files: set[str] = set()
    for s in subs:
        for p in _scope_paths(s):
            if not any(ch in p for ch in "*?["):
                files.add(p)
        # read-only mentions only count when kind is analyze and path appears as path token
        if s.get("kind") == "analyze":
            for m in _FILE.findall(_text(s)):
                if "/" in m:
                    files.add(m.replace("\\", "/").lstrip("./"))
    return files


def quality_metrics(goal: dict[str, Any], plan: dict[str, Any],
                    workspace: Path | None) -> dict[str, float]:
    subs = plan.get("subtasks") or []
    scoped = _scoped_files(subs)
    ef = [f.replace("\\", "/").lstrip("./") for f in (goal.get("expected_files") or [])]
    es = goal.get("expected_symbols") or []
    alltext = " ".join(_text(s) for s in subs)

    file_recall = sum(any(f == s or s.endswith("/" + f) or f.endswith(s) for s in scoped) for f in ef) / len(ef) if ef else 1.0
    # precision: among scoped concrete files, how many are expected (or tests/)
    if scoped and ef:
        good = sum(1 for s in scoped if any(s == f or s.endswith(f) or f.endswith(s) or s.startswith("tests/") for f in ef))
        file_precision = good / len(scoped)
    else:
        file_precision = 1.0 if not scoped else 0.0

    # symbols only count when mentioned alongside a scoped file (or in targets)
    symbol_hits = 0
    for sym in es:
        hit = False
        for s in subs:
            tlist = s.get("targets") or []
            if any(isinstance(t, dict) and t.get("symbol") == sym for t in tlist):
                hit = True
                break
            if sym in _text(s) and (_scope_paths(s) or s.get("kind") == "analyze"):
                hit = True
                break
        symbol_hits += 1 if hit else 0
    symbol_recall = symbol_hits / len(es) if es else 1.0

    n = len(subs)
    gtype = goal.get("type") or _infer_type(goal.get("goal", ""))
    lo, hi = (1, 3) if gtype == "analysis" else (2, 6)
    titles = [str(s.get("title", "")).strip() for s in subs]
    dup = 1.0 if len(titles) == len(set(titles)) else 0.5
    gran = (1.0 if lo <= n <= hi else max(0.0, 1 - 0.25 * min(abs(n - lo), abs(n - hi)))) * dup

    kinds = {s.get("kind") for s in subs}
    exp_k = set(goal.get("expected_kinds") or [])
    kind_ok = len(kinds & exp_k) / len(exp_k) if exp_k else 1.0
    if gtype == "analysis" and any(s.get("kind") == "modify" for s in subs):
        kind_ok = 0.0
    verifies = [s for s in subs if s.get("kind") == "verify"]
    modifies = {s["id"] for s in subs if s.get("kind") == "modify"}
    if verifies and modifies:
        dep_ok = sum(bool(_ancestors(subs, v["id"]) & modifies) for v in verifies) / len(verifies)
    elif not modifies:
        dep_ok = 1.0
    else:
        dep_ok = 0.0
    dependencies = (dep_ok + kind_ok) / 2

    # impact: require real test/caller mention only for modify goals
    if gtype == "analysis":
        impact = 1.0 if any(s.get("kind") == "analyze" for s in subs) else 0.0
    else:
        impact = 1.0 if re.search(r"pytest|test_[A-Za-z0-9_]+|tests?/|调用方|caller", alltext, re.I) else 0.0

    testable = _acceptance_score(subs)

    mods = [s for s in subs if s.get("kind") == "modify"]
    if mods:
        weak = 0.0
        for s in mods:
            scopes = [p for p in _scope_paths(s) if not any(ch in p for ch in "*?[")]
            has_file = bool(scopes)
            has_sym = bool(s.get("targets")) or bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", str(s.get("description", ""))))
            narrow = len(scopes) <= 4
            weak += 1.0 if (has_file and has_sym and narrow) else 0.0
        weak_exec = weak / len(mods)
    else:
        weak_exec = 1.0

    # scope creep: average write_scope size (penalize >3)
    if mods:
        avg_scope = sum(len(s.get("write_scope") or []) for s in mods) / len(mods)
        creep = max(0.0, 1.0 - max(0.0, avg_scope - 3) * 0.25)
    else:
        creep = 1.0

    dims = {
        "file_recall": round(file_recall, 3),
        "file_precision": round(file_precision, 3),
        "symbol_recall": round(symbol_recall, 3),
        "granularity": round(gran, 3),
        "dependencies": round(dependencies, 3),
        "impact": round(impact, 3),
        "testable": round(testable, 3),
        "weak_exec": round(weak_exec, 3),
        "scope_creep": round(creep, 3),
    }
    dims["overall"] = round(sum(dims.values()) / len(dims), 3)
    return dims


def score_plan(goal: dict[str, Any], plan: dict[str, Any],
               workspace: Path | None, index=None) -> dict[str, Any]:
    if index is None and workspace is not None:
        index = _load_index(workspace)
    gates = run_gates(goal, plan, workspace, index)
    if gates:
        return {
            "blocked": True,
            "failed_gates": gates,
            "overall": 0.0,
            "file_recall": 0.0, "file_precision": 0.0, "symbol_recall": 0.0,
            "granularity": 0.0, "dependencies": 0.0, "impact": 0.0,
            "testable": 0.0, "weak_exec": 0.0, "scope_creep": 0.0,
        }
    dims = quality_metrics(goal, plan, workspace)
    dims["blocked"] = False
    dims["failed_gates"] = []
    return dims


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


# ---------------------------------------------------------------------------
# Reporting-only extra columns (phase 4). NOT used by gates or overall.
# ---------------------------------------------------------------------------

def extra_columns(goal: dict[str, Any], plan: dict[str, Any], index=None) -> dict[str, Any]:
    subs = plan.get("subtasks") or []
    targets = [t for s in subs for t in (s.get("targets") or []) if isinstance(t, dict)]
    reg = {e.get("evidence_id"): e for e in (plan.get("evidence_registry") or []) if isinstance(e, dict)}
    real = [t for t in targets if str(t.get("status") or "").lower() != "new"]
    bound = 0
    wrong = 0
    for t in real:
        path = str(t.get("path") or "").replace("\\", "/").lstrip("./")
        sym = str(t.get("symbol") or "")
        e = reg.get(str(t.get("evidence_id") or ""))
        if e and e.get("path") == path and (not sym or e.get("symbol") == sym
                                             or str(e.get("symbol", "")).endswith("." + sym)):
            bound += 1
        if index is not None and (not sym or not _symbol_in_file(index, path, sym)):
            wrong += 1
    es = goal.get("expected_symbols") or []
    tsyms = {str(t.get("symbol") or "").split(".")[-1] for t in targets}
    ef = goal.get("expected_files") or []
    cov = {str(c.get("path") or "") for c in (plan.get("coverage") or []) if isinstance(c, dict)}
    tpaths = {str(t.get("path") or "") for t in targets} | cov
    return {
        "evidence_coverage": round(bound / len(real), 3) if real else 0.0,
        "wrong_target_rate": round(wrong / len(real), 3) if real else 0.0,
        "expected_symbol_as_target": round(sum(1 for x in es if x in tsyms) / len(es), 3) if es else 1.0,
        "expected_file_role": round(sum(1 for f in ef if f in tpaths) / len(ef), 3) if ef else 1.0,
        "n_targets": len(targets),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goals", default="evals/planning_goals.jsonl")
    ap.add_argument("--plans", default="evals/plans")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--gpt-review", default="", help="optional GPT review JSON for correlation")
    ap.add_argument("--gpt-key", default="", help="grounded|ungrounded key inside gpt review")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    goals = [json.loads(l) for l in Path(a.goals).read_text(encoding="utf-8").splitlines() if l.strip()]
    ws = Path(a.workspace).resolve()
    index = _load_index(ws)
    results = {}
    for g in goals:
        p = Path(a.plans) / f"{g['id']}.json"
        if not p.is_file():
            continue
        results[g["id"]] = score_plan(g, json.loads(p.read_text(encoding="utf-8")), ws, index)

    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        passed = sum(1 for r in results.values() if not r.get("blocked"))
        for gid, r in results.items():
            if r.get("blocked"):
                print(gid, "BLOCKED", ",".join(r.get("failed_gates") or [])[:120])
            else:
                print(gid, " ".join(f"{k}={v}" for k, v in r.items() if k not in {"blocked", "failed_gates"}))
        if results:
            mean = sum(r["overall"] for r in results.values()) / len(results)
            print(f"MEAN overall={mean:.3f} gate_pass={passed}/{len(results)}")
        else:
            print("no plans found in", a.plans)

    if results and not a.json:
        print("--- EXTRA (reporting only, not in overall/gates) ---")
        tot: dict[str, float] = {}
        for g in goals:
            p = Path(a.plans) / f"{g['id']}.json"
            if not p.is_file():
                continue
            ex = extra_columns(g, json.loads(p.read_text(encoding="utf-8")), index)
            print(g["id"], " ".join(f"{k}={v}" for k, v in ex.items()))
            for k, v in ex.items():
                tot[k] = tot.get(k, 0.0) + float(v)
        n = len(results)
        print("EXTRA_MEAN " + " ".join(f"{k}={v / n:.3f}" for k, v in tot.items() if k != "n_targets"))
    if a.gpt_review and Path(a.gpt_review).is_file() and results:
        review = json.loads(Path(a.gpt_review).read_text(encoding="utf-8"))
        key = a.gpt_key or ("grounded" if "plans_ungrounded" not in a.plans else "ungrounded")
        gpt = review.get(key) or {}
        xs, ys, ids = [], [], []
        for gid, r in results.items():
            if gid in gpt:
                # map blocked→0, else scale 0-1 overall to 1-5 for correlation
                auto = 1.0 + 4.0 * float(r["overall"]) if not r.get("blocked") else 1.0
                xs.append(auto)
                ys.append(float(gpt[gid]))
                ids.append(gid)
        corr = pearson(xs, ys)
        # agreement within 1.0 on 1-5 scale
        agree = sum(1 for x, y in zip(xs, ys) if abs(x - y) <= 1.0) / len(xs) if xs else 0
        print(f"GPT_CORR key={key} n={len(xs)} pearson={corr if corr is not None else 'n/a'} "
              f"agree±1={agree:.2f} pairs={list(zip(ids, [round(x,2) for x in xs], ys))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
