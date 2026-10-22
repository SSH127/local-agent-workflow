"""Real-execution eval: planned (strong planner + cheap executor) vs direct (cheap model one-shot).

    python scripts/exec_eval.py --project novelflow|an [--goals N1,N2] [--cond planned,direct]

Works only on fresh copies under D:/PythonProject/_exec_eval/results/. Never runs
anything except pytest on the copies. Budget cap via --cap (USD, cumulative ledger).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

E = Path(r"D:\PythonProject\_exec_eval")
RESULTS = E / "results"
LEDGER = RESULTS / "_spend.json"
PROJECTS = {
    "novelflow": {"src": E / "NovelFlow", "py": E / "venv_nf" / "Scripts" / "python.exe",
                  "goals": ROOT / "evals" / "exec_goals_novelflow.jsonl", "checks": E / "checks" / "novelflow"},
    "an": {"src": E / "an", "py": E / "venv_an" / "Scripts" / "python.exe",
           "goals": ROOT / "evals" / "exec_goals_an.jsonl", "checks": E / "checks" / "an"},
}
PRICE_IN, PRICE_OUT = 0.28, 0.42  # USD / 1M tokens (DeepSeek list price used for both models)
EXEC_MODEL = "deepseek-chat"
PLAN_MODEL = os.getenv("EXEC_EVAL_PLANNER_MODEL", "deepseek-reasoner")
FORBIDDEN = re.compile(r"(^|/)\.env|(^|/)\.git/|pyautogui|\.\.")

EDIT_RULES = """只输出严格 JSON，不要任何解释或 markdown：
{"edits":[{"path":"仓库相对路径","search":"文件中逐字存在且唯一的原文片段（新建文件填空字符串）","replace":"替换后的文本"}],"test_command":"可选：验证用的 pytest 命令","notes":"一句话说明"}
规则：search 必须与当前文件内容逐字一致（含缩进）；尽量小范围修改；新建文件 search 为 "" 且 replace 是完整文件内容；不要修改与任务无关的文件。"""


def cost(u: dict) -> float:
    return (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6


def spent() -> float:
    try:
        return float(json.loads(LEDGER.read_text(encoding="utf-8"))["usd"])
    except Exception:  # noqa: BLE001
        return 0.0


def add_spend(usd: float) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps({"usd": round(spent() + usd, 6)}), encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "user.name=eval", "-c", "user.email=eval@local", "-c", "core.autocrlf=false",
                        *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout


PATHSPEC_EXCL = (":(exclude)**/__pycache__/**", ":(exclude)tests_eval/**", ":(exclude).pytest_cache/**")


def fresh_copy(src: Path, dst: Path) -> Path:
    if dst.exists():
        raise SystemExit(f"refusing to reuse existing dir {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".env", ".venv", "__pycache__", ".pytest_cache"))
    git(dst, "init", "-q")
    git(dst, "add", "-A", "-f", "--", ".", *PATHSPEC_EXCL)
    git(dst, "commit", "-q", "-m", "base")
    return dst


def run_pytest(py: Path, repo: Path, paths: list[str], timeout: int = 180) -> tuple[bool, str]:
    cmd = [str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths]
    try:
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8",
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                           errors="replace", timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def pytest_paths_from(cmd: str | None, repo: Path) -> list[str]:
    """Sanitize a model-proposed test command: only pytest/unittest, only existing test paths."""
    if not cmd or not re.search(r"pytest|unittest", cmd):
        return []
    try:
        toks = shlex.split(cmd.replace("\\", "/"))
    except ValueError:
        toks = cmd.split()
    out = []
    for t in toks:
        base = t.split("::")[0]
        if base.endswith(".py") or (base.startswith("tests") and (repo / base).exists()):
            if (repo / base).exists() and not FORBIDDEN.search(base):
                out.append(t)
    return out


def parse_json(text: str | None) -> dict:
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no JSON object in response")
    return json.loads(s[a:b + 1])


def apply_edits(repo: Path, edits: list[dict], allowed: set[str] | None) -> tuple[list[str], list[str]]:
    errs, touched = [], []
    for e in edits or []:
        p = str(e.get("path") or "").replace("\\", "/").lstrip("./")
        if not p or FORBIDDEN.search(p) or p.startswith("tests_eval"):
            errs.append(f"禁止的路径：{p}")
            continue
        if allowed is not None and p not in allowed:
            errs.append(f"路径 {p} 不在本子任务允许的 write_scope 内：{sorted(allowed)}")
            continue
        f = repo / p
        search, replace = e.get("search") or "", e.get("replace") or ""
        if not search:
            if f.exists() and f.read_text(encoding="utf-8"):
                errs.append(f"{p} 已存在，修改已有文件必须提供 search")
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(replace.encode("utf-8"))
            touched.append(p)
            continue
        if not f.exists():
            errs.append(f"文件不存在：{p}")
            continue
        raw = f.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        cur = raw.replace("\r\n", "\n")
        s2, r2 = search.replace("\r\n", "\n"), replace.replace("\r\n", "\n")
        n = cur.count(s2)
        if n != 1:
            errs.append(f"{p}: search 片段匹配 {n} 次（必须恰好 1 次）：{s2[:120]!r}")
            continue
        new = cur.replace(s2, r2, 1)
        f.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
        touched.append(p)
    return errs, touched


class Exec:
    """Cheap executor with token accounting and a JSONL log."""

    def __init__(self, log: Path):
        from app.agents.llm import DeepSeekClient
        self.llm = DeepSeekClient(model=EXEC_MODEL)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.log = log
        self.calls = 0

    def chat(self, messages: list[dict], tag: str) -> str:
        if spent() + cost(self.usage) > CAP:
            raise RuntimeError("budget cap reached")
        r = self.llm.chat(messages, [])
        self.calls += 1
        u = r.usage or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage[k] += int(u.get(k) or 0)
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "prompt": messages[-1]["content"][:6000],
                                 "response": (r.content or "")[:8000], "usage": u}, ensure_ascii=False) + "\n")
        return r.content or ""


def file_block(repo: Path, paths: list[str], cap: int = 12000) -> str:
    out = []
    for p in paths:
        f = repo / p
        if f.is_file() and not FORBIDDEN.search(p):
            out.append(f"===== {p} =====\n{f.read_text(encoding='utf-8', errors='replace')[:cap]}")
        else:
            out.append(f"===== {p} =====\n(文件不存在，可新建)")
    return "\n\n".join(out)


SYS = "你是一个谨慎的 Python 代码执行器。严格按给定任务修改代码，只输出 JSON。"


def edit_round(ex: Exec, repo: Path, prompt: str, allowed: set[str] | None, tag: str,
               metrics: dict) -> tuple[dict, list[str]]:
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt + "\n\n" + EDIT_RULES}]
    data: dict = {}
    touched: list[str] = []
    for attempt in range(2):
        content = ex.chat(msgs, f"{tag}#{attempt}")
        try:
            data = parse_json(content)
            errs, t = apply_edits(repo, data.get("edits") or [], allowed)
            touched += t
        except Exception as exc:  # noqa: BLE001
            errs = [f"JSON 解析失败：{exc}"]
        if not errs:
            return data, touched
        metrics["retries"] += 1
        msgs += [{"role": "assistant", "content": content[:6000]},
                 {"role": "user", "content": "以下编辑未能应用（其余已应用的保持不变）：\n- " + "\n- ".join(errs)
                  + "\n请基于文件当前内容只重发失败的编辑。\n\n当前文件：\n"
                  + file_block(repo, sorted(allowed) if allowed else touched or [])[:20000] + "\n\n" + EDIT_RULES}]
    metrics["apply_failures"] += 1
    return data, touched


def topo(subtasks: list[dict]) -> list[dict]:
    done, order, pending = set(), [], list(subtasks)
    while pending:
        progressed = False
        for s in list(pending):
            if all(d in done for d in s.get("depends_on") or []):
                order.append(s)
                done.add(s["id"])
                pending.remove(s)
                progressed = True
        if not progressed:
            order += pending
            break
    return order


def acceptance_tests(spec: dict, packet: dict, repo: Path) -> list[str]:
    cands = [packet.get("test_command") or ""]
    for a in spec.get("acceptance_criteria") or []:
        cands.append(a.get("test_target", "") if isinstance(a, dict) else str(a))
    paths: list[str] = []
    for c in cands:
        for p in pytest_paths_from(c if "pytest" in c else f"pytest {c}", repo):
            if p not in paths:
                paths.append(p)
    return paths


def run_planned(goal: dict, proj: dict, repo: Path, out: Path, metrics: dict) -> None:
    plan_path = out / "plan.json"
    env = dict(os.environ, PLANNER_MODEL=PLAN_MODEL, LLM_TIMEOUT_SECONDS="400", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "try_plan.py"), "--workspace", str(repo),
                        "--goal", goal["goal"], "--out", str(plan_path)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (out / "plan_stdout.txt").write_text(r.stdout[-30000:] + "\n--- stderr ---\n" + r.stderr[-6000:], encoding="utf-8")
    metrics["plan_seconds"] = round(time.time() - t0, 1)
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {"blocked": True}
    metrics["planner_usage"] = plan.get("usage") or {}
    metrics["planner_model"] = PLAN_MODEL
    metrics["plan_blocked"] = bool(plan.get("blocked"))
    add_spend(cost(metrics["planner_usage"]))
    if plan.get("blocked") or not plan.get("subtasks"):
        metrics["note"] = "plan blocked; nothing executed"
        return
    packets = {str(p.get("id")): p for p in plan.get("handoff_packets") or []}
    intended = set()
    for s in plan["subtasks"]:
        intended.update(s.get("write_scope") or [])
    metrics["plan_intended_files"] = sorted(intended)
    metrics["n_subtasks"] = len(plan["subtasks"])
    ex = Exec(out / "exec_log.jsonl")
    modify_scopes: set[str] = set()
    for s in topo(plan["subtasks"]):
        pk = packets.get(str(s["id"]), {})
        kind = s.get("kind")
        step = {"id": s["id"], "kind": kind, "title": s.get("title")}
        if kind == "modify":
            scope = {p.replace("\\", "/") for p in s.get("write_scope") or []}
            modify_scopes |= scope
            prompt = ("你只负责下面这一个子任务（来自规划器的交接包），不要做交接包以外的事。\n"
                      f"子任务：{s.get('title')}\n说明：{s.get('description')}\n"
                      f"允许修改的文件（write_scope）：{sorted(scope)}\n"
                      f"验收标准：{json.dumps(s.get('acceptance_criteria'), ensure_ascii=False)}\n"
                      f"交接包：{json.dumps(pk, ensure_ascii=False)[:8000]}\n\n"
                      f"write_scope 文件当前内容：\n{file_block(repo, sorted(scope))}")
            _d, touched = edit_round(ex, repo, prompt, scope, f"modify-{s['id']}", metrics)
            step["touched"] = touched
        elif kind == "verify":
            paths = acceptance_tests(s, pk, repo)
            step["tests"] = paths
            if paths:
                ok, outp = run_pytest(proj["py"], repo, paths)
                step["pass_first"] = ok
                if not ok and modify_scopes:
                    metrics["retries"] += 1
                    prompt = (f"验证子任务失败：{s.get('title')}\n测试输出（末尾）：\n{outp[-3000:]}\n\n"
                              f"只允许修改：{sorted(modify_scopes)}\n当前文件：\n{file_block(repo, sorted(modify_scopes))}\n"
                              "请修复使测试通过（不要删除或弱化测试断言）。")
                    edit_round(ex, repo, prompt, modify_scopes, f"repair-{s['id']}", metrics)
                    ok, outp = run_pytest(proj["py"], repo, paths)
                step["pass"] = ok
                step["output_tail"] = outp[-800:]
        else:
            step["skipped"] = "analysis step: no execution"
        metrics["steps"].append(step)
    metrics["executor_usage"] = ex.usage
    metrics["executor_calls"] = ex.calls
    add_spend(cost(ex.usage))


def run_direct(goal: dict, proj: dict, repo: Path, out: Path, metrics: dict) -> None:
    from app.engine import planning as P
    overview = P.build_project_overview(repo) or ""
    pack = P.build_grounding(goal["goal"], repo)
    gtext = (pack or {}).get("text") if isinstance(pack, dict) else (pack or "")
    ex = Exec(out / "exec_log.jsonl")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content":
            f"任务：{goal['goal']}\n\n仓库概览：\n{overview}\n\n相关代码线索：\n{(gtext or '')[:12000]}\n\n"
            "先告诉我你需要阅读哪些文件（最多 6 个，仓库相对路径），只输出 JSON：{\"read\":[...]}"}]
    try:
        want = [str(p).replace("\\", "/") for p in parse_json(ex.chat(msgs, "direct-read")).get("read", [])][:6]
    except Exception:  # noqa: BLE001
        want = []
        metrics["retries"] += 1
    metrics["direct_read"] = want
    prompt = (f"任务：{goal['goal']}\n\n仓库概览：\n{overview}\n\n文件内容：\n{file_block(repo, want)}\n\n"
              "一次性完成该任务所需的全部代码修改（包括需要的单元测试），并给出 test_command。")
    data, touched = edit_round(ex, repo, prompt, None, "direct-edit", metrics)
    paths = pytest_paths_from(data.get("test_command"), repo)
    step = {"id": "direct", "touched": touched, "tests": paths}
    if paths:
        ok, outp = run_pytest(proj["py"], repo, paths)
        step["pass_first"] = ok
        if not ok:
            metrics["retries"] += 1
            files = sorted(set(want) | set(touched))
            edit_round(ex, repo, f"测试失败：\n{outp[-3000:]}\n\n当前文件：\n{file_block(repo, files)}\n请修复（不要弱化断言）。",
                       None, "direct-repair", metrics)
            ok, outp = run_pytest(proj["py"], repo, paths)
        step["pass"] = ok
        step["output_tail"] = outp[-800:]
    metrics["steps"].append(step)
    metrics["executor_usage"] = ex.usage
    metrics["executor_calls"] = ex.calls
    add_spend(cost(ex.usage))


def finish(goal: dict, proj: dict, repo: Path, out: Path, metrics: dict) -> None:
    git(repo, "add", "-A", "-f", "--", ".", *PATHSPEC_EXCL)
    diff = git(repo, "diff", "--cached", "HEAD")
    (out / "diff.patch").write_text(diff, encoding="utf-8")
    num = git(repo, "diff", "--cached", "--numstat", "HEAD")
    files, added, removed = [], 0, 0
    for line in num.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files.append(parts[2])
            added += int(parts[0]) if parts[0].isdigit() else 0
            removed += int(parts[1]) if parts[1].isdigit() else 0
    metrics.update(files_changed=files, diff_added=added, diff_removed=removed)
    exp = set(goal.get("expected_files") or [])
    src_changed = {f for f in files if not re.search(r"(^|/)tests?/|test_", f)}
    metrics["expected_files"] = sorted(exp)
    metrics["expected_hit"] = sorted(exp & set(files))
    metrics["unexpected_src_files"] = sorted(src_changed - exp)
    # model-written / modified tests
    mtests = [f for f in files if re.search(r"(^|/)test_[^/]*\.py$", f) and (repo / f).exists()]
    if mtests:
        ok, outp = run_pytest(proj["py"], repo, mtests)
        metrics["model_tests"] = {"files": mtests, "pass": ok, "tail": outp[-600:]}
    else:
        metrics["model_tests"] = {"files": [], "pass": None}
    # hidden objective check (copied in after the diff is captured)
    te = repo / "tests_eval"
    te.mkdir(exist_ok=True)
    for f in proj["checks"].glob(f"test_eval_{goal['id']}.py"):
        shutil.copy2(f, te / f.name)
    paths = goal["check"].split()[1:]
    ok, outp = run_pytest(proj["py"], repo, paths)
    (out / "check.txt").write_text(f"$ pytest {' '.join(paths)}\n{outp}", encoding="utf-8")
    m = re.search(r"(\d+) passed", outp)
    f2 = re.search(r"(\d+) failed", outp)
    metrics["check_pass"] = ok
    metrics["check_summary"] = {"passed": int(m.group(1)) if m else 0, "failed": int(f2.group(1)) if f2 else 0,
                                "error": "error" in outp.lower() and not ok}
    u = {k: metrics.get("planner_usage", {}).get(k, 0) + metrics.get("executor_usage", {}).get(k, 0)
         for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    metrics["total_usage"] = u
    metrics["cost_usd"] = round(cost(u), 6)


CAP = 2.0


def main() -> int:
    global CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, choices=sorted(PROJECTS))
    ap.add_argument("--goals", default="")
    ap.add_argument("--cond", default="planned,direct")
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--run", default="run2", help="results subfolder (copies are never reused)")
    args = ap.parse_args()
    CAP = args.cap
    proj = PROJECTS[args.project]
    goals = [json.loads(l) for l in proj["goals"].read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.goals:
        goals = [g for g in goals if g["id"] in args.goals.split(",")]
    for g in goals:
        for cond in args.cond.split(","):
            if spent() >= CAP:
                print("BUDGET CAP reached, stopping", spent())
                return 2
            out = RESULTS / args.run / args.project / g["id"] / cond
            out.mkdir(parents=True, exist_ok=True)
            repo = fresh_copy(proj["src"], out / "repo")
            metrics = {"goal_id": g["id"], "condition": cond, "task_type": g.get("task_type"),
                       "retries": 0, "apply_failures": 0, "steps": [], "executor_model": EXEC_MODEL}
            t0 = time.time()
            try:
                (run_planned if cond == "planned" else run_direct)(g, proj, repo, out, metrics)
            except Exception as exc:  # noqa: BLE001
                metrics["error"] = f"{type(exc).__name__}: {exc}"[:500]
            metrics["exec_seconds"] = round(time.time() - t0, 1)
            finish(g, proj, repo, out, metrics)
            metrics["wall_seconds"] = round(time.time() - t0, 1)
            (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{args.project} {g['id']} {cond}: check={metrics['check_pass']} files={metrics['files_changed']} "
                  f"+{metrics['diff_added']}/-{metrics['diff_removed']} retries={metrics['retries']} "
                  f"cost=${metrics['cost_usd']} t={metrics['wall_seconds']}s spent=${spent():.4f} "
                  f"{metrics.get('error', '')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())