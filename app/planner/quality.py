"""Planner quality gates, task-type policy, critic, and weak-model handoff packets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.planner.repo_index import RepoIndex, build_index

_NEW = re.compile(r"新建|新增|创建|新文件|create|new file|add (?:a )?new", re.I)
_ANALYSIS = re.compile(r"分析|评估|说明|梳理|定位.*生效|analy[sz]e|explain|how does|where is", re.I)
_BUGFIX = re.compile(r"修复|缺陷|bug|fix|不正确|不准|过于宽松|累加", re.I)
_REFACTOR = re.compile(r"重构|拆到|迁移|合并为|refactor|extract|rename", re.I)

TASK_LIMITS = {
    "analysis": (1, 3),
    "bugfix": (2, 5),
    "feature": (2, 6),
    "refactor": (2, 5),
}

CRITIC_SYSTEM = """你是代码任务规划的严格评审员。只输出一个 JSON 对象：
{"verdict":"pass"|"revise","issues":["..."],"revised":null|{"subtasks":[...]}}
规则：
- modify 的 write_scope 必须是精确已存在文件路径（禁止 ** 与目录级 glob）；新建文件须在描述中标明。
- modify 应包含 targets:[{path,symbol,reason,evidence_ref,status}]；status=confirmed 时 symbol 必须真实存在。
- 验收标准必须结构化：每条含 precondition、expected_behavior、test_target，并尽量含 edge_case。
- analysis 目标禁止 modify；步骤数遵守任务类型上限；不要无需求。
- verify 必须 depends_on 相关 modify。
- 每个 target 必须带 evidence_id（证据登记表中存在、path/symbol 一致）；不得因名字相似选符号。
- modify 子任务 write_scope 与 targets 路径必须一一对应；顶层 coverage 必须给每个证据文件指定角色。
- revised 必须同时给出 coverage。
若 verdict=revise，必须给出完整 revised.subtasks（不要只给补丁）。"""


def classify_task_type(goal: str) -> str:
    if _ANALYSIS.search(goal) and not _BUGFIX.search(goal) and not _REFACTOR.search(goal):
        # pure analysis / explain goals
        if not re.search(r"修复|新增|添加|实现|重构|迁移", goal):
            return "analysis"
    if _BUGFIX.search(goal):
        return "bugfix"
    if _REFACTOR.search(goal):
        return "refactor"
    return "feature"


def max_subtasks_for(goal: str, default: int = 8) -> int:
    lo, hi = TASK_LIMITS[classify_task_type(goal)]
    return min(default, hi)


def _text(spec: dict[str, Any]) -> str:
    parts = [str(spec.get("title", "")), str(spec.get("description", ""))]
    for c in spec.get("acceptance_criteria") or []:
        parts.append(json.dumps(c, ensure_ascii=False) if isinstance(c, dict) else str(c))
    return "\n".join(parts)


def normalize_acceptance(raw: Any) -> list[Any]:
    """Keep structured dicts; coerce plain strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "precondition": str(item.get("precondition") or item.get("given") or "").strip(),
                "expected_behavior": str(item.get("expected_behavior") or item.get("expected") or "").strip(),
                "test_target": str(item.get("test_target") or item.get("test") or "").strip(),
                "edge_case": str(item.get("edge_case") or item.get("regression") or "").strip(),
            })
        elif str(item).strip():
            out.append(str(item).strip()[:500])
    return out[:10]


def acceptance_as_strings(items: list[Any]) -> list[str]:
    """Flatten structured acceptance for SubtaskSpec (list[str]) storage."""
    out = []
    for c in items:
        if isinstance(c, dict):
            bits = []
            if c.get("precondition"):
                bits.append(f"前提:{c['precondition']}")
            if c.get("expected_behavior"):
                bits.append(f"期望:{c['expected_behavior']}")
            if c.get("test_target"):
                bits.append(f"测试:{c['test_target']}")
            if c.get("edge_case"):
                bits.append(f"边界:{c['edge_case']}")
            out.append(" | ".join(bits) if bits else json.dumps(c, ensure_ascii=False)[:500])
        else:
            out.append(str(c)[:500])
    return out


def enforce_plan_quality(
    specs: list[dict[str, Any]],
    goal: str,
    workspace: str | Path | None,
    *,
    index: RepoIndex | None = None,
    strict: bool = True,
) -> list[str]:
    """Validate plan quality. Raises ValueError (PlanError) when strict and gates fail.
    Returns warning list when not strict."""
    from app.engine.planning import PlanError
    from app.tools.security import resolve_workspace_root

    errors: list[str] = []
    task_type = classify_task_type(goal)
    lo, hi = TASK_LIMITS[task_type]
    if not (lo <= len(specs) <= hi):
        errors.append(f"任务类型 {task_type} 要求 {lo}-{hi} 个子任务，当前 {len(specs)} 个")
    if task_type == "analysis" and any(s.get("kind") == "modify" for s in specs):
        errors.append("分析类目标禁止包含 modify 子任务")

    root = None
    try:
        if workspace is not None:
            root = resolve_workspace_root(workspace)
            index = index or build_index(root)
    except Exception:  # noqa: BLE001
        root = Path(workspace) if workspace else None

    ids = {s["id"] for s in specs}
    for s in specs:
        for d in s.get("depends_on") or []:
            if d not in ids:
                errors.append(f"子任务 {s['id']} depends_on 未知 id {d}")
        if s.get("kind") in {"analyze", "verify"} and (s.get("write_scope") or []):
            errors.append(
                f"子任务 {s['id']} 是 {s.get('kind')}，禁止带 write_scope"
                f"（新增测试请用 kind=modify）"
            )
        # Placeholder symbol names: drop on analyze/verify; hard-fail on modify.
        cleaned = []
        for t in s.get("targets") or []:
            if not isinstance(t, dict):
                continue
            sym = str(t.get("symbol") or "").strip()
            if sym.lower() in {"to-verify", "to_verify", "unconfirmed", "unknown", "n/a", "none", ""}:
                if s.get("kind") == "modify":
                    errors.append(
                        f"子任务 {s['id']} target.symbol「{sym or '空'}」无效；"
                        "请填真实符号名，不确定时用 status=to-verify 且保留真实候选名"
                    )
                continue
            cleaned.append(t)
        s["targets"] = cleaned
        if s.get("kind") != "modify":
            continue
        scopes = [str(p).replace("\\", "/").lstrip("./") for p in (s.get("write_scope") or [])]
        allows_new = bool(_NEW.search(_text(s)))
        if not scopes:
            errors.append(f"子任务 {s['id']} 是 modify 但 write_scope 为空")
        for p in scopes:
            if any(ch in p for ch in "*?[") or p.endswith("/") or "/**" in p:
                errors.append(f"子任务 {s['id']} write_scope「{p}」禁止使用宽泛 glob/目录，必须写精确文件路径")
                continue
            if root is not None and not (root / p).is_file():
                if allows_new and (root / p).parent.exists():
                    continue
                errors.append(f"子任务 {s['id']} write_scope「{p}」不是已存在的文件")
        targets = s.get("targets") or []
        if not targets:
            errors.append(f"子任务 {s['id']} 缺少 targets（path/symbol/reason/evidence_ref）")
        for t in targets:
            if not isinstance(t, dict):
                continue
            path = str(t.get("path") or "").replace("\\", "/").lstrip("./")
            sym = str(t.get("symbol") or "").strip()
            status = str(t.get("status") or "confirmed").lower()
            if not path or not sym:
                errors.append(f"子任务 {s['id']} target 缺少 path 或 symbol")
                continue
            if status in {"to-verify", "to_verify", "unconfirmed", "new"}:
                continue
            if index is not None and not _symbol_defined(index, path, sym):
                errors.append(
                    f"子任务 {s['id']} 声称 confirmed 的符号 {sym} 在 {path} 中不存在（改为 status=to-verify 或修正）"
                )

    verifies = [s for s in specs if s.get("kind") == "verify"]
    modifies = {s["id"] for s in specs if s.get("kind") == "modify"}
    if modifies and not verifies and task_type != "analysis":
        errors.append("存在 modify 但没有 verify 子任务")
    for v in verifies:
        deps = set(v.get("depends_on") or [])
        # allow transitive via simple one-level check + any modify ancestor by walking
        if modifies and not (deps & modifies) and not _has_modify_ancestor(specs, v["id"], modifies):
            errors.append(f"verify 子任务 {v['id']} 未依赖任何 modify 子任务")

    for s in specs:
        crit = s.get("acceptance_criteria") or []
        if not crit:
            errors.append(f"子任务 {s['id']} 缺少 acceptance_criteria")
            continue
        for c in crit:
            if isinstance(c, dict):
                if not (c.get("expected_behavior") and c.get("test_target")):
                    errors.append(f"子任务 {s['id']} 结构化验收缺少 expected_behavior 或 test_target")
            else:
                text = str(c)
                if "测试:" not in text and "pytest" not in text.lower() and "test_" not in text:
                    # soft for analyze
                    if s.get("kind") != "analyze":
                        errors.append(f"子任务 {s['id']} 验收标准应包含可执行的 test_target（如 pytest 路径）")

    if strict and errors:
        raise PlanError("；".join(errors[:8]))
    return errors


def _symbol_defined(index: RepoIndex, path: str, symbol: str) -> bool:
    fi = index.files.get(path)
    if not fi:
        return False
    for s in fi.symbols:
        if s.name == symbol or s.qualname == symbol or s.qualname.endswith("." + symbol):
            return True
    return False


def _has_modify_ancestor(specs: list[dict[str, Any]], sid: str, modifies: set[str]) -> bool:
    by = {s["id"]: s for s in specs}
    seen: set[str] = set()
    stack = list(by.get(sid, {}).get("depends_on") or [])
    while stack:
        x = stack.pop()
        if x in seen or x not in by:
            continue
        if x in modifies:
            return True
        seen.add(x)
        stack.extend(by[x].get("depends_on") or [])
    return False


def build_handoff_packets(
    specs: list[dict[str, Any]],
    workspace: str | Path,
    index: RepoIndex | None = None,
) -> list[dict[str, Any]]:
    root = Path(workspace)
    idx = index or build_index(root)
    packets = []
    for s in specs:
        files = [str(p).replace("\\", "/").lstrip("./") for p in (s.get("write_scope") or [])]
        for t in s.get("targets") or []:
            if isinstance(t, dict) and t.get("path"):
                files.append(str(t["path"]).replace("\\", "/").lstrip("./"))
        files = list(dict.fromkeys(files))
        symbols = []
        snippets = []
        for t in s.get("targets") or []:
            if not isinstance(t, dict):
                continue
            path = str(t.get("path") or "").replace("\\", "/").lstrip("./")
            sym = str(t.get("symbol") or "")
            fi = idx.files.get(path)
            line = None
            if fi:
                for ss in fi.symbols:
                    if ss.name == sym or ss.qualname == sym:
                        line = ss.line
                        snip = _snip(root, path, ss.line, min(ss.end_line, ss.line + 12))
                        if snip:
                            snippets.append({"path": path, "symbol": ss.qualname, "lines": f"{ss.line}-{ss.end_line}", "text": snip})
                        break
            symbols.append({"path": path, "symbol": sym, "line": line, "reason": t.get("reason"), "status": t.get("status", "confirmed")})
        test_cmd = None
        for c in s.get("acceptance_criteria") or []:
            text = json.dumps(c, ensure_ascii=False) if isinstance(c, dict) else str(c)
            m = re.search(r"pytest[^\n|;]*|tests?/[\w./-]+\.py", text)
            if m:
                test_cmd = m.group(0).strip()
                break
            if isinstance(c, dict) and c.get("test_target"):
                test_cmd = str(c["test_target"])
                break
        packets.append({
            "id": s.get("id"),
            "title": s.get("title"),
            "kind": s.get("kind"),
            "files": files,
            "symbols": symbols,
            "snippets": snippets[:3],
            "test_command": test_cmd or ("pytest -q" if s.get("kind") == "verify" else None),
            "depends_on": list(s.get("depends_on") or []),
            "acceptance_criteria": s.get("acceptance_criteria") or [],
        })
    return packets


def _snip(root: Path, rel: str, start: int, end: int) -> str:
    try:
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(f"{i}| {lines[i-1][:140]}" for i in range(start, min(end, len(lines)) + 1) if i >= 1)


def critic_revise(
    llm: Any,
    goal: str,
    specs: list[dict[str, Any]],
    *,
    task_type: str,
    grounding: str | None = None,
    usage_acc: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """One critic round. Returns (revised_specs_or_None_if_pass, critic_meta)."""
    payload = {
        "goal": goal,
        "task_type": task_type,
        "limits": TASK_LIMITS[task_type],
        "candidate": {"subtasks": specs},
    }
    user = (
        f"任务类型：{task_type}（步骤上限 {TASK_LIMITS[task_type][1]}）\n"
        f"{'代码定位摘要：\\n' + grounding[:6000] + '\\n' if grounding else ''}"
        f"候选计划：\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n"
        "请评审；若需修改请输出完整 revised。"
    )
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user},
    ]
    meta: dict[str, Any] = {"verdict": "pass", "issues": []}
    try:
        resp = llm.chat(messages, [])
        if usage_acc is not None and getattr(resp, "usage", None):
            u = resp.usage or {}
            usage_acc["prompt_tokens"] = usage_acc.get("prompt_tokens", 0) + int(u.get("prompt_tokens") or 0)
            usage_acc["completion_tokens"] = usage_acc.get("completion_tokens", 0) + int(u.get("completion_tokens") or 0)
            usage_acc["total_tokens"] = usage_acc.get("total_tokens", 0) + int(u.get("total_tokens") or 0)
        content = getattr(resp, "content", None) or ""
        from app.engine.planning import parse_plan_text
        data = parse_plan_text(content)
        if not isinstance(data, dict):
            return None, meta
        meta["verdict"] = data.get("verdict", "pass")
        meta["issues"] = list(data.get("issues") or [])[:10]
        if meta["verdict"] == "revise" and isinstance(data.get("revised"), dict):
            meta["revised_coverage"] = data["revised"].get("coverage")
            return data["revised"].get("subtasks"), meta
        if meta["verdict"] == "revise" and isinstance(data.get("subtasks"), list):
            return data["subtasks"], meta
        return None, meta
    except Exception as exc:  # noqa: BLE001
        meta["verdict"] = "error"
        meta["issues"] = [str(exc)[:200]]
        return None, meta


PHASE2_SCHEMA_HINT = """输出格式（只输出一个 JSON 对象）：
{"task_type":"analysis|bugfix|feature|refactor",
 "subtasks":[
  {"id":"1","title":"简短标题",
   "description":"做什么；为何改这些符号",
   "depends_on":[],"kind":"analyze|modify|verify",
   "risk":"low|medium|high","reversible":true,
   "write_scope":["精确相对路径文件，禁止 ** 与目录 glob；analyze/verify 为空"],
   "targets":[{"path":"app/foo.py","symbol":"func_name","reason":"为何改它",
               "evidence_ref":"L12-L40 或 grounding 片段名","status":"confirmed|to-verify"}],
   "serves_clauses":["C1"],
   "acceptance_criteria":[
     {"precondition":"给定…","expected_behavior":"应…","test_target":"pytest tests/test_x.py::test_y","edge_case":"边界/回归…"}
   ]}
]}
硬性规则：
- write_scope 只能是精确已存在文件；新建文件须在 description 写明「新建」且父目录存在。
- analyze/verify 的 write_scope 必须为空；新增/修改测试文件用 kind=modify。
- modify 必须带 targets；symbol 只能是真实标识符，禁止把 to-verify 当作 symbol 名；
  不确定时 status=to-verify，禁止瞎编 confirmed。优先引用探索证据中的 path/symbol。
- analysis 任务类型禁止 modify；步骤数 analysis1-3、bugfix/refactor2-5、feature2-6。
- 每个非 analyze 子任务的验收必须含 precondition + expected_behavior + test_target。
- verify 必须 depends_on 至少一个 modify；不要输出最终整体验收子任务。
- 不要发明需求；只拆解用户目标已有的内容。"""

PHASE2_SYSTEM_PROMPT = """你是一个代码任务规划器，专长是「可落地、可被弱模型执行」的任务拆解。
先判定 task_type，再按类型控制步骤数量与是否允许修改。
优先引用代码定位（grounding）里的真实路径与符号；找不到就标 to-verify，不要编造。
每个 modify 子任务 = 精确文件 + 符号 + 理由 + 证据引用；随后安排可执行的 verify。
验收标准必须结构化，弱执行模型应能直接照着跑测试。
不要输出密钥或敏感信息。"""
