"""Explore-before-plan: bounded search/read/caller tools → evidence pack for the planner."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.planner.context import goal_terms, rank_files
from app.planner.repo_index import RepoIndex, Symbol, build_index

DEFAULT_BUDGET = 6
EVIDENCE_MAX_CHARS = 10000

EXPLORE_SYSTEM = """你是代码库探索助手。在规划之前，用工具找到与目标真正相关的文件与符号。
规则：
- 每轮只调用必要工具；优先 search_symbol / search_text，再 read_symbol，再 list_callers。
- 不要猜测不存在的符号；找不到就记录缺口。
- 证据足够（覆盖目标里的关键名词、调用方、相关测试）时输出 JSON：
  {"done": true, "summary": "一句话", "key_files": ["path"], "key_symbols": ["path:symbol"]}
- 未完成时不要输出 done；继续用 tool call。
- 工具预算有限，别重复搜索同一查询。"""

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_symbol",
            "description": "按符号名/标识符查找定义（类/函数/组件）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "符号名或片段"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_path",
            "description": "按路径片段查找源文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 15},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "在源文件中搜索文本/正则（小写匹配）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 15},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_symbol",
            "description": "读取某文件中某符号的签名与代码片段",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbol": {"type": "string", "description": "name 或 qualname；空则读文件头"},
                    "max_lines": {"type": "integer", "default": 20},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_callers",
            "description": "按名称查找调用方（近似）及该文件的 import 邻居",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "path": {"type": "string", "description": "可选，限制邻接"},
                    "limit": {"type": "integer", "default": 12},
                },
                "required": ["symbol"],
            },
        },
    },
]


@dataclass
class EvidenceItem:
    kind: str
    path: str = ""
    symbol: str = ""
    detail: str = ""
    source: str = ""  # seed|tool


@dataclass
class ExploreResult:
    text: str
    items: list[EvidenceItem] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    key_symbols: list[str] = field(default_factory=list)
    budget_used: int = 0
    done_reason: str = ""
    registry: list[dict[str, Any]] = field(default_factory=list)


class ExploreToolkit:
    def __init__(self, workspace: str | Path, index: RepoIndex | None = None):
        self.root = Path(workspace).resolve()
        self.index = index or build_index(self.root)

    def search_symbol(self, query: str, limit: int = 10) -> str:
        q = (query or "").strip()
        if not q:
            return "空查询"
        hits = []
        ql = q.lower()
        for rel, fi in self.index.files.items():
            for s in fi.symbols:
                if ql in s.name.lower() or ql in s.qualname.lower():
                    hits.append(f"{rel}:{s.qualname} L{s.line}-{s.end_line} [{s.kind}] {s.signature[:100]}")
        hits.sort()
        if not hits:
            return f"未找到符号匹配：{q}"
        return "\n".join(hits[:limit])

    def search_path(self, query: str, limit: int = 15) -> str:
        q = (query or "").strip().lower().replace("\\", "/")
        if not q:
            return "空查询"
        hits = [p for p in self.index.all_paths() if q in p.lower()]
        return "\n".join(hits[:limit]) if hits else f"未找到路径：{q}"

    def search_text(self, query: str, limit: int = 15) -> str:
        q = (query or "").strip()
        if len(q) < 2:
            return "查询太短"
        hits = []
        for rel in self.index.all_paths():
            try:
                text = (self.root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if q.lower() not in text.lower():
                continue
            # first matching line
            for i, line in enumerate(text.splitlines(), 1):
                if q.lower() in line.lower():
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    break
            if len(hits) >= limit:
                break
        return "\n".join(hits) if hits else f"未找到文本：{q}"

    def read_symbol(self, path: str, symbol: str = "", max_lines: int = 20) -> str:
        rel = str(path or "").replace("\\", "/").lstrip("./")
        fi = self.index.files.get(rel)
        if not fi:
            return f"文件不在索引中：{rel}"
        try:
            lines = (self.root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"读取失败：{exc}"
        if not symbol:
            head = "\n".join(f"{i}| {ln[:140]}" for i, ln in enumerate(lines[:max_lines], 1))
            return f"{rel} (head)\n{head}"
        sym: Symbol | None = None
        for s in fi.symbols:
            if s.name == symbol or s.qualname == symbol or s.qualname.endswith("." + symbol):
                sym = s
                break
        if sym is None:
            names = ", ".join(s.qualname for s in fi.symbols[:20])
            return f"{rel} 中无符号 {symbol}。可选：{names}"
        end = min(sym.end_line, sym.line + max_lines - 1)
        body = "\n".join(f"{i}| {lines[i-1][:140]}" for i in range(sym.line, end + 1) if 0 < i <= len(lines))
        return f"{rel}:{sym.qualname} L{sym.line}-{sym.end_line}\n{sym.signature}\n{body}"

    def list_callers(self, symbol: str, path: str = "", limit: int = 12) -> str:
        sym = (symbol or "").strip()
        if not sym:
            return "空符号"
        callers = self.index.callers_of(sym, limit=limit)
        lines = [f"callers of {sym}:"]
        if callers:
            for rel, s in callers:
                lines.append(f"  {rel}:{s.qualname}@L{s.line}")
        else:
            lines.append("  （无）")
        if path:
            rel = path.replace("\\", "/").lstrip("./")
            deps = sorted(self.index.import_graph().get(rel, ()))
            importers = sorted(src for src, ds in self.index.import_graph().items() if rel in ds)
            lines.append(f"imports from {rel}: {', '.join(deps[:10]) or '（无）'}")
            lines.append(f"imported by: {', '.join(importers[:10]) or '（无）'}")
        return "\n".join(lines)

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        args = args or {}
        try:
            if name == "search_symbol":
                return self.search_symbol(str(args.get("query", "")), int(args.get("limit") or 10))
            if name == "search_path":
                return self.search_path(str(args.get("query", "")), int(args.get("limit") or 15))
            if name == "search_text":
                return self.search_text(str(args.get("query", "")), int(args.get("limit") or 15))
            if name == "read_symbol":
                return self.read_symbol(str(args.get("path", "")), str(args.get("symbol") or ""),
                                        int(args.get("max_lines") or 20))
            if name == "list_callers":
                return self.list_callers(str(args.get("symbol", "")), str(args.get("path") or ""),
                                         int(args.get("limit") or 12))
            return f"未知工具：{name}"
        except Exception as exc:  # noqa: BLE001
            return f"工具错误：{exc}"


def seed_explore(goal: str, toolkit: ExploreToolkit, max_files: int = 6) -> list[EvidenceItem]:
    """Deterministic first pass from goal terms (no LLM)."""
    items: list[EvidenceItem] = []
    hits = rank_files(toolkit.index, goal, top_k=max_files)
    idents, words, paths = goal_terms(goal)
    for h in hits:
        fi = toolkit.index.files[h.path]
        top = sorted(h.symbols.items(), key=lambda kv: -kv[1])[:3]
        for q, _ in top:
            sym = next((s for s in fi.symbols if s.qualname == q), None)
            if not sym:
                continue
            snip = toolkit.read_symbol(h.path, sym.name, max_lines=12)
            items.append(EvidenceItem("read", h.path, sym.qualname, snip[:1200], "seed"))
            callers = toolkit.list_callers(sym.name, h.path, limit=8)
            items.append(EvidenceItem("callers", h.path, sym.name, callers[:800], "seed"))
        if not top:
            items.append(EvidenceItem("file", h.path, "", toolkit.read_symbol(h.path, "", 15)[:800], "seed"))
    # explicit path hints
    for p in paths:
        if p in toolkit.index.files:
            items.append(EvidenceItem("path", p, "", f"goal mentions {p}", "seed"))
    # identifier direct search
    for ident in list(idents)[:5]:
        out = toolkit.search_symbol(ident, limit=5)
        if "未找到" not in out:
            items.append(EvidenceItem("search", "", ident, out[:800], "seed"))
    # Goal-hinted follow-ups that keyword ranking often misses.
    extra_queries: list[str] = []
    if any(k in goal for k in (".env", "受保护", "保护文件", "protected")):
        extra_queries += ["is_protected_file", "PROTECTED", "protected"]
    if any(k in goal for k in ("概览", "overview", "规划")):
        extra_queries += ["build_project_overview", "build_grounding"]
    if any(k in goal for k in ("workflow", "工作流", "拆解", "合并")):
        extra_queries += ["_generate_specs", "build_grounded_planner_messages", "workflow"]
    for q in list(dict.fromkeys(extra_queries))[:6]:
        out = toolkit.search_symbol(q, limit=6)
        if "未找到" not in out:
            items.append(EvidenceItem("search", "", q, out[:900], "seed"))
            # Read the first hit so the planner sees a real signature.
            line = out.splitlines()[0]
            if ":" in line:
                path = line.split(":", 1)[0]
                sym = q
                items.append(EvidenceItem(
                    "read", path, sym, toolkit.read_symbol(path, sym, 16)[:1200], "seed"))
        outp = toolkit.search_path(q.split("_")[0] if "_" in q else q, limit=8)
        if "未找到" not in outp:
            items.append(EvidenceItem("path", "", q, outp[:600], "seed"))
    return items


def _enough(items: list[EvidenceItem], goal: str) -> bool:
    files = {i.path for i in items if i.path}
    syms = {i.symbol for i in items if i.symbol}
    idents, words, paths = goal_terms(goal)
    if len(files) >= 4 and len(syms) >= 3:
        return True
    covered = sum(1 for p in paths if p in files)
    if paths and covered == len(paths) and syms:
        return True
    if idents and sum(1 for i in idents if any(i in (s or "") for s in syms)) >= max(1, len(idents) // 2):
        if len(files) >= 2:
            return True
    return False


def format_evidence_pack(items: list[EvidenceItem], *, key_files: list[str] | None = None,
                         key_symbols: list[str] | None = None, max_chars: int = EVIDENCE_MAX_CHARS) -> str:
    parts = [
        "## 探索证据（explore-before-plan；仅含工具/种子检索到的内容，勿编造未出现的符号）",
        "规划时 write_scope/targets 必须引用下列 path/symbol；找不到的标 status=to-verify。",
    ]
    if key_files:
        parts.append("关键文件: " + ", ".join(key_files[:12]))
    if key_symbols:
        parts.append("关键符号: " + ", ".join(key_symbols[:16]))
    seen: set[str] = set()
    for i, it in enumerate(items, 1):
        key = f"{it.kind}:{it.path}:{it.symbol}:{it.detail[:80]}"
        if key in seen:
            continue
        seen.add(key)
        head = f"\n### E{i} [{it.source}/{it.kind}] {it.path}" + (f" :: {it.symbol}" if it.symbol else "")
        parts.append(head)
        parts.append(it.detail[:1500])
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（探索证据已截断）"
    return text


def _acc_usage(usage_acc: dict[str, int] | None, resp: Any) -> None:
    if usage_acc is None:
        return
    u = getattr(resp, "usage", None) or {}
    usage_acc["prompt_tokens"] = usage_acc.get("prompt_tokens", 0) + int(u.get("prompt_tokens") or 0)
    usage_acc["completion_tokens"] = usage_acc.get("completion_tokens", 0) + int(u.get("completion_tokens") or 0)
    usage_acc["total_tokens"] = usage_acc.get("total_tokens", 0) + int(
        u.get("total_tokens") or (int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0))
    )


def _parse_done(content: str | None) -> dict[str, Any] | None:
    if not content or "done" not in content:
        return None
    try:
        from app.engine.planning import parse_plan_text
        data = parse_plan_text(content)
        if isinstance(data, dict) and data.get("done") is True:
            return data
    except Exception:  # noqa: BLE001
        pass
    return None


def _explore_repo_raw(
    goal: str,
    workspace: str | Path,
    *,
    llm: Any | None = None,
    budget: int = DEFAULT_BUDGET,
    usage_acc: dict[str, int] | None = None,
    index: RepoIndex | None = None,
    mock_steps: list[dict[str, Any]] | None = None,
) -> ExploreResult:
    """Run seed explore + optional LLM tool loop. ``mock_steps`` drives unit tests."""
    toolkit = ExploreToolkit(workspace, index=index)
    items = seed_explore(goal, toolkit)
    steps: list[dict[str, Any]] = [{"op": "seed", "n_items": len(items)}]
    used = 0
    key_files: list[str] = []
    key_symbols: list[str] = []
    done_reason = "seed_only"

    if _enough(items, goal) and llm is None and mock_steps is None:
        text = format_evidence_pack(items, max_chars=EVIDENCE_MAX_CHARS)
        return ExploreResult(text, items, steps, sorted({i.path for i in items if i.path}),
                             sorted({f"{i.path}:{i.symbol}" for i in items if i.path and i.symbol}),
                             0, "seed_enough")

    # scripted tools for tests
    if mock_steps is not None:
        for step in mock_steps[:budget]:
            name = step.get("name") or step.get("tool")
            args = step.get("args") or {}
            out = toolkit.dispatch(str(name), args)
            items.append(EvidenceItem("tool", str(args.get("path") or ""), str(args.get("symbol") or args.get("query") or ""),
                                      f"{name}({json.dumps(args, ensure_ascii=False)})\n{out}"[:1500], "tool"))
            steps.append({"tool": name, "args": args, "n": len(out)})
            used += 1
        done_reason = "mock"
        text = format_evidence_pack(items)
        return ExploreResult(text, items, steps,
                             sorted({i.path for i in items if i.path}),
                             sorted({f"{i.path}:{i.symbol}" for i in items if i.path and i.symbol}),
                             used, done_reason)

    if llm is None:
        text = format_evidence_pack(items)
        return ExploreResult(text, items, steps,
                             sorted({i.path for i in items if i.path}),
                             sorted({f"{i.path}:{i.symbol}" for i in items if i.path and i.symbol}),
                             used, done_reason)

    seed_brief = format_evidence_pack(items[:8], max_chars=3500)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXPLORE_SYSTEM},
        {"role": "user", "content": f"用户目标：\n{goal}\n\n已有种子证据：\n{seed_brief}\n\n请继续探索缺口（相关测试、配置、调用方、workflow 接线）。预算 {budget} 次工具调用。"},
    ]
    while used < budget:
        try:
            resp = llm.chat(messages, TOOL_SCHEMAS)
        except Exception as exc:  # noqa: BLE001
            steps.append({"error": str(exc)[:200]})
            done_reason = f"llm_error:{exc}"[:80]
            break
        _acc_usage(usage_acc, resp)
        tool_calls = []
        if hasattr(resp, "parsed_tool_calls"):
            tool_calls = resp.parsed_tool_calls()
        content = getattr(resp, "content", None)
        done = _parse_done(content)
        if done and not tool_calls:
            key_files = [str(x) for x in (done.get("key_files") or [])][:12]
            key_symbols = [str(x) for x in (done.get("key_symbols") or [])][:16]
            steps.append({"done": True, "summary": done.get("summary")})
            done_reason = "llm_done"
            break
        if not tool_calls:
            # model replied without tools — treat as done if we have seed evidence
            steps.append({"content": str(content or "")[:300]})
            done_reason = "llm_no_tools"
            break
        # append assistant message with tool_calls for protocol continuity
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": getattr(resp, "tool_calls", None) or [],
        })
        for call_id, name, args in tool_calls:
            if used >= budget:
                break
            out = toolkit.dispatch(name, args if isinstance(args, dict) else {})
            used += 1
            path = str((args or {}).get("path") or "")
            sym = str((args or {}).get("symbol") or (args or {}).get("query") or "")
            items.append(EvidenceItem("tool", path, sym, f"{name}({json.dumps(args, ensure_ascii=False)})\n{out}"[:1500], "tool"))
            steps.append({"tool": name, "args": args, "chars": len(out)})
            messages.append({
                "role": "tool",
                "tool_call_id": call_id or f"call_{used}",
                "content": out[:4000],
            })
        if _enough(items, goal) and used >= min(3, budget):
            done_reason = "enough"
            break
    else:
        done_reason = "budget"

    text = format_evidence_pack(items, key_files=key_files, key_symbols=key_symbols)
    return ExploreResult(
        text, items, steps,
        key_files or sorted({i.path for i in items if i.path})[:12],
        key_symbols or sorted({f"{i.path}:{i.symbol}" for i in items if i.path and i.symbol})[:16],
        used, done_reason,
    )


def explore_repo(
    goal: str,
    workspace: str | Path,
    *,
    llm: Any | None = None,
    budget: int = DEFAULT_BUDGET,
    usage_acc: dict[str, int] | None = None,
    index: RepoIndex | None = None,
    mock_steps: list[dict[str, Any]] | None = None,
    clauses: list[dict[str, str]] | None = None,
) -> ExploreResult:
    """Explore, then emit a structured evidence registry; text is rendered from it."""
    from app.planner.evidence import build_registry, render_registry
    idx = index or build_index(Path(workspace).resolve())
    er = _explore_repo_raw(goal, workspace, llm=llm, budget=budget, usage_acc=usage_acc,
                           index=idx, mock_steps=mock_steps)
    er.registry = build_registry(er.items, idx, clauses)
    er.text = render_registry(er.registry, Path(workspace).resolve(), max_chars=EVIDENCE_MAX_CHARS)
    return er
