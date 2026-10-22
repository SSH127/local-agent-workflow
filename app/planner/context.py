"""Goal-relevant context retrieval → bounded "grounding pack" for the planner."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.planner.repo_index import FileIndex, RepoIndex, Symbol, build_index

# Chinese goal terms → English identifier fragments (extend freely).
ZH_TERMS: dict[str, list[str]] = {
    "规划": ["plan", "planner", "planning"], "计划": ["plan"], "拆解": ["decompose", "plan", "subtask"],
    "子任务": ["subtask"], "任务": ["task", "run"], "审批": ["approval", "approve"], "批准": ["approve"],
    "拒绝": ["reject"], "工作流": ["workflow"], "执行": ["execute", "executor", "runner"],
    "验证": ["verify", "validate"], "校验": ["validate", "check"], "验收": ["acceptance", "verify"],
    "测试": ["test"], "配置": ["config"], "接口": ["api", "route"], "路由": ["route", "router"],
    "存储": ["storage", "store"], "数据库": ["storage", "db", "sqlite"], "记忆": ["memory"],
    "回滚": ["rollback"], "取消": ["cancel"], "并行": ["parallel", "launchable"], "沙箱": ["sandbox", "isolation"],
    "写入": ["write", "write_scope"], "写入范围": ["write_scope"], "范围": ["scope"], "权限": ["permission", "auth"],
    "认证": ["auth"], "登录": ["login", "auth"], "安全": ["security", "protected"], "工具": ["tool", "tools"],
    "命令": ["command", "shell"], "超时": ["timeout"], "重试": ["retry"], "失败": ["failure", "fail"],
    "事件": ["event", "events"], "日志": ["log", "logging"], "前端": ["frontend", "ui"], "页面": ["page", "ui"],
    "模型": ["llm", "model"], "提示词": ["prompt"], "上下文": ["context"], "项目": ["project", "workspace"],
    "工作区": ["workspace"], "文件": ["file"], "编辑": ["edit"], "读取": ["read"], "搜索": ["search"],
    "用户": ["user"], "条款": ["clause"], "目标": ["goal"], "风险": ["risk"], "依赖": ["depends", "dependency"],
    "预算": ["budget", "limit"], "令牌": ["token"], "步数": ["steps"], "恢复": ["resume", "recovery"],
    "租约": ["lease"], "版本": ["version"], "历史": ["history"], "概览": ["overview"], "索引": ["index"],
}
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "it", "be", "add",
         "make", "fix", "so", "that", "this", "when", "should", "from", "into", "use", "new", "all"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATHLIKE = re.compile(r"[\w./-]+\.(?:py|tsx?|jsx?|md|json|toml)")


def _split_ident(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_")
    return [p for p in parts if len(p) > 1]


def goal_terms(goal: str) -> tuple[set[str], set[str], set[str]]:
    """Return (exact identifiers, lowercase word tokens, path hints)."""
    idents: set[str] = set()
    words: set[str] = set()
    for w in _WORD.findall(goal):
        if len(w) < 2 or w.lower() in _STOP:
            continue
        if "_" in w or any(c.isupper() for c in w[1:]):
            idents.add(w)
        words.add(w.lower())
        words.update(_split_ident(w))
    for zh, ens in sorted(ZH_TERMS.items(), key=lambda kv: -len(kv[0])):
        if zh in goal:
            words.update(ens)
    paths = set(m.strip("./") for m in _PATHLIKE.findall(goal))
    return idents, words - _STOP, paths


@dataclass
class Hit:
    path: str
    score: float = 0.0
    symbols: dict[str, float] = field(default_factory=dict)  # qualname → score
    reasons: list[str] = field(default_factory=list)


def rank_files(index: RepoIndex, goal: str, top_k: int = 8) -> list[Hit]:
    idents, words, paths = goal_terms(goal)
    hits: dict[str, Hit] = {}
    for rel, fi in index.files.items():
        h = Hit(rel)
        path_tokens = set(re.split(r"[/._-]+", rel.lower()))
        for p in paths:
            if rel.endswith(p) or p.endswith(rel):
                h.score += 10
                h.reasons.append(f"path:{p}")
        pt = words & path_tokens
        if pt:
            h.score += 2.0 * len(pt)
            h.reasons.append("path~" + ",".join(sorted(pt)))
        for s in fi.symbols:
            sc = 0.0
            if s.name in idents or s.qualname in idents:
                sc += 8
            toks = set(_split_ident(s.name))
            ov = toks & words
            if ov:
                sc += 1.5 * len(ov) + (1.0 if toks and ov == toks else 0)
            if s.doc and words & set(w.lower() for w in _WORD.findall(s.doc)):
                sc += 0.5
            if sc:
                h.symbols[s.qualname] = sc
        if h.symbols:
            top = sorted(h.symbols.values(), reverse=True)
            h.score += top[0] + 0.3 * sum(top[1:6])
        if rel.startswith("tests/") or "/tests/" in rel:
            h.score *= 0.4
        if h.score > 0:
            hits[rel] = h
    ranked = sorted(hits.values(), key=lambda x: (-x.score, x.path))
    return ranked[:top_k]


def expand_one_hop(index: RepoIndex, seeds: list[Hit], limit: int = 4) -> list[Hit]:
    graph = index.import_graph()
    seed_paths = {h.path for h in seeds}
    extra: dict[str, Hit] = {}
    for h in seeds[:4]:
        for dep in graph.get(h.path, ()):  # imported by seed
            if dep not in seed_paths:
                e = extra.setdefault(dep, Hit(dep))
                e.score += h.score * 0.25
                e.reasons.append(f"imported-by:{h.path}")
        for src, deps in graph.items():   # importers of seed
            if h.path in deps and src not in seed_paths:
                e = extra.setdefault(src, Hit(src))
                e.score += h.score * 0.2
                e.reasons.append(f"imports:{h.path}")
    ranked = sorted(extra.values(), key=lambda x: (-x.score, x.path))
    return ranked[:limit]


def _snippet(root: Path, rel: str, sym: Symbol, max_lines: int = 14) -> str:
    try:
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = sym.line - 1
    end = min(sym.end_line, sym.line + max_lines - 1)
    body = lines[start:end]
    out = "\n".join(f"{start + i + 1:>5}| {ln[:160]}" for i, ln in enumerate(body))
    if sym.end_line > end:
        out += f"\n     | …（共 {sym.end_line - sym.line + 1} 行）"
    return out


def build_grounding_pack(goal: str, workspace: str | Path, max_chars: int = 12000,
                         index: RepoIndex | None = None) -> dict[str, Any]:
    idx = index or build_index(workspace)
    seeds = rank_files(idx, goal)
    hop = expand_one_hop(idx, seeds) if seeds else []
    parts: list[str] = [
        "## 代码定位（自动检索，仅供参考；路径均真实存在）",
        f"已索引源文件 {len(idx.files)} 个。下面按相关度列出与目标最相关的文件、符号大纲与关键片段。",
        "规划时请：引用真实路径/符号；write_scope 只写下列或确有需要的新文件；为受影响调用方安排验证。",
    ]
    likely: list[tuple[str, Symbol]] = []
    for rank, h in enumerate(seeds, 1):
        fi: FileIndex = idx.files[h.path]
        parts.append(f"\n### [{rank}] {h.path}  (score {h.score:.1f}; {', '.join(h.reasons[:3]) or 'symbols'})")
        if fi.doc:
            parts.append(f"  doc: {fi.doc}")
        top_syms = sorted(h.symbols.items(), key=lambda kv: -kv[1])
        top_names = {q for q, _ in top_syms[:6]}
        outline = [s for s in fi.symbols if s.kind in {"class", "function", "component"} or s.qualname in top_names]
        for s in outline[:15]:
            mark = " ★" if s.qualname in top_names else ""
            parts.append(f"  L{s.line}-{s.end_line} {s.signature}{mark}")
        if len(outline) > 15:
            parts.append(f"  …（另有 {len(outline) - 15} 个符号）")
        if rank <= 4:
            for q, _ in top_syms[:2]:
                sym = next((s for s in fi.symbols if s.qualname == q), None)
                if sym:
                    likely.append((h.path, sym))
                    snip = _snippet(idx.root, h.path, sym)
                    if snip:
                        parts.append(f"  片段 {q}:\n{snip}")
    if hop:
        parts.append("\n### 一跳关联文件（import 关系）")
        for h in hop:
            fi = idx.files[h.path]
            names = ", ".join(s.name for s in fi.symbols if s.kind != "method")[:200]
            parts.append(f"  - {h.path}: {'; '.join(h.reasons[:2])} | {names}")
    impact_lines: list[str] = []
    for rel, sym in likely[:6]:
        callers = idx.callers_of(sym.name, limit=8)
        if callers:
            cs = ", ".join(f"{r}:{s.qualname}@L{s.line}" for r, s in callers)
            impact_lines.append(f"  - {rel}:{sym.qualname} ← {cs}")
    if impact_lines:
        # placed right after the header so the char budget never truncates it
        parts[3:3] = ["\n### 影响面（若修改这些符号，需检查/验证的调用方，按名称近似匹配）"] + impact_lines
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（已截断）"
    return {
        "text": text,
        "files": [h.path for h in seeds],
        "hop_files": [h.path for h in hop],
        "likely_symbols": [f"{r}:{s.qualname}" for r, s in likely],
        "indexed_files": len(idx.files),
    }
