"""Structured evidence registry (phase 4): explore results → citable entries.

Each entry: evidence_id, path, symbol, kind (definition|caller|test|config|text),
line_start/line_end (real, from the repo index), supports (clause ids), confidence.
The planner must cite evidence_id for every key target; quality checks verify it.
"""
from __future__ import annotations

import re
from typing import Any

from app.planner.repo_index import RepoIndex

MAX_ENTRIES = 60
MAX_COVERAGE_FILES = 12
ROLES = {"read", "modify", "verify", "not-needed"}
_DEF_LINE = re.compile(r"^\s*([\w./-]+\.(?:py|tsx?|jsx?)):([\w.]+) L(\d+)-(\d+)")
_CALLER_LINE = re.compile(r"^\s*([\w./-]+\.(?:py|tsx?|jsx?)):([\w.]+)@L(\d+)")
_TEXT_LINE = re.compile(r"^\s*([\w./-]+\.(?:py|tsx?|jsx?)):(\d+): (.*)$")
_PLACEHOLDER = {"to-verify", "to_verify", "unconfirmed", "unknown", "n/a", "none", ""}


def _find_symbol(index: RepoIndex, path: str, symbol: str):
    fi = index.files.get(path)
    if not fi or not symbol:
        return None
    for s in fi.symbols:
        if s.qualname == symbol:
            return s
    for s in fi.symbols:
        if s.name == symbol or s.qualname.endswith("." + symbol):
            return s
    return None


def _kind_for(path: str, symbol: str, base: str) -> str:
    if path.startswith("tests/") or "/tests/" in path or path.rsplit("/", 1)[-1].startswith("test_"):
        return "test"
    if path.endswith("config.py") or (symbol and symbol.isupper()):
        return "config" if base == "definition" else base
    return base


def _supports(symbol: str, path: str, clauses: list[dict[str, str]] | None) -> list[str]:
    if not clauses:
        return []
    toks = {t for t in re.split(r"[_./]+", f"{symbol} {path}".lower()) if len(t) > 2}
    out = []
    for c in clauses:
        text = str(c.get("text") or "").lower()
        if symbol and symbol.split(".")[-1].lower() in text:
            out.append(c["id"])
        elif any(t in text for t in toks):
            out.append(c["id"])
    return out or [c["id"] for c in clauses[:1]]


def build_registry(items: list[Any], index: RepoIndex,
                   clauses: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Turn explore items (seed + tool outputs) into deduped evidence entries."""
    raw: list[tuple[str, str, str, float, int | None, int | None]] = []
    for it in items:
        path = str(getattr(it, "path", "") or "")
        symbol = str(getattr(it, "symbol", "") or "")
        detail = str(getattr(it, "detail", "") or "")
        conf_base = 0.9 if getattr(it, "source", "") == "tool" else 0.85
        if path and symbol and getattr(it, "kind", "") in {"read", "tool"}:
            raw.append((path, symbol, "definition", conf_base, None, None))
        for line in detail.splitlines():
            m = _DEF_LINE.match(line)
            if m:
                raw.append((m.group(1), m.group(2), "definition", 0.8, int(m.group(3)), int(m.group(4))))
                continue
            m = _CALLER_LINE.match(line)
            if m:
                raw.append((m.group(1), m.group(2), "caller", 0.6, int(m.group(3)), None))
                continue
            m = _TEXT_LINE.match(line)
            if m:
                raw.append((m.group(1), "", "text", 0.4, int(m.group(2)), int(m.group(2))))
    entries: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path, symbol, base, conf, ls, le in raw:
        if path not in index.files:
            continue
        sym_obj = _find_symbol(index, path, symbol) if symbol else None
        if symbol and sym_obj is None:
            continue  # never register a symbol that is not really defined there
        qual = sym_obj.qualname if sym_obj else ""
        line_start = sym_obj.line if sym_obj else (ls or 1)
        line_end = sym_obj.end_line if sym_obj else (le or line_start)
        kind = _kind_for(path, qual, base)
        key = (path, qual, kind if kind != "text" else f"text:{line_start}")
        if key in seen:
            seen[key]["confidence"] = max(seen[key]["confidence"], conf)
            continue
        e = {
            "evidence_id": "",
            "path": path,
            "symbol": qual,
            "kind": kind,
            "line_start": int(line_start),
            "line_end": int(line_end),
            "supports": _supports(qual, path, clauses),
            "confidence": round(conf, 2),
        }
        seen[key] = e
        entries.append(e)
    order = {"definition": 0, "config": 1, "test": 2, "caller": 3, "text": 4}
    entries.sort(key=lambda e: (order.get(e["kind"], 9), -e["confidence"]))
    entries = entries[:MAX_ENTRIES]
    for i, e in enumerate(entries, 1):
        e["evidence_id"] = f"EV{i}"
    return entries


def augment_registry(registry: list[dict[str, Any]], index: RepoIndex, symbols: list[str],
                     clauses: list[dict[str, str]] | None = None) -> list[str]:
    """Extra explore round: look up definitions of missing symbols; returns new ids."""
    have = {(e["path"], e["symbol"]) for e in registry}
    new_ids: list[str] = []
    n = len(registry)
    for sym in symbols:
        short = sym.split(".")[-1]
        for rel, fi in index.files.items():
            for s in fi.symbols:
                if s.name != short and s.qualname != sym:
                    continue
                if (rel, s.qualname) in have:
                    continue
                n += 1
                eid = f"EV{n}"
                registry.append({
                    "evidence_id": eid, "path": rel, "symbol": s.qualname,
                    "kind": _kind_for(rel, s.qualname, "definition"),
                    "line_start": s.line, "line_end": s.end_line,
                    "supports": _supports(s.qualname, rel, clauses), "confidence": 0.75,
                })
                have.add((rel, s.qualname))
                new_ids.append(eid)
    return new_ids


def coverage_files(registry: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for e in registry:
        if e["kind"] in {"definition", "config"} and e["path"] not in out:
            out.append(e["path"])
        if len(out) >= MAX_COVERAGE_FILES:
            break
    return out


def render_registry(registry: list[dict[str, Any]], root=None, *, max_chars: int = 10000,
                    snippet_lines: int = 8) -> str:
    parts = [
        "## 探索证据 · 证据登记表（evidence registry；每个关键 target 必须引用其中一个 evidence_id）",
        "格式：EVn [kind] path:symbol Lstart-end conf supports",
    ]
    for e in registry:
        parts.append(
            f"{e['evidence_id']} [{e['kind']}] {e['path']}:{e['symbol'] or '-'} "
            f"L{e['line_start']}-{e['line_end']} conf={e['confidence']} supports={','.join(e['supports'])}")
    if root is not None:
        parts.append("\n### 定义片段")
        for e in registry:
            if e["kind"] not in {"definition", "config"} or not e["symbol"]:
                continue
            try:
                lines = (root / e["path"]).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            end = min(e["line_end"], e["line_start"] + snippet_lines - 1)
            body = "\n".join(f"{i}| {lines[i-1][:140]}" for i in range(e["line_start"], end + 1)
                             if 0 < i <= len(lines))
            parts.append(f"[{e['evidence_id']}] {e['path']}:{e['symbol']}\n{body}")
            if sum(len(p) for p in parts) > max_chars:
                break
    files = coverage_files(registry)
    if files:
        parts.append("\n### 需在 coverage 中逐一给出角色的文件")
        parts.extend(f"- {f}" for f in files)
    text = "\n".join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + "\n…（证据已截断）"


EVIDENCE_BINDING_HINT = """
证据绑定规则（必须遵守，否则计划会被拒绝）：
- 每个 target 必须带 evidence_id（来自证据登记表），且 path/symbol 必须与该证据条目一致。
- 不能因为名字出现在目标里就选某个符号；必须有对应 definition/config/test 证据。
- modify 的 target 不允许 status=to-verify；分析类子任务的 target 也必须有证据。
- 新建文件的 target 用 status=new，且该路径必须在同一子任务 write_scope 中、description 写明「新建」。
- 每个 modify 子任务：write_scope 中的每个文件都必须是某个 target 的 path（或 status=new 的新文件），
  每个 target 的 path 也必须在 write_scope 中。
- 顶层输出 "coverage":[{"path":"…","role":"read|modify|verify|not-needed","reason":"…","evidence_id":"EVn"}]，
  覆盖「需在 coverage 中逐一给出角色的文件」中的每个文件；not-needed 必须给出 evidence_id 与理由。
- 若关键符号没有证据，在 "missing_evidence":["符号名"] 中列出，系统会补充探索一次；不要编造。"""


def check_evidence_binding(
    specs: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    task_type: str,
    coverage: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    """Return (errors, missing_symbols)."""
    errors: list[str] = []
    missing: list[str] = []
    by_id = {e["evidence_id"]: e for e in registry}
    for s in specs:
        sid = s.get("id")
        kind = s.get("kind")
        targets = [t for t in (s.get("targets") or []) if isinstance(t, dict)]
        if kind == "analyze" and task_type == "analysis" and not targets:
            errors.append(f"子任务 {sid}（analysis）缺少带证据的关键 target")
        scopes = [str(p).replace("\\", "/").lstrip("./") for p in (s.get("write_scope") or [])]
        for t in targets:
            path = str(t.get("path") or "").replace("\\", "/").lstrip("./")
            sym = str(t.get("symbol") or "").strip()
            status = str(t.get("status") or "confirmed").lower()
            eid = str(t.get("evidence_id") or "").strip()
            if status == "new":
                if kind != "modify" or path not in scopes:
                    errors.append(f"子任务 {sid} 新建 target {path} 必须属于 modify 且在 write_scope 中")
                continue
            if kind == "modify" and status in {"to-verify", "to_verify", "unconfirmed"}:
                errors.append(f"子任务 {sid} modify target {sym or path} 不允许 to-verify")
                if sym and sym.lower() not in _PLACEHOLDER:
                    missing.append(sym)
                continue
            if not eid:
                errors.append(f"子任务 {sid} target {path}:{sym} 缺少 evidence_id")
                if sym and sym.lower() not in _PLACEHOLDER:
                    missing.append(sym)
                continue
            e = by_id.get(eid)
            if e is None:
                errors.append(f"子任务 {sid} target 引用的 {eid} 不存在于证据登记表")
                if sym and sym.lower() not in _PLACEHOLDER:
                    missing.append(sym)
                continue
            if e["symbol"]:
                sym_ok = bool(sym) and (e["symbol"] == sym or e["symbol"].endswith("." + sym)
                                        or sym.endswith("." + e["symbol"]))
            else:
                sym_ok = not sym
            if e["path"] != path or not sym_ok:
                errors.append(
                    f"子任务 {sid} target {path}:{sym} 与证据 {eid}（{e['path']}:{e['symbol']}）不一致")
        if kind == "modify":
            tpaths = {str(t.get("path") or "").replace("\\", "/").lstrip("./") for t in targets}
            for p in scopes:
                if p not in tpaths:
                    errors.append(f"子任务 {sid} write_scope {p} 没有对应 target（write_scope 与 targets 不一致）")
            for p in tpaths:
                if p and p not in scopes:
                    errors.append(f"子任务 {sid} target 路径 {p} 不在 write_scope 中")
    # coverage map
    need = coverage_files(registry)
    cov = {}
    for c in coverage or []:
        if isinstance(c, dict) and c.get("path"):
            cov[str(c["path"]).replace("\\", "/").lstrip("./")] = c
    modify_scopes = {str(p).replace("\\", "/").lstrip("./")
                     for s in specs if s.get("kind") == "modify" for p in (s.get("write_scope") or [])}
    for f in need:
        c = cov.get(f)
        if c is None:
            errors.append(f"coverage 缺少证据文件 {f} 的角色")
            continue
        role = str(c.get("role") or "").lower()
        if role not in ROLES:
            errors.append(f"coverage {f} 角色「{role}」无效")
        elif role == "not-needed" and (str(c.get("evidence_id") or "") not in by_id or not c.get("reason")):
            errors.append(f"coverage {f} 标记 not-needed 但缺少有效 evidence_id/理由")
        elif role == "modify" and f not in modify_scopes:
            errors.append(f"coverage {f} 标记 modify 但没有 modify 子任务的 write_scope 包含它")
    for f in modify_scopes:
        c = cov.get(f)
        if c is not None and str(c.get("role") or "").lower() != "modify":
            errors.append(f"coverage {f} 角色与 write_scope 不一致（应为 modify）")
    if task_type == "analysis" and any(s.get("kind") == "modify" for s in specs):
        errors.append("分析类目标禁止 modify")
    return errors, list(dict.fromkeys(missing))
