"""The first batch of real agent tools.

Each tool returns a structured dict ``{"ok": bool, "output": str, "data": Any}``
where ``ok`` reflects the *actual* execution result (exit code, HTTP status,
file operation success), never model-generated text.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from app import config, write_guard
from app.tools import command_policy, security
from app.tools.base import Tool, ToolContext, tool_error, tool_ok
from app.tools.security import SecurityError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    ".git", ".venv", "node_modules", "dist", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "data", ".cursor", ".idea", ".vscode",
}

_BINARY_MAGIC_BYTES = (0x00,)


def _is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(1024)
    except OSError:
        return False
    return not any(b in _BINARY_MAGIC_BYTES for b in head)


def _walk_files(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name in _SKIP_DIRS:
            continue
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir() and recursive:
            files.extend(_walk_files(entry, recursive))
    return files


def _truncate(text: str, limit: int = config.TOOL_OUTPUT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[已截断，剩余 {len(text) - limit} 字符]"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _list_files(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        path = security.resolve_workspace_path(args.get("path", "."), ctx.workspace_path)
        if not path.exists():
            return tool_error(f"路径不存在：{path}")
        if not path.is_dir():
            return tool_error(f"不是目录：{path}")
        recursive = bool(args.get("recursive", False))
        max_entries = int(args.get("max_entries", 200))
        rel_root = path.relative_to(ctx.workspace_path)

        if ctx.project is not None:
            # Cached listing (memoized until the next file mutation).
            raw_entries = ctx.project.list_files(rel_root.as_posix() or ".", recursive)
            listing = raw_entries[:max_entries]
            return tool_ok(
                f"列出 {len(listing)} 个条目（目录：{rel_root.as_posix() or '.'}）。",
                data={"entries": listing, "truncated": len(raw_entries) > max_entries},
            )

        if recursive:
            entries = _walk_files(path, True)
        else:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))

        listing = []
        for entry in entries[:max_entries]:
            if entry.name in _SKIP_DIRS:
                continue
            rel = entry.relative_to(ctx.workspace_path)
            listing.append({
                "path": rel.as_posix(),
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return tool_ok(
            f"列出 {len(listing)} 个条目（目录：{rel_root.as_posix() or '.'}）。",
            data={"entries": listing, "truncated": len(entries) > max_entries},
        )
    except SecurityError as exc:
        return tool_error(str(exc))


def _read_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        path = security.resolve_workspace_path(args.get("path", ""), ctx.workspace_path)
        if not path.exists():
            # The file was deleted or renamed externally: drop any cached
            # content/index for that path so nothing can be served from it.
            if ctx.project is not None:
                try:
                    rel_missing = path.relative_to(Path(ctx.workspace_path).resolve()).as_posix()
                    ctx.project.invalidate(rel_missing)
                except ValueError:
                    pass
            return tool_error(f"文件不存在：{path}")
        if not path.is_file():
            return tool_error(f"不是文件：{path}")
        if security.is_protected_file(path, ctx.workspace_path):
            return tool_error("受保护文件：该文件包含敏感信息，禁止读取其内容。")

        offset = max(1, int(args.get("offset", 1)))
        limit = max(1, int(args.get("limit", 200)))
        rel_key = path.relative_to(ctx.workspace_path).as_posix()
        cache_hit = None
        content_hash = None
        if ctx.project is not None:
            # Security checks (resolve/validity/protected) already ran above;
            # the cache never bypasses them. A hit means the on-disk version
            # is unchanged, so the cached bytes are authoritative.
            entry = ctx.project.get_file(rel_key)
            text = entry.content
            cache_hit = entry.cache_hit
            content_hash = entry.content_hash
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            content_hash = write_guard.content_hash(text)
        lines = text.splitlines()
        total = len(lines)
        # Record which version this run saw: write tools refuse to overwrite a
        # file that changed after this read (stale-write / lost-update guard).
        if ctx.read_tracker is not None:
            ctx.read_tracker.record_version(rel_key, content_hash)

        if total == 0:
            return tool_ok(f"文件为空：{path}", data={
                "path": rel_key,
                "content": "", "total_lines": 0, "truncated": False,
                "content_hash": content_hash,
            })

        start = offset - 1
        requested_end = min(start + limit, total)  # 1-based inclusive

        # Build the content within the per-read char budget. Lines are added
        # until the budget would be exceeded; the remainder is reported (never
        # silently dropped) with the exact continuation offset.
        budget = config.READ_FILE_MAX_CHARS
        content_lines: list[str] = []
        content_chars = 0
        last_included = start - 1  # 0-based index of the last line returned
        for i in range(start, requested_end):
            line = f"{i + 1}: {lines[i]}"
            extra = len(line) + (1 if content_lines else 0)
            if content_lines and content_chars + extra > budget:
                break
            content_lines.append(line)
            content_chars += extra
            last_included = i

        returned_end = last_included + 1  # 1-based inclusive
        truncated = last_included < requested_end - 1
        has_more = requested_end < total  # more of the file beyond this window
        content = "\n".join(content_lines)

        # Duplicate-read detection via the run's shared tracker.
        overlap = 0
        covered: list[tuple[int, int]] = []
        if ctx.read_tracker is not None:
            overlap, covered = ctx.read_tracker.record(rel_key, offset, returned_end)

        notes: list[str] = []
        if cache_hit is True:
            notes.append("内容来自缓存（文件版本未变，未重新读盘）")
        if overlap > 0:
            notes.append(f"本次范围中有 {overlap} 行此前已读取")
        if truncated:
            notes.append(
                f"内容在第 {returned_end} 行被截断（字符预算 {budget}），"
                f"继续读取请使用 offset={returned_end + 1}"
            )
        if not truncated and has_more:
            notes.append(f"文件还有后续内容，继续读取请使用 offset={requested_end + 1}")
        suffix = f"（{'；'.join(notes)}）" if notes else ""

        return tool_ok(
            f"读取 {rel_key}：第 {offset}-{returned_end} 行 / 共 {total} 行。{suffix}",
            data={
                "path": rel_key,
                "content": content,
                "total_lines": total,
                "offset": offset,
                "limit": limit,
                "start_line": offset,
                "end_line": returned_end,
                "truncated": truncated,
                "has_more": has_more,
                "next_offset": returned_end + 1 if (truncated or has_more) else None,
                "overlap_lines": overlap,
                "already_read_ranges": covered,
                "cache_hit": cache_hit,
                "content_hash": content_hash,
            },
        )
    except SecurityError as exc:
        return tool_error(str(exc))


def _search_code(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = str(args.get("query", ""))
    if not query.strip():
        return tool_error("搜索内容为空：请提供 query 参数。")
    try:
        base = security.resolve_workspace_path(args.get("path", "."), ctx.workspace_path)
        glob = args.get("glob") or None
        case_sensitive = bool(args.get("case_sensitive", False))
        max_results = int(args.get("max_results", 50))
        # Optional surrounding-line context per match (default 0 = match line only).
        context_lines = max(0, int(args.get("context_lines", 0)))
        recursive = glob is not None or args.get("recursive", True)
        root = base if base.is_dir() else base.parent

        flags = 0 if case_sensitive else re.IGNORECASE

        # Fast path: a plain identifier query can be answered from the symbol
        # index (class/function definitions + line numbers) without scanning
        # every file. Protected files are still filtered out.
        if ctx.project is not None and root.is_dir() and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", query
        ):
            rel_root = root.relative_to(ctx.workspace_path).as_posix() or "."
            syms = ctx.project.query_symbols(query, rel_root, glob, case_sensitive)
            if syms:
                results = []
                for s in syms[:max_results]:
                    abs_path = ctx.workspace_path / s["rel_path"]
                    if security.is_protected_file(abs_path, ctx.workspace_path):
                        continue
                    item = {
                        "path": s["rel_path"], "line": s["line"],
                        "text": s["name"], "symbol": s["name"], "kind": s["kind"],
                    }
                    if context_lines > 0:
                        try:
                            lines = ctx.project.get_file(s["rel_path"]).content.splitlines()
                            lo = max(1, s["line"] - context_lines)
                            hi = min(len(lines), s["line"] + context_lines)
                            item["context"] = [
                                {"line": n, "text": lines[n - 1].strip()[:300]}
                                for n in range(lo, hi + 1)
                            ]
                        except OSError:
                            pass
                    results.append(item)
                return tool_ok(
                    f"找到 {len(results)} 处匹配（符号索引命中）。",
                    data={"matches": results, "query": query, "index_hit": True},
                )

        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            return tool_error(f"无效的正则表达式：{exc}")

        results = []
        files = _walk_files(root, recursive) if root.is_dir() else [base]
        for file_path in files:
            if glob and not file_path.match(glob):
                continue
            if not _is_text_file(file_path):
                continue
            # Never leak the contents of protected files (e.g. .env) through search.
            if security.is_protected_file(file_path, ctx.workspace_path):
                continue
            rel = file_path.relative_to(ctx.workspace_path).as_posix()
            if ctx.project is not None:
                try:
                    all_lines = ctx.project.get_file(rel).content.splitlines()
                except OSError:
                    continue
            else:
                try:
                    all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
            for lineno, line in enumerate(all_lines, start=1):
                if not pattern.search(line):
                    continue
                match = {"path": rel, "line": lineno, "text": line.strip()[:300]}
                if context_lines > 0:
                    lo = max(1, lineno - context_lines)
                    hi = min(len(all_lines), lineno + context_lines)
                    match["context"] = [
                        {"line": n, "text": all_lines[n - 1].strip()[:300]}
                        for n in range(lo, hi + 1)
                    ]
                results.append(match)
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        return tool_ok(
            f"找到 {len(results)} 处匹配。",
            data={"matches": results, "query": query},
        )
    except SecurityError as exc:
        return tool_error(str(exc))


def _write_requires_approval(args: dict[str, Any], workspace: Any = None) -> bool:
    """Protected file, or an explicit destructive whole-file rewrite."""
    try:
        path = security.resolve_workspace_path(args.get("path", ""), workspace)
    except SecurityError:
        return False  # the handler reports the boundary violation
    if security.is_protected_file(path, workspace):
        return True
    return bool(args.get("allow_truncate", False)) and path.is_file()


def _write_common_checks(
    path: Path, ctx: ToolContext, args: dict[str, Any], tool: str
) -> None:
    """Approval check shared by ``write_file`` / ``edit_file``.

    Reaching this function without a valid authorization means the executor's
    gate was bypassed, so it fails closed.
    """
    protected = security.is_protected_file(path, ctx.workspace_path)
    destructive = (
        tool == "write_file"
        and bool(args.get("allow_truncate", False))
        and path.is_file()
    )
    if not (protected or destructive):
        return
    if ctx.approval_check is None or not ctx.approval_check(tool, args):
        if protected:
            raise SecurityError("受保护文件：修改该文件需要用户批准。")
        raise SecurityError("整体缩短已有文件需要用户批准（allow_truncate=true）。")


def _scope_error(ctx: ToolContext, tool: str, path: Path) -> str | None:
    """Reject writes outside the running subtask's write_scope."""
    if ctx.write_scope is None:
        return None
    workspace = Path(ctx.workspace_path).resolve()
    try:
        rel = path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return None  # boundary violations are reported by the security layer
    return ctx.write_scope.check_path(tool, rel)


def scope_precheck(name: str, args: dict[str, Any], ctx: ToolContext) -> str | None:
    """Scope check BEFORE approval is requested (an out-of-scope write is never
    worth asking the user about). Returns an error message or ``None``."""
    if ctx.write_scope is None:
        return None
    try:
        if name in ("write_file", "edit_file"):
            path = security.resolve_workspace_path(args.get("path", ""), ctx.workspace_path)
            return _scope_error(ctx, name, path)
        if name == "rollback_file":
            target = write_guard.tx_target(ctx.run_id, str(args.get("tx_id", "")).strip(),
                                           ctx.workspace_path)
            if target is None:
                return None  # unknown tx: the handler reports it
            return ctx.write_scope.check_path(name, target)
        if name == "run_project_command":
            decision = security.classify_command(
                str(args.get("program", "")).strip(), [str(a) for a in args.get("args", [])],
                str(args.get("cwd", ".")), ctx.workspace_path,
            )
            if decision.writes:
                return ctx.write_scope.check_command_writes(name, list(decision.writes))
    except SecurityError:
        return None  # the normal path reports security rejections
    return None


def _expected_version(ctx: ToolContext, rel_key: str, explicit: Any) -> str | None:
    """Explicit ``expected_sha256`` wins; otherwise the version this run read."""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if ctx.read_tracker is not None:
        return ctx.read_tracker.version(rel_key)
    return None


def _write_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        path = security.resolve_workspace_path(args.get("path", ""), ctx.workspace_path)
        content = args.get("content")
        if content is None:
            return tool_error("缺少 content 参数。")
        if not isinstance(content, str):
            return tool_error("content 必须是字符串。")

        scope_error = _scope_error(ctx, "write_file", path)
        if scope_error:
            return tool_error(scope_error)
        # The executor gates this via requires_approval; reaching here without
        # approval is refused.
        _write_common_checks(path, ctx, args, "write_file")

        workspace = Path(ctx.workspace_path).resolve()
        rel_key = path.relative_to(workspace).as_posix()
        state = write_guard.inspect(
            path, rel_key, workspace,
            protected=security.is_protected_file(path, workspace),
        )
        allow_truncate = bool(args.get("allow_truncate", False))

        # Refuse a whole-file rewrite that would destroy most of a large file.
        write_guard.check_shrink_guard(state, content, allow_truncate=allow_truncate)
        # Refuse to overwrite a version this run never saw.
        write_guard.check_expected_version(
            state, _expected_version(ctx, rel_key, args.get("expected_sha256")),
            source="写入",
        )

        tx = write_guard.prepare(state, run_id=ctx.run_id, tool="write_file",
                                 workspace=workspace)
        try:
            data = write_guard.commit(tx, content, project=ctx.project,
                                      tracker=ctx.read_tracker)
        except Exception as exc:  # noqa: BLE001 - record the failure, keep the file
            write_guard.abort(tx, str(exc))
            if isinstance(exc, write_guard.WriteConflict):
                return tool_error(str(exc))
            return tool_error(f"写入失败（原文件未被修改）：{exc}")

        syntax_error = write_guard.syntax_error_for(rel_key, content)
        notes = []
        if syntax_error:
            notes.append(f"语法检查未通过（第 {syntax_error}），文件已写入，请修复")
        notes.append(f"写入前版本已快照：{data['tx_id'][:8]}")
        return tool_ok(
            f"已写入 {rel_key}（{len(content)} 字符；{'；'.join(notes)}）。",
            data={
                **data,
                "path": rel_key,
                "bytes": len(content.encode("utf-8")),
                "syntax_ok": syntax_error is None,
                "syntax_error": syntax_error,
            },
        )
    except write_guard.WriteGuardError as exc:
        return tool_error(str(exc))
    except SecurityError as exc:
        return tool_error(str(exc))
    except OSError as exc:
        return tool_error(f"写入失败：{exc}")


def _edit_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Replace an exact snippet in an existing file (surgical edit).

    Whole-file rewrites are the main source of accidental truncation: the model
    has to reproduce every line from memory. This tool only requires the few
    lines being changed, and refuses ambiguous matches instead of guessing.
    """
    try:
        path = security.resolve_workspace_path(args.get("path", ""), ctx.workspace_path)
        old = args.get("old_string")
        new = args.get("new_string")
        replace_all = bool(args.get("replace_all", False))
        if not isinstance(old, str) or not old:
            return tool_error("old_string 必须是非空字符串。")
        if not isinstance(new, str):
            return tool_error("new_string 必须是字符串。")
        if old == new:
            return tool_error("old_string 与 new_string 相同，无需编辑。")

        scope_error = _scope_error(ctx, "edit_file", path)
        if scope_error:
            return tool_error(scope_error)
        _write_common_checks(path, ctx, args, "edit_file")
        if not path.is_file():
            return tool_error(f"文件不存在：{path}")

        workspace = Path(ctx.workspace_path).resolve()
        rel_key = path.relative_to(workspace).as_posix()
        state = write_guard.inspect(
            path, rel_key, workspace,
            protected=security.is_protected_file(path, workspace),
        )
        write_guard.check_expected_version(
            state, _expected_version(ctx, rel_key, args.get("expected_sha256")),
            source="编辑",
        )

        original = state.text  # newline-normalized
        old_norm = old.replace("\r\n", "\n")
        new_norm = new.replace("\r\n", "\n")

        count = original.count(old_norm)
        if count == 0:
            return tool_error(
                "未找到要替换的内容：old_string 与文件内容不匹配。"
                "请先 read_file 确认原文（注意缩进、空格与换行必须完全一致）。"
            )
        if count > 1 and not replace_all:
            return tool_error(
                f"old_string 在文件中出现 {count} 次，无法唯一确定替换位置。"
                "请提供更长的上下文使其唯一，或设置 replace_all=true。"
            )

        updated = (
            original.replace(old_norm, new_norm)
            if replace_all
            else original.replace(old_norm, new_norm, 1)
        )
        # Preserve the file's original newline style: a one-line edit must not
        # rewrite every line ending in the file.
        payload = updated if state.newline == "\n" else updated.replace("\n", "\r\n")

        tx = write_guard.prepare(state, run_id=ctx.run_id, tool="edit_file",
                                 workspace=workspace)
        try:
            data = write_guard.commit(tx, payload, project=ctx.project,
                                      tracker=ctx.read_tracker)
        except Exception as exc:  # noqa: BLE001 - record the failure, keep the file
            write_guard.abort(tx, str(exc))
            if isinstance(exc, write_guard.WriteConflict):
                return tool_error(str(exc))
            return tool_error(f"编辑失败（原文件未被修改）：{exc}")

        replaced = count if replace_all else 1
        syntax_error = write_guard.syntax_error_for(rel_key, updated)
        note = f"；语法检查未通过（{syntax_error}）" if syntax_error else ""
        return tool_ok(
            f"已编辑 {rel_key}（替换 {replaced} 处，"
            f"{len(original)} → {len(updated)} 字符{note}）。",
            data={
                **data,
                "path": rel_key,
                "replacements": replaced,
                "old_chars": len(original),
                "new_chars": len(updated),
                "syntax_ok": syntax_error is None,
                "syntax_error": syntax_error,
            },
        )
    except write_guard.WriteGuardError as exc:
        return tool_error(str(exc))
    except SecurityError as exc:
        return tool_error(str(exc))
    except OSError as exc:
        return tool_error(f"编辑失败：{exc}")


def _rollback_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Restore the pre-write snapshot of one transaction from this run."""
    tx_id = str(args.get("tx_id", "")).strip()
    if not tx_id:
        return tool_error("缺少 tx_id：请使用写入结果中返回的事务 ID。")
    if ctx.write_scope is not None:
        target = write_guard.tx_target(ctx.run_id, tx_id, ctx.workspace_path)
        if target is not None:
            scope_error = ctx.write_scope.check_path("rollback_file", target)
            if scope_error:
                return tool_error(scope_error)
    # Defense in depth: rollback mutates files and must never run ungated.
    if ctx.approval_check is None or not ctx.approval_check("rollback_file", args):
        return tool_error("回滚会修改文件，需要用户批准。")
    try:
        data = write_guard.rollback(
            run_id=ctx.run_id,
            tx_id=tx_id,
            workspace=ctx.workspace_path,
            force=bool(args.get("force", False)),
            project=ctx.project,
            tracker=ctx.read_tracker,
        )
    except write_guard.WriteConflict as exc:
        return tool_error(str(exc))
    except write_guard.WriteGuardError as exc:
        return tool_error(str(exc))
    except OSError as exc:
        return tool_error(f"回滚失败：{exc}")
    state = "已恢复写入前版本" if not data.get("already") else "文件已是写入前版本"
    return tool_ok(f"回滚 {tx_id[:8]}：{state}（{data.get('rollback_status')}）。", data=data)


def command_timeout_seconds(args: dict[str, Any]) -> float:
    """Effective timeout for ``run_project_command``.

    The model-supplied ``timeout_seconds`` is honoured but clamped to
    ``config.MAX_COMMAND_TIMEOUT_SECONDS``; missing/invalid/non-positive values
    fall back to ``config.TOOL_TIMEOUT_SECONDS``.
    """
    default = float(config.TOOL_TIMEOUT_SECONDS)
    raw = args.get("timeout_seconds")
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        value = default
    if math.isnan(value) or value <= 0:
        value = default
    return max(0.1, min(value, float(config.MAX_COMMAND_TIMEOUT_SECONDS)))


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a command and every child it spawned (best effort)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def _run_project_command(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    program = str(args.get("program", "")).strip()
    argv = [str(a) for a in args.get("args", [])]
    cwd = str(args.get("cwd", "."))
    timeout = command_timeout_seconds(args)

    # Cooperative cancel: refuse to start once the run/subtask is cancelled.
    if ctx.cancel_check is not None:
        reason = ctx.cancel_check()
        if reason:
            return tool_error(f"任务已取消，未启动命令：{reason}")

    try:
        # Argument-level analysis decides *what code* this command would run;
        # an allowlisted program name alone is not enough (``python -m pytest
        # script.py`` executes the script). Approval, when required, is bound to
        # this run and to the exact program/args/cwd by the executor.
        decision = security.classify_command(program, argv, cwd, ctx.workspace_path)
        # Read-only subtask: refuse commands that declare concrete writes, and
        # refuse cache_kind tools when soft sandbox cannot redirect them.
        if (
            config.COMMAND_READONLY_FS_HINT
            and ctx.write_scope is not None
            and ctx.write_scope.read_only
        ):
            if decision.writes:
                return tool_error(
                    "只读子任务（write_scope 为空）拒绝会写入工作区的命令："
                    + ", ".join(decision.writes[:8])
                    + "（COMMAND_READONLY_FS_HINT）。"
                )
            if decision.cache_kind and not config.COMMAND_SOFT_SANDBOX:
                return tool_error(
                    f"只读子任务拒绝会写入缓存的命令（cache_kind={decision.cache_kind}）；"
                    "请启用 COMMAND_SOFT_SANDBOX 或扩大 write_scope。"
                )
            if command_policy.looks_networky(program, argv):
                approved = (
                    ctx.approval_check is not None
                    and ctx.approval_check("run_project_command", args)
                )
                if not approved:
                    return tool_error(
                        "只读子任务拒绝疑似联网命令（需批准或扩大 write_scope）："
                        f"{program} {' '.join(argv[:6])}"
                    )
        # Networky allowlisted commands: require approval when configured.
        if (
            config.COMMAND_BLOCK_NETWORK_TOOLS
            and command_policy.looks_networky(program, argv)
            and (
                ctx.approval_check is None
                or not ctx.approval_check("run_project_command", args)
            )
        ):
            return tool_error(
                "疑似联网命令需要用户批准后才能执行"
                f"（COMMAND_BLOCK_NETWORK_TOOLS）：{program} {' '.join(argv[:6])}。"
            )
        if ctx.write_scope is not None and decision.writes:
            scope_error = ctx.write_scope.check_command_writes(
                "run_project_command", list(decision.writes))
            if scope_error:
                return tool_error(scope_error)
        if decision.requires_approval and (
            ctx.approval_check is None
            or not ctx.approval_check("run_project_command", args)
        ):
            reason = f"（{decision.risk}）" if decision.risk else ""
            return tool_error(f"该命令需要用户批准后才能执行{reason}。")
        workdir = security.resolve_workspace_path(cwd, ctx.workspace_path)
    except SecurityError as exc:
        return tool_error(str(exc))

    # Model-supplied env: drop secret-looking values and every variable that can
    # redirect imports/plugin loading (PYTEST_ADDOPTS, PYTHONPATH, PATH, ...).
    raw_env = {str(k): str(v) for k, v in dict(args.get("env") or {}).items()}
    for key in list(raw_env):
        if security._looks_like_secret_key(key):
            raw_env.pop(key)
    extra_env, _dropped_model_env = command_policy.filter_model_env(raw_env)

    # Inherit the parent env (minus import/plugin redirection), then prepend the
    # active venv's bin dirs so ``python``, ``pytest``, ``ruff``, ``npm`` etc.
    # resolve inside the project environment.
    child_env = os.environ.copy()
    command_policy.sanitize_inherited_env(child_env)
    bin_dirs = []
    exe_dir = Path(sys.executable).resolve().parent
    if exe_dir.name.lower() in {"scripts", "bin"}:
        bin_dirs.append(str(exe_dir))
        bin_dirs.append(str(exe_dir.parent))
    else:
        bin_dirs.append(str(exe_dir))
        bin_dirs.append(str(exe_dir / "Scripts"))
        bin_dirs.append(str(exe_dir / "bin"))
    existing_path = child_env.get("PATH", "")
    trusted_path = os.pathsep.join(bin_dirs + ([existing_path] if existing_path else []))
    child_env["PATH"] = trusted_path
    child_env.update(extra_env)
    if config.COMMAND_DENY_NETWORK:
        # Honest best-effort: strip proxy knobs + Python-only socket block via
        # sitecustomize on PYTHONPATH. Native binaries are NOT firewalled.
        for k in list(child_env):
            ku = k.upper()
            if ku in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
                      "http_proxy", "https_proxy", "all_proxy", "ftp_proxy"}:
                child_env.pop(k, None)
        child_env["NO_PROXY"] = "*"
        child_env["no_proxy"] = "*"
        child_env["AGENT_DENY_NETWORK"] = "1"
        prog_l = program.lower()
        if prog_l in {"python", "python3"} or (
            prog_l in {"uv"} and argv and argv[0] == "run"
        ):
            block_dir = str(Path(__file__).resolve().parent / "netblock")
            prev = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = (
                block_dir if not prev else block_dir + os.pathsep + prev
            )

    # Soft sandbox when write_scope is set: always redirect TMP/pyc into a
    # per-subtask scratch and take a pre-command inventory so escapes outside
    # scope (+ non-artifact) can be restored. Cache kinds additionally get
    # tool-specific CLI/env pins. This is snapshot+policy — not a kernel jail.
    from app.tools import command_sandbox

    session = None
    effective_argv = list(argv)
    if ctx.write_scope is not None:
        sid = ctx.subtask_id or "default"
        kind = decision.cache_kind or "generic"
        session = command_sandbox.begin_session(
            workspace=Path(ctx.workspace_path),
            run_id=ctx.run_id,
            subtask_id=sid,
            scope_patterns=list(ctx.write_scope.patterns),
            kind=kind,
        )
        child_env.update(session.env)
        if decision.cache_kind:
            effective_argv = command_policy.inject_cache_argv(
                decision, argv, str(session.scratch))
        if config.COMMAND_SOFT_SANDBOX:
            path_err = command_sandbox.check_argv_outside_workspace(
                [program, *effective_argv],
                Path(ctx.workspace_path),
                session.scratch,
            )
            if path_err:
                return tool_error(path_err)

    # Resolve the program to a full path *using the trusted PATH* so bare-name
    # ambiguity (Windows App Execution Aliases, symlinked interpreters) cannot
    # route to the wrong binary — a model-supplied PATH is ignored.
    if program == "python":
        executable = sys.executable
    else:
        resolved = shutil.which(program, path=trusted_path)
        if resolved is None:
            return tool_error(f"未找到可执行程序：{program}。请确认它在当前环境可用。")
        executable = resolved

    extra_flags = command_policy.python_interpreter_flags(program, effective_argv)

    # Own process group so a timeout / cancel can kill the whole tree (test
    # runners and npm spawn children that would otherwise outlive the kill).
    group_kwargs: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt" else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            [executable, *extra_flags, *effective_argv],
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            **group_kwargs,
        )
    except FileNotFoundError:
        return tool_error(f"未找到可执行程序：{program}。请确认它在当前环境可用。")
    except OSError as exc:
        return tool_error(f"命令执行失败：{exc}")

    # Register with the cancel token (if any) so a concurrent cancel kills us.
    cancel_token = ctx.cancel_token
    if cancel_token is not None:
        cancel_token.register_proc(proc)

    job_handle = None
    with contextlib.suppress(Exception):
        from app.tools.win_job import attach_kill_on_close_job
        job_handle = attach_kill_on_close_job(proc)

    cancelled = False
    timed_out = False
    box: dict[str, str | None] = {"out": None, "err": None}
    done = threading.Event()

    def _reader() -> None:
        try:
            with contextlib.suppress(Exception):
                out, err = proc.communicate()
                box["out"], box["err"] = out, err
        finally:
            done.set()

    reader = threading.Thread(target=_reader, name="cmd-reader", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while not done.wait(0.25):
            if ctx.cancel_check is not None and ctx.cancel_check():
                cancelled = True
                _kill_process_tree(proc)
                done.wait(5)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _kill_process_tree(proc)
                done.wait(5)
                break
    finally:
        if cancel_token is not None:
            cancel_token.unregister_proc(proc)
        if job_handle is not None:
            with contextlib.suppress(Exception):
                from app.tools.win_job import close_job
                close_job(job_handle)

    if cancelled:
        return tool_error(f"任务已取消，已终止进程：{program} {' '.join(argv)}")
    if timed_out:
        return tool_error(
            f"命令超时（>{timeout:g}s），已终止进程：{program} {' '.join(argv)}"
        )
    out_text = box["out"] or ""
    err_text = box["err"] or ""

    stdout = security.redact_text(out_text or "")
    stderr = security.redact_text(err_text or "")
    returncode = proc.returncode
    ok = returncode == 0
    data: dict[str, Any] = {
        "program": program,
        "args": argv,
        "exit_code": returncode,
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
    }
    if session is not None:
        try:
            data["scratch"] = session.scratch.relative_to(session.workspace).as_posix()
        except ValueError:
            data["scratch"] = str(session.scratch)
        data["cache_kind"] = decision.cache_kind
        finish = command_sandbox.finish_session(session)
        if finish.artifacts:
            art_ev = command_sandbox.violations_as_evidence(
                finish.artifacts, tool="run_project_command",
                scope=list(ctx.write_scope.patterns),
                subtask_id=ctx.subtask_id,
            )
            with ctx.write_scope._lock:
                ctx.write_scope.violations.extend(art_ev)
            data["allowed_artifacts"] = [
                {"path": v.path, "action": v.action} for v in finish.artifacts
            ]
        if finish.violations:
            evidence = command_sandbox.violations_as_evidence(
                finish.violations, tool="run_project_command",
                scope=list(ctx.write_scope.patterns),
                subtask_id=ctx.subtask_id,
            )
            # Record on the WriteScope so the subtask's acceptance_evidence
            # picks them up the same way as write_file rejections.
            with ctx.write_scope._lock:
                ctx.write_scope.violations.extend(evidence)
            summary = "；".join(v.message for v in finish.violations[:5])
            data["scope_violations"] = [
                {"path": v.path, "action": v.action} for v in finish.violations
            ]
            return tool_error(
                f"命令写入了 write_scope 之外的路径（已尽量恢复）：{summary}",
                data=data,
            )
    if ok:
        return tool_ok(f"命令成功（退出码 0）：{program} {' '.join(argv)}", data=data)
    return tool_error(
        f"命令失败（退出码 {returncode}）：{program} {' '.join(argv)}",
        data=data,
    )

def _check_port(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    port = int(args.get("port", 0))
    host = str(args.get("host", "127.0.0.1"))
    if not (1 <= port <= 65535):
        return tool_error(f"非法端口：{port}")
    try:
        with socket.create_connection((host, port), timeout=5):
            return tool_ok(f"端口 {host}:{port} 有服务监听。", data={"listening": True})
    except OSError:
        return tool_ok(f"端口 {host}:{port} 无服务监听。", data={"listening": False})


def _check_http(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    url = str(args.get("url", ""))
    method = str(args.get("method", "GET")).upper()
    timeout = float(args.get("timeout_seconds") or 5)

    # Only allow local HTTP endpoints (no SSRF to arbitrary hosts).
    if not re.match(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?(/|$)", url):
        return tool_error("仅允许检查本地 HTTP 服务（localhost / 127.0.0.1）。")
    if method not in {"GET", "HEAD"}:
        return tool_error("仅支持 GET / HEAD 请求。")

    try:
        response = httpx.request(method, url, timeout=timeout)
    except httpx.HTTPError as exc:
        return tool_error(f"HTTP 请求失败：{exc}", data={"ok": False})
    body = security.redact_text(response.text or "")
    ok = 200 <= response.status_code < 400
    data = {
        "status_code": response.status_code,
        "url": url,
        "body_snippet": _truncate(body, 2000),
    }
    if ok:
        return tool_ok(f"HTTP {response.status_code}：{url}", data=data)
    return tool_error(f"HTTP {response.status_code}：{url}", data=data)


def _get_task_logs(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app import storage

    run_id = str(args.get("run_id") or ctx.run_id)
    limit = int(args.get("limit", 100))
    try:
        steps = storage.list_steps(run_id, limit)
    except Exception as exc:  # noqa: BLE001 - defensive; never crash the loop
        return tool_error(f"读取任务日志失败：{exc}")
    if not steps:
        return tool_ok("当前任务还没有执行日志。", data={"steps": []})
    return tool_ok(
        f"返回 {len(steps)} 条执行日志。",
        data={"steps": steps},
    )


def _ask_user(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # The executor intercepts ``ask_user`` (it needs the run's user channel);
    # reaching the handler directly means there is no such channel.
    return tool_error("ask_user 只能在任务执行器中使用（需要用户输入通道）。")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="list_files",
        description="列出允许访问的工作区目录中的文件与子目录。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作区或绝对路径，默认当前目录"},
                "recursive": {"type": "boolean", "description": "是否递归列出"},
                "max_entries": {"type": "integer", "description": "最多返回条目数，默认 200"},
            },
            "required": [],
        },
        handler=_list_files,
    ),
    Tool(
        name="read_file",
        description="读取工作区内某个文本文件的内容（受保护文件不可读）。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作区或绝对路径"},
                "offset": {"type": "integer", "description": "起始行号，从 1 开始"},
                "limit": {"type": "integer", "description": "读取行数，默认 200"},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ),
    Tool(
        name="search_code",
        description="在工作区代码中搜索函数、变量、报错信息或指定文本（支持正则）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜索的文本或正则表达式"},
                "path": {"type": "string", "description": "搜索目录，默认工作区根"},
                "glob": {"type": "string", "description": "文件名匹配，如 *.py"},
                "case_sensitive": {"type": "boolean", "description": "是否区分大小写"},
                "context_lines": {"type": "integer", "description": "每个匹配返回的上下文行数（默认 0，只返回匹配行）"},
                "max_results": {"type": "integer", "description": "最多返回结果数，默认 50"},
            },
            "required": ["query"],
        },
        handler=_search_code,
    ),
    Tool(
        name="write_file",
        description=(
            "创建新文件，或整体重写一个文件（覆盖受保护文件需批准）。"
            "修改已有文件时优先使用 edit_file 做局部替换——整体重写需要复述全部内容，"
            "容易截断或丢失代码。写入前会自动快照，可通过事务 ID 显式回滚。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径（相对或绝对）"},
                "content": {"type": "string", "description": "完整的新文件内容"},
                "allow_truncate": {
                    "type": "boolean",
                    "description": (
                        "仅当确实要把大文件整体缩短时设为 true（需要审批）；"
                        "否则用极小内容覆盖大文件会被拒绝"
                    ),
                },
                "expected_sha256": {
                    "type": "string",
                    "description": (
                        "可选：期望的当前内容哈希（read_file 返回的 content_hash）。"
                        "与磁盘不一致时拒绝写入，避免覆盖他人修改"
                    ),
                },
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
        requires_approval=_write_requires_approval,
    ),
    Tool(
        name="edit_file",
        description=(
            "对已有文件做精确的局部替换：给出唯一的 old_string 和替换后的 new_string。"
            "修改函数体、改几行逻辑时使用本工具，不要用 write_file 重写整个文件。"
            "old_string 必须与文件内容完全一致（含缩进）；匹配不到或不唯一时会报错。"
            "写入前会自动快照，可通过事务 ID 显式回滚。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要编辑的文件路径"},
                "old_string": {
                    "type": "string",
                    "description": "要被替换的原文（必须与文件内容完全一致，且默认唯一）",
                },
                "new_string": {"type": "string", "description": "替换后的新内容"},
                "replace_all": {
                    "type": "boolean",
                    "description": "为 true 时替换所有匹配；默认要求 old_string 唯一",
                },
                "expected_sha256": {
                    "type": "string",
                    "description": "可选：期望的当前内容哈希；不一致时拒绝写入",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=_edit_file,
        requires_approval=_write_requires_approval,
    ),
    Tool(
        name="rollback_file",
        description=(
            "把本任务某次写入回滚到写入前的快照（需要审批）。"
            "tx_id 取自 write_file / edit_file 返回结果。"
            "如果文件在写入后又被其他进程修改，会拒绝回滚以免覆盖新修改。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "tx_id": {"type": "string", "description": "写入事务 ID"},
                "force": {
                    "type": "boolean",
                    "description": "为 true 时即使文件已被外部修改也强制回滚（会先保存当前内容）",
                },
            },
            "required": ["tx_id"],
        },
        handler=_rollback_file,
        requires_approval=lambda args, workspace=None: True,
    ),
    Tool(
        name="run_project_command",
        description=(
            "运行受控的项目开发命令（参数化：program + args），例如运行测试、"
            "构建或类型检查。程序必须在白名单内。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "可执行程序名，如 pytest / npm / python"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参数列表（禁止使用 shell 元字符）",
                },
                "cwd": {"type": "string", "description": "运行目录，默认工作区根"},
                "timeout_seconds": {
                    "type": "number",
                    "description": "超时秒数（默认 TOOL_TIMEOUT_SECONDS，上限 MAX_COMMAND_TIMEOUT_SECONDS）",
                },
            },
            "required": ["program"],
        },
        handler=_run_project_command,
        requires_approval=lambda args, workspace=None: security.requires_approval(
            str(args.get("program", "")),
            [str(a) for a in args.get("args", [])],
            str(args.get("cwd", ".")),
            workspace,
        ),
    ),
    Tool(
        name="check_port",
        description="检查指定本地端口是否有服务监听。",
        parameters={
            "type": "object",
            "properties": {
                "port": {"type": "integer", "description": "端口号，1-65535"},
                "host": {"type": "string", "description": "主机，默认 127.0.0.1"},
            },
            "required": ["port"],
        },
        handler=_check_port,
    ),
    Tool(
        name="check_http",
        description="检查本地 HTTP 服务是否可以正常响应（仅限 localhost）。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "本地 http/https URL"},
                "method": {"type": "string", "description": "GET 或 HEAD"},
                "timeout_seconds": {"type": "number", "description": "超时秒数"},
            },
            "required": ["url"],
        },
        handler=_check_http,
    ),
    Tool(
        name="get_task_logs",
        description="获取当前 Agent 任务的执行日志与工具调用结果。",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "任务 ID，默认当前任务"},
                "limit": {"type": "integer", "description": "最多返回条数，默认 100"},
            },
            "required": [],
        },
        handler=_get_task_logs,
    ),
    Tool(
        name="ask_user",
        description=(
            "向用户提出一个具体、简短的问题，并暂停任务等待回复（状态 waiting_for_input）。"
            "仅在缺少完成任务所必需、且无法通过读取代码获得的信息时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "要问用户的问题"},
            },
            "required": ["question"],
        },
        handler=_ask_user,
    ),
]

_TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}


def get_tool(name: str) -> Tool | None:
    return _TOOL_BY_NAME.get(name)


def tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema() for tool in TOOLS]
