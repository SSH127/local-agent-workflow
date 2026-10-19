"""Subtask write scope: which workspace paths a running subtask may modify.

A subtask contract carries ``write_scope``: a list of paths / globs relative to
the workspace (``app/parser.py``, ``app/**``, ``tests/test_*.py``). A pattern
without glob characters covers that path *and* everything beneath it (so a
directory name means "this directory"). ``**`` matches any number of path
segments; ``*`` / ``?`` / ``[..]`` match inside one segment. An **empty scope
means read-only**.

The scope is enforced in the tool layer (``write_file`` / ``edit_file`` /
``rollback_file`` and commands that declare writes) via :class:`WriteScope`;
every rejected write is recorded as a ``scope_violation`` evidence item.

Overlap (:func:`scopes_overlap`) is a conservative "could some path be matched
by both" test, used by the plan static check (two overlapping scopes need a
dependency path) and by the parallel scheduler (overlapping subtasks never run
at the same time).
"""
from __future__ import annotations

import os
import re
import threading
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

_GLOB = re.compile(r"[*?\[]")
_CASE_INSENSITIVE = os.name == "nt"

# Tools that write files (checked against the scope).
WRITE_TOOLS = {"write_file", "edit_file", "rollback_file"}


class ScopeError(ValueError):
    """Invalid write-scope pattern."""


def normalize_pattern(pattern: str) -> str:
    """Normalize one pattern to a workspace-relative POSIX form.

    Rejects absolute paths, drive letters and ``..`` segments.
    """
    raw = str(pattern or "").strip().replace("\\", "/")
    if not raw:
        raise ScopeError("write_scope 不能包含空路径。")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ScopeError(f"write_scope 必须是相对工作区的路径：{pattern}")
    segs = [s for s in raw.split("/") if s not in ("", ".")]
    if any(s == ".." for s in segs):
        raise ScopeError(f"write_scope 不能包含 ..：{pattern}")
    if not segs:
        return "**"
    return "/".join(segs)


def normalize_scope(patterns: Any) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    out: list[str] = []
    for p in patterns:
        norm = normalize_pattern(p)
        if norm not in out:
            out.append(norm)
    return out


def _fold(text: str) -> str:
    return text.lower() if _CASE_INSENSITIVE else text


def _segs(text: str) -> tuple[str, ...]:
    return tuple(s for s in _fold(text).replace("\\", "/").split("/") if s not in ("", "."))


def _expand(pattern: str) -> tuple[str, ...]:
    """Literal pattern = that path and its whole subtree."""
    segs = _segs(pattern)
    if not _GLOB.search(pattern):
        segs = segs + ("**",)
    return segs


def _seg_match(path_seg: str, pat_seg: str) -> bool:
    return fnmatchcase(path_seg, pat_seg) if _GLOB.search(pat_seg) else path_seg == pat_seg


@lru_cache(maxsize=4096)
def _match(pat: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pat:
        return not path
    if pat[0] == "**":
        return _match(pat[1:], path) or (bool(path) and _match(pat, path[1:]))
    if not path:
        return False
    return _seg_match(path[0], pat[0]) and _match(pat[1:], path[1:])


def path_in_pattern(pattern: str, rel_path: str) -> bool:
    return _match(_expand(normalize_pattern(pattern)), _segs(rel_path))


def path_in_scope(scope: list[str], rel_path: str) -> bool:
    return any(path_in_pattern(p, rel_path) for p in scope)


def dir_in_scope(scope: list[str], rel_dir: str) -> bool:
    """True when EVERY path under ``rel_dir`` is inside the scope (a command
    that may write anywhere below a directory)."""
    target = _segs(rel_dir)
    for p in scope:
        pat = _expand(normalize_pattern(p))
        if "**" not in pat:
            continue
        k = pat.index("**")
        if pat[k + 1:]:
            continue  # trailing segments after ** narrow the subtree
        if len(target) < k:
            continue
        if all(_seg_match(target[i], pat[i]) for i in range(k)):
            return True
    return False


def _seg_overlap(x: str, y: str) -> bool:
    gx, gy = _GLOB.search(x), _GLOB.search(y)
    if not gx and not gy:
        return x == y
    if not gx:
        return fnmatchcase(x, y)
    if not gy:
        return fnmatchcase(y, x)
    # Both globs: only "*.ext" vs "*.other" is decided exactly; otherwise
    # assume they can overlap (conservative).
    if x.startswith("*") and y.startswith("*") and not _GLOB.search(x[1:]) \
            and not _GLOB.search(y[1:]):
        return x[1:].endswith(y[1:]) or y[1:].endswith(x[1:])
    return True


@lru_cache(maxsize=4096)
def _intersect(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    if not a and not b:
        return True
    if a and a[0] == "**":
        return _intersect(a[1:], b) or (bool(b) and _intersect(a, b[1:]))
    if b and b[0] == "**":
        return _intersect(a, b[1:]) or (bool(a) and _intersect(a[1:], b))
    if not a or not b:
        return False
    return _seg_overlap(a[0], b[0]) and _intersect(a[1:], b[1:])


def patterns_overlap(a: str, b: str) -> bool:
    return _intersect(_expand(normalize_pattern(a)), _expand(normalize_pattern(b)))


def scopes_overlap(a: list[str] | None, b: list[str] | None) -> bool:
    """Could any path be written by both scopes? Empty scope never overlaps."""
    return any(patterns_overlap(x, y) for x in (a or []) for y in (b or []))


def overlapping_pairs(a: list[str] | None, b: list[str] | None) -> list[tuple[str, str]]:
    return [(x, y) for x in (a or []) for y in (b or []) if patterns_overlap(x, y)]


# ---------------------------------------------------------------------------
# Enforcement guard (one per running subtask attempt)
# ---------------------------------------------------------------------------


class WriteScope:
    """Write permission of the running subtask, handed to the tool layer.

    ``check_*`` return ``None`` when allowed, else a clear error message; each
    rejection is appended to :attr:`violations` (thread-safe) so the control
    loop can store it as acceptance evidence.
    """

    def __init__(self, patterns: list[str] | None, *, subtask_id: str | None = None) -> None:
        self.patterns = normalize_scope(patterns or [])
        self.subtask_id = subtask_id
        self.violations: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def read_only(self) -> bool:
        return not self.patterns

    def _describe(self) -> str:
        return "只读（write_scope 为空）" if self.read_only else "、".join(self.patterns)

    def _reject(self, tool: str, target: str, why: str) -> str:
        message = (
            f"写入范围限制：子任务 [{self.subtask_id or '?'}] 不允许{why} {target}。"
            f"允许的 write_scope：{self._describe()}。如确需修改，请在计划中调整该子任务的 write_scope。"
        )
        with self._lock:
            self.violations.append({
                "type": "scope_violation",
                "tool": tool,
                "path": target,
                "scope": list(self.patterns),
                "subtask_id": self.subtask_id,
                "message": message,
                "at": datetime.now(UTC).isoformat(),
            })
        return message

    def check_path(self, tool: str, rel_path: str) -> str | None:
        if path_in_scope(self.patterns, rel_path):
            return None
        return self._reject(tool, rel_path, "写入")

    def check_command_writes(self, tool: str, writes: list[str]) -> str | None:
        """``writes``: declared targets; a trailing ``/`` marks a directory
        (the command may write anywhere beneath it)."""
        for target in writes:
            if target.endswith("/"):
                rel = target.rstrip("/") or "."
                allowed = dir_in_scope(self.patterns, "" if rel == "." else rel)
            else:
                allowed = path_in_scope(self.patterns, target)
            if not allowed:
                return self._reject(tool, target, "通过命令写入")
        return None

    def snapshot_violations(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.violations)


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return PurePosixPath(Path(path).resolve().relative_to(Path(workspace).resolve())).as_posix()
    except ValueError:
        return Path(path).as_posix()
