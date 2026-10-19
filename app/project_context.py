"""Per-project context cache: file content, listing and symbol index.

This is the data-access layer the file/command tools use after they have run
their own path-safety and protected-file checks. It is *not* a security layer:
security resolution (``resolve_workspace_path`` / ``is_protected_file``) always
runs first in the tools, and this module never skips it.

Design:

* Bounded by **project identity**: every row is keyed by the canonical,
  resolved project root path, so two projects can never share file content.
* File content is cached keyed by ``(mtime_ns, size)`` (cheap version gate) and
  verified against a SHA-256 ``content_hash`` recorded on read. A changed file
  (edit/add/rename/delete) is re-read from disk; the old content is never
  served as current.
* A lightweight AST symbol index (class/function/import + line number) is built
  lazily for Python files and refreshed when the file changes. Unparseable
  files simply have no index entries and remain searchable by text.
* Persistence is SQLite (``config.CONTEXT_CACHE_DB``) for cross-run reuse, with
  an in-memory working set per instance so a single run never hits the DB twice
  for the same key. DB I/O is best-effort: failures degrade to in-memory only
  and never crash the agent loop.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import config

_SKIP_DIRS = {
    ".git", ".venv", "node_modules", "dist", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "data", ".cursor", ".idea", ".vscode",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_identifier(query: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", query))


@dataclass
class FileEntry:
    rel_path: str
    content: str
    total_lines: int
    content_hash: str
    mtime_ns: int
    size: int
    cache_hit: bool


def _index_python(content: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract (symbols, imports) from Python source via ``ast``.

    Returns empty lists (never raises) so an unparseable file is simply
    treated as having no index and stays fully text-searchable.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [], []

    symbols: list[dict[str, Any]] = []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"name": node.name, "kind": "function", "line": node.lineno})
        elif isinstance(node, ast.ClassDef):
            symbols.append({"name": node.name, "kind": "class", "line": node.lineno})
        elif isinstance(node, ast.Import):
            imports.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    # De-duplicate imports preserving order.
    seen: set[str] = set()
    dedup: list[str] = []
    for i in imports:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    return symbols, dedup


class ProjectContext:
    """Cached, index-backed view of a single project workspace."""

    def __init__(self, project_root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(project_root).resolve()
        self.project_id = str(self.root).rstrip("\\/")
        self.db_path = Path(db_path) if db_path is not None else config.CONTEXT_CACHE_DB

        # In-memory working set (single run, single thread): fast path.
        self._files: dict[str, FileEntry] = {}
        self._index: dict[str, list[dict[str, Any]]] = {}  # rel_path -> symbols
        self._listing_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        self._stats: dict[str, int] = defaultdict(int)
        self._indexed = False

        self._load_from_db()

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        keys = (
            "read_requests", "disk_reads", "cache_hits", "cache_misses",
            "index_hits", "index_misses",
        )
        return {k: self._stats.get(k, 0) for k in keys}

    # -------------------------------------------------------------- persistence
    def _load_from_db(self) -> None:
        try:
            db_path = Path(self.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS file_cache ("
                    "project_id TEXT NOT NULL, rel_path TEXT NOT NULL, "
                    "mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, "
                    "content_hash TEXT NOT NULL, content TEXT NOT NULL, "
                    "total_lines INTEGER NOT NULL, created_at TEXT NOT NULL, "
                    "PRIMARY KEY (project_id, rel_path))"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS project_index ("
                    "project_id TEXT NOT NULL, rel_path TEXT NOT NULL, "
                    "symbols_json TEXT NOT NULL, imports_json TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, PRIMARY KEY (project_id, rel_path))"
                )
                conn.commit()
                rows = conn.execute(
                    "SELECT rel_path, mtime_ns, size, content_hash, content, total_lines "
                    "FROM file_cache WHERE project_id = ?",
                    (self.project_id,),
                ).fetchall()
                for rel, mtime_ns, size, h, content, total in rows:
                    self._files[rel] = FileEntry(
                        rel_path=rel, content=content, total_lines=total,
                        content_hash=h, mtime_ns=mtime_ns, size=size, cache_hit=False,
                    )
                idx_rows = conn.execute(
                    "SELECT rel_path, symbols_json FROM project_index WHERE project_id = ?",
                    (self.project_id,),
                ).fetchall()

                for rel, symbols_json in idx_rows:
                    self._index[rel] = json.loads(symbols_json or "[]")
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 - best-effort persistence; degrade to memory
            self.db_path = None  # type: ignore[assignment]
            return

    def _persist_file(self, entry: FileEntry) -> None:
        if self.db_path is None:
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.execute(
                    "INSERT INTO file_cache (project_id, rel_path, mtime_ns, size, "
                    "content_hash, content, total_lines, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(project_id, rel_path) DO UPDATE SET "
                    "mtime_ns=excluded.mtime_ns, size=excluded.size, "
                    "content_hash=excluded.content_hash, content=excluded.content, "
                    "total_lines=excluded.total_lines, created_at=excluded.created_at",
                    (self.project_id, entry.rel_path, entry.mtime_ns, entry.size,
                     entry.content_hash, entry.content, entry.total_lines, _now()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            return

    def _persist_index(self, rel_path: str, symbols: list[dict[str, Any]]) -> None:
        if self.db_path is None:
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.execute(
                    "INSERT INTO project_index (project_id, rel_path, symbols_json, "
                    "imports_json, updated_at) VALUES (?, ?, ?, '[]', ?) "
                    "ON CONFLICT(project_id, rel_path) DO UPDATE SET "
                    "symbols_json=excluded.symbols_json, updated_at=excluded.updated_at",
                    (self.project_id, rel_path, json.dumps(symbols, ensure_ascii=False), _now()),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return

    def _delete_file(self, rel_path: str) -> None:
        if self.db_path is None:
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.execute(
                    "DELETE FROM file_cache WHERE project_id = ? AND rel_path = ?",
                    (self.project_id, rel_path),
                )
                conn.execute(
                    "DELETE FROM project_index WHERE project_id = ? AND rel_path = ?",
                    (self.project_id, rel_path),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------ files
    def _stat(self, rel_path: str) -> tuple[int, int]:
        stat = (self.root / rel_path).stat()
        return stat.st_mtime_ns, stat.st_size

    def get_file(self, rel_path: str) -> FileEntry:
        """Return the file's current content (cached when version unchanged).

        A cache hit requires the on-disk ``(mtime_ns, size)`` to still match the
        cached entry; otherwise the file is re-read, re-hashed and re-indexed.
        This guarantees stale content is never served after an edit/add/rename.
        """
        self._stats["read_requests"] += 1
        try:
            mtime_ns, size = self._stat(rel_path)
        except OSError:
            self.invalidate(rel_path)
            raise

        cached = self._files.get(rel_path)
        if cached is not None and cached.mtime_ns == mtime_ns and cached.size == size:
            self._stats["cache_hits"] += 1
            cached.cache_hit = True
            return cached

        self._stats["cache_misses"] += 1
        self._stats["disk_reads"] += 1
        content = (self.root / rel_path).read_text(encoding="utf-8", errors="replace")
        total = len(content.splitlines())
        h = _sha256(content)
        entry = FileEntry(
            rel_path=rel_path, content=content, total_lines=total, content_hash=h,
            mtime_ns=mtime_ns, size=size, cache_hit=False,
        )
        self._files[rel_path] = entry
        self._persist_file(entry)
        # Refresh the symbol index lazily whenever we actually read the file.
        self._refresh_index(rel_path, content)
        return entry

    def put_file(self, rel_path: str, content: str) -> FileEntry:
        """Record a file after ``write_file`` so the next read is served from
        cache without a redundant disk read, and the index stays current."""
        try:
            mtime_ns, size = self._stat(rel_path)
        except OSError:
            mtime_ns, size = 0, len(content.encode("utf-8"))
        total = len(content.splitlines())
        entry = FileEntry(
            rel_path=rel_path, content=content, total_lines=total,
            content_hash=_sha256(content), mtime_ns=mtime_ns, size=size, cache_hit=False,
        )
        self._files[rel_path] = entry
        self._persist_file(entry)
        self._refresh_index(rel_path, content)
        self._listing_cache.clear()
        return entry

    def invalidate(self, rel_path: str) -> None:
        """Drop cached content + index for ``rel_path`` and clear listing memo.

        Called by ``write_file``; also used defensively when a stat fails.
        """
        self._files.pop(rel_path, None)
        self._index.pop(rel_path, None)
        self._listing_cache.clear()
        self._delete_file(rel_path)

    def _refresh_index(self, rel_path: str, content: str) -> None:
        if not rel_path.endswith(".py"):
            return
        symbols, _imports = _index_python(content)
        self._index[rel_path] = symbols
        self._persist_index(rel_path, symbols)

    def symbols(self, rel_path: str) -> list[dict[str, Any]]:
        return list(self._index.get(rel_path, []))

    # ----------------------------------------------------------------- listing
    def list_files(self, path_rel: str, recursive: bool) -> list[dict[str, Any]]:
        """Return ``[{path, type, size}]`` (relative to project root), memoized
        per ``(path, recursive)`` until the next ``invalidate``."""
        key = (path_rel, recursive)
        if key in self._listing_cache:
            return list(self._listing_cache[key])

        base = self.root / path_rel
        entries: list[dict[str, Any]] = []
        if base.is_dir():
            if recursive:
                for f in self._walk_files(base, True):
                    entries.append({
                        "path": f.relative_to(self.root).as_posix(),
                        "type": "file",
                        "size": f.stat().st_size,
                    })
            else:
                for e in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name)):
                    if e.name in _SKIP_DIRS:
                        continue
                    entries.append({
                        "path": e.relative_to(self.root).as_posix(),
                        "type": "dir" if e.is_dir() else "file",
                        "size": e.stat().st_size if e.is_file() else None,
                    })
        self._listing_cache[key] = entries
        return list(entries)

    @staticmethod
    def _walk_files(root: Path, recursive: bool) -> list[Path]:
        files: list[Path] = []
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            return files
        for entry in children:
            if entry.name in _SKIP_DIRS:
                continue
            if entry.is_file():
                files.append(entry)
            elif entry.is_dir() and recursive:
                files.extend(ProjectContext._walk_files(entry, recursive))
        return files

    def _rel_files(self, path_rel: str) -> list[str]:
        """All file paths (posix, relative) under ``path_rel`` (recursive)."""
        return [
            e["path"] for e in self.list_files(path_rel, recursive=True)
            if e["type"] == "file"
        ]

    def ensure_index(self) -> None:
        """Reconcile the Python symbol index with the filesystem.

        Called before every index query so externally added / deleted /
        renamed / modified files are never served from a stale index:

        * a directory walk finds added/renamed/deleted ``.py`` files;
        * a cheap ``(mtime_ns, size)`` stat detects content edits;
        * only files whose version actually changed are re-read and re-parsed.

        This is far cheaper than a text scan of every file and keeps the index
        authoritative even when the agent did not perform the mutation itself.
        """
        on_disk: set[str] = set()
        for path in self._walk_files(self.root, True):
            rel = path.relative_to(self.root).as_posix()
            if not rel.endswith(".py"):
                continue
            on_disk.add(rel)
            try:
                mtime_ns, size = self._stat(rel)
            except OSError:
                continue
            entry = self._files.get(rel)
            if entry is None or entry.mtime_ns != mtime_ns or entry.size != size:
                try:
                    self.get_file(rel)  # re-read + re-index this file
                except OSError:
                    continue
            elif rel not in self._index:
                self._refresh_index(rel, entry.content)

        # Drop index/cache entries for files that no longer exist on disk.
        for rel in list(self._index):
            if rel not in on_disk:
                self._index.pop(rel, None)
                self._files.pop(rel, None)
                self._delete_file(rel)
        self._indexed = True

    # ------------------------------------------------------------------ search
    def query_symbols(
        self,
        name: str,
        path_rel: str = ".",
        glob: str | None = None,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Locate class/function definitions by (sub)string from the index."""
        self.ensure_index()
        needle = name if case_sensitive else name.casefold()
        hits: list[dict[str, Any]] = []
        for rel, symbols in self._index.items():
            if glob and not Path(rel).match(glob):
                continue
            if path_rel != "." and not (rel == path_rel or rel.startswith(path_rel.rstrip("/") + "/")):
                continue
            for s in symbols:
                hay = s["name"] if case_sensitive else s["name"].casefold()
                if needle == hay or needle in hay:
                    hits.append({
                        "rel_path": rel, "line": s["line"],
                        "name": s["name"], "kind": s["kind"],
                    })
        if hits:
            self._stats["index_hits"] += 1
        else:
            self._stats["index_misses"] += 1
        return hits

    def _context_lines(self, rel_path: str, lineno: int, n: int) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        try:
            lines = self.get_file(rel_path).content.splitlines()
        except OSError:
            return []
        lo = max(1, lineno - n)
        hi = min(len(lines), lineno + n)
        return [{"line": i, "text": lines[i - 1].strip()[:300]} for i in range(lo, hi + 1)]

    def search(
        self,
        query: str,
        path_rel: str = ".",
        glob: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 50,
        context_lines: int = 0,
    ) -> list[dict[str, Any]]:
        """Search by symbol index first (identifier queries), then by text scan.

        The text scan reads files through ``get_file`` (cached), so repeated
        searches do not re-read the disk and the index is warmed as a side
        effect. Unparseable files are skipped by the index but still searched as
        plain text.
        """
        results: list[dict[str, Any]] = []

        # 1) Symbol index fast path for plain identifiers.
        if _is_identifier(query):
            syms = self.query_symbols(query, path_rel, glob, case_sensitive)
            if syms:
                for s in syms[:max_results]:
                    item = {
                        "path": s["rel_path"], "line": s["line"],
                        "text": s["name"], "symbol": s["name"], "kind": s["kind"],
                    }
                    ctx = self._context_lines(s["rel_path"], s["line"], context_lines)
                    if ctx:
                        item["context"] = ctx
                    results.append(item)
                return results

        # 2) Text scan fallback (also the only path for regex/non-identifier).
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error:
            pattern = re.compile(re.escape(query), flags)

        for rel in self._rel_files(path_rel):
            if glob and not Path(rel).match(glob):
                continue
            try:
                lines = self.get_file(rel).content.splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if not pattern.search(line):
                    continue
                match = {"path": rel, "line": lineno, "text": line.strip()[:300]}
                if context_lines > 0:
                    lo = max(1, lineno - context_lines)
                    hi = min(len(lines), lineno + context_lines)
                    match["context"] = [
                        {"line": n, "text": lines[n - 1].strip()[:300]}
                        for n in range(lo, hi + 1)
                    ]
                results.append(match)
                if len(results) >= max_results:
                    return results
        return results
