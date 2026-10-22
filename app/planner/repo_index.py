"""Fast, dependency-free repository indexer used to ground the planner.

Python files are parsed with ``ast`` (modules, classes, functions with
signatures and line numbers, imports, approximate call edges by name).
TS/JS files are scanned with regexes (exported functions/components, imports).
Results are cached per file by (mtime, size).
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "dist", "build", ".idea", ".vscode", "data",
    ".next", "coverage", ".tox",
}
PROTECTED_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
PY_EXT = {".py"}
JS_EXT = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
MAX_FILE_BYTES = 400_000
MAX_FILES = 3000


@dataclass
class Symbol:
    name: str
    kind: str              # class | function | method | component
    line: int
    end_line: int
    signature: str
    qualname: str
    calls: list[str] = field(default_factory=list)
    doc: str = ""


@dataclass
class FileIndex:
    path: str               # posix relative path
    lang: str
    mtime: float
    size: int
    doc: str = ""
    imports: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    error: str | None = None


def _is_protected(name: str) -> bool:
    low = name.lower()
    return low in PROTECTED_NAMES or (low.startswith(".env") and low != ".env.example") \
        or low.endswith((".pem", ".key"))


def iter_source_files(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in PY_EXT and ext not in JS_EXT:
                continue
            if _is_protected(fn):
                continue
            p = Path(dirpath) / fn
            yield p
            count += 1
            if count >= MAX_FILES:
                return


def _call_name(node: ast.AST) -> str | None:
    f = getattr(node, "func", None)
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _calls_in(node: ast.AST) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            n = _call_name(sub)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:  # noqa: BLE001
        args = "..."
    ret = ""
    if node.returns is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:  # noqa: BLE001
            ret = ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({args}){ret}"
    return sig if len(sig) <= 200 else sig[:197] + "..."


def _first_doc_line(node: Any) -> str:
    try:
        d = ast.get_docstring(node) or ""
    except Exception:  # noqa: BLE001
        d = ""
    return d.strip().splitlines()[0][:160] if d.strip() else ""


def index_python(text: str, fi: FileIndex) -> None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        fi.error = f"syntax: {exc}"
        return
    fi.doc = _first_doc_line(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            fi.imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            fi.imports.append(mod)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fi.symbols.append(Symbol(node.name, "function", node.lineno,
                                     getattr(node, "end_lineno", node.lineno) or node.lineno,
                                     _signature(node), node.name, _calls_in(node),
                                     _first_doc_line(node)))
        elif isinstance(node, ast.ClassDef):
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:  # noqa: BLE001
                bases = ""
            fi.symbols.append(Symbol(node.name, "class", node.lineno,
                                     getattr(node, "end_lineno", node.lineno) or node.lineno,
                                     f"class {node.name}({bases})" if bases else f"class {node.name}",
                                     node.name, [], _first_doc_line(node)))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fi.symbols.append(Symbol(item.name, "method", item.lineno,
                                             getattr(item, "end_lineno", item.lineno) or item.lineno,
                                             _signature(item), f"{node.name}.{item.name}",
                                             _calls_in(item), _first_doc_line(item)))


_JS_IMPORT = re.compile(r"""(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\(\s*)['"]([^'"]+)['"]""")
_JS_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(([^)]*)\)",
    re.M)
_JS_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(([^)]*)\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>",
    re.M)
_JS_CLASS = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)
_JS_CALL = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_JS_KEYWORDS = {"if", "for", "while", "switch", "return", "function", "catch", "typeof", "await", "new"}


def index_js(text: str, fi: FileIndex) -> None:
    fi.imports = _JS_IMPORT.findall(text)
    starts: list[tuple[int, str, str, str]] = []
    for m in _JS_FUNC.finditer(text):
        starts.append((m.start(), m.group(1), "function", f"function {m.group(1)}({m.group(2).strip()})"))
    for m in _JS_ARROW.finditer(text):
        starts.append((m.start(), m.group(1), "function", f"const {m.group(1)} = ({(m.group(2) or '').strip()}) =>"))
    for m in _JS_CLASS.finditer(text):
        starts.append((m.start(), m.group(1), "class", f"class {m.group(1)}"))
    starts.sort()
    line_of = lambda pos: text.count("\n", 0, pos) + 1  # noqa: E731
    total = text.count("\n") + 1
    for i, (pos, name, kind, sig) in enumerate(starts):
        end_pos = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[pos:end_pos]
        calls = []
        for c in _JS_CALL.findall(body):
            if c not in _JS_KEYWORDS and c != name and c not in calls:
                calls.append(c)
        if kind == "function" and name[:1].isupper() and fi.lang in {"tsx", "jsx"}:
            kind = "component"
        end_line = line_of(end_pos) - 1 if i + 1 < len(starts) else total
        fi.symbols.append(Symbol(name, kind, line_of(pos), max(line_of(pos), end_line), sig[:200], name, calls[:40]))


_CACHE: dict[str, FileIndex] = {}


def index_file(path: Path, root: Path) -> FileIndex | None:
    try:
        st = path.stat()
    except OSError:
        return None
    rel = path.relative_to(root).as_posix()
    key = str(path)
    cached = _CACHE.get(key)
    if cached and cached.mtime == st.st_mtime and cached.size == st.st_size:
        return cached
    ext = path.suffix.lower()
    fi = FileIndex(rel, "python" if ext in PY_EXT else ext.lstrip("."), st.st_mtime, st.st_size)
    if st.st_size > MAX_FILE_BYTES:
        fi.error = "too large"
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            fi.error = str(exc)
            text = ""
        if text:
            if ext in PY_EXT:
                index_python(text, fi)
            else:
                index_js(text, fi)
    _CACHE[key] = fi
    return fi


@dataclass
class RepoIndex:
    root: Path
    files: dict[str, FileIndex]

    def all_paths(self) -> list[str]:
        return sorted(self.files)

    def module_to_path(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel in self.files:
            if rel.endswith(".py"):
                mod = rel[:-3].replace("/", ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                out[mod] = rel
        return out

    def resolve_import(self, src: str, imp: str) -> str | None:
        if src.endswith(".py"):
            m2p = self.module_to_path()
            mod = imp
            if imp.startswith("."):
                level = len(imp) - len(imp.lstrip("."))
                base = src[:-3].split("/")[:-1]
                if level > 1:
                    base = base[: len(base) - (level - 1)]
                rest = imp.lstrip(".")
                mod = ".".join(base + ([rest] if rest else []))
            parts = mod.split(".")
            while parts:
                cand = ".".join(parts)
                if cand in m2p:
                    return m2p[cand]
                parts.pop()
            return None
        if not imp.startswith("."):
            return None
        base = (Path(src).parent / imp).as_posix()
        norm = os.path.normpath(base).replace("\\", "/")
        for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
            if norm + ext in self.files:
                return norm + ext
        return None

    def import_graph(self) -> dict[str, set[str]]:
        g: dict[str, set[str]] = {}
        for rel, fi in self.files.items():
            deps = set()
            for imp in fi.imports:
                t = self.resolve_import(rel, imp)
                if t and t != rel:
                    deps.add(t)
            g[rel] = deps
        return g

    def definitions(self) -> dict[str, list[tuple[str, Symbol]]]:
        d: dict[str, list[tuple[str, Symbol]]] = {}
        for rel, fi in self.files.items():
            for s in fi.symbols:
                d.setdefault(s.name, []).append((rel, s))
        return d

    def callers_of(self, name: str, limit: int = 20) -> list[tuple[str, Symbol]]:
        out = []
        for rel, fi in sorted(self.files.items()):
            for s in fi.symbols:
                if name in s.calls and s.name != name:
                    out.append((rel, s))
                    if len(out) >= limit:
                        return out
        return out


def build_index(workspace: str | Path) -> RepoIndex:
    root = Path(workspace).resolve()
    files: dict[str, FileIndex] = {}
    for p in iter_source_files(root):
        fi = index_file(p, root)
        if fi is not None:
            files[fi.path] = fi
    return RepoIndex(root, files)
