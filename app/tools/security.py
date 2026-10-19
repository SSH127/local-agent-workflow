"""Security primitives shared by all agent tools.

- Path normalization and workspace boundary enforcement.
- Secret redaction for tool arguments and outputs.
- Command policy (allowlist + deny-list) and approval classification.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from app import config
from app.tools.command_policy import CommandDecision, classify
from app.tools.errors import SecurityError  # re-exported for existing callers

# ---------------------------------------------------------------------------
# Path boundaries
# ---------------------------------------------------------------------------

def resolve_workspace_path(
    path: str | os.PathLike[str],
    workspace: str | os.PathLike[str] | None = None,
) -> Path:
    """Normalize ``path`` relative to the workspace and reject escapes.

    Relative paths are interpreted against ``workspace`` (or the process-wide
    ``config.WORKSPACE_PATH`` when ``workspace`` is None); absolute paths must
    still live inside that workspace after resolving symlinks. The workspace is
    a parameter, never mutated, so concurrent tasks can safely use different
    workspaces without touching global state.
    """
    root = Path(workspace) if workspace is not None else config.WORKSPACE_PATH
    # Resolve the root itself too, so the containment check always compares two
    # canonical paths (defense-in-depth against a non-canonical workspace root
    # that contains symlinks or ``..`` segments).
    root = root.resolve()
    raw = Path(path)
    candidate = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:  # path escapes the workspace
        raise SecurityError(
            f"路径越界：不允许访问工作区之外的路径（{path}）。"
        ) from exc

    return candidate


def resolve_workspace_root(path: str | os.PathLike[str]) -> Path:
    """Validate a user-supplied workspace root and return its resolved path.

    Rejects non-existent paths, non-directories, the filesystem root and known
    system directories, so a user cannot nominate an arbitrary system location
    as a workspace and then have the agent reach files inside it.
    """
    raw = Path(path)
    try:
        resolved = raw.expanduser().resolve()
    except OSError as exc:
        raise SecurityError(f"工作区路径无效：{path}（{exc}）") from exc
    if not resolved.exists():
        raise SecurityError(f"工作区路径不存在：{resolved}")
    if not resolved.is_dir():
        raise SecurityError(f"工作区路径不是目录：{resolved}")
    if resolved.parent == resolved:
        raise SecurityError(f"不允许将文件系统根目录作为工作区：{resolved}")
    norm = str(resolved).rstrip("\\/").casefold()
    for forbidden in config.FORBIDDEN_WORKSPACE_ROOTS:
        fb = str(Path(forbidden)).rstrip("\\/").casefold()
        if norm == fb:
            raise SecurityError(f"不允许将系统目录作为工作区：{resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Natural-language workspace detection
# ---------------------------------------------------------------------------

# Absolute-path tokens a user may write directly in the task goal. The intended
# workflow (like Cursor) is to *name the target project directory in prose*, so
# the API extracts the first explicit absolute path and binds the run to it.
# ``\\.server\\share`` is UNC; ``D:\\...`` / ``/...`` are local roots.
_QUOTED_PATH = re.compile(
    r'["\'`]([A-Za-z]:[\\/][^"\'\n`]+|/(?:[^"\'\n`]+/)*[^"\'\n`]+)["\'`]'
)
_UNC_PATH = re.compile(r'\\\\[^\s"\'`<>|]+(?:\\[^\s"\'`<>|]+)+')
_WIN_DRIVE_PATH = re.compile(r'(?<![\w])[A-Za-z]:[\\/][^\s"\'`<>|]+')
_POSIX_ABS_PATH = re.compile(r'(?<![\w/.])/(?:[^/\s`]+/)+[^/\s`]+')
# Trailing punctuation/fullwidth punctuation that commonly follows a path in
# Chinese prose and must not become part of the extracted path.
_TRAILING_PATH_PUNCT = '，。；：！？、,.;:!?)】」』}）"\'`]'


def detect_workspace(goal: str) -> str | None:
    """Detect an explicit target workspace from a natural-language task goal.

    Returns the fully-resolved, validated workspace directory as a string when
    the goal names an absolute path (``D:\\...``, ``/...`` or UNC — quoted or
    not), or ``None`` when no path is mentioned (the run then uses the default
    workspace). Raises :class:`SecurityError` when a path *is* mentioned but is
    invalid; callers must surface that to the user instead of silently falling
    back to another workspace.
    """
    if not goal:
        return None

    # Collect every candidate with its position so the *first* path mentioned in
    # the goal wins, regardless of which token shape matched it.
    spans: list[tuple[int, str]] = []
    for match in _QUOTED_PATH.finditer(goal):
        spans.append((match.start(), match.group(1)))
    for match in _UNC_PATH.finditer(goal):
        spans.append((match.start(), match.group(0)))
    for match in _WIN_DRIVE_PATH.finditer(goal):
        spans.append((match.start(), match.group(0)))
    for match in _POSIX_ABS_PATH.finditer(goal):
        spans.append((match.start(), match.group(0)))

    if not spans:
        return None

    spans.sort(key=lambda pair: pair[0])
    raw = spans[0][1].rstrip(_TRAILING_PATH_PUNCT)
    try:
        resolved = Path(raw).expanduser().resolve()
    except OSError as exc:
        raise SecurityError(f"工作区路径无效：{raw}（{exc}）") from exc

    if not resolved.exists():
        raise SecurityError(
            f"工作区路径不存在：{resolved}（任务目标中指定的目录不存在，请确认后重试。）"
        )
    # A goal may name a file inside the project (e.g. ``D:\\proj\\main.py``);
    # treat its containing directory as the project workspace.
    if resolved.is_file():
        resolved = resolved.parent
    return str(resolve_workspace_root(resolved))


def _protected_basenames() -> set[str]:
    """Case-folded filenames that are protected at any directory depth."""
    return {Path(entry).name.casefold() for entry in config.PROTECTED_FILES}


def is_protected_file(
    path: str | os.PathLike[str],
    workspace: str | os.PathLike[str] | None = None,
) -> bool:
    """Return True for files that must not be overwritten without approval.

    Protection is by *filename* (case-insensitive) at any directory depth, so
    ``.env``, ``.env.local``, ``.env.production`` and the configured database
    file are protected wherever they appear. Files that merely contain "env" in
    their name (e.g. ``environment.py``) are NOT matched. Paths that escape the
    workspace raise :class:`SecurityError` (fail-closed) via
    :func:`resolve_workspace_path`.
    """
    resolved = resolve_workspace_path(path, workspace)
    return resolved.name.casefold() in _protected_basenames()


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = (
    r"\b(api[_-]?key|apikey|token|secret|password|passwd|pwd)\b\s*[=:]\s*\S+",
    r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{8,}",
    r"\bbearer\s+[A-Za-z0-9._\-]{8,}",
    r"AKIA[0-9A-Z]{16}",
    r"(deepseek|openai)[_-]?api[_-]?key\s*[=:]\s*\S+",
)

_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS), flags=re.IGNORECASE)


def redact_text(text: str) -> str:
    """Replace common secrets in ``text`` with a placeholder."""
    if not text:
        return text
    return _SECRET_RE.sub("***REDACTED***", text)


def redact_kv(data: Any) -> Any:
    """Recursively redact sensitive keys/values in an arbitrary structure."""
    if isinstance(data, dict):
        return {
            key: "***REDACTED***" if _looks_like_secret_key(key) else redact_kv(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [redact_kv(item) for item in data]
    if isinstance(data, str):
        return redact_text(data)
    return data


def _looks_like_secret_key(key: str) -> bool:
    return bool(
        re.search(r"(?i)(api[_-]?key|apikey|token|secret|password|passwd|pwd|authorization)", key)
    )


# ---------------------------------------------------------------------------
# Command policy
# ---------------------------------------------------------------------------
# The policy itself (argument-level analysis: which code a command will execute)
# lives in ``app.tools.command_policy``. This module keeps the stable
# ``validate_command`` / ``requires_approval`` / classification API used by the
# tools and the executor. Program allowlists and subcommand rules live there so
# there is exactly one source of truth.
# ``ruff`` flags that make an otherwise read-only check mutate files.
_RUFF_FIX_FLAGS = {"--fix", "--fix-only", "--unsafe-fixes"}

# Modules whose run via ``python -m <module>`` / ``uv run <module>`` produce
# read-only pass/fail verification evidence. ``ruff`` is handled separately
# because ``ruff check`` verifies while ``ruff format`` mutates.
_VERIFYING_MODULES = {"pytest", "mypy", "coverage", "compileall", "unittest"}

# npm scripts treated as read-only pass/fail verification.
_VERIFYING_NPM_SCRIPTS = {"test", "build", "typecheck", "lint", "tsc"}


def resolve_effective_command(program: str, args: list[str]) -> tuple[str, list[str]]:
    """Resolve wrapper indirection to the real program and its own arguments.

    ``python -m X ...`` and ``uv run X ...`` delegate to ``X``, so mutation /
    verification classification must look through them to prevent a mutating
    command (e.g. ``uv run ruff format``) from bypassing the policy.
    ``coverage run -m pytest ...`` is also unwrapped, so the covered command
    keeps its verification/mutation classification.
    """
    program = (program or "").strip().lower()
    argv = [str(a) for a in args]
    if program == "python" and len(argv) >= 2 and argv[0] == "-m":
        return argv[1].lower(), argv[2:]
    if program == "uv" and len(argv) >= 2 and argv[0] == "run":
        return argv[1].lower(), argv[2:]
    if program == "coverage" and len(argv) >= 3 and argv[0] == "run" and argv[1] == "-m":
        return argv[2].lower(), argv[3:]
    return program, argv


def is_mutating_command(program: str, args: list[str]) -> bool:
    """True when the command rewrites project files (a *formatter/autofixer*).

    Formatters such as ``ruff format`` and ``ruff check --fix`` mutate source
    files, so (a) they always require approval and (b) their clean exit code is
    never treated as verification evidence — the executor records them as a
    mutation that a later real verification must confirm.
    """
    eff_prog, eff_args = resolve_effective_command(program, args)
    if eff_prog != "ruff":
        return False
    if not eff_args:
        return False
    sub = eff_args[0].lower()
    if sub == "format":
        return True
    if sub == "check":
        return any(flag in _RUFF_FIX_FLAGS for flag in eff_args[1:])
    return False


def is_verification_command(program: str, args: list[str]) -> bool:
    """True when the command is a *read-only* pass/fail verification.

    Formatters/autofixers are excluded: ``ruff check`` (without ``--fix``) is a
    verification, but ``ruff format`` and ``ruff check --fix`` are not.
    """
    eff_prog, eff_args = resolve_effective_command(program, args)
    if eff_prog == "ruff":
        if not eff_args or eff_args[0].lower() != "check":
            return False
        return not any(flag in _RUFF_FIX_FLAGS for flag in eff_args[1:])
    if eff_prog in _VERIFYING_MODULES:
        return True
    if eff_prog == "npx":
        return bool(eff_args) and eff_args[0].lower() == "tsc"
    if eff_prog == "npm":
        return (
            len(eff_args) >= 2
            and eff_args[0].lower() == "run"
            and eff_args[1].lower() in _VERIFYING_NPM_SCRIPTS
        )
    return False


def classify_command(
    program: str,
    args: list[str],
    cwd: str,
    workspace: str | os.PathLike[str] | None = None,
) -> CommandDecision:
    """Full classification (risk + approval) for one command invocation."""
    return classify(program, args, cwd, workspace)


def validate_command(
    program: str,
    args: list[str],
    cwd: str,
    workspace: str | os.PathLike[str] | None = None,
) -> tuple[str, bool]:
    """Validate a parameterized command.

    Returns ``(resolved_program, requires_approval)``. Raises ``SecurityError``
    when the command is disallowed. The command is structured (program + args),
    never a raw shell string. Argument-level analysis decides what code the
    command would actually execute, so wrappers such as ``python -m pytest``
    cannot smuggle an arbitrary script through the program allowlist.
    """
    decision = classify(program, args, cwd, workspace)
    return decision.program, decision.requires_approval


def requires_approval(
    program: str,
    args: list[str],
    cwd: str,
    workspace: str | os.PathLike[str] | None = None,
) -> bool:
    """Return True when a parameterized command needs explicit user approval."""
    _, requires = validate_command(program, args, cwd, workspace)
    return requires


def approval_reason(
    tool_name: str,
    args: dict[str, Any],
    workspace: str | os.PathLike[str] | None = None,
) -> str:
    """Human-readable risk reason shown to the user for an approval request.

    For commands this includes the concrete risk class (test execution scope,
    package installation, formatter mutation, ...), and the caller records the
    full command/args/cwd with the approval request itself.
    """
    if tool_name in {"write_file", "edit_file"}:
        if bool(args.get("allow_truncate", False)):
            return f"整体缩短/清空文件 {args.get('path', '')}（可能丢失内容）"
        return f"覆盖受保护文件 {args.get('path', '')}"
    if tool_name == "rollback_file":
        return f"回滚写入事务 {args.get('tx_id', '')}（还原文件到写入前版本）"
    if tool_name == "run_project_command":
        program = str(args.get("program", ""))
        argv = [str(a) for a in args.get("args", [])]
        cwd = str(args.get("cwd", "."))
        try:
            decision = classify_command(program, argv, cwd, workspace)
        except SecurityError:
            return f"执行命令：{program}"
        return decision.risk or f"执行命令：{program} {' '.join(argv)}"
    return "高风险操作"
