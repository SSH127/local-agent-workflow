"""Redirect known caches/artifacts into a per-subtask scratch directory and
detect (and restore) unintended writes outside the running write_scope.

Used by ``run_project_command`` when a subtask has a write_scope and the
classified command is a *cache writer* (pytest / mypy / npm test|build /
npx tsc / coverage / compileall / unittest). Direct formatters and dependency
installs keep declaring their ``writes`` and are checked up-front instead.

Changes under ``ARTIFACT_PATH_GLOBS`` are ephemeral: handled by
``ARTIFACT_POLICY`` (allow | restore | fail) and never silently skipped from
inventory when restore is requested.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import config
from app.write_scope import path_in_pattern, path_in_scope

# Hard skips: never inventory (vendor / VCS / our scratch). Artifact dirs are
# inventoried so ARTIFACT_POLICY=restore can put them back.
_HARD_SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules", "data", ".agent_scratch",
}

SCRATCH_ROOT_NAME = ".agent_scratch"


@dataclass
class ScratchSession:
    """One command invocation's redirected scratch + pre-command inventory."""

    run_id: str
    subtask_id: str
    workspace: Path
    scratch: Path
    scope_patterns: list[str]
    # path_posix → (mtime_ns, size, sha256 or None)
    pre_files: dict[str, tuple[int, int, str | None]] = field(default_factory=dict)
    pre_bytes: dict[str, bytes] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    kind: str = ""


@dataclass
class SandboxViolation:
    path: str
    action: str  # created | modified | restored | deleted_new | allowed_artifact
    message: str
    kind: str = "scope_violation"  # or "allowed_artifact"


@dataclass
class FinishResult:
    violations: list[SandboxViolation] = field(default_factory=list)
    artifacts: list[SandboxViolation] = field(default_factory=list)


def scratch_dir(workspace: Path, run_id: str, subtask_id: str) -> Path:
    safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)[:64]
    safe_sid = "".join(c if c.isalnum() or c in "-_." else "_" for c in subtask_id)[:64]
    return Path(workspace).resolve() / SCRATCH_ROOT_NAME / safe_run / safe_sid


def redirect_env(kind: str, scratch: Path) -> dict[str, str]:
    """Environment variables that redirect known caches into ``scratch``."""
    scratch.mkdir(parents=True, exist_ok=True)
    pytest_cache = scratch / "pytest_cache"
    mypy_cache = scratch / "mypy_cache"
    pycache = scratch / "pycache"
    tmp = scratch / "tmp"
    npm_cache = scratch / "npm_cache"
    tsbuild = scratch / "tsbuildinfo"
    coverage = scratch / "coverage"
    for d in (pytest_cache, mypy_cache, pycache, tmp, npm_cache, coverage):
        d.mkdir(parents=True, exist_ok=True)
    env = {
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "PYTHONPYCACHEPREFIX": str(pycache),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if kind in {"pytest", "coverage", "unittest", "python_cache"}:
        env["PYTEST_CACHE_DIR"] = str(pytest_cache)
    if kind in {"mypy", "python_cache"}:
        env["MYPY_CACHE_DIR"] = str(mypy_cache)
    if kind in {"npm", "tsc"}:
        env["npm_config_cache"] = str(npm_cache)
        env["NPM_CONFIG_CACHE"] = str(npm_cache)
        env["npm_config_prefer_offline"] = "true"
        env["VITE_CACHE_DIR"] = str(scratch / "vite_cache")
        (scratch / "vite_cache").mkdir(exist_ok=True)
        env["TURBO_CACHE_DIR"] = str(scratch / "turbo_cache")
        (scratch / "turbo_cache").mkdir(exist_ok=True)
        env["TSBUILDINFO"] = str(tsbuild / "tsconfig.tsbuildinfo")
        tsbuild.mkdir(exist_ok=True)
    if kind == "coverage":
        env["COVERAGE_FILE"] = str(coverage / ".coverage")
    return env


def is_artifact_path(rel: str, globs: list[str] | None = None) -> bool:
    """True when ``rel`` matches configured artifact/cache path globs."""
    patterns = globs if globs is not None else list(config.ARTIFACT_PATH_GLOBS)
    if any(path_in_pattern(p, rel) for p in patterns):
        return True
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    for i, part in enumerate(parts):
        if part.endswith(".egg-info"):
            return True
        if part == "node_modules" and i + 1 < len(parts) and parts[i + 1] == ".cache":
            return True
    return False


def _should_skip_dir(name: str) -> bool:
    return name in _HARD_SKIP_DIR_NAMES


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        base = Path(dirpath)
        for name in filenames:
            yield base / name


def _file_sig(path: Path, *, capture_bytes: bool, max_bytes: int = 1_000_000
              ) -> tuple[int, int, str | None, bytes | None]:
    st = path.stat()
    data: bytes | None = None
    digest: str | None = None
    # Always hash small files so same-size content edits are detected even when
    # mtime resolution is coarse; only keep bytes when capture_bytes is set.
    if st.st_size <= max_bytes:
        try:
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if capture_bytes:
                data = blob
        except OSError:
            data = None
            digest = None
    return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size), digest, data


def take_inventory(
    workspace: Path,
    scope_patterns: list[str],
    scratch: Path,
    *,
    capture_bytes: bool = True,
) -> tuple[dict[str, tuple[int, int, str | None]], dict[str, bytes]]:
    """Inventory files outside write_scope and outside scratch (incl. artifacts)."""
    root = workspace.resolve()
    scratch_res = scratch.resolve()
    files: dict[str, tuple[int, int, str | None]] = {}
    blobs: dict[str, bytes] = {}
    for path in _iter_files(root):
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        try:
            resolved.relative_to(scratch_res)
            continue
        except ValueError:
            pass
        if path_in_scope(scope_patterns, rel):
            continue
        mtime, size, digest, data = _file_sig(path, capture_bytes=capture_bytes)
        files[rel] = (mtime, size, digest)
        if data is not None:
            blobs[rel] = data
    return files, blobs



def check_argv_outside_workspace(
    argv: list[str],
    workspace: Path,
    scratch: Path | None = None,
) -> str | None:
    """Refuse absolute argv paths that escape workspace (+ optional scratch).

    Relative paths are left alone (resolved against cwd inside workspace by the
    command itself). Returns an error message or None.
    """
    root = Path(workspace).resolve()
    allowed = [root]
    if scratch is not None:
        allowed.append(Path(scratch).resolve())
    for raw in argv:
        if not raw or raw.startswith("-"):
            continue
        # Skip flag values that look like options (=) or bare names.
        candidate = raw
        if "=" in raw and not raw.startswith("/"):
            # e.g. cache_dir=/tmp/x — check the value side when absolute.
            _, _, rhs = raw.partition("=")
            if rhs.startswith("/") or (len(rhs) > 1 and len(rhs) > 1 and rhs[1] == ":"):
                candidate = rhs
            else:
                continue
        is_abs = candidate.startswith("/") or (
            len(candidate) > 1 and candidate[1] == ":" and candidate[0].isalpha()
        )
        if not is_abs:
            continue
        try:
            resolved = Path(candidate).resolve()
        except OSError:
            continue
        if any(_is_under(resolved, base) for base in allowed):
            continue
        return (
            f"命令参数包含工作区外的绝对路径：{candidate}。"
            "请改用工作区内的相对路径（COMMAND_SOFT_SANDBOX）。"
        )
    return None


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False



def begin_session(
    *,
    workspace: Path,
    run_id: str,
    subtask_id: str,
    scope_patterns: list[str],
    kind: str,
) -> ScratchSession:
    scratch = scratch_dir(workspace, run_id, subtask_id)
    env = redirect_env(kind, scratch)
    pre_files, pre_bytes = take_inventory(workspace, scope_patterns, scratch, capture_bytes=True)
    return ScratchSession(
        run_id=run_id, subtask_id=subtask_id, workspace=Path(workspace).resolve(),
        scratch=scratch, scope_patterns=list(scope_patterns),
        pre_files=pre_files, pre_bytes=pre_bytes, env=env, kind=kind,
    )


def _under_scratch(workspace: Path, scratch: Path, rel: str) -> bool:
    try:
        (workspace / rel).resolve().relative_to(scratch.resolve())
        return True
    except (OSError, ValueError):
        return False


def _restore_or_delete(
    session: ScratchSession, rel: str, *, existed: bool,
) -> tuple[str, str, bool]:
    """Restore from snapshot or delete a new file. Returns (action, msg, ok)."""
    root = session.workspace
    target = root / rel
    if not existed:
        try:
            if target.is_file():
                target.unlink()
            return "deleted_new", f"已删除范围外新建文件 {rel}。", True
        except OSError as exc:
            return "created", f"新建文件 {rel}（删除失败：{exc}）。", False
    blob = session.pre_bytes.get(rel)
    if blob is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            return "restored", f"已从命令前快照恢复 {rel}。", True
        except OSError as exc:
            return "modified", f"修改了 {rel}（恢复失败：{exc}）。", False
    return "modified", f"修改了 {rel}（无可用快照，未能恢复）。", False


def finish_session(session: ScratchSession) -> FinishResult:
    """Diff workspace vs pre-inventory; apply artifact policy + restore escapes."""
    root = session.workspace
    post, _ = take_inventory(root, session.scope_patterns, session.scratch, capture_bytes=False)
    policy = (config.ARTIFACT_POLICY or "allow").lower()
    result = FinishResult()

    def handle_change(rel: str, *, existed: bool, changed: bool) -> None:
        if not changed and existed:
            return
        artifact = is_artifact_path(rel)
        if artifact and policy != "fail":
            if policy == "restore":
                action, msg, _ok = _restore_or_delete(session, rel, existed=existed)
                result.artifacts.append(SandboxViolation(
                    rel, action,
                    f"产物路径 {rel}（ARTIFACT_POLICY=restore）：{msg}",
                    kind="allowed_artifact",
                ))
            else:
                # allow: leave in place, informational only
                action = "allowed_artifact"
                result.artifacts.append(SandboxViolation(
                    rel, action,
                    f"产物路径 {rel} 有变更，已按 ARTIFACT_POLICY=allow 忽略（不计入 scope_violation）。",
                    kind="allowed_artifact",
                ))
            return
        # Real escape (or artifact with policy=fail)
        action, msg, _ok = _restore_or_delete(session, rel, existed=existed)
        prefix = "命令在写入范围外"
        if not existed:
            full = f"{prefix}新建了文件 {rel}" + ("，已删除。" if action == "deleted_new" else f"（{msg}）")
        else:
            full = f"{prefix}修改了文件 {rel}" + (
                "，已从命令前快照恢复。" if action == "restored" else f"（{msg}）"
            )
        if artifact and policy == "fail":
            full = f"产物路径 {rel}（ARTIFACT_POLICY=fail）：" + full
        result.violations.append(SandboxViolation(rel, action, full, kind="scope_violation"))

    for rel, (mtime, size, digest) in post.items():
        if _under_scratch(root, session.scratch, rel):
            continue
        if path_in_scope(session.scope_patterns, rel):
            continue
        prev = session.pre_files.get(rel)
        if prev is None:
            handle_change(rel, existed=False, changed=True)
            continue
        prev_mtime, prev_size, prev_digest = prev
        changed = False
        if digest and prev_digest:
            changed = digest != prev_digest
        else:
            changed = (mtime, size) != (prev_mtime, prev_size)
        if changed:
            handle_change(rel, existed=True, changed=True)

    for rel, prev in session.pre_files.items():
        if rel in post:
            continue
        if path_in_scope(session.scope_patterns, rel):
            continue
        # Deleted outside scope
        artifact = is_artifact_path(rel)
        if artifact and policy == "allow":
            result.artifacts.append(SandboxViolation(
                rel, "allowed_artifact",
                f"产物路径 {rel} 被删除，已按 ARTIFACT_POLICY=allow 忽略。",
                kind="allowed_artifact",
            ))
            continue
        if artifact and policy == "restore":
            action, msg, _ok = _restore_or_delete(session, rel, existed=True)
            result.artifacts.append(SandboxViolation(
                rel, action, f"产物路径 {rel}（restore）：{msg}", kind="allowed_artifact"))
            continue
        # fail policy for artifact, or real source delete
        action, msg, _ok = _restore_or_delete(session, rel, existed=True)
        result.violations.append(SandboxViolation(
            rel, action,
            f"命令删除了写入范围外的文件 {rel}" + (
                "，已从命令前快照恢复。" if action == "restored" else f"（{msg}）"
            ),
            kind="scope_violation",
        ))
    return result


def violations_as_evidence(
    items: list[SandboxViolation], *, tool: str, scope: list[str], subtask_id: str | None,
) -> list[dict[str, Any]]:
    from datetime import UTC, datetime
    at = datetime.now(UTC).isoformat()
    return [
        {
            "type": v.kind,  # scope_violation | allowed_artifact
            "tool": tool,
            "path": v.path,
            "scope": list(scope),
            "subtask_id": subtask_id,
            "message": v.message,
            "action": v.action,
            "at": at,
        }
        for v in items
    ]
