"""Write protection: version checks, snapshots, atomic replace, rollback.

``write_file`` / ``edit_file`` previously wrote in place with no recovery
point. A partial model output therefore replaced a 640-line file with 11 bytes
and a 506-line file with ~180 lines, with nothing to restore from.

This module gives every modifying tool one path:

1. **Inspect** the target and compare its content hash with the version the run
   actually read (or an explicit ``expected_sha256``). A mismatch means another
   process changed the file — the write is refused, never silently applied on
   top of stale content.
2. **Snapshot** the original bytes outside the workspace before touching it,
   together with a manifest (run, tool, hashes, sizes, mtime, pid).
3. **Commit** atomically: write a sibling temp file, fsync, re-check the
   target's ``(mtime_ns, size)``, then ``os.replace``. A crash leaves either the
   old or the new file, never a truncated one.
4. **Roll back** explicitly, but only when the file still matches what the
   write produced. A rollback never clobbers a newer external edit; it refuses
   and reports the conflict instead.

Snapshots live outside the run's workspace so the agent's own file tools cannot
read, overwrite or delete them. Protected files (``.env``) are snapshotted into
that same controlled directory with owner-only permissions, and their contents
never appear in tool output.

What this module deliberately does NOT do: auto-rollback on test failures. A
failing test is frequently the very problem being fixed, and reverting would
undo legitimate work. Only explicit rollback (or a write that never landed)
restores a file. Every transaction stays on disk as an audit record.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app import config

MANIFEST_NAME = "manifest.json"
ORIGINAL_NAME = "original.bin"
APPLIED_NAME = "applied.bin"

STATUS_PREPARED = "prepared"
STATUS_APPLIED = "applied"
STATUS_FAILED = "failed"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_ABANDONED = "abandoned"
STATUS_NEEDS_REVIEW = "needs_review"


class WriteGuardError(RuntimeError):
    """Raised for a write that cannot be safely prepared or committed."""


class WriteConflict(WriteGuardError):
    """Raised when a write would clobber a version the run never saw."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(text: str) -> str:
    """Hash of newline-normalized text (matches ``ProjectContext``)."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _fallback_root() -> Path:
    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / "local-agent-workflow" / "write_backups"
    return Path.home() / ".local" / "share" / "local-agent-workflow" / "write_backups"


def backup_root(workspace: str | Path | None = None) -> Path:
    """Resolved, workspace-external directory that holds write snapshots.

    If the configured directory ends up *inside* the run's workspace it would be
    reachable by ``read_file``/``write_file``, so it is replaced by an
    out-of-workspace fallback. This keeps snapshots (which may include
    protected-file contents) out of the agent's reach.
    """
    configured = Path(config.WRITE_BACKUP_DIR).expanduser().resolve()
    ws = Path(workspace if workspace is not None else config.WORKSPACE_PATH).resolve()
    try:
        configured.relative_to(ws)
    except ValueError:
        return configured
    fallback = _fallback_root().resolve()
    try:
        fallback.relative_to(ws)
    except ValueError:
        return fallback
    return (Path(tempfile.gettempdir()) / "law-write-backups").resolve()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@dataclass
class FileState:
    """The target's state immediately before a write (no side effects)."""

    abs_path: Path
    rel_path: str
    existed: bool
    raw: bytes
    text: str
    sha256: str | None
    size: int
    mtime_ns: int | None
    newline: str
    protected: bool

    @property
    def char_count(self) -> int:
        return len(self.text)


def inspect(path: Path, rel_path: str, workspace: Path, protected: bool) -> FileState:
    """Read the current version of ``path`` and describe it."""
    existed = path.exists()
    if existed and not path.is_file():
        raise WriteGuardError(f"不是文件：{path}")
    raw = path.read_bytes() if existed else b""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WriteGuardError("目标不是有效的 UTF-8 文本，无法安全写入。") from exc
    normalized = text.replace("\r\n", "\n")
    st = path.stat() if existed else None
    return FileState(
        abs_path=path,
        rel_path=rel_path,
        existed=existed,
        raw=raw,
        text=normalized,
        sha256=content_hash(normalized) if existed else None,
        size=st.st_size if st is not None else 0,
        mtime_ns=st.st_mtime_ns if st is not None else None,
        newline="\r\n" if "\r\n" in text else "\n",
        protected=protected,
    )


def check_expected_version(
    state: FileState, expected_sha256: str | None, *, source: str
) -> None:
    """Refuse when the on-disk version differs from what the run saw."""
    if not expected_sha256:
        return
    actual = state.sha256
    if actual == expected_sha256:
        return
    if actual is None:
        raise WriteConflict(
            f"{state.rel_path} 已被外部删除（任务读取时版本 {expected_sha256[:12]}…）。"
            "请重新读取后再写入。"
        )
    raise WriteConflict(
        f"{state.rel_path} 在读取后被外部修改（读取时 {expected_sha256[:12]}…，"
        f"当前 {actual[:12]}…）。为避免覆盖他人修改，本次{source}已中止；"
        "请先 read_file 重新确认内容再写入。"
    )


def check_shrink_guard(
    state: FileState, new_text: str, *, allow_truncate: bool
) -> None:
    """Refuse a whole-file rewrite that would destroy most of a large file."""
    if not state.existed or allow_truncate:
        return
    old_chars = state.char_count
    if old_chars > config.WRITE_SHRINK_GUARD_MIN_CHARS and (
        len(new_text) < old_chars * config.WRITE_SHRINK_GUARD_RATIO
    ):
        raise WriteGuardError(
            f"拒绝覆盖 {state.abs_path.name}：新内容仅 {len(new_text)} 字符，"
            f"而现有文件有 {old_chars} 字符。这通常意味着写入不完整。"
            "请改用 edit_file 做局部替换；若确实要整体缩短，"
            "请显式传入 allow_truncate=true（需要审批）。"
        )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


@dataclass
class WriteTransaction:
    """A prepared write: snapshot on disk + in-memory plan."""

    tx_id: str
    run_id: str
    tool: str
    state: FileState
    tx_dir: Path
    snapshot_path: Path | None
    started_at: str = field(default_factory=_utc_now)

    @property
    def manifest_path(self) -> Path:
        return self.tx_dir / MANIFEST_NAME

    def summary(self) -> dict[str, Any]:
        """Safe-to-log view: hashes and paths, never file contents."""
        return {
            "tx_id": self.tx_id,
            "run_id": self.run_id,
            "tool": self.tool,
            "path": self.state.rel_path,
            "created": not self.state.existed,
            "before_sha256": self.state.sha256,
            "before_size": self.state.size,
            "snapshot": str(self.snapshot_path) if self.snapshot_path else None,
            "tx_dir": str(self.tx_dir),
            "manifest": str(self.manifest_path),
            "protected": self.state.protected,
        }


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_manifest(tx_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((tx_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WriteGuardError(f"写入事务记录不可读：{tx_dir}（{exc}）") from exc


def _secure_dir(path: Path) -> None:
    """Create a private directory; snapshots may hold protected-file content."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass  # Windows: inherits the user-private ACL of the parent


def prepare(
    state: FileState,
    *,
    run_id: str,
    tool: str,
    workspace: Path,
) -> WriteTransaction:
    """Persist a recoverable snapshot before the target is modified."""
    tx_id = uuid4().hex
    root = backup_root(workspace)
    _secure_dir(root)
    tx_dir = root / (run_id or "unknown-run") / tx_id
    _secure_dir(tx_dir)

    snapshot_path: Path | None = None
    if state.existed:
        snapshot_path = tx_dir / ORIGINAL_NAME
        snapshot_path.write_bytes(state.raw)
        if state.protected:
            # Best effort on POSIX; Windows ACLs are inherited from the backup
            # root, which is outside the workspace and never exposed to tools.
            try:
                os.chmod(snapshot_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass

    tx = WriteTransaction(
        tx_id=tx_id, run_id=run_id, tool=tool, state=state,
        tx_dir=tx_dir, snapshot_path=snapshot_path,
    )
    _write_manifest(tx.manifest_path, {
        **tx.summary(),
        "abs_path": str(state.abs_path),
        "existed": state.existed,
        "mtime_ns": state.mtime_ns,
        "raw_size": len(state.raw),
        "status": STATUS_PREPARED,
        "pid": os.getpid(),
        "started_at": tx.started_at,
        "applied_at": None,
        "after_sha256": None,
        "after_size": None,
        "error": None,
        "rollback_status": None,
        "rolled_back_at": None,
    })
    return tx


def _atomic_write(path: Path, data: bytes, *, like: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".law-write-", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if like is not None:
            try:
                os.chmod(tmp, stat.S_IMODE(like.stat().st_mode))
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def commit(
    tx: WriteTransaction,
    new_text: str,
    *,
    project: Any = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Atomically replace the target and finalize the transaction record."""
    state = tx.state
    payload_text = new_text
    payload = payload_text.encode("utf-8")

    # Re-check immediately before the replace: if another process wrote the file
    # while we were preparing, refuse rather than clobber it.
    if state.existed:
        try:
            st = state.abs_path.stat()
        except OSError as exc:
            raise WriteConflict(f"目标文件在写入前消失（{exc}）；已中止。") from exc
        if (st.st_mtime_ns, st.st_size) != (state.mtime_ns, state.size):
            raise WriteConflict(
                f"{state.rel_path} 在写入过程中被其他进程修改，已中止写入，"
                "原文件未被改动。请重新读取后再试。"
            )
    elif state.abs_path.exists():
        raise WriteConflict(
            f"{state.rel_path} 在写入前被其他进程创建，已中止写入，避免覆盖新文件。"
        )

    _atomic_write(
        state.abs_path, payload,
        like=state.abs_path if state.existed else None,
    )

    after_hash = content_hash(payload_text)
    cache_text = payload_text.replace("\r\n", "\n")
    manifest = _read_manifest(tx.tx_dir)
    manifest.update({
        "status": STATUS_APPLIED,
        "applied_at": _utc_now(),
        "after_sha256": after_hash,
        "after_size": len(payload),
    })
    _write_manifest(tx.manifest_path, manifest)
    (tx.tx_dir / APPLIED_NAME).write_bytes(payload)

    if project is not None:
        # Cache newline-normalized text so the hash it computes matches the one
        # recorded below (the on-disk bytes may use CRLF).
        project.put_file(state.rel_path, cache_text)
    if tracker is not None:
        tracker.record_version(state.rel_path, after_hash)

    return {
        **tx.summary(),
        "status": STATUS_APPLIED,
        "after_sha256": after_hash,
        "after_size": len(payload),
    }


def abort(tx: WriteTransaction, error: str) -> None:
    """Record a failed write. The target was not modified (atomic replace)."""
    try:
        manifest = _read_manifest(tx.tx_dir)
    except WriteGuardError:
        return
    if manifest.get("status") == STATUS_APPLIED:
        return
    manifest.update({"status": STATUS_FAILED, "error": error[:2000]})
    try:
        _write_manifest(tx.manifest_path, manifest)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _find_tx(run_id: str, tx_id: str, workspace: Path) -> Path:
    root = backup_root(workspace)
    candidate = root / (run_id or "unknown-run") / tx_id
    if (candidate / MANIFEST_NAME).is_file():
        return candidate
    # Defensive: allow locating a transaction recorded under another run dir,
    # but never restore it unless its own run_id matches (checked by caller).
    for manifest in root.rglob(f"*/{tx_id}/{MANIFEST_NAME}"):
        return manifest.parent
    raise WriteGuardError(f"未找到写入事务：{tx_id}")


def tx_target(run_id: str, tx_id: str, workspace: str | Path) -> str | None:
    """Workspace-relative path a transaction would restore (for scope checks)."""
    ws = Path(workspace).resolve()
    try:
        manifest = _read_manifest(_find_tx(run_id, tx_id, ws))
        return Path(manifest["abs_path"]).resolve().relative_to(ws).as_posix()
    except (WriteGuardError, KeyError, ValueError, OSError):
        return None


def rollback(
    *,
    run_id: str,
    tx_id: str,
    workspace: str | Path,
    force: bool = False,
    project: Any = None,
    tracker: Any = None,
) -> dict[str, Any]:
    """Restore the pre-write snapshot, refusing to clobber newer edits."""
    ws = Path(workspace).resolve()
    tx_dir = _find_tx(run_id, tx_id, ws)
    manifest = _read_manifest(tx_dir)
    if manifest.get("run_id") != run_id:
        raise WriteGuardError("该写入事务不属于当前任务，拒绝回滚。")

    abs_path = Path(manifest["abs_path"]).resolve()
    try:
        abs_path.relative_to(ws)
    except ValueError as exc:
        raise WriteGuardError("快照目标已不在当前工作区内，拒绝回滚。") from exc

    if manifest.get("status") == STATUS_ROLLED_BACK:
        return {"tx_id": tx_id, "status": STATUS_ROLLED_BACK, "already": True}

    before_hash = manifest.get("before_sha256")
    after_hash = manifest.get("after_sha256")
    if after_hash is None:
        raise WriteGuardError("该写入事务未成功提交（无修改后版本），无需回滚。")

    current_text = (
        abs_path.read_text(encoding="utf-8") if abs_path.is_file() else None
    )
    current_hash = content_hash(current_text) if current_text is not None else None

    if current_hash == before_hash:
        manifest.update({"status": STATUS_ROLLED_BACK, "rolled_back_at": _utc_now(),
                         "rollback_status": "already_restored"})
        _write_manifest(tx_dir / MANIFEST_NAME, manifest)
        return {"tx_id": tx_id, "status": STATUS_ROLLED_BACK, "already": True}

    if current_hash != after_hash and not force:
        raise WriteConflict(
            f"{manifest['path']} 在写入后又被其他进程修改（当前 {str(current_hash)[:12]}…，"
            f"写入后 {after_hash[:12]}…）。为避免覆盖新修改，回滚已中止。"
            "如确认要覆盖，请显式使用 force=true。"
        )

    # Keep the state we are about to overwrite so the rollback is auditable
    # and itself reversible.
    if current_text is not None:
        (tx_dir / "superseded.bin").write_text(current_text, encoding="utf-8")
    if force and current_hash != after_hash:
        (tx_dir / "conflict.bin").write_text(current_text or "", encoding="utf-8")

    if manifest.get("existed"):
        original = (tx_dir / ORIGINAL_NAME).read_bytes()
        _atomic_write(abs_path, original, like=abs_path if abs_path.exists() else None)
        restored_text = original.decode("utf-8").replace("\r\n", "\n")
        if project is not None:
            project.put_file(manifest["path"], restored_text)
        if tracker is not None:
            tracker.record_version(manifest["path"], content_hash(restored_text))
    else:
        # The file did not exist before the write: rolling back removes it.
        abs_path.unlink(missing_ok=True)
        restored_text = None
        if project is not None:
            project.invalidate(manifest["path"])
        if tracker is not None:
            tracker.forget_version(manifest["path"])

    manifest.update({
        "status": STATUS_ROLLED_BACK,
        "rolled_back_at": _utc_now(),
        "rollback_status": "restored_original" if manifest.get("existed") else "deleted_created",
        "rollback_forced": bool(force and current_hash != after_hash),
    })
    _write_manifest(tx_dir / MANIFEST_NAME, manifest)
    return {
        "tx_id": tx_id,
        "path": manifest["path"],
        "status": STATUS_ROLLED_BACK,
        "rollback_status": manifest["rollback_status"],
        "restored_sha256": content_hash(restored_text) if restored_text is not None else None,
        "forced": manifest["rollback_forced"],
    }


def scan_incomplete(
    workspace: str | Path | None = None, *, min_age_seconds: float = 60.0
) -> list[dict[str, Any]]:
    """Report transactions left ``prepared`` by a crashed process.

    A manifest only stays ``prepared`` if the process died between taking the
    snapshot and recording the commit. Comparing the target's current hash with
    the snapshot tells us whether the write landed:

    * hash == before  -> the write never happened (``abandoned``, safe).
    * otherwise       -> outcome unknown (``needs_review``, never auto-restored).

    Entries younger than ``min_age_seconds`` are skipped because a live write
    may legitimately be mid-flight.
    """
    root = backup_root(workspace)
    now = datetime.now(UTC)
    findings: list[dict[str, Any]] = []
    if not root.exists():
        return findings
    for manifest_path in root.rglob(MANIFEST_NAME):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("status") != STATUS_PREPARED:
            continue
        try:
            started = datetime.fromisoformat(str(data.get("started_at")))
        except ValueError:
            started = now
        if (now - started).total_seconds() < min_age_seconds:
            continue
        abs_path = Path(str(data.get("abs_path", "")))
        try:
            current = abs_path.read_text(encoding="utf-8") if abs_path.is_file() else None
        except OSError:
            current = None
        current_hash = content_hash(current) if current is not None else None
        if current_hash == data.get("before_sha256"):
            verdict, status = STATUS_ABANDONED, "写入未生效，可安全丢弃该事务"
        elif current_hash is None and not data.get("existed"):
            verdict, status = STATUS_ABANDONED, "写入未生效（文件未创建）"
        else:
            verdict, status = STATUS_NEEDS_REVIEW, "写入可能已生效但未登记，需人工确认"
        findings.append({
            "tx_id": data.get("tx_id"),
            "path": data.get("path"),
            "run_id": data.get("run_id"),
            "verdict": verdict,
            "detail": status,
            "manifest": str(manifest_path),
            "current_sha256": current_hash,
            "before_sha256": data.get("before_sha256"),
        })
    findings.sort(key=lambda item: str(item.get("tx_id")))
    return findings


def syntax_error_for(rel_path: str, text: str) -> str | None:
    """Compile-check a written Python file; return a message when invalid.

    This is a *warning*, never an automatic rollback: a broken intermediate
    state is normal while fixing a test, and the agent must still see its own
    change to repair it.
    """
    if not rel_path.endswith(".py"):
        return None
    try:
        compile(text, rel_path, "exec")
    except SyntaxError as exc:
        return f"第 {exc.lineno} 行：{exc.msg}"
    except (ValueError, TypeError) as exc:  # e.g. NUL bytes
        return str(exc)
    return None
