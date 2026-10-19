import contextlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import DATABASE_PATH

# Terminal status a run is moved into when it was left ``running`` across a
# process restart and cannot be safely resumed (side effects are unverifiable).
STATUS_INTERRUPTED = "interrupted"
INTERRUPTED_REASON = "服务进程重启，任务执行结果无法确认。"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# Every ``with _connect() as conn:`` block in this process is serialized: the
# parallel scheduler runs several subtasks on worker threads that all write
# steps / leases / heartbeats. SQLite's own busy timeout covers other processes.
_DB_LOCK = threading.RLock()


class _SerializedConnection(sqlite3.Connection):
    def __enter__(self):  # type: ignore[override]
        _DB_LOCK.acquire()
        try:
            return super().__enter__()
        except BaseException:
            _DB_LOCK.release()
            raise

    def __exit__(self, *exc):  # type: ignore[override]
        try:
            return super().__exit__(*exc)
        finally:
            _DB_LOCK.release()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, factory=_SerializedConnection)
    conn.row_factory = sqlite3.Row
    # WAL improves concurrent readers (SSE DB-tail) while a writer is active.
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL, "
            "attempts INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT '', "
            "error TEXT, workspace_path TEXT, state_json TEXT NOT NULL DEFAULT '{}', "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        # Migration for databases created before the per-run workspace column.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        if "workspace_path" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN workspace_path TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_steps ("
            "step_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL, "
            "ts TEXT NOT NULL, kind TEXT NOT NULL, tool_name TEXT, "
            "args_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL, "
            "output TEXT, error TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_steps_run ON run_steps(run_id, seq)")
        # Migration: per-subtask step attribution (task-decomposition workflow).
        step_cols = {row[1] for row in conn.execute("PRAGMA table_info(run_steps)")}
        if "subtask_id" not in step_cols:
            conn.execute("ALTER TABLE run_steps ADD COLUMN subtask_id TEXT")
        if "attempt_number" not in step_cols:
            conn.execute("ALTER TABLE run_steps ADD COLUMN attempt_number INTEGER")
        # Task graph per run: plan_version + plan JSON (subtasks with status,
        # attempt_number, acceptance_evidence, ...) + the execution lease.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_plans ("
            "run_id TEXT PRIMARY KEY, plan_version INTEGER NOT NULL DEFAULT 1, "
            "plan_json TEXT NOT NULL DEFAULT '{}', execution_lease_json TEXT, "
            "updated_at TEXT NOT NULL)"
        )
        plan_cols = {row[1] for row in conn.execute("PRAGMA table_info(run_plans)")}
        if "goal_json" not in plan_cols:
            # Frozen goal text + clauses, written once (immutable).
            conn.execute("ALTER TABLE run_plans ADD COLUMN goal_json TEXT")
        # One execution lease (+ executor conversation state) per subtask, so
        # parallel subtasks each hold their own lease and heartbeat.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS subtask_leases ("
            "run_id TEXT NOT NULL, subtask_id TEXT NOT NULL, lease_json TEXT, "
            "exec_state_json TEXT, updated_at TEXT NOT NULL, "
            "PRIMARY KEY (run_id, subtask_id))"
        )
        # Migrate the single run-level lease of v0.3.0 into the per-subtask table.
        for row in conn.execute(
            "SELECT run_id, execution_lease_json FROM run_plans "
            "WHERE execution_lease_json IS NOT NULL"
        ).fetchall():
            try:
                old = json.loads(row[1])
            except (TypeError, ValueError):
                continue
            if isinstance(old, dict) and old.get("subtask_id"):
                conn.execute(
                    "INSERT OR IGNORE INTO subtask_leases (run_id, subtask_id, lease_json, "
                    "updated_at) VALUES (?, ?, ?, ?)",
                    (row[0], str(old["subtask_id"]), row[1], _utc_now()),
                )
        # Append-only plan version history (never updated or deleted).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS plan_versions ("
            "run_id TEXT NOT NULL, version INTEGER NOT NULL, revision INTEGER NOT NULL, "
            "parent_version INTEGER, parent_revision INTEGER, reason TEXT NOT NULL, "
            "failure_class TEXT, created_at TEXT NOT NULL, plan_json TEXT NOT NULL, "
            "diff_json TEXT, PRIMARY KEY (run_id, version, revision))"
        )

        # Append-only SSE / status event log (survives restart; seq is per-run).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_events ("
            "run_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, "
            "payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, "
            "PRIMARY KEY (run_id, seq))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_events_run_seq ON run_events(run_id, seq)"
        )


def save_run(run: dict[str, Any], db_path: Path | None = None) -> None:
    payload = dict(run)
    payload["state_json"] = json.dumps(run.get("state", {}), ensure_ascii=False)
    payload["workspace_path"] = run.get("workspace_path")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, goal, status, attempts, summary, error, "
            "workspace_path, state_json, created_at, updated_at) "
            "VALUES (:run_id, :goal, :status, :attempts, :summary, :error, "
            ":workspace_path, :state_json, :created_at, :updated_at) "
            "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
            "attempts=excluded.attempts, summary=excluded.summary, error=excluded.error, "
            "workspace_path=excluded.workspace_path, "
            "state_json=excluded.state_json, updated_at=excluded.updated_at",
            payload,
        )
    with contextlib.suppress(Exception):
        from app.events import publish_run_event
        publish_run_event(run["run_id"], "status", status=run.get("status"))


def get_run(run_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["state"] = json.loads(result.pop("state_json"))
    return result


def list_runs(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["state"] = json.loads(item.pop("state_json"))
        results.append(item)
    return results


def update_run_state(
    run_id: str,
    state_patch: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Merge ``state_patch`` into the run's ``state`` dict and persist it."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return
        current = json.loads(row["state_json"] or "{}")
        current.update(state_patch)
        conn.execute(
            "UPDATE runs SET state_json = ? WHERE run_id = ?",
            (json.dumps(current, ensure_ascii=False), run_id),
        )
    with contextlib.suppress(Exception):
        from app.events import publish_run_event
        publish_run_event(run_id, "state")


def update_run_status(
    run_id: str,
    status: str,
    *,
    summary: str | None = None,
    error: str | None = None,
    updated_at: str | None = None,
    db_path: Path | None = None,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = COALESCE(?, updated_at) WHERE run_id = ?",
            (status, updated_at, run_id),
        )
        if summary is not None:
            conn.execute("UPDATE runs SET summary = ? WHERE run_id = ?", (summary, run_id))
        if error is not None:
            conn.execute("UPDATE runs SET error = ? WHERE run_id = ?", (error, run_id))
    with contextlib.suppress(Exception):
        from app.events import publish_run_event
        publish_run_event(run_id, "status", status=status, summary=summary, error=error)


def transition_run_status(
    run_id: str,
    from_status: str,
    to_status: str,
    *,
    state_patch: dict[str, Any] | None = None,
    summary: str | None = None,
    error: str | None = None,
    updated_at: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """Atomically transition a run from ``from_status`` to ``to_status``.

    Uses a conditional ``UPDATE ... WHERE status = ?`` so that, when several
    callers race on the same transition (e.g. concurrent HTTP approve requests),
    exactly one caller wins and the rest observe ``False``. ``state_patch`` is
    merged into the persisted state in the *same transaction* as the status
    change, so a carried approval decision and its status change cannot be torn
    apart by a crash between the two writes.

    Returns ``True`` only for the single winning caller.
    """
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE runs SET status = ?, updated_at = COALESCE(?, updated_at) "
            "WHERE run_id = ? AND status = ?",
            (to_status, updated_at, run_id, from_status),
        )
        if cur.rowcount != 1:
            return False
        if state_patch:
            row = conn.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            current = json.loads(row["state_json"] or "{}")
            current.update(state_patch)
            conn.execute(
                "UPDATE runs SET state_json = ? WHERE run_id = ?",
                (json.dumps(current, ensure_ascii=False), run_id),
            )
        if summary is not None:
            conn.execute("UPDATE runs SET summary = ? WHERE run_id = ?", (summary, run_id))
        if error is not None:
            conn.execute("UPDATE runs SET error = ? WHERE run_id = ?", (error, run_id))
        return True


def mark_interrupted_runs(
    db_path: Path | None = None,
    *,
    reason: str = INTERRUPTED_REASON,
    exclude_run_ids: set[str] | None = None,
) -> int:
    """Move every run left in ``running`` across a restart to ``interrupted``.

    A ``running`` run has no surviving execution thread after the process
    restarted, and whether any of its tools already executed side effects is
    unknowable. It must NOT be replayed automatically, so we mark it with a
    clear terminal status and retain its history/steps for human review.

    Returns the number of runs that were interrupted.
    """
    exclude = sorted(exclude_run_ids or set())
    with _connect(db_path) as conn:
        sql = (
            "UPDATE runs SET status = ?, summary = ?, error = ?, updated_at = ? "
            "WHERE status = ?"
        )
        params: list[Any] = [STATUS_INTERRUPTED, "任务中断", reason, _utc_now(), "running"]
        if exclude:
            sql += " AND run_id NOT IN (" + ",".join("?" for _ in exclude) + ")"
            params.extend(exclude)
        cur = conn.execute(sql, params)
        return cur.rowcount


def list_run_ids_with_status(status: str, db_path: Path | None = None) -> list[str]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT run_id FROM runs WHERE status = ?", (status,)).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Task graph (plan) + execution lease
# ---------------------------------------------------------------------------

def get_plan(run_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT plan_json FROM run_plans WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0] or "{}") or None


class GoalImmutableError(ValueError):
    """The frozen goal text / clauses of a run's plan may never change."""


def _goal_payload(plan: dict[str, Any]) -> str | None:
    if "goal_clauses" not in plan or plan.get("goal_clauses") is None:
        return None
    return json.dumps({"goal": plan.get("goal"), "clauses": plan.get("goal_clauses")},
                      ensure_ascii=False, sort_keys=True)


def save_plan(run_id: str, plan: dict[str, Any], db_path: Path | None = None) -> None:
    goal = _goal_payload(plan)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT goal_json FROM run_plans WHERE run_id = ?", (run_id,)
        ).fetchone()
        frozen = row[0] if row is not None else None
        if frozen and goal is not None and frozen != goal:
            raise GoalImmutableError("任务图的原始目标与目标条款已冻结，不能修改。")
        conn.execute(
            "INSERT INTO run_plans (run_id, plan_version, plan_json, goal_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
            "plan_version=excluded.plan_version, plan_json=excluded.plan_json, "
            "goal_json=COALESCE(run_plans.goal_json, excluded.goal_json), "
            "updated_at=excluded.updated_at",
            (run_id, int(plan.get("version") or 1),
             json.dumps(plan, ensure_ascii=False), goal, _utc_now()),
        )


def get_frozen_goal(run_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT goal_json FROM run_plans WHERE run_id = ?", (run_id,)
        ).fetchone()
    return json.loads(row[0]) if row is not None and row[0] else None


# --- per-subtask execution leases ------------------------------------------

def get_leases(run_id: str, db_path: Path | None = None) -> dict[str, dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT subtask_id, lease_json FROM subtask_leases "
            "WHERE run_id = ? AND lease_json IS NOT NULL", (run_id,)
        ).fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def get_lease(
    run_id: str, subtask_id: str | None = None, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Lease of one subtask; without ``subtask_id`` the most recently
    heart-beaten lease of the run (compatibility with the single-lease API)."""
    leases = get_leases(run_id, db_path)
    if subtask_id is not None:
        return leases.get(subtask_id)
    if not leases:
        return None
    return max(leases.values(), key=lambda lease: str(lease.get("heartbeat_at") or ""))


def set_lease(
    run_id: str, lease: dict[str, Any] | None, db_path: Path | None = None
) -> None:
    if lease is None or not lease.get("subtask_id"):
        return
    payload = json.dumps(lease, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO subtask_leases (run_id, subtask_id, lease_json, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(run_id, subtask_id) DO UPDATE SET "
            "lease_json=excluded.lease_json, updated_at=excluded.updated_at",
            (run_id, str(lease["subtask_id"]), payload, _utc_now()),
        )


def acquire_lease(
    run_id: str,
    new_lease: dict[str, Any],
    *,
    expected_lease_id: str | None,
    db_path: Path | None = None,
) -> bool:
    """Compare-and-swap one subtask's lease: succeeds only if the stored lease
    id still equals ``expected_lease_id`` (None = no lease). Two runners racing
    for the same subtask cannot both win."""
    sid = str(new_lease["subtask_id"])
    payload = json.dumps(new_lease, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")  # serialize the read-compare-write
        row = conn.execute(
            "SELECT lease_json FROM subtask_leases WHERE run_id = ? AND subtask_id = ?",
            (run_id, sid),
        ).fetchone()
        current = json.loads(row[0]) if row is not None and row[0] else None
        current_id = current.get("lease_id") if isinstance(current, dict) else None
        if current_id != expected_lease_id:
            return False
        conn.execute(
            "INSERT INTO subtask_leases (run_id, subtask_id, lease_json, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(run_id, subtask_id) DO UPDATE SET "
            "lease_json=excluded.lease_json, updated_at=excluded.updated_at",
            (run_id, sid, payload, _utc_now()),
        )
        return True


def save_exec_state(
    run_id: str, subtask_id: str, state: dict[str, Any], db_path: Path | None = None
) -> None:
    """Executor conversation state of one subtask (parallel-safe resume)."""
    payload = json.dumps(state, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO subtask_leases (run_id, subtask_id, exec_state_json, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(run_id, subtask_id) DO UPDATE SET "
            "exec_state_json=excluded.exec_state_json, updated_at=excluded.updated_at",
            (run_id, subtask_id, payload, _utc_now()),
        )


def get_exec_state(
    run_id: str, subtask_id: str, db_path: Path | None = None
) -> dict[str, Any]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT exec_state_json FROM subtask_leases WHERE run_id = ? AND subtask_id = ?",
            (run_id, subtask_id),
        ).fetchone()
    return json.loads(row[0]) if row is not None and row[0] else {}


# --- append-only plan version history --------------------------------------

def append_plan_version(
    run_id: str, entry: dict[str, Any], db_path: Path | None = None
) -> bool:
    """Insert one immutable (version, revision) snapshot. Never overwrites:
    returns ``False`` when that (version, revision) already exists."""
    with _connect(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO plan_versions (run_id, version, revision, parent_version, "
                "parent_revision, reason, failure_class, created_at, plan_json, diff_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, int(entry["version"]), int(entry.get("revision") or 1),
                 entry.get("parent_version"), entry.get("parent_revision"),
                 str(entry.get("reason") or ""), entry.get("failure_class"),
                 entry.get("created_at") or _utc_now(),
                 json.dumps(entry.get("plan") or {}, ensure_ascii=False),
                 json.dumps(entry.get("diff"), ensure_ascii=False)
                 if entry.get("diff") is not None else None),
            )
        except sqlite3.IntegrityError:
            return False
    return True


def _version_row(row: sqlite3.Row, full: bool) -> dict[str, Any]:
    item = {
        "version": row["version"], "revision": row["revision"],
        "parent_version": row["parent_version"], "parent_revision": row["parent_revision"],
        "reason": row["reason"], "failure_class": row["failure_class"],
        "created_at": row["created_at"],
    }
    plan = json.loads(row["plan_json"] or "{}")
    item["subtask_count"] = len(plan.get("subtasks") or [])
    if full:
        item["plan"] = plan
        item["diff"] = json.loads(row["diff_json"]) if row["diff_json"] else None
    return item


def list_plan_versions(run_id: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM plan_versions WHERE run_id = ? ORDER BY version, revision",
            (run_id,),
        ).fetchall()
    return [_version_row(r, full=False) for r in rows]


def get_plan_version(
    run_id: str, version: int, revision: int | None = None, db_path: Path | None = None
) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        if revision is None:
            row = conn.execute(
                "SELECT * FROM plan_versions WHERE run_id = ? AND version = ? "
                "ORDER BY revision DESC LIMIT 1", (run_id, version),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM plan_versions WHERE run_id = ? AND version = ? AND revision = ?",
                (run_id, version, revision),
            ).fetchone()
    return _version_row(row, full=True) if row is not None else None


def save_step(step: dict[str, Any], db_path: Path | None = None) -> None:
    payload = {
        "step_id": step["step_id"],
        "run_id": step["run_id"],
        "seq": step.get("seq"),
        "ts": step["ts"],
        "kind": step["kind"],
        "tool_name": step.get("tool_name"),
        "args_json": json.dumps(step.get("args", {}), ensure_ascii=False),
        "status": step["status"],
        "output": step.get("output"),
        "error": step.get("error"),
        "subtask_id": step.get("subtask_id"),
        "attempt_number": step.get("attempt_number"),
    }
    with _connect(db_path) as conn:
        # A per-run monotonic ordinal. When the caller does not supply a seq
        # (the agent executor leaves it to storage), assign MAX(seq)+1 so every
        # step gets a unique, reconstructable order — even across a resume.
        if payload["seq"] is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_steps WHERE run_id = ?",
                (step["run_id"],),
            ).fetchone()
            payload["seq"] = row[0]
        conn.execute(
            "INSERT INTO run_steps (step_id, run_id, seq, ts, kind, tool_name, "
            "args_json, status, output, error, subtask_id, attempt_number) "
            "VALUES (:step_id, :run_id, :seq, :ts, :kind, :tool_name, :args_json, "
            ":status, :output, :error, :subtask_id, :attempt_number)",
            payload,
        )
    with contextlib.suppress(Exception):
        from app.events import publish_run_event
        publish_run_event(
            step["run_id"], "step",
            kind=step.get("kind"), tool_name=step.get("tool_name"),
            status=step.get("status"), subtask_id=step.get("subtask_id"),
        )


def list_steps(run_id: str, limit: int = 200, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM run_steps WHERE run_id = ? ORDER BY seq ASC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["args"] = json.loads(item.pop("args_json"))
        results.append(item)
    return results


def append_run_event(
    run_id: str,
    type_: str,
    fields: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert the next per-run event row; returns ``seq`` (1-based, monotonic)."""
    payload = dict(fields or {})
    payload.setdefault("type", type_)
    created = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        seq = int(row[0]) + 1
        conn.execute(
            "INSERT INTO run_events (run_id, seq, type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, type_, json.dumps(payload, ensure_ascii=False, default=str), created),
        )
    return seq


def list_run_events_after(
    run_id: str,
    after_seq: int = 0,
    *,
    limit: int = 500,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return events with ``seq > after_seq`` (oldest first), for SSE replay/tail."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT seq, type, payload_json, created_at FROM run_events "
            "WHERE run_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
            (run_id, int(after_seq), int(limit)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {"data": payload}
        ev = dict(payload)
        ev["seq"] = int(row["seq"])
        ev["type"] = str(row["type"] or ev.get("type") or "")
        ev["run_id"] = run_id
        ev["created_at"] = row["created_at"]
        out.append(ev)
    return out


def max_run_event_seq(run_id: str, db_path: Path | None = None) -> int:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return int(row[0])
