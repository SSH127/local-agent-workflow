"""Cross-process plan writes.

A run is normally controlled by a single process, but the SQLite file is shared:
a second uvicorn worker, a restart overlapping an old process, or a manual API
call can all write the same ``run_plans`` row. ``_DB_LOCK`` only serializes
within one process, so these tests use real OS processes.

What must hold:

1. concurrent whole-plan writes never tear the row (the persisted JSON is one
   writer's complete plan, never a mix or a truncated document) and never fail
   with ``database is locked``;
2. the frozen goal is decided exactly once: when two processes race with
   different goals, exactly one wins and the other always gets
   ``GoalImmutableError`` - the check and the write are one transaction.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from app import storage

ROUNDS = 12


def _write_plan_loop(db_path: str, run_id: str, goal: str, marker: str,
                     rounds: int, out) -> None:
    """Child process: repeatedly save a complete plan for the same run."""
    plan: dict = {
        "version": 1,
        "revision": 1,
        "goal": goal,
        "goal_clauses": [{"id": "C1", "text": goal}],
        "subtasks": [{"id": "1", "marker": marker}],
    }
    successes = 0
    conflicts = 0
    errors: list[str] = []
    for i in range(rounds):
        plan["subtasks"][0]["marker"] = f"{marker}:{i}"
        try:
            storage.save_plan(run_id, plan, db_path=Path(db_path))
            successes += 1
        except storage.GoalImmutableError:
            conflicts += 1
        except Exception as exc:  # noqa: BLE001 - reported back to the parent
            errors.append(f"{type(exc).__name__}: {exc}")
    out.put({"marker": marker, "successes": successes,
             "conflicts": conflicts, "errors": errors})


def _run_workers(
    db_path: Path, run_id: str, goals_and_markers: list[tuple[str, str]],
) -> list[dict]:
    ctx = mp.get_context("spawn")
    procs: list[tuple[mp.Process, mp.Queue]] = []
    for goal, marker in goals_and_markers:
        out: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_write_plan_loop,
            args=(str(db_path), run_id, goal, marker, ROUNDS, out),
        )
        proc.start()
        procs.append((proc, out))
    results: list[dict] = []
    for proc, out in procs:
        try:
            results.append(out.get(timeout=90))
        finally:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - only on a hung child
                proc.terminate()
                proc.join(timeout=10)
    for proc, _ in procs:
        assert proc.exitcode == 0, f"child exited with {proc.exitcode}"
    return results


def test_concurrent_plan_writes_never_tear_the_row(tmp_path):
    """Input: four separate processes write the same run's plan concurrently,
    each round replacing it with that writer's complete document.
    Trigger: real OS processes (spawn) against one SQLite file.
    Assert: all writes succeed (no lock errors), and the persisted plan is one
    writer's complete plan - parseable, with the expected keys and a marker from
    the allowed set."""
    db = tmp_path / "wf.sqlite3"
    storage.init_db(db)
    writers = [("fix a.py bug", f"p{i}") for i in range(4)]

    results = _run_workers(db, "run-race", writers)

    assert all(r["errors"] == [] for r in results), results
    assert sum(r["successes"] for r in results) == len(writers) * ROUNDS
    assert sum(r["conflicts"] for r in results) == 0

    plan = storage.get_plan("run-race", db_path=db)
    assert plan is not None
    assert plan["goal"] == "fix a.py bug"
    assert plan["goal_clauses"] == [{"id": "C1", "text": "fix a.py bug"}]
    assert plan["version"] == 1 and plan["revision"] == 1
    # Exactly one writer's complete subtask survived; never a torn/mixed object.
    assert len(plan["subtasks"]) == 1
    assert set(plan["subtasks"][0]) == {"id", "marker"}
    marker, sep, round_no = plan["subtasks"][0]["marker"].partition(":")
    assert sep and marker in {m for _, m in writers}
    assert 0 <= int(round_no) < ROUNDS


class _ProbeConnection:
    """Delegating wrapper that records ``in_transaction`` for one statement."""

    def __init__(self, conn, flags: list[bool]) -> None:
        self._conn = conn
        self._flags = flags

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)

    def execute(self, sql, *args, **kwargs):
        if "SELECT goal_json FROM run_plans" in sql:
            self._flags.append(bool(self._conn.in_transaction))
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_freeze_check_reads_inside_the_write_transaction(tmp_path, monkeypatch):
    """Input: a normal ``save_plan`` call. Trigger: probe the connection's
    ``in_transaction`` at the moment the freeze SELECT runs.
    Assert: the read happens inside an explicit write transaction. This is the
    structural property the cross-process test observes from outside - if the
    read is moved back out of the transaction, two processes can both see an
    empty goal and the later commit silently replaces the first freeze."""
    db = tmp_path / "wf.sqlite3"
    storage.init_db(db)
    flags: list[bool] = []
    real_connect = storage._connect
    monkeypatch.setattr(
        storage, "_connect",
        lambda db_path=None: _ProbeConnection(real_connect(db_path), flags),
    )
    storage.save_plan("run-probe", {
        "version": 1, "revision": 1, "goal": "fix a.py bug",
        "goal_clauses": [{"id": "C1", "text": "fix a.py bug"}], "subtasks": [],
    }, db_path=db)
    assert flags, "the freeze SELECT was not executed"
    assert all(flags), "the freeze check ran outside the write transaction"


def test_conflicting_frozen_goals_have_exactly_one_winner(tmp_path):
    """Input: two processes race to freeze DIFFERENT goals on a fresh run.
    Trigger: concurrent ``save_plan`` (BEGIN IMMEDIATE read-check-write).
    Assert: exactly one process writes every round and the other is rejected
    every round with ``GoalImmutableError``; the persisted goal is the winner's
    - the freeze check can never be observed outside the write transaction."""
    db = tmp_path / "wf.sqlite3"
    storage.init_db(db)

    results = _run_workers(db, "run-freeze", [("fix a.py bug", "a"), ("fix b.py bug", "b")])

    assert all(r["errors"] == [] for r in results), results
    assert sorted(r["successes"] for r in results) == [0, ROUNDS], results
    assert sorted(r["conflicts"] for r in results) == [0, ROUNDS], results

    winner = next(r for r in results if r["successes"])
    persisted = storage.get_plan("run-freeze", db_path=db)
    assert persisted["goal"] == ("fix a.py bug" if winner["marker"] == "a"
                                 else "fix b.py bug")
    assert persisted["subtasks"][0]["marker"].startswith(winner["marker"] + ":")
