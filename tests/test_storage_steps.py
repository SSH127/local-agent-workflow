from pathlib import Path

from app.storage import get_run, init_db, list_steps, save_run, save_step, update_run_state


def _mk_run(db_path: Path, run_id: str = "run-steps") -> None:
    init_db(db_path)
    save_run({
        "run_id": run_id, "goal": "goal", "status": "running",
        "attempts": 0, "summary": "", "error": None,
        "state": {}, "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }, db_path)


def test_steps_round_trip(tmp_path):
    db = Path(tmp_path) / "steps.sqlite3"
    _mk_run(db)
    for i in range(3):
        save_step({
            "step_id": f"s{i}", "run_id": "run-steps", "seq": i,
            "ts": "2026-01-01T00:00:00+00:00", "kind": "tool",
            "tool_name": "list_files", "args": {"path": "."},
            "status": "succeeded", "output": "ok", "error": None,
        }, db)
    steps = list_steps("run-steps", limit=10, db_path=db)
    assert [s["seq"] for s in steps] == [0, 1, 2]
    assert steps[0]["args"] == {"path": "."}


def test_update_run_state_merges(tmp_path):
    db = Path(tmp_path) / "state.sqlite3"
    _mk_run(db)
    update_run_state("run-steps", {"pending_tool": {"name": "write_file"}}, db)
    run = get_run("run-steps", db)
    assert run["state"]["pending_tool"]["name"] == "write_file"


def test_steps_auto_seq_monotonic(tmp_path):
    """When the caller omits seq, storage assigns a per-run monotonic ordinal."""
    db = Path(tmp_path) / "autoseq.sqlite3"
    _mk_run(db)
    for i in range(3):
        save_step({
            "step_id": f"s{i}", "run_id": "run-steps",
            "ts": "2026-01-01T00:00:00+00:00", "kind": "tool",
            "tool_name": "list_files", "args": {},
            "status": "succeeded", "output": "ok", "error": None,
        }, db)
    steps = list_steps("run-steps", limit=10, db_path=db)
    assert [s["seq"] for s in steps] == [1, 2, 3]


def test_steps_seq_is_per_run(tmp_path):
    """Each run has its own monotonic seq, independent of other runs."""
    db = Path(tmp_path) / "perrun.sqlite3"
    _mk_run(db, "r1")
    _mk_run(db, "r2")
    save_step({"step_id": "a1", "run_id": "r1", "ts": "t", "kind": "tool",
               "tool_name": "list_files", "args": {}, "status": "succeeded",
               "output": "ok", "error": None}, db)
    save_step({"step_id": "b1", "run_id": "r2", "ts": "t", "kind": "tool",
               "tool_name": "list_files", "args": {}, "status": "succeeded",
               "output": "ok", "error": None}, db)
    save_step({"step_id": "a2", "run_id": "r1", "ts": "t", "kind": "tool",
               "tool_name": "read_file", "args": {}, "status": "succeeded",
               "output": "ok", "error": None}, db)
    assert [s["seq"] for s in list_steps("r1", db_path=db)] == [1, 2]
    assert [s["seq"] for s in list_steps("r2", db_path=db)] == [1]
