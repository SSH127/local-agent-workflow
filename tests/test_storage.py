from pathlib import Path

from app.storage import get_run, init_db, save_run


def test_storage_round_trip(tmp_path):
    db_path = Path(tmp_path) / "test.sqlite3"
    init_db(db_path)
    run = {
        "run_id": "run-test", "goal": "test goal", "status": "succeeded",
        "attempts": 1, "summary": "ok", "error": None,
        "state": {"node": "finish"},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
    }
    save_run(run, db_path)
    loaded = get_run("run-test", db_path)
    assert loaded is not None
    assert loaded["state"] == {"node": "finish"}
    assert loaded["status"] == "succeeded"
