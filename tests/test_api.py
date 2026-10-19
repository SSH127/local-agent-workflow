import time

import httpx
from fastapi.testclient import TestClient

from app import storage
from app.main import app


def test_api_lifecycle(monkeypatch, tmp_path):
    db = tmp_path / "api.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db(db)

    # Simulate a developer machine with a real-looking key in the environment.
    # The autouse fixture blanks ``config.DEEPSEEK_API_KEY``, so the background
    # run must fail cleanly WITHOUT touching the network — regardless of this.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-test-key-123456")
    network_calls: list[tuple] = []

    def _forbidden_post(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("real HTTP request attempted during test")

    monkeypatch.setattr(httpx, "post", _forbidden_post)

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        # Create a run; it is queued immediately and executed in the background.
        resp = client.post("/runs", json={"goal": "测试任务目标", "max_attempts": 2})
        assert resp.status_code == 201
        run = resp.json()
        assert run["run_id"]
        run_id = run["run_id"]

        # Without a DeepSeek key the background run must fail cleanly.
        status = run["status"]
        for _ in range(100):
            run = client.get(f"/runs/{run_id}").json()
            status = run["status"]
            if status in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert status == "failed"
        assert run["error"]

        # Steps endpoint returns a list.
        steps = client.get(f"/runs/{run_id}/steps").json()
        assert isinstance(steps, list)

        # Approving a run that is not waiting for approval -> 409.
        assert client.post(f"/runs/{run_id}/approve", json={"approved": True}).status_code == 409

        # The run is present in the history list.
        runs = client.get("/runs").json()
        assert any(item["run_id"] == run_id for item in runs)

    # No real network request may be made during this test.
    assert network_calls == []
