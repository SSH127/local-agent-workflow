"""Task-goal length limit tests (30000 chars).

Verifies end-to-end that:

* the goal maximum is 30000 characters and is applied by the backend schema;
* a normal-length goal submits and round-trips exactly (no silent truncation);
* a near-limit goal is never truncated by the schema, API or SQLite storage;
* an over-limit goal is rejected with a clear 422 that names the limit;
* an empty goal is still rejected;
* authentication is unchanged and enforced independently of goal length.

These tests never touch the real DeepSeek API: the autouse fixtures blank the
API key and block all network I/O, and the background run therefore fails
cleanly before any HTTP request.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import config, storage
from app.main import app
from app.schemas import RunCreate

MAX = config.TASK_GOAL_MAX_CHARS


def _client(monkeypatch, tmp_path, token: str = ""):
    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "goal-length.sqlite3")
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", token)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Schema-level length validation (no HTTP, no background run).
# ---------------------------------------------------------------------------


def test_goal_limit_is_30000():
    assert MAX == 30000


def test_normal_goal_validates():
    goal = "检查项目结构并总结当前工作流的执行情况"
    assert RunCreate(goal=goal).goal == goal


def test_goal_at_max_validates():
    goal = "x" * MAX
    parsed = RunCreate(goal=goal)
    assert parsed.goal == goal
    assert len(parsed.goal) == MAX


def test_goal_over_max_rejected_with_clear_error():
    goal = "x" * (MAX + 1)
    with pytest.raises(ValidationError) as exc:
        RunCreate(goal=goal)
    # The error message must name the limit so the failure is unambiguous.
    assert "30000" in str(exc.value)


def test_empty_goal_rejected():
    with pytest.raises(ValidationError):
        RunCreate(goal="")


def test_short_goal_still_rejected_by_existing_min_length():
    # min_length=3 is preserved: a 2-char goal is still rejected.
    with pytest.raises(ValidationError):
        RunCreate(goal="ab")


# ---------------------------------------------------------------------------
# API-level validation and round-trip fidelity.
# ---------------------------------------------------------------------------


def test_api_normal_goal_submits_and_roundtrips(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        goal = "分析项目结构并总结当前工作流的执行情况"
        resp = client.post("/runs", json={"goal": goal, "max_attempts": 1})
        assert resp.status_code == 201
        run = resp.json()
        assert run["goal"] == goal
        # The persisted goal matches the user input exactly.
        assert storage.get_run(run["run_id"])["goal"] == goal


def test_api_near_limit_goal_not_truncated(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        goal = "x" * (MAX - 1)
        resp = client.post("/runs", json={"goal": goal, "max_attempts": 1})
        assert resp.status_code == 201
        run = resp.json()
        assert run["goal"] == goal
        assert len(run["goal"]) == MAX - 1
        stored = storage.get_run(run["run_id"])
        assert stored["goal"] == goal
        assert len(stored["goal"]) == MAX - 1


def test_api_over_limit_goal_rejected_422(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        goal = "x" * (MAX + 1)
        resp = client.post("/runs", json={"goal": goal, "max_attempts": 1})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        # The clear 4xx error names the maximum length.
        assert any("30000" in str(item.get("msg", "")) for item in detail)
        # Nothing was persisted for the rejected run.
        assert storage.list_runs(limit=10) == []


def test_api_empty_goal_rejected_422(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        resp = client.post("/runs", json={"goal": "", "max_attempts": 1})
        assert resp.status_code == 422


def test_api_auth_enforced_for_valid_max_length_goal(monkeypatch, tmp_path):
    """Authentication is enforced independently of goal length: a valid
    30000-char goal still requires the token when one is configured."""
    with _client(monkeypatch, tmp_path, token="secret-token") as client:
        goal = "x" * MAX
        assert client.post("/runs", json={"goal": goal, "max_attempts": 1}).status_code == 401
        # With the token, the same request is accepted (201), not a length error.
        resp = client.post(
            "/runs",
            json={"goal": goal, "max_attempts": 1},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 201
        assert resp.json()["goal"] == goal
