"""Local API authentication tests (shared token)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app import config, storage
from app.main import app


def _prepare(monkeypatch, tmp_path, token: str):
    db = tmp_path / "auth.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    monkeypatch.setattr(config, "LOCAL_AGENT_TOKEN", token)
    storage.init_db(db)
    return db


def test_health_is_always_open(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "secret-token")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_auth_required_when_token_set(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "secret-token")
    with TestClient(app) as client:
        # Missing token -> 401
        assert client.get("/runs").status_code == 401
        # Wrong token -> 401
        assert client.get("/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        # Correct token -> 200
        assert client.get("/runs", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_auth_disabled_when_token_empty(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "")
    with TestClient(app) as client:
        assert client.get("/runs").status_code == 200
        assert client.get("/health").status_code == 200


def test_protected_endpoints_require_token(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "tok")
    with TestClient(app) as client:
        # create run
        assert client.post("/runs", json={"goal": "任务目标x", "max_attempts": 1}).status_code == 401
        # run details / steps / approve / input
        assert client.get("/runs/nope").status_code == 401
        assert client.get("/runs/nope/steps").status_code == 401
        assert client.post("/runs/nope/approve", json={"approved": True}).status_code == 401
        assert client.post("/runs/nope/input", json={"message": "hi"}).status_code == 401


def test_authorized_requests_proceed(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "tok")
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer tok"}
        resp = client.post("/runs", json={"goal": "测试任务目标", "max_attempts": 1}, headers=headers)
        assert resp.status_code == 201
        run_id = resp.json()["run_id"]
        assert client.get(f"/runs/{run_id}", headers=headers).status_code == 200
        assert client.get(f"/runs/{run_id}/steps", headers=headers).status_code == 200


def test_approve_invalid_approval_id_rejected(monkeypatch, tmp_path):
    db = _prepare(monkeypatch, tmp_path, "")
    storage.save_run({
        "run_id": "r1", "goal": "g", "status": "waiting_for_approval",
        "attempts": 0, "summary": "", "error": None,
        "state": {"pending_approval": {"approval_id": "real-id", "tool_name": "write_file",
                                       "args": {"path": ".env"}, "risk_reason": "risk",
                                       "status": "pending"}},
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
    }, db)
    with TestClient(app) as client:
        # Wrong approval id -> 400 (mismatch with the pending request).
        assert client.post(
            "/runs/r1/approve", json={"approved": True, "approval_id": "wrong-id"},
        ).status_code == 400
