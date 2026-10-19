"""Regression tests proving the test suite is isolated from the real DeepSeek
API and the development database, and that no recorder silently auto-approves.

These tests are read-only with respect to the network: they install a failing
``httpx.post`` so any accidental real HTTP call fails the test immediately.
"""
from __future__ import annotations

import httpx
import pytest

from app import config
from app.agents.llm import DeepSeekClient, LLMError


def test_config_key_blanked_despite_env(monkeypatch):
    """A fake key in the OS env must NOT leak into config: the autouse fixture
    resets the module attribute (``setenv`` alone would leave the import-time
    value intact)."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-env-key-123456")
    assert config.DEEPSEEK_API_KEY == ""


def test_default_client_fails_without_network(monkeypatch):
    """A default ``DeepSeekClient`` (reading the blanked config) must fail before
    any HTTP request, even with a fake key present in the environment."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-env-key-123456")
    calls: list[tuple] = []

    def _forbidden_post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("real HTTP request attempted during test")

    monkeypatch.setattr(httpx, "post", _forbidden_post)

    client = DeepSeekClient()  # default: reads config.DEEPSEEK_API_KEY (blanked)
    assert client.api_key == ""
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}], [])
    assert calls == []


def test_noop_recorder_removed():
    """The auto-approving ``NoopRecorder`` dead code must no longer exist, while
    the real gate-backed recorder must still be present."""
    from app.engine import runner

    assert not hasattr(runner, "NoopRecorder")
    assert hasattr(runner, "StorageRecorder")


def test_no_other_auto_approve_recorder():
    """The only remaining fallback recorder (``_DefaultRecorder``) must deny
    approval-requiring operations; there is no recorder that auto-approves."""
    from app.engine.approvals import ApprovalDecision
    from app.engine.workflow import _DefaultRecorder

    decision = _DefaultRecorder().request_approval(
        "r1", "write_file", {"path": ".env"}, "覆盖受保护文件",
    )
    assert isinstance(decision, ApprovalDecision)
    assert decision.approved is False


# ---------------------------------------------------------------------------
# Network & database isolation (global fixtures).
# ---------------------------------------------------------------------------

def test_network_entry_points_blocked_by_default():
    """Without an explicit mock, every real network entry point raises a clear
    error instead of silently returning a fake success."""
    import socket

    with pytest.raises(RuntimeError, match="真实网络"):
        httpx.post("https://api.deepseek.com/chat/completions", json={})
    with pytest.raises(RuntimeError, match="真实网络"):
        httpx.request("GET", "http://localhost:8000")
    with pytest.raises(RuntimeError, match="真实网络"):
        socket.create_connection(("127.0.0.1", 8000))


def test_default_client_never_issues_real_request():
    """A default DeepSeekClient (blanked key) fails before any HTTP call, and a
    real call would be blocked even if it tried."""
    client = DeepSeekClient()
    assert client.api_key == ""
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}], [])


def test_database_isolated_to_temp_despite_env(monkeypatch, tmp_path):
    """Even when ``DATABASE_PATH`` points at a "development" database in the
    environment, tests write only to the per-test temporary database."""
    from app import storage

    dev_db = tmp_path.parent / "dev-should-not-be-touched.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(dev_db))

    # The autouse fixture points storage at a temp file within tmp_path.
    assert storage.DATABASE_PATH == tmp_path / "test.sqlite3"

    storage.init_db()
    storage.save_run({
        "run_id": "r1", "goal": "g", "status": "running",
        "attempts": 0, "summary": "", "error": None,
        "state": {}, "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    })
    assert (tmp_path / "test.sqlite3").exists()
    assert not dev_db.exists()


def test_check_http_explicit_mock_provides_result(monkeypatch):
    """If a test needs a real HTTP-check result it must mock explicitly; a mock
    that returns a result is honored and does not touch the network."""
    from app.tools.base import ToolContext
    from app.tools.registry import get_tool

    class FakeResponse:
        status_code = 200
        text = "ok"

    def _fake_request(method, url, timeout=None):
        return FakeResponse()

    monkeypatch.setattr(httpx, "request", _fake_request)
    tool = get_tool("check_http")
    result = tool.handler(
        {"url": "http://localhost:8000", "method": "GET"},
        ToolContext(run_id="r"),
    )
    assert result["ok"] is True
    assert result["data"]["status_code"] == 200
