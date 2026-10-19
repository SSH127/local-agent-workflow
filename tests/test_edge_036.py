"""v0.3.6 residual mitigations: durable SSE, net deny, LLM abandon."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from app import config, storage
from app.agents.llm import DeepSeekClient, LLMCancelled
from app.cancel import CancelToken
from app.events import publish_run_event
from app.tools import command_policy, security
from app.tools.base import ToolContext
from app.tools.command_sandbox import begin_session, finish_session
from app.tools.registry import get_tool
from app.write_scope import WriteScope


def test_run_events_persist_and_replay(tmp_path, monkeypatch):
    db = tmp_path / "ev.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db(db)
    storage.save_run({
        "run_id": "r1", "goal": "g", "status": "running", "attempts": 0,
        "summary": "", "error": None, "workspace_path": str(tmp_path),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00", "state": {},
    })
    # save_run already published status (seq >= 1)
    base = storage.max_run_event_seq("r1")
    assert base >= 1
    s2 = publish_run_event("r1", "step", kind="tool")
    assert s2 == base + 1
    replay = storage.list_run_events_after("r1", after_seq=base)
    assert len(replay) == 1
    assert replay[0]["seq"] == s2
    assert replay[0]["type"] == "step"


def test_db_tail_sees_events_without_bus_subscriber(tmp_path, monkeypatch):
    db = tmp_path / "ev2.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db(db)
    seq = storage.append_run_event("r2", "approval", {"tool_name": "write_file"})
    assert seq == 1
    rows = storage.list_run_events_after("r2", 0)
    assert rows[0]["type"] == "approval"
    assert rows[0]["seq"] == 1


def test_looks_networky_patterns():
    assert command_policy.looks_networky("curl", ["https://x"])
    assert command_policy.looks_networky(
        "python", ["-c", "import urllib.request; urllib.request.urlopen('http://x')"],
    )
    assert not command_policy.looks_networky("python", ["-m", "pytest", "-q"])


def test_readonly_refuses_networky_without_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "COMMAND_READONLY_FS_HINT", True)
    monkeypatch.setattr(config, "COMMAND_BLOCK_NETWORK_TOOLS", True)

    def fake_classify(program, argv, cwd, workspace=None):
        return command_policy.CommandDecision(
            program="python", effective_program="python",
        )

    monkeypatch.setattr(security, "classify_command", fake_classify)
    scope = WriteScope([], subtask_id="ro")
    ctx = ToolContext(
        run_id="r", workspace=tmp_path, write_scope=scope, subtask_id="ro",
        approval_check=lambda n, a: False,
    )
    res = get_tool("run_project_command").handler(
        {"program": "python", "args": ["-c", "import urllib.request"], "cwd": "."},
        ctx,
    )
    assert res.get("ok") is False
    out = res.get("output") or ""
    assert "联网" in out or "批准" in out


def test_deny_network_blocks_python_socket_via_sitecustomize():
    block_dir = Path(__file__).resolve().parents[1] / "app" / "tools" / "netblock"
    assert (block_dir / "sitecustomize.py").is_file()
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=0.5)\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED:' + str(e))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(block_dir)
    env["AGENT_DENY_NETWORK"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", code],
        check=False, env=env, capture_output=True, text=True, timeout=10,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert "BLOCKED" in out or "COMMAND_DENY_NETWORK" in out, out
    assert "CONNECTED" not in out


def test_scope_escape_still_fails_with_soft_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACT_POLICY", "allow")
    monkeypatch.setattr(config, "COMMAND_SOFT_SANDBOX", True)
    (tmp_path / "app").mkdir()
    victim = tmp_path / "app" / "main.py"
    victim.write_text("keep\n", encoding="utf-8")
    session = begin_session(
        workspace=tmp_path, run_id="r1", subtask_id="t1",
        scope_patterns=["tests/**"], kind="generic",
    )
    victim.write_text("tampered\n", encoding="utf-8")
    finish = finish_session(session)
    assert any(v.path == "app/main.py" for v in finish.violations)
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_llm_cancel_discards_late_and_next_chat_ok(monkeypatch):
    import app.agents.llm as llm_mod

    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(config, "LLM_CONNECT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(config, "CANCEL_FORCE_MS", 100)
    monkeypatch.setattr(config, "CANCEL_JOIN_MS", 300)
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(config, "DEEPSEEK_BASE_URL", "http://example.invalid")

    calls = {"n": 0}

    class SlowThenOk:
        def __init__(self, *a, **k):
            self._closed = False
            calls["n"] += 1
            self.n = calls["n"]

        def post(self, *a, **k):
            if self.n == 1:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if self._closed:
                        raise httpx.HTTPError("closed")
                    time.sleep(0.05)
                raise TimeoutError("not cancelled")

            class R:
                status_code = 200
                text = "{}"

                def json(self):
                    return {
                        "choices": [{
                            "message": {"content": "ok", "tool_calls": []},
                            "finish_reason": "stop",
                        }],
                        "usage": {},
                    }

            return R()

        def close(self):
            self._closed = True

    monkeypatch.setattr(llm_mod.httpx, "Client", SlowThenOk)
    monkeypatch.setattr(httpx, "Client", SlowThenOk)

    client = DeepSeekClient(timeout=60.0)
    token = CancelToken()
    box: dict = {}

    def worker():
        try:
            client.chat([{"role": "user", "content": "hi"}], [], cancel_token=token)
            box["ok"] = True
        except LLMCancelled as exc:
            box["cancelled"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            box["err"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    token.set("用户取消")
    t.join(timeout=3.0)
    assert time.monotonic() - t0 < 2.0
    assert "cancelled" in box, box

    token2 = CancelToken()
    resp = client.chat([{"role": "user", "content": "again"}], [], cancel_token=token2)
    assert resp.content == "ok"
    assert calls["n"] >= 2
