"""v0.3.5 edge harden: SSE + durable approval wait, sandbox, fast LLM cancel."""
from __future__ import annotations

import threading
import time

import httpx

from app import config, storage
from app.agents.llm import DeepSeekClient, LLMCancelled
from app.cancel import CancelToken
from app.engine.approvals import (
    APPROVAL_GRANTED,
    APPROVAL_PENDING,
    ApprovalGate,
    ApprovalRecord,
)
from app.engine.runner import Runner
from app.events import RunEventBus
from app.main import app
from app.tools.base import ToolContext
from app.tools.registry import get_tool
from app.tools.win_job import attach_kill_on_close_job, close_job
from app.write_scope import WriteScope

# ---------------------------------------------------------------------------
# 1) SSE event bus + HTTP stream
# ---------------------------------------------------------------------------


def test_event_bus_delivers_status_change():
    bus = RunEventBus()
    q = bus.subscribe("run-a")
    bus.publish("run-a", {"type": "status", "status": "running"})
    ev = q.get(timeout=1.0)
    assert ev["type"] == "status"
    assert ev["status"] == "running"
    assert ev["run_id"] == "run-a"
    bus.unsubscribe("run-a", q)


def test_sse_endpoint_emits_on_status_change(tmp_path, monkeypatch):
    """Bus publish on status change + SSE route registered (avoid TestClient stream hang)."""
    from app.events import bus

    db = tmp_path / "sse.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db(db)
    storage.save_run({
        "run_id": "r-sse",
        "goal": "g",
        "status": "running",
        "attempts": 0,
        "summary": "",
        "error": None,
        "workspace_path": str(tmp_path),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "state": {},
    })

    q = bus.subscribe("r-sse")
    storage.update_run_status("r-sse", "succeeded", summary="done")
    ev = q.get(timeout=2.0)
    bus.unsubscribe("r-sse", q)
    assert ev.get("type") == "status"
    assert ev.get("status") == "succeeded"

    # Route is registered (avoid TestClient long-lived stream — it can hang
    # the sync portal when the generator blocks on queue.get).
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/runs/{run_id}/events" in paths


# ---------------------------------------------------------------------------
# 2) Durable approval: thread gone + adopt / resume
# ---------------------------------------------------------------------------


def test_approval_adopt_and_wait_after_event_lost():
    """Simulate waiter thread gone: rehydrate via adopt, then resolve."""
    gate = ApprovalGate()
    record = gate.request("r1", "write_file", {"path": "a.py"}, "risk")
    aid = record.approval_id
    # Drop the Event (thread crashed) but keep the record shape as if from DB.
    persisted = record.to_dict()
    gate.clear_run("r1")
    assert gate.get(aid) is None

    # Rehydrate from persisted state (same approval_id).
    adopted = gate.adopt(ApprovalRecord.from_dict(persisted))
    assert adopted.approval_id == aid
    assert adopted.status == APPROVAL_PENDING

    box: dict = {}

    def waiter():
        box["status"] = gate.wait(aid, timeout=5.0)

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    assert gate.resolve_if_pending(aid, True, "ok") == "resolved"
    t.join(timeout=2.0)
    assert box.get("status") == APPROVAL_GRANTED


def test_approve_after_restart_resumes(monkeypatch, tmp_path):
    """Gate empty + DB pending_approval → approve falls through to resume."""
    from app import main
    from app.schemas import ApprovalRequest

    db = tmp_path / "appr.sqlite3"
    monkeypatch.setattr(storage, "DATABASE_PATH", db)
    storage.init_db(db)
    pending = {
        "approval_id": "deadbeef",
        "run_id": "r-lost",
        "tool_name": "write_file",
        "args": {"path": "x.py", "content": "1"},
        "risk_reason": "overwrite",
        "status": "pending",
        "reason": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    storage.save_run({
        "run_id": "r-lost",
        "goal": "写文件",
        "status": "waiting_for_approval",
        "attempts": 1,
        "summary": "",
        "error": None,
        "workspace_path": str(tmp_path),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "state": {
            "pending_approval": pending,
            "agent_messages": [],
            "pending_tools": [],
        },
    })
    fresh = Runner()  # empty gate — waiter thread is gone
    resume_calls: list[str] = []
    monkeypatch.setattr(main, "runner", fresh)
    monkeypatch.setattr(
        fresh, "resume", lambda run_id, goal: resume_calls.append(run_id) or True,
    )
    result = main.approve_run("r-lost", ApprovalRequest(approved=True, reason="ok"))
    assert resume_calls == ["r-lost"]
    assert result["status"] == "running"
    assert (result.get("state") or {}).get("approval_decision", {}).get("approved") is True


# ---------------------------------------------------------------------------
# 3) Read-only FS hint + Job Object helpers
# ---------------------------------------------------------------------------


def test_readonly_subtask_refuses_write_command_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "COMMAND_READONLY_FS_HINT", True)
    monkeypatch.setattr(config, "COMMAND_SOFT_SANDBOX", True)
    scope = WriteScope([], subtask_id="ro1")  # read_only
    assert scope.read_only
    ctx = ToolContext(
        run_id="run-ro", workspace=tmp_path, write_scope=scope, subtask_id="ro1",
        approval_check=lambda n, a: True,
    )
    res = get_tool("run_project_command").handler(
        {"program": "pip", "args": ["install", "requests"], "cwd": "."}, ctx,
    )
    assert res.get("ok") is False
    err = (res.get("output") or "") + str(res.get("error") or "") + str(res.get("data") or "")
    assert "只读" in err or "COMMAND_READONLY" in err or "拒绝" in err, err


def test_win_job_helpers_noop_off_windows():
    class FakeProc:
        pid = 1
        _handle = None

    job = attach_kill_on_close_job(FakeProc())
    # On Linux (box) this is None; on Windows it may be a handle.
    close_job(job)  # must not raise


def test_cancel_kills_running_command(tmp_path, monkeypatch):
    """CancelToken during a long sleep kills the process tree (Job Object on Win)."""
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "COMMAND_SOFT_SANDBOX", True)
    token = CancelToken()
    scope = WriteScope(["**"], subtask_id="k1")
    ctx = ToolContext(
        run_id="run-k", workspace=tmp_path, write_scope=scope, subtask_id="k1",
        approval_check=lambda n, a: True,
        cancel_token=token,
        cancel_check=token.check,
    )
    box: dict = {}

    def worker():
        box["res"] = get_tool("run_project_command").handler(
            {"program": "python", "args": ["-c", "import time; time.sleep(8)"], "cwd": "."},
            ctx,
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.4)
    t0 = time.monotonic()
    token.set("用户取消")
    t.join(timeout=8.0)
    elapsed = time.monotonic() - t0
    assert not t.is_alive(), f"command handler did not return after cancel; box={box}"
    assert elapsed < 5.0, f"cancel kill too slow: {elapsed:.2f}s; box={box}"


# ---------------------------------------------------------------------------
# 4) LLM cancel abandons slow HTTP well under LLM_TIMEOUT_SECONDS
# ---------------------------------------------------------------------------


def test_llm_cancel_beats_long_read_timeout(monkeypatch):
    """Blocking httpx.Client.post + cancel must return within ~2s even if read timeout is 60s.

    Uses a fake Client whose ``post`` blocks until ``close`` is called (simulates a
    hung read). No real sockets — conftest network isolation stays intact.
    """
    import app.agents.llm as llm_mod

    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(config, "LLM_CONNECT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(config, "CANCEL_FORCE_MS", 200)
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key-not-real")
    monkeypatch.setattr(config, "DEEPSEEK_BASE_URL", "http://example.invalid")

    released = threading.Event()

    class SlowResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "late"}, "finish_reason": "stop"}]}

    class SlowClient:
        def __init__(self, *a, **k):
            self._closed = False

        def post(self, *a, **k):
            # Block until close() (cancel path) or a long ceiling.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self._closed or released.is_set():
                    raise httpx.HTTPError("client closed")
                time.sleep(0.05)
            return SlowResponse()

        def close(self):
            self._closed = True
            released.set()

    monkeypatch.setattr(llm_mod.httpx, "Client", SlowClient)
    # Also ensure module-level httpx.Client is the same object path
    monkeypatch.setattr(httpx, "Client", SlowClient)

    client = DeepSeekClient(timeout=60.0)
    token = CancelToken()
    box: dict = {}

    def worker():
        try:
            client.chat([{"role": "user", "content": "hi"}], tools=[], cancel_token=token)
            box["ok"] = True
        except LLMCancelled as exc:
            box["cancelled"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            box["err"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.25)
    t0 = time.monotonic()
    token.set("用户取消")
    t.join(timeout=5.0)
    elapsed = time.monotonic() - t0
    assert not t.is_alive(), f"chat still blocked; box={box}"
    assert elapsed < 2.0, f"cancel waited {elapsed:.2f}s (should be << 60s); box={box}"
    assert "cancelled" in box, box
