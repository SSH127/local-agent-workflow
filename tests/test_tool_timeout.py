"""Per-tool timeouts are real: a hung handler no longer blocks the loop, the
model-supplied command timeout is clamped, and a timed-out command is killed."""
from __future__ import annotations

import time

from app import config
from app.agents import agent_executor
from app.tools import registry
from app.tools.base import Tool, ToolContext


def test_hung_handler_returns_at_timeout(monkeypatch):
    def hung_handler(args, ctx):
        time.sleep(3)
        return {"ok": True, "output": "too late", "data": None}

    monkeypatch.setattr(
        agent_executor, "get_tool",
        lambda name: Tool(name=name, description="hung", parameters={}, handler=hung_handler),
    )
    started = time.monotonic()
    result, ok = agent_executor._run_tool_handler(
        "hung", {}, ToolContext(run_id="r1"), tool_timeout=0.2,
    )
    elapsed = time.monotonic() - started
    assert ok is False
    assert "超时" in result["output"]
    # Previously ``with ThreadPoolExecutor`` waited for the 3s handler on exit.
    assert elapsed < 1.5


def test_handler_exception_is_reported():
    def boom(args, ctx):
        raise RuntimeError("bad")

    tool = Tool(name="boom", description="", parameters={}, handler=boom)
    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(agent_executor, "get_tool", lambda name: tool)
    try:
        result, ok = agent_executor._run_tool_handler(
            "boom", {}, ToolContext(run_id="r1"), tool_timeout=2,
        )
    finally:
        mp.undo()
    assert ok is False
    assert "RuntimeError" in result["output"]


def test_command_timeout_is_clamped(monkeypatch):
    monkeypatch.setattr(config, "TOOL_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(config, "MAX_COMMAND_TIMEOUT_SECONDS", 600.0)
    f = registry.command_timeout_seconds
    assert f({}) == 60.0
    assert f({"timeout_seconds": 30}) == 30.0
    assert f({"timeout_seconds": 99999}) == 600.0
    assert f({"timeout_seconds": "abc"}) == 60.0
    assert f({"timeout_seconds": -5}) == 60.0
    assert f({"timeout_seconds": float("nan")}) == 60.0
    monkeypatch.setattr(config, "MAX_COMMAND_TIMEOUT_SECONDS", 10.0)
    assert f({}) == 10.0


def test_outer_wait_covers_command_timeout(monkeypatch):
    """The executor's outer wait for run_project_command is sized from the
    (clamped) command timeout, so the handler can kill its own subprocess."""
    seen: dict[str, float] = {}

    def fake_handler(args, ctx):
        return {"ok": True, "output": "", "data": None}

    monkeypatch.setattr(config, "MAX_COMMAND_TIMEOUT_SECONDS", 50.0)
    real_thread = agent_executor.threading.Thread

    class SpyThread(real_thread):
        def join(self, timeout=None):
            seen["timeout"] = timeout
            return super().join(timeout)

    monkeypatch.setattr(agent_executor.threading, "Thread", SpyThread)
    monkeypatch.setattr(
        agent_executor, "get_tool",
        lambda name: Tool(name=name, description="", parameters={}, handler=fake_handler),
    )
    agent_executor._run_tool_handler(
        "run_project_command", {"timeout_seconds": 9999}, ToolContext(run_id="r"),
        tool_timeout=1.0,
    )
    assert seen["timeout"] == 60.0  # clamp 50 + 10s cleanup margin


def test_timed_out_command_is_killed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKSPACE_PATH", tmp_path)
    monkeypatch.setattr(config, "MAX_COMMAND_TIMEOUT_SECONDS", 4.0)
    done = tmp_path / "done.marker"
    (tmp_path / "test_slow_child.py").write_text(
        "import time, pathlib\n"
        "def test_slow():\n"
        "    time.sleep(8)\n"
        f"    pathlib.Path({str(done)!r}).write_text('x')\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = registry._run_project_command(
        {"program": "python", "args": ["-m", "pytest", "test_slow_child.py", "-q"],
         "cwd": ".", "timeout_seconds": 9999},
        ToolContext(run_id="r", workspace=tmp_path),
    )
    elapsed = time.monotonic() - started
    assert result["ok"] is False
    assert "超时" in result["output"]
    assert elapsed < 8
    # Had the process survived, it would write the marker ~8s after start.
    time.sleep(max(0.0, 10 - elapsed))
    assert not done.exists()
