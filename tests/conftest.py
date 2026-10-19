"""Shared test helpers."""
from __future__ import annotations

import json
import socket
from typing import Any

import httpx
import pytest

from app import config, storage
from app.agents.llm import LLMResponse


class NetworkIsolationError(RuntimeError):
    """Raised when a test attempts a real network call without an explicit mock."""


def _block_network(*args: Any, **kwargs: Any) -> Any:
    """Reject any real network access: raise, never return a fake success."""
    raise NetworkIsolationError(
        "测试期间禁止真实网络访问；请显式 mock 相关函数以提供结果。"
    )


@pytest.fixture(autouse=True)
def _block_network_fixture(monkeypatch):
    """Block every real network entry point the project actually uses.

    Project code only performs outbound network I/O through three callables:
    ``httpx.post`` (DeepSeek chat), ``httpx.request`` (check_http) and
    ``socket.create_connection`` (check_port). Blocking them makes "no real
    network in tests" the default. Tests that genuinely need a network result
    must explicitly re-mock the specific function; their ``monkeypatch``
    override takes precedence over this default blocker.

    ``httpx.Client`` / ``httpx.AsyncClient`` are intentionally NOT patched:
    project code never uses them, and ``TestClient`` relies on an in-process
    ASGI transport (no sockets).
    """
    monkeypatch.setattr(httpx, "post", _block_network)
    monkeypatch.setattr(httpx, "request", _block_network)
    monkeypatch.setattr(socket, "create_connection", _block_network)


@pytest.fixture(autouse=True)
def _isolate_llm_config(monkeypatch):
    """Blank the DeepSeek API key for every test.

    ``config.DEEPSEEK_API_KEY`` is read from the environment at import time and
    bound to a module attribute, so ``monkeypatch.delenv("DEEPSEEK_API_KEY")``
    alone would NOT prevent a real client from using it. Reset the module
    attribute instead so no test can accidentally reach the real API.
    """
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")


@pytest.fixture(autouse=True)
def _default_require_approval(monkeypatch):
    """Run every test with approval gating ON (the shipped default), regardless
    of the developer's local ``.env``. Tests for ``REQUIRE_APPROVAL=false``
    patch it explicitly."""
    monkeypatch.setattr(config, "REQUIRE_APPROVAL", True)


@pytest.fixture(autouse=True)
def _legacy_workflow_defaults(monkeypatch):
    """Existing tests drive the ReAct executor directly with a scripted FakeLLM.
    Disable the planner, plan approval and the LLM final review so no scripted
    response is consumed by planning (the run becomes a single-subtask legacy
    task graph). Workflow tests enable these explicitly."""
    monkeypatch.setattr(config, "PLANNER_ENABLED", False)
    monkeypatch.setattr(config, "PLAN_REQUIRES_APPROVAL", False)
    monkeypatch.setattr(config, "FINAL_VERIFY_LLM", False)


@pytest.fixture(autouse=True)
def _isolate_database(monkeypatch, tmp_path):
    """Point storage at a per-test temporary SQLite file.

    ``app/storage.py`` binds ``DATABASE_PATH`` at import time via
    ``from app.config import DATABASE_PATH``, so ``_connect`` reads the
    ``storage`` module attribute (not ``config``). Patch the storage attribute
    (not config) so API, background threads and ``StorageRecorder`` all share
    the isolated temporary database and never touch the development database.
    """
    monkeypatch.setattr(storage, "DATABASE_PATH", tmp_path / "test.sqlite3")


@pytest.fixture(autouse=True)
def _isolate_context_cache(monkeypatch, tmp_path):
    """Point the project context cache at a per-test temporary SQLite file.

    ``ProjectContext`` reads ``config.CONTEXT_CACHE_DB`` at construction time,
    so patching the config attribute routes every run (executor + direct tool
    tests) to an isolated cache and never touches the development cache DB.
    """
    monkeypatch.setattr(config, "CONTEXT_CACHE_DB", tmp_path / "context_cache.sqlite3")


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> LLMResponse:
    """Build an LLMResponse that requests a single tool call."""
    return LLMResponse(
        content="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    )


def multi_tool_call(calls: list[tuple[str, str, dict[str, Any]]]) -> LLMResponse:
    """Build an LLMResponse that requests several tool calls in one message."""
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            for (cid, name, args) in calls
        ],
    )


def final_answer(text: str) -> LLMResponse:
    return LLMResponse(content=text)


class FakeLLM:
    """Scripted LLM. Pops responses in order; repeats the last one forever."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self._last: LLMResponse | None = None
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        if self.responses:
            self._last = self.responses.pop(0)
            return self._last
        if self._last is not None:
            return self._last
        return final_answer("完成。")
