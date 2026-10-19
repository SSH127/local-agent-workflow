"""Cooperative cancellation for a running agent / subtask.

A :class:`CancelToken` is shared by the control loop, the ReAct executor and
the tool layer. Setting it:

* stops new tool starts and new LLM turns (the executor polls between them);
* aborts any in-flight LLM HTTP client registered with the token;
* lets ``run_project_command`` kill its process tree (same path as a timeout);
* lets waiting approvals / ask_user be resolved as denied with reason
  ``cancelled`` (the gate is still woken via ``resolve_if_pending``).

The token is per run (and optionally refined per subtask); it is never cleared
once set — a cancelled attempt must not resume as a normal attempt.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable


class CancelToken:
    """Thread-safe cooperative cancel flag."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancelled"
        self._lock = threading.Lock()
        self._procs: list[object] = []
        self._http: list[object] = []

    def set(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._reason = reason or "cancelled"
            self._event.set()
            procs = list(self._procs)
            clients = list(self._http)
        for proc in procs:
            self._kill(proc)
        for client in clients:
            self._close_http(client)

    def is_set(self) -> bool:
        return self._event.is_set()

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    @staticmethod
    def _kill(proc: object) -> None:
        with contextlib.suppress(Exception):
            from app.tools.registry import _kill_process_tree
            _kill_process_tree(proc)  # type: ignore[arg-type]
            return
        with contextlib.suppress(Exception):
            kill = getattr(proc, "kill", None)
            if callable(kill):
                kill()

    @staticmethod
    def _close_http(client: object) -> None:
        with contextlib.suppress(Exception):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def register_proc(self, proc: object) -> None:
        with self._lock:
            self._procs.append(proc)
            already = self._event.is_set()
        if already:
            self._kill(proc)

    def unregister_proc(self, proc: object) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._procs.remove(proc)

    def register_http(self, client: object) -> None:
        """Register an in-flight HTTP client; closed immediately if already cancelled."""
        with self._lock:
            self._http.append(client)
            already = self._event.is_set()
        if already:
            self._close_http(client)

    def unregister_http(self, client: object) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._http.remove(client)

    def check(self) -> str | None:
        return self.reason() if self.is_set() else None


def as_check(token: CancelToken | None) -> Callable[[], str | None] | None:
    if token is None:
        return None
    return token.check
