"""Run event bus + durable SQLite log for SSE.

Every publish writes an append-only ``run_events`` row (survives restart) and
pushes to the in-process bus. SSE clients reconnect with ``?after_seq=N`` to
replay, then live-tail the bus and/or poll the DB (``EVENTS_DB_TAIL``).
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from typing import Any


class RunEventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, list[queue.Queue]] = {}

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        if not run_id:
            return
        payload = dict(event)
        payload.setdefault("run_id", run_id)
        payload.setdefault("at", time.time())
        with self._lock:
            targets = list(self._subs.get(run_id, ()))
            targets.extend(self._subs.get("*", ()))
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def subscribe(self, run_id: str, *, maxsize: int = 64) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: queue.Queue) -> None:
        with self._lock:
            lst = self._subs.get(run_id) or []
            try:
                lst.remove(q)
            except ValueError:
                pass
            if not lst and run_id in self._subs:
                del self._subs[run_id]


bus = RunEventBus()


def publish_run_event(run_id: str, type_: str, **fields: Any) -> int | None:
    """Persist + in-process publish. Returns the durable ``seq`` (or None on DB failure)."""
    if not run_id:
        return None
    seq: int | None = None
    with contextlib.suppress(Exception):
        from app import storage
        seq = storage.append_run_event(run_id, type_, fields)
    payload = {"type": type_, **fields}
    if seq is not None:
        payload["seq"] = seq
    bus.publish(run_id, payload)
    return seq
