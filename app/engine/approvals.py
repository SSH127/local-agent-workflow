"""Unified approval lifecycle: domain model + the in-process approval gate.

This is the single source of truth for every approval. An approval is bound to
a specific run, tool, the exact validated arguments and a risk reason; it is
single-use — the executor consumes it the moment the authorized tool actually
runs, so the same grant can never authorize a second, changed operation.

The wait Event is in-memory. Pending approval/input records are also persisted
in the run's SQLite ``state`` (``pending_approval`` / ``pending_input``). After
a process restart the live waiter is gone; ``POST /approve`` or ``/input``
detects ``not_found`` on the gate and resumes with the stored decision (see
``app/main.py``). ``adopt`` can rehydrate a DB record into a fresh Event when a
new waiter must be attached without minting a new approval_id.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

APPROVAL_PENDING = "pending"
APPROVAL_GRANTED = "granted"
APPROVAL_DENIED = "denied"
APPROVAL_EXPIRED = "expired"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_args(args: Any) -> str:
    """Stable string for comparing argument sets (order-insensitive)."""
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class ApprovalRecord:
    """A pending/resolved approval request, bound to a run, tool and arguments."""

    approval_id: str
    run_id: str
    tool_name: str
    args: dict[str, Any]
    risk_reason: str
    status: str = APPROVAL_PENDING
    reason: str | None = None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "risk_reason": self.risk_reason,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRecord:
        return cls(
            approval_id=str(data.get("approval_id", "")),
            run_id=str(data.get("run_id", "")),
            tool_name=str(data.get("tool_name", "")),
            args=dict(data.get("args") or {}),
            risk_reason=str(data.get("risk_reason", "")),
            status=str(data.get("status", APPROVAL_PENDING)),
            reason=data.get("reason"),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class ApprovalDecision:
    """What the executor receives after (or instead of) a blocking wait."""

    approval_id: str
    approved: bool
    reason: str | None = None


@dataclass
class Authorization:
    """Backend-generated, single-use authorization for exactly one tool call.

    The handler verifies the operation against this object (tool name + exact
    arguments) rather than trusting any model-supplied ``approved`` field.
    """

    approval_id: str
    run_id: str
    tool_name: str
    args: dict[str, Any]
    granted: bool
    _consumed: bool = False

    def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Return True exactly once, and only for the matching tool + arguments."""
        if not self.granted or self._consumed:
            return False
        if self.tool_name != tool_name:
            return False
        if canonical_args(self.args) != canonical_args(args):
            return False
        self._consumed = True
        return True


class ApprovalGate:
    """In-process registry of approval requests with idempotent resolution.

    Keyed by ``approval_id``; a per-run index maps the currently active request.
    ``resolve_if_pending`` is atomic under the lock, so duplicate approve/deny
    calls are no-ops that cannot cause double side effects.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, ApprovalRecord] = {}
        self._events: dict[str, threading.Event] = {}
        self._by_run: dict[str, str] = {}

    def request(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=uuid4().hex,
            run_id=run_id,
            tool_name=tool_name,
            args=dict(args),
            risk_reason=risk_reason,
        )
        with self._lock:
            self._records[record.approval_id] = record
            self._by_run[run_id] = record.approval_id
            self._events[record.approval_id] = threading.Event()
        return record

    def wait(self, approval_id: str, timeout: float) -> str:
        """Block until resolved; on timeout mark the request expired. Returns status.

        If the Event was lost but the record is still pending (e.g. adopt without
        waiters yet), a fresh Event is created so a reattached waiter can block.
        """
        with self._lock:
            record = self._records.get(approval_id)
            event = self._events.get(approval_id)
            if event is None:
                if record is not None and record.status == APPROVAL_PENDING:
                    event = threading.Event()
                    self._events[approval_id] = event
                else:
                    return APPROVAL_EXPIRED
        signaled = event.wait(timeout)
        with self._lock:
            record = self._records.get(approval_id)
            if record is None:
                return APPROVAL_EXPIRED
            if not signaled and record.status == APPROVAL_PENDING:
                record.status = APPROVAL_EXPIRED
            return record.status

    def resolve_if_pending(self, approval_id: str, approved: bool, reason: str | None = None) -> str:
        """Atomically resolve a pending request. Returns one of
        ``"resolved"`` | ``"already_resolved"`` | ``"not_found"``."""
        with self._lock:
            record = self._records.get(approval_id)
            if record is None:
                return "not_found"
            if record.status != APPROVAL_PENDING:
                return "already_resolved"
            record.status = APPROVAL_GRANTED if approved else APPROVAL_DENIED
            record.reason = reason
            event = self._events.get(approval_id)
        if event is not None:
            event.set()
        return "resolved"

    def get(self, approval_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(approval_id)

    def get_pending_for_run(self, run_id: str) -> ApprovalRecord | None:
        with self._lock:
            approval_id = self._by_run.get(run_id)
            if approval_id is None:
                return None
            return self._records.get(approval_id)


    def adopt(self, record: ApprovalRecord) -> ApprovalRecord:
        """Rehydrate a persisted pending record into this gate with a fresh Event.

        Used when a resume path needs to wait again on the *same* approval_id
        (e.g. tests simulating thread-gone + reattach). Idempotent if the id
        is already registered as pending.
        """
        with self._lock:
            existing = self._records.get(record.approval_id)
            if existing is not None and existing.status == APPROVAL_PENDING:
                if record.approval_id not in self._events:
                    self._events[record.approval_id] = threading.Event()
                self._by_run[record.run_id] = record.approval_id
                return existing
            self._records[record.approval_id] = record
            self._by_run[record.run_id] = record.approval_id
            self._events[record.approval_id] = threading.Event()
            return record

    def clear_run(self, run_id: str) -> None:
        with self._lock:
            approval_id = self._by_run.pop(run_id, None)
            if approval_id is not None:
                self._records.pop(approval_id, None)
                self._events.pop(approval_id, None)
