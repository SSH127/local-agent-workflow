"""Unit tests for the unified approval domain model (approvals.py)."""
from __future__ import annotations

from app.engine.approvals import (
    APPROVAL_EXPIRED,
    ApprovalGate,
    Authorization,
)


def test_authorization_single_use_exact_match():
    auth = Authorization("a1", "r1", "write_file", {"path": ".env", "content": "x"}, granted=True)
    assert auth.check("write_file", {"path": ".env", "content": "x"}) is True
    # Single-use: the same grant cannot authorize a second execution.
    assert auth.check("write_file", {"path": ".env", "content": "x"}) is False


def test_authorization_rejects_param_change():
    auth = Authorization("a1", "r1", "write_file", {"path": ".env", "content": "x"}, granted=True)
    assert auth.check("write_file", {"path": ".env", "content": "y"}) is False


def test_authorization_rejects_wrong_tool():
    auth = Authorization("a1", "r1", "write_file", {"path": ".env", "content": "x"}, granted=True)
    assert auth.check("run_project_command", {"path": ".env"}) is False


def test_authorization_denied_never_allows():
    auth = Authorization("a1", "r1", "write_file", {"path": ".env", "content": "x"}, granted=False)
    assert auth.check("write_file", {"path": ".env", "content": "x"}) is False


def test_gate_duplicate_resolve_no_double_effect():
    gate = ApprovalGate()
    record = gate.request("r1", "write_file", {"path": ".env"}, "risk")
    assert gate.resolve_if_pending(record.approval_id, True, None) == "resolved"
    # Repeating the same decision is an idempotent no-op, not a second effect.
    assert gate.resolve_if_pending(record.approval_id, True, None) == "already_resolved"
    assert gate.resolve_if_pending(record.approval_id, False, None) == "already_resolved"


def test_gate_invalid_id_not_found():
    gate = ApprovalGate()
    assert gate.resolve_if_pending("does-not-exist", True, None) == "not_found"


def test_gate_wait_timeout_expires():
    gate = ApprovalGate()
    record = gate.request("r1", "write_file", {"path": ".env"}, "risk")
    assert gate.wait(record.approval_id, timeout=0.05) == APPROVAL_EXPIRED


def test_gate_request_bound_to_run_tool_args():
    gate = ApprovalGate()
    record = gate.request("r1", "write_file", {"path": ".env", "content": "x"}, "覆盖受保护文件")
    assert record.run_id == "r1"
    assert record.tool_name == "write_file"
    assert record.args == {"path": ".env", "content": "x"}
    assert record.risk_reason == "覆盖受保护文件"
    # The pending request is discoverable for its run.
    assert gate.get_pending_for_run("r1") is record
    assert gate.get_pending_for_run("other-run") is None
