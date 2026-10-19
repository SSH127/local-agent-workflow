"""Shared tool-layer exceptions.

Kept in its own module so ``security`` (path/redaction/policy facade) and
``command_policy`` (argument analysis) can both raise the *same* exception class
without importing each other at module import time.
"""
from __future__ import annotations


class SecurityError(Exception):
    """Raised when a requested operation violates the security policy."""
