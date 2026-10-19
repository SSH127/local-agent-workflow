"""Loaded via PYTHONPATH when COMMAND_DENY_NETWORK injects this directory.

Blocks outbound TCP for *this Python interpreter only* (socket.connect /
create_connection). Native binaries and other languages are unaffected.
"""
from __future__ import annotations

import os
import socket

if os.environ.get("AGENT_DENY_NETWORK", "").strip() in {"1", "true", "yes", "on"}:
    _OrigSocket = socket.socket

    class _NoNetSocket(_OrigSocket):  # type: ignore[misc,valid-type]
        def connect(self, address):
            raise OSError(
                "COMMAND_DENY_NETWORK: outbound socket.connect blocked "
                "(Python child only; not a kernel firewall)"
            )

        def connect_ex(self, address):
            self.connect(address)
            return 0

    def _no_create_connection(*args, **kwargs):
        raise OSError(
            "COMMAND_DENY_NETWORK: socket.create_connection blocked "
            "(Python child only; not a kernel firewall)"
        )

    socket.socket = _NoNetSocket  # type: ignore[misc,assignment]
    socket.create_connection = _no_create_connection  # type: ignore[assignment]
