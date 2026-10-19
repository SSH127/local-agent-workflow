"""Best-effort Windows Job Object helpers (no admin required).

Attaches a child process to a job with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so
closing/killing the job reaps the whole tree. On non-Windows this is a no-op.
"""
from __future__ import annotations

import contextlib
import ctypes
import sys
from typing import Any

# HANDLE values we keep alive for the lifetime of the Popen.
_JOBS: list[Any] = []


def attach_kill_on_close_job(proc: Any) -> Any | None:
    """Create a kill-on-close job and assign ``proc``. Returns job handle or None."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return None
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE often needed; OpenProcess if handle insufficient
        handle = getattr(proc, "_handle", None)
        if handle is None:
            PROCESS_ALL = 0x1F0FFF
            handle = kernel32.OpenProcess(PROCESS_ALL, False, int(proc.pid))
            if not handle:
                kernel32.CloseHandle(job)
                return None
            owned = True
        else:
            owned = False
        try:
            if not kernel32.AssignProcessToJobObject(job, handle):
                kernel32.CloseHandle(job)
                return None
        finally:
            if owned:
                kernel32.CloseHandle(handle)
        _JOBS.append(job)
        return job
    except Exception:  # noqa: BLE001 — optional hardening
        return None


def close_job(job: Any) -> None:
    if job is None or sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        ctypes.windll.kernel32.CloseHandle(job)  # type: ignore[attr-defined]
    with contextlib.suppress(ValueError):
        _JOBS.remove(job)
