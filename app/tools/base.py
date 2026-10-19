"""Tool base types and the shared execution context."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import config


@dataclass
class ReadTracker:
    """Tracks which line ranges of which files a run has already read.

    The executor creates one instance per run and shares it across every tool
    call, so ``read_file`` can report overlaps and steer the model away from
    re-reading the same content. Ranges are 1-based and inclusive.
    """

    _ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # Content hash of the version the run actually saw for each path. Write
    # tools compare against it so a stale write cannot clobber a file another
    # process changed after the read.
    _versions: dict[str, str] = field(default_factory=dict)

    def record_version(self, path: str, sha256: str | None) -> None:
        if sha256:
            self._versions[path] = sha256

    def version(self, path: str) -> str | None:
        return self._versions.get(path)

    def forget_version(self, path: str) -> None:
        self._versions.pop(path, None)

    def record(self, path: str, start: int, end: int) -> tuple[int, list[tuple[int, int]]]:
        """Record ``[start, end]`` (inclusive) and return
        ``(overlap_line_count, merged_ranges)``.``overlap_line_count`` is how many
        of the requested lines had already been read before this call."""
        if start > end:
            return 0, list(self._ranges.get(path, []))
        existing = list(self._ranges.get(path, []))
        overlap = 0
        for (s, e) in existing:
            lo = max(s, start)
            hi = min(e, end)
            if lo <= hi:
                overlap += hi - lo + 1
        merged = existing + [(start, end)]
        merged.sort()
        out: list[tuple[int, int]] = []
        for (s, e) in merged:
            if out and s <= out[-1][1] + 1:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        self._ranges[path] = out
        return overlap, out

    def covered(self, path: str) -> list[tuple[int, int]]:
        """Merged already-read ranges for ``path`` (1-based inclusive)."""
        return list(self._ranges.get(path, []))


@dataclass
class ToolContext:
    """Read-only context handed to each tool invocation.

    ``workspace`` is the run's own (already-validated) workspace root; when it
    is None the tool falls back to the process-wide ``config.WORKSPACE_PATH``.
    It is passed per-invocation (never mutated globally), so concurrent runs
    each operate inside their own workspace. ``read_tracker`` is shared within
    one run (never across runs) and lets file tools avoid re-reading.
    """

    run_id: str
    # ``approve`` is injected by the executor; tools that need approval declare
    # it via ``requires_approval`` and the executor handles the gate.
    approval_check: Callable[[str, dict[str, Any]], bool] | None = None
    workspace: Path | None = None
    read_tracker: ReadTracker | None = None
    # The run's ``ProjectContext`` (cached file/index view). Optional: tools
    # fall back to direct disk access when it is absent (e.g. isolated tests).
    project: Any = None
    # The running subtask's write permission (``app.write_scope.WriteScope``).
    # ``None`` = no task-graph scope (single-subtask / legacy mode).
    write_scope: Any = None
    # Cooperative cancel: callable returning a reason string when the run /
    # subtask should stop (``None`` = keep going). Polled by long-running tools.
    cancel_check: Any = None
    # The CancelToken itself (so long-running tools can register their Popen).
    cancel_token: Any = None
    # Identity of the running subtask (for per-subtask scratch paths).
    subtask_id: str | None = None

    @property
    def workspace_path(self) -> Path:
        return self.workspace if self.workspace is not None else config.WORKSPACE_PATH


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]
    # Returns True when this invocation must pause for user approval. The second
    # argument is the run's workspace root (may be None -> process default).
    requires_approval: Callable[[dict[str, Any], Any], bool] = field(
        default=lambda args, workspace=None: False
    )

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_ok(output: str = "", data: Any = None) -> dict[str, Any]:
    return {"ok": True, "output": output, "data": data}


def tool_error(output: str, data: Any = None) -> dict[str, Any]:
    return {"ok": False, "output": output, "data": data}
