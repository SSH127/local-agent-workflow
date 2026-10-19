"""Task-level memory: a per-run structured record of what has been established.

Each run owns an independent ``RunMemory`` that captures, separately:

* ``verified`` facts — confirmed by real tool output (file content read,
  command exit codes, test results);
* ``inferred`` conclusions — model statements that have not been backed by a
  tool result;
* ``pending`` items — things still to verify or do.

The executor updates it as tools run and persists it in the run state, so the
next turn (and a later resume) reads memory instead of re-deriving conclusions
from the raw history. New evidence can update or discard old conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMemory:
    goal: str = ""
    # The actual, validated target directory for this run. Persisted so a
    # resumed task can verify it is still operating on the same project.
    target_workspace: str = ""
    constraints: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    confirmed_locations: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    # Post-write content hash per modified file. A resumed run compares these
    # against the current disk state so a completed edit is not replayed.
    modified_file_states: dict[str, str] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)
    test_evidence: list[dict[str, Any]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    phase: str = ""
    remaining_budget: int = 0
    # Why the previous attempt stopped (empty while running); persisted in the
    # checkpoint for a precise, non-success terminal state.
    stop_reason: str = ""
    last_summary: str = ""

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "target_workspace": self.target_workspace,
            "constraints": list(self.constraints),
            "plan": list(self.plan),
            "completed_steps": list(self.completed_steps),
            "confirmed_locations": list(self.confirmed_locations),
            "findings": list(self.findings),
            "modified_files": list(self.modified_files),
            "modified_file_states": dict(self.modified_file_states),
            "commands": list(self.commands),
            "test_evidence": list(self.test_evidence),
            "pending": list(self.pending),
            "phase": self.phase,
            "remaining_budget": self.remaining_budget,
            "stop_reason": self.stop_reason,
            "last_summary": self.last_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RunMemory:
        if not data:
            return cls()
        m = cls()
        m.goal = str(data.get("goal", ""))
        m.target_workspace = str(data.get("target_workspace", ""))
        m.constraints = [str(x) for x in data.get("constraints", [])]
        m.plan = [str(x) for x in data.get("plan", [])]
        m.completed_steps = list(data.get("completed_steps", []))
        m.confirmed_locations = [str(x) for x in data.get("confirmed_locations", [])]
        m.findings = list(data.get("findings", []))
        m.modified_files = [str(x) for x in data.get("modified_files", [])]
        m.modified_file_states = {
            str(k): str(v) for k, v in dict(data.get("modified_file_states") or {}).items()
        }
        m.commands = list(data.get("commands", []))
        m.test_evidence = list(data.get("test_evidence", []))
        m.pending = [str(x) for x in data.get("pending", [])]
        m.phase = str(data.get("phase", ""))
        m.remaining_budget = int(data.get("remaining_budget", 0) or 0)
        m.stop_reason = str(data.get("stop_reason", ""))
        m.last_summary = str(data.get("last_summary", ""))
        return m

    # ------------------------------------------------------------------ records
    def record_step(self, tool: str, ok: bool, summary: str, *, max_steps: int = 80) -> None:
        self.completed_steps.append({"tool": tool, "ok": bool(ok), "summary": summary[:300]})
        if len(self.completed_steps) > max_steps:
            self.completed_steps = self.completed_steps[-max_steps:]

    def record_modified_file(self, rel_path: str, sha256: str | None = None) -> None:
        if not rel_path:
            return
        if rel_path not in self.modified_files:
            self.modified_files.append(rel_path)
        if sha256:
            self.modified_file_states[rel_path] = sha256

    def record_command(self, program: str, argv: list[str], exit_code: Any, ok: bool) -> None:
        self.commands.append({
            "program": program,
            "args": [str(a) for a in argv][:20],
            "exit_code": exit_code,
            "ok": bool(ok),
        })
        if len(self.commands) > 40:
            self.commands = self.commands[-40:]

    def record_test_evidence(self, program: str, argv: list[str], ok: bool, summary: str) -> None:
        self.test_evidence.append({
            "program": program,
            "args": [str(a) for a in argv][:20],
            "ok": bool(ok),
            "summary": summary[:300],
        })
        if len(self.test_evidence) > 40:
            self.test_evidence = self.test_evidence[-40:]

    def add_finding(self, text: str, kind: str = "verified") -> None:
        """Record a conclusion with its epistemic status: one of
        ``verified`` (backed by tool output), ``inferred`` (model reasoning) or
        ``pending`` (still to verify)."""
        if kind not in {"verified", "inferred", "pending"}:
            kind = "inferred"
        # A new verified fact supersedes a conflicting pending item.
        if kind == "verified":
            self.pending = [p for p in self.pending if p != text]
        self.findings.append({"text": text, "kind": kind})

    # ------------------------------------------------------------------ assembly
    def context_block(self, max_chars: int = 4000) -> str:
        """Compact, bounded summary of the memory to inject into the request.

        Carries only what is still relevant (plan, modified files, command/test
        evidence, pending items) — never the raw historical tool outputs.
        """
        parts: list[str] = []
        if self.plan:
            parts.append("计划：\n" + "\n".join(f"- {p}" for p in self.plan))
        if self.modified_files:
            parts.append("已修改文件：" + ", ".join(self.modified_files[-30:]))
        if self.commands:
            cmd = "；".join(
                f"{c['program']} {' '.join(c['args'])} (退出码 {c['exit_code']})"
                for c in self.commands[-10:]
            )
            parts.append("已运行命令：" + cmd)
        if self.test_evidence:
            ev = "；".join(
                f"{e['program']} {' '.join(e['args'])} → {'通过' if e['ok'] else '失败'}"
                for e in self.test_evidence[-10:]
            )
            parts.append("测试证据：" + ev)
        if self.confirmed_locations:
            parts.append("已确认位置：" + ", ".join(self.confirmed_locations[-20:]))
        if self.pending:
            parts.append("待办/待验证：" + "；".join(self.pending[-10:]))
        if self.phase:
            parts.append(f"当前阶段：{self.phase}，剩余轮次预算：{self.remaining_budget}")
        text = "\n".join(parts)
        return ("[任务记忆]\n" + text)[:max_chars]
