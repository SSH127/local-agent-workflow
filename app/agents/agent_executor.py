"""The real agent execution loop (ReAct style) with hard limits.

The executor calls the LLM, executes tools, feeds real tool results back, and
enforces max iterations, per-tool timeout and a total task deadline. It never
treats model text as proof of task completion: success is decided only from real
execution evidence, never from the model's prose.

Success (``succeeded``) requires all of the following:

* at least one tool call was actually made (``tool_calls_total > 0``);
* if files were modified, a *verification* command (pytest/ruff/build/...) must
  have passed after the last modification;
* the most recent verification command must not be a failure (a failed
  verification is only cleared by a later passing verification — an unrelated
  successful tool never masks it);
* the last executed tool step must not have failed (this also implies at least
  one tool succeeded, since a fully-failed run ends on a failure).

Read-only / diagnostic tasks (read_file, check_http, check_port, ...) are
verified by their own tool success; pytest is never forced on them.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app import config
from app.agents.llm import LLMCancelled
from app.context_builder import assemble_messages, estimate_prompt_tokens
from app.engine.approvals import ApprovalDecision, Authorization, canonical_args
from app.memory import RunMemory
from app.project_context import ProjectContext
from app.tools import get_tool
from app.tools.base import ReadTracker, ToolContext, tool_error, tool_ok
from app.tools.security import (
    SecurityError,
    approval_reason,
    is_mutating_command,
    is_verification_command,
    redact_kv,
    resolve_workspace_path,
    resolve_workspace_root,
)

# Status values shared with the API/schema layer.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_WAITING_APPROVAL = "waiting_for_approval"
STATUS_WAITING_INPUT = "waiting_for_input"
# Task-decomposition control loop statuses.
STATUS_WAITING_PLAN_APPROVAL = "waiting_for_plan_approval"
STATUS_WAITING_SUBTASK_APPROVAL = "waiting_for_subtask_approval"
STATUS_NEEDS_ATTENTION = "needs_attention"
STATUS_CANCELLED = "cancelled"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _usage_source(token_stats: dict[str, Any]) -> str:
    """Classify where the token numbers came from: ``api`` (every call returned
    real usage), ``estimated`` (no call did) or ``mixed`` (some did)."""
    real = int(token_stats.get("calls_with_real_usage", 0) or 0)
    missing = int(token_stats.get("calls_without_real_usage", 0) or 0)
    if real and missing:
        return "mixed"
    if real:
        return "api"
    return "estimated"


# Commands whose purpose is to *verify* code correctness (they produce pass/fail
# evidence) are classified in ``app.tools.security`` (single source of truth),
# which keeps read-only verification (``ruff check``) distinct from file
# mutation (``ruff format``, ``ruff check --fix``).


def is_verifying_tool_call(name: str, args: dict[str, Any]) -> bool:
    """True if the tool call produces pass/fail *verification evidence*.

    Only ``run_project_command`` can be a verification tool; reads/writes and
    diagnostic tools (check_http, check_port, ...) are not verifications. A
    mutating command (formatter/autofixer) is explicitly *not* a verification.
    """
    if name != "run_project_command":
        return False
    return is_verification_command(
        str(args.get("program", "")),
        [str(a) for a in args.get("args", [])],
    )


# ---------------------------------------------------------------------------
# Turn-phase classification and per-phase budgets
# ---------------------------------------------------------------------------
# A "phase" is derived from the tool calls a single model turn requested — not a
# forced state machine, so a read-only task never enters the modify/verify
# phases. Priority: a command turn is ``verify``, a write turn is ``modify``, a
# read/list/search turn is ``analyze``.
_READ_TOOLS = {"read_file", "list_files", "search_code", "get_task_logs", "check_port", "check_http"}
# Tools that change project files (all of them go through the write guard).
_MUTATING_TOOLS = {"write_file", "edit_file", "rollback_file"}
# Tools whose successful execution is *productive* modification evidence. A
# rollback restores an older version, so it must not count as completing a
# "fix/implement" goal.
_WRITE_TOOLS = {"write_file", "edit_file"}
# Interactive tool: pauses the run in ``waiting_for_input`` until answered.
ASK_USER_TOOL = "ask_user"

# Terminal stop reasons. These are persisted (run state + checkpoint) and
# logged, so "budget exhausted", "max iterations", "tool failed", "workspace
# rejected" and "model call failed" are never conflated in the UI.
STOP_COMPLETED = "completed"
STOP_MAX_ITERATIONS = "max_iterations"
STOP_MODEL_CALL_BUDGET = "model_call_budget"
STOP_PHASE_BUDGET = "phase_budget"
STOP_ANALYZE_TIMEOUT = "analyze_timeout"
STOP_TASK_TIMEOUT = "task_timeout"
STOP_MODEL_ERROR = "model_error"
STOP_NO_PROGRESS = "no_progress"
STOP_MUTATION_REQUIRED = "mutation_required"
STOP_VERIFICATION_REQUIRED = "verification_required"
STOP_WORKSPACE_REJECTED = "workspace_rejected"
STOP_TOOL_FAILURE = "tool_failure"
STOP_BLOCKED = "blocked"
STOP_CANCELLED = "cancelled"

PHASE_LABELS = {
    "analyze": "规划与阅读分析",
    "modify": "执行修改",
    "verify": "验证与修复",
}

PHASE_EXHAUSTION = {
    "analyze": "阅读与分析阶段预算耗尽：连续阅读轮次过多，仍未进入修改或验证阶段。",
    "modify": "修改阶段预算耗尽：修改轮次过多，仍未通过验证。",
    "verify": "测试与修复阶段预算耗尽：多次测试/修复仍未满足完成条件。",
}

PHASE_HINTS = {
    "analyze": (
        "阅读与分析阶段预算已用尽。若本任务为只读分析，请直接给出最终总结；"
        "若需修改代码，请停止继续阅读，基于已读内容形成计划并开始修改。"
    ),
    "modify": "修改阶段预算已用尽。请停止继续修改，运行测试验证当前结果并总结。",
    "verify": "测试与修复阶段预算已用尽。请基于最新测试结果给出最终总结。",
}


def classify_phase(names: list[str]) -> str | None:
    """Map a turn's tool names to one phase (or None for a non-tool turn)."""
    if any(n == "run_project_command" for n in names):
        return "verify"
    if any(n in _MUTATING_TOOLS for n in names):
        return "modify"
    if any(n in _READ_TOOLS for n in names):
        return "analyze"
    return None


# ---------------------------------------------------------------------------
# Goal intent + checkpoint helpers
# ---------------------------------------------------------------------------

_READONLY_DIRECTIVE = re.compile(
    r"(只读|只分析|仅分析|不要修改|无需修改|不要改动|只检查|仅检查|"
    r"read[\s-]?only|do not modify|without modifying|no changes?)",
    re.IGNORECASE,
)
_MUTATION_VERB = re.compile(
    r"(修复|修改|实现|添加|新增|增加|删除|移除|重构|优化|更新|调整|改造|"
    r"编写|创建|写入|补全|修好|改正|替换|迁移|升级|接入|支持|解决|"
    r"\bfix\b|\badd\b|\bimplement\b|\bmodify\b|\bchange\b|\bupdate\b|"
    r"\bcreate\b|\bwrite\b|\brefactor\b|\bremove\b|\bdelete\b|\brename\b|"
    r"\bmigrate\b|\bupgrade\b|\bpatch\b|\brepair\b|\bresolve\b|\bedit\b|"
    r"\boptimi[sz]e\b|\bintroduce\b|\bsupport\b)",
    re.IGNORECASE,
)
_QUESTION_PREFIX = re.compile(
    r"(如何|怎么|怎样|是否|为什么|为何|能否|可否|要不要|"
    r"how\s+to|whether|need\s+to|should\s+i)",
    re.IGNORECASE,
)


def goal_requires_mutation(goal: str) -> bool:
    """Heuristically decide whether a goal is a change request.

    This is intentionally conservative: a read-only task (analyse/review/check)
    must still be able to finish without writing a file, while an explicit
    "fix/add/implement/..." goal must produce at least one real modification.
    A mutation verb inside a question ("如何修复", "whether to change") is not
    treated as an instruction to change code.
    """
    text = (goal or "").strip()
    if not text or _READONLY_DIRECTIVE.search(text):
        return False
    for match in _MUTATION_VERB.finditer(text):
        prefix = text[max(0, match.start() - 16): match.start()]
        if _QUESTION_PREFIX.search(prefix):
            continue
        return True
    return False


def _sha256_text(text: str) -> str:
    """Hash of newline-normalized text (matches ``write_guard.content_hash``)."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _checkpoint_modified_index(checkpoint: dict[str, Any] | None) -> dict[str, str]:
    """Return ``{relative_path: post_write_sha256}`` from a checkpoint/memory."""
    if not checkpoint:
        return {}
    index: dict[str, str] = {}
    files = checkpoint.get("modified_files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                path = item.get("path")
                digest = item.get("sha256")
            else:
                path, digest = item, None
            if path and digest:
                index[str(path).replace("\\", "/")] = str(digest)
    memory = checkpoint.get("memory")
    if isinstance(memory, dict):
        states = memory.get("modified_file_states")
        if isinstance(states, dict):
            for path, digest in states.items():
                if path and digest:
                    index.setdefault(str(path).replace("\\", "/"), str(digest))
    return index


def _mutation_already_applied(
    name: str,
    args: dict[str, Any],
    workspace: Path,
    checkpoint: dict[str, Any] | None,
) -> bool:
    """True when a write/edit is already reflected on disk.

    Used on resume (and for a repeated model-issued write) so completed work is
    not replayed. The check requires checkpoint evidence for the path *and* an
    exact current-content match; anything ambiguous is executed normally and
    still goes through the write guard.
    """
    if name not in _WRITE_TOOLS or not checkpoint:
        return False
    path_arg = str(args.get("path") or "").strip()
    if not path_arg:
        return False
    try:
        abs_path = resolve_workspace_path(path_arg, workspace)
        rel_key = abs_path.relative_to(Path(workspace).resolve()).as_posix()
    except (SecurityError, ValueError):
        return False
    index = _checkpoint_modified_index(checkpoint)
    if rel_key not in index and str(abs_path).replace("\\", "/") not in index:
        return False
    if not abs_path.is_file():
        return False
    try:
        current = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    current_norm = current.replace("\r\n", "\n")

    if name == "write_file":
        desired = args.get("content")
        return isinstance(desired, str) and current_norm == desired.replace("\r\n", "\n")
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    return (
        old.replace("\r\n", "\n") not in current_norm
        and new.replace("\r\n", "\n") in current_norm
    )


def _is_plan_announcement(text: str | None) -> bool:
    """True when the model text looks like an implementation plan."""
    if not text:
        return False
    return bool(
        re.search(
            r"(计划|方案|步骤|实施|\bplan\b|\bsteps?\b|\bapproach\b)",
            text,
            re.IGNORECASE,
        )
    )


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


class RunRecorder(Protocol):
    """Persistence hooks used by the executor (wired to storage in production)."""

    def save_step(self, step: dict[str, Any]) -> None: ...
    def set_status(self, status: str, summary: str | None = None, error: str | None = None) -> None: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
    def request_approval(
        self, run_id: str, tool_name: str, args: dict[str, Any], risk_reason: str
    ) -> ApprovalDecision: ...
    # Optional: ``request_input(run_id, question) -> str | None`` blocks until
    # the user answers an ``ask_user`` question. Recorders without it have no
    # user channel, so ``ask_user`` fails cleanly.


@dataclass
class AgentOutcome:
    status: str
    summary: str
    error: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    # Per-phase turn counts and per-tool invocation counts (for logs/diagnostics).
    phase_used: dict[str, int] = field(default_factory=dict)
    tool_counts: dict[str, int] = field(default_factory=dict)
    # Machine-readable termination reason + recoverable checkpoint. A failed
    # outcome always carries a checkpoint so the task can be resumed instead of
    # being replayed blindly from the beginning.
    stop_reason: str = ""
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeState:
    """Serialized conversation + the remaining (already-approved) tool calls,
    used when a paused run is resumed (e.g. after an approval decision or a
    restart). Storing the full batch keeps the conversation consistent: the
    assistant message lists every tool call, so every call must be answered by
    a matching ``tool`` message."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    # List of ``{"name": str, "args": dict, "call_id": str}``.
    pending_tools: list[dict[str, Any]] = field(default_factory=list)
    # A user decision carried across a restart so the resumed run does not
    # re-prompt. Shape: {"approval_id": str, "run_id": str, "tool_name": str,
    # "args": dict, "approved": bool, "reason": str | None}. Only authorizes a
    # single call in the *same* run with identical arguments.
    preapproved: dict[str, Any] | None = None
    # Task memory carried across a restart (see ``app.memory.RunMemory``).
    memory: dict[str, Any] | None = None
    # Accumulated token/cache statistics carried across a restart, so a resumed
    # run neither loses prior usage nor re-counts it.
    token_stats: dict[str, Any] | None = None
    cache_stats: dict[str, Any] | None = None
    # Persisted execution checkpoint (goal/workspace/completed steps/modified
    # files with hashes/test evidence/pending steps/stop reason). A resume uses
    # this to verify the current disk state before doing anything.
    checkpoint: dict[str, Any] | None = None
    # The user's answer to a pending ``ask_user`` call, carried across a
    # restart (the replayed call is answered with it instead of asking again).
    user_input: str | None = None


def _tool_message_content(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)[: config.TOOL_OUTPUT_MAX_CHARS]


def _run_tool_handler(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    tool_timeout: float,
) -> tuple[dict[str, Any], bool]:
    """Execute a tool handler with a hard per-tool timeout (no approval/log).

    The handler runs on a daemon thread and the caller waits at most the
    timeout: a hung handler can no longer block the run loop (the previous
    ``with ThreadPoolExecutor(...)`` waited for the worker on exit, so the
    timeout was not actually enforced). ``run_project_command`` additionally
    kills its own subprocess tree when its command timeout is exceeded; its
    outer wait is sized slightly above that so the handler can clean up.
    """
    tool = get_tool(name)
    if tool is None:
        return tool_error(f"未知工具：{name}"), False

    timeout = tool_timeout
    if name == "run_project_command":
        from app.tools.registry import command_timeout_seconds

        timeout = max(tool_timeout, command_timeout_seconds(args) + 10.0)

    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = tool.handler(args, ctx)
        except Exception as exc:  # noqa: BLE001 - defensive; never crash the loop
            box["error"] = exc

    worker = threading.Thread(target=_target, daemon=True, name=f"tool-{name}")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        result = tool_error(f"工具执行超时（>{timeout:g}s）：{name}")
    elif "error" in box:
        exc = box["error"]
        result = tool_error(f"工具执行异常：{type(exc).__name__}: {exc}")
    else:
        result = box.get("result")
        if not isinstance(result, dict):
            result = tool_error(f"工具返回结果格式异常：{name}")

    return result, bool(result.get("ok"))


def _record_step(recorder: RunRecorder, run_id: str, **fields: Any) -> None:
    # ``seq`` is deliberately left to the persistence layer (see storage.save_step),
    # which assigns a per-run monotonic ordinal so every step gets a unique,
    # reconstructable order — model turns, tool calls and approvals alike.
    seq = fields.pop("seq", None)
    step = {
        "step_id": uuid4().hex,
        "run_id": run_id,
        "ts": utc_now(),
        "kind": fields.pop("kind", "tool"),
        "tool_name": fields.pop("tool_name", None),
        "args": fields.pop("args", {}),
        "status": fields.pop("status", "succeeded"),
        "output": fields.pop("output", None),
        "error": fields.pop("error", None),
    }
    if seq is not None:
        step["seq"] = seq
    recorder.save_step(step)


def _audit_checkpoint_files(
    checkpoint: dict[str, Any] | None, workspace: Path
) -> list[str]:
    """Compare checkpointed post-write hashes with the current disk state.

    This is what makes a resume safe: completed edits are confirmed before the
    model is asked to continue, so it is told not to redo them. A mismatch is
    reported (never silently "fixed") because the file may have changed outside
    the agent.
    """
    index = _checkpoint_modified_index(checkpoint)
    if not index:
        return []
    notes: list[str] = []
    for rel_path, expected in index.items():
        try:
            abs_path = resolve_workspace_path(rel_path, workspace)
        except SecurityError:
            notes.append(f"{rel_path}：检查点路径越界，已忽略（不会访问工作区外文件）。")
            continue
        if not abs_path.is_file():
            notes.append(f"{rel_path}：文件当前不存在（需要重新创建或确认）。")
            continue
        try:
            current = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            notes.append(f"{rel_path}：无法读取当前内容（{exc}）。")
            continue
        if _sha256_text(current) == expected:
            notes.append(f"{rel_path}：当前内容与检查点一致（已完成的修改，不要重复执行）。")
        else:
            notes.append(f"{rel_path}：当前内容与检查点不一致（可能被外部修改，需重新核对）。")
    return notes


def _checkpoint_resume_message(
    checkpoint: dict[str, Any] | None,
    memory: RunMemory,
    workspace: Path,
    file_notes: list[str],
) -> str:
    """A compact, explicit resume directive injected as a user message."""
    stop_reason = ""
    if checkpoint:
        stop_reason = str(checkpoint.get("stop_reason", ""))
    stop_reason = stop_reason or memory.stop_reason or "unknown"
    lines = [
        "[检查点]",
        f"这是中断后恢复的任务。实际目标目录：{workspace}",
        f"上次停止原因：{stop_reason}",
    ]
    if memory.completed_steps:
        recent = memory.completed_steps[-10:]
        lines.append(
            "已完成步骤："
            + "；".join(f"{s.get('tool')}({'成功' if s.get('ok') else '失败'})" for s in recent)
        )
    if memory.modified_files:
        lines.append("已修改文件：" + "、".join(memory.modified_files[-20:]))
    if file_notes:
        lines.append("当前文件状态核对：" + "；".join(file_notes))
    if memory.test_evidence:
        evidence = "；".join(
            f"{e.get('program')} {' '.join(e.get('args', []))}→{'通过' if e.get('ok') else '失败'}"
            for e in memory.test_evidence[-5:]
        )
        lines.append("上次测试结果：" + evidence)
    pending = [
        p for p in memory.pending
        if not p.startswith(("read_file 失败", "list_files 失败", "search_code 失败"))
    ]
    # Pending steps are part of the persisted checkpoint (not only memory), so a
    # resumed run is told exactly what remains even when the compressed memory
    # block is empty.
    for item in (checkpoint or {}).get("pending_steps", []) or []:
        if isinstance(item, str) and item not in pending:
            pending.append(item)
    if pending:
        lines.append("待执行步骤：" + "；".join(pending[-12:]))
    lines.append(
        "请先核对上面的文件状态，不要重复执行已经完成的修改；"
        "从未完成步骤继续，先完成一个范围明确、可验证的修改，再运行相关测试/构建。"
    )
    return "\n".join(lines)


def run_agent(
    run_id: str,
    goal: str,
    llm: LLMClient,
    tools: list[dict[str, Any]],
    recorder: RunRecorder,
    *,
    max_iterations: int | None = None,
    task_timeout: float | None = None,
    tool_timeout: float | None = None,
    workspace: str | Path | None = None,
    resume: ResumeState | None = None,
    plan: list[str] | None = None,
    max_model_calls: int | None = None,
    no_progress_limit: int | None = None,
    analyze_timeout: float | None = None,
    mutation_required: bool | None = None,
    write_scope: Any = None,
    cancel_token: Any = None,
) -> AgentOutcome:
    max_iterations = max_iterations or config.AGENT_MAX_ITERATIONS
    task_timeout = task_timeout if task_timeout is not None else config.TASK_TIMEOUT_SECONDS
    cancel_check = (cancel_token.check if cancel_token is not None else None)
    tool_timeout = tool_timeout if tool_timeout is not None else config.TOOL_TIMEOUT_SECONDS
    max_model_calls = max_model_calls or config.AGENT_MAX_MODEL_CALLS
    no_progress_limit = no_progress_limit or config.AGENT_NO_PROGRESS_LIMIT
    analyze_timeout = (
        analyze_timeout if analyze_timeout is not None else config.AGENT_ANALYZE_TIMEOUT_SECONDS
    )

    # Per-phase budgets are clamped to the total so the phases can never exceed
    # the overall turn budget (they are soft guards, not additive caps).
    phase_budgets = {
        "analyze": min(config.AGENT_PHASE_BUDGET_ANALYZE, max_iterations),
        "modify": min(config.AGENT_PHASE_BUDGET_MODIFY, max_iterations),
        "verify": min(config.AGENT_PHASE_BUDGET_VERIFY, max_iterations),
    }
    phase_used: dict[str, int] = {"analyze": 0, "modify": 0, "verify": 0}
    phase_hint_sent: set[str] = set()
    last_phase: str | None = None
    tool_counts: dict[str, int] = {}
    # One tracker per run: shared across every tool call so ``read_file`` can
    # detect and report duplicate reads. Never shared between runs.
    read_tracker = ReadTracker()

    # ------------------------------------------------------------------
    # Workspace validation happens BEFORE any model call or tool execution.
    # An explicitly supplied workspace that is invalid is a hard failure; it
    # must never silently fall back to the engine's own project directory.
    # ------------------------------------------------------------------
    try:
        attempted_workspace = (
            str(Path(workspace).expanduser().resolve()) if workspace is not None
            else str(Path(config.WORKSPACE_PATH).expanduser().resolve())
        )
    except OSError:
        # A malformed path must still be reported as a rejected workspace, not
        # crash the run before it can persist a stop reason.
        attempted_workspace = str(workspace if workspace is not None else config.WORKSPACE_PATH)
    if workspace is not None:
        try:
            effective_workspace = resolve_workspace_root(workspace)
        except SecurityError as exc:
            reason = f"工作区无效或被拒绝，任务未执行：{exc}"
            checkpoint = {
                "goal": goal,
                "workspace": attempted_workspace,
                "status": STATUS_FAILED,
                "stop_reason": STOP_WORKSPACE_REJECTED,
                "completed_steps": [],
                "modified_files": [],
                "test_results": [],
                "pending_steps": ["修正目标工作区路径后重新创建任务"],
                "iterations": 0,
                "model_calls": 0,
                "phase_used": {},
                "tool_counts": {},
            }
            _record_step(
                recorder, run_id, kind="workspace", status="failed",
                output=f"工作区被拒绝：{attempted_workspace}（{exc}）", error=str(exc),
            )
            _record_step(
                recorder, run_id, kind="checkpoint", status=STATUS_FAILED,
                output=json.dumps(checkpoint, ensure_ascii=False),
            )
            _record_step(
                recorder, run_id, kind="stop", status=STATUS_FAILED,
                output=json.dumps({"stop_reason": STOP_WORKSPACE_REJECTED, "workspace": attempted_workspace}, ensure_ascii=False),
            )
            recorder.set_state({"stop_reason": STOP_WORKSPACE_REJECTED, "checkpoint": checkpoint})
            recorder.set_status(STATUS_FAILED, summary=reason, error=reason)
            return AgentOutcome(
                STATUS_FAILED, reason, reason,
                stop_reason=STOP_WORKSPACE_REJECTED, checkpoint=checkpoint,
            )
    else:
        # No explicit path: using the process default is intentional (not a
        # fallback), but it still has to be a valid directory.
        try:
            effective_workspace = resolve_workspace_root(config.WORKSPACE_PATH)
        except SecurityError as exc:
            reason = f"默认工作区无效或被拒绝，任务未执行：{exc}"
            checkpoint = {
                "goal": goal,
                "workspace": attempted_workspace,
                "status": STATUS_FAILED,
                "stop_reason": STOP_WORKSPACE_REJECTED,
                "completed_steps": [],
                "modified_files": [],
                "test_results": [],
                "pending_steps": ["配置有效的默认工作区后重试"],
                "iterations": 0,
                "model_calls": 0,
                "phase_used": {},
                "tool_counts": {},
            }
            _record_step(
                recorder, run_id, kind="workspace", status="failed",
                output=f"默认工作区被拒绝：{attempted_workspace}（{exc}）", error=str(exc),
            )
            _record_step(
                recorder, run_id, kind="checkpoint", status=STATUS_FAILED,
                output=json.dumps(checkpoint, ensure_ascii=False),
            )
            recorder.set_state({"stop_reason": STOP_WORKSPACE_REJECTED, "checkpoint": checkpoint})
            recorder.set_status(STATUS_FAILED, summary=reason, error=reason)
            return AgentOutcome(
                STATUS_FAILED, reason, reason,
                stop_reason=STOP_WORKSPACE_REJECTED, checkpoint=checkpoint,
            )

    # Project-level context cache (file content + symbol index) scoped to this
    # run's workspace. Persisted to SQLite and keyed by the normalized path, so
    # a later run on the same project reuses it; other projects never share.
    project = ProjectContext(effective_workspace, db_path=config.CONTEXT_CACHE_DB)

    # Task-level memory (goals/plan/facts/evidence/budget), carried across a
    # resume via the persisted state/checkpoint.
    memory = RunMemory.from_dict(resume.memory) if (resume and resume.memory) else RunMemory(goal=goal)
    if resume and resume.checkpoint and not resume.memory:
        checkpoint_memory = resume.checkpoint.get("memory")
        if isinstance(checkpoint_memory, dict):
            memory = RunMemory.from_dict(checkpoint_memory)
    memory.goal = goal
    memory.target_workspace = str(effective_workspace)
    if plan:
        memory.plan = list(plan)

    # Token/cache accounting for the run (reported in a final stats step and
    # persisted in the run state). ``prompt_tokens`` etc. are the sum of *real*
    # API usage; ``*_estimated`` covers only calls whose provider returned no
    # usage, so real and estimated values are never conflated.
    token_stats: dict[str, Any] = {
        "model_calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_tokens_estimated": 0, "completion_tokens_estimated": 0,
        "calls_with_real_usage": 0, "calls_without_real_usage": 0,
        "usage_source": "estimated",
        "estimated": True,
    }
    # When resuming, seed the counters with what the previous attempt already
    # recorded (the resurrected process only counts its own new calls, so this
    # accumulates exactly once instead of resetting or double-counting).
    if resume and resume.token_stats:
        for key, value in resume.token_stats.items():
            if key in token_stats and isinstance(value, (int, float)):
                token_stats[key] = value
        # Recompute the derived flag from the restored counters.
        token_stats["usage_source"] = _usage_source(token_stats)
        token_stats["estimated"] = token_stats["calls_with_real_usage"] == 0
    cache_stats_base: dict[str, int] = (
        dict(resume.cache_stats) if (resume and resume.cache_stats) else {}
    )

    # Make the actual workspace visible in the execution log from the very first
    # step, so a run never silently operates somewhere the user did not expect.
    _record_step(
        recorder, run_id, kind="workspace",
        output=f"本次任务工作区：{effective_workspace}",
    )
    _record_step(
        recorder, run_id, kind="phase",
        output=(
            f"总预算 {max_iterations} 轮，模型调用≤{max_model_calls} 次；阶段预算："
            + "、".join(f"{PHASE_LABELS[k]}≤{v}" for k, v in phase_budgets.items())
            + f"；连续只读上限 {no_progress_limit} 轮，分析时限 {analyze_timeout:g}s"
        ),
    )

    deadline = time.monotonic() + task_timeout

    # ------------------------------------------------------------------
    # Resume preflight: verify the checkpoint's files against the *current*
    # disk state before the model is called. Completed edits are reported so
    # they are not redone; mismatches are reported for re-checking.
    # ------------------------------------------------------------------
    checkpoint = resume.checkpoint if (resume and resume.checkpoint) else None
    resume_file_notes = _audit_checkpoint_files(checkpoint, effective_workspace)
    if checkpoint:
        _record_step(
            recorder, run_id, kind="checkpoint", status="loaded",
            output=(
                f"加载检查点：workspace={checkpoint.get('workspace')}，"
                f"stop_reason={checkpoint.get('stop_reason')}；"
                + ("；".join(resume_file_notes) if resume_file_notes else "无已修改文件需要核对")
            ),
        )
    resume_context = (
        _checkpoint_resume_message(checkpoint, memory, effective_workspace, resume_file_notes)
        if checkpoint else ""
    )

    if resume and resume.messages:
        messages: list[dict[str, Any]] = list(resume.messages)
    else:
        from app.agents.llm import build_messages

        messages = build_messages(goal, workspace=str(effective_workspace))

    iterations = 0
    model_calls_attempt = 0
    tool_calls_total = 0
    tool_successes = 0
    tool_failures = 0
    last_step_failed = False

    # No-progress tracking. A "read-only turn" is a turn whose classified phase
    # is analyze; it is only progress if it discovers something new. Mutations
    # and real command runs reset the streak (they are execution progress).
    readonly_streak = 0
    stalled_streak = 0
    duplicate_counts: dict[str, int] = {}
    seen_result_signatures: set[str] = set()
    no_progress_hint_sent = False
    plan_announced = False
    plan_idle_turns = 0
    analyze_started: float | None = None

    # Verification evidence. ``tool_calls_total`` doubles as a monotonic sequence
    # number so ordering between mutation and verification can be checked.
    # A checkpoint may already contain real modifications from a previous
    # attempt; they still require a *new* verification after the resume.
    checkpoint_modified = _checkpoint_modified_index(checkpoint)
    mutated = bool(checkpoint_modified)  # a real project file was modified
    mutation_seq = 0 if checkpoint_modified else -1
    verification_pass_seq = -1  # tool_calls_total at last passing verification
    last_verification_failed = False  # most recent verification command failed

    def make_ctx(authorization: Authorization | None = None) -> ToolContext:
        # The handler verifies against this backend-generated authorization; it
        # never trusts a model-supplied ``approved`` flag. ``None`` means "not
        # authorized", so the handler fails closed for any approval-requiring op.
        # The run's workspace is fixed for the lifetime of this run.
        return ToolContext(
            run_id=run_id,
            approval_check=(authorization.check if authorization is not None else None),
            workspace=effective_workspace,
            read_tracker=read_tracker,
            project=project,
            write_scope=write_scope,
            cancel_check=cancel_check,
            cancel_token=cancel_token,
            subtask_id=(getattr(write_scope, "subtask_id", None) if write_scope else None),
        )

    def track(ok: bool) -> None:
        nonlocal tool_calls_total, tool_successes, tool_failures, last_step_failed
        tool_calls_total += 1
        if ok:
            tool_successes += 1
            last_step_failed = False
        else:
            tool_failures += 1
            last_step_failed = True

    def record_evidence(name: str, args: dict[str, Any], ok: bool) -> None:
        nonlocal mutated, mutation_seq, verification_pass_seq, last_verification_failed
        if name in _WRITE_TOOLS and ok:
            mutated = True
            mutation_seq = tool_calls_total
        if name == "run_project_command" and ok:
            program = str(args.get("program", ""))
            argv = [str(a) for a in args.get("args", [])]
            # A successful formatter/autofixer mutates files; it must NOT be
            # counted as verification evidence, so require a later real
            # verification to confirm the result.
            if is_mutating_command(program, argv):
                mutated = True
                mutation_seq = tool_calls_total
        if is_verifying_tool_call(name, args):
            if ok:
                last_verification_failed = False
                verification_pass_seq = tool_calls_total
            else:
                last_verification_failed = True

    def record_tool_memory(name: str, args: dict[str, Any], result: dict[str, Any], ok: bool) -> None:
        """Fold a completed tool call into the task memory (real results only)."""
        output = str(result.get("output", ""))[:300]
        memory.record_step(name, ok, output)
        data = result.get("data") or {}
        if name in _WRITE_TOOLS and ok:
            changed = str(args.get("path") or (data.get("path") if isinstance(data, dict) else "") or "")
            if changed:
                digest = data.get("after_sha256") if isinstance(data, dict) else None
                memory.record_modified_file(changed, str(digest) if digest else None)
        if name == "run_project_command":
            program = str(args.get("program", ""))
            argv = [str(a) for a in args.get("args", [])]
            memory.record_command(program, argv, data.get("exit_code"), ok)
            if is_verifying_tool_call(name, args):
                memory.record_test_evidence(program, argv, ok, output)
        if name in ("check_http", "check_port"):
            # A passed HTTP/port check is pass/fail verification evidence, just
            # like a test/build command — a verify subtask may be accepted on it.
            target = (
                str(args.get("url", ""))
                if name == "check_http"
                else f"{args.get('host', '127.0.0.1')}:{args.get('port', '')}"
            )
            memory.record_test_evidence(name, [target], ok, output)
        if not ok:
            # Failed evidence must survive history compression: record it in
            # memory so a truncated old tool message never erases the reason.
            item = f"{name} 失败：{output[:240]}"
            if item not in memory.pending:
                memory.pending.append(item)
                if len(memory.pending) > 20:
                    memory.pending = memory.pending[-20:]
        if name in ("read_file", "list_files", "search_code") and ok:
            p = str(args.get("path", ""))
            if p and p not in memory.confirmed_locations:
                memory.confirmed_locations.append(p)

    # Per-turn progress probe. It is reset for each model turn (and for the
    # pending-tool replay on resume) and fed by ``execute_tool`` as real results
    # arrive. "new_info" means a read returned content/range/results the run has
    # not seen before; a duplicate call alone is never progress.
    turn_progress: dict[str, Any] = {
        "new_info": False, "read_calls": 0, "duplicate_calls": 0, "tool_names": [],
    }

    def reset_turn_progress() -> None:
        turn_progress.update(
            {"new_info": False, "read_calls": 0, "duplicate_calls": 0, "tool_names": []}
        )

    def observe_tool_progress(
        name: str, args: dict[str, Any], result: dict[str, Any], ok: bool
    ) -> None:
        """Record duplicate calls and whether this call produced new information."""
        fingerprint = f"{name}:{canonical_args(args)}"
        count = duplicate_counts.get(fingerprint, 0) + 1
        duplicate_counts[fingerprint] = count
        if count > 1:
            turn_progress["duplicate_calls"] += 1
        turn_progress["tool_names"].append(name)
        if not ok:
            return
        data = result.get("data")
        if name == "read_file" and isinstance(data, dict):
            turn_progress["read_calls"] += 1
            signature = (
                f"read_file:{data.get('path')}:{data.get('start_line')}:"
                f"{data.get('end_line')}:{data.get('content_hash')}"
            )
            if signature not in seen_result_signatures:
                seen_result_signatures.add(signature)
                turn_progress["new_info"] = True
        elif name in _READ_TOOLS:
            turn_progress["read_calls"] += 1
            try:
                raw = json.dumps(
                    {"name": name, "data": data}, sort_keys=True,
                    ensure_ascii=False, default=str,
                )
            except (TypeError, ValueError):
                raw = f"{name}:{data}"
            signature = f"{name}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"
            if signature not in seen_result_signatures:
                seen_result_signatures.add(signature)
                turn_progress["new_info"] = True

    def finish_turn_progress(phase: str | None) -> None:
        """Update streaks, log metrics and (once) steer a stalled run."""
        nonlocal readonly_streak, stalled_streak, no_progress_hint_sent, plan_idle_turns
        names = list(turn_progress["tool_names"])
        executed = any(n in _WRITE_TOOLS or n == "run_project_command" for n in names)
        if phase == "analyze" and not executed:
            readonly_streak += 1
            if turn_progress["new_info"] and turn_progress["duplicate_calls"] == 0:
                stalled_streak = 0
            else:
                stalled_streak += 1
        else:
            readonly_streak = 0
            stalled_streak = 0
            plan_idle_turns = 0
            duplicate_counts.clear()

        if plan_announced and phase == "analyze" and not mutated and not executed:
            plan_idle_turns += 1
        elif executed:
            plan_idle_turns = 0

        metrics = {
            "phase": phase,
            "iteration": iterations,
            "model_calls": token_stats["model_calls"],
            "readonly_streak": readonly_streak,
            "stalled_streak": stalled_streak,
            "duplicate_calls": turn_progress["duplicate_calls"],
            "read_calls": turn_progress["read_calls"],
            "new_info": bool(turn_progress["new_info"]),
            "tools": names,
            # Explicit read/write classification (not just tool names), so logs
            # make "only reading" vs "actually modifying" unambiguous.
            "read_tools": [n for n in names if n in _READ_TOOLS],
            "write_tools": [n for n in names if n in _MUTATING_TOOLS],
            "command_tools": [n for n in names if n == "run_project_command"],
            "mutated": mutated,
        }
        _record_step(
            recorder, run_id, kind="progress",
            output=json.dumps(metrics, ensure_ascii=False),
        )

        if mutated or executed:
            return
        max_duplicate_calls = max(duplicate_counts.values(), default=0)
        limit_hit = (
            readonly_streak >= no_progress_limit
            or stalled_streak >= no_progress_limit
            # Repeated identical calls are a no-progress signal on their own,
            # even when the read-only streak has not reached its limit yet.
            or max_duplicate_calls >= config.AGENT_DUPLICATE_TOOL_LIMIT
            or (plan_announced and plan_idle_turns >= config.AGENT_PLAN_IDLE_LIMIT)
        )
        if phase == "analyze" and limit_hit and not no_progress_hint_sent:
            no_progress_hint_sent = True
            hint = (
                f"[无进展收敛] 已连续 {readonly_streak} 轮只读分析（重复/无新信息 "
                f"{stalled_streak} 轮，计划后未执行 {plan_idle_turns} 轮）。"
                "不要再继续读取；请基于已获得的信息，在本轮直接执行一个范围明确、"
                "可验证的最小修改（edit_file/write_file），然后运行相关测试/构建。"
                "如果确实无法继续，请明确说明阻塞原因并给出最终总结。"
            )
            _record_step(recorder, run_id, kind="progress", status="hint", output=hint)
            messages.append({"role": "user", "content": hint})

    def _merged_cache_stats() -> dict[str, int]:
        current = project.stats()
        return {
            key: int(cache_stats_base.get(key, 0)) + int(current.get(key, 0))
            for key in current
        }

    def _stats_state() -> dict[str, Any]:
        """The stats/memory snapshot persisted with run state.

        Written after every tool turn (not only at termination) so that a run
        paused for approval or interrupted still carries its accumulated token
        and cache counters into the resumed attempt.
        """
        return {
            "memory": memory.to_dict(),
            "token_stats": token_stats,
            "cache_stats": _merged_cache_stats(),
        }

    def _record_stats(note: str = "") -> None:
        """Persist memory + token/cache stats and log a human-readable summary."""
        snapshot = _stats_state()
        cache_stats = snapshot["cache_stats"]
        real = (
            f"真实 Token（API usage，{token_stats['calls_with_real_usage']} 次调用）："
            f"输入 {token_stats['prompt_tokens']}，"
            f"输出 {token_stats['completion_tokens']}，合计 {token_stats['total_tokens']}"
        )
        est = (
            f"估算 Token（{token_stats['calls_without_real_usage']} 次调用无真实 usage）："
            f"输入 {token_stats['prompt_tokens_estimated']}，"
            f"输出 {token_stats['completion_tokens_estimated']}"
        )
        source_note = {
            "api": "统计来源：全部为 API 真实值。",
            "estimated": "统计来源：全部为估算值（无真实 usage）。",
            "mixed": (
                "统计来源：混合——真实值仅覆盖 "
                f"{token_stats['calls_with_real_usage']} 次调用，"
                f"其余 {token_stats['calls_without_real_usage']} 次为估算，两者不可相加。"
            ),
        }[token_stats["usage_source"]]
        cache_summary = (
            f"缓存统计：读请求 {cache_stats.get('read_requests', 0)}，"
            f"磁盘读 {cache_stats.get('disk_reads', 0)}，"
            f"命中 {cache_stats.get('cache_hits', 0)}，"
            f"未命中 {cache_stats.get('cache_misses', 0)}，"
            f"索引命中 {cache_stats.get('index_hits', 0)}。"
        )
        lines = [
            cache_summary,
            f"模型调用 {token_stats['model_calls']} 次；{real}；{est}。{source_note}",
        ]
        if note:
            lines.append(note)
        _record_step(recorder, run_id, kind="stats", output="\n".join(lines))
        recorder.set_state(snapshot)

    def _default_pending_steps(stop_reason: str) -> list[str]:
        if stop_reason == STOP_COMPLETED:
            return []
        if stop_reason in {
            STOP_NO_PROGRESS, STOP_PHASE_BUDGET, STOP_MUTATION_REQUIRED,
            STOP_ANALYZE_TIMEOUT,
        }:
            return [
                "停止继续只读分析；基于已获得的信息执行一个最小可验证修改",
                "修改后运行相关测试/构建验证",
            ]
        if stop_reason == STOP_MAX_ITERATIONS:
            return ["从检查点继续：先核对已修改文件，再执行剩余步骤"]
        if stop_reason == STOP_MODEL_CALL_BUDGET:
            return ["模型调用预算已用尽；恢复后从检查点继续剩余步骤"]
        if stop_reason == STOP_MODEL_ERROR:
            return ["模型调用失败；恢复后从检查点继续剩余步骤"]
        if stop_reason == STOP_TASK_TIMEOUT:
            return ["任务超时；恢复后从检查点继续剩余步骤"]
        return ["核对检查点后继续未完成步骤"]

    def _checkpoint_payload(
        status: str, stop_reason: str, summary: str
    ) -> dict[str, Any]:
        pending_steps = [p for p in memory.pending if isinstance(p, str)]
        pending_steps.extend(_default_pending_steps(stop_reason))
        # Stable de-duplication while preserving order.
        pending_steps = list(dict.fromkeys(pending_steps))[:20]
        modified = [
            {
                "path": path,
                "sha256": memory.modified_file_states.get(path, ""),
            }
            for path in memory.modified_files
        ]
        # Include any hash recorded without a corresponding list entry.
        known = {item["path"] for item in modified}
        for path, digest in memory.modified_file_states.items():
            if path not in known:
                modified.append({"path": path, "sha256": digest})
        return {
            "goal": goal,
            "workspace": str(effective_workspace),
            "status": status,
            "stop_reason": stop_reason,
            "iterations": iterations,
            "model_calls": token_stats["model_calls"],
            "phase_used": dict(phase_used),
            "tool_counts": dict(tool_counts),
            "readonly_streak": readonly_streak,
            "stalled_streak": stalled_streak,
            "completed_steps": list(memory.completed_steps[-80:]),
            "modified_files": modified,
            "test_results": list(memory.test_evidence[-20:]),
            "pending_steps": pending_steps,
            "last_summary": summary[:2000],
            "memory": memory.to_dict(),
        }

    def _persist_checkpoint(
        status: str, stop_reason: str, summary: str
    ) -> dict[str, Any]:
        memory.stop_reason = stop_reason
        memory.last_summary = summary[:2000]
        snapshot = _checkpoint_payload(status, stop_reason, summary)
        _record_step(
            recorder, run_id, kind="checkpoint", status=status,
            output=json.dumps(snapshot, ensure_ascii=False),
        )
        recorder.set_state(
            {"stop_reason": stop_reason, "checkpoint": snapshot, **_stats_state()}
        )
        return snapshot

    def _finish(
        status: str, summary: str, error: str | None, stop_reason: str
    ) -> AgentOutcome:
        """Persist stats/checkpoint/stop metadata, then return the outcome."""
        _record_stats(error or summary)
        snapshot = _persist_checkpoint(status, stop_reason, summary)
        stop_meta = {
            "status": status,
            "stop_reason": stop_reason,
            "iterations": iterations,
            "model_calls": token_stats["model_calls"],
            "tool_calls": tool_calls_total,
            "phase": last_phase,
            "phase_used": dict(phase_used),
            "tool_counts": dict(tool_counts),
            "readonly_streak": readonly_streak,
            "stalled_streak": stalled_streak,
            "workspace": str(effective_workspace),
            "checkpoint": "saved",
        }
        _record_step(
            recorder, run_id, kind="stop", status=status,
            output=json.dumps(stop_meta, ensure_ascii=False),
        )
        recorder.set_status(status, summary=summary, error=error)
        return AgentOutcome(
            status, summary, error,
            iterations, tool_calls_total, tool_successes, tool_failures,
            phase_used, tool_counts,
            stop_reason=stop_reason, checkpoint=snapshot,
        )

    pending_answer: dict[str, str | None] = {
        "message": resume.user_input if resume is not None else None,
    }

    def handle_ask_user(args: dict[str, Any], call_id: str) -> None:
        """Pause the run for a user answer (mirrors the approval flow)."""
        question = str(args.get("question") or "").strip()
        answer: str | None = None
        if not question:
            result = tool_error("ask_user 需要非空的 question 参数。")
        else:
            if pending_answer["message"]:
                # Restart path: the answer was delivered while no thread waited.
                answer = pending_answer["message"]
                pending_answer["message"] = None
                recorder.set_state({"user_input": None})
            else:
                requester = getattr(recorder, "request_input", None)
                if requester is not None:
                    _record_step(
                        recorder, run_id, kind="input", tool_name=ASK_USER_TOOL,
                        args={"question": question}, status=STATUS_WAITING_INPUT,
                        output=f"等待用户补充信息：{question}",
                    )
                    answer = requester(run_id, question)
            if answer:
                _record_step(
                    recorder, run_id, kind="input", tool_name=ASK_USER_TOOL,
                    args={"question": question}, status="answered",
                    output=f"用户回复：{answer}"[: config.TOOL_OUTPUT_MAX_CHARS],
                )
                result = tool_ok(
                    f"用户回复：{answer}", data={"question": question, "answer": answer},
                )
            elif getattr(recorder, "request_input", None) is None:
                result = tool_error("当前没有可用的用户输入通道，请基于已有信息继续或说明阻塞原因。")
            else:
                result = tool_error("用户未在时限内回复；请基于已有信息继续，或在总结中说明阻塞原因。")
        ok = bool(result.get("ok"))
        _record_step(
            recorder, run_id, kind="tool", tool_name=ASK_USER_TOOL,
            args={"question": question}, status="succeeded" if ok else "failed",
            output=str(result.get("output", ""))[: config.TOOL_OUTPUT_MAX_CHARS],
            error=None if ok else str(result.get("output", "")),
        )
        track(ok)
        observe_tool_progress(ASK_USER_TOOL, args, result, ok)
        record_tool_memory(ASK_USER_TOOL, args, result, ok)
        messages.append({
            "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
        })

    def execute_tool(
        name: str,
        args: dict[str, Any],
        call_id: str,
        preapproved: dict[str, Any] | None = None,
    ) -> None:
        """Execute one tool call with per-call approval (single-use authorization)."""
        if cancel_check is not None and cancel_check():
            reason = f"任务已取消，拒绝启动工具 {name}"
            result = tool_error(reason)
            _record_step(
                recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
                status="failed", output=result["output"], error="cancelled",
            )
            track(False)
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
            })
            return
        tool_counts[name] = tool_counts.get(name, 0) + 1
        tool = get_tool(name)
        if tool is None:
            result = tool_error(f"未知工具：{name}")
            _record_step(
                recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
                status="failed", output=result["output"], error=None,
            )
            track(False)
            observe_tool_progress(name, args, result, False)
            record_tool_memory(name, args, result, False)
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
            })
            return

        # A resumed run may still carry a pending write that already landed
        # before the crash. The checkpoint proves the intended content is on
        # disk, so answer the assistant tool call without touching the file
        # again. A repeated model-issued identical write is handled the same
        # way (idempotent), which also prevents needless snapshots.
        if _mutation_already_applied(name, args, effective_workspace, checkpoint):
            path_arg = str(args.get("path", ""))
            try:
                rel_key = resolve_workspace_path(path_arg, effective_workspace).relative_to(
                    effective_workspace
                ).as_posix()
            except (SecurityError, ValueError):
                rel_key = path_arg
            digest = _checkpoint_modified_index(checkpoint).get(rel_key, "")
            result = {
                "ok": True,
                "output": (
                    f"检查点确认 {rel_key} 的修改已在磁盘生效；跳过重复执行。"
                ),
                "data": {"path": rel_key, "after_sha256": digest, "skipped": True},
            }
            _record_step(
                recorder, run_id, kind="checkpoint", tool_name=name,
                args=redact_kv(args), status="skipped", output=result["output"],
            )
            track(True)
            record_evidence(name, args, True)
            memory.record_modified_file(rel_key, digest or None)
            memory.record_step(name, True, result["output"][:300])
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
            })
            return

        if name == ASK_USER_TOOL:
            handle_ask_user(args, call_id)
            return
        if write_scope is not None:
            # The running subtask's contract limits where it may write; an
            # out-of-scope write is rejected before any approval prompt and is
            # recorded as a scope_violation evidence item by the guard.
            from app.tools.registry import scope_precheck

            scope_error = scope_precheck(name, args, make_ctx(None))
            if scope_error:
                result = tool_error(scope_error)
                _record_step(
                    recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
                    status="failed", output=result["output"], error="rejected by write scope",
                )
                track(False)
                observe_tool_progress(name, args, result, False)
                record_tool_memory(name, args, result, False)
                messages.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": _tool_message_content(result),
                })
                return
        authorization: Authorization | None = None
        # Security classification may reject a disallowed command, illegal
        # arguments or a path escape before any execution. Handle that exactly
        # like a tool failure: record a failed step and hand a clear tool_error
        # back to the model so it can pick a safe alternative. Only SecurityError
        # is caught — unexpected program errors still propagate as system errors.
        try:
            needs_approval = tool.requires_approval(args, effective_workspace)
        except SecurityError as exc:
            result = tool_error(str(exc))
            _record_step(
                recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
                status="failed", output=result["output"], error="rejected by security policy",
            )
            track(False)
            observe_tool_progress(name, args, result, False)
            record_tool_memory(name, args, result, False)
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
            })
            return

        if needs_approval and not config.REQUIRE_APPROVAL:
            # Approval gating disabled by configuration: authorize this exact
            # call without blocking. Hard-deny rules were already enforced by
            # ``requires_approval`` above (it raises SecurityError).
            risk = approval_reason(name, args, effective_workspace)
            auto_id = f"auto-{uuid4().hex[:12]}"
            _record_step(
                recorder, run_id, kind="approval", tool_name=name,
                args=redact_kv(args), status="auto_granted",
                output=f"REQUIRE_APPROVAL=false，自动放行：{name}（{risk}）",
                error=None,
            )
            authorization = Authorization(
                approval_id=auto_id, run_id=run_id,
                tool_name=name, args=args, granted=True,
            )
        elif needs_approval:
            risk = approval_reason(name, args, effective_workspace)
            if preapproved is not None:
                decision = ApprovalDecision(
                    approval_id=str(preapproved.get("approval_id", "")),
                    approved=bool(preapproved.get("approved")),
                    reason=preapproved.get("reason"),
                )
            else:
                _record_step(
                    recorder, run_id, kind="approval", tool_name=name,
                    args=redact_kv(args), status=STATUS_WAITING_APPROVAL,
                    output=f"等待批准：{name}（{risk}）", error=None,
                )
                decision = recorder.request_approval(run_id, name, args, risk)

            _record_step(
                recorder, run_id, kind="approval", tool_name=name,
                args=redact_kv(args),
                status="granted" if decision.approved else "denied",
                output=f"批准请求 {decision.approval_id}：{risk} → "
                + ("已批准" if decision.approved else "已拒绝"),
                error=None if decision.approved else (decision.reason or "用户拒绝"),
            )
            if not decision.approved:
                result = tool_error("用户拒绝了该操作。")
                _record_step(
                    recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
                    status="failed", output=result["output"], error="rejected by user",
                )
                track(False)
                observe_tool_progress(name, args, result, False)
                record_tool_memory(name, args, result, False)
                messages.append({
                    "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
                })
                return
            authorization = Authorization(
                approval_id=decision.approval_id, run_id=run_id,
                tool_name=name, args=args, granted=True,
            )

        result, ok = _run_tool_handler(name, args, make_ctx(authorization), tool_timeout)
        _record_step(
            recorder, run_id, kind="tool", tool_name=name, args=redact_kv(args),
            status="succeeded" if ok else "failed",
            output=str(result.get("output", ""))[: config.TOOL_OUTPUT_MAX_CHARS],
            error=None if ok else str(result.get("output", "")),
        )
        track(ok)
        observe_tool_progress(name, args, result, ok)
        record_evidence(name, args, ok)
        record_tool_memory(name, args, result, ok)
        messages.append({
            "role": "tool", "tool_call_id": call_id, "content": _tool_message_content(result),
        })

    # On resume, re-execute the remaining (un-executed) tool calls so the
    # persisted assistant message is fully answered by tool messages. Approval
    # is re-checked per call: a pre-approved decision carried across a restart is
    # honored only when it matches the exact tool + arguments, otherwise a fresh
    # approval is requested (never guessed from stale conversation state).
    if resume and resume.pending_tools:
        reset_turn_progress()
        for pending in resume.pending_tools:
            name = pending["name"]
            args = pending.get("args", {})
            call_id = pending.get("call_id") or f"resume-{uuid4().hex[:8]}"
            pre = resume.preapproved
            matching_pre = None
            if (
                pre is not None
                # An approval is bound to one run: a decision carried across a
                # restart is honored only for the *same* run, the same tool and
                # the exact same arguments (including cwd).
                and pre.get("run_id") == run_id
                and pre.get("tool_name") == name
                and canonical_args(pre.get("args", {})) == canonical_args(args)
            ):
                matching_pre = pre
            execute_tool(name, args, call_id, preapproved=matching_pre)
        recorder.set_state({
            "pending_tools": None, "agent_messages": None,
            "pending_approval": None, "approval_decision": None,
        })
        reset_turn_progress()

    # Inject the checkpoint directive *after* any pending tool replies, so the
    # OpenAI tool-call protocol (assistant tool_calls followed by tool messages)
    # is never broken by an extra user message.
    if resume_context:
        messages.append({"role": "user", "content": resume_context})

    while iterations < max_iterations:
        if cancel_check is not None and cancel_check():
            reason = f"任务已取消：{cancel_token.reason() if cancel_token else 'cancelled'}"
            return _finish(
                STATUS_CANCELLED, reason, reason, STOP_CANCELLED,
            )
        # Hard budgets are checked before a model call, so an exhausted budget
        # cannot consume one more call or execute another tool.
        if time.monotonic() >= deadline:
            return _finish(
                STATUS_FAILED, "任务超时：达到总时间限制。", "task timeout",
                STOP_TASK_TIMEOUT,
            )
        if model_calls_attempt >= max_model_calls:
            reason = (
                f"模型调用预算耗尽（本次尝试 {model_calls_attempt}/{max_model_calls} 次），"
                "任务未完成；已保存检查点，可从断点继续。"
            )
            _record_step(recorder, run_id, kind="progress", status="blocked", output=reason)
            return _finish(STATUS_FAILED, reason, reason, STOP_MODEL_CALL_BUDGET)
        if (
            not mutated
            and last_phase == "analyze"
            and analyze_started is not None
            and time.monotonic() - analyze_started >= analyze_timeout
        ):
            reason = (
                f"分析阶段超时（>{analyze_timeout:g}s），仍未进入修改或验证；"
                "任务未完成，已保存检查点。"
            )
            _record_step(recorder, run_id, kind="progress", status="blocked", output=reason)
            return _finish(STATUS_FAILED, reason, reason, STOP_ANALYZE_TIMEOUT)

        try:
            # Build the *projected* request: compact the raw history (truncate
            # old tool outputs, inject the task-memory block) while keeping the
            # current tool exchange verbatim so the tool-call protocol holds.
            projected = assemble_messages(
                messages, memory,
                tool_tail_chars=config.CONTEXT_TOOL_TAIL_CHARS,
                memory_block_chars=config.CONTEXT_MEMORY_BLOCK_CHARS,
                failure_tail_chars=config.CONTEXT_FAILURE_TAIL_CHARS,
            )
            try:
                response = llm.chat(projected, tools, cancel_token=cancel_token)
            except TypeError:
                # Fakes / older clients may not accept cancel_token.
                response = llm.chat(projected, tools)
        except LLMCancelled as exc:
            reason = f"任务已取消（模型调用中止）：{exc}"
            return _finish(STATUS_CANCELLED, reason, reason, STOP_CANCELLED)
        except Exception as exc:  # noqa: BLE001 - model/network failure
            if cancel_token is not None and cancel_token.is_set():
                reason = f"任务已取消（模型调用中止）：{cancel_token.reason()}"
                return _finish(STATUS_CANCELLED, reason, reason, STOP_CANCELLED)
            return _finish(
                STATUS_FAILED, f"模型调用失败：{exc}", str(exc), STOP_MODEL_ERROR,
            )

        model_calls_attempt += 1
        iterations += 1
        token_stats["model_calls"] += 1
        prompt_est = estimate_prompt_tokens(projected)
        completion_est = estimate_prompt_tokens([{"content": response.content or ""}])
        if response.usage:
            # Real usage: count it separately from the estimate fallback.
            token_stats["calls_with_real_usage"] += 1
            token_stats["prompt_tokens"] += int(response.usage.get("prompt_tokens") or 0)
            token_stats["completion_tokens"] += int(response.usage.get("completion_tokens") or 0)
            token_stats["total_tokens"] += int(response.usage.get("total_tokens") or 0)
        else:
            # No usage returned: record an explicit estimate for *this* call.
            token_stats["calls_without_real_usage"] += 1
            token_stats["prompt_tokens_estimated"] += prompt_est
            token_stats["completion_tokens_estimated"] += completion_est
        token_stats["usage_source"] = _usage_source(token_stats)
        token_stats["estimated"] = token_stats["calls_with_real_usage"] == 0

        if _is_plan_announcement(response.content):
            plan_announced = True

        if response.has_tool_calls():
            _record_step(
                recorder, run_id, kind="model", tool_name=None,
                args={"iteration": iterations, "model_calls": token_stats["model_calls"]},
                status="running", output=(response.content or "")[:2000], error=None,
            )

            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            })

            parsed = response.parsed_tool_calls()
            pending_batch = [
                {"name": n, "args": a, "call_id": cid} for (cid, n, a) in parsed
            ]
            reset_turn_progress()

            # Classify this turn into a phase *before* running its tools, so a
            # turn that would keep doing an already-exhausted phase (after a
            # steering hint) is stopped with a phase-specific reason instead of
            # burning more turns on redundant work.
            phase = classify_phase([c["name"] for c in pending_batch])

            # No-progress guard: the convergence hint was already sent and the
            # model still asks only for reads. Stop now (with a checkpoint)
            # instead of executing another useless read-only turn.
            if phase == "analyze" and not mutated and no_progress_hint_sent:
                reason = (
                    f"连续只读分析未产生进展（只读 {readonly_streak} 轮，"
                    f"重复/无新信息 {stalled_streak} 轮），"
                    "已提示收敛仍未进入修改或验证，任务未完成。"
                )
                _record_step(recorder, run_id, kind="progress", status="blocked", output=reason)
                return _finish(STATUS_FAILED, reason, reason, STOP_NO_PROGRESS)

            if (
                phase
                and phase_used[phase] >= phase_budgets[phase]
                and phase in phase_hint_sent
                # Reading again *after* a real mutation is normal work (checking
                # the edited region, reading the new test file). Only stop a run
                # that is genuinely stuck in the analyze phase without having
                # changed anything — the abort message promises exactly that.
                and not (phase == "analyze" and mutated)
            ):
                reason = PHASE_EXHAUSTION[phase]
                _record_step(recorder, run_id, kind="phase", output=reason)
                return _finish(STATUS_FAILED, reason, reason, STOP_PHASE_BUDGET)

            # Each approval is requested per-call (not as one batch-wide boolean)
            # and only authorizes that exact tool + args. Before an
            # approval-requiring call we persist only the not-yet-executed suffix
            # of the batch, so a restart during the approval never re-runs tools
            # that already completed.
            for idx, call in enumerate(pending_batch):
                call_id, name, args = call["call_id"], call["name"], call["args"]
                if cancel_check is not None and cancel_check():
                    reason = f"任务已取消，跳过后续工具：{cancel_token.reason() if cancel_token else 'cancelled'}"
                    recorder.set_state({
                        "pending_tools": None, "agent_messages": None, "pending_approval": None,
                    })
                    return _finish(STATUS_CANCELLED, reason, reason, STOP_CANCELLED)
                if time.monotonic() >= deadline:
                    recorder.set_state({
                        "pending_tools": None, "agent_messages": None, "pending_approval": None,
                    })
                    return _finish(
                        STATUS_FAILED, "任务超时", "task timeout", STOP_TASK_TIMEOUT,
                    )
                tool = get_tool(name)
                # A security rejection is handled by ``execute_tool`` (recorded as
                # a failed step); here we only skip persisting a pending state for
                # an invalid call, so a restart never waits on a rejected command.
                try:
                    will_pause = name == ASK_USER_TOOL or (
                        tool is not None
                        and config.REQUIRE_APPROVAL
                        and tool.requires_approval(args, effective_workspace)
                    )
                except SecurityError:
                    will_pause = False
                if will_pause:
                    recorder.set_state({
                        "agent_messages": messages,
                        "pending_tools": pending_batch[idx:],
                        **_stats_state(),
                    })
                execute_tool(name, args, call_id)
                # Advance the persisted pending cursor immediately after the tool
                # completes, so a crash right here cannot leave an already-run
                # side-effecting tool in ``pending_tools`` (which a later resume
                # would otherwise replay).
                if will_pause and idx + 1 < len(pending_batch):
                    recorder.set_state({
                        "agent_messages": messages,
                        "pending_tools": pending_batch[idx + 1:],
                        **_stats_state(),
                    })

            # Account for the phase this turn spent, log phase transitions and
            # emit a one-time steering hint when a phase budget is first hit.
            if phase:
                phase_used[phase] += 1
                if phase == "analyze":
                    if analyze_started is None:
                        analyze_started = time.monotonic()
                else:
                    analyze_started = None
                if phase != last_phase:
                    _record_step(
                        recorder, run_id, kind="phase",
                        output=(
                            f"阶段：{PHASE_LABELS[phase]}"
                            f"（已用 {phase_used[phase]}/{phase_budgets[phase]} 轮，"
                            f"模型调用 {token_stats['model_calls']} 次）"
                        ),
                    )
                    last_phase = phase
                if phase_used[phase] >= phase_budgets[phase] and phase not in phase_hint_sent:
                    _record_step(recorder, run_id, kind="phase", output=PHASE_HINTS[phase])
                    messages.append({"role": "user", "content": PHASE_HINTS[phase]})
                    phase_hint_sent.add(phase)
                memory.phase = phase
                memory.remaining_budget = max(0, phase_budgets[phase] - phase_used[phase])

            finish_turn_progress(phase)

            # Persist a running checkpoint after every turn (not only at
            # termination), so a crash/restart leaves a recoverable state.
            running_checkpoint = _checkpoint_payload("running", "", "")
            recorder.set_state({
                "pending_tools": None, "agent_messages": None, "pending_approval": None,
                "checkpoint": running_checkpoint,
                "stop_reason": "",
                **_stats_state(),
            })
            continue

        # Final answer from the model. Success is decided by real execution
        # evidence, never by the model's prose alone:
        #   * at least one tool must have actually run;
        #   * if files were modified, a verification must have passed afterwards;
        #   * a failed verification is only cleared by a later passing one;
        #   * the last executed step must not have failed;
        #   * a change-request goal must have produced a real modification.
        summary = (response.content or "").strip()
        stop_reason = STOP_COMPLETED

        if tool_calls_total == 0:
            error = "模型未调用任何工具，无法确认任务已实际执行并通过验证。"
            stop_reason = STOP_BLOCKED
        elif mutated and verification_pass_seq < mutation_seq:
            error = "检测到文件修改，但缺少修改后通过的验证证据（如测试或构建），无法确认目标达成。"
            stop_reason = STOP_VERIFICATION_REQUIRED
        elif last_verification_failed:
            error = "最后一次验证命令失败，且未通过后续重新验证。"
            stop_reason = STOP_VERIFICATION_REQUIRED
        elif last_step_failed:
            error = "最后一步工具执行失败且未通过后续验证。"
            stop_reason = STOP_TOOL_FAILURE
        elif (
            goal_requires_mutation(goal) if mutation_required is None else mutation_required
        ) and not mutated:
            error = (
                "任务目标要求修改文件，但本次执行未实际修改任何文件，不能视为完成；"
                "请从检查点继续并先完成一个最小可验证修改。"
            )
            stop_reason = STOP_MUTATION_REQUIRED
        else:
            error = None

        memory.add_finding(summary[:500], kind="verified" if error is None else "inferred")

        _record_step(
            recorder, run_id, kind="final", tool_name=None, args={},
            status="succeeded" if error is None else "failed",
            output=summary[:4000], error=error,
        )
        status = STATUS_SUCCEEDED if error is None else STATUS_FAILED
        return _finish(status, summary, error, stop_reason)

    reason = (
        f"达到最大迭代次数（{iterations}/{max_iterations}）仍未完成；"
        "任务未完成，已保存检查点。"
    )
    _record_step(recorder, run_id, kind="progress", status="blocked", output=reason)
    return _finish(
        STATUS_FAILED, "达到最大迭代次数仍未完成。", "达到最大迭代次数",
        STOP_MAX_ITERATIONS,
    )
