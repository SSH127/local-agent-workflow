from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import TASK_GOAL_MAX_CHARS

RunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "waiting_for_approval",
    "waiting_for_input",
    # legacy alias kept for backward compatibility
    "waiting_for_human",
    # a run left in ``running`` across a process restart: its execution thread
    # is gone and its side effects are unverifiable, so it must NOT be replayed.
    "interrupted",
    # task-decomposition control loop
    "waiting_for_plan_approval",
    "waiting_for_subtask_approval",
    "needs_attention",
    "cancelled",
]

TERMINAL_STATUSES = {"succeeded", "failed", "interrupted", "cancelled"}
# Statuses in which no execution thread is running and the user may edit the
# task graph (manual takeover) or resume.
PAUSED_STATUSES = {
    "waiting_for_plan_approval", "waiting_for_subtask_approval",
    "needs_attention", "interrupted", "failed",
}


class RunCreate(BaseModel):
    goal: str = Field(min_length=3, max_length=TASK_GOAL_MAX_CHARS)
    # Kept for API compatibility only: the workflow is single-pass (recovery is
    # owned by the inner ReAct loop), so this value does not change behaviour.
    max_attempts: int = Field(default=2, ge=1, le=5)
    # Optional absolute path of the target project workspace for this run.
    # When omitted/blank, the run uses the process-wide default workspace
    # (``config.WORKSPACE_PATH``). Validated by the API at creation time.
    workspace_path: str | None = Field(default=None, max_length=4096)


class RunResponse(BaseModel):
    run_id: str
    goal: str
    status: RunStatus
    attempts: int
    summary: str
    error: str | None = None
    workspace_path: str | None = None
    created_at: str
    updated_at: str


class RunStepResponse(BaseModel):
    step_id: str
    run_id: str
    seq: int
    ts: str
    kind: str
    tool_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    status: str
    output: str | None = None
    error: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool
    reason: str | None = None
    # Optional; when provided it must match the run's currently pending approval.
    # When omitted, the pending approval is looked up from the run's state.
    approval_id: str | None = None


class InputRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


# ---------------------------------------------------------------------------
# Task-decomposition workflow requests
# ---------------------------------------------------------------------------

class PlanApprovalRequest(BaseModel):
    """Approve or reject the pending plan version.

    ``decision="reject"`` sends the run back to decompose with ``feedback``
    (bounded by ``MAX_PLAN_REJECTIONS``). ``subtasks`` (approve only) replaces
    the not-yet-done part of the graph with a user-edited version.
    ``approved`` is accepted as a legacy alias of ``decision``.
    """

    decision: Literal["approve", "reject"] | None = None
    approved: bool | None = None
    feedback: str | None = Field(default=None, max_length=4000)
    subtasks: list[dict[str, Any]] | None = None

    def is_approve(self) -> bool:
        if self.decision is not None:
            return self.decision == "approve"
        return bool(self.approved)


class SubtaskConfirmRequest(BaseModel):
    approved: bool
    note: str | None = Field(default=None, max_length=2000)


class SubtaskEditRequest(BaseModel):
    """Manual contract edit of a not-yet-succeeded subtask (while paused)."""

    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    depends_on: list[str] | None = None
    kind: Literal["analyze", "modify", "verify"] | None = None
    acceptance_criteria: list[str] | None = None
    risk: Literal["low", "medium", "high"] | None = None
    reversible: bool | None = None
    # Paths / globs relative to the workspace this subtask may write ([] = read-only).
    write_scope: list[str] | None = None
    # Goal clause ids (C1..Cn) this subtask serves.
    serves_clauses: list[str] | None = None
    note: str | None = Field(default=None, max_length=2000)


class ManualResultRequest(BaseModel):
    """Manually mark a subtask's result (flagged ``manual`` in evidence)."""

    status: Literal["done", "failed", "skipped"]
    note: str = Field(min_length=1, max_length=2000)
    evidence_text: str = Field(default="", max_length=8000)
