"""Application configuration.

Loads settings from environment variables and an optional ``.env`` file at the
project root. Values may be overridden by real environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: the directory containing this ``app`` package's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if present (does not override real env).
load_dotenv(PROJECT_ROOT / ".env")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", PROJECT_ROOT / "data" / "workflow.sqlite3"))

# Filesystem/security boundaries. The agent may only touch files inside this
# directory (resolved against symlinks).
WORKSPACE_PATH = Path(os.getenv("WORKSPACE_PATH", str(PROJECT_ROOT))).resolve()

# ---------------------------------------------------------------------------
# Local API authentication (shared token).
# ---------------------------------------------------------------------------
# Shared secret protecting the local HTTP API. When empty, authentication is
# DISABLED (local development default). Set it and send it as
# ``Authorization: Bearer <token>`` to require authentication on every endpoint
# except /health. This is local access protection only — not a production-grade
# identity / multi-user system.
LOCAL_AGENT_TOKEN = os.getenv("LOCAL_AGENT_TOKEN", "")

# Browser origins allowed to call the API (CORS). Comma separated; defaults to
# the local Vite dev-server ports.
_DEFAULT_CORS_ORIGINS = ",".join(
    f"http://{host}:{port}"
    for port in (5173, 5174, 5175)
    for host in ("localhost", "127.0.0.1")
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible chat completions) configuration.
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_CONNECT_TIMEOUT_SECONDS = float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10"))
# How quickly CancelToken must force-close an in-flight LLM client (ms).
CANCEL_FORCE_MS = int(os.getenv("CANCEL_FORCE_MS", "500"))
SSE_POLL_FALLBACK_MS = int(os.getenv("SSE_POLL_FALLBACK_MS", "2000"))
SSE_POLL_ACTIVE_MS = int(os.getenv("SSE_POLL_ACTIVE_MS", "8000"))
# Replay + live-tail run_events from SQLite (multi-worker friendly with WAL).
EVENTS_DB_TAIL = os.getenv("EVENTS_DB_TAIL", "true").strip().lower() in {"1", "true", "yes", "on"}
EVENTS_DB_TAIL_MS = int(os.getenv("EVENTS_DB_TAIL_MS", "500"))
# Brief join wait after cancel before abandoning the LLM worker thread (ms).
CANCEL_JOIN_MS = int(os.getenv("CANCEL_JOIN_MS", "1000"))

# ---------------------------------------------------------------------------
# Agent execution limits.
# ---------------------------------------------------------------------------
# Total model-turn (ReAct iteration) budget for a single run. Every loop pass
# is one model turn; a single turn may issue several tool calls (batched), so
# this counts *turns*, not tool invocations.
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "40"))
# Per-phase turn budgets. Each is clamped to ``AGENT_MAX_ITERATIONS`` at run
# time so the phases can never exceed the total budget. ``analyze`` covers
# read/list/search turns, ``modify`` covers write_file turns, ``verify`` covers
# run_project_command (test/fix) turns. Read-only tasks never touch modify or
# verify, so they are only bounded by ``analyze`` + the total.
AGENT_PHASE_BUDGET_ANALYZE = int(os.getenv("AGENT_PHASE_BUDGET_ANALYZE", "10"))
AGENT_PHASE_BUDGET_MODIFY = int(os.getenv("AGENT_PHASE_BUDGET_MODIFY", "12"))
AGENT_PHASE_BUDGET_VERIFY = int(os.getenv("AGENT_PHASE_BUDGET_VERIFY", "15"))
# Independent model-call budget for one attempt. Normally one ReAct turn is one
# call, but making the cap explicit lets a configuration reject a runaway run
# even if a future change batches calls differently. A resume starts a fresh
# attempt budget; the prior usage is still carried in the checkpoint/stats.
AGENT_MAX_MODEL_CALLS = int(os.getenv("AGENT_MAX_MODEL_CALLS", "40"))
# Wall-clock cap for a single uninterrupted reading/analysis streak. It is
# reset by a real mutation or a verification/repair command, so a slow but
# productive run is not cut off just because it spent time reading once.
AGENT_ANALYZE_TIMEOUT_SECONDS = float(
    os.getenv("AGENT_ANALYZE_TIMEOUT_SECONDS", "300")
)
# No-progress guards. Consecutive read-only turns (and duplicate tool calls)
# are bounded independently of the phase budget so an agent cannot keep
# exploring until the whole turn budget is exhausted. When the limit is hit the
# executor first sends one convergence instruction ("make a minimal executable
# change or report a blocker"); only a still-read-only turn is aborted.
AGENT_NO_PROGRESS_LIMIT = int(os.getenv("AGENT_NO_PROGRESS_LIMIT", "4"))
AGENT_DUPLICATE_TOOL_LIMIT = int(os.getenv("AGENT_DUPLICATE_TOOL_LIMIT", "3"))
# How many analyzed turns may follow a model-declared plan before the executor
# treats the plan as stalled and forces convergence.
AGENT_PLAN_IDLE_LIMIT = int(os.getenv("AGENT_PLAN_IDLE_LIMIT", "2"))
# Maximum characters accepted for a task goal (mirrored by the frontend in
# ``frontend/src/App.tsx``; keep the two in sync). This is an *input-size*
# guard only: it does NOT guarantee that a 30k-char prompt fits the model's
# context window, which is a separate, model-dependent limit.
TASK_GOAL_MAX_CHARS = 30000
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "60"))
# Hard upper bound for ``run_project_command``. The model may pass its own
# ``timeout_seconds`` (e.g. for a slow build), but it is clamped to this value
# and the command's whole process tree is killed when it is exceeded.
MAX_COMMAND_TIMEOUT_SECONDS = float(os.getenv("MAX_COMMAND_TIMEOUT_SECONDS", "600"))
TASK_TIMEOUT_SECONDS = float(os.getenv("TASK_TIMEOUT_SECONDS", "900"))
# Maximum characters of tool output kept in the event log / stored in DB.
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "8000"))
# Per-read content budget for ``read_file``. A single read returns at most this
# many characters of file text; when the requested range exceeds it the result
# is truncated *explicitly* (with ``truncated`` + ``next_offset``) so the model
# knows how to continue — never a silent cut.
READ_FILE_MAX_CHARS = int(os.getenv("READ_FILE_MAX_CHARS", "8000"))

# ---------------------------------------------------------------------------
# Project context cache (file content + symbol index).
# ---------------------------------------------------------------------------
# SQLite file backing the per-project file/symbol cache. Keyed by the
# normalized project path, so runs of the *same* project reuse it across tasks;
# different projects are isolated by their canonical path and never share rows.
CONTEXT_CACHE_DB = Path(
    os.getenv("CONTEXT_CACHE_DB", PROJECT_ROOT / "data" / "context_cache.sqlite3")
)
# When assembling the model request, older tool messages (before the current
# tool exchange) are truncated to this many characters; their conclusions are
# carried in the task-memory block instead. This is what actually saves input
# tokens — cache hits alone do not.
CONTEXT_TOOL_TAIL_CHARS = int(os.getenv("CONTEXT_TOOL_TAIL_CHARS", "400"))
# Upper bound on the injected task-memory block (chars).
CONTEXT_MEMORY_BLOCK_CHARS = int(os.getenv("CONTEXT_MEMORY_BLOCK_CHARS", "4000"))
# Failed tool results keep more context than successful ones when old history
# is compressed: a pytest/ruff failure must not lose its error detail or
# summary line (head + tail are kept).
CONTEXT_FAILURE_TAIL_CHARS = int(os.getenv("CONTEXT_FAILURE_TAIL_CHARS", "4000"))

# When True (default), high-risk tool calls (install packages, overwrite
# protected files, rollback, unbounded test runs, ...) pause the run and wait
# for explicit user approval. When False they are auto-authorized and run
# without blocking (each auto-grant is still logged as an ``approval`` step).
# Hard ``reject`` rules of the security/command policy are NEVER relaxed by
# this switch: a denied command or an out-of-workspace path stays denied.
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "true").lower() in {"1", "true", "yes", "on"}
# How long a run waits (seconds) for an approval decision before auto-denying.
APPROVAL_WAIT_SECONDS = float(os.getenv("APPROVAL_WAIT_SECONDS", "300"))
# How long a run waits (seconds) for the user's answer to an ``ask_user``
# question before continuing without it.
INPUT_WAIT_SECONDS = float(os.getenv("INPUT_WAIT_SECONDS", "600"))

# ---------------------------------------------------------------------------
# Write protection (snapshot / version check / rollback)
# ---------------------------------------------------------------------------
# Controlled directory holding pre-write snapshots and their manifests. It must
# live OUTSIDE the run's workspace so the agent's own file tools cannot read,
# overwrite or delete recovery data; when it resolves inside the workspace the
# guard falls back to an out-of-workspace location automatically.
WRITE_BACKUP_DIR = Path(
    os.getenv(
        "WRITE_BACKUP_DIR",
        Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "share")
        / "local-agent-workflow"
        / "write_backups",
    )
)
# Whole-file rewrite guard: refuse to replace a file larger than
# ``WRITE_SHRINK_GUARD_MIN_CHARS`` when the new content is smaller than this
# fraction of it (almost always a truncated model output), unless the caller
# explicitly passes ``allow_truncate`` (which requires approval).
WRITE_SHRINK_GUARD_MIN_CHARS = int(os.getenv("WRITE_SHRINK_GUARD_MIN_CHARS", "1000"))
WRITE_SHRINK_GUARD_RATIO = float(os.getenv("WRITE_SHRINK_GUARD_RATIO", "0.5"))

# Comma separated list of absolute/relative paths (relative to WORKSPACE_PATH)
# that must never be modified by write_file without approval, and never exposed
# in full by read_file (their values are redacted).
PROTECTED_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "data/workflow.sqlite3",
}

# Workspace roots that may NOT be nominated as a task's workspace. This is
# defense-in-depth: the real boundary is enforced per-tool via
# ``resolve_workspace_path``, but nominating a system directory as a workspace
# would otherwise let a task reach system files inside that "workspace".
# Compared case-insensitively against the fully-resolved path.
FORBIDDEN_WORKSPACE_ROOTS = {
    "C:\\Windows",
    "C:\\Windows\\System32",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/var",
}

# ---------------------------------------------------------------------------
# Task-decomposition workflow (control loop + per-goal task graph)
# ---------------------------------------------------------------------------
def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


# Decompose the goal with the LLM. When False the task graph is a single
# subtask wrapping the whole goal (legacy single-pass behaviour).
PLANNER_ENABLED = _bool_env("PLANNER_ENABLED", "true")
# Pause for user approval of every new plan version before executing it.
PLAN_REQUIRES_APPROVAL = _bool_env("PLAN_REQUIRES_APPROVAL", "true")
MAX_SUBTASKS = int(os.getenv("MAX_SUBTASKS", "8"))
# Bound on replans (rewrite-remaining + structural) per run.
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))
# Attempts per subtask (transient retries); exhausting escalates to structural.
MAX_SUBTASK_ATTEMPTS = int(os.getenv("MAX_SUBTASK_ATTEMPTS", "3"))
# How many times the user may reject a plan with feedback before the run is
# moved to needs_attention.
MAX_PLAN_REJECTIONS = int(os.getenv("MAX_PLAN_REJECTIONS", "3"))
PLANNER_OVERVIEW_MAX_CHARS = int(os.getenv("PLANNER_OVERVIEW_MAX_CHARS", "4000"))
# Grounded planning: index the repo and feed goal-relevant files/symbols/snippets
# (the "grounding pack") to the planner instead of only a shallow dir tree.
PLANNER_GROUNDING = _bool_env("PLANNER_GROUNDING", "true")
PLANNER_CONTEXT_MAX_CHARS = int(os.getenv("PLANNER_CONTEXT_MAX_CHARS", "12000"))
# Phase-2 planner: reject broad write globs / missing symbols; optional critic round.
PLANNER_STRICT = _bool_env("PLANNER_STRICT", "true")
PLANNER_CRITIC = _bool_env("PLANNER_CRITIC", "true")
# Live workflow: strict gates + evidence binding first, lenient fallback on failure.
PLANNER_WORKFLOW_STRICT = _bool_env("PLANNER_WORKFLOW_STRICT", "true")
# One extra strict re-plan when the model attempted phase-2 format but failed gates.
PLANNER_WORKFLOW_STRICT_RETRY = _bool_env("PLANNER_WORKFLOW_STRICT_RETRY", "true")
# Critic pass (one revise) on strict-passing plans in the live workflow.
PLANNER_WORKFLOW_CRITIC = _bool_env("PLANNER_WORKFLOW_CRITIC", "true")
# Explore-before-plan: bounded search/read/caller loop before drafting.
PLANNER_EXPLORE = _bool_env("PLANNER_EXPLORE", "true")
PLANNER_EXPLORE_BUDGET = int(os.getenv("PLANNER_EXPLORE_BUDGET", "6"))
# LLM review of the combined result against the ORIGINAL goal in final verify.
FINAL_VERIFY_LLM = _bool_env("FINAL_VERIFY_LLM", "true")
# Per-subtask budgets (passed to the ReAct executor).
SUBTASK_MAX_ITERATIONS = int(os.getenv("SUBTASK_MAX_ITERATIONS", "20"))
SUBTASK_TIMEOUT_SECONDS = float(os.getenv("SUBTASK_TIMEOUT_SECONDS", "600"))
# Run-level circuit breakers: tripping one moves the run to needs_attention.
MAX_RUN_STEPS = int(os.getenv("MAX_RUN_STEPS", "200"))          # model turns, all subtasks
MAX_RUN_SECONDS = float(os.getenv("MAX_RUN_SECONDS", "3600"))   # active execution time
MAX_RUN_TOKENS = int(os.getenv("MAX_RUN_TOKENS", "2000000"))    # prompt+completion
# Execution lease: heartbeat older than this = crashed attempt.
LEASE_TTL_SECONDS = float(os.getenv("LEASE_TTL_SECONDS", "120"))
LEASE_HEARTBEAT_SECONDS = float(os.getenv("LEASE_HEARTBEAT_SECONDS", "10"))
# Parallel subtask execution: at most this many ready subtasks with
# non-overlapping write_scope run at once (1 = strictly serial).
MAX_PARALLEL = max(1, int(os.getenv("MAX_PARALLEL", "2")))
# When true (and PLANNER_ENABLED), a still-ambiguous multi-requirement
# goal clause may be split by one planner LLM call. Default false so
# offline tests stay deterministic.
GOAL_CLAUSE_LLM_SPLIT = _bool_env("GOAL_CLAUSE_LLM_SPLIT", "false")

# Ephemeral build/cache dirs written by pytest/npm/tsc etc. Changes under these
# globs are not scope_violations (see ARTIFACT_POLICY). Comma-separated.
_DEFAULT_ARTIFACT_GLOBS = (
    "dist/**,build/**,out/**,target/**,"
    ".next/**,.nuxt/**,.turbo/**,.vite/**,"
    "coverage/**,htmlcov/**,"
    "**/__pycache__/**,__pycache__/**,"
    "**/.pytest_cache/**,.pytest_cache/**,"
    "**/.mypy_cache/**,.mypy_cache/**,"
    "**/.ruff_cache/**,.ruff_cache/**,"
    "**/*.egg-info/**,*.egg-info/**,"
    "**/node_modules/.cache/**,node_modules/.cache/**,"
    ".cache/**,.tox/**,.nox/**"
)
def _csv_globs(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [p.strip() for p in raw.split(",") if p.strip()]

ARTIFACT_PATH_GLOBS = _csv_globs(
    "ARTIFACT_PATH_GLOBS", "".join(_DEFAULT_ARTIFACT_GLOBS),
)
# allow = leave artifacts, record informational evidence (default)
# restore = put artifact paths back to pre-command snapshot
# fail = treat artifact writes like scope violations
ARTIFACT_POLICY = os.getenv("ARTIFACT_POLICY", "allow").strip().lower()
if ARTIFACT_POLICY not in {"allow", "restore", "fail"}:
    ARTIFACT_POLICY = "allow"

# Final-verify failure may trigger a replan (dedicated budget, default 1).
FINAL_VERIFY_REPLAN = _bool_env("FINAL_VERIFY_REPLAN", "true")
FINAL_VERIFY_MAX_REPLANS = max(0, int(os.getenv("FINAL_VERIFY_MAX_REPLANS", "1")))

# Soft command sandbox (snapshot + path policy). Not a kernel sandbox.
# When write_scope is set, always redirect TMP/PYTHONPYCACHEPREFIX into scratch
# and refuse argv absolute paths outside workspace+scratch (if enabled).
COMMAND_SOFT_SANDBOX = _bool_env("COMMAND_SOFT_SANDBOX", "true")
# Strip proxy env vars for child commands (does NOT block raw sockets).
COMMAND_DENY_NETWORK = _bool_env("COMMAND_DENY_NETWORK", "false")
# Force approval for curl/wget/Invoke-WebRequest/python urllib patterns.
COMMAND_BLOCK_NETWORK_TOOLS = _bool_env("COMMAND_BLOCK_NETWORK_TOOLS", "true")
# Read-only subtasks (empty write_scope): refuse commands that declare writes.
COMMAND_READONLY_FS_HINT = _bool_env("COMMAND_READONLY_FS_HINT", "true")
# Legacy plans missing write_scope: strict=true blocks modify until scoped via PATCH.
LEGACY_WRITE_SCOPE_STRICT = _bool_env("LEGACY_WRITE_SCOPE_STRICT", "true")
# Cost estimate (per 1K tokens). Defaults approximate DeepSeek-chat list prices
# in CNY; adjust to your provider. Display only — never used for decisions.
TOKEN_PRICE_INPUT_PER_1K = float(os.getenv("TOKEN_PRICE_INPUT_PER_1K", "0.002"))
TOKEN_PRICE_OUTPUT_PER_1K = float(os.getenv("TOKEN_PRICE_OUTPUT_PER_1K", "0.008"))
COST_CURRENCY = os.getenv("COST_CURRENCY", "CNY")


def is_llm_configured() -> bool:
    return bool(DEEPSEEK_API_KEY)
