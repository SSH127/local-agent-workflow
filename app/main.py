import copy
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import config, storage
from app.agents import agent_executor
from app.config import APP_HOST, APP_PORT
from app.engine import planning as P
from app.engine.runner import Runner
from app.engine.workflow import recover_runs_on_startup
from app.schemas import (
    PAUSED_STATUSES,
    ApprovalRequest,
    InputRequest,
    ManualResultRequest,
    PlanApprovalRequest,
    RunCreate,
    SubtaskConfirmRequest,
    SubtaskEditRequest,
)
from app.storage import (
    get_run,
    init_db,
    list_runs,
    list_steps,
    save_run,
    transition_run_status,
)
from app.tools import security

runner = Runner()


def now() -> str:
    return datetime.now(UTC).isoformat()


def require_token(request: Request) -> None:
    """Shared-token guard for the local API.

    Authentication is disabled when ``LOCAL_AGENT_TOKEN`` is empty (the local
    development default). When set, every protected endpoint requires
    ``Authorization: Bearer <token>``. EventSource cannot set headers, so
    ``GET .../events?access_token=<token>`` is also accepted (SSE only use).
    This is local access protection only — not a production-grade identity /
    multi-user system.
    """
    expected = config.LOCAL_AGENT_TOKEN
    if not expected:
        return
    provided = request.headers.get("authorization", "")
    if provided == f"Bearer {expected}":
        return
    # Query fallback for SSE (EventSource has no custom headers).
    q = request.query_params.get("access_token") or request.query_params.get("token")
    if q and q == expected:
        return
    raise HTTPException(status_code=401, detail="未授权：令牌缺失或不正确")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Runs left in ``running`` across a restart: a live execution lease held by
    # another process means "truly running" (left alone); otherwise the attempt
    # crashed and the run is marked ``interrupted`` (never auto-replayed;
    # ``POST /runs/{id}/resume`` recovers it idempotently from its checkpoint).
    recover_runs_on_startup()
    yield


app = FastAPI(
    title="Local Agent Workflow",
    version="0.3.6",
    description="Local-first AI Agent workflow API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "local-agent-workflow"}


@app.get("/runs/{run_id}/events", dependencies=[Depends(require_token)])
async def run_events(run_id: str, after_seq: int = 0):
    """SSE for one run: durable replay via ``after_seq``, then live bus / DB-tail.

    Polling ``GET /runs/{id}`` remains the fallback. With ``EVENTS_DB_TAIL``
    (default true) each uvicorn worker can serve events produced elsewhere by
    reading SQLite (WAL). Prefer ``workers=1`` for this local app.
    """
    import asyncio
    import json
    import queue as _queue

    from app import config
    from app import storage as store
    from app.events import bus

    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在。")

    after = max(0, int(after_seq or 0))
    q = bus.subscribe(run_id)
    db_tail = bool(getattr(config, "EVENTS_DB_TAIL", True))
    tail_s = max(0.1, float(getattr(config, "EVENTS_DB_TAIL_MS", 500) or 500) / 1000.0)

    async def gen():
        last_seq = after
        try:
            hello = {"type": "connected", "run_id": run_id, "after_seq": after}
            yield f"data: {json.dumps(hello, ensure_ascii=False)}\n\n"
            # Replay missed durable events first.
            for ev in store.list_run_events_after(run_id, last_seq):
                last_seq = max(last_seq, int(ev.get("seq") or 0))
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            loop = asyncio.get_running_loop()
            while True:
                wait = tail_s if db_tail else 15.0
                try:
                    ev = await loop.run_in_executor(None, lambda w=wait: q.get(timeout=w))
                    seq = int(ev.get("seq") or 0)
                    if seq and seq <= last_seq:
                        continue
                    if seq:
                        last_seq = seq
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                except _queue.Empty:
                    if db_tail:
                        for ev in store.list_run_events_after(run_id, last_seq, limit=100):
                            last_seq = max(last_seq, int(ev.get("seq") or 0))
                            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                    else:
                        yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(run_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })



@app.post("/runs", status_code=201, dependencies=[Depends(require_token)])
def create_run(payload: RunCreate):
    # Resolve the run's workspace from (in order) an explicit API field, then a
    # path written directly in the goal prose. Either way the path is validated
    # up front: an explicitly-named-but-invalid directory aborts here with a
    # clear 400 instead of silently falling back to the default workspace.
    workspace: str | None = None
    workspace_source = "default"
    if payload.workspace_path and payload.workspace_path.strip():
        workspace_source = "field"
        try:
            workspace = str(security.resolve_workspace_root(payload.workspace_path))
        except security.SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        try:
            detected = security.detect_workspace(payload.goal)
        except security.SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if detected is not None:
            workspace = detected
            workspace_source = "goal"

    run_id = str(uuid4())
    timestamp = now()
    save_run({
        "run_id": run_id, "goal": payload.goal, "status": agent_executor.STATUS_QUEUED,
        "attempts": 0, "summary": "任务已排队", "error": None,
        "workspace_path": workspace,
        "state": {"workspace_source": workspace_source},
        "created_at": timestamp, "updated_at": timestamp,
    })
    runner.submit(run_id, payload.goal, payload.max_attempts, workspace=workspace)
    return get_run(run_id)


@app.get("/runs", dependencies=[Depends(require_token)])
def get_runs(limit: int = Query(default=50, ge=1, le=200)):
    return list_runs(limit)


@app.get("/runs/{run_id}", dependencies=[Depends(require_token)])
def get_run_by_id(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/steps", dependencies=[Depends(require_token)])
def get_run_steps(run_id: str, limit: int = Query(default=200, ge=1, le=1000)):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return list_steps(run_id, limit)


@app.post("/runs/{run_id}/approve", dependencies=[Depends(require_token)])
def approve_run(run_id: str, payload: ApprovalRequest):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != agent_executor.STATUS_WAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"任务当前状态为 {run['status']}，无需批准。",
        )

    state = dict(run.get("state") or {})
    pending = state.get("pending_approval")
    pending_id = pending.get("approval_id") if isinstance(pending, dict) else None

    # A supplied approval id must match the run's currently pending request.
    if payload.approval_id is not None and payload.approval_id != pending_id:
        raise HTTPException(status_code=400, detail="无效审批 ID：与当前待审批请求不匹配。")

    if pending_id:
        outcome = runner.approve(pending_id, payload.approved, payload.reason)
        if outcome in ("resolved", "already_resolved"):
            return get_run(run_id)
        # outcome == "not_found": the in-memory gate was lost (process restart);
        # fall through to the restart-handling path below.

    # No live waiter (e.g. the process restarted while waiting). Carry the
    # decision into the resumed run instead of silently guessing state. The
    # transition is atomic: concurrent approve/deny requests race on the
    # conditional UPDATE and only one wins, so a single run can never spawn two
    # resume threads (the losing caller returns the run's current state below).
    if payload.approved:
        decision = {
            "approval_id": pending_id or payload.approval_id or "",
            "run_id": run_id,
            "tool_name": pending.get("tool_name") if isinstance(pending, dict) else None,
            "args": pending.get("args") if isinstance(pending, dict) else {},
            "risk": pending.get("risk_reason") if isinstance(pending, dict) else None,
            "approved": True,
            "reason": payload.reason,
        }
        claimed = transition_run_status(
            run_id,
            agent_executor.STATUS_WAITING_APPROVAL,
            agent_executor.STATUS_RUNNING,
            state_patch={"approval_decision": decision},
            updated_at=now(),
        )
        if claimed:
            runner.resume(run_id, run["goal"])
    else:
        transition_run_status(
            run_id,
            agent_executor.STATUS_WAITING_APPROVAL,
            agent_executor.STATUS_FAILED,
            summary="用户拒绝了待批准的操作。",
            error=payload.reason or "rejected by user",
            updated_at=now(),
        )
    return get_run(run_id)


@app.post("/runs/{run_id}/input", dependencies=[Depends(require_token)])
def provide_input(run_id: str, payload: InputRequest):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != agent_executor.STATUS_WAITING_INPUT:
        raise HTTPException(
            status_code=409,
            detail=f"任务当前状态为 {run['status']}，无需补充信息。",
        )
    # Live path: the run thread is blocked in ``ask_user`` waiting on the gate;
    # hand it the answer directly (same pattern as approvals).
    state = dict(run.get("state") or {})
    pending = state.get("pending_input")
    input_id = pending.get("input_id") if isinstance(pending, dict) else None
    if input_id:
        outcome = runner.provide_input(input_id, payload.message)
        if outcome in ("resolved", "already_resolved"):
            return get_run(run_id)
        # outcome == "not_found": the in-memory waiter was lost (process
        # restart); fall through and resume from the persisted state.

    # Restart path: record the user's message and resume the run; the pending
    # ``ask_user`` call is replayed and answered with it. The transition is
    # atomic so concurrent input requests cannot spawn two resume threads.
    claimed = transition_run_status(
        run_id,
        agent_executor.STATUS_WAITING_INPUT,
        agent_executor.STATUS_RUNNING,
        state_patch={"user_input": payload.message},
        updated_at=now(),
    )
    if claimed:
        runner.resume(run_id, run["goal"])
    return get_run(run_id)


# ---------------------------------------------------------------------------
# Task graph (plan) API
# ---------------------------------------------------------------------------

def _run_or_404(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _plan_or_404(run_id: str) -> dict:
    plan = storage.get_plan(run_id)
    if not plan or not plan.get("subtasks"):
        raise HTTPException(status_code=404, detail="该任务还没有任务图")
    return plan


def _cost(tokens: dict | None) -> float:
    tokens = tokens or {}
    return round(
        int(tokens.get("prompt") or 0) / 1000 * config.TOKEN_PRICE_INPUT_PER_1K
        + int(tokens.get("completion") or 0) / 1000 * config.TOKEN_PRICE_OUTPUT_PER_1K,
        6,
    )


def _lazy_recover(run: dict) -> dict:
    """A ``running`` run with no thread here and no live lease anywhere is a
    crashed attempt (e.g. another process died): mark it interrupted."""
    if run["status"] != agent_executor.STATUS_RUNNING or runner.is_active(run["run_id"]):
        return run
    leases = [lease for lease in storage.get_leases(run["run_id"]).values()
              if lease.get("state") != "released"]
    # Parallel subtasks: the run is crashed only when NONE of its unreleased
    # per-subtask leases has a fresh heartbeat.
    if not leases or any(P.lease_is_live(lease) for lease in leases):
        return run
    transition_run_status(
        run["run_id"], agent_executor.STATUS_RUNNING, storage.STATUS_INTERRUPTED,
        summary="任务中断", error="执行租约心跳已过期：判定为崩溃的执行尝试。", updated_at=now(),
    )
    return get_run(run["run_id"]) or run


def _plan_view(run: dict, plan: dict) -> dict:
    view = dict(plan)
    subtasks = []
    ready = {s["id"] for s in P.ready_subtasks(plan)}
    total_prompt = total_completion = 0
    for s in plan.get("subtasks", []):
        item = {k: v for k, v in s.items() if k != "last_checkpoint"}
        tokens = s.get("tokens") or {}
        total_prompt += int(tokens.get("prompt") or 0)
        total_completion += int(tokens.get("completion") or 0)
        item["cost"] = _cost(tokens)
        item["retries"] = max(0, int(s.get("attempt_number") or 0) - 1)
        item["ready"] = s["id"] in ready
        item["needs_confirmation"] = P.needs_preexec_confirmation(s) and not s.get("confirmed")
        subtasks.append(item)
    view["subtasks"] = subtasks
    if plan.get("goal_clauses"):
        view["clause_report"] = P.clause_report(plan)
    totals = dict(plan.get("totals") or {})
    totals.update({
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "cost": _cost({"prompt": total_prompt, "completion": total_completion}),
        "currency": config.COST_CURRENCY,
        "limits": {
            "max_run_steps": config.MAX_RUN_STEPS,
            "max_run_seconds": config.MAX_RUN_SECONDS,
            "max_run_tokens": config.MAX_RUN_TOKENS,
        },
    })
    view["totals"] = totals
    return {
        "run_id": run["run_id"],
        "run_status": run["status"],
        "plan_version": plan.get("version"),
        "plan_revision": plan.get("revision"),
        "plan": view,
        "lease": storage.get_lease(run["run_id"]),  # latest heartbeat (compat)
        "leases": storage.get_leases(run["run_id"]),
        "max_parallel": config.MAX_PARALLEL,
    }


def _resume_from(run: dict, from_status: str, plan: dict | None = None) -> None:
    if plan is not None:
        _save_plan(run["run_id"], plan)
    claimed = transition_run_status(
        run["run_id"], from_status, agent_executor.STATUS_RUNNING, updated_at=now(),
    )
    if claimed:
        runner.resume(run["run_id"], run["goal"])


def _intervene(target: dict, item: dict) -> None:
    target.setdefault("interventions", []).append({**item, "at": now()})


def _save_plan(run_id: str, plan: dict) -> None:
    try:
        storage.save_plan(run_id, plan)
    except storage.GoalImmutableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _record_version(run_id: str, plan: dict, parent: dict | None, reason: str,
                    failure_class: str | None = None) -> None:
    """Append-only plan history. A plan written before 0.3.1 has no history
    yet: its pre-edit state is recorded first so the diff has a parent."""
    if parent is not None and not storage.list_plan_versions(run_id):
        storage.append_plan_version(run_id, P.version_entry(
            parent, None, reason="升级前已有的任务图（补记）"))
    storage.append_plan_version(run_id, P.version_entry(
        plan, parent, reason=reason, failure_class=failure_class))


def _validate_remaining(plan: dict, raw: list, done: list[dict]) -> list[dict]:
    try:
        return P.validate_subtasks(raw, max(config.MAX_SUBTASKS, len(raw)),
                                   known_nodes=done, clause_ids=P.clause_ids_of(plan))
    except P.PlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id}/plan", dependencies=[Depends(require_token)])
def get_run_plan(run_id: str):
    run = _lazy_recover(_run_or_404(run_id))
    return _plan_view(run, _plan_or_404(run_id))


@app.post("/runs/{run_id}/plan/approve", dependencies=[Depends(require_token)])
def approve_plan(run_id: str, payload: PlanApprovalRequest):
    run = _run_or_404(run_id)
    if run["status"] != agent_executor.STATUS_WAITING_PLAN_APPROVAL:
        raise HTTPException(status_code=409, detail=f"任务当前状态为 {run['status']}，没有待批准的计划。")
    plan = _plan_or_404(run_id)
    if payload.is_approve():
        if payload.subtasks is not None:
            done = [s for s in plan["subtasks"] if s["status"] == P.SUBTASK_DONE]
            parent = copy.deepcopy(plan)
            specs = _validate_remaining(plan, payload.subtasks, done)
            old_remaining = [s for s in plan["subtasks"] if s["status"] != P.SUBTASK_DONE]
            diff = P.graph_diff(old_remaining, specs)
            if diff["added"] or diff["removed"] or diff["changed"]:
                plan["history"] = (list(plan.get("history") or []) + [
                    {**P.snapshot_history(plan), "diff": diff, "reason": "用户编辑后批准"}])[-10:]
                old_by_id = {s["id"]: s for s in old_remaining}
                nodes = []
                for spec in specs:
                    prev = old_by_id.get(spec["id"])
                    node = P.with_runtime([spec])[0]
                    if prev is not None and not P.contract_diff(prev, spec):
                        node = prev
                    nodes.append(node)
                plan["subtasks"] = done + nodes
                plan["source"] = "edited"
                P.assign_lineage(parent["subtasks"], nodes)
                structural = P.is_structural_graph_diff(diff)
                if structural:
                    P.bump_version(plan, parent)
                else:
                    P.bump_revision(plan, parent)
                _record_version(run_id, plan, parent,
                                "用户编辑后批准" + ("（结构性）" if structural else "（非结构性修订）"),
                                "manual_edit")
        plan["approval_status"] = P.PLAN_APPROVED
        plan["approved_version"] = plan["version"]
        plan["attention"] = None
        _intervene(plan, {"type": "plan_approval", "approved": True, "version": plan["version"],
                          "edited": payload.subtasks is not None, "note": payload.feedback})
    else:
        plan["approval_status"] = P.PLAN_REJECTED
        plan["plan_rejections"] = int(plan.get("plan_rejections") or 0) + 1
        plan["feedback"] = list(plan.get("feedback") or []) + [
            (payload.feedback or "").strip() or "（用户未填写具体反馈，请换一种拆解思路）"]
        _intervene(plan, {"type": "plan_approval", "approved": False,
                          "version": plan.get("version"), "note": payload.feedback})
    # Both branches continue the control loop: approve → execute the approved
    # version; reject → decompose again with the feedback (bounded).
    _resume_from(run, agent_executor.STATUS_WAITING_PLAN_APPROVAL, plan)
    return _plan_view(get_run(run_id) or run, storage.get_plan(run_id) or plan)


def _sub_or_404(plan: dict, subtask_id: str) -> dict:
    sub = P.find_subtask(plan, subtask_id)
    if sub is None:
        raise HTTPException(status_code=404, detail=f"子任务 {subtask_id} 不存在")
    return sub


@app.post("/runs/{run_id}/subtasks/{subtask_id}/confirm", dependencies=[Depends(require_token)])
def confirm_subtask(run_id: str, subtask_id: str, payload: SubtaskConfirmRequest):
    run = _run_or_404(run_id)
    if run["status"] != agent_executor.STATUS_WAITING_SUBTASK_APPROVAL:
        raise HTTPException(status_code=409, detail=f"任务当前状态为 {run['status']}，没有待确认的子任务。")
    plan = _plan_or_404(run_id)
    if plan.get("pending_confirmation") != subtask_id:
        raise HTTPException(status_code=400, detail="该子任务不是当前待确认的子任务。")
    sub = _sub_or_404(plan, subtask_id)
    _intervene(sub, {"type": "preexec_confirmation", "approved": payload.approved,
                     "note": payload.note})
    if payload.approved:
        sub["confirmed"] = True
    else:
        sub["status"] = P.SUBTASK_SKIPPED
        sub["verify_note"] = f"用户拒绝执行：{payload.note or ''}"
    plan["pending_confirmation"] = None
    _resume_from(run, agent_executor.STATUS_WAITING_SUBTASK_APPROVAL, plan)
    return _plan_view(get_run(run_id) or run, storage.get_plan(run_id) or plan)


def _require_paused(run: dict) -> None:
    if run["status"] not in PAUSED_STATUSES or runner.is_active(run["run_id"]):
        raise HTTPException(
            status_code=409,
            detail=f"任务当前状态为 {run['status']}，只能在暂停/待处理时人工接管。",
        )


@app.patch("/runs/{run_id}/subtasks/{subtask_id}", dependencies=[Depends(require_token)])
def edit_subtask(run_id: str, subtask_id: str, payload: SubtaskEditRequest):
    run = _run_or_404(run_id)
    _require_paused(run)
    plan = _plan_or_404(run_id)
    sub = _sub_or_404(plan, subtask_id)
    if sub["status"] == P.SUBTASK_DONE:
        raise HTTPException(status_code=409, detail="已成功的子任务不能修改契约。")
    changes = payload.model_dump(exclude_none=True, exclude={"note"})
    candidate = {**sub, **changes}
    done = [s for s in plan["subtasks"] if s["status"] == P.SUBTASK_DONE]
    remaining = [candidate if s["id"] == subtask_id else s
                 for s in plan["subtasks"] if s["status"] != P.SUBTASK_DONE]
    spec_keys = ("id", "title", "description", "depends_on", "kind",
                 "acceptance_criteria", "risk", "reversible", "write_scope", "serves_clauses")
    raw = [{k: s.get(k) for k in spec_keys if k in s or k in ("id", "title")} for s in remaining]
    normalized = {n["id"]: n for n in _validate_remaining(plan, raw, done)}
    if "write_scope" in changes:
        changes["write_scope"] = normalized[subtask_id]["write_scope"]
        candidate["write_scope"] = changes["write_scope"]
    diff = P.contract_diff(sub, candidate)
    if not diff:
        return _plan_view(run, plan)
    parent = copy.deepcopy(plan)
    sub.update(changes)
    _intervene(sub, {"type": "manual_edit", "diff": diff, "note": payload.note})
    if P.is_structural_diff(diff):
        plan["history"] = (list(plan.get("history") or []) + [{
            **P.snapshot_history(plan), "diff": {"changed": {subtask_id: diff}},
            "reason": "人工修改子任务契约（结构性）"}])[-10:]
        P.bump_version(plan, parent)
        plan["source"] = "edited"
        if sub["status"] in (P.SUBTASK_FAILED, P.SUBTASK_SKIPPED, P.SUBTASK_CANCELLED):
            sub["status"] = P.SUBTASK_PENDING
        sub["confirmed"] = False
        _record_version(run_id, plan, parent, f"人工修改子任务 {subtask_id} 契约（结构性）",
                        "manual_edit")
        if config.PLAN_REQUIRES_APPROVAL:
            plan["approval_status"] = P.PLAN_PENDING
            plan["approved_version"] = None
            _save_plan(run_id, plan)
            storage.update_run_status(run_id, agent_executor.STATUS_WAITING_PLAN_APPROVAL,
                                      summary=f"任务图 v{plan['version']}（人工修改）等待批准",
                                      updated_at=now())
            return _plan_view(get_run(run_id) or run, plan)
        plan["approval_status"] = P.PLAN_APPROVED
        plan["approved_version"] = plan["version"]
    else:
        P.bump_revision(plan, parent)
        _record_version(run_id, plan, parent, f"人工修改子任务 {subtask_id}（非结构性修订）",
                        "manual_edit")
    _save_plan(run_id, plan)
    return _plan_view(run, plan)


@app.get("/runs/{run_id}/plan/versions", dependencies=[Depends(require_token)])
def list_plan_versions(run_id: str):
    """Append-only plan history (summaries, oldest first)."""
    _run_or_404(run_id)
    plan = storage.get_plan(run_id) or {}
    return {"run_id": run_id, "current_version": plan.get("version"),
            "current_revision": plan.get("revision"),
            "goal": plan.get("goal"), "goal_clauses": plan.get("goal_clauses") or [],
            "versions": storage.list_plan_versions(run_id)}


@app.get("/runs/{run_id}/plan/versions/{version}", dependencies=[Depends(require_token)])
def get_plan_version(run_id: str, version: int, revision: int | None = Query(default=None, ge=1)):
    """One historical plan: full graph + diff against its parent (read-only)."""
    _run_or_404(run_id)
    entry = storage.get_plan_version(run_id, version, revision)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"任务图版本 v{version} 不存在")
    return {"run_id": run_id, **entry}


@app.post("/runs/{run_id}/subtasks/{subtask_id}/manual-result",
          dependencies=[Depends(require_token)])
def manual_result(run_id: str, subtask_id: str, payload: ManualResultRequest):
    run = _run_or_404(run_id)
    _require_paused(run)
    plan = _plan_or_404(run_id)
    sub = _sub_or_404(plan, subtask_id)
    sub["status"] = payload.status
    sub["manual"] = True
    sub["result_summary"] = f"[人工] {payload.note}"
    sub["verify_note"] = "人工标记的结果（manual），未经执行器证据验收"
    sub["finished_at"] = now()
    sub.setdefault("acceptance_evidence", []).append(
        P.manual_evidence(payload.note, payload.evidence_text, payload.status))
    _intervene(sub, {"type": "manual_result", "status": payload.status, "note": payload.note})
    if plan.get("pending_confirmation") == subtask_id:
        plan["pending_confirmation"] = None
    storage.save_plan(run_id, plan)
    return _plan_view(run, plan)


@app.post("/runs/{run_id}/resume", dependencies=[Depends(require_token)])
def resume_run(run_id: str):
    run = _run_or_404(run_id)
    allowed = {agent_executor.STATUS_NEEDS_ATTENTION, storage.STATUS_INTERRUPTED,
               agent_executor.STATUS_FAILED, agent_executor.STATUS_WAITING_SUBTASK_APPROVAL}
    if run["status"] not in allowed or runner.is_active(run_id):
        raise HTTPException(status_code=409, detail=f"任务当前状态为 {run['status']}，不能继续。")
    plan = storage.get_plan(run_id)
    if plan is not None:
        plan["attention"] = None
        _intervene(plan, {"type": "resume", "from_status": run["status"]})
    _resume_from(run, run["status"], plan)
    return get_run(run_id)


@app.post("/runs/{run_id}/cancel", dependencies=[Depends(require_token)])
def cancel_run(run_id: str):
    run = _run_or_404(run_id)
    if run["status"] in (agent_executor.STATUS_SUCCEEDED, agent_executor.STATUS_CANCELLED):
        raise HTTPException(status_code=409, detail=f"任务当前状态为 {run['status']}，无法取消。")
    plan = storage.get_plan(run_id)
    if run["status"] in PAUSED_STATUSES and not runner.is_active(run_id):
        if plan is not None:
            for s in plan.get("subtasks", []):
                if s["status"] in (P.SUBTASK_PENDING, P.SUBTASK_RUNNING, P.SUBTASK_FAILED):
                    s["status"] = P.SUBTASK_CANCELLED
            _intervene(plan, {"type": "cancel"})
            storage.save_plan(run_id, plan)
        transition_run_status(run_id, run["status"], agent_executor.STATUS_CANCELLED,
                              summary="用户取消了剩余子任务", updated_at=now())
        return get_run(run_id)
    # A live thread: request cancellation; the control loop stops at the next
    # subtask boundary. A blocked approval / question is resolved as declined.
    if plan is not None:
        plan["cancel_requested"] = True
        _intervene(plan, {"type": "cancel_requested"})
        storage.save_plan(run_id, plan)
    pending = runner.gate.get_pending_for_run(run_id)
    if pending is not None:
        runner.gate.resolve_if_pending(pending.approval_id, False, "用户取消任务")
    return get_run(run_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=True)
