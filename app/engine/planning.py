"""Task graph (per-goal) + helpers for the fixed control loop.

Two layers — keep them separate:

1. **Control loop** (fixed, never rewritten per goal) lives in
   ``app/engine/workflow.py``: decompose → plan approval → pick next →
   execute → accept/reject by evidence → (retry / rewrite remaining /
   re-approve) → final verify exit → finish/fail.

2. **Task graph** (this module): the per-goal DAG of subtasks with
   ``depends_on``, status, acceptance criteria and attempt number. Regenerated
   on each new goal; on failure only the *remaining* part may change (or the
   whole plan is bumped and re-approved). The frontend draws THIS graph.

Persistence shape (stored in ``runs.state_json.plan`` + the ``plan_version`` /
``execution_lease`` columns)::

    {
      "version": 1,                 # plan_version; bumps on structural change
      "source": "llm" | "fallback" | "single" | "edited" | "replan",
      "approval_status": "pending" | "approved" | "rejected",
      "rewrites": 0,                # remaining-only rewrites (same version)
      "structural_replans": 0,      # version bumps that need re-approval
      "current_subtask_id": null,
      "subtasks": [ {
          id, title, description, depends_on, kind, acceptance_criteria,
          status, attempt_number, result_summary, error, stop_reason,
          acceptance_evidence, started_at, finished_at
      } ],
      "history": [ {version, source, done_ids, snapshot_at} ],
      "notes": []
    }

Acceptance is decided from ``acceptance_evidence`` (command exit codes, file
hashes, tool outcomes) — never from the executor's prose "done" claim.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator

from app import config
from app.tools.security import SecurityError, is_protected_file, resolve_workspace_root
from app.write_scope import ScopeError, normalize_scope, overlapping_pairs, scopes_overlap

# --- Subtask / plan status -------------------------------------------------
SUBTASK_PENDING = "pending"
SUBTASK_READY = "ready"  # computed, not persisted; deps satisfied
SUBTASK_RUNNING = "running"
SUBTASK_DONE = "done"
SUBTASK_FAILED = "failed"
SUBTASK_SKIPPED = "skipped"      # user declined / superseded; does NOT satisfy deps
SUBTASK_CANCELLED = "cancelled"  # run cancelled before this subtask finished
SUBTASK_STATUSES = (
    SUBTASK_PENDING, SUBTASK_RUNNING, SUBTASK_DONE, SUBTASK_FAILED,
    SUBTASK_SKIPPED, SUBTASK_CANCELLED,
)
RISK_LEVELS = ("low", "medium", "high")
# Contract fields whose change is *structural* (new plan version + approval).
STRUCTURAL_FIELDS = (
    "depends_on", "acceptance_criteria", "kind", "description", "risk", "reversible",
    "write_scope", "serves_clauses",
)
# Spec (contract) fields stored in version snapshots.
SPEC_FIELDS = (
    "id", "title", "description", "depends_on", "kind", "acceptance_criteria",
    "risk", "reversible", "write_scope", "serves_clauses", "derived_from",
)
# Static risk heuristic: text that implies deletion / external side effects /
# config changes must not be marked ``risk=low``.
_RISKY_TEXT = re.compile(
    r"(删除|移除文件|清空|drop\s+table|rm\s+-|truncate|部署|发布|上线|git\s+push|推送|"
    r"迁移数据库|数据库迁移|修改配置|配置文件|\.env\b|生产环境|外部\s*api|发送邮件|"
    r"delete|deploy|publish|migrat)",
    re.IGNORECASE,
)

PLAN_PENDING = "pending"
PLAN_APPROVED = "approved"
PLAN_REJECTED = "rejected"

# Failure routing after evidence-based reject (control-loop branches).
FAIL_TRANSIENT = "transient"          # retry same subtask, bump attempt
FAIL_REWRITE = "rewrite_remaining"    # rewrite remaining, keep plan_version
FAIL_STRUCTURAL = "structural"        # bump plan_version → plan approval
FAIL_FINAL_VERIFY = "final_verify"    # final verify → replan (plan gaps)

# Plans that wrap the whole goal in one subtask keep legacy single-pass
# semantics (subtask goal == run goal; final verify still separate).
LEGACY_SOURCES = {"single", "fallback", "legacy"}

KIND_LABELS = {"analyze": "分析", "modify": "修改", "verify": "验证"}

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")

# Transient failures: retry the SAME subtask (attempt+1), plan unchanged.
# Deliberately narrow — a wrong approach must not be blindly re-run.
_TRANSIENT_STOP = {"task_timeout", "model_error", "crashed", "lease_expired"}
_TRANSIENT_ERROR_MARKERS = (
    "database is locked", "锁冲突", "文件被占用", "being used by another process",
    "进程重启", "Connection reset", "连接重置", "网络错误", "HTTP 429", "HTTP 502",
    "HTTP 503", "HTTP 504",
)


class PlanError(ValueError):
    """Raised when a plan / task-graph is structurally invalid."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Spec (what the LLM / user proposes)
# ---------------------------------------------------------------------------

class SubtaskSpec(BaseModel):
    """One node of the per-goal task DAG."""

    id: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    depends_on: list[str] = Field(default_factory=list)
    kind: Literal["analyze", "modify", "verify"] = "analyze"
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=10)
    # Contract risk marking (set by the planner, checked statically).
    risk: Literal["low", "medium", "high"] = "low"
    reversible: bool = True
    # Paths/globs (workspace-relative) this subtask may write. Empty = read-only.
    # analyze / verify default to read-only; modify must declare a scope.
    write_scope: list[str] = Field(default_factory=list, max_length=30)
    # Goal clause ids (e.g. "C1") this subtask serves (goal traceability).
    serves_clauses: list[str] = Field(default_factory=list, max_length=30)
    # Lineage: ids in the parent plan version this node derives from.
    derived_from: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        value = value.strip()
        if not _ID_RE.match(value):
            raise ValueError("子任务 id 只能包含字母、数字、_ . -（1-32 字符）")
        return value

    @field_validator("title", "description")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("depends_on", mode="before")
    @classmethod
    def _deps(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            return [str(value)]
        return [str(v).strip() for v in value]

    @field_validator("write_scope", mode="before")
    @classmethod
    def _scope(cls, value: Any) -> Any:
        try:
            return normalize_scope(value)
        except ScopeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("serves_clauses", "derived_from", mode="before")
    @classmethod
    def _id_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            value = [value]
        return list(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))

    @field_validator("acceptance_criteria", mode="before")
    @classmethod
    def _criteria(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v).strip()[:500] for v in value if str(v).strip()]


class PlanSpec(BaseModel):
    subtasks: list[SubtaskSpec] = Field(min_length=1)


def validate_subtasks(
    raw: Any,
    max_subtasks: int | None = None,
    *,
    known_ids: set[str] | None = None,
    known_nodes: list[dict[str, Any]] | None = None,
    clause_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate a list (or ``{"subtasks": [...]}``) of subtask specs.

    Checks pydantic schema, count ≤ max, unique ids (also vs ``known_ids`` —
    e.g. already-done subtasks kept across a rewrite), every dependency exists,
    no self-dep, no cycle, risk marking, and (static checks before approval):

    * a ``modify`` subtask must declare a non-empty ``write_scope``;
    * two subtasks whose ``write_scope`` overlap must be connected by a
      dependency path (either direction) — otherwise they could run
      concurrently / in either order and clobber each other;
    * goal traceability (when ``clause_ids`` is given): each subtask serves at
      least one valid clause and every clause is covered (by these specs or by
      ``known_nodes``, the already-done subtasks kept across a replan).

    Returns normalized specs in topological order.
    """
    limit = max_subtasks if max_subtasks is not None else config.MAX_SUBTASKS
    known = set(known_ids or set())
    if isinstance(raw, dict):
        raw = raw.get("subtasks")
    try:
        spec = PlanSpec.model_validate({"subtasks": raw})
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise PlanError(f"计划格式无效（{loc}）：{first.get('msg', exc)}") from exc
    items = [s.model_dump() for s in spec.subtasks]
    if len(items) > limit:
        raise PlanError(f"子任务数量 {len(items)} 超过上限 {limit}。")
    ids = [s["id"] for s in items]
    dup = {i for i in ids if ids.count(i) > 1} | (set(ids) & known)
    if dup:
        raise PlanError(f"子任务 id 重复：{', '.join(sorted(dup))}")
    known_list = list(known_nodes or [])
    known |= {n["id"] for n in known_list}
    if dup := (set(ids) & {n["id"] for n in known_list}):
        raise PlanError(f"子任务 id 重复：{', '.join(sorted(dup))}")
    all_ids = set(ids) | known
    for s in items:
        check_risk_marking(s)
        s["depends_on"] = list(dict.fromkeys(s["depends_on"]))
        if s["id"] in s["depends_on"]:
            raise PlanError(f"子任务 {s['id']} 不能依赖自己。")
        missing = [d for d in s["depends_on"] if d not in all_ids]
        if missing:
            raise PlanError(
                f"子任务 {s['id']} 依赖了不存在的子任务：{', '.join(missing)}"
            )
    ordered = topological_order(items, known)
    check_write_scopes(ordered, known_list)
    if clause_ids is not None:
        check_clause_coverage(ordered, clause_ids, known_list)
    return ordered


def _ancestors(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(node: str, stack: frozenset[str]) -> set[str]:
        if node in memo:
            return memo[node]
        result: set[str] = set()
        for dep in graph.get(node, []):
            if dep in stack:
                continue
            result.add(dep)
            result |= visit(dep, stack | {dep})
        memo[node] = result
        return result

    for node in graph:
        visit(node, frozenset({node}))
    return memo


def check_write_scopes(
    specs: list[dict[str, Any]], known_nodes: list[dict[str, Any]] | None = None,
) -> None:
    """Static write-scope checks (see :func:`validate_subtasks`)."""
    for s in specs:
        if s.get("kind") == "modify" and not s.get("write_scope"):
            raise PlanError(
                f"子任务 {s['id']} 是修改类（modify），但 write_scope 为空；"
                "请列出它允许修改的文件/目录（相对工作区的路径或 glob）。"
            )
    graph = {n["id"]: list(n.get("depends_on") or []) for n in (known_nodes or [])}
    graph.update({s["id"]: list(s.get("depends_on") or []) for s in specs})
    anc = _ancestors(graph)
    for i, a in enumerate(specs):
        for b in specs[i + 1:]:
            pairs = overlapping_pairs(a.get("write_scope"), b.get("write_scope"))
            if not pairs:
                continue
            if a["id"] in anc.get(b["id"], set()) or b["id"] in anc.get(a["id"], set()):
                continue
            shown = "、".join(f"{x} ↔ {y}" for x, y in pairs[:3])
            raise PlanError(
                f"子任务 {a['id']} 与 {b['id']} 的 write_scope 重叠（{shown}），"
                "但两者之间没有依赖路径；请增加 depends_on 使其先后执行，或拆开写入范围。"
            )


def check_clause_coverage(
    specs: list[dict[str, Any]],
    clause_ids: list[str],
    known_nodes: list[dict[str, Any]] | None = None,
) -> None:
    valid = set(clause_ids)
    for s in specs:
        if not s.get("serves_clauses"):
            raise PlanError(
                f"子任务 {s['id']} 没有引用任何目标条款（serves_clauses 为空）；"
                f"可用条款：{', '.join(clause_ids)}。"
            )
        unknown = [c for c in s["serves_clauses"] if c not in valid]
        if unknown:
            raise PlanError(f"子任务 {s['id']} 引用了不存在的目标条款：{', '.join(unknown)}")
    covered = {c for s in specs for c in s.get("serves_clauses", [])}
    covered |= {c for n in (known_nodes or []) for c in n.get("serves_clauses", [])}
    missing = [c for c in clause_ids if c not in covered]
    if missing:
        raise PlanError(f"目标条款 {', '.join(missing)} 没有被任何子任务覆盖。")


# ---------------------------------------------------------------------------
# Goal traceability: frozen goal text + numbered clauses
# ---------------------------------------------------------------------------

_CLAUSE_SPLIT = re.compile(r"(?<=[。；;！!？?])|\n+")
_ENUM_PREFIX = re.compile(
    r"^\s*(?:[-*•]|\(?\d{1,2}[).、）]|[（(]\d{1,2}[)）]|第[一二三四五六七八九十]+[步点]?[，,：:]?)\s*"
)
# Conjunctions that often join two independent requirements inside one sentence.
_CONJ_SPLIT = re.compile(
    r"(?:，|,|;)?\s*(?:并且|以及|同时|另外|此外|还有|并要|并请)"
    r"|(?:，|,|;|\s)+(?:and\s+then|and\s+also|and|also|then)\s+",
    re.IGNORECASE,
)
_ZH_VERB = re.compile(
    r"(修复|修改|添加|补充|实现|删除|更新|重构|优化|检查|验证|分析|编写|创建|"
    r"安装|配置|迁移|测试|梳理|整理|说明|文档化)"
)
_EN_VERB = re.compile(
    r"\b(fix|add|implement|remove|update|refactor|optimize|verify|analyze|"
    r"write|create|install|configure|migrate|test|document)\b",
    re.IGNORECASE,
)
MAX_GOAL_CLAUSES = 12
_MULTI_LEN = 24
_MULTI_VERB_MIN = 2


def _strip_clause(text: str) -> str:
    text = _ENUM_PREFIX.sub("", (text or "").strip()).strip()
    text = re.sub(r"^(?:also|then|并且|以及|同时)\s+", "", text, flags=re.IGNORECASE)
    return text.rstrip("。；;！!？?").strip()


def _looks_multi(text: str) -> bool:
    verbs = len(_ZH_VERB.findall(text)) + len(_EN_VERB.findall(text))
    if verbs >= 3:
        return True
    if len(text) < _MULTI_LEN:
        return False
    return verbs >= _MULTI_VERB_MIN or bool(_CONJ_SPLIT.search(text))


def _heuristic_split_clause(text: str) -> list[str]:
    """Split one sentence on conjunctions when both sides look substantial."""
    if not _CONJ_SPLIT.search(text):
        return [text]
    parts = [p.strip() for p in _CONJ_SPLIT.split(text) if p and p.strip()]
    kept = [p for p in parts if len(p) >= 2]
    if len(kept) < 2:
        return [text]
    return kept


def _llm_split_clause(text: str, llm: Any) -> list[str] | None:
    """Ask the planner LLM to split one multi-requirement clause. Strict JSON."""
    prompt = (
        "把下面这一条用户目标拆成若干条互不重叠的原子要求。"
        "只输出 JSON：{\"clauses\":[\"...\",...]}，不要解释。"
        "每条至少两个字；不要添加原文没有的要求；若无法再拆就返回原文一条。\n\n"
        f"原文：{text}"
    )
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": "你是目标条款拆分器。只输出严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            [],
        )
        data = parse_plan_text(getattr(resp, "content", None))
        if not isinstance(data, dict):
            return None
        raw = data.get("clauses")
        if not isinstance(raw, list):
            return None
        out = [_strip_clause(str(x)) for x in raw if str(x).strip()]
        out = [x for x in out if len(x) >= 2]
        return out or None
    except Exception:  # noqa: BLE001 - advisory; fall back to heuristic
        return None


def split_goal_clauses(
    goal: str,
    max_clauses: int = MAX_GOAL_CLAUSES,
    *,
    llm: Any = None,
) -> list[dict[str, str]]:
    """Hybrid split of the goal into numbered clauses ``C1..Cn``.

    1. Deterministic split on sentence terminators（。；！？ / ;!?）and line breaks;
       strip list enumerators.
    2. Further split each piece on conjunctions（并且/以及/同时/and/also…）when
       both sides look like distinct requirements.
    3. Optional LLM assist (``config.GOAL_CLAUSE_LLM_SPLIT`` and a provided
       ``llm``) for pieces that still look multi-requirement after step 2.

    Always returns at least one clause; overflow beyond ``max_clauses`` is
    merged into the last one. The resulting set is frozen on the plan and must
    never change for that run.
    """
    seed: list[str] = []
    for piece in _CLAUSE_SPLIT.split(goal or ""):
        text = _strip_clause(piece)
        if len(text) >= 2:
            seed.append(text)
    if not seed:
        seed = [(goal or "").strip() or "完成用户目标"]

    parts: list[str] = []
    use_llm = (
        llm is not None
        and bool(getattr(config, "GOAL_CLAUSE_LLM_SPLIT", False))
        and bool(getattr(config, "PLANNER_ENABLED", True))
    )
    for text in seed:
        pieces = _heuristic_split_clause(text)
        for piece in pieces:
            if use_llm and _looks_multi(piece):
                llm_parts = _llm_split_clause(piece, llm)
                parts.extend(llm_parts if llm_parts else [piece])
            else:
                parts.append(piece)

    seen: set[str] = set()
    uniq: list[str] = []
    for t in parts:
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t[:1000])
    if not uniq:
        uniq = [(goal or "").strip() or "完成用户目标"]
    if len(uniq) > max_clauses:
        uniq = uniq[: max_clauses - 1] + ["；".join(uniq[max_clauses - 1:])]
    return [{"id": f"C{i}", "text": t} for i, t in enumerate(uniq, 1)]

def clause_ids_of(plan: dict[str, Any]) -> list[str] | None:
    clauses = plan.get("goal_clauses")
    if not clauses:
        return None
    return [c["id"] for c in clauses]


def goal_fingerprint(goal: str, clauses: list[dict[str, str]]) -> str:
    import hashlib

    payload = json.dumps({"goal": goal, "clauses": clauses}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_risk_marking(spec: dict[str, Any]) -> None:
    """Static check: irreversible steps must be ``risk=high``; steps whose text
    implies deletion / external side effects / config changes must not be
    ``risk=low``."""
    if not spec.get("reversible", True) and spec.get("risk") != "high":
        raise PlanError(
            f"子任务 {spec['id']} 标记为不可逆（reversible=false），必须设置 risk=high。"
        )
    text = f"{spec.get('title', '')} {spec.get('description', '')}"
    if spec.get("risk", "low") == "low" and _RISKY_TEXT.search(text):
        raise PlanError(
            f"子任务 {spec['id']} 涉及删除/外部副作用/配置变更，但风险等级为 low；"
            "请标记 risk=medium/high（不可逆时 reversible=false 且 risk=high）。"
        )


def needs_preexec_confirmation(subtask: dict[str, Any]) -> bool:
    """High-risk or irreversible subtasks pause BEFORE they start, even when
    the plan was approved."""
    return (not subtask.get("reversible", True)) or subtask.get("risk") == "high"


def topological_order(
    items: list[dict[str, Any]], known: set[str] | None = None
) -> list[dict[str, Any]]:
    """Stable Kahn ordering (keeps given order among ready items)."""
    satisfied = set(known or set())
    remaining = list(items)
    ordered: list[dict[str, Any]] = []
    while remaining:
        progress = False
        for item in list(remaining):
            if all(d in satisfied for d in item.get("depends_on", [])):
                ordered.append(item)
                satisfied.add(item["id"])
                remaining.remove(item)
                progress = True
        if not progress:
            raise PlanError(
                "子任务依赖存在环：" + ", ".join(i["id"] for i in remaining)
            )
    return ordered


def _runtime_fields() -> dict[str, Any]:
    return {
        "status": SUBTASK_PENDING,
        "attempt_number": 0,
        "result_summary": "",
        "error": None,
        "stop_reason": "",
        "acceptance_evidence": [],
        "manual": False,
        "confirmed": False,
        "interventions": [],
        "attempts_log": [],
        "duration_seconds": 0.0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0, "estimated": False},
        "started_at": None,
        "finished_at": None,
        "finished_seq": None,
    }


def with_runtime(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**spec, **_runtime_fields()} for spec in specs]


def new_plan_record(
    specs: list[dict[str, Any]],
    source: str,
    *,
    requires_approval: bool,
    version: int = 1,
    notes: list[str] | None = None,
    goal: str | None = None,
    goal_clauses: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    clauses = goal_clauses if goal_clauses is not None else (
        split_goal_clauses(goal) if goal is not None else None)
    record = {
        "version": int(version),
        "revision": 1,
        "parent_version": None,
        "parent_revision": None,
        "source": source,
        "approval_status": PLAN_PENDING if requires_approval else PLAN_APPROVED,
        "approved_version": None if requires_approval else int(version),
        "rewrites": 0,
        "structural_replans": 0,
        "final_verify_replans": 0,
        "plan_rejections": 0,
        "feedback": [],
        "cancel_requested": False,
        "attention": None,
        "totals": {"steps": 0, "active_seconds": 0.0, "tokens": 0},
        "current_subtask_id": None,
        "subtasks": with_runtime(specs),
        "history": [],
        "notes": list(notes or []),
        "pending_failures": [],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    if goal is not None:
        # Frozen at creation and never regenerated: every later version (replan,
        # rejection, manual edit) keeps exactly these clauses.
        record["goal"] = goal
        record["goal_clauses"] = clauses
        record["goal_fingerprint"] = goal_fingerprint(goal, clauses or [])
    return record


def single_subtask_spec(goal: str, clause_ids: list[str] | None = None) -> list[dict[str, Any]]:
    title = goal.strip().splitlines()[0][:80] if goal.strip() else "完成任务"
    return [{
        "id": "1",
        "title": f"执行：{title}",
        "description": goal[:4000],
        "depends_on": [],
        "kind": "modify",
        "acceptance_criteria": [
            "满足用户目标，并通过执行器成功判定（真实工具证据，非模型自述）",
        ],
        "risk": "low",
        "reversible": True,
        # Single-subtask mode keeps v0.2 behaviour: the whole workspace.
        "write_scope": ["**"],
        "serves_clauses": list(clause_ids or ["C1"]),
        "derived_from": [],
    }]


def is_legacy(plan: dict[str, Any]) -> bool:
    return plan.get("source") in LEGACY_SOURCES and len(plan.get("subtasks", [])) == 1


def plan_titles(plan: dict[str, Any]) -> list[str]:
    return [f"[{s['id']}] {s['title']}" for s in plan.get("subtasks", [])]


def describe_plan(plan: dict[str, Any]) -> str:
    lines = [(
        f"任务图 v{plan.get('version', 1)}（来源：{plan.get('source')}，"
        f"共 {len(plan.get('subtasks', []))} 个子任务）："
    )]
    for s in plan.get("subtasks", []):
        deps = f"，依赖 {','.join(s['depends_on'])}" if s.get("depends_on") else ""
        lines.append(
            f"[{s['id']}] {KIND_LABELS.get(s['kind'], s['kind'])}：{s['title']}{deps}"
        )
    return "\n".join(lines)


def done_ids(plan: dict[str, Any]) -> set[str]:
    """Ids that satisfy dependencies: only *succeeded* subtasks."""
    return {s["id"] for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE}


def migrate_legacy_write_scopes(
    plan: dict[str, Any],
    *,
    strict: bool | None = None,
) -> list[str]:
    """Fill missing ``write_scope`` keys on legacy subtasks.

    * analyze / verify → ``[]`` (read-only)
    * modify → blocked (``needs_write_scope=True``) when
      ``LEGACY_WRITE_SCOPE_STRICT`` (default), else ``['**']`` with a note

    Returns human-readable migration notes. Idempotent for already-scoped nodes.
    """
    from app import config as _cfg
    use_strict = _cfg.LEGACY_WRITE_SCOPE_STRICT if strict is None else strict
    notes: list[str] = []
    at = _utc_now()
    for s in plan.get("subtasks") or []:
        if "write_scope" in s:
            # Clear stale block flag once scoped.
            if s.pop("needs_write_scope", None):
                notes.append(f"子任务 {s['id']} 已具备 write_scope，解除启动阻塞。")
            continue
        kind = (s.get("kind") or "analyze").lower()
        if kind in {"analyze", "verify"}:
            s["write_scope"] = []
            note = f"旧计划迁移：子任务 {s['id']}（{kind}）→ write_scope=[]（只读）"
            notes.append(note)
            ev = list(s.get("acceptance_evidence") or [])
            ev.append({"type": "legacy_scope_migration", "message": note, "at": at,
                       "write_scope": []})
            s["acceptance_evidence"] = ev
            s.pop("needs_write_scope", None)
        else:
            # modify (or unknown mutating kinds)
            if use_strict:
                s["needs_write_scope"] = True
                note = (
                    f"旧计划子任务 {s['id']}（{kind}）缺少 write_scope；"
                    "LEGACY_WRITE_SCOPE_STRICT=true，请 PATCH /runs/.../subtasks/... "
                    "设置 write_scope 后再继续。"
                )
                notes.append(note)
            else:
                s["write_scope"] = ["**"]
                s.pop("needs_write_scope", None)
                note = (
                    f"旧计划迁移：子任务 {s['id']}（{kind}）→ write_scope=['**']"
                    "（LEGACY_WRITE_SCOPE_STRICT=false）"
                )
                notes.append(note)
                ev = list(s.get("acceptance_evidence") or [])
                ev.append({"type": "legacy_scope_migration", "message": note, "at": at,
                           "write_scope": ["**"]})
                s["acceptance_evidence"] = ev
    if notes:
        plan_notes = list(plan.get("notes") or [])
        for n in notes:
            if n not in plan_notes:
                plan_notes.append(n)
        plan["notes"] = plan_notes[-30:]
        plan["updated_at"] = at
    return notes


def unscoped_modify_ids(plan: dict[str, Any]) -> list[str]:
    """Subtask ids that still need a write_scope before they may start."""
    return [
        s["id"] for s in plan.get("subtasks") or []
        if s.get("needs_write_scope")
        or (
            "write_scope" not in s
            and (s.get("kind") or "").lower() not in {"analyze", "verify"}
            and s.get("status") in {SUBTASK_PENDING, SUBTASK_RUNNING, SUBTASK_FAILED}
        )
    ]



def ready_subtasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """All pending subtasks whose dependencies succeeded (DAG ready set).

    Independent ready subtasks MAY run in parallel; v1 of the control loop
    still picks one at a time, but the data model is a real DAG.
    """
    finished = done_ids(plan)
    ready: list[dict[str, Any]] = []
    for s in plan.get("subtasks", []):
        if s.get("status") != SUBTASK_PENDING:
            continue
        if all(d in finished for d in s.get("depends_on", [])):
            ready.append(s)
    return ready


def descendants_of(plan: dict[str, Any], subtask_id: str) -> set[str]:
    children: dict[str, set[str]] = {}
    for s in plan.get("subtasks", []):
        for d in s.get("depends_on", []):
            children.setdefault(d, set()).add(s["id"])
    out: set[str] = set()
    stack = [subtask_id]
    while stack:
        for c in children.get(stack.pop(), set()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def launchable_subtasks(
    plan: dict[str, Any],
    in_flight: set[str],
    capacity: int,
    *,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Subtasks the parallel scheduler may start now.

    Candidates are ``running`` subtasks not in flight in this process (crash
    recovery / restart) followed by ready ``pending`` ones, in topological
    order. A candidate is taken only if its write_scope overlaps neither a
    subtask in flight nor one already chosen in this batch; overlapping ones
    queue until the conflicting subtask finishes.
    """
    if capacity <= 0:
        return []
    by_id = {s["id"]: s for s in plan.get("subtasks", [])}

    def scope_of(node: dict[str, Any]) -> list[str]:
        # After migrate_legacy_write_scopes, every node has the key. A still-
        # missing key (strict-blocked modify) conflicts with everything so it
        # never runs in parallel with others.
        if "write_scope" in node:
            return list(node.get("write_scope") or [])
        return ["**"]

    busy_scopes = [scope_of(by_id[i]) for i in in_flight if i in by_id]
    skip = set(exclude or set()) | set(in_flight)
    candidates = [
        s for s in plan.get("subtasks", [])
        if s.get("status") == SUBTASK_RUNNING and s["id"] not in skip
    ] + [s for s in ready_subtasks(plan) if s["id"] not in skip]
    chosen: list[dict[str, Any]] = []
    for s in candidates:
        if len(chosen) >= capacity:
            break
        if s.get("needs_write_scope"):
            continue  # strict legacy modify — wait for PATCH
        scope = scope_of(s)
        if any(scopes_overlap(scope, other) for other in busy_scopes):
            continue
        chosen.append(s)
        busy_scopes.append(scope)
    return chosen


def next_ready_subtask(plan: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Pick next subtask for the sequential v1 scheduler.

    Prefer a ``running`` subtask (resume under the same lease), else the first
    ready pending node (topological list order). Returns
    ``(subtask, "resume"|"ready"|"complete"|"blocked")``.
    """
    for s in plan.get("subtasks", []):
        if s.get("status") == SUBTASK_RUNNING:
            return s, "resume"
    ready = ready_subtasks(plan)
    if ready:
        return ready[0], "ready"
    pending = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_PENDING]
    if not pending:
        return None, "complete"
    return None, "blocked"


def find_subtask(plan: dict[str, Any], subtask_id: str | None) -> dict[str, Any] | None:
    for s in plan.get("subtasks", []):
        if s["id"] == subtask_id:
            return s
    return None


def compose_subtask_goal(goal: str, plan: dict[str, Any], subtask: dict[str, Any]) -> str:
    if is_legacy(plan):
        return goal
    done = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE]
    lines = [
        f"总体目标：{goal}",
        "",
        (f"当前子任务 [{subtask['id']}]（类型："
         f"{KIND_LABELS.get(subtask['kind'], subtask['kind'])}）：{subtask['title']}"),
    ]
    if subtask.get("description"):
        lines.append(f"子任务说明：{subtask['description']}")
    if subtask.get("acceptance_criteria"):
        lines.append("验收标准：")
        lines.extend(f"- {c}" for c in subtask["acceptance_criteria"])
    clauses = {c["id"]: c["text"] for c in plan.get("goal_clauses") or []}
    served = [f"{c}：{clauses[c]}" for c in subtask.get("serves_clauses", []) if c in clauses]
    if served:
        lines.append("本子任务服务的目标条款：")
        lines.extend(f"- {c}" for c in served)
    scope = subtask.get("write_scope") or []
    lines.append(
        "允许写入的范围（write_scope）：" + ("、".join(scope) if scope else "无（只读子任务）")
        + "。范围之外的写入会被拒绝。"
    )
    if done:
        lines.append("")
        lines.append("已完成的子任务（结论摘要）：")
        for s in done:
            lines.append(
                f"- [{s['id']}] {s['title']}：{(s.get('result_summary') or '')[:300]}"
            )
    lines.append("")
    rules = {
        "analyze": "本子任务为只读分析：只读取/搜索/检查，不要修改任何文件。",
        "modify": "本子任务需要做出最小、可验证的修改，修改后运行相关测试/构建验证。",
        "verify": "本子任务为验证：运行相关测试/构建/检查，失败时可做最小修复后重新验证。",
    }
    lines.append(rules.get(subtask["kind"], ""))
    lines.append(
        "只完成当前子任务，不要提前做后续子任务。"
        "完成后给出总结，并逐条说明验收标准是否满足（以工具证据为准）。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Acceptance evidence (never trust executor prose)
# ---------------------------------------------------------------------------

def extract_acceptance_evidence(outcome: Any) -> list[dict[str, Any]]:
    """Build evidence list from an ``AgentOutcome`` / checkpoint.

    Evidence comes from real tool results recorded in the checkpoint
    (test_results, modified_files with hashes) — not from ``summary`` text.
    """
    evidence: list[dict[str, Any]] = []
    checkpoint = getattr(outcome, "checkpoint", None) or {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    for item in checkpoint.get("test_results") or []:
        if not isinstance(item, dict):
            continue
        evidence.append({
            "type": "command",
            "program": item.get("program"),
            "args": item.get("args", []),
            "ok": bool(item.get("ok")),
            "summary": str(item.get("summary", ""))[:300],
        })
    for item in checkpoint.get("modified_files") or []:
        if isinstance(item, dict):
            evidence.append({
                "type": "file",
                "path": item.get("path"),
                "sha256": item.get("sha256") or "",
            })
        elif isinstance(item, str):
            evidence.append({"type": "file", "path": item, "sha256": ""})
    status = getattr(outcome, "status", None)
    stop = getattr(outcome, "stop_reason", None) or checkpoint.get("stop_reason")
    evidence.append({
        "type": "executor_status",
        "status": status,
        "stop_reason": stop,
        "tool_calls": getattr(outcome, "tool_calls", 0),
        "tool_successes": getattr(outcome, "tool_successes", 0),
        "tool_failures": getattr(outcome, "tool_failures", 0),
    })
    return evidence


def manual_evidence(
    note: str, evidence_text: str, status: str, files: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evidence items for a manually-marked subtask result.

    The manual note and any *declared file changes* are kept as separate items
    so overall acceptance can weigh them independently: a note alone must never
    satisfy a "code must change" requirement, while a declared change is counted
    as real file evidence (without a hash — the user asserts it on disk).
    """
    at = _utc_now()
    items: list[dict[str, Any]] = []
    for path in (files or []):
        path = str(path).strip().replace("\\", "/")
        if path:
            items.append({"type": "file", "path": path, "sha256": "",
                          "manual": True, "at": at})
    items.append({
        "type": "manual",
        "by": "user",
        "status": status,
        "note": (note or "")[:2000],
        "evidence_text": (evidence_text or "")[:8000],
        "at": at,
    })
    return items


def next_finish_seq(plan: dict[str, Any]) -> int:
    """Monotonic completion ordinal for the whole run (survives resume/restart).

    Final verification must order evidence by *actual* execution/completion,
    not by the static ``plan["subtasks"]`` list order (parallel subtasks finish
    in scheduler order). A wall-clock timestamp alone can collide or skew across
    workers, so each completed subtask is stamped with an integer ordinal that
    is strictly increasing in completion order.

    Derived from the persisted plan (``max(existing)+1``) rather than an
    in-process counter, so the sequence stays monotonic even when the control
    loop restarts mid-run (``finished_at`` remains as a secondary tie-break).
    """
    subs = plan.get("subtasks") or []
    return max((int(s.get("finished_seq") or 0) for s in subs), default=0) + 1


def contract_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Field-level diff of two subtask contracts (spec fields only)."""
    fields = ("title",) + STRUCTURAL_FIELDS
    return {
        f: {"before": before.get(f), "after": after.get(f)}
        for f in fields
        if before.get(f) != after.get(f)
    }


def is_structural_diff(diff: dict[str, Any]) -> bool:
    return any(f in STRUCTURAL_FIELDS for f in diff)


def graph_diff(
    old_subtasks: list[dict[str, Any]], new_specs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diff of the remaining graph: added / removed / changed contracts."""
    old = {s["id"]: s for s in old_subtasks}
    new = {s["id"]: s for s in new_specs}
    changed = {
        sid: contract_diff(old[sid], new[sid])
        for sid in old.keys() & new.keys()
        if contract_diff(old[sid], new[sid])
    }
    return {
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "changed": changed,
    }


def is_structural_graph_diff(diff: dict[str, Any]) -> bool:
    if diff.get("added") or diff.get("removed"):
        return True
    return any(is_structural_diff(d) for d in diff.get("changed", {}).values())


# ---------------------------------------------------------------------------
# Append-only plan version history + lineage
# ---------------------------------------------------------------------------

def spec_of(subtask: dict[str, Any]) -> dict[str, Any]:
    return {k: subtask.get(k) for k in SPEC_FIELDS}


def assign_lineage(
    parent_subtasks: list[dict[str, Any]] | None,
    subtasks: list[dict[str, Any]],
    *,
    by_id: bool = True,
) -> None:
    """Fill ``derived_from``: keep planner-provided ids that exist in the
    parent; otherwise (when ``by_id``) a node with the same id derives from it."""
    parent_ids = {s["id"] for s in parent_subtasks or []}
    for s in subtasks:
        given = [d for d in s.get("derived_from") or [] if d in parent_ids]
        if not given and by_id and s["id"] in parent_ids:
            given = [s["id"]]
        s["derived_from"] = given


def version_entry(
    plan: dict[str, Any],
    parent: dict[str, Any] | None,
    *,
    reason: str,
    failure_class: str | None = None,
) -> dict[str, Any]:
    """Immutable snapshot of one plan (version, revision) for the history."""
    subtasks = [
        {**spec_of(s), "status": s.get("status"), "manual": bool(s.get("manual"))}
        for s in plan.get("subtasks", [])
    ]
    diff = graph_diff(parent.get("subtasks", []), plan.get("subtasks", [])) if parent else None
    return {
        "version": int(plan.get("version") or 1),
        "revision": int(plan.get("revision") or 1),
        "parent_version": parent.get("version") if parent else None,
        "parent_revision": parent.get("revision") if parent else None,
        "reason": reason,
        "failure_class": failure_class,
        "created_at": _utc_now(),
        "plan": {
            "goal": plan.get("goal"),
            "goal_clauses": plan.get("goal_clauses"),
            "source": plan.get("source"),
            "subtasks": subtasks,
        },
        "diff": diff,
    }


def bump_revision(plan: dict[str, Any], parent: dict[str, Any]) -> None:
    plan["parent_version"] = parent.get("version")
    plan["parent_revision"] = parent.get("revision")
    plan["revision"] = int(parent.get("revision") or 1) + 1


def bump_version(plan: dict[str, Any], parent: dict[str, Any]) -> None:
    plan["parent_version"] = parent.get("version")
    plan["parent_revision"] = parent.get("revision")
    plan["version"] = int(parent.get("version") or 1) + 1
    plan["revision"] = 1


# ---------------------------------------------------------------------------
# Final verify: per-clause traceability report
# ---------------------------------------------------------------------------

def _evidence_brief(e: dict[str, Any]) -> str | None:
    t = e.get("type")
    if t == "command":
        return f"{'✔' if e.get('ok') else '✘'} {e.get('program')} {' '.join(e.get('args') or [])}".strip()
    if t == "file":
        sha = (e.get("sha256") or "")[:12]
        return f"文件 {e.get('path')}" + (f"（{sha}…）" if sha else "")
    if t == "manual":
        return f"人工：{(e.get('note') or '')[:120]}"
    return None


def subtask_is_verified(subtask: dict[str, Any]) -> bool:
    """A done subtask counts as *verified* only when it carries evidence that
    substantiates its own acceptance rule. A bare ``done`` flag — e.g. a manual
    marker on a modify/verify subtask with no file/check evidence — is a claim,
    not proof, so it does not count as implemented.
    """
    if subtask.get("status") != SUBTASK_DONE:
        return False
    kind = subtask.get("kind") or "analyze"
    ev = subtask.get("acceptance_evidence") or []
    files = [e for e in ev if e.get("type") == "file"]
    commands = [e for e in ev if e.get("type") == "command"]
    passed = [c for c in commands if c.get("ok")]
    manual = [e for e in ev if e.get("type") == "manual"]
    if kind == "modify":
        # A modification is substantiated by actual file-change evidence only.
        return bool(files)
    if kind == "verify":
        # A verification is substantiated by a passing command/check or a manual
        # verification note.
        return bool(passed) or bool(manual)
    # analyze: a real read/check happened (tool calls) or the user confirmed it.
    calls = next((e.get("tool_calls", 0) for e in ev if e.get("type") == "executor_status"), 0)
    return calls > 0 or bool(manual)


def clause_report(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Per goal clause: which subtasks *plan* to cover it vs which have actually
    been *verified* (done + substantive evidence). ``status`` is ``covered``
    (verified), ``planned`` (declared but not yet backed by evidence) or
    ``missing`` (no subtask declares it)."""
    report = []
    for clause in plan.get("goal_clauses") or []:
        serving = [
            s for s in plan.get("subtasks", [])
            if clause["id"] in (s.get("serves_clauses") or [])
        ]
        verified = [s for s in serving if subtask_is_verified(s)]
        evidence: list[str] = []
        for s in verified:
            briefs = [b for b in (_evidence_brief(e) for e in s.get("acceptance_evidence", [])) if b]
            if not briefs:
                calls = next((e.get("tool_calls") for e in s.get("acceptance_evidence", [])
                              if e.get("type") == "executor_status"), 0)
                briefs = [f"只读执行：工具调用 {calls} 次"]
            evidence.extend(f"[{s['id']}] {b}" for b in briefs[:5])
        status = "covered" if verified else ("missing" if not serving else "planned")
        report.append({
            "id": clause["id"],
            "text": clause["text"],
            "status": status,
            "covered_by": [s["id"] for s in verified],
            "planned_by": [s["id"] for s in serving],
            "manual": [s["id"] for s in verified if s.get("manual")],
            "evidence": evidence,
        })
    return report


def accept_subtask(
    subtask: dict[str, Any],
    outcome: Any,
    *,
    evidence: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Decide accept/reject from evidence + existing executor success rules.

    Subtask success ≠ goal success. Final whole-goal verification is a
    separate control-loop exit, not a normal subtask.
    """
    evidence = evidence if evidence is not None else extract_acceptance_evidence(outcome)
    status = getattr(outcome, "status", None)
    stop = getattr(outcome, "stop_reason", "") or ""
    if status != "succeeded":
        return False, f"执行器未成功（status={status}, stop_reason={stop}）"

    kind = subtask.get("kind") or "analyze"
    tool_calls = getattr(outcome, "tool_calls", 0) or 0
    if tool_calls <= 0:
        return False, "没有任何工具调用证据，不能认定子任务完成。"

    files = [e for e in evidence if e.get("type") == "file"]
    commands = [e for e in evidence if e.get("type") == "command"]
    passed_cmds = [c for c in commands if c.get("ok")]
    failed_cmds = [c for c in commands if not c.get("ok")]

    if kind == "analyze":
        # Read-only: executor success + at least one tool is enough.
        return True, "只读分析子任务：工具调用成功，已记录证据。"

    if kind == "modify":
        if not files:
            return False, "修改类子任务缺少文件变更证据（路径/哈希）。"
        if failed_cmds and not passed_cmds:
            return False, "修改后验证命令失败，且无后续通过的验证。"
        return True, "修改类子任务：存在文件变更证据，执行器判定成功。"

    if kind == "verify":
        # Verification must be substantiated by at least one passing command /
        # check (pytest / build / check_http / check_port). Reading or searching
        # code is preparation, not proof, so a bare tool call never passes.
        if not passed_cmds:
            return False, "验证类子任务缺少通过的命令/检查证据（测试、构建或端口/HTTP 检查均未通过）。"
        return True, "验证类子任务：存在通过的验证证据。"

    return True, "已按执行器成功规则接受。"


def classify_failure(subtask: dict[str, Any], outcome: Any) -> str:
    """Map a rejected subtask to a control-loop failure branch.

    * ``transient`` — timeout / model-network error / crash / lock conflict and
      attempts remain: retry the SAME subtask (attempt+1), plan unchanged.
    * ``structural`` — forced: the executor was blocked (no tools at all), the
      workspace was rejected, or the attempt budget is exhausted.
    * ``rewrite_remaining`` — otherwise (wrong approach). The rewrite proposal
      is still diffed: if it changes steps/deps/criteria it is escalated to
      ``structural`` by the control loop.
    """
    stop = getattr(outcome, "stop_reason", None) or ""
    error = getattr(outcome, "error", None) or ""
    attempt = int(subtask.get("attempt_number") or 0)
    exhausted = attempt >= config.MAX_SUBTASK_ATTEMPTS
    transient = stop in _TRANSIENT_STOP or any(m in error for m in _TRANSIENT_ERROR_MARKERS)
    if transient and not exhausted:
        return FAIL_TRANSIENT
    if stop in {"blocked", "workspace_rejected"} or exhausted:
        return FAIL_STRUCTURAL
    return FAIL_REWRITE


# ---------------------------------------------------------------------------
# Rewrite remaining / structural bump
# ---------------------------------------------------------------------------

def snapshot_history(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": plan.get("version", 1),
        "source": plan.get("source"),
        "done_ids": sorted(done_ids(plan)),
        "snapshot_at": _utc_now(),
    }


def apply_rewrite_remaining(
    plan: dict[str, Any],
    new_specs: list[dict[str, Any]],
    *,
    failed_id: str,
) -> dict[str, Any]:
    """Replace failed + pending subtasks; keep done ones and plan_version."""
    kept = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE]
    old_by_id = {s["id"]: s for s in plan.get("subtasks", [])}
    remaining = []
    for spec in new_specs:
        node = {**spec, **_runtime_fields()}
        prev = old_by_id.get(spec["id"])
        if prev is not None:
            # Same node, new approach: keep attempt history / interventions.
            for key in ("attempt_number", "attempts_log", "interventions",
                        "duration_seconds", "tokens", "acceptance_evidence", "confirmed"):
                node[key] = prev.get(key, node[key])
        remaining.append(node)
    parent = plan
    assign_lineage(parent.get("subtasks"), remaining)
    plan = dict(plan)
    history = list(plan.get("history") or [])
    history.append(snapshot_history(plan))
    plan["history"] = history[-10:]
    plan["subtasks"] = kept + remaining
    plan["source"] = "replan"
    bump_revision(plan, parent)
    plan["rewrites"] = int(plan.get("rewrites") or 0) + 1
    plan["current_subtask_id"] = None
    plan["updated_at"] = _utc_now()
    _ = failed_id  # the failed node is replaced (or kept by id) by new_specs
    return plan


def apply_structural_replan(
    plan: dict[str, Any],
    new_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bump plan_version, keep done subtasks, mark approval pending."""
    kept = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE]
    parent = plan
    fresh = with_runtime(new_specs)
    assign_lineage(parent.get("subtasks"), fresh)
    plan = dict(plan)
    history = list(plan.get("history") or [])
    history.append(snapshot_history(plan))
    plan["history"] = history[-10:]
    bump_version(plan, parent)
    plan["subtasks"] = kept + fresh
    plan["source"] = "replan"
    plan["approval_status"] = PLAN_PENDING if config.PLAN_REQUIRES_APPROVAL else PLAN_APPROVED
    plan["approved_version"] = None if config.PLAN_REQUIRES_APPROVAL else plan["version"]
    plan["structural_replans"] = int(plan.get("structural_replans") or 0) + 1
    plan["current_subtask_id"] = None
    plan["updated_at"] = _utc_now()
    return plan


def reset_subtask_for_retry(subtask: dict[str, Any]) -> dict[str, Any]:
    """Transient failure: same node, bump attempt, clear running fields."""
    s = dict(subtask)
    s["status"] = SUBTASK_PENDING
    s["attempt_number"] = int(s.get("attempt_number") or 0)  # bumped when leased
    s["error"] = None
    s["stop_reason"] = ""
    s["result_summary"] = ""
    s["finished_at"] = None
    s["finished_seq"] = None
    # Keep acceptance_evidence from the failed attempt for diagnostics.
    return s


# ---------------------------------------------------------------------------
# Execution lease (prevents two runners stealing the same step)
# ---------------------------------------------------------------------------

def new_lease(
    run_id: str, subtask_id: str, attempt_number: int, holder: str,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "lease_id": uuid4().hex,
        "run_id": run_id,
        "subtask_id": subtask_id,
        "attempt_number": int(attempt_number),
        "holder": holder,
        "state": "running",          # running | waiting (approval/input) | released
        "acquired_at": now,
        "heartbeat_at": now,
    }


def lease_age_seconds(lease: dict[str, Any] | None, now: datetime | None = None) -> float:
    if not isinstance(lease, dict) or not lease.get("heartbeat_at"):
        return float("inf")
    try:
        beat = datetime.fromisoformat(str(lease["heartbeat_at"]))
    except ValueError:
        return float("inf")
    return ((now or datetime.now(UTC)) - beat).total_seconds()


def lease_is_live(lease: dict[str, Any] | None, ttl: float | None = None) -> bool:
    """A ``running`` lease whose heartbeat is younger than the TTL is held by a
    live executor; an older one is a crashed attempt."""
    if not isinstance(lease, dict) or lease.get("state") != "running":
        return False
    limit = ttl if ttl is not None else config.LEASE_TTL_SECONDS
    return lease_age_seconds(lease) < limit


def lease_matches(lease: dict[str, Any] | None, subtask_id: str, attempt: int) -> bool:
    if not isinstance(lease, dict):
        return False
    return (
        lease.get("subtask_id") == subtask_id
        and int(lease.get("attempt_number") or 0) == int(attempt)
    )


# ---------------------------------------------------------------------------
# Project overview + LLM planning
# ---------------------------------------------------------------------------

_OVERVIEW_SKIP = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "dist", "build", ".idea", ".vscode", "data",
}
_OVERVIEW_FILES = ("README.md", "README.rst", "README.txt", "pyproject.toml", "package.json")


def build_project_overview(workspace: str | Path | None, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else config.PLANNER_OVERVIEW_MAX_CHARS
    try:
        root = resolve_workspace_root(
            workspace if workspace is not None else config.WORKSPACE_PATH
        )
    except SecurityError:
        return ""
    lines: list[str] = [f"工作区：{root}", "目录结构（最多两层）："]
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        entries = []
    for entry in entries[:60]:
        if entry.name in _OVERVIEW_SKIP:
            continue
        if entry.is_dir():
            lines.append(f"  {entry.name}/")
            try:
                children = sorted(
                    entry.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
                )
            except OSError:
                children = []
            shown = [c for c in children if c.name not in _OVERVIEW_SKIP][:15]
            for child in shown:
                lines.append(f"    {child.name}{'/' if child.is_dir() else ''}")
            if len(children) > len(shown):
                lines.append("    …")
        else:
            lines.append(f"  {entry.name}")
    for name in _OVERVIEW_FILES:
        path = root / name
        if not path.is_file() or is_protected_file(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:1200]
        except OSError:
            continue
        lines.append(f"\n--- {name}（开头）---\n{text}")
    overview = "\n".join(lines)
    if len(overview) > limit:
        overview = overview[:limit] + "\n…（已截断）"
    return overview


_SCHEMA_HINT = """输出格式（只输出一个 JSON 对象，不要输出其他文字）：
{"subtasks": [
  {"id": "1", "title": "简短标题", "description": "要做什么、涉及哪些文件",
   "depends_on": [], "kind": "analyze|modify|verify",
   "risk": "low|medium|high", "reversible": true,
   "write_scope": ["允许修改的相对路径或 glob，例如 app/parser.py、tests/test_*.py"],
   "serves_clauses": ["C1"],
   "acceptance_criteria": ["可检验的完成标准"]}
]}
说明：depends_on 构成 DAG；彼此独立的子任务可以并行执行。
write_scope：modify 子任务必须列出允许写入的文件/目录；analyze/verify 通常为空（只读）。
两个子任务的 write_scope 若有重叠，必须用 depends_on 让其中一个依赖另一个。
serves_clauses：每个子任务至少服务一个目标条款，且所有条款都必须被覆盖。
整体目标的最终验收由控制环单独执行，不要把「最终验收」做成普通子任务。"""

PLANNER_SYSTEM_PROMPT = """你是一个代码任务规划器。把用户目标拆解为少量、按依赖排序、可独立验证的子任务（任务图 / DAG）。
规则：
- kind 只能是 analyze（只读分析）、modify（修改代码）、verify（运行测试/构建验证）。
- 修改类子任务之后应安排 verify 子任务（或在验收标准中要求运行具体测试）。
- 只读的分析/问答类目标只需要 analyze 子任务，不要安排修改。
- depends_on 只能引用前面已定义的子任务 id；不要形成环。独立子任务可互不依赖。
- 子任务要具体（指明文件/模块），验收标准要可检验；子任务数量尽量少。
- 每个子任务必须标注 risk（low/medium/high）与 reversible（是否可撤销）：
  删除文件/数据、外部副作用（部署、推送、发送请求）、修改配置等至少为 medium；
  不可逆（reversible=false）必须为 high。高风险子任务执行前会单独请用户确认。
- 不要输出「最终整体验收」子任务——那是控制环的固定出口。
- 每个子任务声明 write_scope（允许写入的路径/glob，只读子任务为空）与 serves_clauses
  （它服务的目标条款编号）；写入范围之外的修改会被系统拒绝。
- 不要输出任何密钥或敏感信息。"""


def parse_plan_text(text: str | None) -> Any:
    if not text or not text.strip():
        raise PlanError("模型没有返回计划内容。")
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(body[start:end + 1])
        except json.JSONDecodeError as exc:
            raise PlanError(f"计划不是合法 JSON：{exc}") from exc
    raise PlanError("计划不是合法 JSON。")


def _clause_text(clauses: list[dict[str, str]] | None) -> str:
    if not clauses:
        return ""
    return "目标条款（编号固定，不可修改）：\n" + "\n".join(
        f"- {c['id']}：{c['text']}" for c in clauses) + "\n\n"


def build_planner_messages(
    goal: str, overview: str, max_subtasks: int,
    clauses: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    user = (
        f"用户目标：\n{goal}\n\n{_clause_text(clauses)}"
        f"项目概览：\n{overview or '（无法读取项目概览）'}\n\n"
        f"请拆解为不超过 {max_subtasks} 个子任务。\n{_SCHEMA_HINT}"
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def classify_final_verify_failure(
    missing: list[str],
    inconsistent: list[str],
    notes: list[str] | None = None,
) -> str:
    """Return ``structural`` | ``rewrite_remaining`` | ``fail``.

    Missing goal-clause coverage → structural (need new/adjusted subtasks).
    Soft inconsistencies that look like approach gaps → rewrite_remaining.
    Transient / flaky evidence → fail (do not thrash).
    """
    notes = notes or []
    blob = " ".join(missing + inconsistent + notes).lower()
    transient_markers = (
        "timeout", "超时", "429", "rate limit", "503", "502", "临时", "transient",
        "connection reset", "连接重置", "不可用，仅使用规则",
    )
    if any(m in blob for m in transient_markers) and not missing:
        return "fail"
    if missing:
        return FAIL_STRUCTURAL
    if inconsistent:
        # Cross-subtask file digest / verify-order issues: usually not fixed by
        # inventing new steps; fail loudly unless clearly a coverage gap.
        coverageish = any(
            k in " ".join(inconsistent)
            for k in ("条款", "覆盖", "未完成", "遗漏", "missing", "未达成")
        )
        return FAIL_REWRITE if coverageish else "fail"
    return "fail"


def build_final_verify_replan_messages(
    goal: str,
    plan: dict[str, Any],
    *,
    missing: list[str],
    inconsistent: list[str],
    max_subtasks: int,
    mode: str = FAIL_STRUCTURAL,
) -> list[dict[str, Any]]:
    """Planner prompt after final verify failed (plan gaps / missing clauses)."""
    done = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE]

    def brief(s: dict[str, Any]) -> dict[str, Any]:
        return {
            k: s.get(k)
            for k in ("id", "title", "kind", "depends_on", "acceptance_criteria",
                      "write_scope", "serves_clauses", "result_summary")
        }

    payload = {
        "plan_version": plan.get("version", 1),
        "mode": mode,
        "trigger": "final_verify",
        "missing_clauses": missing[:20],
        "inconsistencies": inconsistent[:20],
        "done": [brief(s) for s in done],
        "goal_clauses": plan.get("goal_clauses") or [],
    }
    if mode == FAIL_STRUCTURAL:
        instruction = (
            "最终验收发现原始目标仍有缺口。请给出**新增或替换**的后续子任务"
            f"（不超过 {max_subtasks} 个），专门覆盖 missing_clauses / 修复 inconsistencies。"
            "已完成的子任务会保留；新 id 不能与已完成子任务重复。"
            "整体最终验收仍由控制环执行，不要做成普通子任务。"
        )
    else:
        instruction = (
            "最终验收发现覆盖不足。请在不改变计划版本语义的前提下，重写后续子任务"
            f"（不超过 {max_subtasks} 个）以补齐缺失条款。已完成子任务会保留。"
        )
    user = (
        f"用户目标：\n{goal}\n\n{_clause_text(plan.get('goal_clauses'))}"
        "新任务图必须沿用上面的条款编号。\n\n"
        "最终验收失败详情（JSON）：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}\n\n"
        f"{instruction}\n{_SCHEMA_HINT}"
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_replan_messages(
    goal: str,
    plan: dict[str, Any],
    failed: dict[str, Any],
    error: str,
    max_subtasks: int,
    *,
    mode: str = FAIL_REWRITE,
) -> list[dict[str, Any]]:
    done = [s for s in plan.get("subtasks", []) if s.get("status") == SUBTASK_DONE]
    remaining = [
        s for s in plan.get("subtasks", [])
        if s.get("status") in (SUBTASK_PENDING, SUBTASK_RUNNING, SUBTASK_FAILED)
        and s["id"] != failed["id"]
    ]

    def brief(s: dict[str, Any], with_result: bool = False) -> dict[str, Any]:
        item = {
            k: s.get(k)
            for k in ("id", "title", "kind", "depends_on", "acceptance_criteria",
                      "write_scope", "serves_clauses")
        }
        if with_result:
            item["result_summary"] = (s.get("result_summary") or "")[:400]
            item["acceptance_evidence"] = (s.get("acceptance_evidence") or [])[:5]
        return item

    payload = {
        "plan_version": plan.get("version", 1),
        "mode": mode,
        "done": [brief(s, True) for s in done],
        "failed": {
            **brief(failed, True),
            "error": (error or "")[:1500],
            "stop_reason": failed.get("stop_reason", ""),
            "attempt_number": failed.get("attempt_number", 0),
        },
        "remaining": [brief(s) for s in remaining],
    }
    if mode == FAIL_STRUCTURAL:
        instruction = (
            "失败属于结构性问题。请给出**替换**失败子任务和所有剩余子任务的新任务图"
            f"（不超过 {max_subtasks} 个）。可以增删步骤、调整 depends_on 或验收标准。"
            "已完成的子任务会保留；新 id 不能与已完成子任务重复。"
            "整体最终验收仍由控制环执行，不要做成普通子任务。"
        )
    else:
        instruction = (
            "失败属于「做法不对但仍在已批准计划范围内」。请在**不改变计划版本语义**的前提下，"
            "重写失败子任务与剩余子任务（可调整描述与做法，尽量保持 id/依赖/验收标准意图一致；"
            f"不超过 {max_subtasks} 个）。已完成子任务会保留；新 id 不能与已完成重复。"
            "不要把整体最终验收做成普通子任务。"
        )
    user = (
        f"用户目标：\n{goal}\n\n{_clause_text(plan.get('goal_clauses'))}"
        "新任务图必须沿用上面的条款编号；可以用 derived_from 标注新子任务来自哪些旧子任务。\n\n"
        "当前任务图执行情况（JSON）：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}\n\n"
        f"{instruction}\n{_SCHEMA_HINT}"
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def request_subtasks(
    llm: Any,
    messages: list[dict[str, Any]],
    max_subtasks: int,
    *,
    known_ids: set[str] | None = None,
    known_nodes: list[dict[str, Any]] | None = None,
    clause_ids: list[str] | None = None,
    retries: int = 1,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    errors: list[str] = []
    convo = list(messages)
    for _ in range(retries + 1):
        content: Any = None
        try:
            response = llm.chat(convo, [])
            content = getattr(response, "content", None)
            specs = validate_subtasks(
                parse_plan_text(content), max_subtasks, known_ids=known_ids,
                known_nodes=known_nodes, clause_ids=clause_ids,
            )
            return specs, errors
        except PlanError as exc:
            errors.append(str(exc))
            convo = convo + [
                {"role": "assistant", "content": str(content or "")[:4000]},
                {
                    "role": "user",
                    "content": f"上面的计划无效：{exc}。请严格按格式重新输出完整 JSON。",
                },
            ]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"模型调用失败：{exc}")
    return None, errors
