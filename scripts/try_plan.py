"""Try the planner on one goal, without executing anything.

    python scripts/try_plan.py --workspace PATH --goal "..." [--dry-run] [--out plan.json]

Phase 4: classify → explore → evidence registry → plan with evidence_id-bound
targets + coverage map → gates/binding → (one extra explore round on missing
evidence) → critic (one revise) → handoff packets. Records tokens/cost.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

_PRICE_IN = 0.28
_PRICE_OUT = 0.42


def _cost_usd(usage: dict) -> float:
    return (usage.get("prompt_tokens", 0) * _PRICE_IN
            + usage.get("completion_tokens", 0) * _PRICE_OUT) / 1_000_000


def _missing_from_errors(errors: list[str]) -> list[str]:
    out: list[str] = []
    for e in errors:
        for m in re.finditer(r"target ([\w./-]+):([A-Za-z_][\w.]*)", e):
            out.append(m.group(2))
        for m in re.finditer(r"modify target ([A-Za-z_][\w.]*)", e):
            out.append(m.group(1))
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    from app import config
    from app.engine import planning as P
    from app.planner import quality as Q
    from app.planner.evidence import EVIDENCE_BINDING_HINT, augment_registry, render_registry
    from app.planner.explore import explore_repo
    from app.planner.repo_index import build_index

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-subtasks", type=int, default=None)
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--no-explore", action="store_true")
    ap.add_argument("--out", help="save the plan JSON here")
    args = ap.parse_args(argv)

    ws = Path(args.workspace).resolve()
    task_type = Q.classify_task_type(args.goal)
    max_st = args.max_subtasks or Q.max_subtasks_for(args.goal, config.MAX_SUBTASKS)
    clauses = P.split_goal_clauses(args.goal)
    ids = [c["id"] for c in clauses]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    budget = int(getattr(config, "PLANNER_EXPLORE_BUDGET", 6) or 6)
    do_explore = (not args.no_explore) and getattr(config, "PLANNER_EXPLORE", True)
    index = build_index(ws)

    llm = None
    registry: list[dict] = []
    explore_meta: dict = {}
    explore_text = None
    if do_explore:
        if not args.dry_run:
            from app.agents.llm import DeepSeekClient
            llm = DeepSeekClient()
        er = explore_repo(args.goal, ws, llm=llm, budget=budget, usage_acc=usage,
                          index=index, clauses=clauses)
        registry = er.registry
        explore_text = er.text
        explore_meta = {"budget_used": er.budget_used, "done_reason": er.done_reason,
                        "key_files": er.key_files, "key_symbols": er.key_symbols,
                        "steps": er.steps[:20], "extra_round": False}

    def _messages():
        text = render_registry(registry, ws) if registry else explore_text
        msgs, pk = P.build_grounded_planner_messages(
            args.goal, ws, max_st, clauses, phase2=True, explore_text=text)
        if registry:
            msgs[-1]["content"] += "\n" + EVIDENCE_BINDING_HINT
        return msgs, pk

    messages, pack = _messages()
    print("=" * 30 + " EXPLORE / EVIDENCE " + "=" * 30)
    print(f"task_type={task_type} max_subtasks={max_st} registry={len(registry)} "
          f"explore={ {k: explore_meta.get(k) for k in ('budget_used', 'done_reason')} if explore_meta else 'off'}")
    if registry:
        print(render_registry(registry, None)[:3000])
    if args.dry_run:
        print("=" * 30 + " PLANNER MESSAGES " + "=" * 30)
        for m in messages:
            print(f"--- [{m['role']}] ---")
            print(m["content"])
        return 0

    if llm is None:
        from app.agents.llm import DeepSeekClient
        llm = DeepSeekClient()
    meta: dict = {}
    reg_arg = registry if registry else None
    specs, errors, usage = P.request_subtasks_phase2(
        llm, messages, max_st, goal=args.goal, workspace=ws, clause_ids=ids,
        retries=2, usage_acc=usage, registry=reg_arg, meta_out=meta)

    # One extra explore round (deterministic lookup) when key targets lacked evidence.
    if specs is None and registry:
        wanted = list(dict.fromkeys(list(meta.get("binding_missing") or [])
                                    + list(meta.get("missing_evidence") or [])
                                    + _missing_from_errors(errors)))[:8]
        new_ids = augment_registry(registry, index, wanted, clauses) if wanted else []
        explore_meta["extra_round"] = {"wanted": wanted, "new_ids": new_ids}
        messages, pack = _messages()
        if errors:
            messages[-1]["content"] += "\n\n上一版计划被拒绝：" + "；".join(errors[-2:])[:1500]
        meta = {}
        specs, err2, usage = P.request_subtasks_phase2(
            llm, messages, max_st, goal=args.goal, workspace=ws, clause_ids=ids,
            retries=1, usage_acc=usage, registry=reg_arg, meta_out=meta)
        errors.extend(err2)

    critic_meta = {"verdict": "skipped", "issues": []}
    if specs is not None and not args.no_critic and getattr(config, "PLANNER_CRITIC", True):
        grounding = render_registry(registry, None) if registry else ((pack or {}).get("text"))
        revised, critic_meta = Q.critic_revise(
            llm, args.goal, specs, task_type=task_type, grounding=grounding, usage_acc=usage)
        if revised is not None:
            m2: dict = {}
            specs2, err2, usage = P.request_subtasks_phase2(
                llm, [], max_st, goal=args.goal, workspace=ws, clause_ids=ids, retries=0,
                usage_acc=usage, raw_specs=revised, registry=reg_arg, meta_out=m2,
                raw_coverage=critic_meta.get("revised_coverage") or meta.get("coverage"))
            if specs2 is not None:
                specs = specs2
                if critic_meta.get("revised_coverage"):
                    meta["coverage"] = critic_meta["revised_coverage"]
                critic_meta["applied"] = True
            else:
                critic_meta["applied"] = False
                critic_meta["rejected_by_gates"] = err2[:3]

    print("=" * 30 + " PLAN " + "=" * 30)
    blocked = specs is None
    if blocked:
        print("规划失败（blocked）：", *errors, sep="\n- ")
    warnings = [] if blocked else P.check_plan_grounding(specs, ws)
    handoff = [] if blocked else Q.build_handoff_packets(specs, ws)
    if not blocked:
        print(json.dumps({"task_type": task_type, "subtasks": specs,
                          "coverage": meta.get("coverage")}, ensure_ascii=False, indent=2))
    print("validation retries/errors:", errors or "none")
    print("critic:", {k: critic_meta.get(k) for k in ("verdict", "applied")})
    print("usage:", usage, "cost_usd≈", round(_cost_usd(usage), 6))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "goal": args.goal, "task_type": task_type, "blocked": blocked,
            "subtasks": specs or [], "coverage": meta.get("coverage") or [],
            "evidence_registry": registry, "grounding_warnings": warnings,
            "handoff_packets": handoff, "critic": critic_meta, "explore": explore_meta,
            "errors": errors[-6:], "usage": usage, "cost_usd": round(_cost_usd(usage), 6),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
