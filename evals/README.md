# Planning evals (offline)

Goal: measure whether task decomposition is *grounded and executable by a weak model*,
without calling any LLM during scoring.

## Workflow
1. Generate a plan per goal (this one calls the planner model once per goal):
   `python scripts/try_plan.py --workspace . --goal "<goal>" --out evals/plans/G01.json`
   Use `--dry-run` first to inspect the grounding pack and exact prompt (no model call).
2. Score saved plans offline:
   `python scripts/eval_plans.py --goals evals/planning_goals.jsonl --plans evals/plans --workspace .`

Each plan JSON: `{"goal": ..., "subtasks": [...], "grounding_warnings": [...]}` (what `--out` writes),
file name `<goal id>.json`.

## Rubric (0–1 each; automatic part in parentheses)
| Dimension | What good looks like | Auto metric |
|---|---|---|
| Grounded files exist | every write_scope / mentioned path is real or explicitly new | share of write_scope paths that exist or are declared new |
| Expected coverage | plan touches the files/symbols a human would | recall of `expected_files`, `expected_symbols` in plan text/scopes |
| Granularity | 2–6 subtasks for normal goals; no "do everything" node; no trivial nodes | penalty outside [2,6] (analysis goals: 1–3) |
| Dependency correctness | modify → verify ordering; overlapping scopes ordered | every verify depends (transitively) on a modify; kinds match `expected_kinds` |
| Impact coverage | callers of changed symbols are verified | any verify/acceptance text mentions tests or callers |
| Acceptance testability | criteria are checkable (command, test name, observable) | share of criteria containing a test/command/file/symbol token |
| Weak-model executable | each subtask names concrete files + symbols, scope is narrow | share of modify subtasks whose description names a file and a symbol, scope ≤ 4 entries |

Manual review is still recommended for the last two; the script prints per-dimension scores
and an overall mean so prompt/grounding changes can be compared run to run.
