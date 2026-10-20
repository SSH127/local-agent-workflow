// Types + pure helpers for the persisted structured failure report.
//
// The shape mirrors ``app/engine/planning.py::build_failure_report``. These
// helpers are kept free of React/DOM so they can be unit-tested with Node's
// built-in test runner. The one invariant they enforce is: never fabricate
// content — a missing / empty / old-format field degrades to an explicit empty
// state instead of being auto-filled.

export type FailureEvidence = {
  subtask_id?: string
  kind?: string
  evidence?: string[]
  result_full?: string
}

export type FailureReport = {
  run_id?: string
  status?: string
  failure_reason?: string
  unmet_clauses?: string[]
  inconsistencies?: string[]
  available_evidence?: FailureEvidence[]
  evidence_gaps?: string[]
  suggested_next_steps?: string[]
  generated_at?: string
}

export type FailureSection = {
  key: string
  title: string
  items: string[]
  isEmpty: boolean
}

export type EvidenceEntry = {
  subtask_id?: string
  kind?: string
  evidence: string[]
  result_full?: string
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string' && item.length > 0)
}

/** A report is "present" when it is a non-empty plain object. Old-format or
 *  junk values are rejected without throwing. */
export function hasFailureReport(value: unknown): value is FailureReport {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.keys(value as Record<string, unknown>).length > 0
  )
}

/** Normalized, render-ready sections. Missing / empty fields become an empty
 *  list with an explicit ``isEmpty`` flag — never invented text. */
export function failureReportSections(
  report: FailureReport | null | undefined,
): FailureSection[] {
  const r: FailureReport = hasFailureReport(report) ? report : {}
  const defs: Array<{ key: string; title: string; value: unknown }> = [
    { key: 'unmet_clauses', title: '未满足条款', value: r.unmet_clauses },
    { key: 'inconsistencies', title: '矛盾 / 不一致', value: r.inconsistencies },
    { key: 'evidence_gaps', title: '证据缺口', value: r.evidence_gaps },
    { key: 'suggested_next_steps', title: '建议的下一步', value: r.suggested_next_steps },
  ]
  return defs.map((d) => {
    const items = asStringList(d.value)
    return { key: d.key, title: d.title, items, isEmpty: items.length === 0 }
  })
}

/** Normalized ``available_evidence`` entries. Entries with neither a brief
 *  evidence line nor a full result are dropped (they carry no information). */
export function failureEvidence(
  report: FailureReport | null | undefined,
): { items: EvidenceEntry[]; isEmpty: boolean } {
  const r: FailureReport = hasFailureReport(report) ? report : {}
  const raw = Array.isArray(r.available_evidence) ? r.available_evidence : []
  const items: EvidenceEntry[] = []
  for (const entry of raw) {
    if (typeof entry !== 'object' || entry === null) continue
    const e = entry as FailureEvidence
    const evidence = asStringList(e.evidence)
    const resultFull = typeof e.result_full === 'string' && e.result_full.length > 0
      ? e.result_full
      : undefined
    if (evidence.length === 0 && !resultFull) continue
    items.push({
      subtask_id: typeof e.subtask_id === 'string' ? e.subtask_id : undefined,
      kind: typeof e.kind === 'string' ? e.kind : undefined,
      evidence,
      result_full: resultFull,
    })
  }
  return { items, isEmpty: items.length === 0 }
}

/** True when a report exists but carries no displayable content at all. */
export function isReportEmpty(report: FailureReport | null | undefined): boolean {
  if (!hasFailureReport(report)) return true
  const reason = typeof report.failure_reason === 'string' && report.failure_reason.length > 0
  const sections = failureReportSections(report)
  const evidence = failureEvidence(report)
  return !reason && sections.every((s) => s.isEmpty) && evidence.isEmpty
}

/** Marked truncation: returns the full text unchanged when it fits, otherwise
 *  an explicit "truncated + full length" marker. Never silently cuts text. */
export function truncateMarked(text: string, limit: number): string {
  if (text.length <= limit) return text
  return `${text.slice(0, limit)}…（已截断，共 ${text.length} 字符）`
}
