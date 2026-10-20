import test from 'node:test'
import assert from 'node:assert/strict'
import {
  failureEvidence,
  failureReportSections,
  hasFailureReport,
  isReportEmpty,
  truncateMarked,
  type FailureReport,
} from '../src/failureReportModel.ts'

const fullReport: FailureReport = {
  run_id: 'run-1',
  status: 'final_verify_failed',
  failure_reason: '最终验收未通过：缺少部署说明',
  unmet_clauses: ['目标条款 C1「部署」没有覆盖', '子任务 [2] 状态为 failed'],
  inconsistencies: ['文件 a.py 与记录证据不一致'],
  available_evidence: [
    { subtask_id: '1', kind: 'analyze', evidence: ['工具调用成功 2 次（共 3 次）'], result_full: '完整的原始结果' },
  ],
  evidence_gaps: ['目标条款 C1「部署」缺少充分证据（状态：missing）'],
  suggested_next_steps: ['补齐未覆盖的目标条款'],
  generated_at: '2026-10-21T00:00:00+00:00',
}

test('hasFailureReport accepts a populated report and rejects junk', () => {
  assert.equal(hasFailureReport(fullReport), true)
  assert.equal(hasFailureReport({}), false)
  assert.equal(hasFailureReport(null), false)
  assert.equal(hasFailureReport(undefined), false)
  assert.equal(hasFailureReport('not an object'), false)
  assert.equal(hasFailureReport([1, 2, 3]), false)
})

test('failureReportSections extracts every field and flags empties', () => {
  const sections = failureReportSections(fullReport)
  const byKey = new Map(sections.map((s) => [s.key, s]))
  assert.equal(byKey.get('unmet_clauses')!.items.length, 2)
  assert.equal(byKey.get('inconsistencies')!.items.length, 1)
  assert.equal(byKey.get('evidence_gaps')!.items.length, 1)
  assert.equal(byKey.get('suggested_next_steps')!.items.length, 1)
  assert.equal(byKey.get('unmet_clauses')!.isEmpty, false)
})

test('missing / empty / old-format fields degrade to empty, never fabricated', () => {
  const sections = failureReportSections({ run_id: 'x', status: 'failed' })
  for (const s of sections) {
    assert.equal(s.items.length, 0)
    assert.equal(s.isEmpty, true)
  }
  // Old format: an unexpected type for a list field must not throw or invent text.
  const weird = failureReportSections({ unmet_clauses: 'not-an-array' } as unknown as FailureReport)
  assert.deepEqual(weird[0].items, [])
  assert.equal(weird[0].isEmpty, true)
})

test('isReportEmpty distinguishes empty reports from populated ones', () => {
  assert.equal(isReportEmpty(null), true)
  assert.equal(isReportEmpty({}), true)
  assert.equal(isReportEmpty({ run_id: 'x', status: 'failed' }), true)
  assert.equal(isReportEmpty({ failure_reason: '失败' }), false)
  assert.equal(isReportEmpty(fullReport), false)
})

test('failureEvidence normalizes entries and drops information-less ones', () => {
  const evidence = failureEvidence(fullReport)
  assert.equal(evidence.isEmpty, false)
  assert.equal(evidence.items.length, 1)
  assert.equal(evidence.items[0].subtask_id, '1')
  assert.equal(evidence.items[0].evidence[0], '工具调用成功 2 次（共 3 次）')
  assert.equal(evidence.items[0].result_full, '完整的原始结果')
  // Entries with neither evidence lines nor a full result carry nothing.
  const empty = failureEvidence({ available_evidence: [{ subtask_id: '1' }] })
  assert.equal(empty.isEmpty, true)
})

test('truncateMarked never silently cuts text', () => {
  const long = '结'.repeat(2500)
  assert.equal(truncateMarked('short', 1000), 'short')
  const marked = truncateMarked(long, 1000)
  assert.ok(marked.length < long.length)
  assert.ok(marked.includes('已截断'))
  assert.ok(marked.includes('2500'))
})
