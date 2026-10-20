import { CircleAlert, FileWarning } from 'lucide-react'
import {
  failureEvidence,
  failureReportSections,
  hasFailureReport,
  isReportEmpty,
  truncateMarked,
  type FailureReport,
} from './failureReportModel'

function formatDate(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function Section({ title, items, isEmpty }: { title: string; items: string[]; isEmpty: boolean }) {
  return (
    <div className="fr-section">
      <div className="fr-section-title">{title}</div>
      {isEmpty ? (
        <div className="fr-empty">（未提供）</div>
      ) : (
        <ul className="fr-list">
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * Renders the persisted structured failure report inline, in the user's normal
 * task-detail view. Text is rendered as React text nodes only (no raw HTML), so
 * model/tool output can never be interpreted as executable markup. Long full
 * results are shown inside a collapsible `<details>`, never silently truncated.
 */
export default function FailureReportCard({
  report,
  runStatus,
}: {
  report?: FailureReport | null
  runStatus?: string
}) {
  // Successful runs keep their original summary view untouched.
  if (runStatus === 'succeeded') return null

  if (!hasFailureReport(report)) {
    if (runStatus === 'failed') {
      return (
        <div className="failure-report failure-report-missing">
          <FileWarning size={16} />
          <div>
            <strong>失败报告不可用</strong>
            <p>该失败运行没有持久化的结构化失败报告。</p>
          </div>
        </div>
      )
    }
    return null
  }

  if (isReportEmpty(report)) {
    return (
      <div className="failure-report failure-report-empty">
        <CircleAlert size={16} />
        <div>
          <strong>失败报告为空</strong>
          <p>该运行已记录失败，但失败报告中没有可展示的内容。</p>
        </div>
      </div>
    )
  }

  const sections = failureReportSections(report)
  const evidence = failureEvidence(report)

  return (
    <div className="failure-report">
      <div className="failure-report-head">
        <div className="fr-head-icon"><CircleAlert size={16} /></div>
        <div>
          <strong>失败报告</strong>
          <p>
            {report.status ? `状态 ${report.status}` : ''}
            {report.run_id ? ` · 运行 ${report.run_id}` : ''}
            {report.generated_at ? ` · ${formatDate(report.generated_at)}` : ''}
          </p>
        </div>
      </div>

      {report.failure_reason && (
        <div className="fr-reason">{truncateMarked(report.failure_reason, 4000)}</div>
      )}

      {sections.map((section) => (
        <Section
          key={section.key}
          title={section.title}
          items={section.items}
          isEmpty={section.isEmpty}
        />
      ))}

      <div className="fr-section">
        <div className="fr-section-title">可用证据</div>
        {evidence.isEmpty ? (
          <div className="fr-empty">（未提供）</div>
        ) : (
          <div className="fr-evidence">
            {evidence.items.map((item, i) => (
              <details key={i} className="fr-evidence-item">
                <summary>
                  {item.subtask_id ? `子任务 [${item.subtask_id}]` : '子任务'}
                  {item.kind ? ` · ${item.kind}` : ''}
                  {item.evidence.length > 0 ? ` · ${item.evidence[0]}` : ''}
                </summary>
                <div className="fr-evidence-body">
                  {item.evidence.length > 0 && (
                    <ul className="fr-list">
                      {item.evidence.map((line, j) => (
                        <li key={j}>{line}</li>
                      ))}
                    </ul>
                  )}
                  {item.result_full && (
                    <pre className="fr-full-result">{item.result_full}</pre>
                  )}
                </div>
              </details>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
