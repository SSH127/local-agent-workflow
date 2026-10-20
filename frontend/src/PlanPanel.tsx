import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import { Ban, Check, Hand, History, Pencil, Play, ShieldAlert, X } from 'lucide-react'
import { apiFetch, sendJson } from './api'

// ---------------------------------------------------------------------------
// Types (mirror GET /runs/{id}/plan)
// ---------------------------------------------------------------------------

type Evidence = {
  type: string
  program?: string
  args?: string[]
  ok?: boolean
  summary?: string
  path?: string
  sha256?: string
  tool?: string
  message?: string
  scope?: string[]
  status?: string
  stop_reason?: string
  note?: string
  evidence_text?: string
  tool_calls?: number
}

type Attempt = {
  attempt: number
  status: string
  stop_reason?: string
  seconds?: number
  iterations?: number
  tokens?: { total?: number }
}

type Intervention = { type: string; at?: string; note?: string | null; approved?: boolean; [k: string]: unknown }

export type Subtask = {
  id: string
  title: string
  description: string
  depends_on: string[]
  kind: 'analyze' | 'modify' | 'verify'
  acceptance_criteria: string[]
  risk: 'low' | 'medium' | 'high'
  reversible: boolean
  status: string
  attempt_number: number
  retries: number
  result_summary?: string | null
  verify_note?: string | null
  error?: string | null
  stop_reason?: string | null
  acceptance_evidence: Evidence[]
  manual: boolean
  confirmed: boolean
  needs_confirmation: boolean
  ready: boolean
  interventions: Intervention[]
  attempts_log: Attempt[]
  duration_seconds?: number | null
  tokens?: { prompt?: number; completion?: number; total?: number; estimated?: boolean }
  cost: number
  write_scope?: string[]
  serves_clauses?: string[]
  derived_from?: string[]
}

type GoalClause = { id: string; text: string }

type ClauseReport = {
  id: string
  text: string
  status: 'covered' | 'missing'
  covered_by: string[]
  planned_by: string[]
  manual: string[]
  evidence: string[]
}

type GraphDiff = {
  added: string[]
  removed: string[]
  changed: Record<string, Record<string, { before: unknown; after: unknown }>>
}

type VersionSummary = {
  version: number
  revision: number
  parent_version: number | null
  parent_revision: number | null
  reason: string
  failure_class: string | null
  created_at: string
  subtask_count: number
}

type VersionEntry = VersionSummary & {
  plan: { goal?: string; goal_clauses?: GoalClause[]; source?: string; subtasks: Partial<Subtask>[] }
  diff: GraphDiff | null
}

type Plan = {
  version: number
  revision?: number
  goal?: string
  goal_clauses?: GoalClause[]
  clause_report?: ClauseReport[]
  source: string
  approval_status: string
  approved_version?: number | null
  rewrites: number
  structural_replans: number
  final_verify_replans?: number
  plan_rejections: number
  feedback: string[]
  attention?: { reason: string; at?: string } | null
  pending_confirmation?: string | null
  subtasks: Subtask[]
  history: { version: number; reason?: string; diff?: unknown }[]
  notes: string[]
  final_verify?: {
    passed: boolean
    missing_clauses: string[]
    inconsistencies: string[]
    notes: string[]
    clauses?: ClauseReport[]
    replan_triggered?: boolean
    replan_reason?: string | null
    suggested_next_steps?: string[]
  } | null
  totals: {
    steps: number
    active_seconds: number
    tokens: number
    cost: number
    currency: string
    limits: { max_run_steps: number; max_run_seconds: number; max_run_tokens: number }
  }
}

type PlanView = {
  run_id: string
  run_status: string
  plan_version: number
  plan_revision?: number
  plan: Plan
  lease?: Lease | null
  leases?: Record<string, Lease>
  max_parallel?: number
}

type Lease = { holder?: string; state?: string; heartbeat_at?: string; attempt_number?: number }

const FAILURE_CLASS_LABEL: Record<string, string> = {
  rejected: '计划被拒绝',
  rewrite_remaining: '重写剩余子任务',
  structural: '结构性重规划',
  final_verify: '最终验收重规划',
  manual_edit: '人工修改',
  transient: '瞬时失败',
}

function versionKey(v: { version: number; revision: number }) {
  return `${v.version}.${v.revision}`
}

// Fill the runtime fields a historical snapshot does not carry.
function fromSnapshot(s: Partial<Subtask>): Subtask {
  return {
    id: s.id ?? '?',
    title: s.title ?? '',
    description: s.description ?? '',
    depends_on: s.depends_on ?? [],
    kind: s.kind ?? 'analyze',
    acceptance_criteria: s.acceptance_criteria ?? [],
    risk: s.risk ?? 'low',
    reversible: s.reversible ?? true,
    status: s.status ?? 'pending',
    attempt_number: 0,
    retries: 0,
    acceptance_evidence: [],
    manual: Boolean(s.manual),
    confirmed: false,
    needs_confirmation: false,
    ready: false,
    interventions: [],
    attempts_log: [],
    cost: 0,
    write_scope: s.write_scope ?? [],
    serves_clauses: s.serves_clauses ?? [],
    derived_from: s.derived_from ?? [],
  }
}

function fmtValue(value: unknown): string {
  if (Array.isArray(value)) return value.length ? value.join(', ') : '（空）'
  if (value === null || value === undefined || value === '') return '（空）'
  return String(value)
}

const SUBTASK_STATUS_LABEL: Record<string, string> = {
  pending: '待执行',
  running: '执行中',
  done: '已通过',
  failed: '未通过',
  skipped: '已跳过',
  cancelled: '已取消',
}

const KIND_LABEL: Record<string, string> = { analyze: '分析', modify: '修改', verify: '验证' }
const RISK_LABEL: Record<string, string> = { low: '低风险', medium: '中风险', high: '高风险' }

// Run statuses where the user may take over (edit contract / mark result).
const TAKEOVER_STATUSES = [
  'waiting_for_plan_approval',
  'waiting_for_subtask_approval',
  'needs_attention',
  'interrupted',
  'failed',
]
const RESUMABLE_STATUSES = ['needs_attention', 'interrupted', 'failed']
const FINISHED_STATUSES = ['succeeded', 'cancelled']

function topoLevels(subtasks: Subtask[]): Map<string, number> {
  const level = new Map<string, number>()
  const byId = new Map(subtasks.map((s) => [s.id, s]))
  const visit = (id: string, seen: Set<string>): number => {
    if (level.has(id)) return level.get(id) as number
    if (seen.has(id)) return 0
    seen.add(id)
    const node = byId.get(id)
    const deps = (node?.depends_on ?? []).filter((d) => byId.has(d))
    const value = deps.length === 0 ? 0 : Math.max(...deps.map((d) => visit(d, seen))) + 1
    level.set(id, value)
    return value
  }
  subtasks.forEach((s) => visit(s.id, new Set()))
  return level
}

function fmtSeconds(value?: number | null) {
  if (value == null) return '—'
  return value < 60 ? `${value.toFixed(1)} 秒` : `${(value / 60).toFixed(1)} 分钟`
}

function fmtCost(value: number, currency: string) {
  return `${value.toFixed(4)} ${currency}`
}

function evidenceText(e: Evidence): string {
  switch (e.type) {
    case 'command':
      return `${e.ok ? '✔' : '✘'} 命令 ${e.program ?? ''} ${(e.args ?? []).join(' ')} — ${e.summary ?? ''}`
    case 'file':
      return `文件 ${e.path ?? ''}${e.sha256 ? `（sha256 ${e.sha256.slice(0, 12)}…）` : ''}`
    case 'executor_status':
      return `执行器状态 ${e.status ?? ''} / ${e.stop_reason ?? ''}（工具调用 ${e.tool_calls ?? 0}）— 仅作参考，不作为验收依据`
    case 'manual':
      return `【人工】${e.note ?? ''}${e.evidence_text ? ` — ${e.evidence_text}` : ''}`
    case 'scope_violation':
      return `⛔ 越界写入被拦截：${e.tool ?? ''} → ${e.path ?? ''}（允许范围 ${(e.scope ?? []).join(', ') || '只读'}）`
    default:
      return JSON.stringify(e)
  }
}

function contractOf(s: Subtask) {
  return {
    id: s.id,
    title: s.title,
    description: s.description,
    depends_on: s.depends_on,
    kind: s.kind,
    acceptance_criteria: s.acceptance_criteria,
    risk: s.risk,
    reversible: s.reversible,
    write_scope: s.write_scope ?? [],
    serves_clauses: s.serves_clauses ?? [],
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PlanPanel({
  runId,
  runStatus,
  onChanged,
  fallback,
}: {
  runId: string
  runStatus: string
  onChanged: () => void
  fallback: ReactNode
}) {
  const [view, setView] = useState<PlanView | null>(null)
  const [missing, setMissing] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [rejecting, setRejecting] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [editingPlan, setEditingPlan] = useState(false)
  const [planJson, setPlanJson] = useState('')
  const [manualOpen, setManualOpen] = useState(false)
  const [manualStatus, setManualStatus] = useState('done')
  const [manualNote, setManualNote] = useState('')
  const [manualEvidence, setManualEvidence] = useState('')
  const [contractOpen, setContractOpen] = useState(false)
  const [contractJson, setContractJson] = useState('')
  const [versions, setVersions] = useState<VersionSummary[]>([])
  const [versionSel, setVersionSel] = useState('current')
  const [historical, setHistorical] = useState<VersionEntry | null>(null)

  const load = useCallback(async () => {
    try {
      const response = await apiFetch(`/runs/${runId}/plan`)
      if (response.status === 404) {
        setMissing(true)
        setView(null)
        return
      }
      if (!response.ok) return
      setMissing(false)
      setView((await response.json()) as PlanView)
    } catch {
      // 网络错误时保留上一次的任务图。
    }
  }, [runId])

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      if (!cancelled) await load()
    }
    void tick()
    const timer = window.setInterval(() => void tick(), 2000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [load, runStatus])

  const act = useCallback(
    async (path: string, method: string, body?: unknown) => {
      setBusy(true)
      setError('')
      try {
        await sendJson(path, method, body)
        await load()
        onChanged()
        return true
      } catch (err) {
        setError(err instanceof Error ? err.message : '操作失败')
        return false
      } finally {
        setBusy(false)
      }
    },
    [load, onChanged],
  )

  const plan = view?.plan
  const status = view?.run_status ?? runStatus
  const currentKey = plan ? `${plan.version}.${plan.revision ?? 1}` : ''

  // Append-only version history: reload the list whenever the plan moves on.
  useEffect(() => {
    if (!currentKey) return
    let cancelled = false
    void (async () => {
      try {
        const response = await apiFetch(`/runs/${runId}/plan/versions`)
        if (!response.ok || cancelled) return
        const data = (await response.json()) as { versions: VersionSummary[] }
        if (!cancelled) setVersions(data.versions)
      } catch {
        // 历史版本列表加载失败时保持原样。
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runId, currentKey])

  useEffect(() => {
    if (versionSel === 'current') return
    let cancelled = false
    const [v, r] = versionSel.split('.')
    void (async () => {
      try {
        const response = await apiFetch(`/runs/${runId}/plan/versions/${v}?revision=${r}`)
        if (!response.ok || cancelled) return
        const entry = (await response.json()) as VersionEntry
        if (!cancelled) setHistorical(entry)
      } catch {
        // 加载失败时继续显示当前版本。
      }
    })()
    return () => {
      cancelled = true
    }
  }, [runId, versionSel])

  // Only an entry that matches the current selection is shown (no stale view).
  const viewingHistory =
    versionSel !== 'current' && historical !== null && versionKey(historical) === versionSel
  const shownSubtasks = useMemo<Subtask[]>(() => {
    if (viewingHistory && historical) return historical.plan.subtasks.map(fromSnapshot)
    return plan?.subtasks ?? []
  }, [viewingHistory, historical, plan])
  const diff = viewingHistory ? historical?.diff ?? null : null
  const selected = shownSubtasks.find((s) => s.id === selectedId) ?? null

  const { nodes, edges } = useMemo(() => {
    if (!plan) return { nodes: [] as Node[], edges: [] as Edge[] }
    const levels = topoLevels(shownSubtasks)
    const rows = new Map<number, number>()
    const diffClass = (id: string) =>
      !diff ? '' : diff.added.includes(id) ? ' plan-diff-added' : diff.changed[id] ? ' plan-diff-changed' : ''
    const nodes: Node[] = shownSubtasks.map((s) => {
      const lvl = levels.get(s.id) ?? 0
      const row = rows.get(lvl) ?? 0
      rows.set(lvl, row + 1)
      const badges = [
        s.manual ? '人工' : '',
        s.risk === 'high' || !s.reversible ? '高风险' : '',
        s.retries > 0 ? `重试×${s.retries}` : '',
        !viewingHistory && plan.pending_confirmation === s.id ? '待确认' : '',
        diff?.added.includes(s.id) ? '新增' : '',
        diff?.changed[s.id] ? '已修改' : '',
        s.write_scope && s.write_scope.length === 0 ? '只读' : '',
      ].filter(Boolean)
      return {
        id: s.id,
        position: { x: 40 + lvl * 260, y: 30 + row * 120 },
        data: {
          label: (
            <div className="plan-node-body">
              <div className="plan-node-title">[{s.id}] {s.title}</div>
              <div className="plan-node-meta">
                {KIND_LABEL[s.kind] ?? s.kind} · {SUBTASK_STATUS_LABEL[s.status] ?? s.status}
                {s.attempt_number > 0 ? ` · 第 ${s.attempt_number} 次` : ''}
              </div>
              {badges.length > 0 && (
                <div className="plan-node-badges">
                  {badges.map((b) => <span key={b} className="plan-badge">{b}</span>)}
                </div>
              )}
            </div>
          ),
        },
        className: `plan-node plan-node-${s.status}${s.manual ? ' plan-node-manual' : ''}${
          selectedId === s.id ? ' plan-node-selected' : ''}${diffClass(s.id)}${
          viewingHistory ? ' plan-node-history' : ''}`,
      }
    })
    const edges: Edge[] = shownSubtasks.flatMap((s) =>
      s.depends_on.map((dep) => ({
        id: `${dep}->${s.id}`,
        source: dep,
        target: s.id,
        animated: s.status === 'running',
        markerEnd: { type: MarkerType.ArrowClosed },
        style: { strokeWidth: 2, stroke: s.status === 'done' ? '#59c89b' : '#9aa3b8' },
      })),
    )
    return { nodes, edges }
  }, [plan, shownSubtasks, selectedId, diff, viewingHistory])

  if (missing || !plan) return <>{fallback}</>

  const totals = plan.totals
  const canTakeOver = TAKEOVER_STATUSES.includes(status) && !viewingHistory
  const clauses = plan.final_verify?.clauses ?? plan.clause_report ?? []
  const runningLeases = Object.entries(view?.leases ?? {}).filter(([, l]) => l.state !== 'released')
  const pendingSub = plan.subtasks.find((s) => s.id === plan.pending_confirmation)

  return (
    <div className="plan-panel">
      <div className="plan-header">
        <div>
          <strong>任务图 v{plan.version}{(plan.revision ?? 1) > 1 ? `.r${plan.revision}` : ''}</strong>
          <span className="plan-sub">
            来源 {plan.source} · 计划审批 {plan.approval_status}
            {plan.rewrites ? ` · 重写 ${plan.rewrites} 次` : ''}
            {plan.structural_replans ? ` · 结构性重规划 ${plan.structural_replans} 次` : ''}{plan.final_verify_replans ? ` · 最终验收重规划 ${plan.final_verify_replans} 次` : ''}
            {plan.plan_rejections ? ` · 被拒绝 ${plan.plan_rejections} 次` : ''}
            {view?.max_parallel ? ` · 并行上限 ${view.max_parallel}` : ''}
            {runningLeases.length ? ` · 执行中 ${runningLeases.map(([id]) => id).join(', ')}` : ''}
          </span>
          {versions.length > 0 && (
            <label className="plan-version-select">
              <History size={14} />
              <select value={versionSel} onChange={(e) => {
                setVersionSel(e.target.value)
                setSelectedId(null)
              }}>
                <option value="current">当前版本（v{plan.version}.r{plan.revision ?? 1}）</option>
                {[...versions].reverse().map((v) => (
                  <option key={versionKey(v)} value={versionKey(v)}>
                    v{v.version}.r{v.revision} · {FAILURE_CLASS_LABEL[v.failure_class ?? ''] ?? '初始'} · {v.reason.slice(0, 40)}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <div className="plan-totals">
          <span>步数 {totals.steps}/{totals.limits.max_run_steps}</span>
          <span>时长 {fmtSeconds(totals.active_seconds)} / {totals.limits.max_run_seconds}s</span>
          <span>Token {totals.tokens}/{totals.limits.max_run_tokens}</span>
          <span>估算费用 {fmtCost(totals.cost, totals.currency)}</span>
        </div>
      </div>

      {error && <div className="detail-error">{error}</div>}

      {viewingHistory && historical && (
        <div className="plan-history-banner">
          <strong>只读：历史版本 v{historical.version}.r{historical.revision}</strong>
          <span>
            {historical.parent_version != null
              ? `基于 v${historical.parent_version}.r${historical.parent_revision ?? 1}`
              : '初始版本'}
            {historical.failure_class ? ` · 原因分类 ${FAILURE_CLASS_LABEL[historical.failure_class] ?? historical.failure_class}` : ''}
            {` · ${new Date(historical.created_at).toLocaleString()}`}
          </span>
          <p>{historical.reason}</p>
          {diff && (
            <p className="plan-diff-legend">
              <span className="diff-chip diff-added">新增 {diff.added.length}</span>
              <span className="diff-chip diff-changed">修改 {Object.keys(diff.changed).length}</span>
              <span className="diff-chip diff-removed">
                删除 {diff.removed.length}{diff.removed.length ? `（${diff.removed.join(', ')}）` : ''}
              </span>
            </p>
          )}
          <button className="button button-secondary" onClick={() => setVersionSel('current')}>回到当前版本</button>
        </div>
      )}

      {plan.attention && (
        <div className="approval-banner attention-banner">
          <ShieldAlert size={18} />
          <div className="approval-text">
            <strong>需要人工处理</strong>
            <p>{plan.attention.reason}</p>
          </div>
        </div>
      )}

      {!viewingHistory && status === 'waiting_for_plan_approval' && (
        <div className="approval-banner">
          <ShieldAlert size={18} />
          <div className="approval-text">
            <strong>任务图 v{plan.version} 等待批准</strong>
            <p>批准后按依赖顺序执行；拒绝需填写反馈，系统会据此重新拆解。</p>
            {rejecting && (
              <textarea
                className="input-reply"
                rows={2}
                value={feedback}
                placeholder="说明哪里不对，例如：拆得太粗，先读 README 再改代码…"
                onChange={(e) => setFeedback(e.target.value)}
              />
            )}
            {editingPlan && (
              <textarea
                className="input-reply plan-json"
                rows={10}
                value={planJson}
                onChange={(e) => setPlanJson(e.target.value)}
              />
            )}
          </div>
          {!rejecting && !editingPlan && (
            <>
              <button className="button button-primary" disabled={busy}
                onClick={() => void act(`/runs/${runId}/plan/approve`, 'POST', { decision: 'approve' })}>
                <Check size={16} /> 批准
              </button>
              <button className="button button-secondary" disabled={busy}
                onClick={() => {
                  setPlanJson(JSON.stringify(
                    plan.subtasks.filter((s) => s.status !== 'done').map(contractOf), null, 2))
                  setEditingPlan(true)
                }}>
                <Pencil size={16} /> 编辑
              </button>
              <button className="button button-secondary" disabled={busy} onClick={() => setRejecting(true)}>
                <X size={16} /> 拒绝
              </button>
            </>
          )}
          {rejecting && (
            <>
              <button className="button button-primary" disabled={busy || !feedback.trim()}
                onClick={async () => {
                  if (await act(`/runs/${runId}/plan/approve`, 'POST', { decision: 'reject', feedback })) {
                    setRejecting(false)
                    setFeedback('')
                  }
                }}>
                <X size={16} /> 提交反馈并重新拆解
              </button>
              <button className="button button-secondary" onClick={() => setRejecting(false)}>取消</button>
            </>
          )}
          {editingPlan && (
            <>
              <button className="button button-primary" disabled={busy}
                onClick={async () => {
                  let subtasks: unknown
                  try {
                    subtasks = JSON.parse(planJson)
                  } catch {
                    setError('JSON 格式不正确')
                    return
                  }
                  if (await act(`/runs/${runId}/plan/approve`, 'POST', { decision: 'approve', subtasks })) {
                    setEditingPlan(false)
                  }
                }}>
                <Check size={16} /> 保存并批准
              </button>
              <button className="button button-secondary" onClick={() => setEditingPlan(false)}>取消</button>
            </>
          )}
        </div>
      )}

      {!viewingHistory && status === 'waiting_for_subtask_approval' && pendingSub && (
        <div className="approval-banner">
          <ShieldAlert size={18} />
          <div className="approval-text">
            <strong>高风险 / 不可逆子任务 [{pendingSub.id}] 执行前确认</strong>
            <p>{pendingSub.title} — {RISK_LABEL[pendingSub.risk]}{pendingSub.reversible ? '' : '，不可逆'}</p>
          </div>
          <button className="button button-primary" disabled={busy}
            onClick={() => void act(`/runs/${runId}/subtasks/${pendingSub.id}/confirm`, 'POST', { approved: true })}>
            <Check size={16} /> 允许执行
          </button>
          <button className="button button-secondary" disabled={busy}
            onClick={() => void act(`/runs/${runId}/subtasks/${pendingSub.id}/confirm`, 'POST', { approved: false })}>
            <X size={16} /> 跳过
          </button>
        </div>
      )}

      <div className="plan-actions">
        {!viewingHistory && RESUMABLE_STATUSES.includes(status) && (
          <button className="button button-primary" disabled={busy}
            onClick={() => void act(`/runs/${runId}/resume`, 'POST')}>
            <Play size={16} /> 继续执行
          </button>
        )}
        {!viewingHistory && !FINISHED_STATUSES.includes(status) && (
          <button className="button button-secondary" disabled={busy}
            onClick={() => void act(`/runs/${runId}/cancel`, 'POST')}>
            <Ban size={16} /> 取消任务
          </button>
        )}
      </div>

      <div className="plan-body">
        <div className="canvas-wrap plan-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            nodesDraggable
            nodesConnectable={false}
            onNodeClick={(_, node) => {
              setSelectedId(node.id)
              setManualOpen(false)
              setContractOpen(false)
            }}
          >
            <Background color="#dfe4ed" gap={20} />
            <Controls />
          </ReactFlow>
        </div>

        <div className="plan-detail">
          {!selected ? (
            <div className="detail-empty">点击任务图中的节点查看子任务详情。</div>
          ) : (
            <div className="detail-content">
              <div className="detail-status">
                <span className={`status-pill subtask-${selected.status}`}>
                  {SUBTASK_STATUS_LABEL[selected.status] ?? selected.status}
                </span>
                {selected.manual && <span className="plan-badge">人工标记</span>}
              </div>
              <div className="detail-field"><span>[{selected.id}] {selected.title}</span>
                <p>{selected.description || '—'}</p></div>
              <div className="detail-field"><span>类型 / 风险</span>
                <p>{KIND_LABEL[selected.kind]} · {RISK_LABEL[selected.risk]}{selected.reversible ? '' : ' · 不可逆'}
                  {selected.depends_on.length ? ` · 依赖 ${selected.depends_on.join(', ')}` : ''}</p></div>
              <div className="detail-field"><span>写入范围（write_scope）</span>
                <p>{(selected.write_scope ?? []).length ? (selected.write_scope ?? []).join(', ') : '只读（不允许写入）'}</p></div>
              <div className="detail-field"><span>服务的目标条款</span>
                <p>{(selected.serves_clauses ?? []).length ? (selected.serves_clauses ?? []).map((id) => {
                  const c = (plan.goal_clauses ?? []).find((x) => x.id === id)
                  return c ? `${id}：${c.text}` : id
                }).join('；') : '—'}</p></div>
              {(selected.derived_from ?? []).length > 0 && (
                <div className="detail-field"><span>来源（上一版本中的子任务）</span>
                  <p>{(selected.derived_from ?? []).join(', ')}</p></div>
              )}
              {diff?.changed[selected.id] && (
                <div className="detail-field"><span>相对上一版本的修改</span>
                  <ul className="diff-list">{Object.entries(diff.changed[selected.id]).map(([field, d]) => (
                    <li key={field}><code>{field}</code>：{fmtValue(d.before)} → {fmtValue(d.after)}</li>
                  ))}</ul></div>
              )}
              <div className="detail-field"><span>验收标准</span>
                <ul>{selected.acceptance_criteria.map((c) => <li key={c}>{c}</li>)}</ul></div>
              <div className="detail-field"><span>尝试 / 重试</span>
                <p>第 {selected.attempt_number} 次尝试，重试 {selected.retries} 次</p>
                {selected.attempts_log.length > 0 && (
                  <ul>{selected.attempts_log.map((a) => (
                    <li key={a.attempt}>#{a.attempt} {a.status}（{a.stop_reason || '—'}）· {fmtSeconds(a.seconds)} · {a.tokens?.total ?? 0} token</li>
                  ))}</ul>
                )}
              </div>
              <div className="detail-field"><span>耗时 / Token / 费用</span>
                <p>{fmtSeconds(selected.duration_seconds)} · 输入 {selected.tokens?.prompt ?? 0} / 输出 {selected.tokens?.completion ?? 0}
                  {selected.tokens?.estimated ? '（含估算）' : ''} · {fmtCost(selected.cost, totals.currency)}</p></div>
              {selected.verify_note && <div className="detail-field"><span>验收结论</span><p>{selected.verify_note}</p></div>}
              {selected.error && <div className="detail-error">{selected.error}</div>}
              <div className="detail-field"><span>验收证据</span>
                {selected.acceptance_evidence.length === 0 ? <p>暂无</p> : (
                  <ul className="evidence-list">{selected.acceptance_evidence.map((e, i) => (
                    <li key={i} className={`evidence-${e.type}`}>{evidenceText(e)}</li>
                  ))}</ul>
                )}
              </div>
              <div className="detail-field"><span>人工干预</span>
                {selected.interventions.length === 0 ? <p>无</p> : (
                  <ul>{selected.interventions.map((iv, i) => (
                    <li key={i}>{iv.type}{iv.approved === undefined ? '' : iv.approved ? '（同意）' : '（拒绝）'}{iv.note ? `：${iv.note}` : ''}</li>
                  ))}</ul>
                )}
              </div>
              {selected.result_summary && <div className="detail-field"><span>结果摘要</span><p>{selected.result_summary}</p></div>}

              {canTakeOver && selected.status !== 'done' && (
                <div className="takeover">
                  <div className="takeover-buttons">
                    <button className="button button-secondary" onClick={() => {
                      setManualOpen((v) => !v)
                      setContractOpen(false)
                    }}><Hand size={15} /> 人工标记结果</button>
                    <button className="button button-secondary" onClick={() => {
                      const c = contractOf(selected)
                      setContractJson(JSON.stringify({
                        title: c.title, description: c.description, depends_on: c.depends_on, kind: c.kind,
                        acceptance_criteria: c.acceptance_criteria, risk: c.risk, reversible: c.reversible,
                        write_scope: c.write_scope, serves_clauses: c.serves_clauses,
                      }, null, 2))
                      setContractOpen((v) => !v)
                      setManualOpen(false)
                    }}><Pencil size={15} /> 修改契约</button>
                  </div>
                  {manualOpen && (
                    <div className="takeover-form">
                      <select value={manualStatus} onChange={(e) => setManualStatus(e.target.value)}>
                        <option value="done">标记为已完成</option>
                        <option value="failed">标记为失败</option>
                        <option value="skipped">标记为跳过</option>
                      </select>
                      <textarea rows={2} placeholder="说明（必填）" value={manualNote}
                        onChange={(e) => setManualNote(e.target.value)} />
                      <textarea rows={3} placeholder="证据（命令输出、文件路径等）" value={manualEvidence}
                        onChange={(e) => setManualEvidence(e.target.value)} />
                      <button className="button button-primary" disabled={busy || !manualNote.trim()}
                        onClick={async () => {
                          if (await act(`/runs/${runId}/subtasks/${selected.id}/manual-result`, 'POST', {
                            status: manualStatus, note: manualNote, evidence_text: manualEvidence,
                          })) {
                            setManualOpen(false)
                            setManualNote('')
                            setManualEvidence('')
                          }
                        }}>提交</button>
                    </div>
                  )}
                  {contractOpen && (
                    <div className="takeover-form">
                      <p className="input-hint">修改依赖 / 描述 / 验收标准 / 类型 / 风险 / 写入范围 / 目标条款属于结构性修改：会生成新版本任务图并回到计划审批；只改标题记为同版本的新修订。</p>
                      <textarea className="plan-json" rows={10} value={contractJson}
                        onChange={(e) => setContractJson(e.target.value)} />
                      <button className="button button-primary" disabled={busy}
                        onClick={async () => {
                          let body: unknown
                          try {
                            body = JSON.parse(contractJson)
                          } catch {
                            setError('JSON 格式不正确')
                            return
                          }
                          if (await act(`/runs/${runId}/subtasks/${selected.id}`, 'PATCH', body)) {
                            setContractOpen(false)
                          }
                        }}>保存修改</button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {(plan.goal_clauses ?? []).length > 0 && (
        <div className="clause-report">
          <strong>目标条款追踪（原始目标已冻结）</strong>
          <ul>
            {(plan.goal_clauses ?? []).map((c) => {
              const r = clauses.find((x) => x.id === c.id)
              const covered = r?.status === 'covered'
              return (
                <li key={c.id} className={covered ? 'clause-covered' : 'clause-missing'}>
                  <code>{c.id}</code> {c.text}
                  <span className="clause-status">
                    {covered
                      ? `✔ 已覆盖：${r?.covered_by.join(', ')}${r?.evidence.length ? `（证据：${r.evidence.slice(0, 3).join('；')}）` : ''}`
                      : `○ 未覆盖${r?.planned_by.length ? `（计划由 ${r.planned_by.join(', ')} 完成）` : ''}`}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {plan.final_verify && (
        <div className={`final-verify ${plan.final_verify.passed ? 'final-ok' : 'final-bad'}`}>
          <strong>最终验收（对照原始目标）：{plan.final_verify.passed ? '通过' : '未通过'}</strong>
          {plan.final_verify.replan_triggered && (
            <div className="final-replan">↻ 已触发重规划：{plan.final_verify.replan_reason || '见计划版本历史'}</div>
          )}
          {[...plan.final_verify.missing_clauses, ...plan.final_verify.inconsistencies, ...plan.final_verify.notes]
            .map((m, i) => <div key={i}>· {m}</div>)}
          {(plan.final_verify.suggested_next_steps || []).map((m, i) => (
            <div key={`tip-${i}`}>→ {m}</div>
          ))}
        </div>
      )}
      {(plan.notes ?? []).length > 0 && <div className="input-hint">{(plan.notes ?? []).join('；')}</div>}
    </div>
  )
}
