
import { useCallback, useEffect, useState } from 'react'
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  Activity,
  Check,
  CheckCircle2,
  CircleAlert,
  Clock3,
  GitBranch,
  MessageSquare,
  Play,
  Send,
  RefreshCw,
  ShieldAlert,
  Terminal,
  Workflow,
  X,
} from 'lucide-react'
import './App.css'

import {
  API_BASE,
  apiFetch,
  formatApiError,
  openRunEventSource,
  SSE_POLL_ACTIVE_MS,
  SSE_POLL_FALLBACK_MS,
} from './api'
import PlanPanel from './PlanPanel'
import FailureReportCard from './FailureReport'
import type { FailureReport } from './failureReportModel'

// Must stay in sync with ``app/main.py`` InputRequest.message max_length.
const INPUT_MAX_CHARS = 4000

// Must stay in sync with ``app/config.py`` -> ``TASK_GOAL_MAX_CHARS`` (backend).
const GOAL_MAX_CHARS = 30000

type Run = {
  run_id: string
  goal: string
  status: string
  attempts?: number
  summary?: string | null
  error?: string | null
  workspace_path?: string | null
  created_at?: string
  updated_at?: string
  state?: {
    pending_input?: { input_id?: string; question?: string } | null
    [key: string]: unknown
  } | null
  failure_report?: FailureReport | null
}

type RunStep = {
  step_id: string
  run_id: string
  seq: number
  ts: string
  kind: string
  tool_name?: string | null
  args?: Record<string, unknown>
  status: string
  output?: string | null
  error?: string | null
}

type WorkspaceView = 'canvas' | 'runs'

// Paused-for-the-user statuses are polled too so the plan view stays live.
const ACTIVE_STATUSES = [
  'running',
  'pending',
  'queued',
  'waiting_for_approval',
  'waiting_for_input',
  'waiting_for_plan_approval',
  'waiting_for_subtask_approval',
]

function isActiveStatus(status: string | undefined) {
  return status != null && ACTIVE_STATUSES.includes(status)
}

const initialNodes: Node[] = [
  {
    id: 'input',
    position: { x: 40, y: 130 },
    data: { label: '任务输入' },
    type: 'input',
  },
  {
    id: 'agent',
    position: { x: 300, y: 130 },
    data: { label: 'Agent 执行器' },
  },
  {
    id: 'result',
    position: { x: 560, y: 130 },
    data: { label: '执行结果' },
    type: 'output',
  },
]


const initialEdges: Edge[] = [
  {
    id: 'e1',
    source: 'input',
    target: 'agent',
    animated: true,
    style: { stroke: '#8d83e8', strokeWidth: 2 },
  },
  {
    id: 'e2',
    source: 'agent',
    target: 'result',
    animated: true,
    style: { stroke: '#8d83e8', strokeWidth: 2 },
  },
]

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    succeeded: '执行成功',
    failed: '执行失败',
    running: '执行中',
    pending: '等待执行',
    queued: '等待执行',
    waiting_for_approval: '等待批准',
    waiting_for_input: '等待补充',
    waiting_for_human: '等待用户',
    waiting_for_plan_approval: '等待批准计划',
    waiting_for_subtask_approval: '等待确认子任务',
    needs_attention: '需要处理',
    interrupted: '已中断',
    cancelled: '已取消',
  }
  return labels[status] ?? status
}

function kindLabel(kind: string) {
  const labels: Record<string, string> = {
    model: '思考',
    tool: '工具',
    approval: '审批',
    input: '补充信息',
    final: '总结',
    workspace: '工作区',
    phase: '阶段',
  }
  return labels[kind] ?? kind
}

function formatDate(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function normalizeRuns(data: unknown): Run[] {
  if (Array.isArray(data)) return data as Run[]
  if (data && typeof data === 'object') {
    const record = data as { runs?: Run[]; items?: Run[] }
    if (Array.isArray(record.runs)) return record.runs
    if (Array.isArray(record.items)) return record.items
  }
  return []
}

function ExecutionLog({ steps }: { steps: RunStep[] }) {
  if (steps.length === 0) {
    return (
      <div className="log-empty">
        <Terminal size={18} />
        <span>暂无执行日志。提交任务后，这里会实时显示 Agent 的每一步操作与工具结果。</span>
      </div>
    )
  }
  return (
    <div className="step-log">
      {steps.map((step) => (
        <div key={step.step_id} className={`step-item step-${step.status}`}>
          <div className="step-head">
            <span className={`step-kind step-kind-${step.kind}`}>{kindLabel(step.kind)}</span>
            {step.tool_name && <span className="step-tool">{step.tool_name}</span>}
            <span className="step-time">{formatDate(step.ts)}</span>
          </div>
          {step.output && <pre className="step-output">{step.output}</pre>}
          {step.error && <div className="step-error">{step.error}</div>}
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [goal, setGoal] = useState('')
  const [runs, setRuns] = useState<Run[]>([])
  const [selectedRun, setSelectedRun] = useState<Run | null>(null)
  /** SSE connection for the selected active run: 'live' | 'poll' | 'off' */
  const [liveStream, setLiveStream] = useState<'live' | 'poll' | 'off'>('off')
  const [activeView, setActiveView] = useState<WorkspaceView>('canvas')
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [approving, setApproving] = useState(false)
  const [inputText, setInputText] = useState('')
  const [sendingInput, setSendingInput] = useState(false)
  const [error, setError] = useState('')
  const [apiOnline, setApiOnline] = useState(false)
  const [steps, setSteps] = useState<RunStep[]>([])
  /** A run-detail fetch is in flight (list rows never carry the plan-backed
   *  failure report), so "report missing" must not be shown yet. */
  const [detailFetching, setDetailFetching] = useState(false)

  const loadRunDetail = useCallback(async (runId: string) => {
    setDetailFetching(true)
    try {
      const response = await apiFetch(`/runs/${runId}`)
      if (!response.ok) return
      const detail = (await response.json()) as Run
      // Ignore a late response for a run the user already navigated away from.
      setSelectedRun((current) => (current?.run_id === runId ? detail : current))
    } catch {
      // 网络错误时保留上一次的详情，下一次轮询会重试。
    } finally {
      setDetailFetching(false)
    }
  }, [])

  const loadRuns = useCallback(async () => {
    setRefreshing(true)
    try {
      const response = await apiFetch('/runs?limit=50')
      if (!response.ok) throw new Error(`获取任务列表失败：HTTP ${response.status}`)
      const data: unknown = await response.json()
      const items = normalizeRuns(data)
      setRuns(items)
      setApiOnline(true)

      setSelectedRun((current) => {
        if (!current) return items[0] ?? null
        const fresh = items.find((run) => run.run_id === current.run_id)
        if (!fresh) return current
        // Keep the detail-only field: the list endpoint omits the plan-backed
        // failure report, so replacing the object wholesale would erase a
        // report the detail fetch already delivered on every poll.
        return { ...fresh, failure_report: fresh.failure_report ?? current.failure_report }
      })
      setError('')
    } catch (err) {
      setApiOnline(false)
      setError(
        err instanceof Error
          ? `${err.message}。请确认后端运行在 ${API_BASE}。`
          : '无法连接后端 API。',
      )
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    // Deferred one tick: the first paint commits before the load starts, so the
    // fetch's setState calls never run synchronously inside the effect body.
    const timer = window.setTimeout(() => void loadRuns(), 0)
    return () => window.clearTimeout(timer)
  }, [loadRuns])

  const loadSteps = useCallback(async (runId: string) => {
    try {
      const response = await apiFetch(`/runs/${runId}/steps`)
      if (!response.ok) return
      const data: unknown = await response.json()
      if (Array.isArray(data)) setSteps(data as RunStep[])
    } catch {
      // 日志拉取失败不打断主流程。
    }
  }, [])

  useEffect(() => {
    const runId = selectedRun?.run_id
    const status = selectedRun?.status

    if (!runId || !isActiveStatus(status)) {
      // No subscription needed. The displayed indicator is derived from status
      // below, so no setState is required here (and none may run synchronously
      // inside an effect body).
      return
    }

    let cancelled = false
    let sseConnected = false
    let es: EventSource | null = null
    let lastSeq = 0
    let reconnectTimer: number | null = null

    const refreshSelectedRun = async () => {
      try {
        const [runResponse, stepsResponse] = await Promise.all([
          apiFetch(`/runs/${runId}`),
          apiFetch(`/runs/${runId}/steps`),
        ])

        if (runResponse.ok) {
          const updatedRun: Run = await runResponse.json()
          if (!cancelled) setSelectedRun(updatedRun)
        }
        if (stepsResponse.ok) {
          const stepsData: unknown = await stepsResponse.json()
          if (!cancelled && Array.isArray(stepsData)) setSteps(stepsData as RunStep[])
        }
        if (!cancelled) await loadRuns()
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '自动刷新任务失败')
        }
      }
    }

    void refreshSelectedRun()

    const schedulePoll = () => {
      const ms = sseConnected ? SSE_POLL_ACTIVE_MS : SSE_POLL_FALLBACK_MS
      return window.setInterval(() => {
        void refreshSelectedRun()
      }, ms)
    }

    let timer = schedulePoll()

    const restartPoll = () => {
      window.clearInterval(timer)
      timer = schedulePoll()
    }

    const attachEventSource = () => {
      if (cancelled) return
      if (es) {
        try {
          es.close()
        } catch {
          /* ignore */
        }
        es = null
      }
      es = openRunEventSource(
        runId,
        (data) => {
          if (cancelled) return
          const seq = Number(data.seq)
          if (Number.isFinite(seq) && seq > lastSeq) lastSeq = seq
          const t = String(data.type || '')
          if (t === 'connected') {
            sseConnected = true
            setLiveStream('live')
            restartPoll()
            return
          }
          if (
            t === 'status' ||
            t === 'step' ||
            t === 'approval' ||
            t === 'input' ||
            t === 'state' ||
            t === 'plan'
          ) {
            void refreshSelectedRun()
          }
        },
        () => {
          if (cancelled) return
          sseConnected = false
          setLiveStream('poll')
          restartPoll()
          // Manual reconnect with after_seq so restart/missed events replay.
          if (reconnectTimer != null) window.clearTimeout(reconnectTimer)
          reconnectTimer = window.setTimeout(() => {
            attachEventSource()
          }, 1500)
        },
        () => {
          if (cancelled) return
          sseConnected = true
          setLiveStream('live')
          restartPoll()
        },
        lastSeq,
      )
      if (!es) setLiveStream('poll')
    }

    attachEventSource()

    return () => {
      cancelled = true
      window.clearInterval(timer)
      if (reconnectTimer != null) window.clearTimeout(reconnectTimer)
      if (es) {
        try {
          es.close()
        } catch {
          /* ignore */
        }
      }
      setLiveStream('off')
    }
  }, [selectedRun?.run_id, selectedRun?.status, loadRuns])

  // When a run reaches a terminal state, fetch the final authoritative snapshot
  // once: steps, plus the run detail (which carries the plan-backed failure
  // report) so a just-finished failure is never displayed from a stale list row.
  useEffect(() => {
    const runId = selectedRun?.run_id
    const status = selectedRun?.status
    const settled = ['succeeded', 'failed', 'needs_attention', 'cancelled', 'interrupted']
    if (!runId || !status || !settled.includes(status)) return
    const timer = window.setTimeout(() => {
      void loadRunDetail(runId)
      void loadSteps(runId)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [selectedRun?.run_id, selectedRun?.status, loadRunDetail, loadSteps])

  // A failed run selected from the list (or auto-selected at startup) has no
  // report yet because list rows never carry it. Fetch the detail once so the
  // report appears even on a terminal run that no longer polls.
  const hasFailureReport = Boolean(selectedRun?.failure_report)
  useEffect(() => {
    const runId = selectedRun?.run_id
    if (!runId || selectedRun?.status !== 'failed' || hasFailureReport) return
    const timer = window.setTimeout(() => void loadRunDetail(runId), 0)
    return () => window.clearTimeout(timer)
  }, [selectedRun?.run_id, selectedRun?.status, hasFailureReport, loadRunDetail])

  const submitRun = async () => {
    if (loading) return
    const trimmedGoal = goal.trim()

    // Guard against over-limit input (the textarea ``maxLength`` also caps
    // typing/paste, but this is the authoritative client-side check that never
    // silently truncates the user's text).
    if (goal.length > GOAL_MAX_CHARS) {
      setError(`任务目标过长：最多 ${GOAL_MAX_CHARS} 字符，当前 ${goal.length} 字符，请精简后再提交。`)
      return
    }
    if (!trimmedGoal) return

    setLoading(true)
    setError('')

    try {
      const response = await apiFetch('/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          goal: trimmedGoal,
          max_attempts: 2,
        }),
      })

      const data = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(
          formatApiError(data?.detail, `提交任务失败：HTTP ${response.status}`),
        )
      }

      const run = data as Run
      setSelectedRun(run)
      setGoal('')
      void loadSteps(run.run_id)
      await loadRuns()
      setSelectedRun(run)
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message}。如果浏览器提示 CORS，请告诉我，我们会配置后端跨域。`
          : '提交任务时发生未知错误。',
      )
    } finally {
      setLoading(false)
    }
  }

  const selectRun = (run: Run) => {
    setSelectedRun(run)
    void loadSteps(run.run_id)
    void loadRunDetail(run.run_id)
  }

  const approveRun = async (approved: boolean) => {
    const runId = selectedRun?.run_id
    if (!runId || approving) return
    setApproving(true)
    setError('')
    try {
      const response = await apiFetch(`/runs/${runId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      })
      const data = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(formatApiError(data?.detail, `审批失败：HTTP ${response.status}`))
      }
      setSelectedRun(data as Run)
      await loadRuns()
      void loadSteps(runId)
    } catch (err) {
      setError(err instanceof Error ? err.message : '审批失败')
    } finally {
      setApproving(false)
    }
  }

  const submitInput = async () => {
    const runId = selectedRun?.run_id
    const message = inputText.trim()
    if (!runId || sendingInput || !message) return
    setSendingInput(true)
    setError('')
    try {
      const response = await apiFetch(`/runs/${runId}/input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
      const data = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(formatApiError(data?.detail, `提交回复失败：HTTP ${response.status}`))
      }
      setInputText('')
      setSelectedRun(data as Run)
      await loadRuns()
      void loadSteps(runId)
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交回复失败')
    } finally {
      setSendingInput(false)
    }
  }

  const refreshSelected = useCallback(() => {
    const runId = selectedRun?.run_id
    if (!runId) return
    void loadRunDetail(runId)
    void loadSteps(runId)
    void loadRuns()
  }, [selectedRun?.run_id, loadRunDetail, loadSteps, loadRuns])

  const currentStatus = selectedRun?.status
  // Derived, not stored: an inactive run has no live subscription regardless of
  // what the last connection attempt left in state.
  const liveStreamShown = isActiveStatus(currentStatus) ? liveStream : 'off'

  const isRunning = isActiveStatus(currentStatus)
  const isWaitingApproval = currentStatus === 'waiting_for_approval'
  const isWaitingInput = currentStatus === 'waiting_for_input'
  const pendingQuestion = selectedRun?.state?.pending_input?.question ?? ''

  const nodes: Node[] = initialNodes.map((node) => {

      let label: string
      let className = 'flow-node'
  
      if (node.id === 'input') {
        label = selectedRun ? '任务已提交' : '等待输入'
        if (selectedRun) className += ' node-success'
      } else if (node.id === 'agent') {
        if (isWaitingApproval) {
          label = '等待批准'
          className += ' node-active'
        } else if (isWaitingInput) {
          label = '等待补充信息'
          className += ' node-active'
        } else if (isRunning || loading) {
          label = 'Agent 执行中…'
          className += ' node-active'
        } else if (currentStatus === 'succeeded') {
          label = 'Agent 执行成功'
          className += ' node-success'
        } else if (currentStatus === 'failed') {
          label = 'Agent 执行失败'
          className += ' node-failed'
        } else {
          label = '等待执行'
        }
      } else {
        if (currentStatus === 'succeeded') {
          label = '执行成功'
          className += ' node-success'
        } else if (currentStatus === 'failed') {
          label = '执行失败'
          className += ' node-failed'
        } else {
          label = '等待执行结果'
        }
      }
  
      return {
        ...node,
        data: { ...node.data, label },
        className,
      }
    })
  
  const edges: Edge[] = initialEdges.map((edge) => {
    const status = selectedRun?.status

    const isRunning = isActiveStatus(status)

    const isSuccess = status === 'succeeded'
    const isFailed = status === 'failed'

    const edgeFinished = edge.id === 'e1' || isSuccess || isFailed

    return {
      ...edge,
      animated: isRunning,
      style: {
        stroke: isFailed
          ? '#ef7770'
          : isSuccess && edgeFinished
            ? '#59c89b'
            : isRunning
              ? '#8a7bf0'
              : '#c5cad8',
        strokeWidth: isRunning || isSuccess || isFailed ? 2.5 : 2,
      },
    }
  })

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Workflow size={22} /></div>
          <div>
            <div className="brand-name">Local Agent</div>
            <div className="brand-subtitle">WORKFLOW STUDIO</div>
          </div>
        </div>

        <div className="sidebar-section-label">工作区</div>
        <button
          type="button"
          className={`nav-item ${activeView === 'canvas' ? 'nav-item-active' : ''}`}
          aria-current={activeView === 'canvas' ? 'page' : undefined}
          onClick={() => setActiveView('canvas')}
        >
          <GitBranch size={18} />
          <span>工作流画布</span>
        </button>
        <button
          type="button"
          className={`nav-item ${activeView === 'runs' ? 'nav-item-active' : ''}`}
          aria-current={activeView === 'runs' ? 'page' : undefined}
          onClick={() => setActiveView('runs')}
        >
          <Activity size={18} />
          <span>运行记录</span>
          <span className="nav-count">{runs.length}</span>
        </button>

        <div className="sidebar-bottom">
          <div className="connection-card">
            <span className={`status-dot ${apiOnline ? 'online' : ''}`} />
            <div>
              <strong>{apiOnline ? '后端已连接' : '检查后端连接'}</strong>
              <small>{API_BASE.replace(/^https?:\/\//, '')}</small>
            </div>
          </div>
          <div className="sidebar-version">本地开发版 · v0.3.6</div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              工作区 <span>/</span> {activeView === 'canvas' ? '工作流画布' : '运行记录'}
            </div>
            <h1>{activeView === 'canvas' ? '工作流控制台' : '运行记录'}</h1>
          </div>
          <button
            className="button button-secondary"
            onClick={() => void loadRuns()}
            disabled={refreshing}
          >
            <RefreshCw size={16} className={refreshing ? 'spin' : ''} />
            刷新数据
          </button>
        </header>

        {error && (
          <div className="error-banner">
            <CircleAlert size={18} />
            <span>{error}</span>
          </div>
        )}

        {isWaitingApproval && (
          <div className="approval-banner">
            <ShieldAlert size={18} />
            <div className="approval-text">
              <strong>Agent 正在等待你的批准</strong>
              <p>检测到高风险操作（如安装依赖或覆盖受保护文件）。请确认是否继续。</p>
            </div>
            <button
              className="button button-primary"
              onClick={() => void approveRun(true)}
              disabled={approving}
            >
              <Check size={16} /> 批准
            </button>
            <button
              className="button button-secondary"
              onClick={() => void approveRun(false)}
              disabled={approving}
            >
              <X size={16} /> 拒绝
            </button>
          </div>
        )}

        {isWaitingInput && (
          <div className="approval-banner input-banner">
            <MessageSquare size={18} />
            <div className="approval-text">
              <strong>Agent 需要你补充信息</strong>
              <p>{pendingQuestion || 'Agent 正在等待你的回复。'}</p>
              <textarea
                className="input-reply"
                value={inputText}
                maxLength={INPUT_MAX_CHARS}
                rows={2}
                placeholder="输入你的回复…（Ctrl+Enter 发送）"
                onChange={(event) => setInputText(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                    event.preventDefault()
                    void submitInput()
                  }
                }}
                disabled={sendingInput}
              />
            </div>
            <button
              className="button button-primary"
              onClick={() => void submitInput()}
              disabled={sendingInput || !inputText.trim()}
            >
              <Send size={16} /> 发送
            </button>
          </div>
        )}

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon purple"><Workflow size={19} /></div>
            <div><span>任务总数</span><strong>{runs.length}</strong></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green"><CheckCircle2 size={19} /></div>
            <div><span>执行成功</span><strong>{runs.filter((r) => r.status === 'succeeded').length}</strong></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue"><Activity size={19} /></div>
            <div><span>执行中 / 等待</span><strong>{runs.filter((r) => isActiveStatus(r.status)).length}</strong></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon orange"><CircleAlert size={19} /></div>
            <div><span>执行失败</span><strong>{runs.filter((r) => r.status === 'failed').length}</strong></div>
          </div>
        </section>

        {activeView === 'canvas' ? (
        <section className="workspace-grid">
          <div className="workspace-main">
            <div className="section-heading">
              <div>
                <h2>任务图</h2>
                <p>每个目标拆解出的子任务 DAG，按状态着色；点击节点查看详情</p>
              </div>
              <span className="demo-tag">TASK GRAPH</span>
            </div>

            {selectedRun ? (
              <PlanPanel
                key={selectedRun.run_id}
                runId={selectedRun.run_id}
                runStatus={selectedRun.status}
                onChanged={refreshSelected}
                fallback={
                <div className="canvas-wrap">
                  <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    fitView
                    fitViewOptions={{ padding: 0.25 }}
                    nodesDraggable
                    nodesConnectable={false}
                    elementsSelectable
                  >
                    <Background color="#dfe4ed" gap={20} />
                    <Controls />
                    <MiniMap zoomable pannable />
                  </ReactFlow>
                  <div className="canvas-caption">
                    <span className="status-dot online" /> 任务输入
                    <span className="caption-line" /> Agent 执行
                    <span className="caption-line" /> 输出结果
                  </div>
                </div>
                }
              />
            ) : (
            <div className="canvas-wrap">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                fitView
                fitViewOptions={{ padding: 0.25 }}
                nodesDraggable
                nodesConnectable={false}
                elementsSelectable
              >
                <Background color="#dfe4ed" gap={20} />
                <Controls />
                <MiniMap zoomable pannable />
              </ReactFlow>
              <div className="canvas-caption">
                <span className="status-dot online" /> 任务输入
                <span className="caption-line" /> Agent 执行
                <span className="caption-line" /> 输出结果
              </div>
            </div>
            )}

            <div className="section-heading history-heading">
              <div>
                <h2>最近运行</h2>
                <p>从本地后端加载的任务记录</p>
              </div>
              <button className="text-button" onClick={() => void loadRuns()}>查看最新数据</button>
            </div>

            <div className="runs-table">
              <div className="runs-header">
                <span>任务目标</span><span>状态</span><span>更新时间</span>
              </div>
              {runs.length === 0 ? (
                <div className="empty-state">
                  <Clock3 size={25} />
                  <strong>还没有运行记录</strong>
                  <span>在右侧输入任务目标，创建你的第一个工作流。</span>
                </div>
              ) : (
                runs.slice(0, 8).map((run) => (
                  <button
                    className={`run-row ${selectedRun?.run_id === run.run_id ? 'run-row-selected' : ''}`}
                    key={run.run_id}
                    onClick={() => void selectRun(run)}
                  >
                    <span className="run-goal">{run.goal}</span>
                    <span className={`status-pill status-${run.status}`}>{statusLabel(run.status)}</span>
                    <span className="run-date">{formatDate(run.updated_at)}</span>
                  </button>
                ))
              )}
            </div>

            <div className="section-heading log-heading">
              <div>
                <h2>Agent 执行日志</h2>
                <p>实时记录当前任务的思考、工具调用与结果</p>
              </div>
            </div>
            <div className="log-panel">
              <ExecutionLog steps={steps} />
            </div>
          </div>

          <aside className="right-panel">
            <section className="panel-card">
              <div className="panel-card-heading">
                <div className="panel-icon"><Play size={17} /></div>
                <div><h2>启动新任务</h2><p>描述你希望 Agent 完成的目标</p></div>
              </div>
              <label htmlFor="goal">任务目标</label>
              <textarea
                id="goal"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                placeholder="例如：帮我检查 D:\PythonProject\an 项目，修复 FAL 自动化脚本的三个稳定性 BUG。先阅读相关代码，分析问题，完成修改并运行离线测试。"
                rows={5}
                maxLength={GOAL_MAX_CHARS}
              />
              <div className={`input-hint${goal.length > GOAL_MAX_CHARS ? ' input-hint-over' : ''}`}>
                {goal.length} / {GOAL_MAX_CHARS} 字符
              </div>
              <div className="input-hint">
                在任务目标中写明目标项目的绝对路径（例如 D:\PythonProject\an），Agent 会自动识别并在该项目内执行；未指定则使用默认工作区。
              </div>
              <div className="setting-row">
                <span>最大尝试次数</span>
                <strong>2 次</strong>
              </div>
              <div className="executor-note">
                <span className="status-dot online" />
                当前执行器：DeepSeek Agent
              </div>
              <button
                className="button button-primary submit-button"
                onClick={() => void submitRun()}
                disabled={!goal.trim() || goal.length > GOAL_MAX_CHARS || loading}
              >
                <Play size={17} />
                {loading ? '正在提交…' : '运行工作流'}
              </button>
              <p className="disclaimer">
                Agent 通过 DeepSeek 模型调用工具，实际读取文件、运行命令并验证结果；高风险操作需要你批准。
              </p>
            </section>

            <section className="panel-card detail-card">
              <div className="panel-card-heading">
                <div className="panel-icon"><Activity size={17} /></div>
                <div><h2>任务详情</h2><p>当前选中的运行记录</p></div>
              </div>
              {selectedRun ? (
                <div className="detail-content">
                  <div className="detail-status">
                    <span className={`status-pill status-${selectedRun.status}`}>{statusLabel(selectedRun.status)}</span>
                    {liveStreamShown !== 'off' && (
                      <span className={`live-hint live-${liveStreamShown}`} title={liveStreamShown === 'live' ? 'Server-Sent Events 已连接' : '轮询刷新（SSE 未连接）'}>
                        {liveStreamShown === 'live' ? '实时' : '轮询'}
                      </span>
                    )}
                  </div>
                  <div className="detail-field"><span>任务目标</span><p>{selectedRun.goal}</p></div>
                  <div className="detail-field"><span>运行 ID</span><code>{selectedRun.run_id}</code></div>
                  <div className="detail-field"><span>目标项目</span><code>{selectedRun.workspace_path || '默认工作区（未在目标中指定）'}</code></div>
                  <div className="detail-field"><span>尝试次数</span><p>{selectedRun.attempts ?? '—'}</p></div>
                  <div className="detail-field"><span>执行摘要</span><p>{selectedRun.summary || '暂无摘要'}</p></div>
                  {selectedRun.error && <div className="detail-error">{selectedRun.error}</div>}
                  <FailureReportCard report={selectedRun.failure_report} runStatus={selectedRun.status} loading={detailFetching} />
                  <div className="detail-field"><span>更新时间</span><p>{formatDate(selectedRun.updated_at)}</p></div>
                </div>
              ) : (
                <div className="detail-empty">选择一条运行记录以查看详情。</div>
              )}
            </section>
          </aside>
        </section>
        ) : (
        <section className="workspace-grid runs-view">
          <div className="workspace-main">
            <div className="section-heading">
              <div>
                <h2>运行记录</h2>
                <p>从本地后端加载的全部任务记录</p>
              </div>
              <button className="text-button" onClick={() => void loadRuns()}>查看最新数据</button>
            </div>

            <div className="runs-table">
              <div className="runs-header">
                <span>任务目标</span><span>状态</span><span>更新时间</span>
              </div>
              {runs.length === 0 ? (
                <div className="empty-state">
                  <Clock3 size={25} />
                  <strong>还没有运行记录</strong>
                  <span>回到工作流画布，输入任务目标后即可创建记录。</span>
                </div>
              ) : (
                runs.map((run) => (
                  <button
                    className={`run-row ${selectedRun?.run_id === run.run_id ? 'run-row-selected' : ''}`}
                    key={run.run_id}
                    onClick={() => void selectRun(run)}
                  >
                    <span className="run-goal">{run.goal}</span>
                    <span className={`status-pill status-${run.status}`}>{statusLabel(run.status)}</span>
                    <span className="run-date">{formatDate(run.updated_at)}</span>
                  </button>
                ))
              )}
            </div>

            <div className="section-heading log-heading">
              <div>
                <h2>Agent 执行日志</h2>
                <p>该任务的思考、工具调用与结果</p>
              </div>
            </div>
            <div className="log-panel">
              <ExecutionLog steps={steps} />
            </div>
          </div>

          <aside className="right-panel">
            <section className="panel-card detail-card">
              <div className="panel-card-heading">
                <div className="panel-icon"><Activity size={17} /></div>
                <div><h2>任务详情</h2><p>点击左侧记录查看完整信息</p></div>
              </div>
              {selectedRun ? (
                <div className="detail-content">
                  <div className="detail-status">
                    <span className={`status-pill status-${selectedRun.status}`}>{statusLabel(selectedRun.status)}</span>
                    {liveStreamShown !== 'off' && (
                      <span className={`live-hint live-${liveStreamShown}`} title={liveStreamShown === 'live' ? 'Server-Sent Events 已连接' : '轮询刷新（SSE 未连接）'}>
                        {liveStreamShown === 'live' ? '实时' : '轮询'}
                      </span>
                    )}
                  </div>
                  <div className="detail-field"><span>任务目标</span><p>{selectedRun.goal}</p></div>
                  <div className="detail-field"><span>运行 ID</span><code>{selectedRun.run_id}</code></div>
                  <div className="detail-field"><span>目标项目</span><code>{selectedRun.workspace_path || '默认工作区（未在目标中指定）'}</code></div>
                  <div className="detail-field"><span>尝试次数</span><p>{selectedRun.attempts ?? '—'}</p></div>
                  <div className="detail-field"><span>执行摘要</span><p>{selectedRun.summary || '暂无摘要'}</p></div>
                  {selectedRun.error && <div className="detail-error">{selectedRun.error}</div>}
                  <FailureReportCard report={selectedRun.failure_report} runStatus={selectedRun.status} loading={detailFetching} />
                  <div className="detail-field"><span>更新时间</span><p>{formatDate(selectedRun.updated_at)}</p></div>
                </div>
              ) : (
                <div className="detail-empty">选择一条运行记录以查看详情。</div>
              )}
            </section>
          </aside>
        </section>
        )}

        <footer className="app-footer">
          <span>Local Agent Workflow</span>
          <span><span className={`status-dot ${apiOnline ? 'online' : ''}`} /> API {apiOnline ? '已连接' : '未连接'}</span>
        </footer>
      </section>
    </main>
  )
}
