// Backend base URL and optional shared token, from Vite env (``frontend/.env``
// or ``.env.local``): VITE_API_BASE / VITE_AGENT_TOKEN. The token must match the
// backend's LOCAL_AGENT_TOKEN; leave it empty when backend auth is disabled.
export const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000').replace(/\/+$/, '')
const AGENT_TOKEN = import.meta.env.VITE_AGENT_TOKEN || ''

export function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  if (AGENT_TOKEN) headers.set('Authorization', `Bearer ${AGENT_TOKEN}`)
  return fetch(`${API_BASE}${path}`, { ...init, headers })
}

export function formatApiError(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg: unknown }).msg).replace(/^Value error,\s*/, '')
        }
        return ''
      })
      .filter((message) => message.length > 0)
    if (messages.length > 0) return messages.join('；')
  }
  return fallback
}

/** POST/PATCH JSON and throw a readable error on non-2xx. */
export async function sendJson<T>(path: string, method: string, body?: unknown): Promise<T> {
  const response = await apiFetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(formatApiError(data?.detail, `请求失败：HTTP ${response.status}`))
  }
  return data as T
}

/** Poll intervals (ms). Prefer SSE; use ACTIVE when EventSource is healthy. */
export const SSE_POLL_FALLBACK_MS = 2000
export const SSE_POLL_ACTIVE_MS = 8000

/**
 * Open an SSE stream for one run.
 * Pass ``afterSeq`` so reconnects replay durable ``run_events`` after a restart.
 * EventSource cannot set Authorization headers — when VITE_AGENT_TOKEN is set
 * we pass ?access_token= (backend require_token accepts it).
 */
export function openRunEventSource(
  runId: string,
  onEvent: (data: Record<string, unknown>) => void,
  onError?: () => void,
  onOpen?: () => void,
  afterSeq?: number,
): EventSource | null {
  if (typeof EventSource === 'undefined') return null
  const params = new URLSearchParams()
  if (AGENT_TOKEN) params.set('access_token', AGENT_TOKEN)
  if (afterSeq != null && afterSeq > 0) params.set('after_seq', String(afterSeq))
  const qs = params.toString() ? `?${params.toString()}` : ''
  const url = `${API_BASE}/runs/${encodeURIComponent(runId)}/events${qs}`
  const es = new EventSource(url)
  es.onopen = () => {
    onOpen?.()
  }
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data) as Record<string, unknown>
      onEvent(data)
    } catch {
      /* ignore malformed frames */
    }
  }
  es.onerror = () => {
    onError?.()
  }
  return es
}
