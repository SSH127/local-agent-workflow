/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL, default http://127.0.0.1:8000 */
  readonly VITE_API_BASE?: string
  /** Shared token; must match the backend LOCAL_AGENT_TOKEN (empty = no auth) */
  readonly VITE_AGENT_TOKEN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
