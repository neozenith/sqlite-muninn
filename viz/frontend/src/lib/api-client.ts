/**
 * Typed wrappers around the muninn-viz backend API.
 *
 * This file is the SINGLE code path that talks to the backend. Pages and
 * components must not call `fetch` directly — go through these functions so
 * types, error handling, and URL construction stay in one place.
 */

export interface DatabaseInfo {
  id: string
  book_id: number
  model: string
  dim: number
  file: string
  size_bytes: number
  label: string
}

export interface HealthResponse {
  status: string
}

/**
 * Thrown on any non-2xx response. `status` is the HTTP code, `body` is the
 * response body (usually `{"detail": "..."}` from FastAPI).
 */
export class ApiError extends Error {
  status: number
  body: string

  constructor(status: number, body: string) {
    super(`API error ${status}: ${body}`)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

const API_BASE = '/api'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const body = await response.text().catch(() => '')
    throw new ApiError(response.status, body)
  }
  return (await response.json()) as T
}

/** GET /api/health — returns `{status: 'ok'}` when the backend is alive. */
export async function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/health')
}

/** GET /api/databases — returns every database entry in manifest order. */
export async function fetchDatabases(): Promise<DatabaseInfo[]> {
  const body = await getJson<{ databases: DatabaseInfo[] }>('/databases')
  return body.databases
}

/**
 * GET /api/databases/:id — returns a single database's metadata.
 * Throws ApiError(404, ...) if the id is unknown.
 */
export async function fetchDatabase(id: string): Promise<DatabaseInfo> {
  return getJson<DatabaseInfo>(`/databases/${encodeURIComponent(id)}`)
}
