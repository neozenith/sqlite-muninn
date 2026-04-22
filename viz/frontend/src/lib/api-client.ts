/**
 * Typed wrappers around the muninn-viz backend API.
 *
 * This file is the SINGLE code path that talks to the backend. Pages and
 * components must not call `fetch` directly — go through these functions so
 * types, error handling, and URL construction stay in one place.
 */

export interface DatabaseInfo {
  id: string
  /** Project Gutenberg book id. Absent for session-log / non-book demos. */
  book_id: number | null
  model: string
  dim: number
  file: string
  size_bytes: number
  label: string
}

export interface HealthResponse {
  status: string
}

/** Which viz tables are available for a specific database. */
export interface TablesResponse {
  database_id: string
  embed_tables: string[]
  kg_tables: string[]
  resolutions: number[]
}

/** A single point in the 3D UMAP space. */
export interface EmbedPoint {
  id: number
  x: number
  y: number
  z: number
  label: string
  category: string | null
}

export interface EmbedPayload {
  table_id: string
  count: number
  points: EmbedPoint[]
}

export interface KGNode {
  id: string
  label: string
  entity_type: string | null
  community_id: number | null
  mention_count: number | null
  /** Betweenness centrality over the FULL graph, not the filtered subset. */
  node_betweenness: number | null
}

export interface KGEdge {
  source: string
  target: string
  rel_type: string | null
  weight: number | null
  /** Betweenness centrality over the FULL graph, not the filtered subset. */
  edge_betweenness: number | null
}

export interface KGCommunity {
  id: number
  label: string | null
  member_count: number
  node_ids: string[]
}

export type SeedMetric = 'degree' | 'node_betweenness' | 'edge_betweenness'

export interface KGPayload {
  table_id: string
  resolution: number
  seed_metric: SeedMetric
  max_depth: number
  node_count: number
  edge_count: number
  community_count: number
  /** Total nodes in the DB before top-N filtering (>= node_count). */
  total_node_count: number
  /** Total edges in the DB before top-N filtering (>= edge_count). */
  total_edge_count: number
  nodes: KGNode[]
  edges: KGEdge[]
  communities: KGCommunity[]
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

/** GET /api/databases/:id/tables — which viz tables exist for this database. */
export async function fetchTables(id: string): Promise<TablesResponse> {
  return getJson<TablesResponse>(`/databases/${encodeURIComponent(id)}/tables`)
}

/**
 * GET /api/databases/:id/embed/:table_id — 3D UMAP points for Deck.GL.
 * `tableId` is one of {'chunks', 'entities'}.
 */
export async function fetchEmbed(
  databaseId: string,
  tableId: string,
): Promise<EmbedPayload> {
  return getJson<EmbedPayload>(
    `/databases/${encodeURIComponent(databaseId)}/embed/${encodeURIComponent(tableId)}`,
  )
}

/**
 * GET /api/databases/:id/kg/:table_id — KG payload for Cytoscape.
 *
 * `topN` picks the N highest-scoring seed nodes by `seedMetric` (default
 * edge_betweenness). The backend then BFS-expands from those seeds through
 * the undirected edge view up to `maxDepth` hops — `maxDepth=0` is
 * unlimited (union of connected components containing a seed).
 */
export async function fetchKG(
  databaseId: string,
  tableId: string,
  options: {
    resolution?: number
    topN?: number
    seedMetric?: SeedMetric
    maxDepth?: number
  } = {},
): Promise<KGPayload> {
  const params = new URLSearchParams()
  if (options.resolution !== undefined) params.set('resolution', String(options.resolution))
  if (options.topN !== undefined) params.set('top_n', String(options.topN))
  if (options.seedMetric !== undefined) params.set('seed_metric', options.seedMetric)
  if (options.maxDepth !== undefined) params.set('max_depth', String(options.maxDepth))
  const query = params.toString()
  return getJson<KGPayload>(
    `/databases/${encodeURIComponent(databaseId)}/kg/${encodeURIComponent(tableId)}${
      query ? `?${query}` : ''
    }`,
  )
}
