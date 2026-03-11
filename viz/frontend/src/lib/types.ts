/** All TypeScript interfaces for the muninn-viz application. */

// ── Health ──────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string
  db_path: string
  db_exists: boolean
  extension_path: string
  extension_loaded: boolean
  active_database: string | null
  hnsw_index_count?: number
  edge_table_count?: number
  error?: string
}

// ── Databases ──────────────────────────────────────────────────────────

export interface DatabaseInfo {
  id: string
  book_id: number
  model: string
  dim: number
  file: string
  size_bytes: number
  label: string
}

export interface DatabasesResponse {
  databases: DatabaseInfo[]
  active: string | null
}

// ── VSS ─────────────────────────────────────────────────────────────

export interface HnswIndexInfo {
  name: string
  dimensions: number
  metric: string
  m: number
  ef_construction: number
  node_count: number
}

export interface EmbeddingPoint {
  id: number
  x: number
  y: number
  z?: number
  label: string
  metadata: Record<string, unknown>
  /** Optional per-point color override (RGB tuple). Used by KG query rank gradient. */
  color?: readonly [number, number, number]
}

export interface EmbeddingsResponse {
  index: string
  count: number
  original_dimensions: number
  projected_dimensions: number
  points: EmbeddingPoint[]
}

export interface SearchNeighbor {
  id: number
  distance: number
  label: string
  metadata: Record<string, unknown>
}

export interface SearchResponse {
  index: string
  query_id: number
  k: number
  count: number
  neighbors: SearchNeighbor[]
}

// ── Graph ───────────────────────────────────────────────────────────

export interface EdgeTableInfo {
  table_name: string
  src_col: string
  dst_col: string
  weight_col: string | null
  edge_count: number
}

export interface GraphNode {
  id: string
  label: string
  mention_count?: number
  entity_type?: string
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
  rel_type?: string
}

export interface SubgraphResponse {
  edge_table: string
  node_count: number
  edge_count: number
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface BfsNode {
  node: string
  depth: number
}

export interface BfsResponse {
  edge_table: string
  start_node: string
  max_depth: number
  direction: string
  count: number
  nodes: BfsNode[]
}

export interface CommunitiesResponse {
  edge_table: string
  resolution: number
  community_count: number
  node_count: number
  communities: Record<string, string[]>
  node_community: Record<string, number>
}

export interface CentralityScore {
  node: string
  centrality: number
}

export interface CentralityResponse {
  edge_table: string
  measure: string
  direction: string
  count: number
  scores: CentralityScore[]
}

// ── KG Search ──────────────────────────────────────────────────────

export interface KGSearchFtsResult {
  chunk_id: number
  text: string
}

export interface KGSearchVssResult {
  chunk_id: number
  similarity: number
  distance: number
  text: string
  x3d: number | null
  y3d: number | null
  z3d: number | null
}

export interface KGSearchGraphNode {
  name: string
  depth: number
  similarity: number
  is_anchor: boolean
}

export interface KGSearchGraphEdge {
  src: string
  rel: string
  dst: string
}

export interface KGSearchResult {
  query: string
  fts_results: KGSearchFtsResult[]
  vss_results: KGSearchVssResult[]
  graph_nodes: KGSearchGraphNode[]
  graph_edges: KGSearchGraphEdge[]
  node_community?: Record<string, number>
  community_count?: number
}

// ── UI State ────────────────────────────────────────────────────────

export type CentralityMeasure = 'degree' | 'betweenness' | 'closeness'

export type Direction = 'out' | 'in' | 'both'
