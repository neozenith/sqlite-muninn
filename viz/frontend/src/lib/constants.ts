/** Application constants — colors, defaults, config. */

/** Community color palette (10 distinct colors for Leiden communities). */
export const COMMUNITY_COLORS: readonly string[] = [
  '#4e79a7',
  '#f28e2b',
  '#e15759',
  '#76b7b2',
  '#59a14f',
  '#edc948',
  '#b07aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ac',
] as const

/** Get a color for a community ID (wraps around if > 10 communities). */
export const communityColor = (id: number): string => COMMUNITY_COLORS[id % COMMUNITY_COLORS.length]

/** Default HNSW search parameters. */
export const DEFAULT_K = 20
export const DEFAULT_EF_SEARCH = 64

/** Default graph parameters. */
export const DEFAULT_MAX_DEPTH = 3
export const DEFAULT_RESOLUTION = 1.0
export const DEFAULT_SUBGRAPH_LIMIT = 500

/** Default UMAP projection dimensions. */
export const DEFAULT_PROJECTION_DIMS = 2

/** Search highlight color (orange). */
export const SEARCH_HIGHLIGHT_COLOR: [number, number, number] = [255, 140, 0]

/** Background point color (muted blue). */
export const BACKGROUND_POINT_COLOR: [number, number, number] = [100, 149, 237]

/** Selected point color (red). */
export const SELECTED_POINT_COLOR: [number, number, number] = [255, 50, 50]

/** Available Cytoscape layout algorithms. */
export const LAYOUT_OPTIONS = [
  { value: 'fcose', label: 'Force-directed (fCoSE)' },
  { value: 'cose', label: 'Force-directed (CoSE)' },
  { value: 'circle', label: 'Circle' },
  { value: 'grid', label: 'Grid' },
  { value: 'concentric', label: 'Concentric' },
  { value: 'breadthfirst', label: 'Breadth-first' },
  { value: 'random', label: 'Random' },
] as const

export type LayoutName = (typeof LAYOUT_OPTIONS)[number]['value']

