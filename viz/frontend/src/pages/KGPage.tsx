import cytoscape from 'cytoscape'
// cytoscape-fcose doesn't ship a perfect type for its default export — the
// cast below hands the plugin to cytoscape.use() without relaxing strict mode.
// cytoscape-svg adds `cy.svg(opts)` for SVG export.
import fcose from 'cytoscape-fcose'
import cytoscapeSvg from 'cytoscape-svg'
import { useEffect, useMemo, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import { Link, useParams } from 'react-router-dom'
import { PanelSection, RightPanel } from '../components/RightPanel'
import {
  ApiError,
  type KGCommunity,
  type KGEdge,
  type KGNode,
  type KGPayload,
  type SeedMetric,
  fetchKG,
} from '../lib/api-client'
import { useTheme } from '../lib/theme-context'

cytoscape.use(fcose as unknown as cytoscape.Ext)
cytoscape.use(cytoscapeSvg as unknown as cytoscape.Ext)

type LoadState = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; payload: KGPayload }

type SizeMode = 'uniform' | 'degree' | 'betweenness'
type NodeColorMode = 'uniform' | 'entity_type'
type EdgeColorMode = 'uniform' | 'rel_type'
type EdgeThicknessMode = 'uniform' | 'weight' | 'edge_betweenness'

const DEFAULT_TOP_N = 2
const DEFAULT_MAX_DEPTH = 1
const DEFAULT_SEED_METRIC: SeedMetric = 'edge_betweenness'
const DEFAULT_MIN_DEGREE = 1
const DEFAULT_COMMUNITY_OPACITY = 0.2
const DEFAULT_RELATIVE_PLACEMENT_GAP = 80
const DEFAULT_SIZE_MODE: SizeMode = 'betweenness'
/**
 * Debounce window for auto-applying any user input that triggers either a
 * data refetch or a fcose layout re-run. Tuned to feel "instant" on a
 * single click but to coalesce slider scrubbing into one final apply.
 */
const AUTO_APPLY_DEBOUNCE_MS = 250

interface KGSelection {
  nodes: Set<string>
  edges: Set<string>
  communities: Set<number>
}

const EMPTY_SELECTION: KGSelection = {
  nodes: new Set(),
  edges: new Set(),
  communities: new Set(),
}

const DEFAULT_FCOSE_CONFIG = {
  quality: 'default',
  randomize: true,
  animate: false,
  animationDuration: 800,
  fit: true,
  padding: 30,
  nodeRepulsion: 4500,
  idealEdgeLength: 50,
  edgeElasticity: 0.45,
  nestingFactor: 0.1,
  gravity: 0.25,
  gravityRangeCompound: 1.5,
  gravityCompound: 1.0,
  gravityRange: 3.8,
  initialEnergyOnIncremental: 0.3,
  tile: true,
  tilingPaddingVertical: 10,
  tilingPaddingHorizontal: 10,
  numIter: 2500,
}

const DEFAULT_FCOSE_CONFIG_TEXT = JSON.stringify(DEFAULT_FCOSE_CONFIG, null, 2)

const communityIdFor = (parentNodeId: string): number => {
  const match = /^community_(\d+)$/.exec(parentNodeId)
  return match && match[1] !== undefined ? Number(match[1]) : NaN
}

const hashHue = (s: string): number => {
  let h = 5381
  for (let i = 0; i < s.length; i++) h = (h * 33) ^ s.charCodeAt(i)
  return Math.abs(h) % 360
}

const hueToHex = (hue: number, saturation = 0.6, lightness = 0.55): string => {
  const c = (1 - Math.abs(2 * lightness - 1)) * saturation
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = lightness - c / 2
  let r = 0,
    g = 0,
    b = 0
  if (hue < 60) [r, g, b] = [c, x, 0]
  else if (hue < 120) [r, g, b] = [x, c, 0]
  else if (hue < 180) [r, g, b] = [0, c, x]
  else if (hue < 240) [r, g, b] = [0, x, c]
  else if (hue < 300) [r, g, b] = [x, 0, c]
  else [r, g, b] = [c, 0, x]
  const toHex = (v: number) =>
    Math.round((v + m) * 255)
      .toString(16)
      .padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

const colorForKey = (key: string): string => hueToHex(hashHue(key))

const buildElements = (payload: KGPayload): cytoscape.ElementDefinition[] => {
  const parentIdFor = (communityId: number) => `community_${communityId}`

  const parents: cytoscape.ElementDefinition[] = payload.communities.map((c) => ({
    group: 'nodes',
    data: {
      id: parentIdFor(c.id),
      label: c.label ?? `community #${c.id}`,
      isCommunity: true,
      memberCount: c.member_count,
    },
  }))

  const children: cytoscape.ElementDefinition[] = payload.nodes.map((n) => ({
    group: 'nodes',
    data: {
      id: n.id,
      label: n.label,
      entityType: n.entity_type,
      mentionCount: n.mention_count,
      nodeBetweenness: n.node_betweenness,
      parent: n.community_id !== null ? parentIdFor(n.community_id) : undefined,
    },
  }))

  const nodeIdSet = new Set(children.map((c) => c.data.id as string))
  const edges: cytoscape.ElementDefinition[] = payload.edges
    .filter((e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map((e, i) => ({
      group: 'edges',
      data: {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        relType: e.rel_type ?? '',
        weight: e.weight ?? 1,
        edgeBetweenness: e.edge_betweenness,
      },
    }))

  return [...parents, ...children, ...edges]
}

const baseLightNodeStyle = {
  'background-color': '#4F86C6',
  label: 'data(label)',
  'font-size': '8px',
  color: '#222',
  'text-wrap': 'ellipsis',
  'text-max-width': '80px',
  'text-valign': 'center',
  'text-halign': 'center',
  width: 14,
  height: 14,
} as const satisfies cytoscape.Css.Node

const baseDarkNodeStyle = {
  ...baseLightNodeStyle,
  'background-color': '#7AB3FF',
  color: '#e6e6e6',
} as const satisfies cytoscape.Css.Node

const LIGHT_STYLESHEET: cytoscape.StylesheetStyle[] = [
  { selector: 'node', style: baseLightNodeStyle },
  {
    selector: 'node[?isCommunity]',
    style: {
      'background-color': '#F4E3A1',
      'background-opacity': 0.2,
      'border-color': '#C19A00',
      'border-width': 1,
      label: 'data(label)',
      'font-size': '14px',
      'font-weight': 'bold',
      color: '#5A4300',
      'text-valign': 'top',
      'text-halign': 'center',
      'text-margin-y': -6,
      'text-wrap': 'ellipsis',
      'text-max-width': '200px',
      shape: 'round-rectangle',
      padding: '16px',
    } as unknown as cytoscape.Css.Node,
  },
  {
    selector: 'edge',
    style: {
      width: 1,
      'line-color': '#999',
      'target-arrow-color': '#999',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      opacity: 0.6,
    },
  },
  { selector: 'node:selected', style: { 'border-color': '#ff8800', 'border-width': 3 } },
  {
    selector: 'edge:selected',
    style: {
      'line-color': '#ff8800',
      'target-arrow-color': '#ff8800',
      width: 3,
      opacity: 1,
    },
  },
]

const DARK_STYLESHEET: cytoscape.StylesheetStyle[] = [
  { selector: 'node', style: baseDarkNodeStyle },
  {
    selector: 'node[?isCommunity]',
    style: {
      'background-color': '#3a341a',
      'background-opacity': 0.35,
      'border-color': '#E0C765',
      'border-width': 1,
      label: 'data(label)',
      'font-size': '14px',
      'font-weight': 'bold',
      color: '#F5DC78',
      'text-valign': 'top',
      'text-halign': 'center',
      'text-margin-y': -6,
      'text-wrap': 'ellipsis',
      'text-max-width': '200px',
      shape: 'round-rectangle',
      padding: '16px',
    } as unknown as cytoscape.Css.Node,
  },
  {
    selector: 'edge',
    style: {
      width: 1,
      'line-color': '#5a6375',
      'target-arrow-color': '#5a6375',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      opacity: 0.7,
    },
  },
  { selector: 'node:selected', style: { 'border-color': '#ffbb33', 'border-width': 3 } },
  {
    selector: 'edge:selected',
    style: {
      'line-color': '#ffbb33',
      'target-arrow-color': '#ffbb33',
      width: 3,
      opacity: 1,
    },
  },
]

interface ParsedConfig {
  ok: true
  config: Record<string, unknown>
}

interface ParseError {
  ok: false
  error: string
}

/**
 * Parse the layout-config textarea into a fcose options object. The textarea
 * is the single source of truth for the layout (including constraints), so
 * an invalid string parks the layout — the auto-apply effect skips,
 * constraint-mutation buttons are disabled, and an inline error is shown.
 */
const parseLayoutConfig = (text: string): ParsedConfig | ParseError => {
  try {
    const parsed = JSON.parse(text)
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { ok: false, error: 'config must be a JSON object' }
    }
    return { ok: true, config: parsed as Record<string, unknown> }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}

const applyNodeSizes = (cy: cytoscape.Core, mode: SizeMode, scale: number, hiddenIds: Set<string>): void => {
  const base = 14 * scale
  const leaves = cy.nodes().filter((n: cytoscape.NodeSingular) => !n.data('isCommunity'))
  if (mode === 'uniform') {
    leaves.style({ width: base, height: base })
    return
  }
  const metric = (n: cytoscape.NodeSingular): number => {
    if (mode === 'degree') return n.degree(false)
    return Number(n.data('nodeBetweenness') ?? 0)
  }
  // Normalize over the visible subset so hidden outliers don't compress the
  // visible range. Hidden nodes are still sized (so re-showing them feels
  // consistent) but using the visible-set min/max.
  const values: number[] = []
  leaves.forEach((n: cytoscape.NodeSingular) => {
    if (!hiddenIds.has(n.id())) values.push(metric(n))
  })
  const max = values.length > 0 ? Math.max(...values) : 1
  const min = values.length > 0 ? Math.min(...values) : 0
  const range = max - min || 1
  leaves.forEach((n: cytoscape.NodeSingular) => {
    const m = metric(n)
    const norm = (m - min) / range
    const size = base * (0.6 + 2.4 * norm)
    n.style({ width: size, height: size })
  })
}

const applyNodeColors = (cy: cytoscape.Core, mode: NodeColorMode): void => {
  const leaves = cy.nodes().filter((n) => !n.data('isCommunity'))
  if (mode === 'uniform') {
    leaves.removeStyle('background-color')
    return
  }
  leaves.forEach((n) => {
    const key = String(n.data('entityType') ?? 'unknown')
    n.style('background-color', colorForKey(key))
  })
}

const applyEdgeColors = (cy: cytoscape.Core, mode: EdgeColorMode): void => {
  const edges = cy.edges()
  if (mode === 'uniform') {
    edges.removeStyle('line-color')
    edges.removeStyle('target-arrow-color')
    return
  }
  edges.forEach((e) => {
    const key = String(e.data('relType') || 'unknown')
    const c = colorForKey(key)
    e.style({ 'line-color': c, 'target-arrow-color': c })
  })
}

const applyEdgeThickness = (
  cy: cytoscape.Core,
  mode: EdgeThicknessMode,
  scale: number,
  hiddenEdgeIds: Set<string>,
): void => {
  const base = 1 * scale
  const edges = cy.edges()
  if (mode === 'uniform') {
    edges.style({ width: base })
    return
  }
  const metric = (e: cytoscape.EdgeSingular): number =>
    mode === 'weight' ? Number(e.data('weight') ?? 1) : Number(e.data('edgeBetweenness') ?? 0)
  const values: number[] = []
  edges.forEach((e: cytoscape.EdgeSingular) => {
    if (!hiddenEdgeIds.has(e.id())) values.push(metric(e))
  })
  const max = values.length > 0 ? Math.max(...values) : 1
  const min = values.length > 0 ? Math.min(...values) : 0
  const range = max - min || 1
  edges.forEach((e: cytoscape.EdgeSingular) => {
    const m = metric(e)
    const norm = (m - min) / range
    const w = base * (0.4 + 4.6 * norm)
    e.style('width', w)
  })
}

const applyCommunityOpacity = (cy: cytoscape.Core, opacity: number): void => {
  // Use the per-component opacity properties rather than the element-level
  // `opacity`, because cytoscape's compound-node `opacity` cascades to
  // children. We want the community box fade independent of its members.
  const parents = cy.nodes().filter((n: cytoscape.NodeSingular) => n.data('isCommunity'))
  parents.style({
    'background-opacity': opacity,
    'border-opacity': opacity,
    'text-opacity': opacity,
  })
}

interface VisibilityOptions {
  hiddenEntityTypes: Set<string>
  hiddenRelTypes: Set<string>
}

interface VisibilityResult {
  hiddenNodeIds: Set<string>
  hiddenEdgeIds: Set<string>
}

const applyVisibility = (cy: cytoscape.Core, opts: VisibilityOptions): VisibilityResult => {
  const hiddenNodeIds = new Set<string>()
  const hiddenEdgeIds = new Set<string>()
  const leaves = cy.nodes().filter((n: cytoscape.NodeSingular) => !n.data('isCommunity'))
  leaves.forEach((n: cytoscape.NodeSingular) => {
    const entityType = String(n.data('entityType') ?? 'unknown')
    const hide = opts.hiddenEntityTypes.has(entityType)
    if (hide) hiddenNodeIds.add(n.id())
    n.style('display', hide ? 'none' : 'element')
  })
  cy.edges().forEach((e: cytoscape.EdgeSingular) => {
    const rel = String(e.data('relType') || 'unknown')
    const hiddenByLegend = opts.hiddenRelTypes.has(rel)
    const srcHidden = hiddenNodeIds.has(e.source().id())
    const tgtHidden = hiddenNodeIds.has(e.target().id())
    const hide = hiddenByLegend || srcHidden || tgtHidden
    if (hide) hiddenEdgeIds.add(e.id())
    e.style('display', hide ? 'none' : 'element')
  })
  return { hiddenNodeIds, hiddenEdgeIds }
}

const groupCounts = (values: Array<string | null | undefined>): Array<[string, number]> => {
  const counts = new Map<string, number>()
  for (const v of values) {
    const key = v && v.length > 0 ? v : 'unknown'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
}

/**
 * fcose layout constraints — match the cytoscape-fcose v2 API surface that the
 * official demo uses (https://ivis-at-bilkent.github.io/cytoscape.js-fcose/demo/demo-constraint.html).
 *
 * The JSON `layoutConfig` textarea is the SOURCE OF TRUTH for constraints.
 * Constraint-mutation buttons round-trip through the JSON: parse → mutate →
 * re-stringify → setLayoutConfig. The active-constraint list rendered in
 * the right panel is derived by parsing the same JSON each render.
 *
 * - `fixedNodeConstraint`: `Array<{nodeId, position: {x, y}}>` — pinning.
 *   Compound parents are valid targets; their position is the centroid.
 * - `alignmentConstraint.vertical | .horizontal`: groups of node ids. Nodes
 *   in a vertical group share x; horizontal share y.
 * - `relativePlacementConstraint`: `Array<{top, bottom, gap} | {left, right, gap}>`.
 */
interface ParsedConstraints {
  pinned: Array<{ nodeId: string; position: { x: number; y: number } }>
  vertical: string[][]
  horizontal: string[][]
  relative: Array<{ kind: 'vertical' | 'horizontal'; first: string; second: string; gap: number; index: number }>
}

const EMPTY_PARSED_CONSTRAINTS: ParsedConstraints = {
  pinned: [],
  vertical: [],
  horizontal: [],
  relative: [],
}

const isPositionRecord = (value: unknown): value is { x: number; y: number } =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as { x?: unknown }).x === 'number' &&
  typeof (value as { y?: unknown }).y === 'number'

const isPinEntry = (v: unknown): v is { nodeId: string; position: { x: number; y: number } } =>
  typeof v === 'object' &&
  v !== null &&
  typeof (v as { nodeId?: unknown }).nodeId === 'string' &&
  isPositionRecord((v as { position?: unknown }).position)

const isStringArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === 'string')

const parseConstraintsFromConfig = (config: Record<string, unknown>): ParsedConstraints => {
  const out: ParsedConstraints = { pinned: [], vertical: [], horizontal: [], relative: [] }
  const fixed = config.fixedNodeConstraint
  if (Array.isArray(fixed)) {
    out.pinned = fixed
      .filter(isPinEntry)
      .map((p) => ({ nodeId: p.nodeId, position: { x: p.position.x, y: p.position.y } }))
  }
  const align = config.alignmentConstraint
  if (typeof align === 'object' && align !== null) {
    const v = (align as { vertical?: unknown }).vertical
    const h = (align as { horizontal?: unknown }).horizontal
    if (Array.isArray(v)) out.vertical = v.filter(isStringArray)
    if (Array.isArray(h)) out.horizontal = h.filter(isStringArray)
  }
  const rel = config.relativePlacementConstraint
  if (Array.isArray(rel)) {
    rel.forEach((r, index) => {
      if (typeof r !== 'object' || r === null) return
      const obj = r as Record<string, unknown>
      const gap = typeof obj.gap === 'number' ? obj.gap : DEFAULT_RELATIVE_PLACEMENT_GAP
      if (typeof obj.top === 'string' && typeof obj.bottom === 'string') {
        out.relative.push({ kind: 'vertical', first: obj.top, second: obj.bottom, gap, index })
      } else if (typeof obj.left === 'string' && typeof obj.right === 'string') {
        out.relative.push({ kind: 'horizontal', first: obj.left, second: obj.right, gap, index })
      }
    })
  }
  return out
}

const constraintTotal = (c: ParsedConstraints): number =>
  c.pinned.length + c.vertical.length + c.horizontal.length + c.relative.length

/**
 * Apply `mutator` to a parsed copy of `prevText` and return the re-stringified
 * JSON. If `prevText` doesn't parse, returns it unchanged so the caller's UI
 * disable-on-invalid logic does the right thing without further branching.
 */
const mutateLayoutConfigText = (prevText: string, mutator: (cfg: Record<string, unknown>) => void): string => {
  const result = parseLayoutConfig(prevText)
  if (!result.ok) return prevText
  const next = { ...result.config }
  mutator(next)
  return JSON.stringify(next, null, 2)
}

/** True iff `hidden` hides everything in `allKeys` except `key`. */
const isIsolatedTo = (hidden: Set<string>, key: string, allKeys: string[]): boolean => {
  if (allKeys.length <= 1) return false
  if (hidden.has(key)) return false
  for (const k of allKeys) {
    if (k === key) continue
    if (!hidden.has(k)) return false
  }
  return true
}

export function KGPage() {
  const { databaseId, tableId } = useParams<{ databaseId: string; tableId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  // Data-axis (server-side). Direct state — every change auto-refetches via
  // a debounced effect, so slider scrubbing collapses into one final fetch.
  const [topN, setTopN] = useState<number>(DEFAULT_TOP_N)
  const [maxDepth, setMaxDepth] = useState<number>(DEFAULT_MAX_DEPTH)
  const [seedMetric, setSeedMetric] = useState<SeedMetric>(DEFAULT_SEED_METRIC)
  const [minDegree, setMinDegree] = useState<number>(DEFAULT_MIN_DEGREE)

  // Client-side filters.
  const [hiddenEntityTypes, setHiddenEntityTypes] = useState<Set<string>>(new Set())
  const [hiddenRelTypes, setHiddenRelTypes] = useState<Set<string>>(new Set())

  // Layout. fcose-only (we dropped elk + grid). The JSON textarea is the
  // single source of truth — including for fcose constraints. Constraint
  // buttons round-trip through it; the active-constraint list is derived.
  const [layoutConfig, setLayoutConfig] = useState<string>(DEFAULT_FCOSE_CONFIG_TEXT)

  // Aesthetics. Defaults: node size by node-betweenness, edge thickness by
  // edge-betweenness — both metrics give honest visual cues for centrality.
  const [sizeMode, setSizeMode] = useState<SizeMode>(DEFAULT_SIZE_MODE)
  const [nodeColorMode, setNodeColorMode] = useState<NodeColorMode>('entity_type')
  const [edgeColorMode, setEdgeColorMode] = useState<EdgeColorMode>('rel_type')
  const [nodeScale, setNodeScale] = useState<number>(1)
  const [edgeOpacity, setEdgeOpacity] = useState<number>(0.6)
  const [communityOpacity, setCommunityOpacity] = useState<number>(DEFAULT_COMMUNITY_OPACITY)
  const [edgeThicknessMode, setEdgeThicknessMode] = useState<EdgeThicknessMode>('edge_betweenness')
  const [edgeThicknessScale, setEdgeThicknessScale] = useState<number>(1)

  // Gap value used when the user clicks one of the relative-placement
  // buttons. Doesn't live in the JSON because it's a per-click input,
  // not a stored constraint.
  const [relativeGap, setRelativeGap] = useState<number>(DEFAULT_RELATIVE_PLACEMENT_GAP)

  const [selection, setSelection] = useState<KGSelection>(EMPTY_SELECTION)

  const cyRef = useRef<cytoscape.Core | null>(null)
  const legendClickTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { resolved } = useTheme()
  const stylesheet = resolved === 'dark' ? DARK_STYLESHEET : LIGHT_STYLESHEET

  // Debounced fetch — coalesces slider scrubbing on topN/maxDepth/etc.
  useEffect(() => {
    if (!databaseId || !tableId) return
    const handle = setTimeout(() => {
      // Reset to loading + clear selection / legend filters so the previous
      // graph's state doesn't bleed into the refetch. Layout config is
      // intentionally preserved across refetches so a pinned-layout doesn't
      // reset on every parameter tweak.
      setState({ status: 'loading' })
      setSelection(EMPTY_SELECTION)
      setHiddenEntityTypes(new Set())
      setHiddenRelTypes(new Set())
      fetchKG(databaseId, tableId, { topN, seedMetric, maxDepth, minDegree })
        .then((payload) => setState({ status: 'ready', payload }))
        .catch((err: unknown) => {
          const message =
            err instanceof ApiError
              ? `API error ${err.status}: ${err.body}`
              : err instanceof Error
                ? err.message
                : 'unknown error'
          setState({ status: 'error', message })
        })
    }, AUTO_APPLY_DEBOUNCE_MS)
    return () => clearTimeout(handle)
  }, [databaseId, tableId, topN, seedMetric, maxDepth, minDegree])

  // Parse the JSON config once per render — the parsed config drives the
  // auto-apply layout effect AND the active-constraint list display.
  const parsedLayoutConfig = useMemo(() => parseLayoutConfig(layoutConfig), [layoutConfig])
  const parsedConstraints = useMemo<ParsedConstraints>(
    () => (parsedLayoutConfig.ok ? parseConstraintsFromConfig(parsedLayoutConfig.config) : EMPTY_PARSED_CONSTRAINTS),
    [parsedLayoutConfig],
  )

  const elements = useMemo(() => (state.status === 'ready' ? buildElements(state.payload) : []), [state])

  const nodeIndex = useMemo<Map<string, KGNode>>(() => {
    if (state.status !== 'ready') return new Map()
    return new Map(state.payload.nodes.map((n) => [n.id, n]))
  }, [state])

  const edgeIndex = useMemo<Map<string, KGEdge>>(() => {
    if (state.status !== 'ready') return new Map()
    return new Map(state.payload.edges.map((e, i) => [`e${i}`, e]))
  }, [state])

  const communityIndex = useMemo<Map<number, KGCommunity>>(() => {
    if (state.status !== 'ready') return new Map()
    return new Map(state.payload.communities.map((c) => [c.id, c]))
  }, [state])

  const entityTypeLegend = useMemo<Array<[string, number]>>(
    () =>
      state.status === 'ready' && nodeColorMode === 'entity_type'
        ? groupCounts(state.payload.nodes.map((n) => n.entity_type))
        : [],
    [state, nodeColorMode],
  )

  const relTypeLegend = useMemo<Array<[string, number]>>(
    () =>
      state.status === 'ready' && edgeColorMode === 'rel_type'
        ? groupCounts(state.payload.edges.map((e) => e.rel_type))
        : [],
    [state, edgeColorMode],
  )

  // Initial-position seed: when new data arrives, run a synchronous grid
  // layout so nodes have *some* position before the debounced fcose effect
  // takes over a moment later. Without this, the first paint can show all
  // nodes stacked at (0,0). No `setState` in here — readiness is conveyed
  // by `state.status === 'ready'`, which already flipped when the fetch
  // resolved.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || state.status !== 'ready') return
    cy.layout({ name: 'grid', animate: false } as cytoscape.LayoutOptions).run()
  }, [elements, state.status])

  // Auto-apply layout: any change to layoutConfig (manual JSON edit OR a
  // constraint button mutation) re-runs fcose, debounced. Same effect
  // covers data refresh because elements is a dep.
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || state.status !== 'ready') return
    if (!parsedLayoutConfig.ok) return
    const handle = setTimeout(() => {
      const current = cyRef.current
      if (!current) return
      try {
        current.layout({ ...parsedLayoutConfig.config, name: 'fcose' } as cytoscape.LayoutOptions).run()
      } catch {
        /* fcose throws on bad node IDs in constraints; the inline JSON-error
           panel surfaces this — no need to re-throw. */
      }
    }, AUTO_APPLY_DEBOUNCE_MS)
    return () => clearTimeout(handle)
  }, [elements, parsedLayoutConfig, state.status])

  useEffect(() => {
    cyRef.current?.style(stylesheet as unknown as cytoscape.StylesheetStyle[])
  }, [stylesheet])

  // Selection-sync is bound INSIDE the cy={...} mount callback below — the
  // useEffect-based approach that lived here had a fatal timing bug: the
  // effect's `cyRef.current` read was stale on the run where elements
  // first became non-empty, so cy.on() was never executed. Registering at
  // mount time guarantees we bind to the live cy.

  // Unified re-styling pass. Order matters:
  // visibility first (computes hidden id sets), then the
  // normalization-aware size/thickness passes that respect the visible
  // subset, then colors + opacities which don't depend on visibility.
  // Inlined as a single callback so it can fire from BOTH the cy={...}
  // mount callback (covers first paint, before any effect runs) AND the
  // effect below (covers later updates when only spec deps change).
  const applyAllStyling = (cy: cytoscape.Core) => {
    const { hiddenNodeIds, hiddenEdgeIds } = applyVisibility(cy, {
      hiddenEntityTypes,
      hiddenRelTypes,
    })
    applyNodeSizes(cy, sizeMode, nodeScale, hiddenNodeIds)
    applyEdgeThickness(cy, edgeThicknessMode, edgeThicknessScale, hiddenEdgeIds)
    applyNodeColors(cy, nodeColorMode)
    applyEdgeColors(cy, edgeColorMode)
    applyCommunityOpacity(cy, communityOpacity)
    cy.edges().style('opacity', edgeOpacity)
  }

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyAllStyling(cy)
    // applyAllStyling reads every spec listed in deps via closure, so the
    // exhaustive-deps lint is satisfied without listing the helper itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    elements,
    hiddenEntityTypes,
    hiddenRelTypes,
    sizeMode,
    nodeScale,
    edgeThicknessMode,
    edgeThicknessScale,
    nodeColorMode,
    edgeColorMode,
    communityOpacity,
    edgeOpacity,
  ])

  const resetLayoutConfig = () => setLayoutConfig(DEFAULT_FCOSE_CONFIG_TEXT)

  const clearSelection = () => {
    cyRef.current?.elements().unselect()
  }

  // ── Constraint handlers ────────────────────────────────────────────────
  // Constraint mutations round-trip through the JSON: parse → mutate →
  // re-stringify. Compound (community) parents are valid constraint
  // targets — fcose's fixedNodeConstraint / alignmentConstraint /
  // relativePlacementConstraint all accept compound-node IDs and operate
  // on their centroid (pinning a parent pins the cluster's centroid;
  // children rearrange around it).

  const selectedConstraintIds = (): string[] => {
    const cy = cyRef.current
    if (!cy) return []
    return cy.nodes(':selected').map((n) => n.id())
  }

  const updateLayoutConfig = (mutator: (cfg: Record<string, unknown>) => void) => {
    setLayoutConfig((prev) => mutateLayoutConfigText(prev, mutator))
  }

  const pinSelected = () => {
    const cy = cyRef.current
    if (!cy) return
    const ids = selectedConstraintIds()
    if (ids.length === 0) return
    const positions = ids.map((id) => {
      const pos = cy.getElementById(id).position()
      return { nodeId: id, position: { x: pos.x, y: pos.y } }
    })
    updateLayoutConfig((cfg) => {
      const existing = Array.isArray(cfg.fixedNodeConstraint) ? [...(cfg.fixedNodeConstraint as unknown[])] : []
      const filtered = existing.filter((e) => !(isPinEntry(e) && ids.includes(e.nodeId)))
      cfg.fixedNodeConstraint = [...filtered, ...positions]
    })
  }

  const unpinSelected = () => {
    const ids = selectedConstraintIds()
    if (ids.length === 0) return
    updateLayoutConfig((cfg) => {
      const existing = Array.isArray(cfg.fixedNodeConstraint) ? (cfg.fixedNodeConstraint as unknown[]) : []
      const next = existing.filter((e) => !(isPinEntry(e) && ids.includes(e.nodeId)))
      if (next.length > 0) cfg.fixedNodeConstraint = next
      else delete cfg.fixedNodeConstraint
    })
  }

  const unpinAll = () =>
    updateLayoutConfig((cfg) => {
      delete cfg.fixedNodeConstraint
    })

  const alignSelected = (axis: 'vertical' | 'horizontal') => {
    const ids = selectedConstraintIds()
    if (ids.length < 2) return
    updateLayoutConfig((cfg) => {
      const align = (
        typeof cfg.alignmentConstraint === 'object' && cfg.alignmentConstraint !== null
          ? { ...(cfg.alignmentConstraint as Record<string, unknown>) }
          : {}
      ) as Record<string, unknown>
      const groups = Array.isArray(align[axis]) ? (align[axis] as unknown[]).filter(isStringArray) : []
      align[axis] = [...groups, ids]
      cfg.alignmentConstraint = align
    })
  }

  const removeAlignmentGroup = (axis: 'vertical' | 'horizontal', index: number) => {
    updateLayoutConfig((cfg) => {
      if (typeof cfg.alignmentConstraint !== 'object' || cfg.alignmentConstraint === null) return
      const align = { ...(cfg.alignmentConstraint as Record<string, unknown>) }
      const groups = Array.isArray(align[axis]) ? (align[axis] as unknown[]).filter(isStringArray) : []
      const next = groups.filter((_, i) => i !== index)
      if (next.length > 0) align[axis] = next
      else delete align[axis]
      if (Object.keys(align).length === 0) delete cfg.alignmentConstraint
      else cfg.alignmentConstraint = align
    })
  }

  const addRelativePlacement = (
    kind: 'vertical' | 'horizontal',
    direction: 'first-then-second' | 'second-then-first',
  ) => {
    const ids = selectedConstraintIds()
    if (ids.length !== 2) return
    const [a, b] = ids as [string, string]
    const [first, second] = direction === 'first-then-second' ? [a, b] : [b, a]
    const entry =
      kind === 'vertical'
        ? { top: first, bottom: second, gap: relativeGap }
        : { left: first, right: second, gap: relativeGap }
    updateLayoutConfig((cfg) => {
      const existing = Array.isArray(cfg.relativePlacementConstraint)
        ? (cfg.relativePlacementConstraint as unknown[])
        : []
      cfg.relativePlacementConstraint = [...existing, entry]
    })
  }

  const removeRelativePlacement = (index: number) => {
    updateLayoutConfig((cfg) => {
      const existing = Array.isArray(cfg.relativePlacementConstraint)
        ? (cfg.relativePlacementConstraint as unknown[])
        : []
      const next = existing.filter((_, i) => i !== index)
      if (next.length > 0) cfg.relativePlacementConstraint = next
      else delete cfg.relativePlacementConstraint
    })
  }

  const clearAllConstraints = () =>
    updateLayoutConfig((cfg) => {
      delete cfg.fixedNodeConstraint
      delete cfg.alignmentConstraint
      delete cfg.relativePlacementConstraint
    })

  // ── Export ─────────────────────────────────────────────────────────────
  // PNG via cytoscape's built-in cy.png(); SVG via cytoscape-svg's cy.svg().
  // Both export with `full: true` so the full graph is captured (not just
  // the visible viewport), and with the current theme's surface as the
  // background — that way the saved file looks the same as on screen.

  const exportBackground = (): string => (resolved === 'dark' ? '#1c1c20' : '#ffffff')

  const triggerDownload = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const exportFilenameStem = (): string => {
    const stamp = new Date().toISOString().replace(/[:T]/g, '-').replace(/\..+$/, '')
    return `kg-${databaseId ?? 'unknown'}-${tableId ?? 'unknown'}-${stamp}`
  }

  const exportPng = () => {
    const cy = cyRef.current
    if (!cy) return
    const blob = cy.png({ output: 'blob', full: true, scale: 2, bg: exportBackground() })
    triggerDownload(blob, `${exportFilenameStem()}.png`)
  }

  const exportSvg = () => {
    const cy = cyRef.current
    if (!cy) return
    // cytoscape-svg adds .svg() at runtime; the @types package doesn't
    // include it, so cast through `unknown` to a minimal callable shape.
    const cyWithSvg = cy as unknown as {
      svg: (opts: { scale?: number; full?: boolean; bg?: string }) => string
    }
    const svgString = cyWithSvg.svg({ scale: 1, full: true, bg: exportBackground() })
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
    triggerDownload(blob, `${exportFilenameStem()}.svg`)
  }

  const selectedConstraintCount = selection.nodes.size + selection.communities.size

  const legendAllKeys = (group: 'entity' | 'rel'): string[] =>
    (group === 'entity' ? entityTypeLegend : relTypeLegend).map(([k]) => k)

  const doSingleToggle = (key: string, group: 'entity' | 'rel') => {
    const setter = group === 'entity' ? setHiddenEntityTypes : setHiddenRelTypes
    setter((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const doDoubleIsolate = (key: string, group: 'entity' | 'rel') => {
    const setter = group === 'entity' ? setHiddenEntityTypes : setHiddenRelTypes
    const allKeys = legendAllKeys(group)
    setter((prev) => {
      if (isIsolatedTo(prev, key, allKeys)) return new Set()
      return new Set(allKeys.filter((k) => k !== key))
    })
  }

  /** Plotly-style click timer: single-click after 250ms, double cancels the timer. */
  const handleLegendClick = (key: string, group: 'entity' | 'rel') => {
    if (legendClickTimer.current) {
      clearTimeout(legendClickTimer.current)
      legendClickTimer.current = null
      doDoubleIsolate(key, group)
      return
    }
    legendClickTimer.current = setTimeout(() => {
      legendClickTimer.current = null
      doSingleToggle(key, group)
    }, 250)
  }

  const selectedNodes = useMemo<KGNode[]>(() => {
    const items: KGNode[] = []
    for (const id of selection.nodes) {
      const n = nodeIndex.get(id)
      if (n) items.push(n)
    }
    return items
  }, [selection.nodes, nodeIndex])

  const selectedEdges = useMemo<KGEdge[]>(() => {
    const items: KGEdge[] = []
    for (const id of selection.edges) {
      const e = edgeIndex.get(id)
      if (e) items.push(e)
    }
    return items
  }, [selection.edges, edgeIndex])

  const selectedCommunities = useMemo<KGCommunity[]>(() => {
    const items: KGCommunity[] = []
    for (const id of selection.communities) {
      const c = communityIndex.get(id)
      if (c) items.push(c)
    }
    return items
  }, [selection.communities, communityIndex])

  const totalSelected = selectedNodes.length + selectedEdges.length + selectedCommunities.length

  const legendItemClass = (hidden: boolean): string =>
    [
      'flex items-center gap-1 rounded px-1 py-0.5 text-left transition',
      hidden ? 'opacity-40 line-through hover:opacity-70' : 'hover:bg-[var(--color-surface-elevated)]',
    ].join(' ')

  return (
    <main
      className="flex h-screen flex-col overflow-hidden bg-[var(--color-surface)] text-[var(--color-foreground)]"
      data-testid="kg-page"
      data-database-id={databaseId ?? ''}
      data-table-id={tableId ?? ''}
    >
      <header className="shrink-0 border-b border-[var(--color-border-subtle)] p-4">
        <nav className="flex gap-4 text-sm">
          <Link to="/" className="text-[var(--color-accent)] hover:underline">
            Home
          </Link>
          <Link to={`/${databaseId}/`} className="text-[var(--color-accent)] hover:underline">
            ← {databaseId}
          </Link>
        </nav>
        <h1 className="mt-2 text-2xl font-bold">
          Knowledge Graph: <span className="font-mono">{tableId}</span>
        </h1>
        {state.status === 'ready' && (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            {state.payload.node_count.toLocaleString()} nodes
            {state.payload.total_node_count > state.payload.node_count
              ? ` (of ${state.payload.total_node_count.toLocaleString()})`
              : ''}
            {' · '}
            {state.payload.edge_count.toLocaleString()} edges · {state.payload.community_count.toLocaleString()}{' '}
            communities · seeds by <span className="font-mono">{state.payload.seed_metric}</span>, depth=
            {state.payload.max_depth}, min-deg={state.payload.min_degree}
          </p>
        )}
      </header>

      <section className="flex min-h-0 flex-1">
        <div className="relative flex-1">
          {state.status === 'loading' && (
            <div data-testid="kg-loading" className="p-8">
              Loading knowledge graph…
            </div>
          )}
          {state.status === 'error' && (
            <div
              data-testid="kg-error"
              className="m-8 rounded border border-red-400 bg-red-50 p-4 text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
            >
              <p className="font-semibold">Failed to load knowledge graph</p>
              <p className="text-sm">{state.message}</p>
            </div>
          )}
          {state.status === 'ready' && (
            <div className="absolute inset-0">
              <CytoscapeComponent
                elements={elements}
                stylesheet={stylesheet}
                cy={(cy: cytoscape.Core) => {
                  cyRef.current = cy
                  cy.selectionType('additive')
                  // Apply colors / sizes / opacities synchronously inside
                  // the same tick that cytoscape was constructed. Without
                  // this, the per-element inline overrides land in a later
                  // effect tick — and the first paint shows the base
                  // stylesheet's uniform colors instead of entity-typed.
                  applyAllStyling(cy)
                  // Selection sync. Bound here (not in a useEffect) because
                  // the effect-keyed-on-elements approach hits a timing
                  // race where cyRef.current is null when the effect first
                  // runs and never re-runs. The cy={...} callback can fire
                  // multiple times for the same cy instance (every parent
                  // render), so we gate registration via cy.scratch — a
                  // per-cy-instance flag that survives callback re-fires.
                  // (We can't use a namespaced `select.viz` listener for
                  // idempotent removal: in cytoscape 3.33, namespaced
                  // listeners don't fire on internally-emitted events.)
                  if (!cy.scratch('viz-sync-registered')) {
                    cy.scratch('viz-sync-registered', true)
                    cy.on('select unselect', () => {
                      const nodes = new Set<string>()
                      const communities = new Set<number>()
                      cy.nodes(':selected').forEach((n) => {
                        if (n.data('isCommunity')) {
                          const cid = communityIdFor(n.id())
                          if (!Number.isNaN(cid)) communities.add(cid)
                        } else {
                          nodes.add(n.id())
                        }
                      })
                      const edges = new Set<string>(cy.edges(':selected').map((e) => e.id()))
                      setSelection({ nodes, edges, communities })
                    })
                  }
                }}
                style={{ width: '100%', height: '100%' }}
                wheelSensitivity={0.2}
              />
              {state.status === 'ready' && (
                <div
                  data-testid="kg-canvas-ready"
                  data-node-count={state.payload.node_count}
                  data-edge-count={state.payload.edge_count}
                  data-community-count={state.payload.community_count}
                  className="hidden"
                />
              )}
            </div>
          )}
        </div>

        {state.status === 'ready' && (
          <RightPanel title="Graph inspector" storageKey="kg" testId="kg">
            {/* 1. Filters — entity-type & relation-type legends. Live at the
                top because they're the most-used controls during exploration. */}
            <PanelSection
              title="Filters"
              storageKey="kg-filters"
              meta={
                <span className="font-mono">
                  {entityTypeLegend.length}+{relTypeLegend.length}
                </span>
              }
            >
              {entityTypeLegend.length === 0 && relTypeLegend.length === 0 && (
                <p className="text-[11px] text-[var(--color-muted-foreground)]">
                  No legend yet — entity / relation typing depends on the selected node and edge color modes
                  (Aesthetics).
                </p>
              )}

              {entityTypeLegend.length > 0 && (
                <div data-testid="kg-legend-entity-types">
                  <div className="mb-1 flex items-baseline justify-between text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                    <span>Entity types · {entityTypeLegend.length}</span>
                    <span className="normal-case tracking-normal">click / dbl-click</span>
                  </div>
                  <ul className="flex flex-col gap-0.5 text-[11px]">
                    {entityTypeLegend.slice(0, 24).map(([key, count]) => {
                      const hidden = hiddenEntityTypes.has(key)
                      return (
                        <li key={`et-${key}`}>
                          <button
                            type="button"
                            onClick={() => handleLegendClick(key, 'entity')}
                            data-testid={`kg-legend-entity-${key}`}
                            data-hidden={hidden}
                            className={legendItemClass(hidden)}
                          >
                            <span
                              className="inline-block h-2.5 w-2.5 rounded"
                              style={{ background: colorForKey(key) }}
                            />
                            <span className="truncate">
                              {key} <span className="text-[var(--color-muted-foreground)]">·{count}</span>
                            </span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}

              {relTypeLegend.length > 0 && (
                <div data-testid="kg-legend-rel-types">
                  <div className="mb-1 flex items-baseline justify-between text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                    <span>Relation types · {relTypeLegend.length}</span>
                    <span className="normal-case tracking-normal">click / dbl-click</span>
                  </div>
                  <ul className="flex flex-col gap-0.5 text-[11px]">
                    {relTypeLegend.slice(0, 24).map(([key, count]) => {
                      const hidden = hiddenRelTypes.has(key)
                      return (
                        <li key={`rt-${key}`}>
                          <button
                            type="button"
                            onClick={() => handleLegendClick(key, 'rel')}
                            data-testid={`kg-legend-rel-${key}`}
                            data-hidden={hidden}
                            className={legendItemClass(hidden)}
                          >
                            <span
                              className="inline-block h-2.5 w-2.5 rounded"
                              style={{ background: colorForKey(key) }}
                            />
                            <span className="truncate">
                              {key} <span className="text-[var(--color-muted-foreground)]">·{count}</span>
                            </span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}

              <p className="text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                Single-click toggles, double-click isolates (or restores all if already isolated).
              </p>
            </PanelSection>

            {/* 2. Selection — selected nodes / edges / communities. */}
            <PanelSection
              title="Selection"
              storageKey="kg-selection"
              meta={
                <span className="font-mono" data-testid="kg-selection-count">
                  {totalSelected}
                </span>
              }
            >
              {totalSelected === 0 ? (
                <p className="text-[12px] text-[var(--color-muted-foreground)]">
                  Nothing selected. Click nodes, edges, or community boxes; ctrl/cmd-click to add.
                </p>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={clearSelection}
                    data-testid="kg-clear-selection"
                    className="self-start rounded border border-[var(--color-border-subtle)] px-2 py-0.5 text-[11px] hover:border-[var(--color-accent)]"
                  >
                    Clear
                  </button>

                  {selectedNodes.length > 0 && (
                    <div>
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                        Nodes · {selectedNodes.length}
                      </div>
                      <ul className="flex flex-col gap-1" data-testid="kg-selection-nodes">
                        {selectedNodes.map((n) => (
                          <li
                            key={`n-${n.id}`}
                            className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-2"
                          >
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="truncate font-mono text-xs">{n.id}</span>
                              {n.entity_type && (
                                <span className="rounded bg-[var(--color-surface-sunken)] px-1 text-[10px]">
                                  {n.entity_type}
                                </span>
                              )}
                            </div>
                            <div className="mt-1 break-words text-[12px] leading-snug">{n.label}</div>
                            <div className="mt-1 flex gap-3 font-mono text-[10px] text-[var(--color-muted-foreground)]">
                              {n.mention_count !== null && <span>mentions = {n.mention_count}</span>}
                              {n.node_betweenness !== null && <span>bc = {n.node_betweenness.toFixed(4)}</span>}
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {selectedEdges.length > 0 && (
                    <div>
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                        Edges · {selectedEdges.length}
                      </div>
                      <ul className="flex flex-col gap-1" data-testid="kg-selection-edges">
                        {selectedEdges.map((e, i) => (
                          <li
                            key={`e-${i}`}
                            className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-2"
                          >
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="truncate font-mono text-xs">
                                {e.source} → {e.target}
                              </span>
                              {e.weight !== null && (
                                <span className="font-mono text-[10px] text-[var(--color-muted-foreground)]">
                                  w={e.weight.toFixed(2)}
                                </span>
                              )}
                            </div>
                            {e.rel_type && <div className="mt-1 text-[12px] leading-snug">{e.rel_type}</div>}
                            {e.edge_betweenness !== null && (
                              <div className="mt-1 font-mono text-[10px] text-[var(--color-muted-foreground)]">
                                bc = {e.edge_betweenness.toFixed(4)}
                              </div>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {selectedCommunities.length > 0 && (
                    <div>
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                        Communities · {selectedCommunities.length}
                      </div>
                      <ul className="flex flex-col gap-1" data-testid="kg-selection-communities">
                        {selectedCommunities.map((c) => (
                          <li
                            key={`c-${c.id}`}
                            className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-2"
                          >
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="font-mono text-xs">#{c.id}</span>
                              <span className="font-mono text-[10px] text-[var(--color-muted-foreground)]">
                                members = {c.member_count}
                              </span>
                            </div>
                            <div className="mt-1 break-words text-[12px] leading-snug">
                              {c.label ?? `community #${c.id}`}
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </PanelSection>

            {/* 3. Graph filters — server-side data axis. Each control
                auto-refetches via a debounced effect; no Reload button. */}
            <PanelSection
              title="Graph filters"
              storageKey="kg-graph-filters"
              meta={
                <span className="font-mono" data-testid="kg-loaded-count">
                  {state.payload.node_count}/{state.payload.total_node_count}
                </span>
              }
            >
              <label className="flex flex-col gap-1" data-testid="kg-control-top-n">
                <span>Max seed nodes (0 = all)</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                />
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                  aria-label="Top-N slider"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-seed-metric">
                <span>Seed sort metric</span>
                <select
                  value={seedMetric}
                  onChange={(e) => setSeedMetric(e.target.value as SeedMetric)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="edge_betweenness">edge betweenness</option>
                  <option value="node_betweenness">node betweenness</option>
                  <option value="degree">degree</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-max-depth">
                <span>Max depth (0 = unlimited)</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(Number(e.target.value))}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                />
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                  aria-label="Max-depth slider"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-min-degree">
                <span>Min degree (prune isolates)</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={minDegree}
                  onChange={(e) => setMinDegree(Number(e.target.value))}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                />
              </label>
            </PanelSection>

            {/* 4. fcose constraints + JSON config. The JSON textarea is the
                source of truth; constraint buttons round-trip through it. */}
            <PanelSection
              title="fcose constraints & config"
              storageKey="kg-constraints"
              meta={
                <span className="font-mono" data-testid="kg-constraint-total">
                  {constraintTotal(parsedConstraints)}
                </span>
              }
            >
              <p className="text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                Select nodes or community boxes in the canvas, then click a constraint button to mutate the JSON below.
                Layout re-runs automatically. Pinning a community anchors the cluster's centroid.
              </p>

              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                  Pin (selected: {selectedConstraintCount})
                </span>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={pinSelected}
                    disabled={selectedConstraintCount === 0 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-pin"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    Pin selected
                  </button>
                  <button
                    type="button"
                    onClick={unpinSelected}
                    disabled={selectedConstraintCount === 0 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-unpin"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    Unpin selected
                  </button>
                  <button
                    type="button"
                    onClick={unpinAll}
                    disabled={parsedConstraints.pinned.length === 0 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-unpin-all"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    Unpin all
                  </button>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                  Alignment (≥2 selected)
                </span>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => alignSelected('vertical')}
                    disabled={selectedConstraintCount < 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-align-vertical"
                    title="Selected nodes share the same x"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    Align vertical
                  </button>
                  <button
                    type="button"
                    onClick={() => alignSelected('horizontal')}
                    disabled={selectedConstraintCount < 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-align-horizontal"
                    title="Selected nodes share the same y"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    Align horizontal
                  </button>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                  Relative placement (exactly 2 selected)
                </span>
                <label className="flex items-center gap-2 text-xs">
                  <span>Gap</span>
                  <input
                    type="number"
                    min={1}
                    step={10}
                    value={relativeGap}
                    onChange={(e) => setRelativeGap(Number(e.target.value))}
                    data-testid="kg-constraint-gap"
                    className="w-20 rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-0.5 text-xs"
                  />
                  <span className="text-[var(--color-muted-foreground)]">px</span>
                </label>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => addRelativePlacement('vertical', 'first-then-second')}
                    disabled={selectedConstraintCount !== 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-place-above"
                    title="First selected goes above the second"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    1st above 2nd
                  </button>
                  <button
                    type="button"
                    onClick={() => addRelativePlacement('vertical', 'second-then-first')}
                    disabled={selectedConstraintCount !== 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-place-below"
                    title="First selected goes below the second"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    1st below 2nd
                  </button>
                  <button
                    type="button"
                    onClick={() => addRelativePlacement('horizontal', 'first-then-second')}
                    disabled={selectedConstraintCount !== 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-place-left"
                    title="First selected goes left of the second"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    1st left of 2nd
                  </button>
                  <button
                    type="button"
                    onClick={() => addRelativePlacement('horizontal', 'second-then-first')}
                    disabled={selectedConstraintCount !== 2 || !parsedLayoutConfig.ok}
                    data-testid="kg-constraint-place-right"
                    title="First selected goes right of the second"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                  >
                    1st right of 2nd
                  </button>
                </div>
              </div>

              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={clearAllConstraints}
                  disabled={constraintTotal(parsedConstraints) === 0 || !parsedLayoutConfig.ok}
                  data-testid="kg-constraint-clear-all"
                  className="rounded border border-[var(--color-border-subtle)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
                >
                  Clear all constraints
                </button>
                <button
                  type="button"
                  onClick={resetLayoutConfig}
                  data-testid="kg-reset-config"
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)]"
                >
                  Reset config
                </button>
              </div>

              {constraintTotal(parsedConstraints) > 0 && (
                <div data-testid="kg-constraint-list" className="flex flex-col gap-1 text-[11px]">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
                    Active
                  </div>

                  {parsedConstraints.pinned.length > 0 && (
                    <div className="flex items-center justify-between rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1">
                      <span>
                        <span className="font-mono">pinned</span> · {parsedConstraints.pinned.length} node
                        {parsedConstraints.pinned.length === 1 ? '' : 's'}
                      </span>
                      <button
                        type="button"
                        onClick={unpinAll}
                        aria-label="remove pinned"
                        className="text-[var(--color-muted-foreground)] hover:text-[var(--color-accent)]"
                      >
                        ×
                      </button>
                    </div>
                  )}

                  {parsedConstraints.vertical.map((group, i) => (
                    <div
                      key={`v-${i}`}
                      className="flex items-center justify-between rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1"
                    >
                      <span className="truncate">
                        <span className="font-mono">align ⇕</span> · {group.length} nodes
                      </span>
                      <button
                        type="button"
                        onClick={() => removeAlignmentGroup('vertical', i)}
                        aria-label="remove vertical group"
                        className="text-[var(--color-muted-foreground)] hover:text-[var(--color-accent)]"
                      >
                        ×
                      </button>
                    </div>
                  ))}

                  {parsedConstraints.horizontal.map((group, i) => (
                    <div
                      key={`h-${i}`}
                      className="flex items-center justify-between rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1"
                    >
                      <span className="truncate">
                        <span className="font-mono">align ⇔</span> · {group.length} nodes
                      </span>
                      <button
                        type="button"
                        onClick={() => removeAlignmentGroup('horizontal', i)}
                        aria-label="remove horizontal group"
                        className="text-[var(--color-muted-foreground)] hover:text-[var(--color-accent)]"
                      >
                        ×
                      </button>
                    </div>
                  ))}

                  {parsedConstraints.relative.map((r) => {
                    const arrow = r.kind === 'vertical' ? '↑' : '←'
                    return (
                      <div
                        key={`r-${r.index}`}
                        className="flex items-center justify-between rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1"
                      >
                        <span className="truncate font-mono text-[10px]">
                          {r.first} {arrow} {r.second} · gap={r.gap}
                        </span>
                        <button
                          type="button"
                          onClick={() => removeRelativePlacement(r.index)}
                          aria-label="remove relative placement"
                          className="text-[var(--color-muted-foreground)] hover:text-[var(--color-accent)]"
                        >
                          ×
                        </button>
                      </div>
                    )
                  })}
                </div>
              )}

              <label className="flex flex-col gap-1" data-testid="kg-control-config">
                <span>Layout config (JSON)</span>
                <textarea
                  value={layoutConfig}
                  onChange={(e) => setLayoutConfig(e.target.value)}
                  rows={14}
                  spellCheck={false}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 font-mono text-[10px] leading-snug"
                />
                {!parsedLayoutConfig.ok && (
                  <p
                    data-testid="kg-layout-error"
                    className="rounded border border-red-400 bg-red-50 px-2 py-1 text-[10px] text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
                  >
                    {parsedLayoutConfig.error}
                  </p>
                )}
              </label>
            </PanelSection>

            {/* 5. Aesthetics — node/edge visual style. */}
            <PanelSection title="Aesthetics" storageKey="kg-aesthetics" defaultOpen={false}>
              <label className="flex flex-col gap-1" data-testid="kg-control-size-mode">
                <span>Node size by</span>
                <select
                  value={sizeMode}
                  onChange={(e) => setSizeMode(e.target.value as SizeMode)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="betweenness">node betweenness</option>
                  <option value="degree">degree</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-node-color">
                <span>Node color</span>
                <select
                  value={nodeColorMode}
                  onChange={(e) => setNodeColorMode(e.target.value as NodeColorMode)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="entity_type">by entity type</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-node-scale">
                <span className="flex items-center justify-between">
                  <span>Node size scale</span>
                  <span className="font-mono text-xs text-[var(--color-muted-foreground)]">
                    {nodeScale.toFixed(2)}×
                  </span>
                </span>
                <input
                  type="range"
                  min={0.5}
                  max={3}
                  step={0.1}
                  value={nodeScale}
                  onChange={(e) => setNodeScale(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-edge-thickness-mode">
                <span>Edge thickness by</span>
                <select
                  value={edgeThicknessMode}
                  onChange={(e) => setEdgeThicknessMode(e.target.value as EdgeThicknessMode)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="edge_betweenness">edge betweenness</option>
                  <option value="weight">weight</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-edge-color">
                <span>Edge color</span>
                <select
                  value={edgeColorMode}
                  onChange={(e) => setEdgeColorMode(e.target.value as EdgeColorMode)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="rel_type">by relation type</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-edge-thickness-scale">
                <span className="flex items-center justify-between">
                  <span>Edge thickness scale</span>
                  <span className="font-mono text-xs text-[var(--color-muted-foreground)]">
                    {edgeThicknessScale.toFixed(2)}×
                  </span>
                </span>
                <input
                  type="range"
                  min={0.3}
                  max={5}
                  step={0.1}
                  value={edgeThicknessScale}
                  onChange={(e) => setEdgeThicknessScale(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-edge-opacity">
                <span className="flex items-center justify-between">
                  <span>Edge opacity</span>
                  <span className="font-mono text-xs text-[var(--color-muted-foreground)]">
                    {edgeOpacity.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={edgeOpacity}
                  onChange={(e) => setEdgeOpacity(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-community-opacity">
                <span className="flex items-center justify-between">
                  <span>Community opacity</span>
                  <span className="font-mono text-xs text-[var(--color-muted-foreground)]">
                    {communityOpacity.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={communityOpacity}
                  onChange={(e) => setCommunityOpacity(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
              </label>
            </PanelSection>

            {/* 6. Export — PNG (raster) and SVG (vector). Both export the
                full graph at the current layout / styling, against the
                current theme's background so the saved file looks the
                same as on screen. */}
            <PanelSection title="Export" storageKey="kg-export" defaultOpen={false}>
              <p className="text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                Saves the graph as it currently appears (full extent, current zoom-independent layout, current colors).
                Filename includes the database id, table id, and timestamp.
              </p>
              <div className="flex flex-wrap gap-1">
                <button
                  type="button"
                  onClick={exportPng}
                  data-testid="kg-export-png"
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)]"
                >
                  Download PNG
                </button>
                <button
                  type="button"
                  onClick={exportSvg}
                  data-testid="kg-export-svg"
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)]"
                >
                  Download SVG
                </button>
              </div>
              <p className="text-[10px] leading-snug text-[var(--color-muted-foreground)]">
                PNG is rasterized at 2× device pixels; SVG preserves vector geometry for scalable / editable output.
              </p>
            </PanelSection>
          </RightPanel>
        )}
      </section>
    </main>
  )
}
