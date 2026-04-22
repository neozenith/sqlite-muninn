import cytoscape from 'cytoscape'
// cytoscape-fcose and cytoscape-elk don't ship perfect types for their default
// exports — the casts below hand the plugins to cytoscape.use() without
// relaxing our strict mode.
import fcose from 'cytoscape-fcose'
import elk from 'cytoscape-elk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
import { useTheme } from '../lib/ThemeProvider'

cytoscape.use(fcose as unknown as cytoscape.Ext)
cytoscape.use(elk as unknown as cytoscape.Ext)

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; payload: KGPayload }

type LayoutEngine = 'grid' | 'fcose' | 'elk'
type SizeMode = 'uniform' | 'degree' | 'betweenness'
type NodeColorMode = 'uniform' | 'entity_type'
type EdgeColorMode = 'uniform' | 'rel_type'
type EdgeThicknessMode = 'uniform' | 'weight' | 'edge_betweenness'

const DEFAULT_TOP_N = 50
const DEFAULT_MAX_DEPTH = 0
const DEFAULT_SEED_METRIC: SeedMetric = 'edge_betweenness'
const DEFAULT_MIN_DEGREE = 1
const DEFAULT_COMMUNITY_OPACITY = 0.2

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

const ELK_PRESETS: Record<string, object> = {
  'layered (top→down)': {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': 40,
      'elk.layered.spacing.nodeNodeBetweenLayers': 60,
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
    },
  },
  'layered (left→right)': {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': 40,
      'elk.layered.spacing.nodeNodeBetweenLayers': 60,
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
    },
  },
  stress: {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'stress',
      'elk.stress.desiredEdgeLength': 100,
      'elk.stress.epsilon': 0.0001,
      'elk.stress.iterationLimit': 400,
    },
  },
  force: {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'force',
      'elk.force.iterations': 300,
      'elk.force.model': 'FRUCHTERMAN_REINGOLD',
      'elk.force.repulsivePower': 2,
    },
  },
  mrtree: {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'mrtree',
      'elk.mrtree.weighting': 'DESCENDANTS',
      'elk.spacing.nodeNode': 30,
    },
  },
  radial: {
    fit: true,
    padding: 30,
    elk: {
      algorithm: 'radial',
      'elk.radial.radius': 400,
      'elk.spacing.nodeNode': 40,
    },
  },
}

const DEFAULT_ELK_PRESET = 'layered (top→down)'

const defaultElkConfig = (): string =>
  JSON.stringify(ELK_PRESETS[DEFAULT_ELK_PRESET], null, 2)

const DEFAULT_CONFIGS: Record<Exclude<LayoutEngine, 'grid'>, string> = {
  fcose: JSON.stringify(DEFAULT_FCOSE_CONFIG, null, 2),
  elk: defaultElkConfig(),
}

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

interface LayoutRunResult {
  ok: boolean
  error?: string
}

const runLayout = (
  cy: cytoscape.Core,
  engine: LayoutEngine,
  configJson: string,
): LayoutRunResult => {
  try {
    if (engine === 'grid') {
      cy.layout({ name: 'grid', animate: false } as cytoscape.LayoutOptions).run()
      return { ok: true }
    }
    const parsed = JSON.parse(configJson) as Record<string, unknown>
    cy.layout({ ...parsed, name: engine } as cytoscape.LayoutOptions).run()
    return { ok: true }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}

const applyNodeSizes = (cy: cytoscape.Core, mode: SizeMode, scale: number): void => {
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
  const values: number[] = []
  leaves.forEach((n: cytoscape.NodeSingular) => {
    values.push(metric(n))
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
): void => {
  const base = 1 * scale
  const edges = cy.edges()
  if (mode === 'uniform') {
    edges.style({ width: base })
    return
  }
  const metric = (e: cytoscape.EdgeSingular): number =>
    mode === 'weight'
      ? Number(e.data('weight') ?? 1)
      : Number(e.data('edgeBetweenness') ?? 0)
  const values: number[] = []
  edges.forEach((e: cytoscape.EdgeSingular) => {
    values.push(metric(e))
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
  const parents = cy.nodes().filter((n) => n.data('isCommunity'))
  parents.style('opacity', opacity)
}

interface VisibilityOptions {
  minDegree: number
  hiddenEntityTypes: Set<string>
  hiddenRelTypes: Set<string>
}

const applyVisibility = (cy: cytoscape.Core, opts: VisibilityOptions): void => {
  const leaves = cy.nodes().filter((n: cytoscape.NodeSingular) => !n.data('isCommunity'))
  // Pre-compute total (not filtered) degree — min-degree is a structural
  // filter against the loaded subgraph, not a recursive one.
  const degreeMap = new Map<string, number>()
  const hiddenNodeIds = new Set<string>()
  leaves.forEach((n: cytoscape.NodeSingular) => {
    degreeMap.set(n.id(), n.degree(false))
  })
  leaves.forEach((n: cytoscape.NodeSingular) => {
    const entityType = String(n.data('entityType') ?? 'unknown')
    const hiddenByLegend = opts.hiddenEntityTypes.has(entityType)
    const belowMinDegree = (degreeMap.get(n.id()) ?? 0) < opts.minDegree
    const hide = hiddenByLegend || belowMinDegree
    if (hide) hiddenNodeIds.add(n.id())
    n.style('display', hide ? 'none' : 'element')
  })
  cy.edges().forEach((e: cytoscape.EdgeSingular) => {
    const rel = String(e.data('relType') || 'unknown')
    const hiddenByLegend = opts.hiddenRelTypes.has(rel)
    const srcHidden = hiddenNodeIds.has(e.source().id())
    const tgtHidden = hiddenNodeIds.has(e.target().id())
    const hide = hiddenByLegend || srcHidden || tgtHidden
    e.style('display', hide ? 'none' : 'element')
  })
}

const groupCounts = (values: Array<string | null | undefined>): Array<[string, number]> => {
  const counts = new Map<string, number>()
  for (const v of values) {
    const key = v && v.length > 0 ? v : 'unknown'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
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
  const [layoutReady, setLayoutReady] = useState(false)

  // Data-axis (server-side). `pending*` mirrors what the user has typed but
  // not yet applied; `*` triggers a refetch when it flips.
  const [topN, setTopN] = useState<number>(DEFAULT_TOP_N)
  const [pendingTopN, setPendingTopN] = useState<number>(DEFAULT_TOP_N)
  const [maxDepth, setMaxDepth] = useState<number>(DEFAULT_MAX_DEPTH)
  const [pendingMaxDepth, setPendingMaxDepth] = useState<number>(DEFAULT_MAX_DEPTH)
  const [seedMetric, setSeedMetric] = useState<SeedMetric>(DEFAULT_SEED_METRIC)
  const [pendingSeedMetric, setPendingSeedMetric] = useState<SeedMetric>(DEFAULT_SEED_METRIC)

  // Client-side filters.
  const [minDegree, setMinDegree] = useState<number>(DEFAULT_MIN_DEGREE)
  const [hiddenEntityTypes, setHiddenEntityTypes] = useState<Set<string>>(new Set())
  const [hiddenRelTypes, setHiddenRelTypes] = useState<Set<string>>(new Set())

  // Layout-axis.
  const [layoutEngine, setLayoutEngine] = useState<LayoutEngine>('fcose')
  const [layoutConfigs, setLayoutConfigs] = useState<Record<'fcose' | 'elk', string>>(
    () => ({ ...DEFAULT_CONFIGS }),
  )
  const [elkPreset, setElkPreset] = useState<string>(DEFAULT_ELK_PRESET)
  const [layoutError, setLayoutError] = useState<string | null>(null)

  // Styling.
  const [sizeMode, setSizeMode] = useState<SizeMode>('degree')
  const [nodeColorMode, setNodeColorMode] = useState<NodeColorMode>('entity_type')
  const [edgeColorMode, setEdgeColorMode] = useState<EdgeColorMode>('rel_type')
  const [nodeScale, setNodeScale] = useState<number>(1)
  const [edgeOpacity, setEdgeOpacity] = useState<number>(0.6)
  const [communityOpacity, setCommunityOpacity] =
    useState<number>(DEFAULT_COMMUNITY_OPACITY)
  const [edgeThicknessMode, setEdgeThicknessMode] =
    useState<EdgeThicknessMode>('uniform')
  const [edgeThicknessScale, setEdgeThicknessScale] = useState<number>(1)

  const [selection, setSelection] = useState<KGSelection>(EMPTY_SELECTION)

  const cyRef = useRef<cytoscape.Core | null>(null)
  const legendClickTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { resolved } = useTheme()
  const stylesheet = resolved === 'dark' ? DARK_STYLESHEET : LIGHT_STYLESHEET

  useEffect(() => {
    if (!databaseId || !tableId) return
    setState({ status: 'loading' })
    setLayoutReady(false)
    setSelection(EMPTY_SELECTION)
    setHiddenEntityTypes(new Set())
    setHiddenRelTypes(new Set())
    fetchKG(databaseId, tableId, { topN, seedMetric, maxDepth })
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
  }, [databaseId, tableId, topN, seedMetric, maxDepth])

  const elements = useMemo(
    () => (state.status === 'ready' ? buildElements(state.payload) : []),
    [state],
  )

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

  useEffect(() => {
    const cy = cyRef.current
    if (!cy || state.status !== 'ready') return
    setLayoutReady(false)
    cy.layout({ name: 'grid', animate: false } as cytoscape.LayoutOptions).run()
    setLayoutReady(true)
    if (layoutEngine !== 'grid') {
      const engine: 'fcose' | 'elk' = layoutEngine
      const config = layoutConfigs[engine]
      queueMicrotask(() => {
        if (!cyRef.current) return
        const result = runLayout(cyRef.current, engine, config)
        if (!result.ok) setLayoutError(result.error ?? 'unknown layout error')
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements, state.status])

  useEffect(() => {
    cyRef.current?.style(stylesheet as unknown as cytoscape.StylesheetStyle[])
  }, [stylesheet])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    const sync = () => {
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
    }
    cy.on('select unselect', sync)
    return () => {
      cy.off('select unselect', sync)
    }
  }, [elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyNodeSizes(cy, sizeMode, nodeScale)
  }, [sizeMode, nodeScale, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyNodeColors(cy, nodeColorMode)
  }, [nodeColorMode, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyEdgeColors(cy, edgeColorMode)
  }, [edgeColorMode, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyEdgeThickness(cy, edgeThicknessMode, edgeThicknessScale)
  }, [edgeThicknessMode, edgeThicknessScale, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.edges().style('opacity', edgeOpacity)
  }, [edgeOpacity, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyCommunityOpacity(cy, communityOpacity)
  }, [communityOpacity, elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    applyVisibility(cy, { minDegree, hiddenEntityTypes, hiddenRelTypes })
  }, [minDegree, hiddenEntityTypes, hiddenRelTypes, elements])

  const applyAndRunLayout = useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    setLayoutError(null)
    const config = layoutEngine === 'grid' ? '' : layoutConfigs[layoutEngine]
    const result = runLayout(cy, layoutEngine, config)
    if (!result.ok) setLayoutError(result.error ?? 'unknown layout error')
  }, [layoutEngine, layoutConfigs])

  const resetLayoutConfig = () => {
    if (layoutEngine === 'grid') return
    const def =
      layoutEngine === 'elk' ? JSON.stringify(ELK_PRESETS[elkPreset], null, 2) : DEFAULT_CONFIGS.fcose
    setLayoutConfigs((prev) => ({ ...prev, [layoutEngine]: def }))
    setLayoutError(null)
  }

  const handleElkPresetChange = (preset: string) => {
    setElkPreset(preset)
    const cfg = ELK_PRESETS[preset]
    if (!cfg) return
    setLayoutConfigs((prev) => ({ ...prev, elk: JSON.stringify(cfg, null, 2) }))
    setLayoutError(null)
  }

  const clearSelection = () => {
    cyRef.current?.elements().unselect()
  }

  const handleReload = () => {
    if (Number.isNaN(pendingTopN) || pendingTopN < 0) return
    if (Number.isNaN(pendingMaxDepth) || pendingMaxDepth < 0) return
    setTopN(pendingTopN)
    setMaxDepth(pendingMaxDepth)
    setSeedMetric(pendingSeedMetric)
  }

  const reloadDirty =
    pendingTopN !== topN ||
    pendingMaxDepth !== maxDepth ||
    pendingSeedMetric !== seedMetric

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

  const totalSelected =
    selectedNodes.length + selectedEdges.length + selectedCommunities.length

  const currentConfig = layoutEngine === 'grid' ? '' : layoutConfigs[layoutEngine]

  const legendItemClass = (hidden: boolean): string =>
    [
      'flex items-center gap-1 rounded px-1 py-0.5 text-left transition',
      hidden
        ? 'opacity-40 line-through hover:opacity-70'
        : 'hover:bg-[var(--color-surface-elevated)]',
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
          <Link to="/" className="text-[var(--color-accent)] hover:underline">Home</Link>
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
            {state.payload.edge_count.toLocaleString()} edges ·{' '}
            {state.payload.community_count.toLocaleString()} communities · seeds by{' '}
            <span className="font-mono">{state.payload.seed_metric}</span>, depth={state.payload.max_depth}
          </p>
        )}
      </header>

      <section className="flex min-h-0 flex-1">
        <div className="relative flex-1">
          {state.status === 'loading' && (
            <div data-testid="kg-loading" className="p-8">Loading knowledge graph…</div>
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
                }}
                style={{ width: '100%', height: '100%' }}
                wheelSensitivity={0.2}
              />
              {layoutReady && (
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
            <PanelSection
              title="Data"
              meta={
                <span className="font-mono" data-testid="kg-loaded-count">
                  {state.payload.node_count}/{state.payload.total_node_count}
                </span>
              }
            >
              <label className="flex flex-col gap-1" data-testid="kg-control-seed-metric">
                <span>Seed metric</span>
                <select
                  value={pendingSeedMetric}
                  onChange={(e) => setPendingSeedMetric(e.target.value as SeedMetric)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="edge_betweenness">edge betweenness</option>
                  <option value="node_betweenness">node betweenness</option>
                  <option value="degree">degree</option>
                </select>
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-top-n">
                <span>Max seed nodes (0 = all)</span>
                <input
                  type="number"
                  min={0}
                  step={10}
                  value={pendingTopN}
                  onChange={(e) => setPendingTopN(Number(e.target.value))}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                />
                <input
                  type="range"
                  min={0}
                  max={1000}
                  step={10}
                  value={pendingTopN}
                  onChange={(e) => setPendingTopN(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                  aria-label="Top-N slider"
                />
              </label>

              <label className="flex flex-col gap-1" data-testid="kg-control-max-depth">
                <span>Max depth (0 = unlimited)</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={pendingMaxDepth}
                  onChange={(e) => setPendingMaxDepth(Number(e.target.value))}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                />
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={pendingMaxDepth}
                  onChange={(e) => setPendingMaxDepth(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                  aria-label="Max-depth slider"
                />
              </label>

              <button
                type="button"
                onClick={handleReload}
                disabled={!reloadDirty}
                data-testid="kg-reload"
                className="self-start rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)] disabled:opacity-50"
              >
                Reload
              </button>

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

            <PanelSection title="Layout">
              <label className="flex flex-col gap-1" data-testid="kg-control-layout">
                <span>Engine</span>
                <select
                  value={layoutEngine}
                  onChange={(e) => {
                    setLayoutEngine(e.target.value as LayoutEngine)
                    setLayoutError(null)
                  }}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="fcose">fcose (force-directed)</option>
                  <option value="elk">elk (hierarchical)</option>
                  <option value="grid">grid</option>
                </select>
              </label>

              {layoutEngine === 'elk' && (
                <label className="flex flex-col gap-1" data-testid="kg-control-elk-preset">
                  <span>ELK preset</span>
                  <select
                    value={elkPreset}
                    onChange={(e) => handleElkPresetChange(e.target.value)}
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                  >
                    {Object.keys(ELK_PRESETS).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {layoutEngine !== 'grid' && (
                <label className="flex flex-col gap-1" data-testid="kg-control-config">
                  <span>Config (JSON)</span>
                  <textarea
                    value={currentConfig}
                    onChange={(e) =>
                      setLayoutConfigs((prev) => ({ ...prev, [layoutEngine]: e.target.value }))
                    }
                    rows={12}
                    spellCheck={false}
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 font-mono text-[10px] leading-snug"
                  />
                  {layoutError && (
                    <p
                      data-testid="kg-layout-error"
                      className="rounded border border-red-400 bg-red-50 px-2 py-1 text-[10px] text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
                    >
                      {layoutError}
                    </p>
                  )}
                </label>
              )}
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={applyAndRunLayout}
                  data-testid="kg-apply-layout"
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)]"
                >
                  Apply & run
                </button>
                {layoutEngine !== 'grid' && (
                  <button
                    type="button"
                    onClick={resetLayoutConfig}
                    data-testid="kg-reset-config"
                    className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs hover:border-[var(--color-accent)]"
                  >
                    Reset config
                  </button>
                )}
              </div>
            </PanelSection>

            <PanelSection title="Styling">
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

              <label className="flex flex-col gap-1" data-testid="kg-control-size-mode">
                <span>Node size by</span>
                <select
                  value={sizeMode}
                  onChange={(e) => setSizeMode(e.target.value as SizeMode)}
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="degree">degree</option>
                  <option value="betweenness">node betweenness</option>
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
                  onChange={(e) =>
                    setEdgeThicknessMode(e.target.value as EdgeThicknessMode)
                  }
                  className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] px-2 py-1 text-xs"
                >
                  <option value="uniform">uniform</option>
                  <option value="weight">weight</option>
                  <option value="edge_betweenness">edge betweenness</option>
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
                              {key}{' '}
                              <span className="text-[var(--color-muted-foreground)]">·{count}</span>
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
                              {key}{' '}
                              <span className="text-[var(--color-muted-foreground)]">·{count}</span>
                            </span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}

              <p className="text-[11px] leading-snug text-[var(--color-muted-foreground)]">
                Legend: single-click toggles, double-click isolates (or restores all if already
                isolated). Click a node, edge, or community to select. Ctrl/Cmd-click adds.
              </p>
            </PanelSection>

            <PanelSection
              title="Selection"
              meta={
                <span className="font-mono" data-testid="kg-selection-count">
                  {totalSelected}
                </span>
              }
            >
              {totalSelected === 0 ? (
                <p className="text-[12px] text-[var(--color-muted-foreground)]">Nothing selected.</p>
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
                              {n.node_betweenness !== null && (
                                <span>bc = {n.node_betweenness.toFixed(4)}</span>
                              )}
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
                            {e.rel_type && (
                              <div className="mt-1 text-[12px] leading-snug">{e.rel_type}</div>
                            )}
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
          </RightPanel>
        )}
      </section>
    </main>
  )
}
