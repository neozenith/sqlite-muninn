import cytoscape from 'cytoscape'
// cytoscape-fcose doesn't ship perfect types for its default export — the cast
// below hands the plugin to cytoscape.use() without relaxing our strict mode.
import fcose from 'cytoscape-fcose'
import { useEffect, useMemo, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import { Link, useParams } from 'react-router-dom'
import { ApiError, type KGPayload, fetchKG } from '../lib/api-client'
import { useTheme } from '../lib/ThemeProvider'

cytoscape.use(fcose as unknown as cytoscape.Ext)

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; payload: KGPayload }

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
      },
    }))

  return [...parents, ...children, ...edges]
}

const LIGHT_STYLESHEET: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
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
    },
  },
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
]

const DARK_STYLESHEET: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': '#7AB3FF',
      label: 'data(label)',
      'font-size': '8px',
      color: '#e6e6e6',
      'text-wrap': 'ellipsis',
      'text-max-width': '80px',
      'text-valign': 'center',
      'text-halign': 'center',
      width: 14,
      height: 14,
    },
  },
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
]

export function KGPage() {
  const { databaseId, tableId } = useParams<{ databaseId: string; tableId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  const [layoutReady, setLayoutReady] = useState(false)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const { resolved } = useTheme()
  const stylesheet = resolved === 'dark' ? DARK_STYLESHEET : LIGHT_STYLESHEET

  useEffect(() => {
    if (!databaseId || !tableId) return
    setState({ status: 'loading' })
    setLayoutReady(false)
    fetchKG(databaseId, tableId)
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
  }, [databaseId, tableId])

  const elements = useMemo(
    () => (state.status === 'ready' ? buildElements(state.payload) : []),
    [state],
  )

  useEffect(() => {
    const cy = cyRef.current
    if (!cy || state.status !== 'ready') return
    // Two-phase layout: (1) quick `grid` to paint something deterministic
    // immediately — flips ready so downstream viewers (E2E, humans) know the
    // graph is mounted and interactive; (2) fcose in the background to
    // refine positions. This keeps ready-semantics tight: "data rendered"
    // rather than "physics converged", which matters for 6K-node graphs
    // where fcose can take minutes.
    setLayoutReady(false)
    cy.layout({ name: 'grid', animate: false } as cytoscape.LayoutOptions).run()
    setLayoutReady(true)
    queueMicrotask(() => {
      if (!cyRef.current) return
      const fcoseLayout = cyRef.current.layout({
        name: 'fcose',
        animate: false,
        randomize: false,
        nodeRepulsion: 4500,
        idealEdgeLength: 50,
        tile: true,
      } as unknown as cytoscape.LayoutOptions)
      fcoseLayout.run()
    })
  }, [elements, state.status])

  // Recolor in place on theme change so we don't have to rebuild the layout.
  useEffect(() => {
    cyRef.current?.style(stylesheet as unknown as cytoscape.StylesheetStyle[])
  }, [stylesheet])

  return (
    <main
      className="flex min-h-screen flex-col bg-[var(--color-surface)] text-[var(--color-foreground)]"
      data-testid="kg-page"
      data-database-id={databaseId ?? ''}
      data-table-id={tableId ?? ''}
    >
      <header className="border-b border-[var(--color-border-subtle)] p-4">
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
              ? ` (of ${state.payload.total_node_count.toLocaleString()} — showing top by degree)`
              : ''}
            {' · '}
            {state.payload.edge_count.toLocaleString()} edges ·{' '}
            {state.payload.community_count.toLocaleString()} communities (res={state.payload.resolution})
          </p>
        )}
      </header>

      <section className="relative flex-1">
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
      </section>
    </main>
  )
}
