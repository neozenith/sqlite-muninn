import { useState, useRef, useCallback } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { GraphCanvas } from '@/components/GraphCanvas'
import { useGraphs, useSubgraph, useBFS, useCommunities, useCentrality } from '@/hooks/useGraphData'
import {
  toCytoscapeElements,
  applyCommunityColors,
  applyCentralitySizing,
  highlightBfsNodes,
  applyCommunityGrouping,
} from '@/lib/transforms/cytoscape'
import { LAYOUT_OPTIONS } from '@/lib/constants'
import type { LayoutName } from '@/lib/constants'
import type { CentralityMeasure, EdgeTableInfo } from '@/lib/types'

export function GraphPage() {
  const { dataset } = useParams<{ dataset: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const edgeTable = dataset ? decodeURIComponent(dataset) : null

  const measureParam = (searchParams.get('measure') ?? 'betweenness') as CentralityMeasure
  const layoutParam = (searchParams.get('layout') ?? 'fcose') as LayoutName
  const groupParam = searchParams.get('group') !== '0'
  const nodeParam = searchParams.get('node') ?? null

  const [resolution] = useState(1.0)
  const [maxDepth] = useState(3)

  const cyRef = useRef<cytoscape.Core | null>(null)

  const { data: graphs, isLoading: graphsLoading } = useGraphs()
  const { data: subgraph, isLoading: subgraphLoading, isError: subgraphError } = useSubgraph(edgeTable)
  const { data: bfsData } = useBFS(edgeTable, nodeParam, maxDepth)
  const { data: communities } = useCommunities(edgeTable, resolution)
  const { data: centrality } = useCentrality(edgeTable, measureParam)

  const setMeasure = (m: CentralityMeasure) => {
    setSearchParams((prev) => {
      if (m === 'betweenness') prev.delete('measure')
      else prev.set('measure', m)
      return prev
    })
  }

  const setLayout = (l: LayoutName) => {
    setSearchParams((prev) => {
      if (l === 'fcose') prev.delete('layout')
      else prev.set('layout', l)
      return prev
    })
  }

  const setCommunityGrouping = (on: boolean) => {
    setSearchParams((prev) => {
      if (on) prev.delete('group')
      else prev.set('group', '0')
      return prev
    })
  }

  const setSelectedNode = (node: string | null) => {
    setSearchParams((prev) => {
      if (node == null) prev.delete('node')
      else prev.set('node', node)
      return prev
    })
  }

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.layout({ name: layoutParam, animate: true, animationDuration: 300 } as any).run()
    }
  }, [layoutParam])

  // No dataset selected — show dataset picker
  if (!edgeTable) {
    return (
      <div className="max-w-2xl mx-auto space-y-4">
        <h2 className="text-lg font-semibold">Edge Tables</h2>
        <p className="text-sm text-muted-foreground">Select an edge table to explore its graph.</p>
        {graphsLoading ? (
          <div className="text-muted-foreground">Loading edge tables...</div>
        ) : !graphs?.length ? (
          <div className="text-muted-foreground">No edge tables found in this database.</div>
        ) : (
          <div className="grid gap-3">
            {graphs.map((g: EdgeTableInfo) => (
              <Card
                key={g.table_name}
                className="hover:border-primary/50 transition-colors cursor-pointer"
                onClick={() => navigate(`/graph/${encodeURIComponent(g.table_name)}`)}
              >
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">{g.table_name}</CardTitle>
                </CardHeader>
                <CardContent className="flex gap-2">
                  <Badge variant="secondary">{g.edge_count} edges</Badge>
                  <Badge variant="outline">
                    {g.src_col} → {g.dst_col}
                  </Badge>
                  {g.weight_col && <Badge variant="outline">weight: {g.weight_col}</Badge>}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    )
  }

  // Build Cytoscape elements
  let elements = subgraph ? toCytoscapeElements(subgraph.nodes, subgraph.edges) : []

  if (communities?.node_community) {
    elements = applyCommunityColors(elements, communities.node_community)
  }
  if (centrality?.scores) {
    elements = applyCentralitySizing(elements, centrality.scores)
  }
  if (groupParam && communities?.node_community) {
    elements = applyCommunityGrouping(elements, communities.node_community)
  }
  if (bfsData?.nodes) {
    const bfsNodeIds = new Set(bfsData.nodes.map((n) => n.node))
    elements = highlightBfsNodes(elements, bfsNodeIds)
  }

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar */}
      <div className="w-64 flex flex-col gap-3 shrink-0">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Edge Table</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-xs bg-background"
              value={edgeTable}
              onChange={(e) => navigate(`/graph/${encodeURIComponent(e.target.value)}`)}
            >
              {graphs?.map((g: EdgeTableInfo) => (
                <option key={g.table_name} value={g.table_name}>
                  {g.table_name}
                </option>
              ))}
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Centrality</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={measureParam}
              onChange={(e) => setMeasure(e.target.value as CentralityMeasure)}
            >
              <option value="degree">Degree</option>
              <option value="betweenness">Betweenness</option>
              <option value="closeness">Closeness</option>
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Layout</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={layoutParam}
              onChange={(e) => setLayout(e.target.value as LayoutName)}
            >
              {LAYOUT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={groupParam} onChange={(e) => setCommunityGrouping(e.target.checked)} />
              Group by community
            </label>
            <Button variant="outline" size="sm" className="w-full h-7 text-xs" onClick={runLayout}>
              Re-run layout
            </Button>
          </CardContent>
        </Card>

        {subgraph && (
          <Card data-testid="graph-stats" data-node-count={subgraph.node_count}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Graph Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-xs">
              <div>
                Nodes: <Badge data-testid="graph-node-count" variant="secondary">{subgraph.node_count}</Badge>
              </div>
              <div>
                Edges: <Badge data-testid="graph-edge-count" variant="secondary">{subgraph.edge_count}</Badge>
              </div>
              {communities && (
                <div>
                  Communities: <Badge variant="secondary">{communities.community_count}</Badge>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {nodeParam && bfsData && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">BFS from {nodeParam}</CardTitle>
            </CardHeader>
            <CardContent className="text-xs">
              <div>Reachable: {bfsData.count} nodes</div>
              <div className="mt-1 space-y-0.5 max-h-40 overflow-y-auto">
                {bfsData.nodes.slice(0, 15).map((n) => (
                  <div key={n.node} className="flex justify-between">
                    <span className="truncate">{n.node}</span>
                    <Badge variant="outline" className="text-[10px] ml-1">
                      d={n.depth}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {centrality?.scores && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Top Centrality</CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-0.5 max-h-40 overflow-y-auto">
              {centrality.scores.slice(0, 10).map((s) => (
                <div key={s.node} className="flex justify-between">
                  <span className="truncate">{s.node}</span>
                  <span className="font-mono text-muted-foreground">{s.centrality.toFixed(4)}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative" data-testid="graph-canvas-area">
        {subgraphLoading ? (
          <div
            data-testid="graph-state-loading"
            className="absolute inset-0 flex items-center justify-center text-muted-foreground"
          >
            Loading graph...
          </div>
        ) : subgraphError ? (
          <div
            data-testid="graph-state-error"
            className="absolute inset-0 flex items-center justify-center text-muted-foreground"
          >
            Edge table not found in current database
          </div>
        ) : (
          <GraphCanvas elements={elements} layout={layoutParam} onNodeSelect={setSelectedNode} cyRef={cyRef} />
        )}
      </div>
    </div>
  )
}
