/** KG Query page: 3-panel search — FTS | Embedding 3D | Knowledge Graph. */

import { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import type cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { EmbeddingCanvas } from '@/components/EmbeddingCanvas'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphControls } from '@/components/GraphControls'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '@/lib/fcose'
import type { FcoseParams } from '@/lib/fcose'
import { useKGSearch } from '@/hooks/useKGPipeline'
import type { KGSearchResult, EmbeddingPoint } from '@/lib/types'
import { applyCommunityColors, applyCommunityGrouping } from '@/lib/transforms/cytoscape'
import type { CytoscapeElement } from '@/lib/transforms/cytoscape'
import { communityColor } from '@/lib/constants'

/**
 * Rank-based color gradient: red (top) → orange → purple (lowest).
 * Matches the WASM demo's rankColor function.
 */
function rankColor(rank: number, total: number): readonly [number, number, number] {
  if (rank === 0) return [255, 60, 60]
  if (total <= 1) return [255, 60, 60]
  const t = (rank - 1) / (total - 1)
  const r = Math.round(255 + (140 - 255) * t)
  const g = Math.round(140 + (60 - 140) * t)
  const b = Math.round(50 + (220 - 50) * t)
  return [r, g, b]
}

/** Convert VSS results to EmbeddingPoint format for EmbeddingCanvas. */
function toEmbeddingPoints(result: KGSearchResult): EmbeddingPoint[] {
  const total = result.vss_results.length
  return result.vss_results.map((vss, i) => ({
    id: vss.chunk_id,
    x: vss.x3d ?? Math.cos((i / total || 1) * Math.PI * 2) * 30,
    y: vss.y3d ?? (vss.similarity - 0.5) * 60,
    z: vss.z3d ?? Math.sin((i / total || 1) * Math.PI * 2) * 30,
    label: `Chunk #${vss.chunk_id} (${(vss.similarity * 100).toFixed(1)}%)`,
    metadata: {
      similarity: vss.similarity,
      rank: i,
      text: `Chunk #${vss.chunk_id} \u00b7 ${(vss.similarity * 100).toFixed(1)}%\n${vss.text}`,
    },
    color: rankColor(i, total),
  }))
}

/** Build Cytoscape elements from graph search results. */
function toCytoscapeElements(result: KGSearchResult): CytoscapeElement[] {
  const elements: CytoscapeElement[] = []
  const nodeSet = new Set<string>()

  for (const n of result.graph_nodes) {
    nodeSet.add(n.name)
    elements.push({
      data: {
        id: n.name,
        label: n.name,
        size: n.is_anchor ? 40 : 20 + Math.min(n.similarity, 1) * 20,
        color: n.is_anchor ? '#f87171' : n.similarity > 0.5 ? '#f59e0b' : n.similarity > 0.2 ? '#8b5cf6' : '#6b7280',
      },
    })
  }

  for (const e of result.graph_edges) {
    if (nodeSet.has(e.src) && nodeSet.has(e.dst)) {
      elements.push({
        data: {
          id: `${e.src}-${e.rel}-${e.dst}`,
          source: e.src,
          target: e.dst,
          weight: 1,
        },
      })
    }
  }

  return elements
}

export function KGQueryPage() {
  const [queryText, setQueryText] = useState('')
  const [resolution, setResolution] = useState<number | undefined>(undefined)
  const kgSearch = useKGSearch()

  const [fcoseParams, setFcoseParams] = useState<FcoseParams>(DEFAULT_FCOSE_PARAMS)
  const cyRef = useRef<cytoscape.Core | null>(null)

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current
        .layout({
          name: 'fcose',
          animate: true,
          animationDuration: 300,
          ...toLayoutOptions(fcoseParams),
        } as unknown as cytoscape.LayoutOptions)
        .run()
    }
  }, [fcoseParams])

  const handleQuery = useCallback(() => {
    if (queryText.trim()) {
      kgSearch.mutate({ query: queryText.trim(), resolution })
    }
  }, [queryText, kgSearch, resolution])

  // Debounced search-as-you-type (300ms)
  useEffect(() => {
    if (!queryText.trim()) return
    const timer = setTimeout(() => {
      kgSearch.mutate({ query: queryText.trim(), resolution })
    }, 300)
    return () => clearTimeout(timer)
  }, [queryText]) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-query when resolution changes (community grouping only — no need to re-type)
  useEffect(() => {
    if (!queryText.trim() || resolution === undefined) return
    kgSearch.mutate({ query: queryText.trim(), resolution })
  }, [resolution]) // eslint-disable-line react-hooks/exhaustive-deps

  // Derived data from search results
  const embeddingPoints = useMemo(() => (kgSearch.data ? toEmbeddingPoints(kgSearch.data) : []), [kgSearch.data])

  const graphElements = useMemo(() => {
    if (!kgSearch.data) return []
    let elements = toCytoscapeElements(kgSearch.data)
    const nc = kgSearch.data.node_community
    if (nc && Object.keys(nc).length > 0) {
      elements = applyCommunityColors(elements, nc)
      elements = applyCommunityGrouping(elements, nc, kgSearch.data.community_labels)
    }
    return elements
  }, [kgSearch.data])

  // Build community legend: [{id, label, color, memberCount}]
  const communityLegend = useMemo(() => {
    const nc = kgSearch.data?.node_community
    if (!nc || Object.keys(nc).length === 0) return []
    const labels = kgSearch.data?.community_labels ?? {}
    // Count members per community
    const counts = new Map<number, number>()
    for (const cid of Object.values(nc)) {
      counts.set(cid, (counts.get(cid) ?? 0) + 1)
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id, count]) => ({
        id,
        label: labels[String(id)] ?? `Community ${id}`,
        color: communityColor(id),
        memberCount: count,
      }))
  }, [kgSearch.data])

  // Available resolutions from the server (only populated with precomputed communities)
  const availableResolutions = kgSearch.data?.available_resolutions ?? []

  const hasResults = kgSearch.data != null

  return (
    <div className="max-w-screen-2xl mx-auto space-y-4">
      {/* Search bar */}
      <div className="flex gap-2">
        <Input
          placeholder="Search the knowledge graph... e.g., division of labor and trade"
          value={queryText}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQueryText(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleQuery()}
          className="flex-1"
        />
        {availableResolutions.length > 1 && (
          <select
            value={resolution ?? ''}
            onChange={(e) => setResolution(e.target.value ? Number(e.target.value) : undefined)}
            className="rounded border border-input bg-background px-2 py-1 text-xs"
            data-testid="resolution-selector"
          >
            <option value="">Resolution: auto</option>
            {availableResolutions.map((r) => (
              <option key={r} value={r}>
                {r === 0.25 ? 'Coarse' : r === 1.0 ? 'Medium' : r === 3.0 ? 'Fine' : `\u03B3=${r}`} ({r})
              </option>
            ))}
          </select>
        )}
        <Button onClick={handleQuery} disabled={kgSearch.isPending || !queryText.trim()}>
          {kgSearch.isPending ? 'Searching...' : 'Search'}
        </Button>
      </div>

      {/* Error display */}
      {kgSearch.error && (
        <div className="text-destructive text-sm p-3 bg-destructive/10 rounded">
          Search failed: {(kgSearch.error as Error).message}
        </div>
      )}

      {/* 3-Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: FTS Results */}
        <Card className="flex flex-col">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              FTS Results
              {kgSearch.data && (
                <span className="text-xs text-muted-foreground font-normal">
                  {kgSearch.data.fts_results.length} results
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto max-h-[500px] space-y-2">
            {hasResults && kgSearch.data!.fts_results.length > 0 ? (
              kgSearch.data!.fts_results.map((r) => (
                <div key={r.chunk_id} className="border rounded p-2 text-xs space-y-1">
                  <span className="text-primary font-medium">Chunk #{r.chunk_id}</span>
                  <p className="text-muted-foreground">{r.text}</p>
                </div>
              ))
            ) : hasResults ? (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                No FTS matches found.
              </div>
            ) : (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                Search to see text matches
              </div>
            )}
          </CardContent>
        </Card>

        {/* Center: Embedding Space (3D) */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              Embedding Space (3D)
              {kgSearch.data && (
                <span className="text-xs text-muted-foreground font-normal">
                  {kgSearch.data.vss_results.length} points
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col">
            {embeddingPoints.length > 0 ? (
              <>
                <EmbeddingCanvas points={embeddingPoints} dimensions={3} className="h-[250px]" />
                <div className="flex-1 overflow-y-auto max-h-[200px] space-y-1 mt-2 border-t pt-2">
                  {kgSearch.data!.vss_results.map((vss) => {
                    const pct = vss.similarity * 100
                    return (
                      <div key={vss.chunk_id} className="border rounded p-2 text-xs space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-primary font-medium">Chunk #{vss.chunk_id}</span>
                          <span
                            className={`font-mono ${pct > 50 ? 'text-amber-400' : pct > 20 ? 'text-purple-400' : 'text-muted-foreground'}`}
                          >
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-muted-foreground">{vss.text}</p>
                      </div>
                    )
                  })}
                </div>
              </>
            ) : (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                {hasResults ? 'No embedding results.' : 'Search to visualize embeddings in 3D'}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right: Knowledge Graph */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              Knowledge Graph
              {kgSearch.data && (
                <span className="text-xs text-muted-foreground font-normal">
                  {kgSearch.data.graph_nodes.length} nodes
                  {(kgSearch.data.community_count ?? 0) > 0 && (
                    <> · {kgSearch.data.community_count} communities</>
                  )}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col">
            {graphElements.length > 0 ? (
              <>
                <GraphCanvas
                  elements={graphElements}
                  layout="fcose"
                  layoutOptions={toLayoutOptions(fcoseParams)}
                  onNodeSelect={() => {}}
                  cyRef={cyRef}
                  className="h-[350px]"
                />
                <GraphControls params={fcoseParams} onChange={setFcoseParams} onRunLayout={runLayout} />
                {communityLegend.length > 0 && (
                  <div className="mt-2 border-t pt-2 space-y-1" data-testid="community-legend">
                    <div className="text-xs font-medium text-muted-foreground">
                      Communities ({communityLegend.length})
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-1">
                      {communityLegend.map((c) => (
                        <div key={c.id} className="flex items-center gap-1 text-xs">
                          <span
                            className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ backgroundColor: c.color }}
                          />
                          <span className="truncate max-w-[140px]" title={c.label}>
                            {c.label}
                          </span>
                          <span className="text-muted-foreground">({c.memberCount})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                {hasResults ? 'No graph results.' : 'Search to explore the knowledge graph'}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* No results message */}
      {!hasResults && !kgSearch.isPending && (
        <div className="text-muted-foreground text-sm text-center py-8">
          Enter a search query to explore the knowledge graph.
        </div>
      )}
    </div>
  )
}
