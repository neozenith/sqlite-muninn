import { useState } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { EmbeddingCanvas } from '@/components/EmbeddingCanvas'
import { useIndexes, useEmbeddings, useVSSSearch, useVSSTextSearch } from '@/hooks/useVSSData'
import type { SearchNeighbor, HnswIndexInfo } from '@/lib/types'

export function EmbeddingsPage() {
  const { dataset } = useParams<{ dataset: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const dimensions = (searchParams.get('d') === '2' ? 2 : 3) as 2 | 3
  const k = Number(searchParams.get('k')) || 20
  const queryFromUrl = searchParams.get('q') ?? ''
  const pointFromUrl = searchParams.get('point')
  const selectedPointId = pointFromUrl ? Number(pointFromUrl) : null

  const [searchText, setSearchText] = useState(queryFromUrl)

  const indexName = dataset ? decodeURIComponent(dataset) : null

  const { data: indexes } = useIndexes()
  const {
    data: embeddings,
    isLoading: embeddingsLoading,
    isError: embeddingsError,
  } = useEmbeddings(indexName, dimensions)
  const { data: searchResults } = useVSSSearch(indexName, selectedPointId, k)
  const textSearch = useVSSTextSearch(indexName, searchText, k)

  const activeResults = textSearch.data ?? searchResults

  const setDimensions = (d: 2 | 3) => {
    setSearchParams((prev) => {
      if (d === 3) prev.delete('d')
      else prev.set('d', String(d))
      return prev
    })
  }

  const setSelectedPointId = (id: number | null) => {
    setSearchParams((prev) => {
      if (id == null) prev.delete('point')
      else prev.set('point', String(id))
      return prev
    })
  }

  const setKParam = (newK: number) => {
    setSearchParams((prev) => {
      if (newK === 20) prev.delete('k')
      else prev.set('k', String(newK))
      return prev
    })
  }

  const is3D = dimensions === 3
  const selectedNeighbor = activeResults?.neighbors.find((n: SearchNeighbor) => n.id === selectedPointId)

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar */}
      <div className="w-64 flex flex-col gap-3 shrink-0">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Index</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-xs bg-background"
              value={indexName}
              onChange={(e) => navigate(`/embeddings/${encodeURIComponent(e.target.value)}`)}
            >
              {indexes?.map((idx: HnswIndexInfo) => (
                <option key={idx.name} value={idx.name}>
                  {idx.name}
                </option>
              ))}
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">View</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1">
              <Button
                variant={!is3D ? 'default' : 'outline'}
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={() => setDimensions(2)}
              >
                2D
              </Button>
              <Button
                variant={is3D ? 'default' : 'outline'}
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={() => setDimensions(3)}
              >
                3D
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Text Search</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1">
              <Input
                placeholder="Search text..."
                value={searchText}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
                className="h-8 text-xs"
              />
            </div>
            {textSearch.isLoading && <div className="text-xs text-muted-foreground mt-1">Searching...</div>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Search K</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                min={1}
                max={100}
                value={k}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setKParam(Number(e.target.value))}
                className="w-20 h-8"
              />
              <span className="text-xs text-muted-foreground">neighbors</span>
            </div>
          </CardContent>
        </Card>

        {embeddings && (
          <Card data-testid="embedding-stats" data-count={embeddings.count}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-xs">
              <div>
                Points:{' '}
                <Badge data-testid="embedding-point-count" variant="secondary">
                  {embeddings.count}
                </Badge>
              </div>
              <div>
                Original dim: <Badge variant="secondary">{embeddings.original_dimensions}</Badge>
              </div>
              <div>
                Projected: <Badge variant="secondary">{embeddings.projected_dimensions}D</Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {activeResults && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Search Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {activeResults.neighbors.slice(0, 10).map((n: SearchNeighbor) => (
                  <div
                    key={n.id}
                    className="text-xs p-1 rounded cursor-pointer hover:bg-muted truncate"
                    onClick={() => setSelectedPointId(n.id)}
                  >
                    <span className="font-mono">{n.distance.toFixed(4)}</span>{' '}
                    <span className="text-muted-foreground">{n.label || `#${n.id}`}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative" data-testid="embedding-canvas-area">
        {embeddingsLoading ? (
          <div
            data-testid="embedding-state-loading"
            className="absolute inset-0 flex items-center justify-center text-muted-foreground"
          >
            Projecting embeddings with UMAP...
          </div>
        ) : embeddingsError ? (
          <div
            data-testid="embedding-state-error"
            className="absolute inset-0 flex items-center justify-center text-muted-foreground"
          >
            Index not found in current database
          </div>
        ) : embeddings ? (
          <EmbeddingCanvas
            points={embeddings.points}
            searchResults={activeResults?.neighbors}
            selectedId={selectedPointId ?? undefined}
            onSelect={setSelectedPointId}
            dimensions={dimensions}
          />
        ) : (
          <div
            data-testid="embedding-state-no-selection"
            className="absolute inset-0 flex items-center justify-center text-muted-foreground"
          >
            No index selected
          </div>
        )}
      </div>

      {/* Detail Panel */}
      {selectedNeighbor && (
        <div className="w-56 shrink-0">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Selected Point</CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-1">
              <div>ID: {selectedNeighbor.id}</div>
              <div>Distance: {selectedNeighbor.distance.toFixed(6)}</div>
              {selectedNeighbor.label && <div>Label: {selectedNeighbor.label}</div>}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
