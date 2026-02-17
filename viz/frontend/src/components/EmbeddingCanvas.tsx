import { useMemo } from 'react'
import { DeckGL } from '@deck.gl/react'
import { OrthographicView, OrbitView } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'
import { SimpleMeshLayer } from '@deck.gl/mesh-layers'
import { SphereGeometry } from '@luma.gl/engine'
import { toScatterplotData, computeViewState } from '@/lib/transforms/deckgl'
import type { ScatterplotDatum } from '@/lib/transforms/deckgl'
import type { EmbeddingPoint, SearchNeighbor } from '@/lib/types'
import type { Color } from '@deck.gl/core'

const SPHERE_GEOMETRY = new SphereGeometry({ radius: 1, nlat: 16, nlong: 16 })

interface EmbeddingCanvasProps {
  points: EmbeddingPoint[]
  searchResults?: SearchNeighbor[]
  selectedId?: number
  onSelect?: (id: number) => void
  dimensions?: 2 | 3
  className?: string
}

export function EmbeddingCanvas({
  points,
  searchResults,
  selectedId,
  onSelect,
  dimensions = 2,
  className = 'h-full',
}: EmbeddingCanvasProps) {
  const is3D = dimensions === 3
  const pickable = !!onSelect

  const data = useMemo(() => toScatterplotData(points, searchResults, selectedId), [points, searchResults, selectedId])

  const initialViewState = useMemo(
    () => (is3D ? computeViewState(data, 800, 600, true) : computeViewState(data)),
    [data, is3D],
  )

  const sphereScale = useMemo(() => {
    if (!is3D || data.length === 0) return 1
    let minX = Infinity,
      maxX = -Infinity
    let minY = Infinity,
      maxY = -Infinity
    let minZ = Infinity,
      maxZ = -Infinity
    for (const d of data) {
      const [x, y] = d.position
      const z = d.position[2] ?? 0
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }
    const avgRange = (maxX - minX + (maxY - minY) + (maxZ - minZ)) / 3 || 1
    return avgRange / Math.sqrt(data.length) / 16
  }, [data, is3D])

  const getPos = (d: ScatterplotDatum): [number, number, number] =>
    d.position.length === 3 ? (d.position as [number, number, number]) : [d.position[0], d.position[1], 0]

  const handleClick = onSelect
    ? (info: { object?: unknown }) => {
        if (info.object) onSelect((info.object as ScatterplotDatum).id)
      }
    : undefined

  const view = is3D ? new OrbitView({ orbitAxis: 'Y' }) : new OrthographicView({})

  const layer = is3D
    ? new SimpleMeshLayer<ScatterplotDatum>({
        id: 'embeddings-3d',
        data,
        mesh: SPHERE_GEOMETRY,
        getPosition: getPos,
        getColor: (d: ScatterplotDatum): Color => [d.color[0], d.color[1], d.color[2], d.opacity],
        getScale: (d: ScatterplotDatum) => {
          const s = d.radius * sphereScale
          return [s, s, s]
        },
        pickable,
        ...(handleClick ? { onClick: handleClick } : {}),
      })
    : new ScatterplotLayer<ScatterplotDatum>({
        id: 'embeddings',
        data,
        getPosition: getPos,
        getFillColor: (d: ScatterplotDatum): Color => [d.color[0], d.color[1], d.color[2], d.opacity],
        getRadius: (d: ScatterplotDatum) => d.radius,
        radiusMinPixels: 2,
        radiusMaxPixels: pickable ? 15 : 10,
        pickable,
        ...(handleClick ? { onClick: handleClick } : {}),
      })

  if (points.length === 0) {
    return (
      <div className={`${className} flex items-center justify-center text-muted-foreground text-xs`}>
        No embedding data
      </div>
    )
  }

  return (
    <div className={`${className} relative bg-muted/30 rounded-lg overflow-hidden`}>
      <DeckGL
        views={view}
        initialViewState={initialViewState}
        layers={[layer]}
        controller={true}
        getTooltip={({ object }: { object?: ScatterplotDatum }) =>
          object && object.text
            ? {
                html: `<div style="max-width:250px">${object.text.replace(/\n/g, '<br/>')}</div>`,
                style: {
                  backgroundColor: 'rgba(0,0,0,0.85)',
                  color: 'white',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                },
              }
            : object
              ? `${object.label || `Point #${object.id}`}`
              : null
        }
      />
    </div>
  )
}
