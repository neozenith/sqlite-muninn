import { OrbitView } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'
import DeckGL from '@deck.gl/react'
import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, type EmbedPayload, type EmbedPoint, fetchEmbed } from '../lib/api-client'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; payload: EmbedPayload }

/** djb2 string hash → 0..1 float for deterministic palette selection. */
const hashHue = (s: string): number => {
  let h = 5381
  for (let i = 0; i < s.length; i++) h = (h * 33) ^ s.charCodeAt(i)
  return Math.abs(h) % 360
}

/** HSL (h, 70%, 55%) → [r, g, b, 255] so Deck.GL ScatterplotLayer can render it. */
const hueToRgba = (hue: number): [number, number, number, number] => {
  const s = 0.7
  const l = 0.55
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l - c / 2
  let r = 0,
    g = 0,
    b = 0
  if (hue < 60) [r, g, b] = [c, x, 0]
  else if (hue < 120) [r, g, b] = [x, c, 0]
  else if (hue < 180) [r, g, b] = [0, c, x]
  else if (hue < 240) [r, g, b] = [0, x, c]
  else if (hue < 300) [r, g, b] = [x, 0, c]
  else [r, g, b] = [c, 0, x]
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255), 255]
}

interface DataStats {
  centroid: [number, number, number]
  /** Largest axis-aligned half-extent — used to size zoom + point radius. */
  extent: number
}

const VIEWPORT_FALLBACK = 700

const dataStats = (points: EmbedPoint[]): DataStats => {
  if (points.length === 0) return { centroid: [0, 0, 0], extent: 1 }
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity
  let sx = 0,
    sy = 0,
    sz = 0
  for (const p of points) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.z < minZ) minZ = p.z
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
    if (p.z > maxZ) maxZ = p.z
    sx += p.x
    sy += p.y
    sz += p.z
  }
  const n = points.length
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1) / 2
  return { centroid: [sx / n, sy / n, sz / n], extent }
}

interface EmbedViewState {
  target: [number, number, number]
  rotationX: number
  rotationOrbit: number
  zoom: number
}

export function EmbedPage() {
  const { databaseId, tableId } = useParams<{ databaseId: string; tableId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    if (!databaseId || !tableId) return
    setState({ status: 'loading' })
    fetchEmbed(databaseId, tableId)
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

  const stats = useMemo<DataStats>(
    () => (state.status === 'ready' ? dataStats(state.payload.points) : { centroid: [0, 0, 0], extent: 1 }),
    [state],
  )

  const layers = useMemo(() => {
    if (state.status !== 'ready') return []
    const radius = Math.max(stats.extent * 0.005, 0.02)
    return [
      new ScatterplotLayer<EmbedPoint>({
        id: 'umap-points',
        data: state.payload.points,
        pickable: true,
        getPosition: (p: EmbedPoint) => [p.x, p.y, p.z],
        getRadius: radius,
        radiusUnits: 'common',
        getFillColor: (p: EmbedPoint) =>
          p.category ? hueToRgba(hashHue(p.category)) : [80, 150, 220, 255],
        updateTriggers: {
          getFillColor: state.payload.table_id,
          getRadius: stats.extent,
        },
      }),
    ]
  }, [state, stats])

  const initialViewState = useMemo<EmbedViewState>(() => {
    // OrbitView zoom: higher = closer. log2(viewport / dataExtent) frames the
    // data so it fills ~half the viewport; subtract 0.5 so there's margin.
    const zoom = Math.max(Math.log2(VIEWPORT_FALLBACK / (stats.extent * 2)) - 0.5, 0)
    return { target: stats.centroid, rotationX: 30, rotationOrbit: 30, zoom }
  }, [stats])

  return (
    <main className="min-h-screen flex flex-col" data-testid="embed-page" data-database-id={databaseId ?? ''} data-table-id={tableId ?? ''}>
      <header className="p-4 border-b">
        <nav className="flex gap-4 text-sm">
          <Link to="/" className="text-blue-600 hover:underline">Home</Link>
          <Link to={`/${databaseId}/`} className="text-blue-600 hover:underline">
            ← {databaseId}
          </Link>
        </nav>
        <h1 className="text-2xl font-bold mt-2">
          3D UMAP: <span className="font-mono">{tableId}</span>
        </h1>
        {state.status === 'ready' && (
          <p className="text-sm text-muted-foreground">
            {state.payload.count.toLocaleString()} points
          </p>
        )}
      </header>

      <section className="flex-1 relative">
        {state.status === 'loading' && (
          <div data-testid="embed-loading" className="p-8">Loading embeddings…</div>
        )}
        {state.status === 'error' && (
          <div data-testid="embed-error" className="m-8 rounded border border-red-400 bg-red-50 p-4 text-red-800">
            <p className="font-semibold">Failed to load embeddings</p>
            <p className="text-sm">{state.message}</p>
          </div>
        )}
        {state.status === 'ready' && (
          <div className="absolute inset-0">
            <DeckGL
              views={new OrbitView({ orbitAxis: 'Y', fovy: 50 })}
              initialViewState={initialViewState}
              controller={true}
              layers={layers}
              getTooltip={({ object }: { object?: EmbedPoint | null }) =>
                object
                  ? {
                      html: `<div style="padding:6px 8px; background:#111; color:#fff; border-radius:4px; max-width:320px">
                        <strong>#${object.id}</strong>
                        ${object.category ? ` <em style="opacity:0.8">(${object.category})</em>` : ''}
                        <br/>${object.label}
                      </div>`,
                    }
                  : null
              }
            />
            <div
              data-testid="embed-canvas-ready"
              data-point-count={state.payload.count}
              className="hidden"
            />
          </div>
        )}
      </section>
    </main>
  )
}
