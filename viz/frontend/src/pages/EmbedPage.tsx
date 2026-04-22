import { AmbientLight, DirectionalLight, LightingEffect, OrbitView } from '@deck.gl/core'
import { SimpleMeshLayer } from '@deck.gl/mesh-layers'
import DeckGL from '@deck.gl/react'
import { SphereGeometry } from '@luma.gl/engine'
import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, type EmbedPayload, type EmbedPoint, fetchEmbed } from '../lib/api-client'
import { useTheme } from '../lib/ThemeProvider'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; payload: EmbedPayload }

const hashHue = (s: string): number => {
  let h = 5381
  for (let i = 0; i < s.length; i++) h = (h * 33) ^ s.charCodeAt(i)
  return Math.abs(h) % 360
}

const hueToRgb = (hue: number): [number, number, number] => {
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
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)]
}

interface DataStats {
  centroid: [number, number, number]
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

const SPHERE_MESH = new SphereGeometry({ nlat: 16, nlong: 16, radius: 1 })

const LIGHTING_EFFECT = new LightingEffect({
  ambient: new AmbientLight({ color: [255, 255, 255], intensity: 0.55 }),
  key: new DirectionalLight({
    color: [255, 255, 255],
    intensity: 1.3,
    direction: [-1, -2, -3],
  }),
  fill: new DirectionalLight({
    color: [200, 220, 255],
    intensity: 0.6,
    direction: [2, 1, 1],
  }),
})

export function EmbedPage() {
  const { databaseId, tableId } = useParams<{ databaseId: string; tableId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  const { resolved } = useTheme()

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
    const radius = Math.max(stats.extent * 0.012, 0.05)
    const scale: [number, number, number] = [radius, radius, radius]
    return [
      new SimpleMeshLayer<EmbedPoint>({
        id: 'umap-spheres',
        data: state.payload.points,
        mesh: SPHERE_MESH,
        pickable: true,
        material: { ambient: 0.35, diffuse: 0.7, shininess: 64, specularColor: [255, 255, 255] },
        getPosition: (p: EmbedPoint) => [p.x, p.y, p.z],
        getColor: (p: EmbedPoint) =>
          p.category ? [...hueToRgb(hashHue(p.category)), 255] : [110, 165, 230, 255],
        getScale: () => scale,
        updateTriggers: {
          getColor: state.payload.table_id,
          getScale: stats.extent,
        },
      }),
    ]
  }, [state, stats])

  const initialViewState = useMemo<EmbedViewState>(() => {
    const zoom = Math.max(Math.log2(VIEWPORT_FALLBACK / (stats.extent * 2)) - 0.5, 0)
    return { target: stats.centroid, rotationX: 30, rotationOrbit: 30, zoom }
  }, [stats])

  const canvasBg = resolved === 'dark' ? '#12162288' : '#f8f8fa88'

  return (
    <main
      className="flex min-h-screen flex-col bg-[var(--color-surface)] text-[var(--color-foreground)]"
      data-testid="embed-page"
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
          3D UMAP: <span className="font-mono">{tableId}</span>
        </h1>
        {state.status === 'ready' && (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            {state.payload.count.toLocaleString()} points
          </p>
        )}
      </header>

      <section className="relative flex-1">
        {state.status === 'loading' && (
          <div data-testid="embed-loading" className="p-8">Loading embeddings…</div>
        )}
        {state.status === 'error' && (
          <div
            data-testid="embed-error"
            className="m-8 rounded border border-red-400 bg-red-50 p-4 text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
          >
            <p className="font-semibold">Failed to load embeddings</p>
            <p className="text-sm">{state.message}</p>
          </div>
        )}
        {state.status === 'ready' && (
          <div className="absolute inset-0" style={{ background: canvasBg }}>
            <DeckGL
              views={new OrbitView({ orbitAxis: 'Y', fovy: 50 })}
              initialViewState={initialViewState}
              controller={true}
              layers={layers}
              effects={[LIGHTING_EFFECT]}
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
