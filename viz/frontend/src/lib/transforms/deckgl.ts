/** Deck.GL data transforms — convert API data to Deck.GL layer data. */

import type { EmbeddingPoint, SearchNeighbor } from '../types'
import { BACKGROUND_POINT_COLOR, SEARCH_HIGHLIGHT_COLOR, SELECTED_POINT_COLOR } from '../constants'

export interface ScatterplotDatum {
  position: [number, number] | [number, number, number]
  id: number
  label: string
  color: [number, number, number]
  radius: number
  /** Alpha channel (0–255). Background points = 128, search results = 220, selected = 255. */
  opacity: number
  /** Optional rich text for HTML tooltip (chunk text + similarity). */
  text?: string
}

export interface ViewState {
  target: [number, number, number]
  zoom: number
}

export interface ViewState3D extends ViewState {
  rotationX: number
  rotationOrbit: number
  orbitAxis: 'Y'
}

/**
 * Compute an initial view state that fits all data points into the viewport.
 * Returns target (center) and zoom level so the bounding box fits comfortably.
 * When is3D=true, also computes Z center and returns OrbitView rotation props.
 */
export function computeViewState(
  data: ScatterplotDatum[],
  viewWidth?: number,
  viewHeight?: number,
  is3D?: false,
): ViewState
export function computeViewState(
  data: ScatterplotDatum[],
  viewWidth: number,
  viewHeight: number,
  is3D: true,
): ViewState3D
export function computeViewState(
  data: ScatterplotDatum[],
  viewWidth: number = 800,
  viewHeight: number = 600,
  is3D: boolean = false,
): ViewState | ViewState3D {
  if (data.length === 0) {
    const base: ViewState = { target: [0, 0, 0], zoom: 0 }
    if (is3D) return { ...base, rotationX: 30, rotationOrbit: -30, orbitAxis: 'Y' } as ViewState3D
    return base
  }

  let minX = Infinity,
    maxX = -Infinity
  let minY = Infinity,
    maxY = -Infinity
  let minZ = Infinity,
    maxZ = -Infinity
  for (const d of data) {
    const x = d.position[0]
    const y = d.position[1]
    const z = d.position[2] ?? 0
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }

  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1

  // Add 10% padding so points at the edge aren't clipped
  const paddedX = rangeX * 1.1
  const paddedY = rangeY * 1.1

  // zoom = log2(viewportSize / dataRange)
  const zoomX = Math.log2(viewWidth / paddedX)
  const zoomY = Math.log2(viewHeight / paddedY)
  let zoom = Math.min(zoomX, zoomY)

  if (is3D) {
    const rangeZ = maxZ - minZ || 1
    const paddedZ = rangeZ * 1.1
    // For 3D, zoom must also accommodate the Z range projected onto screen
    const zoomZ = Math.log2(Math.min(viewWidth, viewHeight) / paddedZ)
    zoom = Math.min(zoom, zoomZ)
    return {
      target: [cx, cy, cz],
      zoom,
      rotationX: 30,
      rotationOrbit: -30,
      orbitAxis: 'Y',
    } as ViewState3D
  }

  return { target: [cx, cy, 0], zoom }
}

/**
 * Convert embedding points to Deck.GL ScatterplotLayer data.
 * Optionally highlights search results and a selected point.
 */
export function toScatterplotData(
  points: EmbeddingPoint[],
  searchResults?: SearchNeighbor[],
  selectedId?: number,
): ScatterplotDatum[] {
  const searchIds = new Set(searchResults?.map((r) => r.id) ?? [])
  const hasActiveSearch = searchIds.size > 0 || selectedId != null

  return points.map((p) => {
    let color: [number, number, number] = p.color ? [...p.color] : BACKGROUND_POINT_COLOR
    // 25% when a search is active so results stand out; 50% otherwise
    let opacity = p.color ? 220 : hasActiveSearch ? 64 : 128

    if (p.id === selectedId) {
      color = SELECTED_POINT_COLOR
      opacity = 255
    } else if (searchIds.has(p.id)) {
      color = SEARCH_HIGHLIGHT_COLOR
      opacity = 220
    }

    const position: [number, number] | [number, number, number] = p.z != null ? [p.x, p.y, p.z] : [p.x, p.y]

    return {
      position,
      id: p.id,
      label: p.label,
      color,
      radius: 5,
      opacity,
      text: typeof p.metadata?.text === 'string' ? p.metadata.text : undefined,
    }
  })
}

/**
 * Extract just the search result IDs as a Set for quick lookup.
 */
export function searchResultIds(results: SearchNeighbor[]): Set<number> {
  return new Set(results.map((r) => r.id))
}
