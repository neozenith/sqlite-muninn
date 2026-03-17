/** Cytoscape.js data transforms — convert API data to Cytoscape elements. */

import type { CentralityScore, GraphEdge, GraphNode } from '../types'
import { communityColor } from '../constants'

export interface CytoscapeNode {
  data: {
    id: string
    label: string
    color?: string
    size?: number
    mention_count?: number
    entity_type?: string
    parent?: string
  }
}

export interface CytoscapeEdge {
  data: {
    id: string
    source: string
    target: string
    weight: number
    rel_type?: string
  }
}

export type CytoscapeElement = CytoscapeNode | CytoscapeEdge

/**
 * Convert API graph data to Cytoscape elements array.
 */
export function toCytoscapeElements(nodes: GraphNode[], edges: GraphEdge[]): CytoscapeElement[] {
  const nodeElements: CytoscapeNode[] = nodes.map((n) => ({
    data: {
      id: n.id,
      label: n.label,
      mention_count: n.mention_count,
      entity_type: n.entity_type,
      size: Math.max(8, Math.min(24, (n.mention_count ?? 1) * 3)),
    },
  }))

  const edgeElements: CytoscapeEdge[] = edges.map((e, i) => ({
    data: {
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      weight: e.weight,
      ...(e.rel_type != null ? { rel_type: e.rel_type } : {}),
    },
  }))

  return [...nodeElements, ...edgeElements]
}

/**
 * Apply community colors to Cytoscape node elements.
 */
export function applyCommunityColors(
  elements: CytoscapeElement[],
  nodeCommunity: Record<string, number>,
): CytoscapeElement[] {
  return elements.map((el) => {
    if ('source' in el.data) return el // Edge, skip
    const communityId = nodeCommunity[el.data.id]
    if (communityId !== undefined) {
      return {
        ...el,
        data: { ...el.data, color: communityColor(communityId) },
      }
    }
    return el
  })
}

/**
 * Apply centrality-based sizing to Cytoscape node elements.
 */
export function applyCentralitySizing(elements: CytoscapeElement[], scores: CentralityScore[]): CytoscapeElement[] {
  const scoreMap = new Map(scores.map((s) => [s.node, s.centrality]))
  const maxScore = Math.max(...scores.map((s) => s.centrality), 0.001)

  return elements.map((el) => {
    if ('source' in el.data) return el // Edge, skip
    const score = scoreMap.get(el.data.id)
    if (score !== undefined) {
      const normalized = score / maxScore // 0..1
      const size = 8 + normalized * 24 // 8..32
      return {
        ...el,
        data: { ...el.data, size },
      }
    }
    return el
  })
}

/**
 * Highlight BFS results by adding a `highlighted` flag to node data.
 */
export function highlightBfsNodes(elements: CytoscapeElement[], bfsNodeIds: Set<string>): CytoscapeElement[] {
  return elements.map((el) => {
    if ('source' in el.data) return el
    const highlighted = bfsNodeIds.has(el.data.id)
    return {
      ...el,
      data: { ...el.data, highlighted } as CytoscapeNode['data'] & { highlighted: boolean },
    }
  })
}

/**
 * Add compound parent nodes per community and set `parent` on member nodes.
 * Cytoscape's cose layout natively groups children within parent bounding boxes.
 *
 * When `communityLabels` is provided, parent nodes use the precomputed label
 * (e.g. "Economic Trade Theory") instead of the generic "Community 3" fallback.
 */
export function applyCommunityGrouping(
  elements: CytoscapeElement[],
  nodeCommunity: Record<string, number>,
  communityLabels?: Record<string, string>,
): CytoscapeElement[] {
  // Collect unique community IDs
  const communityIds = new Set(Object.values(nodeCommunity))

  // Create parent nodes — use precomputed labels when available
  const parentNodes: CytoscapeNode[] = [...communityIds].map((id) => ({
    data: {
      id: `community-${id}`,
      label: communityLabels?.[String(id)] ?? `Community ${id}`,
    },
  }))

  // Set parent on member nodes
  const updatedElements = elements.map((el) => {
    if ('source' in el.data) return el // Edge, skip
    const communityId = nodeCommunity[el.data.id]
    if (communityId !== undefined) {
      return {
        ...el,
        data: { ...el.data, parent: `community-${communityId}` },
      }
    }
    return el
  })

  return [...parentNodes, ...updatedElements]
}
