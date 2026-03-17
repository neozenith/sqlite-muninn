import { describe, it, expect } from 'vitest'
import {
  toCytoscapeElements,
  applyCommunityColors,
  applyCentralitySizing,
  highlightBfsNodes,
  applyCommunityGrouping,
} from '../transforms/cytoscape'
import type { GraphNode, GraphEdge, CentralityScore } from '../types'
import { COMMUNITY_COLORS } from '../constants'

const nodes: GraphNode[] = [
  { id: 'a', label: 'Alice', mention_count: 5, entity_type: 'PERSON' },
  { id: 'b', label: 'Bob', mention_count: 2 },
  { id: 'c', label: 'Carol' },
]

const edges: GraphEdge[] = [
  { source: 'a', target: 'b', weight: 1.0 },
  { source: 'b', target: 'c', weight: 2.0 },
]

describe('toCytoscapeElements', () => {
  it('converts nodes and edges to Cytoscape format', () => {
    const elements = toCytoscapeElements(nodes, edges)
    expect(elements).toHaveLength(5) // 3 nodes + 2 edges

    const nodeEl = elements[0]
    expect(nodeEl.data.id).toBe('a')
    expect(nodeEl.data.label).toBe('Alice')

    const edgeEl = elements[3]
    expect('source' in edgeEl.data).toBe(true)
  })

  it('sizes nodes by mention_count', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const aliceEl = elements.find((e) => e.data.id === 'a')!
    const bobEl = elements.find((e) => e.data.id === 'b')!
    expect(aliceEl.data.size).toBeGreaterThan(bobEl.data.size!)
  })

  it('clamps size to 8-24 range', () => {
    const elements = toCytoscapeElements(nodes, edges)
    for (const el of elements) {
      if ('source' in el.data) continue
      expect(el.data.size).toBeGreaterThanOrEqual(8)
      expect(el.data.size).toBeLessThanOrEqual(24)
    }
  })

  it('passes rel_type through on edges', () => {
    const edgesWithType: GraphEdge[] = [
      { source: 'a', target: 'b', weight: 1.0, rel_type: 'KNOWS' },
      { source: 'b', target: 'c', weight: 2.0 },
    ]
    const elements = toCytoscapeElements(nodes, edgesWithType)
    const edgeEls = elements.filter((e) => 'source' in e.data)
    expect((edgeEls[0].data as any).rel_type).toBe('KNOWS')
    expect((edgeEls[1].data as any).rel_type).toBeUndefined()
  })
})

describe('applyCommunityColors', () => {
  it('adds color to nodes based on community', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const colored = applyCommunityColors(elements, { a: 0, b: 1, c: 0 })

    const aliceEl = colored.find((e) => e.data.id === 'a')!
    expect(aliceEl.data.color).toBe(COMMUNITY_COLORS[0])

    const bobEl = colored.find((e) => e.data.id === 'b')!
    expect(bobEl.data.color).toBe(COMMUNITY_COLORS[1])
  })

  it('does not modify edges', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const colored = applyCommunityColors(elements, { a: 0 })
    const edgeEls = colored.filter((e) => 'source' in e.data)
    expect(edgeEls).toHaveLength(2)
  })

  it('wraps community colors for ids > palette length', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const colored = applyCommunityColors(elements, { a: 15 })
    const aliceEl = colored.find((e) => e.data.id === 'a')!
    expect(aliceEl.data.color).toBe(COMMUNITY_COLORS[15 % COMMUNITY_COLORS.length])
  })
})

describe('applyCentralitySizing', () => {
  it('sizes nodes by centrality score', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const scores: CentralityScore[] = [
      { node: 'a', centrality: 1.0 },
      { node: 'b', centrality: 0.5 },
      { node: 'c', centrality: 0.1 },
    ]
    const sized = applyCentralitySizing(elements, scores)

    const aliceEl = sized.find((e) => e.data.id === 'a')!
    const bobEl = sized.find((e) => e.data.id === 'b')!
    expect(aliceEl.data.size).toBeGreaterThan(bobEl.data.size!)
  })
})

describe('highlightBfsNodes', () => {
  it('marks BFS nodes as highlighted', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const highlighted = highlightBfsNodes(elements, new Set(['a', 'c']))

    const aliceEl = highlighted.find((e) => e.data.id === 'a')! as any
    expect(aliceEl.data.highlighted).toBe(true)

    const bobEl = highlighted.find((e) => e.data.id === 'b')! as any
    expect(bobEl.data.highlighted).toBe(false)
  })
})

describe('applyCommunityGrouping', () => {
  it('creates parent nodes per community', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const grouped = applyCommunityGrouping(elements, { a: 0, b: 1, c: 0 })

    // Should have 2 parent nodes + 3 original nodes + 2 edges = 7
    expect(grouped).toHaveLength(7)

    const parentNodes = grouped.filter((e) => e.data.id.startsWith('community-'))
    expect(parentNodes).toHaveLength(2)
    expect(parentNodes.map((p) => p.data.id).sort()).toEqual(['community-0', 'community-1'])
  })

  it('sets parent on member nodes', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const grouped = applyCommunityGrouping(elements, { a: 0, b: 1, c: 0 })

    const aliceEl = grouped.find((e) => e.data.id === 'a')! as any
    expect(aliceEl.data.parent).toBe('community-0')

    const bobEl = grouped.find((e) => e.data.id === 'b')! as any
    expect(bobEl.data.parent).toBe('community-1')
  })

  it('does not modify edges', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const grouped = applyCommunityGrouping(elements, { a: 0 })
    const edgeEls = grouped.filter((e) => 'source' in e.data)
    expect(edgeEls).toHaveLength(2)
  })

  it('skips nodes not in community map', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const grouped = applyCommunityGrouping(elements, { a: 0 })

    const carolEl = grouped.find((e) => e.data.id === 'c')! as any
    expect(carolEl.data.parent).toBeUndefined()
  })

  it('uses precomputed labels when provided', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const labels = { '0': 'Classical Economics', '1': 'Trade Theory' }
    const grouped = applyCommunityGrouping(elements, { a: 0, b: 1, c: 0 }, labels)

    const parentNodes = grouped.filter((e) => e.data.id.startsWith('community-'))
    const labelMap = new Map(parentNodes.map((p) => [p.data.id, p.data.label]))
    expect(labelMap.get('community-0')).toBe('Classical Economics')
    expect(labelMap.get('community-1')).toBe('Trade Theory')
  })

  it('falls back to generic labels when no labels provided', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const grouped = applyCommunityGrouping(elements, { a: 0, b: 1 })

    const parentNodes = grouped.filter((e) => e.data.id.startsWith('community-'))
    const labelMap = new Map(parentNodes.map((p) => [p.data.id, p.data.label]))
    expect(labelMap.get('community-0')).toBe('Community 0')
    expect(labelMap.get('community-1')).toBe('Community 1')
  })

  it('falls back for communities missing from labels map', () => {
    const elements = toCytoscapeElements(nodes, edges)
    const labels = { '0': 'Classical Economics' } // community 1 has no label
    const grouped = applyCommunityGrouping(elements, { a: 0, b: 1 }, labels)

    const parentNodes = grouped.filter((e) => e.data.id.startsWith('community-'))
    const labelMap = new Map(parentNodes.map((p) => [p.data.id, p.data.label]))
    expect(labelMap.get('community-0')).toBe('Classical Economics')
    expect(labelMap.get('community-1')).toBe('Community 1')
  })
})
