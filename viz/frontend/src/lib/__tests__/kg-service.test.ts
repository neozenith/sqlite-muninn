import { describe, it, expect, vi, beforeEach } from 'vitest'
import { queryKGSearch } from '../services/kg-service'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => {
  mockFetch.mockReset()
})

function mockOk(data: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(data),
  })
}

describe('queryKGSearch', () => {
  it('sends POST with query body', async () => {
    mockOk({
      query: 'test',
      fts_results: [],
      vss_results: [],
      graph_nodes: [],
      graph_edges: [],
    })
    const result = await queryKGSearch('division of labor', 5)
    expect(result.query).toBe('test')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/query',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ query: 'division of labor', k: 5 }),
      }),
    )
  })

  it('uses default k', async () => {
    mockOk({
      query: 'test',
      fts_results: [],
      vss_results: [],
      graph_nodes: [],
      graph_edges: [],
    })
    await queryKGSearch('test')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/query',
      expect.objectContaining({
        body: JSON.stringify({ query: 'test', k: 10 }),
      }),
    )
  })

  it('sends resolution when provided', async () => {
    mockOk({
      query: 'test',
      fts_results: [],
      vss_results: [],
      graph_nodes: [],
      graph_edges: [],
    })
    await queryKGSearch('test', 10, 0.25)
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/query',
      expect.objectContaining({
        body: JSON.stringify({ query: 'test', k: 10, resolution: 0.25 }),
      }),
    )
  })

  it('omits resolution when not provided', async () => {
    mockOk({
      query: 'test',
      fts_results: [],
      vss_results: [],
      graph_nodes: [],
      graph_edges: [],
    })
    await queryKGSearch('test')
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body).not.toHaveProperty('resolution')
  })

  it('propagates API errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      text: () => Promise.resolve('KG search failed'),
    })
    await expect(queryKGSearch('fail')).rejects.toThrow()
  })
})
