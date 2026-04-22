import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import {
  ApiError,
  type DatabaseInfo,
  type EmbedPayload,
  type KGPayload,
  type TablesResponse,
  fetchDatabase,
  fetchDatabases,
  fetchEmbed,
  fetchHealth,
  fetchKG,
  fetchTables,
} from '../api-client'

const SAMPLE_DB: DatabaseInfo = {
  id: '3300_MiniLM',
  book_id: 3300,
  model: 'MiniLM',
  dim: 384,
  file: '3300_MiniLM.db',
  size_bytes: 52285440,
  label: 'Book 3300 + MiniLM (384d)',
}

const makeResponse = (body: unknown, init: ResponseInit = {}): Response =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  })

const makeTextResponse = (body: string, status: number): Response =>
  new Response(body, { status, headers: { 'content-type': 'text/plain' } })

describe('api-client', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('fetchHealth', () => {
    test('returns body on 200', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse({ status: 'ok' }))
      const result = await fetchHealth()
      expect(result).toEqual({ status: 'ok' })
      expect(fetchMock).toHaveBeenCalledWith('/api/health')
    })

    test('throws ApiError on 500', async () => {
      fetchMock.mockResolvedValueOnce(makeTextResponse('boom', 500))
      await expect(fetchHealth()).rejects.toMatchObject({ status: 500, body: 'boom' })
    })
  })

  describe('fetchDatabases', () => {
    test('unwraps the databases array', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse({ databases: [SAMPLE_DB] }))
      const dbs = await fetchDatabases()
      expect(dbs).toHaveLength(1)
      expect(dbs[0].id).toBe('3300_MiniLM')
      expect(fetchMock).toHaveBeenCalledWith('/api/databases')
    })

    test('returns an empty array when manifest is empty', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse({ databases: [] }))
      expect(await fetchDatabases()).toEqual([])
    })

    test('throws ApiError on non-2xx', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({ detail: 'manifest not found' }, { status: 500 }),
      )
      await expect(fetchDatabases()).rejects.toBeInstanceOf(ApiError)
    })
  })

  describe('fetchDatabase', () => {
    test('requests the single-database endpoint', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse(SAMPLE_DB))
      const db = await fetchDatabase('3300_MiniLM')
      expect(db.id).toBe('3300_MiniLM')
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/3300_MiniLM')
    })

    test('URL-encodes the id', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse(SAMPLE_DB))
      await fetchDatabase('weird/id with spaces')
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/weird%2Fid%20with%20spaces')
    })

    test('throws ApiError(404) on unknown id', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({ detail: "Database 'x' not found" }, { status: 404 }),
      )
      await expect(fetchDatabase('x')).rejects.toMatchObject({ status: 404 })
    })
  })

  describe('ApiError', () => {
    test('carries status and body through .message', () => {
      const err = new ApiError(418, 'teapot')
      expect(err.status).toBe(418)
      expect(err.body).toBe('teapot')
      expect(err.message).toContain('418')
      expect(err.message).toContain('teapot')
      expect(err.name).toBe('ApiError')
      expect(err).toBeInstanceOf(Error)
    })
  })

  describe('error body fallback', () => {
    test('swallows a failing response.text() so the ApiError still surfaces', async () => {
      const brokenResponse = {
        ok: false,
        status: 500,
        text: () => Promise.reject(new Error('body read failed')),
      } as unknown as Response
      fetchMock.mockResolvedValueOnce(brokenResponse)
      const err = await fetchHealth().catch((e: unknown) => e)
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(500)
      expect((err as ApiError).body).toBe('')
    })
  })

  describe('fetchTables', () => {
    test('returns the discovery payload', async () => {
      const body: TablesResponse = {
        database_id: '3300_MiniLM',
        embed_tables: ['chunks', 'entities'],
        kg_tables: ['base', 'er'],
        resolutions: [0.25, 1.0, 3.0],
      }
      fetchMock.mockResolvedValueOnce(makeResponse(body))
      const result = await fetchTables('3300_MiniLM')
      expect(result).toEqual(body)
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/3300_MiniLM/tables')
    })

    test('encodes the database id', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({
          database_id: 'weird/id',
          embed_tables: [],
          kg_tables: [],
          resolutions: [],
        }),
      )
      await fetchTables('weird/id')
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/weird%2Fid/tables')
    })

    test('propagates ApiError on 404', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse({ detail: 'nope' }, { status: 404 }))
      await expect(fetchTables('x')).rejects.toBeInstanceOf(ApiError)
    })
  })

  describe('fetchEmbed', () => {
    test('returns the embed payload for a valid table', async () => {
      const body: EmbedPayload = {
        table_id: 'chunks',
        count: 2,
        points: [
          { id: 1, x: 0.1, y: 0.2, z: 0.3, label: 'chunk 1', category: null },
          { id: 2, x: 1.1, y: 1.2, z: 1.3, label: 'chunk 2', category: null },
        ],
      }
      fetchMock.mockResolvedValueOnce(makeResponse(body))
      const result = await fetchEmbed('3300_MiniLM', 'chunks')
      expect(result.count).toBe(2)
      expect(result.points[0].x).toBe(0.1)
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/3300_MiniLM/embed/chunks')
    })

    test('throws ApiError on 400 (invalid table)', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({ detail: 'invalid table' }, { status: 400 }),
      )
      await expect(fetchEmbed('3300_MiniLM', 'bogus')).rejects.toMatchObject({ status: 400 })
    })

    test('URL-encodes both ids', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({ table_id: 'x', count: 0, points: [] }),
      )
      await fetchEmbed('weird/db', 'weird/table')
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/databases/weird%2Fdb/embed/weird%2Ftable',
      )
    })
  })

  describe('fetchKG', () => {
    const emptyKG = (overrides: Partial<KGPayload> = {}): KGPayload => ({
      table_id: 'base',
      resolution: 0.25,
      seed_metric: 'edge_betweenness',
      max_depth: 0,
      node_count: 0,
      edge_count: 0,
      community_count: 0,
      total_node_count: 0,
      total_edge_count: 0,
      nodes: [],
      edges: [],
      communities: [],
      ...overrides,
    })

    test('returns the KG payload without resolution query', async () => {
      const body: KGPayload = emptyKG({
        node_count: 1,
        community_count: 1,
        total_node_count: 1,
        nodes: [
          {
            id: 'a',
            label: 'a',
            entity_type: null,
            community_id: 0,
            mention_count: 1,
            node_betweenness: 0.5,
          },
        ],
        communities: [{ id: 0, label: null, member_count: 1, node_ids: ['a'] }],
      })
      fetchMock.mockResolvedValueOnce(makeResponse(body))
      const result = await fetchKG('3300_MiniLM', 'base')
      expect(result.resolution).toBe(0.25)
      expect(result.seed_metric).toBe('edge_betweenness')
      expect(fetchMock).toHaveBeenCalledWith('/api/databases/3300_MiniLM/kg/base')
    })

    test('appends the resolution query string when provided', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse(emptyKG({ resolution: 1 })))
      await fetchKG('3300_MiniLM', 'base', { resolution: 1.0 })
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/databases/3300_MiniLM/kg/base?resolution=1',
      )
    })

    test('appends the top_n query string when provided', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse(emptyKG()))
      await fetchKG('3300_MiniLM', 'base', { topN: 1500 })
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/databases/3300_MiniLM/kg/base?top_n=1500',
      )
    })

    test('appends seed_metric and max_depth when provided', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse(emptyKG({ seed_metric: 'degree', max_depth: 2 })),
      )
      await fetchKG('3300_MiniLM', 'base', {
        seedMetric: 'degree',
        maxDepth: 2,
      })
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/databases/3300_MiniLM/kg/base?seed_metric=degree&max_depth=2',
      )
    })

    test('combines resolution and top_n when both are given', async () => {
      fetchMock.mockResolvedValueOnce(makeResponse(emptyKG({ resolution: 1 })))
      await fetchKG('3300_MiniLM', 'base', { resolution: 1, topN: 0 })
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/databases/3300_MiniLM/kg/base?resolution=1&top_n=0',
      )
    })

    test('throws ApiError on 422 (missing tables)', async () => {
      fetchMock.mockResolvedValueOnce(
        makeResponse({ detail: 'nodes table missing' }, { status: 422 }),
      )
      await expect(fetchKG('x', 'base')).rejects.toMatchObject({ status: 422 })
    })
  })
})
