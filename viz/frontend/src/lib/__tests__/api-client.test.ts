import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import {
  ApiError,
  type DatabaseInfo,
  fetchDatabase,
  fetchDatabases,
  fetchHealth,
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
})
