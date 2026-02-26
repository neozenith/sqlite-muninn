import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchDatabases, selectDatabase, getStoredDatabaseId } from '../services/db-service'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

const mockStorage: Record<string, string> = {}
vi.stubGlobal('localStorage', {
  getItem: (key: string) => mockStorage[key] ?? null,
  setItem: (key: string, value: string) => {
    mockStorage[key] = value
  },
  removeItem: (key: string) => {
    delete mockStorage[key]
  },
})

beforeEach(() => {
  mockFetch.mockReset()
  Object.keys(mockStorage).forEach((key) => delete mockStorage[key])
})

function mockOk(data: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(data),
  })
}

describe('fetchDatabases', () => {
  it('calls GET /api/databases', async () => {
    const response = { databases: [], active: null }
    mockOk(response)
    const result = await fetchDatabases()
    expect(result).toEqual(response)
    expect(mockFetch).toHaveBeenCalledWith('/api/databases', expect.anything())
  })

  it('returns typed DatabasesResponse', async () => {
    const response = {
      databases: [
        {
          id: '3300_MiniLM',
          book_id: 3300,
          model: 'MiniLM',
          dim: 384,
          file: '3300_MiniLM.db',
          size_bytes: 1000,
          label: 'Book 3300 + MiniLM (384d)',
        },
      ],
      active: '3300_MiniLM',
    }
    mockOk(response)
    const result = await fetchDatabases()
    expect(result.databases).toHaveLength(1)
    expect(result.databases[0].id).toBe('3300_MiniLM')
    expect(result.active).toBe('3300_MiniLM')
  })
})

describe('selectDatabase', () => {
  it('calls POST /api/databases/select and sets localStorage', async () => {
    mockOk({ status: 'ok', active: '3300_MiniLM' })
    await selectDatabase('3300_MiniLM')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/databases/select',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ id: '3300_MiniLM' }),
      }),
    )
    expect(mockStorage['muninn-selected-db']).toBe('3300_MiniLM')
  })
})

describe('getStoredDatabaseId', () => {
  it('returns null when no stored value', () => {
    expect(getStoredDatabaseId()).toBeNull()
  })

  it('returns stored value', () => {
    mockStorage['muninn-selected-db'] = '3300_MiniLM'
    expect(getStoredDatabaseId()).toBe('3300_MiniLM')
  })
})
