import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { DatabasePage } from '../DatabasePage'

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const DB_SAMPLE = {
  id: '3300_MiniLM',
  book_id: 3300,
  model: 'MiniLM',
  dim: 384,
  file: '3300_MiniLM.db',
  size_bytes: 52285440,
  label: 'Book 3300 + MiniLM (384d)',
}

const TABLES_SAMPLE = {
  database_id: '3300_MiniLM',
  embed_tables: ['chunks', 'entities'],
  kg_tables: ['base', 'er'],
  resolutions: [0.25, 1.0, 3.0],
}

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/:databaseId/" element={<DatabasePage />} />
      </Routes>
    </MemoryRouter>,
  )

/** Route-aware fetch stub: matches the request URL path to a canned response. */
const mockByPath = (responses: Record<string, Response | Promise<Response>>) =>
  vi.fn((url: string) => {
    const path = url.replace(/^\/api/, '')
    for (const key of Object.keys(responses)) {
      if (path === key) return Promise.resolve(responses[key])
    }
    return Promise.resolve(new Response('no route', { status: 500 }))
  })

describe('DatabasePage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('shows loading state initially', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {})),
    )
    renderAt('/3300_MiniLM/')
    expect(screen.getByTestId('database-loading')).toBeInTheDocument()
  })

  test('renders detail + embed/kg links on success', async () => {
    vi.stubGlobal(
      'fetch',
      mockByPath({
        '/databases/3300_MiniLM': jsonResponse(DB_SAMPLE),
        '/databases/3300_MiniLM/tables': jsonResponse(TABLES_SAMPLE),
      }),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-detail')).toBeInTheDocument()
    })
    expect(screen.getByRole('heading', { name: 'Book 3300 + MiniLM (384d)' })).toBeInTheDocument()

    // Embed links
    expect(screen.getByTestId('embed-link-chunks')).toHaveAttribute(
      'href',
      '/3300_MiniLM/embed/chunks/',
    )
    expect(screen.getByTestId('embed-link-entities')).toHaveAttribute(
      'href',
      '/3300_MiniLM/embed/entities/',
    )
    // KG links
    expect(screen.getByTestId('kg-link-base')).toHaveAttribute('href', '/3300_MiniLM/kg/base/')
    expect(screen.getByTestId('kg-link-er')).toHaveAttribute('href', '/3300_MiniLM/kg/er/')
  })

  test('shows not-found state on 404', async () => {
    vi.stubGlobal(
      'fetch',
      mockByPath({
        '/databases/bogus': jsonResponse({ detail: "Database 'bogus' not found" }, 404),
        '/databases/bogus/tables': jsonResponse({ detail: 'nope' }, 404),
      }),
    )
    renderAt('/bogus/')
    await waitFor(() => {
      expect(screen.getByTestId('database-not-found')).toBeInTheDocument()
    })
  })

  test('shows generic error state on 500', async () => {
    vi.stubGlobal(
      'fetch',
      mockByPath({
        '/databases/3300_MiniLM': new Response('manifest not found', { status: 500 }),
        '/databases/3300_MiniLM/tables': new Response('manifest not found', { status: 500 }),
      }),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('shows error state when fetch rejects', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('network down'))),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('handles non-Error rejection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject({ weird: true })),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('encodes the database id in the data attribute', async () => {
    vi.stubGlobal(
      'fetch',
      mockByPath({
        '/databases/3300_MiniLM': jsonResponse(DB_SAMPLE),
        '/databases/3300_MiniLM/tables': jsonResponse(TABLES_SAMPLE),
      }),
    )
    renderAt('/3300_MiniLM/')
    expect(screen.getByTestId('database-page')).toHaveAttribute('data-database-id', '3300_MiniLM')
  })

  test('renders empty sections when manifest has no tables', async () => {
    vi.stubGlobal(
      'fetch',
      mockByPath({
        '/databases/3300_MiniLM': jsonResponse(DB_SAMPLE),
        '/databases/3300_MiniLM/tables': jsonResponse({
          database_id: '3300_MiniLM',
          embed_tables: [],
          kg_tables: [],
          resolutions: [],
        }),
      }),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('embed-links')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('embed-link-chunks')).not.toBeInTheDocument()
    expect(screen.queryByTestId('kg-link-base')).not.toBeInTheDocument()
  })
})
