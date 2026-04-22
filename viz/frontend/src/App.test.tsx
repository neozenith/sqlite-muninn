import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App'
import { ThemeProvider } from './lib/ThemeProvider'

const SAMPLE_DBS = [
  {
    id: '3300_MiniLM',
    book_id: 3300,
    model: 'MiniLM',
    dim: 384,
    file: '3300_MiniLM.db',
    size_bytes: 52285440,
    label: 'Book 3300 + MiniLM (384d)',
  },
]

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const renderApp = (initialPath: string) =>
  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[initialPath]}>
        <App />
      </MemoryRouter>
    </ThemeProvider>,
  )

describe('App', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('home route renders the database list', async () => {
    // Both HomePage and Sidebar fetch /api/databases — the impl handles either
    // URL by returning the same payload.
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith('/databases')) {
        return Promise.resolve(jsonResponse({ databases: SAMPLE_DBS }))
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`))
    })
    renderApp('/')
    await waitFor(() => {
      expect(screen.getByTestId('home-database-list')).toBeInTheDocument()
    })
    expect(screen.getByTestId('db-card-3300_MiniLM')).toBeInTheDocument()
  })

  test('database route renders the detail page', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith('/databases')) {
        return Promise.resolve(jsonResponse({ databases: SAMPLE_DBS }))
      }
      if (url.endsWith('/tables')) {
        return Promise.resolve(
          jsonResponse({
            database_id: '3300_MiniLM',
            embed_tables: ['chunks'],
            kg_tables: ['base'],
            resolutions: [0.25],
          }),
        )
      }
      return Promise.resolve(jsonResponse(SAMPLE_DBS[0]))
    })
    renderApp('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-detail')).toBeInTheDocument()
    })
    expect(screen.getByRole('heading', { name: 'Book 3300 + MiniLM (384d)' })).toBeInTheDocument()
  })

  test('unknown route redirects to home', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ databases: [] })),
    )
    renderApp('/totally/bogus/path')
    await waitFor(() => {
      expect(screen.getByTestId('home-page')).toBeInTheDocument()
    })
  })
})
