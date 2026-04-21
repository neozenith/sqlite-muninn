import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App'

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
    fetchMock.mockResolvedValueOnce(jsonResponse({ databases: SAMPLE_DBS }))
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )
    await waitFor(() => {
      expect(screen.getByTestId('home-database-list')).toBeInTheDocument()
    })
    expect(screen.getByTestId('db-card-3300_MiniLM')).toBeInTheDocument()
  })

  test('database route renders the detail page', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(SAMPLE_DBS[0]))
    render(
      <MemoryRouter initialEntries={['/3300_MiniLM/']}>
        <App />
      </MemoryRouter>,
    )
    await waitFor(() => {
      expect(screen.getByTestId('database-detail')).toBeInTheDocument()
    })
    expect(screen.getByRole('heading', { name: 'Book 3300 + MiniLM (384d)' })).toBeInTheDocument()
  })

  test('unknown route redirects to home', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ databases: [] }))
    render(
      <MemoryRouter initialEntries={['/totally/bogus/path']}>
        <App />
      </MemoryRouter>,
    )
    await waitFor(() => {
      expect(screen.getByTestId('home-page')).toBeInTheDocument()
    })
  })
})
