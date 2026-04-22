import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { HomePage } from '../HomePage'

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const renderHome = () =>
  render(
    <MemoryRouter>
      <HomePage />
    </MemoryRouter>,
  )

describe('HomePage', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('shows loading state initially', () => {
    fetchMock.mockReturnValueOnce(new Promise(() => {})) // never resolves
    renderHome()
    expect(screen.getByTestId('home-loading')).toBeInTheDocument()
  })

  test('renders the database grid on success', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        databases: [
          {
            id: '3300_MiniLM',
            book_id: 3300,
            model: 'MiniLM',
            dim: 384,
            file: '3300_MiniLM.db',
            size_bytes: 52285440,
            label: 'Book 3300 + MiniLM (384d)',
          },
        ],
      }),
    )
    renderHome()
    await waitFor(() => {
      expect(screen.getByTestId('home-database-list')).toBeInTheDocument()
    })
    expect(screen.getByTestId('home-database-list')).toHaveAttribute('data-count', '1')
    expect(screen.getByText('Book 3300 + MiniLM (384d)')).toBeInTheDocument()
    expect(screen.getByText('49.9 MB')).toBeInTheDocument() // 52285440 / 1024^2
  })

  test('shows empty state when manifest has no databases', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ databases: [] }))
    renderHome()
    await waitFor(() => {
      expect(screen.getByTestId('home-empty')).toBeInTheDocument()
    })
  })

  test('shows error state when the API returns 500', async () => {
    fetchMock.mockResolvedValueOnce(new Response('manifest not found', { status: 500 }))
    renderHome()
    await waitFor(() => {
      expect(screen.getByTestId('home-error')).toBeInTheDocument()
    })
    expect(screen.getByText(/API error 500/)).toBeInTheDocument()
  })

  test('shows error state when fetch itself rejects', async () => {
    fetchMock.mockRejectedValueOnce(new Error('network down'))
    renderHome()
    await waitFor(() => {
      expect(screen.getByTestId('home-error')).toBeInTheDocument()
    })
    expect(screen.getByText(/network down/)).toBeInTheDocument()
  })

  test('handles a non-Error rejection value', async () => {
    fetchMock.mockRejectedValueOnce('weird string error')
    renderHome()
    await waitFor(() => {
      expect(screen.getByTestId('home-error')).toBeInTheDocument()
    })
    expect(screen.getByText(/unknown error/)).toBeInTheDocument()
  })

  test('database card links to /:database_id/', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        databases: [
          {
            id: '3300_MiniLM',
            book_id: 3300,
            model: 'MiniLM',
            dim: 384,
            file: '3300_MiniLM.db',
            size_bytes: 52285440,
            label: 'Book 3300 + MiniLM (384d)',
          },
        ],
      }),
    )
    renderHome()
    const link = await screen.findByTestId('db-card-3300_MiniLM')
    expect(link).toHaveAttribute('href', '/3300_MiniLM/')
  })
})
