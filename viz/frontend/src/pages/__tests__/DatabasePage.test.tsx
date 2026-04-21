import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { DatabasePage } from '../DatabasePage'

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/:databaseId/" element={<DatabasePage />} />
      </Routes>
    </MemoryRouter>,
  )

describe('DatabasePage', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('shows loading state initially', () => {
    fetchMock.mockReturnValueOnce(new Promise(() => {}))
    renderAt('/3300_MiniLM/')
    expect(screen.getByTestId('database-loading')).toBeInTheDocument()
  })

  test('renders detail on success', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        id: '3300_MiniLM',
        book_id: 3300,
        model: 'MiniLM',
        dim: 384,
        file: '3300_MiniLM.db',
        size_bytes: 52285440,
        label: 'Book 3300 + MiniLM (384d)',
      }),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-detail')).toBeInTheDocument()
    })
    expect(screen.getByRole('heading', { name: 'Book 3300 + MiniLM (384d)' })).toBeInTheDocument()
    expect(screen.getByText('384')).toBeInTheDocument()
    expect(screen.getByText('49.9 MB')).toBeInTheDocument()
  })

  test('shows not-found state on 404', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: "Database 'bogus' not found" }, 404),
    )
    renderAt('/bogus/')
    await waitFor(() => {
      expect(screen.getByTestId('database-not-found')).toBeInTheDocument()
    })
  })

  test('shows generic error state on 500', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response('manifest not found', { status: 500 }),
    )
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('shows error state when fetch rejects', async () => {
    fetchMock.mockRejectedValueOnce(new Error('network down'))
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('handles non-Error rejection', async () => {
    fetchMock.mockRejectedValueOnce({ weird: true })
    renderAt('/3300_MiniLM/')
    await waitFor(() => {
      expect(screen.getByTestId('database-error')).toBeInTheDocument()
    })
  })

  test('encodes the database id in the data attribute', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        id: '3300_MiniLM',
        book_id: 3300,
        model: 'MiniLM',
        dim: 384,
        file: '3300_MiniLM.db',
        size_bytes: 52285440,
        label: 'Book 3300 + MiniLM (384d)',
      }),
    )
    renderAt('/3300_MiniLM/')
    expect(screen.getByTestId('database-page')).toHaveAttribute('data-database-id', '3300_MiniLM')
  })
})
