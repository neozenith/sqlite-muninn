/**
 * EmbedPage state-machine tests.
 *
 * We don't assert WebGL rendering in vitest — jsdom has no WebGL context,
 * and Deck.GL will simply not mount the canvas. The three render states
 * (loading / error / ready-with-data-ready-testid) are still verifiable
 * because they live in the DOM outside the canvas.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { ThemeProvider } from '../../lib/ThemeProvider'
import { EmbedPage } from '../EmbedPage'

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const renderAt = (path: string) =>
  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/:databaseId/embed/:tableId/" element={<EmbedPage />} />
        </Routes>
      </MemoryRouter>
    </ThemeProvider>,
  )

const SAMPLE_PAYLOAD = {
  table_id: 'chunks',
  count: 2,
  points: [
    { id: 1, x: 0, y: 0, z: 0, label: 'a', category: null },
    { id: 2, x: 1, y: 1, z: 1, label: 'b', category: 'person' },
  ],
}

describe('EmbedPage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('shows loading state before fetch resolves', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {})),
    )
    renderAt('/3300_MiniLM/embed/chunks/')
    expect(screen.getByTestId('embed-loading')).toBeInTheDocument()
  })

  test('renders canvas-ready marker with point count on success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(jsonResponse(SAMPLE_PAYLOAD))),
    )
    renderAt('/3300_MiniLM/embed/chunks/')
    await waitFor(() => {
      expect(screen.getByTestId('embed-canvas-ready')).toBeInTheDocument()
    })
    expect(screen.getByTestId('embed-canvas-ready')).toHaveAttribute('data-point-count', '2')
  })

  test('shows error state on 400', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(jsonResponse({ detail: 'invalid table' }, 400))),
    )
    renderAt('/3300_MiniLM/embed/bogus/')
    await waitFor(() => {
      expect(screen.getByTestId('embed-error')).toBeInTheDocument()
    })
  })

  test('shows error state when fetch rejects', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('network down'))),
    )
    renderAt('/3300_MiniLM/embed/chunks/')
    await waitFor(() => {
      expect(screen.getByTestId('embed-error')).toBeInTheDocument()
    })
  })

  test('handles a non-Error rejection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject('weird')),
    )
    renderAt('/3300_MiniLM/embed/chunks/')
    await waitFor(() => {
      expect(screen.getByTestId('embed-error')).toBeInTheDocument()
    })
  })
})
