/**
 * KGPage state-machine tests.
 *
 * Cytoscape fails to initialize in jsdom because it depends on DOM measurement
 * APIs (`getBoundingClientRect`, `offsetWidth`) that return zeros. So we test:
 *  - loading / error states (DOM-only)
 *  - ready state arrives (via header node-count text) before the layout
 *    converges. The `kg-canvas-ready` flip is an E2E concern.
 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { ThemeProvider } from '../../lib/ThemeProvider'
import { KGPage } from '../KGPage'

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
          <Route path="/:databaseId/kg/:tableId/" element={<KGPage />} />
        </Routes>
      </MemoryRouter>
    </ThemeProvider>,
  )

describe('KGPage', () => {
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
    renderAt('/3300_MiniLM/kg/base/')
    expect(screen.getByTestId('kg-loading')).toBeInTheDocument()
  })

  // NB: we don't render the Cytoscape success state in vitest — react-cytoscapejs
  // crashes at unmount in jsdom because `this._cy` never initializes without DOM
  // measurement APIs. The ready-state assertion lives in the E2E suite instead.

  test('shows error state on 422', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(jsonResponse({ detail: 'data missing' }, 422)),
      ),
    )
    renderAt('/3300_MiniLM/kg/base/')
    await waitFor(() => {
      expect(screen.getByTestId('kg-error')).toBeInTheDocument()
    })
  })

  test('shows error state when fetch rejects', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('network down'))),
    )
    renderAt('/3300_MiniLM/kg/base/')
    await waitFor(() => {
      expect(screen.getByTestId('kg-error')).toBeInTheDocument()
    })
  })

  test('handles non-Error rejection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject('something weird')),
    )
    renderAt('/3300_MiniLM/kg/base/')
    await waitFor(() => {
      expect(screen.getByTestId('kg-error')).toBeInTheDocument()
    })
  })
})
