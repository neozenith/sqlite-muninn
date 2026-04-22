import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, type DatabaseInfo, fetchDatabases } from '../lib/api-client'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; databases: DatabaseInfo[] }

const formatBytes = (bytes: number): string => {
  const mb = bytes / (1024 * 1024)
  return `${mb.toFixed(1)} MB`
}

export function HomePage() {
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    fetchDatabases()
      .then((databases) => setState({ status: 'ready', databases }))
      .catch((err: unknown) => {
        const message =
          err instanceof ApiError
            ? `API error ${err.status}: ${err.body}`
            : err instanceof Error
              ? err.message
              : 'unknown error'
        setState({ status: 'error', message })
      })
  }, [])

  return (
    <main
      className="min-h-screen bg-[var(--color-surface)] p-8 text-[var(--color-foreground)]"
      data-testid="home-page"
    >
      <header className="mb-8">
        <h1 className="text-4xl font-bold">muninn-viz</h1>
        <p className="text-[var(--color-muted-foreground)]">Select a database to explore.</p>
      </header>

      {state.status === 'loading' && (
        <p data-testid="home-loading">Loading databases…</p>
      )}

      {state.status === 'error' && (
        <div
          data-testid="home-error"
          className="rounded border border-red-400 bg-red-50 p-4 text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
        >
          <p className="font-semibold">Failed to load databases</p>
          <p className="text-sm">{state.message}</p>
        </div>
      )}

      {state.status === 'ready' && state.databases.length === 0 && (
        <p data-testid="home-empty">No databases found in the demos manifest.</p>
      )}

      {state.status === 'ready' && state.databases.length > 0 && (
        <ul
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
          data-testid="home-database-list"
          data-count={state.databases.length}
        >
          {state.databases.map((db) => (
            <li key={db.id}>
              <Link
                to={`/${encodeURIComponent(db.id)}/`}
                data-testid={`db-card-${db.id}`}
                className="block rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-4 transition hover:border-[var(--color-accent)] hover:shadow"
              >
                <div className="font-semibold">{db.label}</div>
                <dl className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-sm text-[var(--color-muted-foreground)]">
                  <dt>ID</dt>
                  <dd className="font-mono">{db.id}</dd>
                  <dt>Model</dt>
                  <dd>{db.model}</dd>
                  <dt>Dimensions</dt>
                  <dd>{db.dim}</dd>
                  <dt>Size</dt>
                  <dd>{formatBytes(db.size_bytes)}</dd>
                </dl>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  )
}
