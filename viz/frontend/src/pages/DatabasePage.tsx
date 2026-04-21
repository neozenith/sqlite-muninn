import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, type DatabaseInfo, fetchDatabase } from '../lib/api-client'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string; notFound: boolean }
  | { status: 'ready'; database: DatabaseInfo }

const formatBytes = (bytes: number): string => `${(bytes / (1024 * 1024)).toFixed(1)} MB`

export function DatabasePage() {
  const { databaseId } = useParams<{ databaseId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    if (!databaseId) return
    setState({ status: 'loading' })
    fetchDatabase(databaseId)
      .then((database) => setState({ status: 'ready', database }))
      .catch((err: unknown) => {
        const isNotFound = err instanceof ApiError && err.status === 404
        const message =
          err instanceof ApiError
            ? `API error ${err.status}: ${err.body}`
            : err instanceof Error
              ? err.message
              : 'unknown error'
        setState({ status: 'error', message, notFound: isNotFound })
      })
  }, [databaseId])

  return (
    <main className="min-h-screen p-8" data-testid="database-page" data-database-id={databaseId ?? ''}>
      <nav className="mb-4">
        <Link to="/" className="text-sm text-blue-600 hover:underline" data-testid="back-to-home">
          ← Back to all databases
        </Link>
      </nav>

      {state.status === 'loading' && (
        <p data-testid="database-loading">Loading database…</p>
      )}

      {state.status === 'error' && state.notFound && (
        <div data-testid="database-not-found" className="rounded border border-amber-400 bg-amber-50 p-4 text-amber-900">
          <p className="font-semibold">Database not found</p>
          <p className="text-sm font-mono">{databaseId}</p>
        </div>
      )}

      {state.status === 'error' && !state.notFound && (
        <div data-testid="database-error" className="rounded border border-red-400 bg-red-50 p-4 text-red-800">
          <p className="font-semibold">Failed to load database</p>
          <p className="text-sm">{state.message}</p>
        </div>
      )}

      {state.status === 'ready' && (
        <article data-testid="database-detail">
          <h1 className="text-4xl font-bold">{state.database.label}</h1>
          <p className="mt-1 font-mono text-sm text-muted-foreground">{state.database.id}</p>

          <dl className="mt-6 grid max-w-xl grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="font-semibold">Book ID</dt>
            <dd>{state.database.book_id}</dd>
            <dt className="font-semibold">Embedding Model</dt>
            <dd>{state.database.model}</dd>
            <dt className="font-semibold">Dimensions</dt>
            <dd>{state.database.dim}</dd>
            <dt className="font-semibold">File</dt>
            <dd className="font-mono">{state.database.file}</dd>
            <dt className="font-semibold">Size</dt>
            <dd>{formatBytes(state.database.size_bytes)}</dd>
          </dl>
        </article>
      )}
    </main>
  )
}
