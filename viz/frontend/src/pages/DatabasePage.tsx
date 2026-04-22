import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ApiError,
  type DatabaseInfo,
  type TablesResponse,
  fetchDatabase,
  fetchTables,
} from '../lib/api-client'

interface Loaded {
  database: DatabaseInfo
  tables: TablesResponse
}

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string; notFound: boolean }
  | { status: 'ready'; data: Loaded }

const formatBytes = (bytes: number): string => `${(bytes / (1024 * 1024)).toFixed(1)} MB`

const EMBED_LABELS: Record<string, string> = {
  chunks: 'Chunk embeddings (3D UMAP)',
  entities: 'Entity embeddings (3D UMAP)',
}

const KG_LABELS: Record<string, string> = {
  base: 'Base knowledge graph (raw NER + RE)',
  er: 'Entity-resolved knowledge graph',
}

export function DatabasePage() {
  const { databaseId } = useParams<{ databaseId: string }>()
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    if (!databaseId) return
    setState({ status: 'loading' })
    Promise.all([fetchDatabase(databaseId), fetchTables(databaseId)])
      .then(([database, tables]) =>
        setState({ status: 'ready', data: { database, tables } }),
      )
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
    <main
      className="min-h-screen bg-[var(--color-surface)] p-8 text-[var(--color-foreground)]"
      data-testid="database-page"
      data-database-id={databaseId ?? ''}
    >
      <nav className="mb-4">
        <Link
          to="/"
          className="text-sm text-[var(--color-accent)] hover:underline"
          data-testid="back-to-home"
        >
          ← Back to all databases
        </Link>
      </nav>

      {state.status === 'loading' && (
        <p data-testid="database-loading">Loading database…</p>
      )}

      {state.status === 'error' && state.notFound && (
        <div
          data-testid="database-not-found"
          className="rounded border border-amber-400 bg-amber-50 p-4 text-amber-900 dark:border-amber-500 dark:bg-amber-950/40 dark:text-amber-200"
        >
          <p className="font-semibold">Database not found</p>
          <p className="font-mono text-sm">{databaseId}</p>
        </div>
      )}

      {state.status === 'error' && !state.notFound && (
        <div
          data-testid="database-error"
          className="rounded border border-red-400 bg-red-50 p-4 text-red-800 dark:border-red-500 dark:bg-red-950/40 dark:text-red-200"
        >
          <p className="font-semibold">Failed to load database</p>
          <p className="text-sm">{state.message}</p>
        </div>
      )}

      {state.status === 'ready' && (
        <article data-testid="database-detail">
          <h1 className="text-4xl font-bold">{state.data.database.label}</h1>
          <p className="mt-1 font-mono text-sm text-[var(--color-muted-foreground)]">
            {state.data.database.id}
          </p>

          <dl className="mt-6 grid max-w-xl grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="font-semibold">Book ID</dt>
            <dd>{state.data.database.book_id}</dd>
            <dt className="font-semibold">Embedding Model</dt>
            <dd>{state.data.database.model}</dd>
            <dt className="font-semibold">Dimensions</dt>
            <dd>{state.data.database.dim}</dd>
            <dt className="font-semibold">File</dt>
            <dd className="font-mono">{state.data.database.file}</dd>
            <dt className="font-semibold">Size</dt>
            <dd>{formatBytes(state.data.database.size_bytes)}</dd>
          </dl>

          <section className="mt-8" data-testid="embed-links">
            <h2 className="mb-2 text-xl font-semibold">3D UMAP Embeddings</h2>
            <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {state.data.tables.embed_tables.map((tid) => (
                <li key={`embed-${tid}`}>
                  <Link
                    to={`/${encodeURIComponent(state.data.database.id)}/embed/${encodeURIComponent(tid)}/`}
                    data-testid={`embed-link-${tid}`}
                    className="block rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-3 transition hover:border-[var(--color-accent)] hover:shadow"
                  >
                    <span className="font-mono text-sm">{tid}</span>
                    <span className="block text-xs text-[var(--color-muted-foreground)]">
                      {EMBED_LABELS[tid] ?? tid}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </section>

          <section className="mt-6" data-testid="kg-links">
            <h2 className="mb-2 text-xl font-semibold">Knowledge Graphs</h2>
            <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {state.data.tables.kg_tables.map((tid) => (
                <li key={`kg-${tid}`}>
                  <Link
                    to={`/${encodeURIComponent(state.data.database.id)}/kg/${encodeURIComponent(tid)}/`}
                    data-testid={`kg-link-${tid}`}
                    className="block rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] p-3 transition hover:border-[var(--color-accent)] hover:shadow"
                  >
                    <span className="font-mono text-sm">{tid}</span>
                    <span className="block text-xs text-[var(--color-muted-foreground)]">
                      {KG_LABELS[tid] ?? tid}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        </article>
      )}
    </main>
  )
}
