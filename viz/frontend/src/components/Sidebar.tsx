import { useEffect, useMemo, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { ApiError, type DatabaseInfo, type TablesResponse, fetchDatabases, fetchTables } from '../lib/api-client'
import { ThemeToggle } from './ThemeToggle'

const COLLAPSE_KEY = 'muninn-viz:sidebar-collapsed'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; databases: DatabaseInfo[] }

const EMBED_LABELS: Record<string, string> = {
  chunks: 'Chunks',
  entities: 'Entities',
}

const KG_LABELS: Record<string, string> = {
  base: 'Base',
  er: 'Entity-resolved',
}

const readCollapsed = (): boolean => {
  try {
    return (
      typeof window !== 'undefined' &&
      typeof window.localStorage?.getItem === 'function' &&
      window.localStorage.getItem(COLLAPSE_KEY) === '1'
    )
  } catch {
    return false
  }
}

const writeCollapsed = (next: boolean): void => {
  try {
    if (typeof window !== 'undefined' && typeof window.localStorage?.setItem === 'function') {
      window.localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0')
    }
  } catch {
    /* ignore */
  }
}

/**
 * Parse the first path segment as the active database id. Returns null on /
 * or any URL without a segment. Decodes percent-encoding so slugs with
 * unusual characters still match manifest ids.
 */
const activeDbIdFromPath = (pathname: string): string | null => {
  const first = pathname.split('/').filter(Boolean)[0]
  if (!first) return null
  try {
    return decodeURIComponent(first)
  } catch {
    return first
  }
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed())
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  const [tablesByDb, setTablesByDb] = useState<Record<string, TablesResponse>>({})
  const { pathname } = useLocation()
  const activeDbId = useMemo(() => activeDbIdFromPath(pathname), [pathname])

  useEffect(() => {
    fetchDatabases()
      .then((databases) => setState({ status: 'ready', databases }))
      .catch((err: unknown) => {
        const message =
          err instanceof ApiError ? `API error ${err.status}` : err instanceof Error ? err.message : 'unknown error'
        setState({ status: 'error', message })
      })
  }, [])

  // Lazy-fetch table lists for the active DB. Cache by id so navigating back
  // to a previously-viewed DB doesn't round-trip again.
  useEffect(() => {
    if (!activeDbId || tablesByDb[activeDbId]) return
    if (state.status !== 'ready') return
    if (!state.databases.some((db) => db.id === activeDbId)) return
    let cancelled = false
    fetchTables(activeDbId)
      .then((tables) => {
        if (cancelled) return
        setTablesByDb((prev) => ({ ...prev, [activeDbId]: tables }))
      })
      .catch(() => {
        /* surface errors only on the page itself — keep the sidebar quiet */
      })
    return () => {
      cancelled = true
    }
  }, [activeDbId, state, tablesByDb])

  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev
      writeCollapsed(next)
      return next
    })
  }

  const renderSubtree = (db: DatabaseInfo) => {
    const tables = tablesByDb[db.id]
    if (!tables) {
      return (
        <p
          className="px-2 pb-1 pl-4 text-[10px] text-[var(--color-muted-foreground)]"
          data-testid={`sidebar-subtree-loading-${db.id}`}
        >
          Loading…
        </p>
      )
    }

    const embedItems = tables.embed_tables
    const kgItems = tables.kg_tables

    const leafClass = ({ isActive }: { isActive: boolean }) =>
      [
        'block rounded px-2 py-1 text-[11px] transition',
        isActive
          ? 'bg-[var(--color-surface-elevated)] text-[var(--color-foreground)] ring-1 ring-[var(--color-accent)]'
          : 'text-[var(--color-muted-foreground)] hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-foreground)]',
      ].join(' ')

    return (
      <div
        className="mt-1 ml-3 flex flex-col gap-1 border-l border-[var(--color-border-subtle)] pl-2"
        data-testid={`sidebar-subtree-${db.id}`}
      >
        {embedItems.length > 0 && (
          <div>
            <div className="px-1 pb-0.5 text-[9px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
              Embeddings
            </div>
            <ul className="flex flex-col gap-0.5">
              {embedItems.map((tid) => (
                <li key={`embed-${tid}`}>
                  <NavLink
                    to={`/${encodeURIComponent(db.id)}/embed/${encodeURIComponent(tid)}/`}
                    end
                    data-testid={`sidebar-embed-${db.id}-${tid}`}
                    className={leafClass}
                  >
                    {EMBED_LABELS[tid] ?? tid}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        )}

        {kgItems.length > 0 && (
          <div>
            <div className="px-1 pb-0.5 text-[9px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
              Knowledge graph
            </div>
            <ul className="flex flex-col gap-0.5">
              {kgItems.map((tid) => (
                <li key={`kg-${tid}`}>
                  <NavLink
                    to={`/${encodeURIComponent(db.id)}/kg/${encodeURIComponent(tid)}/`}
                    end
                    data-testid={`sidebar-kg-${db.id}-${tid}`}
                    className={leafClass}
                  >
                    {KG_LABELS[tid] ?? tid}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        )}

        {embedItems.length === 0 && kgItems.length === 0 && (
          <p className="px-1 text-[10px] text-[var(--color-muted-foreground)]">No viz tables</p>
        )}
      </div>
    )
  }

  return (
    <aside
      data-testid="sidebar"
      data-collapsed={collapsed}
      className={`flex h-screen shrink-0 flex-col border-r border-[var(--color-border-subtle)] bg-[var(--color-surface-sunken)] text-[var(--color-foreground)] transition-[width] duration-200 ${
        collapsed ? 'w-12' : 'w-64'
      }`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border-subtle)] px-3 py-2">
        {!collapsed && (
          <NavLink
            to="/"
            end
            className="truncate text-sm font-semibold tracking-tight hover:text-[var(--color-accent)]"
            data-testid="sidebar-home-link"
          >
            muninn-viz
          </NavLink>
        )}
        <button
          type="button"
          onClick={toggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          data-testid="sidebar-toggle"
          className="inline-flex h-7 w-7 items-center justify-center rounded border border-[var(--color-border-subtle)] text-xs hover:border-[var(--color-accent)]"
        >
          {collapsed ? '»' : '«'}
        </button>
      </div>

      {!collapsed && (
        <div className="flex-1 overflow-y-auto px-2 py-3" data-testid="sidebar-db-list">
          <div className="px-1 pb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)]">
            Databases
          </div>
          {state.status === 'loading' && (
            <p className="px-1 py-2 text-xs text-[var(--color-muted-foreground)]" data-testid="sidebar-loading">
              Loading…
            </p>
          )}
          {state.status === 'error' && (
            <p className="px-1 py-2 text-xs text-red-500" data-testid="sidebar-error">
              {state.message}
            </p>
          )}
          {state.status === 'ready' && state.databases.length === 0 && (
            <p className="px-1 py-2 text-xs text-[var(--color-muted-foreground)]">None</p>
          )}
          {state.status === 'ready' && state.databases.length > 0 && (
            <ul className="flex flex-col gap-0.5">
              {state.databases.map((db) => {
                const expanded = activeDbId === db.id
                return (
                  <li key={db.id} data-expanded={expanded}>
                    <NavLink
                      to={`/${encodeURIComponent(db.id)}/`}
                      data-testid={`sidebar-db-${db.id}`}
                      className={({ isActive }) =>
                        [
                          'block rounded px-2 py-1.5 text-xs transition',
                          isActive
                            ? 'bg-[var(--color-surface-elevated)] text-[var(--color-foreground)] ring-1 ring-[var(--color-accent)]'
                            : 'text-[var(--color-muted-foreground)] hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-foreground)]',
                        ].join(' ')
                      }
                    >
                      <span className="block truncate font-medium">{db.label}</span>
                      <span className="block truncate font-mono text-[10px] opacity-70">{db.id}</span>
                    </NavLink>
                    {expanded && renderSubtree(db)}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}

      <div
        className={`border-t border-[var(--color-border-subtle)] p-2 ${collapsed ? 'flex justify-center' : 'flex items-center justify-between'}`}
      >
        {!collapsed && (
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-muted-foreground)]">Theme</span>
        )}
        <ThemeToggle />
      </div>
    </aside>
  )
}
