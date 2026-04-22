import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { ApiError, type DatabaseInfo, fetchDatabases } from '../lib/api-client'
import { ThemeToggle } from './ThemeToggle'

const COLLAPSE_KEY = 'muninn-viz:sidebar-collapsed'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; databases: DatabaseInfo[] }

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

export function Sidebar() {
  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed())
  const [state, setState] = useState<LoadState>({ status: 'loading' })

  useEffect(() => {
    fetchDatabases()
      .then((databases) => setState({ status: 'ready', databases }))
      .catch((err: unknown) => {
        const message =
          err instanceof ApiError
            ? `API error ${err.status}`
            : err instanceof Error
              ? err.message
              : 'unknown error'
        setState({ status: 'error', message })
      })
  }, [])

  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev
      writeCollapsed(next)
      return next
    })
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
              {state.databases.map((db) => (
                <li key={db.id}>
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
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className={`border-t border-[var(--color-border-subtle)] p-2 ${collapsed ? 'flex justify-center' : 'flex items-center justify-between'}`}>
        {!collapsed && (
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-muted-foreground)]">Theme</span>
        )}
        <ThemeToggle />
      </div>
    </aside>
  )
}
