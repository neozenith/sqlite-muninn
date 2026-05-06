import { useState } from 'react'
import type { ReactNode } from 'react'

interface RightPanelProps {
  /** Stable key for persisting collapsed state in localStorage. */
  storageKey: string
  /** Heading shown when the panel is open. */
  title: string
  /** Panel body. Sections are rendered vertically in the order given. */
  children: ReactNode
  /** Optional testid suffix — the outer aside gets `right-panel-${testId}`. */
  testId?: string
}

const storageKeyFor = (key: string): string => `muninn-viz:right-panel:${key}`

const readCollapsed = (key: string): boolean => {
  try {
    if (typeof window === 'undefined' || typeof window.localStorage?.getItem !== 'function') {
      return false
    }
    return window.localStorage.getItem(storageKeyFor(key)) === '1'
  } catch {
    return false
  }
}

const writeCollapsed = (key: string, next: boolean): void => {
  try {
    if (typeof window !== 'undefined' && typeof window.localStorage?.setItem === 'function') {
      window.localStorage.setItem(storageKeyFor(key), next ? '1' : '0')
    }
  } catch {
    /* ignore */
  }
}

/**
 * Collapsible right-side panel. Defaults to open on first visit; collapsed
 * state is persisted per `storageKey` so different pages can remember
 * independently.
 */
export function RightPanel({ storageKey, title, children, testId }: RightPanelProps) {
  const [collapsed, setCollapsed] = useState<boolean>(() => readCollapsed(storageKey))
  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev
      writeCollapsed(storageKey, next)
      return next
    })
  }

  return (
    <aside
      data-testid={testId ? `right-panel-${testId}` : 'right-panel'}
      data-collapsed={collapsed}
      className={`flex h-full shrink-0 flex-col border-l border-[var(--color-border-subtle)] bg-[var(--color-surface-sunken)] text-[var(--color-foreground)] transition-[width] duration-200 ${
        collapsed ? 'w-10' : 'w-80'
      }`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border-subtle)] px-3 py-2">
        {!collapsed && <h2 className="truncate text-sm font-semibold tracking-tight">{title}</h2>}
        <button
          type="button"
          onClick={toggle}
          aria-label={collapsed ? `Expand ${title}` : `Collapse ${title}`}
          data-testid={testId ? `right-panel-toggle-${testId}` : 'right-panel-toggle'}
          className="inline-flex h-7 w-7 items-center justify-center rounded border border-[var(--color-border-subtle)] text-xs hover:border-[var(--color-accent)]"
        >
          {collapsed ? '«' : '»'}
        </button>
      </div>
      {!collapsed && <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-3 text-sm">{children}</div>}
    </aside>
  )
}

interface PanelSectionProps {
  title: string
  /** Right-aligned meta text next to the title, e.g. count or status. */
  meta?: ReactNode
  children: ReactNode
  /**
   * Stable key for persisting open/closed state in localStorage. Optional —
   * sections without a key are still collapsible, but their state is
   * in-memory only and resets on page reload.
   */
  storageKey?: string
  /** Default open state on first visit. Defaults to true. */
  defaultOpen?: boolean
  /** Optional testid suffix — the outer section gets `panel-section-${testId}`. */
  testId?: string
}

const sectionStorageKey = (key: string): string => `muninn-viz:panel-section:${key}`

const readSectionOpen = (key: string | undefined, fallback: boolean): boolean => {
  if (key === undefined) return fallback
  try {
    if (typeof window === 'undefined' || typeof window.localStorage?.getItem !== 'function') {
      return fallback
    }
    const raw = window.localStorage.getItem(sectionStorageKey(key))
    if (raw === '1') return true
    if (raw === '0') return false
    return fallback
  } catch {
    return fallback
  }
}

const writeSectionOpen = (key: string | undefined, next: boolean): void => {
  if (key === undefined) return
  try {
    if (typeof window !== 'undefined' && typeof window.localStorage?.setItem === 'function') {
      window.localStorage.setItem(sectionStorageKey(key), next ? '1' : '0')
    }
  } catch {
    /* ignore */
  }
}

/**
 * Titled, collapsible section used inside a RightPanel. Always collapsible —
 * pass `storageKey` to additionally persist the open/closed state across
 * reloads.
 */
export function PanelSection({ title, meta, children, storageKey, defaultOpen = true, testId }: PanelSectionProps) {
  const [open, setOpen] = useState<boolean>(() => readSectionOpen(storageKey, defaultOpen))
  const toggle = () => {
    setOpen((prev) => {
      const next = !prev
      writeSectionOpen(storageKey, next)
      return next
    })
  }

  // Stable testid suffix: explicit `testId` wins, otherwise fall back to
  // `storageKey` (which is already a stable per-section identifier). This
  // keeps the toggle reachable from E2E tests without forcing every caller
  // to pass two redundant props.
  const idSuffix = testId ?? storageKey
  return (
    <section data-testid={idSuffix ? `panel-section-${idSuffix}` : undefined} data-open={open}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="mb-1 flex w-full cursor-pointer items-baseline justify-between gap-2 rounded-sm px-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]"
        data-testid={idSuffix ? `panel-section-toggle-${idSuffix}` : undefined}
      >
        <span className="flex items-center gap-1">
          <span aria-hidden className="inline-block w-3 text-center">
            {open ? '▾' : '▸'}
          </span>
          <span>{title}</span>
        </span>
        {meta !== undefined && meta !== null && <span className="normal-case tracking-normal">{meta}</span>}
      </button>
      {open && <div className="flex flex-col gap-2">{children}</div>}
    </section>
  )
}
