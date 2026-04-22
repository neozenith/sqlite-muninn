import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { type ResolvedTheme, ThemeContext, type ThemeContextValue, type ThemeMode } from './theme-context'

const STORAGE_KEY = 'muninn-viz:theme'

const safeGet = (key: string): string | null => {
  try {
    return typeof window !== 'undefined' && typeof window.localStorage?.getItem === 'function'
      ? window.localStorage.getItem(key)
      : null
  } catch {
    return null
  }
}

const safeSet = (key: string, value: string): void => {
  try {
    if (typeof window !== 'undefined' && typeof window.localStorage?.setItem === 'function') {
      window.localStorage.setItem(key, value)
    }
  } catch {
    /* storage unavailable — ignore */
  }
}

const readStoredMode = (): ThemeMode => {
  const raw = safeGet(STORAGE_KEY)
  return raw === 'light' || raw === 'dark' || raw === 'system' ? raw : 'system'
}

const prefersDark = (): boolean => {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(() => readStoredMode())
  const [osPrefersDark, setOsPrefersDark] = useState<boolean>(() => prefersDark())

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const mql = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = () => setOsPrefersDark(mql.matches)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])

  // `resolved` is a pure function of mode + OS preference — derive it with
  // useMemo rather than duplicating it into state. The DOM side-effects
  // (html.dark class, data-theme attribute) live in a separate effect.
  const resolved = useMemo<ResolvedTheme>(
    () => (mode === 'system' ? (osPrefersDark ? 'dark' : 'light') : mode),
    [mode, osPrefersDark],
  )

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', resolved === 'dark')
    root.dataset.theme = resolved
  }, [resolved])

  const setMode = useCallback((next: ThemeMode) => {
    setModeState(next)
    safeSet(STORAGE_KEY, next)
  }, [])

  const toggle = useCallback(() => {
    setModeState((prev) => {
      const currentResolved: ResolvedTheme = prev === 'system' ? (prefersDark() ? 'dark' : 'light') : prev
      const next: ThemeMode = currentResolved === 'dark' ? 'light' : 'dark'
      safeSet(STORAGE_KEY, next)
      return next
    })
  }, [])

  const value = useMemo<ThemeContextValue>(
    () => ({ mode, resolved, setMode, toggle }),
    [mode, resolved, setMode, toggle],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}
