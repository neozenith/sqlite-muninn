import { useTheme } from '../lib/theme-context'

export function ThemeToggle() {
  const { resolved, toggle } = useTheme()
  const nextLabel = resolved === 'dark' ? 'light' : 'dark'

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${nextLabel} theme`}
      data-testid="theme-toggle"
      data-theme={resolved}
      className="inline-flex h-8 w-8 items-center justify-center rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-elevated)] text-[var(--color-foreground)] transition hover:border-[var(--color-accent)]"
    >
      <span aria-hidden className="text-sm">
        {resolved === 'dark' ? '☀' : '☾'}
      </span>
    </button>
  )
}
