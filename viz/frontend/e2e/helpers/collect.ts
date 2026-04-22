/**
 * Per-test telemetry collector — ported from claude-code-sessions.
 *
 * Wires Playwright page events to in-memory buffers, then writes three
 * artifacts per test (`{slug}.log`, `{slug}.network.json`) and asserts
 * zero browser console errors (after filtering known noise).
 *
 * The screenshot itself is taken by the test body — we handle everything
 * that isn't the pixel image.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, type Page, type Request as PlaywrightRequest } from '@playwright/test'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUTPUT_DIR = resolve(HERE, '..', '..', 'e2e-screenshots')

export interface NetworkTiming {
  url: string
  method: string
  status: number | null
  /** ms since test start */
  start_offset_ms: number
  /** wall-clock duration */
  duration_ms: number
  /** xhr | fetch | document | script | stylesheet | image | ... */
  resource_type: string
  /** true iff url contains /api/ — used to surface backend-specific stats */
  is_api: boolean
}

export interface TestCollector {
  /** Flush {slug}.log and {slug}.network.json to e2e-screenshots/ */
  writeLog: (slug: string) => void
  /**
   * Assert no unexpected console errors. Global noise (vite HMR, React
   * DevTools banner, act() warnings, favicon) is always filtered.
   *
   * `allowedSubstrings` lets a test whitelist expected errors — e.g. a
   * "not found" page that intentionally triggers a 404 passes
   * `['404']` or `['Failed to load resource']`.
   */
  assertNoErrors: (allowedSubstrings?: string[]) => void
  /** The output directory — useful for tests that also save custom artifacts */
  readonly outputDir: string
}

/**
 * Filters that suppress known-noise from the error list. An entry that
 * CONTAINS any of these substrings is treated as not-an-error.
 */
const ERROR_NOISE = ['act(', 'favicon', '[vite]', 'Download the React DevTools']

export function collectTestIO(page: Page): TestCollector {
  const testStart = Date.now()
  const lines: string[] = []
  const errors: string[] = []
  const network: NetworkTiming[] = []
  const pending = new Map<PlaywrightRequest, number>()

  page.on('pageerror', (err) => {
    lines.push(`[PAGE_ERROR] ${err.message}`)
    errors.push(err.message)
  })

  page.on('console', (msg) => {
    const level = msg.type().toUpperCase().padEnd(7)
    lines.push(`[${level}] ${msg.text()}`)
    if (msg.type() === 'error') errors.push(msg.text())
  })

  page.on('request', (req) => {
    pending.set(req, Date.now())
  })

  page.on('requestfinished', async (req) => {
    const start = pending.get(req)
    if (start === undefined) return
    pending.delete(req)
    const res = await req.response()
    network.push({
      url: req.url(),
      method: req.method(),
      status: res ? res.status() : null,
      start_offset_ms: start - testStart,
      duration_ms: Date.now() - start,
      resource_type: req.resourceType(),
      is_api: req.url().includes('/api/'),
    })
  })

  page.on('requestfailed', (req) => {
    const start = pending.get(req)
    if (start === undefined) return
    pending.delete(req)
    network.push({
      url: req.url(),
      method: req.method(),
      status: null,
      start_offset_ms: start - testStart,
      duration_ms: Date.now() - start,
      resource_type: req.resourceType(),
      is_api: req.url().includes('/api/'),
    })
  })

  return {
    outputDir: OUTPUT_DIR,

    writeLog(slug: string) {
      mkdirSync(OUTPUT_DIR, { recursive: true })
      writeFileSync(`${OUTPUT_DIR}/${slug}.log`, lines.join('\n') + '\n', 'utf-8')

      const apiCalls = network.filter((n) => n.is_api)
      const wallClockEnd = network.reduce(
        (max, n) => Math.max(max, n.start_offset_ms + n.duration_ms),
        0,
      )
      const summary = {
        test_start_ms: testStart,
        wall_clock_duration_ms: wallClockEnd,
        total_requests: network.length,
        total_duration_ms: network.reduce((s, n) => s + n.duration_ms, 0),
        api_requests: apiCalls.length,
        api_duration_ms: apiCalls.reduce((s, n) => s + n.duration_ms, 0),
        slowest_api: [...apiCalls]
          .sort((a, b) => b.duration_ms - a.duration_ms)
          .slice(0, 5)
          .map((n) => ({
            url: n.url,
            start_offset_ms: n.start_offset_ms,
            duration_ms: n.duration_ms,
            status: n.status,
          })),
        all_requests: [...network].sort((a, b) => a.start_offset_ms - b.start_offset_ms),
      }
      writeFileSync(
        `${OUTPUT_DIR}/${slug}.network.json`,
        JSON.stringify(summary, null, 2) + '\n',
        'utf-8',
      )
    },

    assertNoErrors(allowedSubstrings: string[] = []) {
      const filters = [...ERROR_NOISE, ...allowedSubstrings]
      const real = errors.filter((e) => !filters.some((n) => e.includes(n)))
      expect(real, `Browser console errors:\n${real.join('\n')}`).toHaveLength(0)
    },
  }
}
