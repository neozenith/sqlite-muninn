/**
 * Warm the Vite dev server once before any workers start.
 *
 * Vite compiles routes lazily. Without this, the first test in each
 * worker pays the compile cost, which often blows past Playwright's
 * per-test timeout and produces flaky "React never mounted" failures.
 */
import { chromium } from '@playwright/test'

const FRONTEND_URL = 'http://localhost:5282'

export default async function globalSetup(): Promise<void> {
  const browser = await chromium.launch()
  const page = await browser.newPage()
  try {
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle', timeout: 60000 })
    await page
      .waitForFunction(
        () => (document.getElementById('root')?.children.length ?? 0) > 0,
        { timeout: 30000 },
      )
      .catch(() => {
        console.warn(`Global setup: React failed to mount at ${FRONTEND_URL}`)
      })
  } finally {
    await page.close()
    await browser.close()
  }
}
