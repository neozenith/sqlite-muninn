import type { Page } from '@playwright/test'

/**
 * Assert the React app has mounted and the initial fetches have settled.
 *
 * Ordering matters:
 *  1. Mount: the React tree has rendered into #root
 *  2. Network idle: no in-flight requests for ~500ms
 *  3. No stuck loading text visible
 *
 * The mount wait is strict (fails the test if React doesn't mount in 15s);
 * the networkidle/loading waits are best-effort because some legitimate
 * pages keep a background poll running.
 */
export async function waitForPageLoad(page: Page): Promise<void> {
  await page.waitForFunction(
    () => (document.getElementById('root')?.children.length ?? 0) > 0,
    { timeout: 15000 },
  )
  await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})
  await page
    .waitForFunction(
      () => !/Loading(…|\.\.\.)/i.test(document.body.innerText),
      { timeout: 5000 },
    )
    .catch(() => {})
}
