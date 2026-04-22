/**
 * Permutation coverage — one test per (section × database × table) tuple.
 *
 * Every test follows the same checklist:
 *   1. Navigate to the URL
 *   2. Wait for React mount + network idle
 *   3. Assert the section's "loaded" testid is visible
 *   4. Take screenshot
 *   5. Assert zero console errors (after noise filtering)
 *
 * The whole body is wrapped in try/finally so {slug}.log and
 * {slug}.network.json are flushed even when the test fails.
 *
 * See viz/CLAUDE.md § "E2E Testing Pattern" for the full spec.
 */
import { expect, test } from '@playwright/test'
import { collectTestIO } from './helpers/collect'
import { allPermutations, permutationLabel, screenshotSlug } from './helpers/sitemap'
import { waitForPageLoad } from './helpers/wait'

for (const permutation of allPermutations()) {
  const { section, database, table } = permutation
  const slug = screenshotSlug(section, database, table)
  const label = permutationLabel(permutation)

  test(label, async ({ page }) => {
    // fcose layout on a 6K-node KG can take ~30-60s — per-section override.
    test.setTimeout(Math.max(90000, section.loadTimeoutMs + 30000))

    const io = collectTestIO(page)

    try {
      await page.goto(section.pathFor(database, table))
      await waitForPageLoad(page)

      await expect(page.getByTestId(section.loadedTestId)).toHaveCount(1, {
        timeout: section.loadTimeoutMs,
      })

      await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
      io.assertNoErrors()
    } finally {
      io.writeLog(slug)
    }
  })
}
