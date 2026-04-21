/**
 * Permutation coverage — one test per (section × database) tuple.
 *
 * Every test follows the same checklist:
 *   1. Navigate to the URL
 *   2. Wait for React mount + network idle
 *   3. Assert the section's "loaded" testid is visible
 *   4. Take screenshot
 *   5. Assert zero console errors (after noise filtering)
 *
 * The whole body is wrapped in try/finally so {slug}.log and
 * {slug}.network.json are flushed even when the test fails — post-mortem
 * telemetry is most valuable exactly when the test didn't pass.
 *
 * See viz/CLAUDE.md § "E2E Testing Pattern" for the full spec.
 */
import { expect, test } from '@playwright/test'
import { collectTestIO } from './helpers/collect'
import { allPermutations, permutationLabel, screenshotSlug } from './helpers/sitemap'
import { waitForPageLoad } from './helpers/wait'

for (const permutation of allPermutations()) {
  const { section, database } = permutation

  test(permutationLabel(permutation), async ({ page }) => {
    const io = collectTestIO(page)
    const slug = screenshotSlug(section, database)

    try {
      await page.goto(section.pathFor(database))
      await waitForPageLoad(page)

      await expect(page.getByTestId(section.loadedTestId(database))).toBeVisible({
        timeout: 10000,
      })

      await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
      io.assertNoErrors()
    } finally {
      io.writeLog(slug)
    }
  })
}
