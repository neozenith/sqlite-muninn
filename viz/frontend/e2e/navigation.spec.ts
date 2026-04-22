/**
 * Behavioral tests — click-through navigation + URL contracts.
 *
 * These are NOT permutation tests. They still capture the full telemetry
 * trio ({slug}.png + .log + .network.json) because the artifacts are
 * valuable even when the test passes.
 */
import { expect, test } from '@playwright/test'
import { collectTestIO } from './helpers/collect'
import { loadManifest } from './helpers/sitemap'
import { waitForPageLoad } from './helpers/wait'

const MANIFEST = loadManifest()
const FIRST_DB = MANIFEST.databases[0]

test('home → database card click navigates to /:database_id/', async ({ page }) => {
  const io = collectTestIO(page)
  const slug = 'BEHAVIORAL-home-to-database'

  try {
    await page.goto('/')
    await waitForPageLoad(page)

    const card = page.getByTestId(`db-card-${FIRST_DB.id}`)
    await expect(card).toBeVisible()
    await card.click()
    await waitForPageLoad(page)

    await expect(page).toHaveURL(new RegExp(`/${FIRST_DB.id}/?$`))
    await expect(page.getByTestId('database-detail')).toBeVisible()

    await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
    io.assertNoErrors()
  } finally {
    io.writeLog(slug)
  }
})

test('database page → back link returns home', async ({ page }) => {
  const io = collectTestIO(page)
  const slug = 'BEHAVIORAL-database-back-to-home'

  try {
    await page.goto(`/${FIRST_DB.id}/`)
    await waitForPageLoad(page)
    await expect(page.getByTestId('database-detail')).toBeVisible()

    await page.getByTestId('back-to-home').click()
    await waitForPageLoad(page)

    await expect(page).toHaveURL(/\/$/)
    await expect(page.getByTestId('home-database-list')).toBeVisible()

    await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
    io.assertNoErrors()
  } finally {
    io.writeLog(slug)
  }
})

test('unknown database id shows not-found state without crashing', async ({ page }) => {
  const io = collectTestIO(page)
  const slug = 'BEHAVIORAL-database-not-found'

  try {
    await page.goto('/definitely_not_a_real_database_id/')
    await waitForPageLoad(page)

    await expect(page.getByTestId('database-not-found')).toBeVisible()

    await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
    // Chromium logs the 404 fetch as a "Failed to load resource" console
    // error — we expect that exactly here, so allowlist it.
    io.assertNoErrors(['Failed to load resource'])
  } finally {
    io.writeLog(slug)
  }
})

test('deep link to a specific database loads directly', async ({ page }) => {
  const io = collectTestIO(page)
  const lastDb = MANIFEST.databases[MANIFEST.databases.length - 1]
  const slug = 'BEHAVIORAL-deep-link'

  try {
    await page.goto(`/${lastDb.id}/`)
    await waitForPageLoad(page)

    await expect(page.getByTestId('database-detail')).toBeVisible()
    await expect(page.getByRole('heading', { name: lastDb.label })).toBeVisible()

    await page.screenshot({ path: `${io.outputDir}/${slug}.png`, fullPage: true })
    io.assertNoErrors()
  } finally {
    io.writeLog(slug)
  }
})
