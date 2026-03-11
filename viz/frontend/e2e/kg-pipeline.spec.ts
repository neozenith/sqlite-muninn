import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('KG Query (Escalator: Python Server-Backed)', () => {
  test('redirects / to /kg/query/', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');
    await page.waitForURL(/\/kg\/query\//);
    await checkpoint(page, 'kg-root-redirect');
  });

  test('redirects /kg/ to /kg/query/', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/kg/');
    await page.waitForURL(/\/kg\/query\//);
    await checkpoint(page, 'kg-redirect');
  });

  test('3-panel KG search with matching checkpoints', async ({ page }) => {
    test.setTimeout(60_000);
    setupConsoleMonitor(page);

    // Navigate directly to the KG query page
    await page.goto('/kg/query/');

    // Ensure the default DB (3300_NomicEmbed) is selected — previous tests may
    // have switched to a DB without full KG tables.
    const dbSelector = page.getByTestId('db-selector');
    await expect(dbSelector).toBeVisible({ timeout: 10_000 });
    await dbSelector.selectOption('3300_NomicEmbed');
    // Wait for DB switch to complete
    await expect(page.getByTestId('db-switching')).not.toBeVisible({ timeout: 15_000 });

    // 1. Initial state — 3-panel layout before search
    const queryInput = page.locator('input[placeholder*="knowledge graph"]');
    await expect(queryInput).toBeVisible({ timeout: 10_000 });
    const searchButton = page.getByRole('button', { name: 'Search' });
    await expect(searchButton).toBeVisible();
    await checkpoint(page, 'viz-checkpoint-01-initial-state');

    // 2. Type query character by character for video
    await queryInput.pressSequentially('trade and commerce between nations', { delay: 50 });
    await searchButton.click();

    // 3. Wait for FTS results in left panel (look for "X results" count text)
    await expect(page.getByText(/\d+ results/)).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, 'viz-checkpoint-02-fts-results');

    // 4. Check embedding panel shows points count
    await expect(page.getByText(/\d+ points/)).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, 'viz-checkpoint-03-embedding-results');

    // 5. Check graph panel shows nodes count
    await expect(page.getByText(/\d+ nodes/)).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, 'viz-checkpoint-04-graph-results');

    // 6. Final state — all three panels populated, hold for 6s
    await page.waitForTimeout(6_000);
    await checkpoint(page, 'viz-checkpoint-05-final-state');
  });
});

test.describe('Embeddings Route', () => {
  test('redirects /embeddings/ to /embeddings/chunks_vec/', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/embeddings/');
    await page.waitForURL(/\/embeddings\/chunks_vec\//);
    await checkpoint(page, 'embeddings-redirect');
  });
});

test.describe('Graph Route', () => {
  test('redirects /graph/ to /graph/edges/', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/graph/');
    await page.waitForURL(/\/graph\/edges\//);
    await checkpoint(page, 'graph-redirect');
  });
});
