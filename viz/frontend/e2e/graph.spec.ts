import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('Graph Explorer', () => {
  test('renders graph page and loads network', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Click the "Graph" sidebar link to navigate
    await page.getByRole('link', { name: 'Graph' }).click();
    await page.waitForURL(/\/graph/);
    await checkpoint(page, 'graph-page-loaded');

    // The dataset picker shows "Edge Tables" heading when no dataset selected
    await expect(page.getByRole('heading', { name: 'Edge Tables' })).toBeVisible({ timeout: 10_000 });

    // Wait for at least one edge table card to appear (contains "edges" badge)
    await expect(async () => {
      const cards = await page.locator('text=/\\d+ edges/').count();
      expect(cards).toBeGreaterThanOrEqual(1);
    }).toPass({ timeout: 10_000 });

    await checkpoint(page, 'graph-tables-discovered');

    // Click the first edge table card to select it
    const firstCard = page.locator('[class*="cursor-pointer"]').first();
    await firstCard.click();
    await page.waitForURL(/\/graph\/.+/);

    // Graph canvas must show a definitive state: data loaded, empty, or error.
    // (Never a blank canvas with no indication of state.)
    await expect(async () => {
      const stats = page.getByTestId('graph-stats');
      const empty = page.getByTestId('graph-empty-state');
      const error = page.getByTestId('graph-state-error');
      const isStats = await stats.isVisible();
      const isEmpty = await empty.isVisible();
      const isError = await error.isVisible();
      expect(isStats || isEmpty || isError, 'Graph canvas must show definitive state').toBe(true);
    }).toPass({ timeout: 15_000 });

    // Stronger check: the graph should have loaded actual data for these demo databases
    await expect(page.getByTestId('graph-stats')).toBeVisible({ timeout: 5_000 });

    await checkpoint(page, 'graph-network-rendered');

    // Centrality selector should be present in the sidebar
    const centralitySelect = page.locator('select').first();
    await expect(centralitySelect).toBeVisible();

    await checkpoint(page, 'graph-centrality-visible');
  });
});
