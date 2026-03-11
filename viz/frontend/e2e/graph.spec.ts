import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('Graph Explorer', () => {
  test('renders graph page and loads network', async ({ page }) => {
    setupConsoleMonitor(page);

    // Navigate directly to the default graph dataset (index pages removed)
    await page.goto('/graph/edges');
    await checkpoint(page, 'graph-page-loaded');

    // The sidebar "Graph" link should be visible
    const sidebarLink = page.getByRole('link', { name: 'Graph' });
    await expect(sidebarLink).toBeVisible({ timeout: 10_000 });

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

    await checkpoint(page, 'graph-definitive-state');

    // If stats are visible, verify node count > 0
    const stats = page.getByTestId('graph-stats');
    if (await stats.isVisible()) {
      const count = Number(await stats.getAttribute('data-node-count'));
      expect(count, 'edges graph should have nodes > 0').toBeGreaterThan(0);

      // Centrality selector should be present in the sidebar
      const centralitySelect = page.locator('select').first();
      await expect(centralitySelect).toBeVisible();

      await checkpoint(page, 'graph-network-rendered');
    }
  });
});
