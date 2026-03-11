import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('VSS Explorer', () => {
  test('renders embeddings page and loads canvas state', async ({ page }) => {
    setupConsoleMonitor(page);

    // Navigate directly to the default embeddings dataset (index pages removed)
    await page.goto('/embeddings/chunks_vec');
    await checkpoint(page, 'vss-page-loaded');

    // The sidebar "Embeddings" link should be visible
    const sidebarLink = page.getByRole('link', { name: 'Embeddings' });
    await expect(sidebarLink).toBeVisible({ timeout: 10_000 });

    // The canvas area must show a definitive state: loading, stats, empty, or error.
    // (Never a blank canvas with no indication of state.)
    const loading = page.getByTestId('embedding-state-loading');
    const stats = page.getByTestId('embedding-stats');
    const empty = page.getByTestId('embedding-empty-state');
    const error = page.getByTestId('embedding-state-error');

    await expect(loading.or(stats).or(empty).or(error)).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, 'vss-canvas-definitive-state');

    // If stats are visible, verify point count > 0
    if (await stats.isVisible()) {
      const count = Number(await stats.getAttribute('data-count'));
      expect(count, 'chunks_vec should have points > 0').toBeGreaterThan(0);
      await checkpoint(page, 'vss-data-loaded');
    }
  });
});
