/**
 * E2E tests for the muninn WASM demo.
 *
 * Tests the full flow: page load → library init → DB load → search → visualization.
 * Timeouts are generous to accommodate first-time CDN downloads and model loading.
 *
 * Screenshots use wasm-checkpoint-{N} naming for side-by-side comparison with viz/.
 */
import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('muninn WASM Demo', () => {
  test.setTimeout(120_000); // 2 minutes for CDN + model downloads

  test('loads and searches the knowledge graph', async ({ page }) => {
    setupConsoleMonitor(page);
    page.on('console', (msg) => {
      if (msg.text().includes('CTE graph')) console.log(`[page] ${msg.text()}`);
    });

    // 1. Navigate to the demo page
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('muninn');
    await checkpoint(page, 'wasm-checkpoint-01-page-loaded');

    // 2. Wait for WASM module to initialize
    await expect(page.locator('#status-wasm[data-status="ready"]')).toBeVisible({
      timeout: 30_000,
    });
    await checkpoint(page, 'wasm-checkpoint-02-wasm-ready');

    // 3. Verify database was loaded (footer shows schema info)
    await expect(page.locator('#db-status')).toContainText('chunks', {
      timeout: 10_000,
    });
    await checkpoint(page, 'wasm-checkpoint-03-database-loaded');

    // 4. Wait for libraries to load
    await expect(
      page.locator('#status-deckgl[data-status="ready"], #status-deckgl[data-status="error"]')
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.locator('#status-cytoscape[data-status="ready"], #status-cytoscape[data-status="error"]')
    ).toBeVisible({ timeout: 10_000 });
    await checkpoint(page, 'wasm-checkpoint-04-viz-libraries-loaded');

    // 5. Wait for Transformers.js model (first download can be 30-60s for fp32)
    await expect(page.locator('#status-transformers[data-status="ready"]')).toBeVisible({
      timeout: 90_000,
    });
    await checkpoint(page, 'wasm-checkpoint-05-transformers-ready');

    // 6. Search input should now be enabled — type character by character for video
    const searchInput = page.locator('#search-input');
    await expect(searchInput).toBeEnabled({ timeout: 5_000 });
    await checkpoint(page, 'wasm-checkpoint-06-initial-state');

    // 7. Perform a search with human-like typing
    await searchInput.pressSequentially('trade and commerce between nations', { delay: 50 });

    // Wait for FTS results in left panel (instant) and embedding results
    await expect(page.locator('.result-card').first()).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, 'wasm-checkpoint-07-fts-results');

    // 8. Verify Deck.GL point count updated (embedding search complete)
    const deckglCount = page.locator('#deckgl-count');
    await expect(deckglCount).not.toHaveText('0 points', { timeout: 30_000 });
    await checkpoint(page, 'wasm-checkpoint-08-embedding-results');

    // 9. Verify Cytoscape graph has nodes (CTE graph search)
    const cytoscapeCount = page.locator('#cytoscape-count');
    await expect(cytoscapeCount).not.toHaveText('0 nodes', { timeout: 10_000 });
    await checkpoint(page, 'wasm-checkpoint-09-graph-results');

    // 10. Verify graph has edges (BFS neighborhood should have connections)
    const graphInfo = await page.evaluate(() => {
      // @ts-expect-error cyInstance is a global
      const cy = window.cyInstance;
      if (!cy) return { nodes: -1, edges: -1, nodeIds: [] };
      return {
        nodes: cy.nodes().length,
        edges: cy.edges().length,
        nodeIds: cy.nodes().map((n: any) => n.id()).slice(0, 10),
      };
    });
    console.log('Graph info:', JSON.stringify(graphInfo));
    expect(graphInfo.edges, 'Graph should have edges connecting nodes').toBeGreaterThan(0);

    // 11. Verify graph controls (sliders + Re-run Layout button) are visible
    await expect(page.locator('#graph-controls')).toBeVisible({ timeout: 5_000 });
    await expect(page.locator('#run-layout-btn')).toBeVisible();
    await checkpoint(page, 'wasm-checkpoint-10-graph-controls');

    // 12. Final state — all three panels populated, hold for 6s
    await page.waitForTimeout(6_000);
    await checkpoint(page, 'wasm-checkpoint-11-final-state');
  });
});
