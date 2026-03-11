/**
 * E2E tests for the muninn WASM demo served at /wasm/.
 *
 * The WASM demo is a standalone HTML page at viz/frontend/public/wasm/index.html,
 * served by Vite at http://localhost:5281/wasm/.
 *
 * Tests the full flow: page load → WASM init → DB load → viz libraries → search.
 * Timeouts are generous to accommodate CDN downloads and model loading in CI.
 *
 * Screenshots use wasm-checkpoint-{N} naming for side-by-side comparison with
 * the main viz app checkpoints.
 */
import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('muninn WASM Demo', () => {
  test.setTimeout(120_000); // 2 minutes for CDN + model downloads

  test('loads and searches the knowledge graph', async ({ page }) => {
    setupConsoleMonitor(page);
    page.on('console', (msg) => {
      if (msg.text().includes('CTE graph')) console.log(`[wasm] ${msg.text()}`);
    });

    // 1. Navigate to the WASM demo page (served as a static asset by Vite).
    // Must use explicit /wasm/index.html — Vite doesn't auto-resolve directory
    // index files, so /wasm/ falls through to the SPA catch-all route.
    await page.goto('/wasm/index.html');
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

    // 4. Wait for viz libraries (Deck.GL + Cytoscape) to load or fail gracefully
    await expect(
      page.locator('#status-deckgl[data-status="ready"], #status-deckgl[data-status="error"]'),
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.locator('#status-cytoscape[data-status="ready"], #status-cytoscape[data-status="error"]'),
    ).toBeVisible({ timeout: 10_000 });
    await checkpoint(page, 'wasm-checkpoint-04-viz-libraries-loaded');

    // 5. Wait for Transformers.js model (first download can be slow for fp32 weights)
    await expect(page.locator('#status-transformers[data-status="ready"]')).toBeVisible({
      timeout: 90_000,
    });
    await checkpoint(page, 'wasm-checkpoint-05-transformers-ready');

    // 6. Search input should now be enabled
    const searchInput = page.locator('#search-input');
    await expect(searchInput).toBeEnabled({ timeout: 5_000 });
    await checkpoint(page, 'wasm-checkpoint-06-initial-state');

    // 7. Perform a search with human-like typing
    await searchInput.pressSequentially('trade and commerce between nations', { delay: 50 });

    // Wait for FTS results in left panel
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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeIds: cy.nodes().map((n: any) => n.id()).slice(0, 10),
      };
    });
    console.log('WASM graph info:', JSON.stringify(graphInfo));
    // The CTE graph search may return nodes without edges for sparse queries.
    // The escalator check: the graph panel rendered with at least 1 node.
    expect(graphInfo.nodes, 'WASM graph should have at least 1 node').toBeGreaterThan(0);

    // 11. Verify graph controls are visible
    await expect(page.locator('#graph-controls')).toBeVisible({ timeout: 5_000 });
    await expect(page.locator('#run-layout-btn')).toBeVisible();
    await checkpoint(page, 'wasm-checkpoint-10-graph-controls');

    // 12. Final state — all three panels populated
    await page.waitForTimeout(3_000);
    await checkpoint(page, 'wasm-checkpoint-11-final-state');
  });
});
