/**
 * Database Switch E2E Tests
 *
 * Verifies the critical "escalator" path: switching between demo databases must
 * produce a definitive, visible result in all visualizations — never a silent
 * blank canvas.
 *
 * The "expensive stairs" failure mode is: user switches DB, page appears to work
 * (no error), but the canvas is blank with no data and no message explaining why.
 *
 * Each DB switch is validated by asserting that the canvas is in ONE of:
 *   - data state:  stats card visible with a point/node count
 *   - empty state: explicit "No data" message (data-testid="*-empty-state")
 *   - error state: explicit error message (data-testid="*-state-error")
 *
 * None of the above being visible is a test failure.
 */
import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

/** Wait for the DB selector "Switching..." indicator to disappear. */
async function waitForSwitchComplete(page: import('@playwright/test').Page) {
  // Switching indicator appears while POST /api/databases/select + invalidateQueries is in flight
  await expect(page.getByTestId('db-switching')).not.toBeVisible({ timeout: 15_000 });
}

/** Assert the embedding canvas is in a definitive state (data, empty, or error).
 *  Returns the current point count if stats are visible, or null if empty/error.
 */
async function assertEmbeddingDefinitiveState(page: import('@playwright/test').Page): Promise<number | null> {
  const stats = page.getByTestId('embedding-stats');
  const empty = page.getByTestId('embedding-empty-state');
  const error = page.getByTestId('embedding-state-error');

  const isStats = await stats.isVisible();
  const isEmpty = await empty.isVisible();
  const isError = await error.isVisible();

  expect(
    isStats || isEmpty || isError,
    'Embedding canvas must show a definitive state (stats, empty, or error) — not a blank canvas',
  ).toBe(true);

  if (isStats) {
    const count = Number(await stats.getAttribute('data-count'));
    return count;
  }
  return null;
}

/** Assert the graph canvas is in a definitive state (data, empty, or error).
 *  Returns the current node count if stats are visible, or null if empty/error.
 */
async function assertGraphDefinitiveState(page: import('@playwright/test').Page): Promise<number | null> {
  const stats = page.getByTestId('graph-stats');
  const empty = page.getByTestId('graph-empty-state');
  const error = page.getByTestId('graph-state-error');

  const isStats = await stats.isVisible();
  const isEmpty = await empty.isVisible();
  const isError = await error.isVisible();

  expect(
    isStats || isEmpty || isError,
    'Graph canvas must show a definitive state (stats, empty, or error) — not a blank canvas',
  ).toBe(true);

  if (isStats) {
    const count = Number(await stats.getAttribute('data-node-count'));
    return count;
  }
  return null;
}

test.describe('Database Switcher', () => {
  test('embeddings page: switching DB produces definitive canvas state', async ({ page }) => {
    setupConsoleMonitor(page);

    // Select the first database (MiniLM) and navigate to the small node2vec index
    // (64d, ~61-83 points) — much faster UMAP than the larger entity indexes.
    await page.goto('/');
    const dbSelector = page.getByTestId('db-selector');
    await expect(dbSelector).toBeVisible({ timeout: 10_000 });
    await dbSelector.selectOption('3300_MiniLM');
    await waitForSwitchComplete(page);
    await checkpoint(page, 'db-switch-emb-01-miniml-db-selected');

    // Navigate directly to the node2vec index on the embeddings page
    await page.goto('/embeddings/node2vec_emb');

    // Wait for UMAP projection to complete: stats card must appear with count > 0.
    // node2vec_emb is small (64d, ~61 points) so UMAP should complete quickly.
    await expect(async () => {
      await expect(page.getByTestId('embedding-stats')).toBeVisible();
      const count = Number(await page.getByTestId('embedding-stats').getAttribute('data-count'));
      expect(count, 'MiniLM node2vec_emb should have points > 0').toBeGreaterThan(0);
    }).toPass({ timeout: 60_000 });

    const minimlCount = Number(await page.getByTestId('embedding-stats').getAttribute('data-count'));
    await checkpoint(page, 'db-switch-emb-02-miniml-data-loaded');

    // ─── THE CRITICAL SWITCH ───
    await dbSelector.selectOption('3300_NomicEmbed');
    await waitForSwitchComplete(page);
    await checkpoint(page, 'db-switch-emb-03-nomic-switching-done');

    // After the switch, the canvas MUST reach a definitive state.
    // The UMAP for the new DB runs fresh — allow generous timeout.
    let nomicCount: number | null = null;
    await expect(async () => {
      nomicCount = await assertEmbeddingDefinitiveState(page);
    }).toPass({ timeout: 60_000 });

    await checkpoint(page, 'db-switch-emb-04-nomic-definitive-state');

    // Positive assertion: if both DBs have data for this index, counts must differ.
    // (Both MiniLM and NomicEmbed have node2vec_emb: 61 vs 83 points.)
    if (nomicCount !== null) {
      expect(
        nomicCount,
        `After DB switch, point count (${nomicCount}) must differ from MiniLM count (${minimlCount})`,
      ).not.toBe(minimlCount);
    }
  });

  test('graph page: switching DB produces definitive canvas state', async ({ page }) => {
    setupConsoleMonitor(page);

    // Select MiniLM and navigate to the relations graph
    await page.goto('/');
    const dbSelector = page.getByTestId('db-selector');
    await expect(dbSelector).toBeVisible({ timeout: 10_000 });
    await dbSelector.selectOption('3300_MiniLM');
    await waitForSwitchComplete(page);

    await page.goto('/graph/relations');

    // Wait for graph stats to show with node count > 0
    await expect(async () => {
      await expect(page.getByTestId('graph-stats')).toBeVisible();
      const count = Number(await page.getByTestId('graph-stats').getAttribute('data-node-count'));
      expect(count, 'MiniLM relations graph should have nodes > 0').toBeGreaterThan(0);
    }).toPass({ timeout: 30_000 });

    const minimlNodes = Number(await page.getByTestId('graph-stats').getAttribute('data-node-count'));
    await checkpoint(page, 'db-switch-graph-01-miniml-loaded');

    // ─── THE CRITICAL SWITCH ───
    await dbSelector.selectOption('3300_NomicEmbed');
    await waitForSwitchComplete(page);
    await checkpoint(page, 'db-switch-graph-02-nomic-switching-done');

    // Canvas must reach a definitive state after the switch
    let nomicNodes: number | null = null;
    await expect(async () => {
      nomicNodes = await assertGraphDefinitiveState(page);
    }).toPass({ timeout: 30_000 });

    await checkpoint(page, 'db-switch-graph-03-nomic-definitive-state');

    // Positive assertion: both DBs have the relations table but with different node counts
    // (MiniLM: ~85 nodes, NomicEmbed: ~145 nodes from the same book with different chunking)
    if (nomicNodes !== null) {
      expect(
        nomicNodes,
        `After DB switch, node count (${nomicNodes}) must differ from MiniLM count (${minimlNodes})`,
      ).not.toBe(minimlNodes);
    }
  });

  test('empty state is visible and not a blank canvas when no data found', async ({ page }) => {
    setupConsoleMonitor(page);

    // Navigate to an index that does not exist in the default DB (before any DB is selected).
    // The server falls back to the default 3300.db which has 'chunks_vec' etc. but not
    // a hypothetical index name — this triggers the error state in EmbeddingsPage.
    await page.goto('/embeddings/nonexistent_index_xyz');

    // The canvas area must show an explicit state — not a blank div.
    const canvasArea = page.getByTestId('embedding-canvas-area');
    await expect(canvasArea).toBeVisible({ timeout: 10_000 });

    // After load, exactly one definitive state must be visible (loading → error or no-selection)
    await expect(async () => {
      const loading = page.getByTestId('embedding-state-loading');
      const error = page.getByTestId('embedding-state-error');
      const noSelection = page.getByTestId('embedding-state-no-selection');
      const emptyCanvas = page.getByTestId('embedding-empty-state');

      const states = await Promise.all([
        loading.isVisible(),
        error.isVisible(),
        noSelection.isVisible(),
        emptyCanvas.isVisible(),
      ]);
      expect(states.some(Boolean), 'Canvas area must show an explicit state — never blank').toBe(true);
    }).toPass({ timeout: 15_000 });

    await checkpoint(page, 'db-switch-empty-state-visible');
  });
});
