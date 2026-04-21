# viz/ — Muninn Visualization Tool

## What This Is

A FastAPI + React app that surfaces the pre-generated demo databases in
`viz/frontend/public/demos/` (produced by `benchmarks/demo_builder/`). The
frontend exposes one page per database; the backend is a thin, typed façade
over the demos manifest plus whatever per-database queries we add over time.

## Current Scope

**Backend** (`server/`):
- `GET /api/health`
- `GET /api/databases` — list manifest entries
- `GET /api/databases/{id}` — one manifest entry
- `GET /api/databases/{id}/tables` — discover which embed/kg tables exist
- `GET /api/databases/{id}/embed/{table_id}` — 3D UMAP points
  (`table_id ∈ {chunks, entities}`)
- `GET /api/databases/{id}/kg/{table_id}?resolution=X&top_n=N` — KG payload
  with nodes + edges + communities (`table_id ∈ {base, er}`).
  Default `top_n=500` — full KG can exceed 6K nodes which makes Cytoscape
  fcose layout unusable; the `total_node_count` / `total_edge_count` fields
  expose the full size for UI banners.

**Frontend** (`frontend/src/`):
- `/` — list all databases from the manifest
- `/:databaseId/` — database detail + links to per-database viz tables
- `/:databaseId/embed/:tableId/` — Deck.GL 3D UMAP scatter (tableId=chunks|entities)
- `/:databaseId/kg/:tableId/` — Cytoscape with compound community parents
  (tableId=base|er)

**Demo assets**: `viz/frontend/public/demos/manifest.json` is the source of
truth. Adding a new DB is a manifest edit, not a code change. Per-database
`*.db` files contain the HNSW shadow tables, UMAP coords, Leiden
communities, and entity resolution clusters.

## Build & Dev

```bash
make -C viz install    # Install Python + npm deps
make -C viz dev        # Start backend (8200) + frontend (5280)
make -C viz test       # Run pytest + vitest (unit + API tests)
make -C viz test-e2e   # Run Playwright permutation + behavioral suite
make -C viz lint       # ruff + eslint + prettier
make -C viz typecheck  # mypy + tsc
make -C viz ci         # lint + typecheck + test + e2e
```

## Architecture

- **Independent UV project** — `viz/.venv/` is separate from the repo root
- **Backend**: `server/` package, run via `python -m server [--port 8200]`
- **Frontend**: Vite + React 19 + TypeScript + Tailwind v4 + react-router-dom
- **Ports**: backend 8200, frontend 5282. Vite proxies `/api/*` → backend.
- **Single API client** (`frontend/src/lib/api-client.ts`) is the only code
  path that talks to the backend. Components and pages never call `fetch`
  directly. All API types live in this file.

## Conventions

- All Python imports at top of file
- Use `pathlib` for file paths
- Use `logging` module, never `print()`
- No mocks in Python tests — use real `TestClient(app)` with tmp-dir fixtures
- Frontend `fetch` is legitimate to stub at the network boundary via
  `vi.stubGlobal('fetch', ...)` — that is trust-boundary isolation, not
  internal-code mocking

## E2E Testing Pattern — Ruthless Permutation Coverage

The E2E suite enumerates every route permutation in the sitemap and runs the
**same checklist** against each one:

1. **Navigate** to the URL
2. **Assert the page mounts** (React root has children, no `Loading...` stuck)
3. **Wait for network idle** (3 s best-effort — if it doesn't settle, that's
   a test bug we want to see)
4. **Assert zero browser console errors** (filtered for known noise:
   `act(`, `favicon`, `[vite]`)
5. **Save three artifacts** per test, paired by slug:
   - `{slug}.png` — full-page screenshot
   - `{slug}.log` — every console level + `pageerror` line (preserved for
     post-mortem even when the test passes)
   - `{slug}.network.json` — chronologically sorted request timeline with
     `start_offset_ms` + `duration_ms` per request (Gantt-ready), plus a
     rolled-up summary with `wall_clock_duration_ms`, `total_requests`,
     `api_requests`, `api_duration_ms`, and the 5 slowest API calls

All three artifacts land in `viz/frontend/e2e-screenshots/` — one directory,
lexicographically sorted by slug.

### Slug naming convention

```
S{section_id}_{SECTION_SLUG}[-D{db_id}_{DB_SLUG}][-T{table_id}_{TABLE_SLUG}].png
```

- `S{id}` — two-digit zero-padded section ID. Current sections:
  - `00_HOME` (`/`) — no DB, no table
  - `01_DATABASE` (`/:databaseId/`) — DB only
  - `02_EMBED` (`/:databaseId/embed/:tableId/`) — DB + table (chunks/entities)
  - `03_KG` (`/:databaseId/kg/:tableId/`) — DB + table (base/er)
- `D{id}` — two-digit zero-padded database index (or `NA`). Database slugs
  come from `manifest.json` (`id` field, uppercased).
- `T{id}` — two-digit zero-padded table index (or omitted). Table slugs:
  `CHUNKS`/`ENTITIES` for embed, `BASE`/`ER` for kg.

Examples:
- `S00_HOME-DNA.png` — homepage
- `S01_DATABASE-D00_3300_MINILM.png` — `/3300_MiniLM/`
- `S02_EMBED-D00_3300_MINILM-T00_CHUNKS.png` — `/3300_MiniLM/embed/chunks/`
- `S03_KG-D03_39653_NOMICEMBED-T01_ER.png` — `/39653_NomicEmbed/kg/er/`

Zero-padding + uppercase keeps `ls e2e-screenshots/` in the order a human
would expect: all S00 first, then all S01 grouped by DB, then S02 grouped
by DB + embed table, then S03 grouped by DB + kg table.

When a new axis appears (filter, tab), extend the slug schema alphabetically:
`S-D-T-F-V`. Never reorder existing axes — that breaks screenshot diff
comparisons across commits.

### Canvas-ready testids for async viz pages

Deck.GL and Cytoscape mount asynchronously. E2E waits on dedicated testids
that expose the data population as attributes so tests can assert both
"rendered" and "rendered with expected data":

| Page | `data-testid` | Attributes |
|------|---------------|-----------|
| EmbedPage | `embed-canvas-ready` | `data-point-count` |
| KGPage | `kg-canvas-ready` | `data-node-count`, `data-edge-count`, `data-community-count` |

Ready semantics: "Cytoscape mounted + initial grid layout landed" rather
than "fcose layout converged". On the full 5K-6K-node KG, fcose can take
minutes — running it as a background refinement after the initial paint
keeps E2E fast and the UI interactive during refinement.

### Sitemap as data

The sitemap lives in `e2e/helpers/sitemap.ts` as a plain data structure.
Adding a new route = adding one entry to `SECTIONS`. Expanding coverage on
an existing route = tweaking its `databases` slice (or other axes when they
appear). The permutation generator in `permutations.spec.ts` reads the
sitemap at module load and emits one `test(...)` per cross-product entry.

When a new axis appears (filter, tab, view mode), extend the slug schema
alphabetically: `S{id}-D{id}-F{id}-V{id}`. Never reorder existing axes —
that breaks screenshot diff comparisons across commits.

### Collector pattern

`e2e/helpers/collect.ts` exports `collectTestIO(page)` which returns:

- `writeLog(slug)` — flushes `{slug}.log` and `{slug}.network.json`
- `assertNoErrors()` — filters known-noise, asserts remaining errors is 0

The collector is a closure over per-test arrays, wired to Playwright page
events at test start. No per-test cleanup needed — the `Page` is disposed
after each test.

### Per-test outline

```typescript
test(testLabel, async ({ page }) => {
  const io = collectTestIO(page)
  const slug = screenshotSlug(section, database)

  await page.goto(section.pathFor(database))
  await waitForPageLoad(page)

  await page.screenshot({ path: `e2e-screenshots/${slug}.png`, fullPage: true })
  io.writeLog(slug)
  io.assertNoErrors()
})
```

The three artifacts are always written, even on failure paths, because the
`.log` and `.network.json` are the post-mortem evidence for why a test
failed. Don't gate them behind success.

### Behavioral tests live separately

`permutations.spec.ts` covers the "every page loads cleanly" axis.
`navigation.spec.ts` covers click-through journeys (home → database →
back → home). Keep them in separate files so a permutation failure is
visually distinct from a nav regression.
