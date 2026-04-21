# viz/ — Muninn Visualization Tool

## What This Is

A FastAPI + React app that surfaces the pre-generated demo databases in
`viz/frontend/public/demos/` (produced by `benchmarks/demo_builder/`). The
frontend exposes one page per database; the backend is a thin, typed façade
over the demos manifest plus whatever per-database queries we add over time.

## Current Scope

- **Backend** (`server/`): `GET /api/health`, `GET /api/databases`, `GET /api/databases/{id}`
- **Frontend** (`frontend/src/`):
  - `/` — lists all databases from the manifest
  - `/:databaseId/` — database detail page (single-database context)
- **Demo assets**: `viz/frontend/public/demos/manifest.json` is the source of
  truth. Adding a new DB is a manifest edit, not a code change.

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
S{section_id}_{SECTION_SLUG}-D{database_id}_{DATABASE_SLUG}.png
```

- `S{id}` — two-digit zero-padded section ID. `00_HOME` for `/`,
  `01_DATABASE` for `/:databaseId/`.
- `D{id}` — two-digit zero-padded database ID, or `NA` for routes that don't
  take a database. Database slugs come from `manifest.json` (`id` field,
  uppercased).

Examples:

- `S00_HOME-DNA.png` — homepage, no database in URL
- `S01_DATABASE-D00_3300_MINILM.png` — `/3300_MiniLM/`
- `S01_DATABASE-D01_3300_NOMICEMBED.png` — `/3300_NomicEmbed/`

Zero-padding + uppercase keeps `ls e2e-screenshots/` in the order a human
would expect: all S00 screenshots together, then all S01 grouped by database.

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
