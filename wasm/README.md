# wasm/ — WASM E2E Testing & Demo

This directory is the **end-to-end testing space** for the muninn SQLite extension compiled to WebAssembly. It contains a self-contained web demo that exercises the full muninn pipeline in-browser, plus Playwright E2E tests that validate the entire flow.

## Purpose

The WASM demo serves two roles:

1. **E2E validation** — Proves that the muninn extension (HNSW search, graph traversal, centrality, community detection) works correctly when compiled to WASM via Emscripten.
2. **Interactive demo** — A zero-backend web app that loads a pre-built knowledge graph database and lets users search it with natural language queries, visualizing results as 3D embeddings and a graph network.

## Directory Structure

```
wasm/
├── index.html              # Demo page (Tailwind CSS dark theme)
├── script.js               # App logic: WASM init → search → visualization
├── styles.css              # Custom CSS (animations, result cards)
├── Makefile                # Build, serve, test, and CI targets
├── assets/
│   └── 3300.db             # Pre-built SQLite DB (Wealth of Nations KG, ~13MB)
├── scripts/
│   └── encode_umap_embeddings.py  # Pre-calculate 2D/3D UMAP coords for visualization
├── e2e/
│   ├── demo.spec.ts        # Playwright E2E test suite (10 checkpoints)
│   └── helpers/
│       └── checkpoint.ts   # Screenshot + console error assertion helper
├── playwright.config.ts    # Playwright config (video recording, single worker)
├── package.json            # Dev dependencies (Playwright, Prettier)
├── build/                  # Generated: WASM artifacts + SQLite amalgamation
├── screenshots/            # Generated: E2E checkpoint screenshots
└── test-results/           # Generated: Playwright output + video recordings
```

## Quick Start

```bash
# Build the WASM module (requires Emscripten SDK)
make -C wasm build

# Start the dev server on port 8300
make -C wasm dev

# Run E2E tests (builds first, installs Playwright)
make -C wasm test

# Full CI pipeline (format + test + video conversion)
make -C wasm ci
```

## How It Works

The demo exercises five muninn subsystems in a single page:

1. **WASM Init** — Compiles SQLite + muninn into a single `.wasm` binary via Emscripten. The 13MB database is fetched and loaded into Emscripten's virtual filesystem.

2. **Embedding Generation** — Uses [Transformers.js](https://huggingface.co/docs/transformers.js) to run `all-MiniLM-L6-v2` (384-dim) entirely in-browser. First load downloads ~30MB of model weights.

3. **HNSW Vector Search** — Packs the query embedding as a `float32` blob and runs `SELECT rowid, distance FROM chunks_vec WHERE vector MATCH ? AND k = 20` via the muninn HNSW virtual table.

4. **Graph Traversal** — Finds entities mentioned in matching chunks, then discovers relationships between them using SQL queries against the `entities` and `relations` tables.

5. **Visualization** — Renders results in three coordinated views:
   - **Text cards** with similarity scores and gradient bars
   - **Deck.GL 3D point cloud** using pre-calculated UMAP coordinates (stored in `chunks_vec_umap`)
   - **Cytoscape.js graph** with entity-type coloring and COSE layout

## E2E Test Checkpoints

The Playwright test (`e2e/demo.spec.ts`) validates 10 sequential checkpoints, each taking a screenshot and asserting zero unexpected console errors:

| # | Checkpoint | What It Validates |
|---|-----------|-------------------|
| 01 | page-loaded | HTML renders, title visible |
| 02 | wasm-ready | WASM module initialized |
| 03 | database-loaded | Schema discovered, footer shows counts |
| 04 | viz-libraries-loaded | Deck.GL + Cytoscape status resolved |
| 05 | transformers-ready | ML model downloaded and loaded |
| 06 | search-enabled | Search input becomes interactive |
| 07 | search-results | Results panel appears after query |
| 08 | results-verified | At least one result card present |
| 09 | graph-populated | Cytoscape shows graph nodes |
| 10 | embeddings-visualized | Deck.GL shows embedding points |

## Pre-calculating UMAP Coordinates

3D visualization coordinates are pre-computed (not generated in-browser) to avoid expensive dimensionality reduction at runtime:

```bash
python wasm/scripts/encode_umap_embeddings.py --db wasm/assets/3300.db --scale 30
```

This creates a `chunks_vec_umap` table with `x2d, y2d, x3d, y3d, z3d` columns, which the demo queries at search time for instant 3D positioning.

## Build Prerequisites

- **Emscripten SDK** (`emcc`) for WASM compilation
- **Node.js** for Playwright tests and Prettier formatting
- **Python 3** for the dev server (`python3 -m http.server`)
- **ffmpeg** (optional) for converting E2E video recordings to `.mp4`/`.gif`
