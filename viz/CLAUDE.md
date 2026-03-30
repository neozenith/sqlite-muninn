# viz/ — Muninn Visualization Tool

## What This Is

A self-contained FastAPI + React visualization tool for the muninn SQLite extension.
Three modes: Embeddings Explorer (Deck.GL), Graph Explorer (Cytoscape.js), KG Query (3-panel search).

Two parallel implementations (escalator functions):
- **Python Server-Backed**: `http://localhost:5280/kg/query` — server-side embedding + FTS + VSS + graph
- **WASM**: `http://localhost:5280/public/wasm/index.html` — all in-browser via Transformers.js + WASM SQLite

## Build & Dev

```bash
make -C viz install    # Install Python + npm deps
make -C viz dev        # Start backend (8200) + frontend (5280)
make -C viz test       # Run all tests
make -C viz test-api   # Python API tests only (>90% coverage)
make -C viz test-frontend  # TypeScript tests only (>90% coverage)
make -C viz test-e2e   # Playwright E2E tests
make -C viz ci         # Full CI: format + lint + typecheck + test + e2e
```

## Routes

| Route | Behavior |
|-------|----------|
| `/` | Redirect → `/kg/query/` |
| `/kg/` | Redirect → `/kg/query/` |
| `/kg/query/` | KG Query page (3-panel: FTS, Embedding 3D, Knowledge Graph) |
| `/embeddings/` | Redirect → `/embeddings/chunks_vec/` |
| `/embeddings/:dataset` | Embeddings Explorer for a specific HNSW index |
| `/graph/` | Redirect → `/graph/edges/` |
| `/graph/:dataset` | Graph Explorer for a specific edge table |

## Architecture

- **Independent UV project** — `viz/.venv/` is separate from root `.venv/`
- **Backend:** `server/` package, run via `python -m server`
- **Frontend:** Vite + React + TypeScript + Tailwind CSS v4 + shadcn/ui
- **UMAP:** Pre-computed in `*_umap` tables by demo_builder — no runtime UMAP needed

### Code Organization (Service → Hook → Component)

All logic in `.ts` files. `.tsx` files are thin presentation shells.

```
lib/services/*.ts    → API calls, data transforms (pure functions, no React)
lib/transforms/*.ts  → Data transformation (pure functions)
hooks/*.ts           → Custom hooks combining services + state
components/*.tsx     → Thin JSX wrappers that destructure hooks
```

### Coverage Targets

| Layer | Target | Strategy |
|-------|--------|----------|
| Python `server/` | >90% | pytest + httpx.TestClient |
| `lib/services/*.ts` | >90% | vitest, mocked fetch |
| `lib/transforms/*.ts` | >90% | vitest, pure functions |
| `hooks/*.ts` | >90% | renderHook tests |
| `components/*.tsx` | minimal | E2E via Playwright |

### Ports

| Mode | Backend | Frontend |
|------|---------|----------|
| Human (`make dev`) | 8200 | 5280 |
| Agentic (`make dev-agentic`) | 8201 | 5281 |

## Key Conventions

- All Python imports at top of file (project rule)
- Use `pathlib` for file paths
- Use `logging` module, never `print()`
- SQL identifier validation on all user-provided table/column names
- TanStack Query for server state, Zustand for client state
