# CLAUDE.md -- Benchmark Harness

Guidance for Claude Code when working within `benchmarks/harness/`.

## What This Is

A Python package providing a unified CLI (`python -m benchmarks.harness.cli`) for running, tracking, and analysing benchmarks against the muninn SQLite extension. Uses the Treatment ABC pattern where each benchmark permutation is a subclass that the harness executes in a setup-run-teardown lifecycle.

## Build & Test Commands

```bash
make -C benchmarks/harness test       # Run all tests (builds muninn extension first)
make -C benchmarks/harness test-quick # Tests excluding slow markers (no extension build)
make -C benchmarks/harness lint       # ruff check + format check
make -C benchmarks/harness fix        # Auto-fix lint + format
make -C benchmarks/harness typecheck  # mypy
make -C benchmarks/harness coverage   # Tests with coverage report
make -C benchmarks/harness ci         # Full CI: lint + typecheck + test
```

Alternative without Make:

```bash
uv run -m pytest benchmarks/harness/tests/ -v --no-cov
```

Test location: `benchmarks/harness/tests/`

## Key Conventions

### Treatment ABC Properties

Every Treatment subclass must implement these abstract properties and methods:

- `category` (str) -- e.g., `"vss"`, `"graph"`, `"adjacency"`, `"kg-extract"`
- `permutation_id` (str) -- unique slug used as folder name under `results/` and as `--id` CLI value. Must be a valid path component (no slashes, no spaces).
- `label` (str) -- human-readable description for manifest display
- `sort_key` (tuple) -- for ordering within a category. Primary scaling dimension first, then secondary tie-breakers. Comparable only within the same category (different categories have different tuple shapes).
- `setup(conn, db_path)` -- returns dict of setup metrics
- `run(conn)` -- returns dict of treatment-specific metrics
- `teardown(conn)` -- cleanup, no return value
- `params_dict()` (optional override) -- flat dict of parameters merged into JSONL output

### Import Rules

All imports must be at the top of the file per project Python rules, with one exception: `cli.py` uses intentional deferred imports inside command handler functions (`_cmd_prep`, `_cmd_manifest`, `_cmd_benchmark`, `_cmd_analyse`) for lazy loading of heavy dependencies. The registry also uses deferred imports inside `_*_permutations()` functions for the same reason. These are the only sanctioned exceptions.

### Extension Loading

Use `load_muninn(conn)` from `common.py` to load the muninn extension. This calls `conn.load_extension(MUNINN_PATH)` where `MUNINN_PATH` points to `build/muninn` relative to the project root (not the repo root's `muninn.dylib` -- the build directory).

```python
from benchmarks.harness.common import load_muninn

load_muninn(conn)  # Enables load_extension and loads build/muninn
```

### Vector Packing

Vectors are float32 BLOBs. Use `pack_vector()` from `common.py`:

```python
from benchmarks.harness.common import pack_vector

blob = pack_vector([1.0, 2.0, 3.0])       # From list
blob = pack_vector(numpy_array)             # From numpy (fast path via tobytes)
```

### Graph Generation

Two generators in `common.py`:

- `generate_erdos_renyi(n_nodes, avg_degree, weighted=False, seed=42)` -- returns `(edges, adjacency_dict)`
- `generate_barabasi_albert(n_nodes, m, weighted=False, seed=42)` -- returns `(edges, adjacency_dict)`

Both return edges as `(src, dst, weight)` tuples and adjacency as `{node: [(neighbor, weight), ...]}`.

## Common Patterns

### Adding Permutations to the Registry

In `registry.py`, each category has a `_<category>_permutations()` function that returns a list of Treatment instances. These use deferred imports to avoid loading heavy treatment modules until needed. Add new generators to `all_permutations()`.

### ChartSpec for Analysis

Charts are defined declaratively via `ChartSpec` dataclasses in `analysis/charts_*.py` files. The aggregator loads JSONL, filters, groups by `repeat_fields`, aggregates, then builds `ChartSeries` for the renderer.

## Gotchas

- **sort_key is comparable within category only.** Different treatment categories return tuples of different shapes and types. Never sort a mixed list by sort_key without grouping by category first.

- **permutation_id must be a valid path component.** It is used as a directory name under `results/`. No slashes, no spaces, no special characters. Convention: `{category}_{variant_slug}`.

- **MUNINN_PATH points to `build/muninn`**, not the project root's `muninn.dylib`. The extension is expected at `<project_root>/build/muninn.<ext>`. This is set in `common.py` via `PROJECT_ROOT / "build" / "muninn"`.

- **OUTPUT_DIR_PREFIX migration switch.** `common.py` defines `OUTPUT_DIR_PREFIX = Path("refactored_outputs")`. During the migration from legacy scripts, all output goes under `benchmarks/refactored_outputs/`. Change to `Path("")` when migration is complete.

- **cli.py deferred imports are intentional.** The `_cmd_*` handler functions import inside the function body to avoid loading all treatment modules and heavy dependencies (numpy, sentence-transformers) when only running a simple subcommand like `manifest`.

- **Harness catches muninn load failure gracefully.** If the extension cannot be loaded, it logs a warning and continues -- some treatments handle their own extensions (e.g., vectorlite, sqlite-vec).

## Path Constants (from common.py)

| Constant | Resolves to |
|---|---|
| `BENCHMARKS_ROOT` | `benchmarks/` |
| `OUTPUT_ROOT` | `benchmarks/refactored_outputs/` (during migration) |
| `RESULTS_DIR` | `benchmarks/refactored_outputs/results/` |
| `CHARTS_DIR` | `benchmarks/refactored_outputs/charts/` |
| `VECTORS_DIR` | `benchmarks/refactored_outputs/vectors/` |
| `TEXTS_DIR` | `benchmarks/refactored_outputs/texts/` |
| `KG_DIR` | `benchmarks/refactored_outputs/kg/` |
| `MUNINN_PATH` | `build/muninn` (no extension suffix -- SQLite adds it) |
| `DOCS_BENCHMARKS_DIR` | `docs/benchmarks/refactored_output/` |

## Benchmark Defaults (from common.py)

| Constant | Value | Purpose |
|---|---|---|
| `K` | 10 | Top-K for search queries |
| `N_QUERIES` | 100 | Number of search queries per run |
| `HNSW_M` | 16 | HNSW max connections per layer |
| `HNSW_EF_CONSTRUCTION` | 200 | HNSW build-time beam width |
| `HNSW_EF_SEARCH` | 64 | HNSW search-time beam width |

## Makefile Targets

Invoke via `make -C benchmarks/harness <target>`:

| Target | Description |
|---|---|
| `help` | Show all targets |
| `extension` | Build the muninn C extension (delegates to root Makefile) |
| `format` | Format with ruff + isort |
| `lint` | Check with ruff (no auto-fix) |
| `fix` | Auto-fix lint + format |
| `typecheck` | mypy type checking |
| `test` | Run all tests (builds extension first) |
| `test-quick` | Fast tests only (skip slow markers) |
| `coverage` | Tests with HTML coverage report |
| `clean` | Remove `__pycache__`, `.pyc`, coverage artifacts |
| `ci` | Full pipeline: lint + typecheck + test |
