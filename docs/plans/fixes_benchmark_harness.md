# Fixes & Improvements: Benchmark Harness CLI

> Collected from hands-on testing of `benchmarks.harness.cli` after Phases 0-9 completed.
> Each item is a self-contained improvement. Implement in any order.

---

## 1. `manifest --limit N` — Return Next N Cheapest Benchmarks

**Problem:** When running benchmarks incrementally, there's no way to ask "give me the next cheapest unrun benchmark." You must pipe all missing commands and manually pick.

**Current behaviour:**
```bash
uv run -m benchmarks.harness.cli manifest --missing --commands
# Emits ALL 403 missing permutations
```

**Desired behaviour:**
```bash
uv run -m benchmarks.harness.cli manifest --missing --limit 1 --commands
# Emits only the 1 cheapest missing benchmark (by sort_key)
uv run -m benchmarks.harness.cli manifest --missing --limit 5
# Shows the 5 cheapest missing benchmarks in table form
```

**Implementation:**
- Add `--limit N` argument to the manifest subparser (type=int, default=None)
- After sorting within each category, flatten the sorted groups and apply `[:limit]` across the flattened list
- When `--limit` is set, sort globally by sort_key (cross-category). This requires a universal sort_key — e.g., prefix with estimated cost or use category priority ordering
- **Alternative (simpler):** Keep within-category sorting but apply limit per category. Discuss trade-offs

**Files to modify:**
- `benchmarks/harness/cli.py` — add `--limit` arg, apply truncation in `_cmd_manifest()`

---

## 2. `manifest --category` — List Available Categories & Show Summary

**Problem:** `--category` requires knowing valid category names. If you omit the value, argparse errors. If you pass a wrong name, you get an empty manifest with no hint of what's valid.

**Current behaviour:**
```bash
uv run -m benchmarks.harness.cli manifest --category
# Error: expected one argument

uv run -m benchmarks.harness.cli manifest --category typo
# Silent empty output
```

**Desired behaviour:**
```bash
uv run -m benchmarks.harness.cli manifest --category
# Lists available categories with counts:
#   vss         180 permutations (0 done)
#   graph       110 permutations (0 done)
#   centrality   21 permutations (0 done)
#   ...

uv run -m benchmarks.harness.cli manifest --help
# Shows: --category {vss,graph,centrality,community,adjacency,kg-extract,kg-resolve,kg-graphrag,node2vec}
```

**Implementation:**
- Change `--category` to use `nargs='?'` with `const=None` so omitting the value triggers category listing
- Enumerate categories from the registry: `sorted(set(p.category for p in all_permutations()))`
- Populate `choices` on the `--category` argument dynamically (or list them in the help string)
- When `--category` is given without a value, print category summary table and exit

**Files to modify:**
- `benchmarks/harness/cli.py` — modify `--category` arg and add category listing logic
- `benchmarks/harness/registry.py` — add `available_categories() -> list[str]` helper

---

## 3. `prep` Sub-Subcommands with `--status` and `--force`

**Problem:** `prep` is a flat positional arg (`vectors`, `texts`, `kg-chunks`, `er-datasets`, `all`). There's no way to check prep status or force re-creation. Each prep target has different options but they all share the same flat argument namespace.

**Current behaviour:**
```bash
uv run -m benchmarks.harness.cli prep vectors
# Downloads/creates .npy files. No status check. No force flag.
# If files exist, behaviour depends on the prep function's internals.
```

**Desired behaviour:**
```bash
uv run -m benchmarks.harness.cli prep vectors --status
# Shows which .npy files exist, their sizes, and which are missing

uv run -m benchmarks.harness.cli prep vectors --force
# Re-downloads and recreates all .npy files even if they exist

uv run -m benchmarks.harness.cli prep texts --status
# Shows which Gutenberg texts are cached

uv run -m benchmarks.harness.cli prep texts --book-id 3300 --force
# Re-downloads book 3300 even if cached
```

**Implementation:**
- Convert `prep` from a positional arg to nested subcommands:
  ```
  prep vectors [--status] [--force] [--model MODEL] [--dataset DATASET]
  prep texts [--status] [--force] [--book-id ID]
  prep kg-chunks [--status] [--force] [--book-id ID]
  prep er-datasets [--status] [--force] [--dataset DATASET]
  prep all [--status] [--force]
  ```
- Each prep function gains a `status_only: bool` and `force: bool` parameter
- `--status` prints a table of expected outputs, their existence, and file sizes
- Default behaviour (no flags): skip if target exists, create if missing
- `--force`: always recreate regardless of existing files

**Files to modify:**
- `benchmarks/harness/cli.py` — restructure prep subparser to use nested sub-subcommands
- `benchmarks/harness/prep/vectors.py` — add `status_only` and `force` params
- `benchmarks/harness/prep/texts.py` — add `status_only` and `force` params
- `benchmarks/harness/prep/kg_chunks.py` — add `status_only` and `force` params
- `benchmarks/harness/prep/er_datasets.py` — add `status_only` and `force` params

---

## 4. `prep texts` — Self-Documenting Help for Gutenberg Books

**Problem:** How do you discover which books are available? How do you download a random economics book from Gutenberg? The CLI gives no guidance.

**Current behaviour:**
```bash
uv run -m benchmarks.harness.cli prep texts --help
# Shows: --book-id (int). No context about Gutenberg, categories, or discovery.
```

**Desired behaviour:**
```bash
uv run -m benchmarks.harness.cli prep texts --help
# Shows:
#   Download Gutenberg texts for benchmark corpora.
#
#   Options:
#     --book-id ID     Download a specific Gutenberg book by ID
#     --random         Download a random book from the economics category
#     --category CAT   Gutenberg subject category (default: economics)
#     --list           List cached texts and their metadata
#
#   Examples:
#     prep texts                          # Download default corpus (Wealth of Nations)
#     prep texts --random                 # Download a random economics book
#     prep texts --random --category law  # Download a random law book
#     prep texts --book-id 3300           # Download Wealth of Nations specifically
#     prep texts --list                   # Show all cached texts
```

**Implementation:**
- Add `--random` flag that picks a random book from the Gutendex catalog for the given category
- Add `--category` for Gutenberg subject filtering (default: "economics")
- Add `--list` to show cached texts with metadata (title, author, word count, file size)
- Enhance the help text with examples and context
- The existing `kg_gutenberg.py` logic for Gutendex catalog queries should be reused

**Files to modify:**
- `benchmarks/harness/cli.py` — add flags to texts subparser
- `benchmarks/harness/prep/texts.py` — implement random selection, category filtering, list display

---

## 5. `benchmarks/harness/` — README.md and CLAUDE.md

**Problem:** The harness package has no README or CLAUDE.md. New contributors (human or AI) need to understand the architecture, conventions, and how to add new treatments.

### README.md

Should cover:
- **Purpose**: What the harness does (unified CLI for benchmark execution + analysis)
- **Quick start**: How to run a benchmark end-to-end
- **Architecture**: Treatment pattern, registry, harness execution flow
- **Adding a new treatment**: Step-by-step guide with example
- **Directory layout**: What goes where
- **Output format**: JSONL schema, sqlite structure, chart output
- **MermaidJS diagram**: CLI → Registry → Treatment → Harness → JSONL + SQLite flow

### CLAUDE.md

Should cover:
- **Key conventions**: Treatment ABC, sort_key, permutation_id format
- **Import rules**: Top-level imports (per project Python rules)
- **Test location**: `benchmarks/harness/tests/`
- **How to run tests**: `uv run -m pytest benchmarks/harness/tests/ -v`
- **Common patterns**: Extension loading, graph generation, vector packing
- **Gotchas**: sort_key must be comparable within category only, permutation_id must be valid path component

**Files to create:**
- `benchmarks/harness/README.md`
- `benchmarks/harness/CLAUDE.md`

---

## 6. `benchmarks/harness/Makefile` — Python Tooling Targets

**Problem:** No Makefile for the harness package itself. Formatting, linting, type checking, and testing require manual commands. Should mirror the patterns in the root `Makefile` and `viz/Makefile`.

**Desired targets:**

```makefile
.PHONY: format lint typecheck test coverage clean

format:          ## Format Python code
	uvx ruff format benchmarks/harness/ --line-length 120
	uvx isort benchmarks/harness/

lint:            ## Lint Python code
	uvx ruff check benchmarks/harness/ --line-length 120 --statistics

fix:             ## Auto-fix lint issues
	uvx ruff check benchmarks/harness/ --line-length 120 --fix-only
	uvx ruff format benchmarks/harness/ --line-length 120

typecheck:       ## Run type checking
	uvx mypy benchmarks/harness/

test:            ## Run harness tests
	uv run -m pytest benchmarks/harness/tests/ -v

coverage:        ## Run tests with coverage
	uv run -m pytest benchmarks/harness/tests/ -v --cov=benchmarks.harness --cov-report=term-missing

clean:           ## Remove __pycache__ and .pyc files
	find benchmarks/harness/ -type d -name __pycache__ -exec rm -rf {} +
```

**Files to create:**
- `benchmarks/harness/Makefile`

---

## 7. `benchmark --id` — Handle Existing SQLite Files Gracefully

**Problem:** If a benchmark's `db.sqlite` already exists (from a previous run), the harness either appends data to it or fails silently. There's no warning or safe re-run mechanism.

**Current behaviour:**
```bash
uv run -m benchmarks.harness.cli benchmark --id adjacency_csr_xsmall_erdos_renyi
# If db.sqlite exists: silently opens it and appends/overwrites tables
```

**Desired behaviour (default):**
```bash
uv run -m benchmarks.harness.cli benchmark --id adjacency_csr_xsmall_erdos_renyi
# WARNING: db.sqlite already exists for adjacency_csr_xsmall_erdos_renyi
# WARNING: Will delete in 30 seconds. Press Ctrl+C to cancel.
# [30s countdown...]
# Deleting existing results...
# Running: Adjacency: csr / xsmall / erdos_renyi / N=500
```

**Desired behaviour (--force):**
```bash
uv run -m benchmarks.harness.cli benchmark --id adjacency_csr_xsmall_erdos_renyi --force
# Running: Adjacency: csr / xsmall / erdos_renyi / N=500
# (immediate start, no warning or sleep)
```

**Implementation:**
- In `_cmd_benchmark()` or `run_treatment()`, check if `db_path.exists()` before starting
- If exists and `--force` is not set:
  1. Log a WARNING with the file path
  2. Sleep 30 seconds with a visible countdown (e.g., `time.sleep(1)` in a loop with `\r` progress)
  3. Delete the existing `db.sqlite`
  4. Continue with the benchmark
- If exists and `--force` is set: delete immediately, no warning
- Add `--force` flag to the benchmark subparser

**Files to modify:**
- `benchmarks/harness/cli.py` — add `--force` arg to benchmark subparser, pass to `run_treatment()`
- `benchmarks/harness/harness.py` — add `force` param to `run_treatment()`, implement countdown + deletion

---

## 8. BUG: `benchmark --id adjacency_csr_xsmall_erdos_renyi` Fails

**Problem:** Running the simplest benchmark permutation fails at runtime.

**Reproduction:**
```bash
uv run -m benchmarks.harness.cli benchmark --id adjacency_csr_xsmall_erdos_renyi
```

**Investigation needed:**
- Check that the muninn extension loads correctly for the adjacency treatment
- The `AdjacencyTreatment.setup()` calls `load_muninn(conn)` again after the harness already tries to load it — this may cause issues if the extension path resolution fails
- Verify `graph_adjacency` virtual table is registered (this is a CSR caching vtable — confirm it exists in the muninn extension or if it's not implemented yet)
- Check if `graph_degree`, `graph_betweenness`, `graph_closeness`, `graph_leiden` TVFs work with string node IDs (`"n0"`, `"n1"`, etc.)
- The `_algo_sql()` method uses `graph_degree('edges', 'src', 'dst', 'weight')` — verify this matches the actual TVF signature
- The `csr` approach creates `CREATE VIRTUAL TABLE g USING graph_adjacency(...)` — verify `graph_adjacency` module exists

**Likely root cause candidates:**
1. `graph_adjacency` virtual table module may not exist in the current muninn build (it was planned/designed but may not be implemented yet in C)
2. Extension path resolution in `load_muninn()` may fail depending on the working directory
3. The TVF signatures in `_algo_sql()` may not match the actual C implementation

**Files to investigate:**
- `benchmarks/harness/common.py` — `load_muninn()` implementation
- `src/muninn.c` — verify which modules are registered
- `benchmarks/harness/treatments/adjacency.py` — TVF SQL and virtual table creation
- Run `make all && sqlite3 :memory: '.load ./muninn' '.tables'` to verify available modules

---

## Implementation Priority

| # | Item | Effort | Impact | Priority |
|---|------|--------|--------|----------|
| 8 | Fix adjacency benchmark bug | Small | Blocking | **P0** |
| 7 | Handle existing sqlite files | Small | UX | **P1** |
| 1 | `--limit` for manifest | Small | Workflow | **P1** |
| 2 | `--category` improvements | Small | UX | **P2** |
| 3 | `prep` sub-subcommands | Medium | UX | **P2** |
| 4 | `prep texts` help + features | Medium | Discoverability | **P2** |
| 6 | Harness Makefile | Small | Dev workflow | **P2** |
| 5 | README.md + CLAUDE.md | Medium | Documentation | **P3** |
