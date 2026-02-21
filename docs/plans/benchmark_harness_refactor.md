# Spec: Refactor Benchmark Suite into Unified CLI with Treatment Pattern

> **Status**: In Progress
> **Created**: 2026-02-20

## Context

The benchmark suite grew organically over ~2 weeks across 15+ sessions into 7,680 LOC across 10 Python scripts in `benchmarks/scripts/`. Each domain (VSS, graph, adjacency, KG) was built independently with significant code duplication.

**Goal**: Consolidate into **1 entry point CLI** with 4 subcommands (`prep`, `manifest`, `benchmark`, `analyse`) using a **Treatment** pattern for extensible benchmark execution.

**Rollout strategy**: Parallel — new code lives in `benchmarks/harness/`, outputs go to `benchmarks/refactored_outputs/`. Legacy scripts stay untouched.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Permutation ID | Human-readable slug | Self-documenting filesystem |
| Code location | `benchmarks/harness/` | Parallel to legacy `benchmarks/scripts/` |
| Output location | `benchmarks/refactored_outputs/` | Swappable to `benchmarks/` later |
| Rollout | Parallel + self-verify | Run both old and new, compare fidelity |

## Architecture

### CLI Subcommands

```
python -m benchmarks.harness.cli prep [vectors|texts|kg-chunks|er-datasets|all]
python -m benchmarks.harness.cli manifest [--missing|--done|--category|--commands]
python -m benchmarks.harness.cli benchmark --id {permutation_id}
python -m benchmarks.harness.cli analyse [--category|--render-docs]
```

### Treatment Pattern

Each benchmark permutation is a `Treatment` subclass with:
- `category` — e.g., 'vss', 'graph', 'adjacency', 'kg-extract'
- `permutation_id` — human-readable slug used as folder name
- `setup(conn, db_path)` → create tables, load data
- `run(conn)` → execute benchmark, return metrics
- `teardown(conn)` → clean up

### Registry

Enumerates all permutations across all treatment categories. Used by `manifest` and `benchmark` subcommands.

### Harness

Executes a single Treatment: creates DB, times setup/run/teardown, collects common metrics (RSS, db size, platform), writes JSONL.

## Treatment Categories

1. **VSS**: muninn-hnsw, sqlite-vector-quantize, sqlite-vector-fullscan, vectorlite-hnsw, sqlite-vec-brute
2. **Graph Traversal**: BFS, DFS, shortest_path, components, pagerank
3. **Graph Centrality**: degree, betweenness, closeness
4. **Graph Community**: Leiden
5. **Adjacency**: TVF, CSR, full_rebuild, incremental, blocked
6. **KG Extract**: NER model adapter pattern (GLiNER, NuNerZero, GNER-T5, spaCy, FTS5)
7. **KG Resolve**: Entity resolution with Pairwise/B-Cubed F1
8. **KG GraphRAG**: VSS+Graph retrieval quality
9. **Node2Vec**: Training time, p/q sweep, dimensionality trade-offs

## Implementation Phases

| Phase | What | Test Files |
|-------|------|-----------|
| 0 | Spec + introspection artifacts | test_phase0.py |
| 1 | common.py + treatments/base.py | test_common.py, test_treatment_base.py |
| 2 | harness.py + registry.py + vss.py | test_registry.py, test_harness.py, test_vss_treatment.py |
| 3 | Remaining treatments | test_treatments.py |
| 4 | Prep subcommand | test_prep.py |
| 5 | Manifest subcommand | test_cli_manifest.py |
| 6 | Benchmark subcommand | test_cli_benchmark.py |
| 7 | Analysis subcommand | test_analysis.py, test_cli_analyse.py |
| 8 | mkdocs.yml + docs | test_docs.py |
| 9 | Makefile.refactored | test_makefile.py |

## Companion Documents

- `docs/plans/benchmarks_refactor/work_events.json` — Historical work events
- `docs/plans/benchmarks_refactor/original_user_requirement_events.json` — Original user prompts
- `docs/plans/benchmarks_refactor/script_requirements_todate.md` — Distilled requirements
