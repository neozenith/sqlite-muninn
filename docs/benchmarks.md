# Benchmarks

Performance measurements of muninn's vector-search and graph-traversal subsystems against alternatives — real embedding models, synthetic graphs, production-relevant scales. The benchmark harness lives in `benchmarks/` and drives JSONL result files + Plotly charts.

This page is a reproduction guide; detailed results, charts, and analysis live on the per-suite pages — [VSS](benchmarks/vss.md), [Graph](benchmarks/graph.md), [Graph VT](benchmarks/graph_vt.md), [Embed](benchmarks/embed.md), [KG](benchmarks/kg.md).

## Quick start

```bash
# Vector: download models + cache embeddings (~5 min)
make -C benchmarks models-prep

# Vector: run all engines × datasets (~30 min)
make -C benchmarks models

# Graph: small graphs (~5 min)
make -C benchmarks graph-small

# Aggregate JSONL results into charts and docs
make -C benchmarks analyze
```

All benchmark targets run from the project root via `make -C benchmarks ...` — do not `cd benchmarks/` first.

## What's benchmarked

| Suite | Compared engines | Scales |
|-------|------------------|--------|
| Vector search | muninn, sqlite-vec, sqlite-vss, sqlite-vector-fullscan | 1k – 250k vectors, dims 384 / 768 / 1024 |
| Graph traversal | muninn TVFs, recursive CTEs, GraphQLite | 1k – 100k nodes |
| Graph centrality | muninn TVFs, NetworkX reference | 1k – 10k nodes |
| Community detection | muninn Leiden, NetworkX Louvain | 1k – 10k nodes |

## Results

Published result pages:

- **[Vector Search](benchmarks/vss.md)** — HNSW vs brute-force across 3 embedding models, 2 datasets, 4 SQLite extensions
- **[Graph](benchmarks/graph.md)** — BFS, DFS, shortest path, components, PageRank, centrality, Leiden, across engines

Check which scenarios have been run:

```bash
make -C benchmarks vss-manifest     # vector search completeness
make -C benchmarks graph-manifest   # graph completeness
```

## Methodology

- **Storage**: disk-persisted SQLite databases (shared-cache off)
- **Ground truth (vector)**: Python brute-force KNN for N ≤ 50k; `sqlite-vector-fullscan` above that
- **Ground truth (graph)**: Python reference implementations (BFS, Dijkstra, union-find, NetworkX)
- **Query count**: 100 random queries per vector config, 50 random start nodes per graph config
- **Vector sources**: pre-computed from AG News and "Wealth of Nations" text, cached as `.npy` files
- **Saturation**: 10,000 sampled pairwise distances per configuration
- **HNSW defaults**: `M=16`, `ef_construction=200`, `ef_search=64`
- **Results**: JSONL at `benchmarks/results/<suite>.jsonl` for cross-run aggregation

## Custom runs

All benchmark drivers accept the same flag family. Examples:

```bash
# Vector: specific model, engine, dataset
uv run python benchmarks/scripts/benchmark_vss.py \
  --source model:all-MiniLM-L6-v2 --sizes 1000,5000 \
  --engine muninn --dataset ag_news

# Vector: random synthetic vectors
uv run python benchmarks/scripts/benchmark_vss.py \
  --source random --dim 384 --sizes 1000,5000

# Graph: specific topology
uv run python benchmarks/scripts/benchmark_graph.py \
  --nodes 1000 --avg-degree 10 --engine muninn

# Graph: scale-free (Barabási–Albert)
uv run python benchmarks/scripts/benchmark_graph.py \
  --graph-model barabasi_albert --nodes 5000 --avg-degree 5
```

### Profile presets

```bash
make -C benchmarks models           # Real embeddings, 3 models × 2 datasets, N ≤ 250k
make -C benchmarks graph-small      # Erdős–Rényi, N ≤ 1k
make -C benchmarks graph-medium     # Erdős–Rényi, N ≤ 10k
make -C benchmarks graph-large      # Erdős–Rényi, N ≤ 100k
make -C benchmarks graph-scale-free # Barabási–Albert, N ≤ 50k
```

See `make -C benchmarks help` for every target.

## Notes

- HNSW doesn't win at small N — at N=500, dim=128, HNSW search (~0.19 ms) is *slower* than quantize-scan (~0.03 ms). The interesting question is where the crossover sits, which depends on dim, metric, and `ef_search`. The result pages trace this crossover for each model.
- Benchmarks run against `make all` (no ASan). An ASan build cannot be loaded into a non-ASan Python interpreter.
- Cached model vectors live at `benchmarks/vectors/` as `.npy`. Safe to delete — they re-compute from source text.

## See also

- [VSS](benchmarks/vss.md), [Graph](benchmarks/graph.md), [Graph VT](benchmarks/graph_vt.md), [Embed](benchmarks/embed.md), [KG](benchmarks/kg.md) — detailed result pages and charts
- [Architecture — HNSW storage](architecture.md#hnsw-index-storage) — how the benchmarked index is built
- [API Reference — `hnsw_index`](api.md#hnsw_index) — the parameters being tuned
