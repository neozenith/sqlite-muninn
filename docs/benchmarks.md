# Benchmarks

Performance comparison of **vector search engines** and **graph traversal engines** in SQLite, using real embedding models and synthetic graphs at production-relevant scales.

## Quick Start

```bash
# Vector: pre-download models + cache embeddings (~5 min)
make -C benchmarks models-prep

# Vector: run model benchmarks across all engines and datasets (~30 min)
make -C benchmarks models

# Graph: small graph benchmarks (~5 min)
make -C benchmarks graph-small

# Analyze all results and generate charts + docs
make -C benchmarks analyze
```

## Results

- **[Vector Search](benchmarks/vss.md)** — HNSW vs brute-force across 3 embedding models, 2 datasets, and 4 SQLite extensions
- **[Graph](benchmarks/graph.md)** — BFS, DFS, shortest path, components, PageRank, degree/betweenness/closeness centrality, and Leiden community detection across muninn TVFs, recursive CTEs, and GraphQLite

## Completeness

Check which benchmark scenarios have been run:

```bash
make -C benchmarks vss-manifest     # VSS completeness report
make -C benchmarks graph-manifest   # Graph completeness report
```

## Methodology

- **Storage**: Disk-persisted SQLite databases (default for model benchmarks)
- **Ground truth (vector)**: Python brute-force KNN for N <= 50K, `sqlite-vector-fullscan` for larger datasets
- **Ground truth (graph)**: Python reference implementations (BFS, Dijkstra, Union-Find, etc.)
- **Vector queries**: 100 random queries per configuration
- **Graph queries**: 50 random start nodes from largest connected component
- **Embeddings**: Pre-computed from AG News / Wealth of Nations, cached as `.npy` files
- **Saturation**: 10K sampled pairwise distances per configuration
- **HNSW params**: M=16, ef_construction=200, ef_search=64
- **Results**: Stored as JSONL for cross-run aggregation

## Running Custom Benchmarks

```bash
# Vector: specific model, engine, and dataset
.venv/bin/python benchmarks/scripts/benchmark_vss.py --source model:all-MiniLM-L6-v2 --sizes 1000,5000 --engine muninn --dataset ag_news

# Vector: random vectors
.venv/bin/python benchmarks/scripts/benchmark_vss.py --source random --dim 384 --sizes 1000,5000

# Graph: specific topology
.venv/bin/python benchmarks/scripts/benchmark_graph.py --nodes 1000 --avg-degree 10 --engine muninn

# Graph: scale-free
.venv/bin/python benchmarks/scripts/benchmark_graph.py --graph-model barabasi_albert --nodes 5000 --avg-degree 5

# All vector profiles
make -C benchmarks saturation  # Random vectors, 8 dims, N=50K
make -C benchmarks models      # Real embeddings, 3 models x 2 datasets, N<=250K

# All graph profiles
make -C benchmarks graph-small       # Erdos-Renyi, N<=1K
make -C benchmarks graph-medium      # Erdos-Renyi, N<=10K
make -C benchmarks graph-large       # Erdos-Renyi, N<=100K
make -C benchmarks graph-scale-free  # Barabasi-Albert, N<=50K
```
