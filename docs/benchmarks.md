# Benchmarks

Performance comparison of **vec_graph-hnsw** vs **sqlite-vector-quantize** and **sqlite-vector-fullscan** using real embedding models at production-relevant dimensions and data volumes.

Both `sqlite-vector-quantize` (approximate search via Product Quantization) and `sqlite-vector-fullscan` (brute-force exact search) come from the [`sqliteai/sqlite-vector`](https://github.com/nichdiekuh/sqlite-vector) library. They represent two different search strategies from the same extension.

## Quick Start

```bash
# Pre-download models and cache embeddings (~5 min)
make benchmark-models-prep

# Run model benchmarks (~30 min)
make benchmark-models

# Analyze results and generate charts
make benchmark-analyze
```

## Key Finding

`vec_graph-hnsw` delivers **sub-millisecond latency** regardless of dataset size, while `sqlite-vector-quantize` latency grows linearly with N. With real embeddings from production models, HNSW is **14x faster** at 50K vectors (384d-MiniLM) and the gap only widens as data grows.

## Models Tested

Three popular sentence-transformer models covering the common embedding dimension range:

| Model | Dimension | Use Case | Dataset |
|-------|-----------|----------|---------|
| **all-MiniLM-L6-v2** | 384 | Fast, lightweight semantic search | AG News (120K texts) |
| **all-mpnet-base-v2** | 768 | Balanced quality/speed | AG News (120K texts) |
| **BAAI/bge-large-en-v1.5** | 1024 | High-quality retrieval | AG News (120K texts) |

## Search Latency

The core question: **at what dataset size does `vec_graph-hnsw` beat `sqlite-vector-quantize`?**

With real embeddings, HNSW wins as early as N=1,000 for smaller dimensions. By N=5,000, HNSW is faster across all models. By N=50,000, it's an order of magnitude faster.

### 384d-MiniLM

```plotly
--8<-- "benchmarks/charts/tipping_point_MiniLM.json"
```

### 768d-MPNet

```plotly
--8<-- "benchmarks/charts/tipping_point_MPNet.json"
```

### 1024d-BGE-Large

```plotly
--8<-- "benchmarks/charts/tipping_point_BGE-Large.json"
```

### Cross-Model Comparison

How does embedding dimension affect latency scaling? `vec_graph-hnsw` latency barely moves with dimension, while `sqlite-vector-quantize` cost scales linearly with both N and dimension.

```plotly
--8<-- "benchmarks/charts/model_comparison.json"
```

## Recall

Both search methods achieve excellent recall with real embeddings. `vec_graph-hnsw` recall stabilizes around 96% at larger N — the trade-off for sub-millisecond search. `sqlite-vector-quantize` maintains ~99% recall but at much higher latency.

```plotly
--8<-- "benchmarks/charts/recall_models.json"
```

## Insert Throughput

HNSW inserts are slower due to graph construction overhead (building neighbor connections at each layer). `sqlite-vector` batch insert is ~500-1000x faster for bulk loading. This is the expected trade-off: HNSW pays at insert time to win at search time.

```plotly
--8<-- "benchmarks/charts/insert_throughput_models.json"
```

## Storage Overhead

On-disk, `vec_graph` stores HNSW graph structure (nodes + edges in shadow tables) alongside the vectors. The storage ratio is consistent: `vec_graph` uses 1.0-1.3x the space of `sqlite-vector`, with the overhead decreasing as dimension increases (graph metadata becomes a smaller fraction of total storage).

```plotly
--8<-- "benchmarks/charts/db_size_models.json"
```

## Vector Space Saturation

Real embeddings occupy a lower-dimensional manifold within their nominal dimension, which means they resist the **curse of dimensionality** better than random vectors. The saturation metrics confirm this — all three models show healthy relative contrast (>1.4) and distance CV (>0.05), meaning nearest-neighbor search remains meaningful.

```plotly
--8<-- "benchmarks/charts/saturation.json"
```

## Methodology

- **Storage**: Disk-persisted SQLite databases (default for model benchmarks)
- **Ground truth**: Python brute-force KNN for N ≤ 50K, `sqlite-vector-fullscan` for larger datasets
- **Queries**: 100 random queries per configuration
- **Embeddings**: Pre-computed from AG News dataset, cached as `.npy` files
- **Saturation**: 10K sampled pairwise distances per configuration
- **HNSW params**: M=16, ef_construction=200, ef_search=64
- **Results**: Stored as JSONL for cross-run aggregation

## Running Custom Benchmarks

```bash
# Specific model and sizes
python python/benchmark_compare.py --source model:all-MiniLM-L6-v2 --sizes 1000,5000,10000

# Random vectors for comparison
python python/benchmark_compare.py --source random --dim 384 --sizes 1000,5000,10000

# All profiles
make benchmark-small       # Random vectors, 3 dims, N≤50K
make benchmark-medium      # Random vectors, 2 dims, N=100K-500K
make benchmark-saturation  # Random vectors, 8 dims, N=50K
make benchmark-models      # Real embeddings, 3 models, N≤50K
```
