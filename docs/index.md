# sqlite-muninn

<div align="center">
    <img src="assets/muninn_logo_transparent.png" alt="Muninn Raven Logo" width=480px/>
    <p><i>Odin's mythic <a href="https://en.wikipedia.org/wiki/Huginn_and_Muninn">raven of Memory</a>.</i></p>
</div>

A zero-dependency C extension for SQLite to add an advanced collection of knowledge graph primitives like Vector Similarity Search, HNSW Indexes, Graph database, Centrality Measures, Community Detection, and Node2Vec capabilities.

## Features

- **HNSW Vector Index** — O(log N) approximate nearest neighbor search with incremental insert/delete
- **Graph Traversal** — BFS, DFS, shortest path, connected components, PageRank on any edge table
- **Centrality Measures** — Degree, betweenness (Brandes), and closeness centrality with weighted/temporal support
- **Community Detection** — Leiden algorithm for discovering graph communities with modularity scoring
- **Graph Adjacency Index** — Persistent CSR-cached adjacency with trigger-based dirty tracking and incremental rebuild
- **Graph Select** — dbt-inspired node selection syntax for lineage queries (ancestors, descendants, closures, set operations)
- **Node2Vec** — Learn structural node embeddings from graph topology, store in HNSW for similarity search
- **Zero dependencies** — Pure C11, compiles to a single `.dylib`/`.so`/`.dll`
- **SIMD accelerated** — ARM NEON and x86 SSE distance functions

## Quick Start

```bash
# Build
brew install sqlite  # macOS
make all

# Run tests
make test        # C unit tests
make test-python # Python integration tests
```

```sql
.load ./muninn

-- Create an HNSW vector index
CREATE VIRTUAL TABLE my_vectors USING hnsw_index(
    dimensions=384, metric='cosine', m=16, ef_construction=200
);

-- KNN search
SELECT rowid, distance FROM my_vectors
WHERE vector MATCH ?query AND k = 10 AND ef_search = 64;

-- Graph traversal on any edge table
SELECT node, depth FROM graph_bfs
WHERE edge_table = 'friendships' AND src_col = 'user_a'
  AND dst_col = 'user_b' AND start_node = 'alice' AND max_depth = 3;

-- Betweenness centrality (find bridge nodes)
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'friendships' AND src_col = 'user_a'
  AND dst_col = 'user_b' AND direction = 'both'
ORDER BY centrality DESC LIMIT 10;

-- Community detection (Leiden algorithm)
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'friendships' AND src_col = 'user_a'
  AND dst_col = 'user_b';

-- Persistent adjacency cache (auto-rebuilds on edge changes)
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='friendships', src_col='user_a', dst_col='user_b'
);
SELECT node, in_degree, out_degree FROM g ORDER BY out_degree DESC;

-- dbt-style lineage query (all descendants of alice)
SELECT node, depth FROM graph_select(
    'friendships', 'user_a', 'user_b', 'alice+'
);
```

## Available Functions

| Function | Type | Purpose | Output Columns |
|----------|------|---------|----------------|
| `hnsw_index` | Virtual Table | HNSW vector index | rowid, vector, distance |
| `graph_bfs` | TVF | Breadth-first traversal | node, depth, parent |
| `graph_dfs` | TVF | Depth-first traversal | node, depth, parent |
| `graph_shortest_path` | TVF | Shortest path (Dijkstra) | node, distance, path_order |
| `graph_components` | TVF | Connected components | node, component_id, component_size |
| `graph_pagerank` | TVF | PageRank scores | node, rank |
| `graph_degree` | TVF | Degree centrality | node, in_degree, out_degree, degree, centrality |
| `graph_betweenness` | TVF | Betweenness centrality | node, centrality |
| `graph_closeness` | TVF | Closeness centrality | node, centrality |
| `graph_leiden` | TVF | Leiden community detection | node, community_id, modularity |
| `graph_adjacency` | Virtual Table | Persistent CSR adjacency cache | node, node_idx, in/out_degree, weighted degrees |
| `graph_select` | TVF | dbt-style node selection | node, depth, direction |
| `node2vec_train()` | Scalar | Graph embedding generation | (scalar: nodes embedded) |

## Installation

=== "Python"

    ```bash
    pip install sqlite-muninn
    ```

=== "Node.js"

    ```bash
    npm install sqlite-muninn
    ```

=== "From Source"

    ```bash
    brew install sqlite  # macOS (or libsqlite3-dev on Linux)
    make all
    ```

See the [Getting Started guide](getting-started.md) for full installation instructions and platform notes.

## Learn More

- [Getting Started](getting-started.md) — Installation, setup, and quick tour of all seven subsystems
- [Architecture](architecture.md) — How the extension is organized, module layering, and design patterns
- [API Reference](api.md) — Complete reference for all functions and virtual tables
- [Centrality & Community Guide](centrality-community.md) — When and how to use centrality measures and Leiden
- [Node2Vec Guide](node2vec.md) — Parameter tuning and integration with HNSW
- [GraphRAG Cookbook](graphrag-cookbook.md) — End-to-end tutorial combining all subsystems
- [Benchmark Results](benchmarks.md) — Performance data across real-world workloads
