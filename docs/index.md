# sqlite-muninn

A zero-dependency C extension for SQLite that combines **HNSW vector similarity search**, **graph database primitives**, and **graph embedding generation** in a single shared library.

## Features

- **HNSW Vector Index** — O(log N) approximate nearest neighbor search with incremental insert/delete
- **Graph Traversal** — BFS, DFS, shortest path, connected components, PageRank on any edge table
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
```
