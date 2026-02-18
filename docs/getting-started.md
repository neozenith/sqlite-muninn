# Getting Started

## Installation

### From Source (C)

Build the extension from source — requires a C11 compiler and SQLite development headers.

=== "macOS (Homebrew)"

    ```bash
    brew install sqlite
    git clone https://github.com/neozenith/sqlite-muninn.git
    cd sqlite-muninn
    make all
    ```

=== "Linux (apt)"

    ```bash
    sudo apt-get install libsqlite3-dev
    git clone https://github.com/neozenith/sqlite-muninn.git
    cd sqlite-muninn
    make all
    ```

This produces `muninn.dylib` (macOS) or `muninn.so` (Linux) in the project root.

### Python

```bash
pip install sqlite-muninn
```

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)
```

!!! warning "macOS System Python"
    Apple's system Python ships with `SQLITE_OMIT_LOAD_EXTENSION` enabled, which prevents loading any SQLite extension. Use Homebrew Python (`brew install python`) or install `pysqlite3-binary` as a drop-in replacement.

### Node.js

```bash
npm install sqlite-muninn
```

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);
```

### SQLite CLI

```bash
# Load the extension directly
sqlite3
.load ./muninn
```

## Verify Installation

After loading the extension, run a quick smoke test:

```sql
-- Create a small HNSW index
CREATE VIRTUAL TABLE test_vec USING hnsw_index(dimensions=4, metric='l2');

-- Insert a vector (4 floats as a blob)
INSERT INTO test_vec(rowid, vector) VALUES (1, X'0000803F0000003F000080BE0000803F');

-- Search (returns the inserted vector)
SELECT rowid, distance FROM test_vec WHERE vector MATCH X'0000803F0000003F000080BE0000803F' AND k = 1;
-- Expected: rowid=1, distance=0.0

-- Clean up
DROP TABLE test_vec;
```

## Quick Tour

muninn provides seven subsystems, all registered by a single `.load`:

### 1. HNSW Vector Index

Create an index, insert embeddings, and run KNN search:

```sql
CREATE VIRTUAL TABLE my_vectors USING hnsw_index(
    dimensions=384, metric='cosine', m=16, ef_construction=200
);

-- Insert (rowid + float32 blob)
INSERT INTO my_vectors(rowid, vector) VALUES (1, ?);

-- KNN search
SELECT rowid, distance FROM my_vectors
WHERE vector MATCH ?query AND k = 10 AND ef_search = 64;
```

### 2. Graph Traversal

Run BFS, DFS, shortest path, connected components, or PageRank on **any** existing edge table:

```sql
-- BFS traversal
SELECT node, depth, parent FROM graph_bfs
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = 'alice' AND max_depth = 3 AND direction = 'both';

-- Connected components
SELECT node, component_id, component_size FROM graph_components
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

### 3. Centrality Measures

Identify important nodes with degree, betweenness, or closeness centrality:

```sql
-- Find bridge nodes (betweenness centrality)
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both'
ORDER BY centrality DESC LIMIT 10;
```

### 4. Community Detection

Discover clusters with the Leiden algorithm:

```sql
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

### 5. Node2Vec Embeddings

Learn structural embeddings from graph topology and store them in HNSW for similarity search:

```sql
CREATE VIRTUAL TABLE node_emb USING hnsw_index(dimensions=64, metric='cosine');

SELECT node2vec_train(
    'edges', 'src', 'dst', 'node_emb',
    64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 5
);
```

**Vector Encoding**

Vectors are passed as **raw float32 blobs** (little-endian, `sizeof(float) * dim` bytes). They are **not** JSON arrays.

=== "Python"

    ```python
    import struct
    dim = 384
    values = [0.1] * dim
    blob = struct.pack(f'{dim}f', *values)
    ```

=== "Node.js"

    ```javascript
    const dim = 384;
    const values = new Float32Array(dim).fill(0.1);
    const blob = Buffer.from(values.buffer);
    ```

=== "C"

    ```c
    float vec[384];
    // fill vec...
    sqlite3_bind_blob(stmt, 1, vec, sizeof(vec), SQLITE_STATIC);
    ```

### 6. Graph Adjacency Cache

Maintain a persistent CSR-cached adjacency index that auto-rebuilds when edges change:

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst', weight_col='weight'
);

-- Pre-computed degrees (reads from cache, not edge table)
SELECT node, in_degree, out_degree FROM g ORDER BY out_degree DESC;

-- Algorithm TVFs can use the adjacency cache as their source
SELECT node, centrality FROM graph_degree('g', 'src', 'dst', 'weight');
```

### 7. Graph Select (Lineage Queries)

Query graph lineage with dbt-inspired selector syntax:

```sql
-- All descendants of node C
SELECT node, depth FROM graph_select('edges', 'src', 'dst', 'C+');

-- All ancestors of node C
SELECT node, depth FROM graph_select('edges', 'src', 'dst', '+C');

-- Build closure: descendants of C plus all their ancestors
SELECT node FROM graph_select('edges', 'src', 'dst', '@C');

-- Common ancestors of D and E (intersection)
SELECT node FROM graph_select('edges', 'src', 'dst', '+D,+E');
```

## Next Steps

- [Architecture](architecture.md) — How the extension is organized internally
- [API Reference](api.md) — Complete reference for all functions and virtual tables
- [Centrality & Community Guide](centrality-community.md) — When and how to use centrality measures and Leiden
- [Node2Vec Guide](node2vec.md) — Parameter tuning and integration patterns
- [GraphRAG Cookbook](graphrag-cookbook.md) — End-to-end tutorial combining all subsystems
- [Benchmark Results](benchmarks.md) — Performance data across real-world workloads
