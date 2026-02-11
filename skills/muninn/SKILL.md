---
name: muninn
description: >
  Add HNSW vector similarity search, graph traversal (BFS, DFS, shortest path,
  PageRank, connected components, degree/betweenness/closeness centrality,
  Leiden community detection), and Node2Vec embedding generation to any SQLite
  database. Use when users need vector search, knowledge graphs, graph algorithms,
  semantic search, or RAG retrieval in SQLite. Triggers on "vector search",
  "nearest neighbor", "HNSW", "graph traversal", "knowledge graph", "PageRank",
  "Node2Vec", "embedding search", "similarity search in SQLite".
license: MIT
compatibility: >
  Requires muninn extension. Python: `pip install sqlite-muninn`.
  Node.js: `npm install sqlite-muninn`. C: download amalgamation from GitHub Releases.
metadata:
  author: joshpeak
  version: "{{VERSION}}"
  repository: https://github.com/user/sqlite-muninn
allowed-tools: Bash(sqlite3:*), Bash(python:*), Bash(node:*)
---

# muninn — HNSW Vector Search + Graph Traversal for SQLite

Zero-dependency C11 SQLite extension. Three subsystems in one `.load`:
HNSW approximate nearest neighbor search, graph traversal TVFs, and Node2Vec.

## Quick Start (Python)

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)
```

## Quick Start (Node.js)

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);
```

## Quick Start (SQLite CLI)

```sql
.load ./muninn
```

## HNSW Vector Index

### Create an index

```sql
CREATE VIRTUAL TABLE my_embeddings USING hnsw_index(
    dimensions=384, metric='l2'
);
```

Supported metrics: `l2` (Euclidean), `cosine`, `inner_product`.

Optional tuning parameters: `m` (max connections per node, default 16),
`ef_construction` (build-time beam width, default 200).

### Insert vectors

Vectors are raw float32 blobs (little-endian). In Python:

```python
import struct
dim = 384
vector = [0.1] * dim
blob = struct.pack(f'{dim}f', *vector)
db.execute("INSERT INTO my_embeddings(rowid, vector) VALUES (?, ?)", (1, blob))
```

In Node.js:

```javascript
const dim = 384;
const vector = new Float32Array(dim).fill(0.1);
const blob = Buffer.from(vector.buffer);
db.prepare("INSERT INTO my_embeddings(rowid, vector) VALUES (?, ?)").run(1, blob);
```

### Search (k-nearest neighbors)

```sql
SELECT rowid, distance
FROM my_embeddings
WHERE vector MATCH ?
  AND k = 10;
```

### Delete vectors

```sql
DELETE FROM my_embeddings WHERE rowid = 42;
```

## Graph Traversal TVFs

Graph TVFs operate on ANY existing SQLite table with source/target columns.
They do NOT require HNSW — they work on plain relational edge tables.

### BFS traversal

```sql
SELECT * FROM graph_bfs(
    'edges',        -- table name
    'source',       -- source column
    'target',       -- target column
    '1'             -- start node ID
);
```

### DFS traversal

```sql
SELECT * FROM graph_dfs(
    'edges', 'source', 'target', '1'
);
```

### Shortest path

```sql
SELECT * FROM graph_shortest_path(
    'edges', 'source', 'target',
    '1',            -- from node
    '42'            -- to node
);
```

### Connected components

```sql
SELECT * FROM graph_components(
    'edges', 'source', 'target'
);
```

### PageRank

```sql
SELECT * FROM graph_pagerank(
    'edges', 'source', 'target'
);
```

### Available TVFs

| Function | Purpose | Output Columns |
|----------|---------|----------------|
| `graph_bfs(table, src, dst, start)` | Breadth-first traversal | node, depth, parent |
| `graph_dfs(table, src, dst, start)` | Depth-first traversal | node, depth, parent |
| `graph_shortest_path(table, src, dst, from, to)` | Shortest path | node, depth, parent |
| `graph_components(table, src, dst)` | Connected components | node, component |
| `graph_pagerank(table, src, dst)` | PageRank scores | node, rank |
| `graph_degree(table, src, dst)` | Degree centrality | node, in_degree, out_degree, degree |
| `graph_betweenness(table, src, dst)` | Betweenness centrality | node, centrality |
| `graph_closeness(table, src, dst)` | Closeness centrality | node, centrality |
| `graph_leiden(table, src, dst)` | Leiden community detection | node, community |

## Node2Vec Embedding Generation

Generate graph embeddings and store them directly in an HNSW index:

```sql
CREATE VIRTUAL TABLE node_embeddings USING hnsw_index(
    dimensions=64, metric='cosine'
);

SELECT node2vec_train(
    'edges',            -- edge table
    'source',           -- source column
    'target',           -- target column
    'node_embeddings',  -- destination HNSW table
    64                  -- embedding dimension
);
```

## Common Mistakes

- **DO NOT** pass vectors as JSON arrays — they must be raw float32 blobs
- **DO NOT** forget `db.enable_load_extension(True)` before loading (Python)
- **DO NOT** mismatch vector dimensions — blob byte length must equal `dimensions * 4`
- **DO NOT** use `id` or `embedding` as column names — the columns are `rowid` and `vector`
- **DO NOT** use on macOS system Python without Homebrew Python or pysqlite3-binary
  (Apple's SQLite has `SQLITE_OMIT_LOAD_EXTENSION`)
- **DO** call `enable_load_extension(False)` after loading for security
- **DO** use `struct.pack(f'{dim}f', *values)` for vector encoding in Python
- **DO** use `Buffer.from(new Float32Array(values).buffer)` for vectors in Node.js

## Platform Notes

- macOS: System Python's sqlite3 disables load_extension(). Use Homebrew Python
  (`brew install python`) or install `pysqlite3-binary`.
- All platforms: The extension is a native binary (.so/.dylib/.dll) — it must match
  your OS and architecture.

## Further Reading

See `references/` for detailed per-language cookbooks:
- [cookbook-python.md](references/cookbook-python.md) — Semantic search, RAG, batch loading
- [cookbook-node.md](references/cookbook-node.md) — Express endpoints, buffer helpers
- [cookbook-c.md](references/cookbook-c.md) — Static linking, sqlite3_auto_extension
- [cookbook-sql.md](references/cookbook-sql.md) — Pure SQL workflows, graph pipelines
- [vector-encoding.md](references/vector-encoding.md) — Cross-language float32 blob format
- [platform-caveats.md](references/platform-caveats.md) — macOS, Windows, glibc details
