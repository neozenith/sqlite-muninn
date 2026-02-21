---
name: muninn
description: >
  Add HNSW vector similarity search, graph traversal (BFS, DFS, shortest path,
  PageRank, connected components, degree/betweenness/closeness centrality,
  Leiden community detection), persistent CSR adjacency caching, dbt-style
  graph selection (lineage queries), and Node2Vec embedding generation to any
  SQLite database. Composes with sqlite-lembed (local GGUF models) and
  sqlite-rembed (OpenAI/Ollama API) for text-in, semantic-search-out workflows.
  Use when users need vector search, knowledge graphs, graph algorithms,
  semantic search, text embeddings, dependency analysis, lineage queries, or RAG
  retrieval in SQLite. Triggers on "vector search", "nearest neighbor", "HNSW",
  "graph traversal", "knowledge graph", "PageRank", "Node2Vec", "embedding search",
  "similarity search in SQLite", "graph adjacency", "CSR cache", "graph select",
  "dbt selector", "lineage query", "dependency graph", "graph lineage",
  "text embedding", "lembed", "rembed", "GGUF", "semantic search".
license: MIT
compatibility: >
  Requires muninn extension. Python: `pip install sqlite-muninn`.
  Node.js: `npm install sqlite-muninn`. C: download amalgamation from GitHub Releases.
  Optional companions: `pip install sqlite-lembed` (GGUF embeddings),
  `pip install sqlite-rembed` (API embeddings).
metadata:
  author: joshpeak
  version: "0.2.0"
  repository: https://github.com/user/sqlite-muninn
allowed-tools: Bash(sqlite3:*), Bash(python:*), Bash(node:*)
---

# muninn — HNSW Vector Search + Graph Traversal for SQLite

C11 SQLite extension. Seven subsystems in one `.load`:
HNSW approximate nearest neighbor search, graph traversal TVFs, centrality measures,
Leiden community detection, persistent graph adjacency caching, dbt-style graph
selection, and Node2Vec. Composes with companion extensions for text-in,
semantic-search-out workflows.

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

### Available Functions

| Function | Type | Purpose | Output Columns |
|----------|------|---------|----------------|
| `graph_bfs(table, src, dst, start)` | TVF | Breadth-first traversal | node, depth, parent |
| `graph_dfs(table, src, dst, start)` | TVF | Depth-first traversal | node, depth, parent |
| `graph_shortest_path(table, src, dst, from, to)` | TVF | Shortest path | node, depth, parent |
| `graph_components(table, src, dst)` | TVF | Connected components | node, component |
| `graph_pagerank(table, src, dst)` | TVF | PageRank scores | node, rank |
| `graph_degree(table, src, dst)` | TVF | Degree centrality | node, in_degree, out_degree, degree |
| `graph_betweenness(table, src, dst)` | TVF | Betweenness centrality | node, centrality |
| `graph_closeness(table, src, dst)` | TVF | Closeness centrality | node, centrality |
| `graph_leiden(table, src, dst)` | TVF | Leiden community detection | node, community |
| `graph_adjacency` | VT | Persistent CSR adjacency cache | node, node_idx, in/out_degree |
| `graph_select(table, src, dst, selector)` | TVF | dbt-style lineage selection | node, depth, direction |

## Graph Adjacency Cache

Maintain a persistent CSR-cached adjacency index with trigger-based dirty tracking:

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='source', dst_col='target',
    weight_col='weight'  -- optional
);

-- Pre-computed degrees (reads from CSR cache, not edge table)
SELECT node, in_degree, out_degree FROM g ORDER BY out_degree DESC;

-- Force a full rebuild
INSERT INTO g(g) VALUES ('rebuild');

-- Force an incremental rebuild
INSERT INTO g(g) VALUES ('incremental_rebuild');
```

## Graph Select (Lineage Queries)

dbt-inspired node selection for dependency and lineage analysis:

```sql
-- All descendants of C
SELECT node, depth FROM graph_select('edges', 'source', 'target', 'C+');

-- All ancestors of C
SELECT node FROM graph_select('edges', 'source', 'target', '+C');

-- Build closure (descendants + their ancestors)
SELECT node FROM graph_select('edges', 'source', 'target', '@C');

-- Depth-limited: 2 hops up, 1 hop down
SELECT node FROM graph_select('edges', 'source', 'target', '2+C+1');

-- Intersection: common ancestors of D and E
SELECT node FROM graph_select('edges', 'source', 'target', '+D,+E');

-- Union: everything reachable from A or X
SELECT node FROM graph_select('edges', 'source', 'target', 'A+ X+');
```

Selector syntax: `+node` (ancestors), `node+` (descendants), `+node+` (both),
`N+node+M` (depth-limited), `@node` (closure), space (union), comma (intersection),
`not` (complement).

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

## Text Embedding Integration

muninn stores and searches vectors but does not generate embeddings. Two companion
SQLite extensions provide text-to-vector conversion that composes directly with
muninn's HNSW index (identical raw float32 blob format).

### sqlite-lembed (Local GGUF Models)

```python
import sqlite3
import sqlite_lembed
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
sqlite_lembed.load(db)
db.enable_load_extension(False)

# Register a GGUF embedding model
db.execute("""
    INSERT INTO temp.lembed_models(name, model)
    SELECT 'MiniLM', lembed_model_from_file('models/all-MiniLM-L6-v2.Q8_0.gguf')
""")

# Create HNSW index
db.execute("""
    CREATE VIRTUAL TABLE doc_vectors USING hnsw_index(
        dimensions=384, metric='cosine'
    )
""")

# Embed and index in one SQL statement
db.execute("""
    INSERT INTO doc_vectors(rowid, vector)
    SELECT id, lembed('MiniLM', content) FROM documents
""")

# Semantic search — embed the query inline
results = db.execute("""
    SELECT v.rowid, v.distance, d.content
    FROM doc_vectors v
    JOIN documents d ON d.id = v.rowid
    WHERE v.vector MATCH lembed('MiniLM', 'search query')
      AND k = 5
""").fetchall()
```

### sqlite-rembed (Remote API)

```python
import sqlite_rembed

sqlite_rembed.load(db)
db.execute("INSERT INTO temp.rembed_clients(name, options) VALUES ('text-embedding-3-small', 'openai')")

# Requires OPENAI_API_KEY environment variable
embedding = db.execute("SELECT rembed('text-embedding-3-small', 'hello world')").fetchone()[0]
```

### Auto-Embed Trigger

Automatically embed new documents on insert using a TEMP trigger:

```sql
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO doc_vectors(rowid, vector)
    VALUES (NEW.id, lembed('MiniLM', NEW.content));
END;

-- Now just insert text — the trigger handles embedding + indexing
INSERT INTO documents(id, content) VALUES (1, 'The quick brown fox');
```

**Important:** Use `TEMP` triggers. Persistent triggers fail if the extension is not
loaded before `sqlite3_open`.

## Common Mistakes

- **DO NOT** pass vectors as JSON arrays — they must be raw float32 blobs
- **DO NOT** forget `db.enable_load_extension(True)` before loading (Python)
- **DO NOT** mismatch vector dimensions — blob byte length must equal `dimensions * 4`
- **DO NOT** use `id` or `embedding` as column names — the columns are `rowid` and `vector`
- **DO NOT** use `max_elements` as a parameter — it does not exist. The HNSW index auto-resizes
- **DO NOT** use persistent triggers with `lembed()`/`rembed()` — use `TEMP` triggers only
- **DO NOT** use on macOS system Python without Homebrew Python or pysqlite3-binary
  (Apple's SQLite has `SQLITE_OMIT_LOAD_EXTENSION`)
- **DO** call `enable_load_extension(False)` after loading for security
- **DO** use `struct.pack(f'{dim}f', *values)` for vector encoding in Python
- **DO** use `Buffer.from(new Float32Array(values).buffer)` for vectors in Node.js
- **DO** load muninn before companion extensions (sqlite-lembed, sqlite-rembed)

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
