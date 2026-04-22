---
name: muninn-node2vec
description: >
  Trains Node2Vec structural graph embeddings (Grover & Leskovec, 2016) from a
  SQLite edge table and writes them directly into an hnsw_index virtual table.
  Covers the node2vec_train scalar function, p/q walk bias tuning, window
  size, negative sampling, epochs, and composing with KNN vector search.
  Use when the user mentions "node2vec", "graph embedding", "structural
  embedding", "random walk embedding", "DeepWalk", "node2vec_train",
  "similar nodes", "graph representation learning", or wants to compute
  embeddings from graph topology (not text).
license: MIT
---

# muninn-node2vec — Structural graph embeddings

`node2vec_train` learns a low-dimensional embedding for every node in an edge table by running biased random walks and training a Skip-gram-with-Negative-Sampling (SGNS) model. The output is written directly into an HNSW index, which means you can immediately run KNN queries on it ("nodes structurally similar to this one") via [muninn-vector-search](../muninn-vector-search/SKILL.md).

## Signature

```sql
node2vec_train(
    edge_table    TEXT,      -- source edge table
    src_col       TEXT,      -- source column
    dst_col       TEXT,      -- destination column
    output_table  TEXT,      -- HNSW virtual table (must already exist)
    dimensions    INTEGER,   -- MUST match the HNSW table
    p             REAL,      -- return parameter (walk bias)
    q             REAL,      -- in-out parameter (walk bias)
    num_walks     INTEGER,   -- walks per node
    walk_length   INTEGER,   -- nodes per walk
    window        INTEGER,   -- skip-gram context window
    neg_samples   INTEGER,   -- negative samples per positive
    learning_rate REAL,      -- SGD learning rate
    epochs        INTEGER    -- training passes
) -> INTEGER                 -- count of nodes embedded
```

All 12 arguments are **required and positional**. Edges are treated as undirected.

## Minimal recipe

```sql
.load ./muninn

-- 1. Ensure you have an edge table
CREATE TABLE edges (src TEXT, dst TEXT);
INSERT INTO edges VALUES
  ('alice','bob'),('bob','carol'),('carol','dave'),
  ('dave','eve'),('alice','eve'),('eve','frank');

-- 2. Create an HNSW index sized to the embedding dimension
CREATE VIRTUAL TABLE user_emb USING hnsw_index(dimensions=32, metric='cosine');

-- 3. Train — embeddings land in user_emb (one row per unique node)
SELECT node2vec_train(
  'edges', 'src', 'dst', 'user_emb',
  32,                    -- dimensions (must match HNSW)
  1.0, 1.0,              -- p, q  (both = 1.0 → DeepWalk)
  10, 40,                -- num_walks, walk_length
  5, 5,                  -- window, negative samples
  0.025, 3               -- learning_rate, epochs
);
```

```text
node2vec_train('edges','src','dst','user_emb',32,...)
-----------------------------------------------------
6                                                        -- nodes embedded
```

Now `user_emb` contains 6 rows keyed by internal node IDs.

## Query "structurally similar" nodes

Retrieve a seed node's embedding, then run KNN against the index:

```sql
-- Fetch alice's embedding blob (rowid is the internal node index; map via the node name column in shadow tables if needed)
SELECT rowid, distance FROM user_emb
  WHERE vector MATCH (SELECT vector FROM user_emb WHERE rowid = 1)
    AND k = 3;
```

Because node2vec uses random-walk context similarity, these are nodes that play *structurally similar roles* in the graph — not textually similar, not metadata-similar.

## p/q tuning

The two walk bias parameters control the random walk's tendency to revisit vs. explore.

| Setting | Behavior | Use case |
|---------|----------|----------|
| `p = q = 1.0` | Unbiased — equivalent to DeepWalk | Baseline |
| `p < 1.0, q > 1.0` | BFS-like (local neighborhoods) | Structural equivalence, role detection |
| `p > 1.0, q < 1.0` | DFS-like (deep exploration) | Community / homophily embeddings |

Typical ranges: `p, q ∈ [0.25, 4.0]`. Sweep `q` first — it has more impact than `p` on downstream clustering quality.

## Sizing guidance

| Graph size | Recommended dims | num_walks × walk_length | epochs |
|------------|------------------|-------------------------|--------|
| |V| < 1k | 32–64 | 10 × 40 | 3–5 |
| 1k ≤ |V| < 10k | 64–128 | 20 × 80 | 3 |
| 10k ≤ |V| < 100k | 128 | 10 × 80 | 1–2 |
| |V| ≥ 100k | 128 | 5 × 40 | 1 |

HNSW memory ≈ `(dim × 4 + 356)` bytes/node — a 100k × 128-dim index uses ~90 MB before peripheral overhead.

## Runtime variants

### Python

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)

db.execute("CREATE VIRTUAL TABLE emb USING hnsw_index(dimensions=64, metric='cosine')")

count = db.execute("""
  SELECT node2vec_train(
    'edges', 'src', 'dst', 'emb',
    64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 3
  )
""").fetchone()[0]

print(f"Embedded {count} nodes")
```

### Node.js

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);

db.exec(`CREATE VIRTUAL TABLE emb USING hnsw_index(dimensions=64, metric='cosine')`);
const n = db.prepare(`
  SELECT node2vec_train(
    'edges', 'src', 'dst', 'emb',
    64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 3
  )
`).pluck().get();
```

## Common pitfalls

- **`dimensions mismatch`** — the `dimensions` argument to `node2vec_train` must exactly match the HNSW table's `dimensions` setting. A 64-dim train into a 32-dim index raises.
- **HNSW index is not empty** — `node2vec_train` will INSERT into the target. If the index already has rows with colliding rowids, the behavior is undefined. Use a fresh HNSW table.
- **Slow on very wide graphs** — `num_walks × walk_length × epochs` dominates runtime. Start with `10 × 40 × 3` and only scale up if embedding quality plateaus.
- **Directed edges interpreted as undirected** — that's intentional; Node2Vec is defined on undirected graphs. If direction matters, use [muninn-graph-algorithms](../muninn-graph-algorithms/SKILL.md) instead.
- **No incremental update** — adding new edges requires a full re-train. For streaming graphs, retrain periodically rather than per-edge.

## When to use Node2Vec vs. text embeddings

| Use case | Best approach |
|----------|---------------|
| "Users who friend similar sets of people" | Node2Vec — pure topology |
| "Nodes with similar descriptions" | `muninn_embed` on text + HNSW — pure content |
| "Combine topology and content" | Concatenate node2vec + text embeddings, L2-normalize, index in HNSW (`metric='cosine'`) |

## See also

- [muninn-vector-search](../muninn-vector-search/SKILL.md) — the HNSW index the embeddings land in
- [muninn-graph-algorithms](../muninn-graph-algorithms/SKILL.md) — alternative graph analytics
- [node2vec.md](../../docs/node2vec.md) — full reference
