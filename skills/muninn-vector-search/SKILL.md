---
name: muninn-vector-search
description: >
  Builds and queries HNSW approximate-nearest-neighbor vector indexes in SQLite
  using the hnsw_index virtual table. Covers CREATE VIRTUAL TABLE with
  dimensions/metric/m/ef_construction, INSERT of raw float32 blobs, MATCH
  queries with k and ef_search, and vector blob encoding in Python
  (struct.pack), Node.js (Float32Array), and C. Use when the user mentions
  "vector search", "HNSW", "nearest neighbor", "KNN", "similarity search in
  SQLite", "hnsw_index", "cosine similarity", "L2 distance", "inner product",
  "vector MATCH", "ef_search", "float32 blob", or asks to add semantic /
  similarity search to a SQLite database.
license: MIT
---

# muninn-vector-search — HNSW vector search in SQLite

The `hnsw_index` virtual table implements Hierarchical Navigable Small World indexes (Malkov & Yashunin, 2018) with SIMD distance kernels. Read this skill when the user wants to store vectors and run KNN queries. If the user also wants to generate those vectors from text, combine with [muninn-embed-text](../muninn-embed-text/SKILL.md).

## Create an index

```sql
.load ./muninn

CREATE VIRTUAL TABLE vec USING hnsw_index(
    dimensions=384,           -- required
    metric='cosine',          -- 'l2' | 'cosine' | 'inner_product'
    m=16,                     -- optional, max neighbors per layer (default 16)
    ef_construction=200       -- optional, insert beam width (default 200)
);
```

| Metric | Formula | Range | When to use |
|--------|---------|-------|-------------|
| `l2` | Squared Euclidean, no sqrt | `[0, ∞)` | Raw vectors without normalization |
| `cosine` | `1 - cos(a, b)` | `[0, 2]` | Normalized or text embeddings — default for most NLP |
| `inner_product` | `-dot(a, b)` | `(-∞, ∞)` | When you want larger magnitudes to win |

The index auto-resizes; there is no `max_elements` parameter. Higher `m` and `ef_construction` trade insert time and memory for recall.

## Insert vectors (per runtime)

Vectors are **raw little-endian IEEE 754 float32 blobs** — no header, `4 × dimensions` bytes. A dimension-mismatched blob raises an error on INSERT (no silent truncation).

### SQLite CLI

```sql
-- Vectors are typically inserted from a host language; for hand-crafted examples:
INSERT INTO vec(rowid, vector) VALUES
  (1, X'0000803F000000000000000000000000');  -- (1.0, 0.0, 0.0, 0.0) for dim=4
```

### Python

```python
import struct

dim = 384
vector = [0.1] * dim
blob = struct.pack(f'{dim}f', *vector)
db.execute("INSERT INTO vec(rowid, vector) VALUES (?, ?)", (1, blob))
```

Batch loading:

```python
import numpy as np

# vectors is an (N, 384) np.float32 array
rows = [(i, v.tobytes()) for i, v in enumerate(vectors)]
db.executemany("INSERT INTO vec(rowid, vector) VALUES (?, ?)", rows)
```

### Node.js

```javascript
const dim = 384;
const vector = new Float32Array(dim).fill(0.1);
const blob = Buffer.from(vector.buffer);
db.prepare("INSERT INTO vec(rowid, vector) VALUES (?, ?)").run(1, blob);
```

### C

```c
float vec[384] = { /* ... */ };
sqlite3_stmt *stmt;
sqlite3_prepare_v2(db, "INSERT INTO vec(rowid, vector) VALUES (?1, ?2)", -1, &stmt, NULL);
sqlite3_bind_int64(stmt, 1, 1);
sqlite3_bind_blob(stmt, 2, vec, sizeof(vec), SQLITE_STATIC);
sqlite3_step(stmt);
sqlite3_finalize(stmt);
```

### WASM

```javascript
const dim = 384;
const blob = new Uint8Array(new Float32Array(dim).fill(0.1).buffer);
db.exec({
  sql: "INSERT INTO vec(rowid, vector) VALUES (?, ?)",
  bind: [1, blob]
});
```

## KNN search

```sql
SELECT rowid, distance FROM vec
  WHERE vector MATCH ?query_blob
    AND k = 10;
```

```text
rowid  distance
-----  --------
42     0.0321
17     0.0877
...
```

### Tuning recall vs speed

```sql
-- Default ef_search = k * 2
SELECT rowid, distance FROM vec
  WHERE vector MATCH ?query
    AND k = 10
    AND ef_search = 100;   -- higher = better recall, slower
```

Start at `ef_search = k * 2`, raise until recall plateaus on your golden set.

## Delete and drop

```sql
DELETE FROM vec WHERE rowid = 42;   -- soft delete + neighbor reconnect
DROP TABLE vec;                      -- removes shadow tables too
-- UPDATE is not supported. Delete and re-insert instead.
```

Shadow tables auto-managed: `vec_config`, `vec_nodes`, `vec_edges`.

## End-to-end Python recipe

```python
import sqlite3, struct
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)

db.execute("CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=4, metric='l2')")

pack = lambda v: struct.pack('4f', *v)

db.executemany("INSERT INTO vec(rowid, vector) VALUES (?, ?)", [
    (1, pack([1.0, 0.0, 0.0, 0.0])),
    (2, pack([0.9, 0.1, 0.0, 0.0])),
    (3, pack([0.0, 0.0, 1.0, 0.0])),
])

query = pack([1.0, 0.0, 0.0, 0.0])
rows = db.execute(
    "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = 2",
    (query,)
).fetchall()

for rowid, dist in rows:
    print(rowid, round(dist, 4))
```

```text
1 0.0
2 0.02
```

## Common pitfalls

- **DO NOT** pass vectors as JSON arrays — they must be raw float32 blobs.
- **DO NOT** mismatch dimensions — blob byte length must equal `dimensions * 4`; otherwise INSERT raises.
- **DO NOT** use `id` or `embedding` as column names — the fixed names are `rowid` and `vector`.
- **DO NOT** use `max_elements` — it does not exist; the index grows on demand.
- **DO** normalize inputs before using `cosine` — the metric does not L2-normalize for you.
- **DO** prefer `l2` when you don't know whether inputs are normalized — its monotonicity does not depend on vector norm.

## See also

- [muninn-embed-text](../muninn-embed-text/SKILL.md) — generate the vectors in SQL via GGUF models
- [muninn-node2vec](../muninn-node2vec/SKILL.md) — write graph embeddings into an HNSW index
- [api.md#hnsw_index](../../docs/api.md) — full signature reference
