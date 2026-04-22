---
name: muninn-graph-algorithms
description: >
  Runs graph algorithms (BFS, DFS, shortest path, PageRank, connected components,
  degree / node-betweenness / edge-betweenness / closeness centrality, Leiden
  community detection) on any SQLite edge table via table-valued functions.
  Covers the WHERE edge_table = ... constraint calling convention and the
  persistent CSR adjacency cache (graph_adjacency). Use when the user mentions
  "graph traversal", "BFS", "DFS", "shortest path", "Dijkstra", "PageRank",
  "connected components", "centrality", "betweenness", "closeness",
  "Leiden community", "community detection", "graph algorithm in SQLite",
  "graph_bfs", "graph_pagerank", "graph_leiden", "CSR adjacency",
  "graph_adjacency", or wants to analyze a network / dependency / social graph
  stored in SQLite.
license: MIT
---

# muninn-graph-algorithms — Graph analytics on SQLite edge tables

muninn ships graph algorithms as **table-valued functions (TVFs)** that read any SQLite table with a source column and a destination column. You do not need a specialized graph storage layer — if you have an `edges(src, dst, weight?)` table, you can run every algorithm below.

All TVFs except `graph_select` use **constraint-based calling convention** (hidden columns in the WHERE clause). Positional-argument calls do not parse.

## Prepare an edge table

```sql
.load ./muninn

CREATE TABLE edges (
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  weight REAL DEFAULT 1.0
);

INSERT INTO edges(src, dst, weight) VALUES
  ('alice','bob', 1.0),
  ('bob','carol', 1.0),
  ('carol','dave', 1.0),
  ('dave','eve', 1.0),
  ('alice','eve', 2.0),
  ('eve','frank', 1.0);
```

## Shared constraint syntax

Every graph TVF accepts these constraints (except `graph_select`, which uses positional args):

| Constraint | Type | Required | Description |
|-----------|------|----------|-------------|
| `edge_table` | TEXT | yes | Name of the edge table |
| `src_col` | TEXT | yes | Source column name |
| `dst_col` | TEXT | yes | Destination column name |
| `weight_col` | TEXT | no | Optional weight column (REAL) |
| `direction` | TEXT | no | `'forward'` (default), `'reverse'`, `'both'` |
| `timestamp_col`, `time_start`, `time_end` | TEXT | no | Temporal filtering (ISO 8601) |

Identifiers are validated through `id_validate.c` — SQL-injection safe.

## BFS / DFS

```sql
SELECT node, depth, parent FROM graph_bfs
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'alice' AND max_depth = 2 AND direction = 'both';
```

```text
node   depth  parent
-----  -----  ------
alice  0      NULL
bob    1      alice
eve    1      alice
carol  2      bob
frank  2      eve
```

Swap `graph_bfs` → `graph_dfs` to change traversal order; same constraints.

## Shortest path

Unweighted BFS if `weight_col` is omitted; weighted Dijkstra if provided.

```sql
SELECT node, distance, path_order FROM graph_shortest_path
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'alice' AND end_node = 'frank'
    AND weight_col = 'weight';
```

```text
node   distance  path_order
-----  --------  ----------
alice  0.0       0
eve    2.0       1
frank  3.0       2
```

## Connected components

Treats edges as undirected regardless of `direction`.

```sql
SELECT node, component_id, component_size FROM graph_components
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

## PageRank

```sql
SELECT node, round(rank, 4) AS rank FROM graph_pagerank
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND damping = 0.85 AND iterations = 30
  ORDER BY rank DESC LIMIT 5;
```

Ranks sum to ~1.0. Typical settings: `damping = 0.85`, `iterations = 20–50`.

## Centrality

Four centrality measures share the constraint syntax plus optional `normalized` (0/1).

```sql
-- Degree — cheapest; run this first as a sanity check
SELECT node, in_degree, out_degree, degree FROM graph_degree
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  ORDER BY degree DESC;

-- Node betweenness — O(VE) exact via Brandes; auto-approx on big graphs
SELECT node, round(centrality, 4) AS bc FROM graph_node_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND normalized = 1
    AND auto_approx_threshold = 50000
  ORDER BY bc DESC LIMIT 10;

-- Edge betweenness — one row per edge
SELECT src, dst, round(centrality, 4) FROM graph_edge_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  ORDER BY centrality DESC LIMIT 10;

-- Closeness — Wasserman-Faust normalization (default on)
SELECT node, round(centrality, 4) FROM graph_closeness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both';
```

**Performance:** Betweenness is O(VE). For |V| > ~50k, `auto_approx_threshold` flips to a ceil(√N) source-sampling approximation. Set lower to accept approximation earlier.

## Leiden community detection

```sql
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND resolution = 1.0;
```

```text
node   community_id
-----  ------------
alice  0
bob    0
carol  0
dave   1
eve    1
frank  1
```

`resolution` ↑ → more, smaller communities. `resolution` ↓ → fewer, larger. The output also includes a `modularity` column (repeated per row). Leiden guarantees well-connected communities (unlike Louvain).

## Persistent CSR adjacency cache

If you run many algorithms on the same edge table, create a `graph_adjacency` virtual table — it caches CSR forward/reverse adjacency with delta-triggered lazy rebuild.

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
  edge_table='edges', src_col='src', dst_col='dst', weight_col='weight'
);

-- Pre-computed degrees
SELECT node, in_degree, out_degree, weighted_in_degree, weighted_out_degree
  FROM g ORDER BY out_degree DESC;

-- Edge mutations mark the cache dirty — next query rebuilds lazily
INSERT INTO edges VALUES ('alice','frank', 0.5);

-- Force a rebuild manually
INSERT INTO g(g) VALUES ('rebuild');              -- full rebuild
INSERT INTO g(g) VALUES ('incremental_rebuild');  -- merge deltas only
```

Cache thresholds: selective rebuild for <5% delta, block rebuild for 5–30%, full rebuild for >30%. For one-off queries on small graphs, the plain TVFs are equally fast and simpler — only use `graph_adjacency` when you need the persistence.

## Runtime variants

### Python

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)

# Graph TVFs accept parameterized constraints
rows = db.execute("""
  SELECT node, round(centrality, 4) FROM graph_node_betweenness
    WHERE edge_table = ? AND src_col = ? AND dst_col = ?
      AND direction = 'both' AND normalized = 1
    ORDER BY centrality DESC LIMIT 5
""", ('edges', 'src', 'dst')).fetchall()

for node, bc in rows:
    print(node, bc)
```

### Node.js

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);

const rows = db.prepare(`
  SELECT node, community_id FROM graph_leiden
    WHERE edge_table = ? AND src_col = ? AND dst_col = ?
      AND direction = 'both'
`).all("edges", "src", "dst");
```

## Common pitfalls

- **`no such function: graph_bfs` in positional form** — positional calls (`graph_bfs('edges', 'src', 'dst', 'a')`) do not parse. Use the `WHERE edge_table = ...` constraint form.
- **Zero results from betweenness on bidirectional data** — pre-2026 Brandes dedup bug is fixed; check you're on muninn ≥ 0.4.
- **`graph_adjacency` not triggering incremental rebuild** — it needs triggers on the source edge table; those are installed by `CREATE VIRTUAL TABLE`. If you DROP and re-create the edge table, re-create `graph_adjacency` too.
- **Leiden `modularity` column repeats per row** — that's correct; it's a global value, repeated for convenience so every row carries the score.
- **Direction ignored on `graph_components`** — components are intrinsically undirected. Use `graph_bfs` with `direction='forward'` if you need reachability.

## See also

- [muninn-graph-select](../muninn-graph-select/SKILL.md) — dbt-style lineage DSL (different calling convention)
- [muninn-node2vec](../muninn-node2vec/SKILL.md) — structural embeddings from graph topology
- [muninn-graphrag](../muninn-graphrag/SKILL.md) — KG retrieval combining extraction + graph
- [api.md § Graph](../../docs/api.md) — full reference
- [centrality-community.md](../../docs/centrality-community.md) — resolution sweeps and supernode patterns
