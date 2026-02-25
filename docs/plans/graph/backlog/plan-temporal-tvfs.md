# Plan: Temporal Awareness for Base Graph TVFs

## Motivation

Graphiti/Zep (getzep/graphiti, arXiv:2501.13956) implements a **bi-temporal knowledge graph** for agent memory on top of Neo4j + Python. Their core insight: temporal awareness isn't a separate algorithm — it's a **filter** applied to standard graph operations. Every graph TVF becomes temporally aware if it only considers edges valid at a given point in time.

Our new modules (`graph_centrality.c`, `graph_community.c`) already support temporal filtering via `graph_load.c`. But the original five TVFs in `graph_tvf.c` — **BFS, DFS, shortest path, components, PageRank** — have zero temporal awareness. This is the gap.

### What Graphiti's Bi-Temporal Model Looks Like

Every edge carries four timestamps:

| Field | Timeline | Meaning |
|-------|----------|---------|
| `valid_at` | Event time | When the fact became true in the real world |
| `invalid_at` | Event time | When the fact stopped being true |
| `created_at` | Transaction time | When the system learned the fact |
| `expired_at` | Transaction time | When the system marked it superseded |

A "point-in-time snapshot" query filters edges where:
```sql
valid_at <= ? AND (invalid_at IS NULL OR invalid_at > ?)
```

We don't need to implement the bi-temporal model itself — that's application-layer. What we need is the plumbing: **all graph TVFs accept optional temporal columns and filter edges accordingly.**

---

## Current State

### What Already Has Temporal Support

| Module | TVFs | Temporal? | Uses graph_load.c? |
|--------|------|-----------|---------------------|
| `graph_centrality.c` | degree, betweenness, closeness | Yes | Yes |
| `graph_community.c` | leiden | Yes | Yes |

### What Does NOT

| Module | TVFs | Load Strategy | Graph Structure |
|--------|------|---------------|-----------------|
| `graph_tvf.c` | BFS, DFS | **Lazy** (per-node SQL queries) | None (streaming) |
| `graph_tvf.c` | shortest_path | **Lazy** (per-node SQL queries) | None (streaming) |
| `graph_tvf.c` | components | **Eager** (load all edges) | UnionFind |
| `graph_tvf.c` | pagerank | **Eager** (load all edges) | PRAdjList (O(N) lookup, no weights) |

### Shared Infrastructure (Already Exists)

- `graph_common.h` — `graph_best_index_common()`, `graph_safe_text()`, `graph_str_hash()`
- `graph_load.c/.h` — `GraphData` with O(1) hash-map lookup, `GraphLoadConfig` with temporal WHERE clause
- `id_validate.c` — SQL identifier safety

---

## Design Decision: Lazy vs. Eager Loading

The key architectural question is how to add temporal filtering to lazy (streaming) TVFs like BFS/DFS.

### Option A: Keep Lazy, Add WHERE Clause to Per-Node Queries

BFS/DFS currently prepare two statements:
```sql
-- Forward: "SELECT dst FROM edges WHERE src = ?"
-- Reverse: "SELECT src FROM edges WHERE dst = ?"
```

With temporal filtering, these become:
```sql
SELECT dst FROM edges WHERE src = ?
  AND (ts >= ?2 OR ?2 IS NULL) AND (ts <= ?3 OR ?3 IS NULL)
```

**Pros:** No behavior change, same memory profile (O(visited) not O(E)), works on huge graphs.
**Cons:** Temporal params must be threaded through the SQL builder; doesn't unify with `graph_load.c`.

### Option B: Migrate All TVFs to Eager Loading via graph_load.c

Load the full (filtered) graph upfront, then run algorithms on in-memory `GraphData`.

**Pros:** Unified code path, temporal+weight+direction for free, shared `graph_load.c` gets better tested.
**Cons:** BFS/DFS become eager (O(E) memory), can't stream on graphs larger than RAM.

### Recommendation: Hybrid (Option A for BFS/DFS/shortest_path, Option B for components/pagerank)

- **BFS, DFS, shortest_path** — keep lazy, extend SQL queries with temporal WHERE clause
- **Components, PageRank** — migrate to `graph_load.c` (they already load eagerly; we just replace `UnionFind`/`PRAdjList` with `GraphData`)

This preserves the lazy streaming advantage for traversal algorithms while unifying the eager algorithms.

---

## Phase 1: Add Temporal + Weight Columns to BFS/DFS

### Schema Changes

Current BFS/DFS hidden columns:
```
edge_table, src_col, dst_col, start_node, max_depth, direction
```

New (append to end):
```
edge_table, src_col, dst_col, start_node, max_depth, direction,
weight_col, timestamp_col, time_start, time_end
```

**Why add `weight_col` to BFS?** Not for BFS itself, but for consistency — every graph TVF should accept the same base parameters. BFS ignores `weight_col`; shortest_path already has it.

### SQL Builder Changes

Extract a shared SQL builder for per-node neighbor queries:

```c
/* Build a parameterized neighbor-lookup statement.
 * Returns a prepared statement with:
 *   ?1 = source/target node ID
 *   ?2 = time_start (or NULL)
 *   ?3 = time_end (or NULL)
 */
static sqlite3_stmt *prepare_neighbor_query(
    sqlite3 *db,
    const char *edge_table, const char *src_col, const char *dst_col,
    const char *weight_col,
    const char *timestamp_col,
    int forward,        /* 1 = src→dst, 0 = dst→src */
    char **pzErrMsg
);
```

The generated SQL:
```sql
-- Without temporal (current behavior):
SELECT "dst" FROM "edges" WHERE "src" = ?1

-- With temporal:
SELECT "dst" FROM "edges" WHERE "src" = ?1
  AND ("ts" >= ?2 OR ?2 IS NULL)
  AND ("ts" <= ?3 OR ?3 IS NULL)

-- With weight + temporal:
SELECT "dst", "weight" FROM "edges" WHERE "src" = ?1
  AND ("ts" >= ?2 OR ?2 IS NULL)
  AND ("ts" <= ?3 OR ?3 IS NULL)
```

### xBestIndex Migration

Replace the hand-rolled xBestIndex in `gtrav_best_index` with `graph_best_index_common()` from `graph_common.h`. This gives us the bitmask pattern for handling optional params (weight_col, timestamp_col, time_start, time_end).

### Test Coverage

Add to `pytests/test_graph_tvf.py`:
- `test_bfs_temporal_filter` — BFS on graph with timestamps, edges outside window not traversed
- `test_dfs_temporal_filter` — same for DFS
- `test_bfs_weight_col_ignored` — confirm weight_col param accepted but ignored by BFS

---

## Phase 2: Add Temporal to Shortest Path

### Schema Changes

Current shortest_path hidden columns:
```
edge_table, src_col, dst_col, start_node, end_node, weight_col
```

New:
```
edge_table, src_col, dst_col, start_node, end_node, weight_col,
direction, timestamp_col, time_start, time_end
```

Note: shortest_path currently lacks a `direction` parameter. Adding it aligns with every other TVF.

### SQL Builder

Reuse the shared `prepare_neighbor_query()` from Phase 1. The Dijkstra variant already uses `weight_col`; we just bind the temporal params too.

### Test Coverage

- `test_shortest_path_temporal` — path exists in full graph but not in time-filtered view
- `test_shortest_path_direction` — verify direction param works (currently missing)

---

## Phase 3: Migrate Components to graph_load.c

### Why Migrate

Components currently uses `UnionFind` with its own SQL loading:
```c
// Current: ~50 lines of duplicated SQL construction + UnionFind
char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"", ...);
```

After migration:
```c
GraphData g;
graph_data_init(&g);
GraphLoadConfig config = { .edge_table = ..., .timestamp_col = ..., ... };
graph_data_load(db, &config, &g, &pzErrMsg);
// Run Union-Find on g.out[] adjacency lists
```

This gives us temporal filtering, weight support, and direction support for free.

### Schema Changes

Current:
```
node, component_id, component_size, edge_table, src_col, dst_col
```

New:
```
node, component_id, component_size,
edge_table, src_col, dst_col, direction,
timestamp_col, time_start, time_end
```

### Algorithm Adaptation

Union-Find currently iterates raw SQL rows. After migration, iterate `GraphData.out[]` adjacency lists:

```c
for (int i = 0; i < g.node_count; i++) {
    for (int e = 0; e < g.out[i].count; e++) {
        uf_union(&uf, i, g.out[i].edges[e].target);
    }
}
```

### Test Coverage

- `test_components_temporal` — two components exist in full graph; time filter splits into more
- `test_components_direction` — direction param now available

---

## Phase 4: Migrate PageRank to graph_load.c

### Why Migrate

PageRank's `PRAdjList` has O(N) linear scan for node lookup vs `GraphData`'s O(1) hash map. Migration is a performance improvement as well as a feature addition.

Current `PRAdjList`:
- `out_edges[][]` — forward adjacency only
- No weights, no reverse adjacency, no temporal support
- `pr_adj_find_or_add()` — linear scan: `for (int i = 0; i < a->count; i++)`

After migration to `GraphData`:
- O(1) node lookup
- Weight support (weighted PageRank)
- Temporal filtering
- Direction support

### Schema Changes

Current:
```
node, rank, edge_table, src_col, dst_col, damping, iterations
```

New:
```
node, rank,
edge_table, src_col, dst_col, weight_col, damping, iterations,
direction, timestamp_col, time_start, time_end
```

### Algorithm Adaptation

PageRank power iteration currently uses `PRAdjList.out_edges`. After migration:

```c
// For each node, distribute rank to out-neighbors
for (int i = 0; i < g.node_count; i++) {
    int out_count = g.out[i].count;
    if (out_count == 0) continue;
    double share = rank[i] / out_count;  // or weight-proportional
    for (int e = 0; e < out_count; e++) {
        new_rank[g.out[i].edges[e].target] += share;
    }
}
```

For weighted PageRank, distribute proportional to edge weight:
```c
double total_weight = 0;
for (int e = 0; e < out_count; e++) total_weight += g.out[i].edges[e].weight;
for (int e = 0; e < out_count; e++) {
    new_rank[g.out[i].edges[e].target] += rank[i] * g.out[i].edges[e].weight / total_weight;
}
```

### Test Coverage

- `test_pagerank_temporal` — rank changes when time window excludes some edges
- `test_pagerank_weighted` — weighted edges influence rank distribution
- `test_pagerank_direction` — direction param now available

---

## Phase 5: Temporal Utility Functions

Beyond making TVFs time-aware, add scalar/aggregate functions for temporal workflows.

### `temporal_decay(timestamp, half_life_days)` Scalar Function

Returns a decay factor `2^(-age / half_life)` for use in scoring:

```sql
-- Recency-weighted PageRank results
SELECT node, rank * temporal_decay(created_at, 30.0) AS decayed_rank
FROM graph_pagerank
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
ORDER BY decayed_rank DESC;
```

### `graph_edge_timeline` TVF (Future)

Returns the lifecycle of edges between entities:

```sql
SELECT * FROM graph_edge_timeline
WHERE edge_table = 'edges'
  AND src_col = 'src' AND dst_col = 'dst'
  AND node = 'Alice'
  AND valid_at_col = 'valid_at'
  AND invalid_at_col = 'invalid_at'
ORDER BY valid_at;
```

This is the "audit trail" view of how facts about an entity evolved.

---

## Implementation Order & Estimates

| Phase | What | Files Modified | Files Created | Risk |
|-------|------|----------------|---------------|------|
| 1 | BFS/DFS temporal + weight params | `graph_tvf.c` | — | Low (additive) |
| 2 | Shortest path temporal + direction | `graph_tvf.c` | — | Low (additive) |
| 3 | Components → graph_load.c | `graph_tvf.c` | — | Medium (rewrite) |
| 4 | PageRank → graph_load.c + weights | `graph_tvf.c` | — | Medium (rewrite) |
| 5 | temporal_decay scalar | `graph_tvf.c` or new file | — | Low |

### Risk Notes

- **Phases 1–2** are purely additive: existing queries without temporal params work identically
- **Phases 3–4** are rewrites: must preserve exact existing behavior for non-temporal queries. Mitigation: existing Python tests cover all current behavior
- **Phase 5** is independent and can be done at any time

### Dependency Graph

```
Phase 1 (BFS/DFS)  ─┐
Phase 2 (shortest)  ─┤── can be parallel
Phase 3 (components) ┤
Phase 4 (pagerank)  ─┘
Phase 5 (decay fn)  ─── independent
```

All phases are independent of each other. The shared infrastructure (`graph_common.h`, `graph_load.c`) is already in place.

---

## Graphiti Parity Checklist

What muninn enables vs. what Graphiti provides:

| Capability | Graphiti (Neo4j) | muninn (current) | muninn (after this plan) |
|------------|-----------------|-------------------|--------------------------|
| Temporal BFS | Cypher query | No | **Yes** (Phase 1) |
| Temporal DFS | Cypher query | No | **Yes** (Phase 1) |
| Temporal shortest path | Not implemented | No | **Yes** (Phase 2) |
| Temporal components | Not implemented | No | **Yes** (Phase 3) |
| Temporal PageRank | Not implemented | No | **Yes** (Phase 4) |
| Weighted PageRank | Not implemented | No | **Yes** (Phase 4) |
| Temporal centrality | Not implemented | **Yes** | Yes |
| Temporal community | Label propagation | **Yes** (Leiden) | Yes |
| Temporal decay scoring | Custom Python | No | **Yes** (Phase 5) |
| Vector search | External API | **Yes** (HNSW) | Yes |
| Node2Vec embeddings | Not available | **Yes** | Yes |
| Edge invalidation | LLM-driven | App-layer SQL | App-layer SQL |
| Runtime dependency | Neo4j + Python + API keys | **Zero** | Zero |
| Deployment | Server cluster | `.load ./muninn` | `.load ./muninn` |

### What Remains Application-Layer

Graphiti's LLM-driven capabilities (entity extraction, edge invalidation, community summarization) are application concerns. muninn provides the graph engine; a Python/TypeScript wrapper performs the LLM calls and writes results into SQLite tables. The temporal filtering happens transparently at the TVF level.

Example application-layer workflow:
```sql
-- 1. Application inserts a new edge with temporal metadata
INSERT INTO edges (src, dst, fact, valid_at, created_at, weight)
VALUES ('Alice', 'Berlin', 'lives_in', '2024-03-15', datetime('now'), 1.0);

-- 2. Application invalidates contradicted edge
UPDATE edges SET invalid_at = '2024-03-15', expired_at = datetime('now')
WHERE src = 'Alice' AND dst = 'NYC' AND fact = 'lives_in' AND invalid_at IS NULL;

-- 3. Query: "Who was connected to Alice in early 2024?"
SELECT node, depth FROM graph_bfs
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = 'Alice' AND max_depth = 2 AND direction = 'both'
  AND timestamp_col = 'valid_at'
  AND time_start = '2024-01-01' AND time_end = '2024-06-30';

-- 4. "What communities existed before the reorg?"
SELECT node, community_id FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND timestamp_col = 'valid_at'
  AND time_start = '2023-01-01' AND time_end = '2023-12-31';
```

---

## Bi-Temporal Support: Valid Time vs. Transaction Time

Graphiti's full bi-temporal model uses TWO timestamp columns for edge filtering. Our current `graph_load.c` supports a single `timestamp_col` with `time_start`/`time_end` range.

### Option A: Single Timestamp Column (Current)

Users pick which temporal dimension to filter on per query:
```sql
-- Filter by valid_at (event time)
... AND timestamp_col = 'valid_at' AND time_start = '2024-01-01' ...

-- Filter by created_at (transaction time)
... AND timestamp_col = 'created_at' AND time_start = '2024-01-01' ...
```

**Limitation:** Can't filter on both dimensions simultaneously.

### Option B: Dual Timestamp Columns (Enhancement)

Add `valid_at_col`/`invalid_at_col` as alternative parameters to `timestamp_col`:

```sql
-- Point-in-time snapshot: edges valid at a specific moment
... AND valid_at_col = 'valid_at' AND invalid_at_col = 'invalid_at'
    AND as_of = '2024-06-15' ...
```

Generated WHERE clause:
```sql
WHERE ("valid_at" <= ?1 OR "valid_at" IS NULL)
  AND ("invalid_at" > ?1 OR "invalid_at" IS NULL)
```

### Recommendation

Start with **Option A** (already works). Add **Option B** in a follow-up when there's a concrete use case driving it. The single-column approach covers 80% of temporal queries, and users can pre-filter with a VIEW for the rest:

```sql
-- Application creates a temporal view
CREATE VIEW edges_at_june AS
SELECT * FROM edges
WHERE (valid_at IS NULL OR valid_at <= '2024-06-15')
  AND (invalid_at IS NULL OR invalid_at > '2024-06-15');

-- TVFs query the view
SELECT * FROM graph_bfs
WHERE edge_table = 'edges_at_june' AND ...;
```
