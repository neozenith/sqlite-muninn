# API Reference

Complete reference for all functions and virtual tables in the muninn extension.

## HNSW Virtual Table (`hnsw_index`)

Based on the [Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320) algorithm (Malkov & Yashunin, 2018).

```sql
CREATE VIRTUAL TABLE name USING hnsw_index(
    dimensions=N,            -- vector dimensionality (required)
    metric='l2',             -- 'l2' | 'cosine' | 'inner_product'
    m=16,                    -- max connections per node per layer
    ef_construction=200      -- beam width during index construction
);
```

**Columns:**

| Column | Type | Hidden | Description |
|--------|------|--------|-------------|
| `rowid` | INTEGER | Yes | User-assigned ID for joining with application tables |
| `vector` | BLOB | No | `float32[dim]` — input for INSERT, MATCH constraint for search |
| `distance` | REAL | No | Computed distance (output only, during search) |
| `k` | INTEGER | Yes | Top-k parameter (search constraint) |
| `ef_search` | INTEGER | Yes | Search beam width (search constraint) |

**Operations:**

```sql
-- Insert
INSERT INTO t (rowid, vector) VALUES (42, ?blob);

-- KNN search
SELECT rowid, distance FROM t WHERE vector MATCH ?query AND k = 10;

-- Point lookup
SELECT vector FROM t WHERE rowid = 42;

-- Delete (with automatic neighbor reconnection)
DELETE FROM t WHERE rowid = 42;

-- Drop (removes index and all shadow tables)
DROP TABLE t;
```

**Shadow tables** (auto-managed): `{name}_config`, `{name}_nodes`, `{name}_edges`.

---

## Graph Traversal TVFs

All graph TVFs work on **any** existing SQLite table with source/target columns. Table and column names are validated against SQL injection.

### Common Constraints

All graph TVFs accept these WHERE-clause constraints:

| Constraint | Type | Required | Description |
|-----------|------|----------|-------------|
| `edge_table` | TEXT | Yes | Name of the edge table |
| `src_col` | TEXT | Yes | Source column name |
| `dst_col` | TEXT | Yes | Destination column name |
| `weight_col` | TEXT | No | Weight column for weighted operations |
| `direction` | TEXT | No | `'forward'` (default), `'reverse'`, or `'both'` |
| `timestamp_col` | TEXT | No | Column for temporal filtering |
| `time_start` | TEXT | No | Start of time window (ISO 8601) |
| `time_end` | TEXT | No | End of time window (ISO 8601) |

### `graph_bfs` / `graph_dfs`

Breadth-first or depth-first traversal from a start node.

```sql
SELECT node, depth, parent FROM graph_bfs
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = 'node-42' AND max_depth = 5
  AND direction = 'forward';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `depth` | INTEGER | Hop distance from start |
| `parent` | TEXT | Parent node in traversal tree (NULL for start) |

Additional constraints: `start_node` (required), `max_depth` (default: unlimited).

### `graph_shortest_path`

Unweighted (BFS) or weighted (Dijkstra) shortest path.

```sql
SELECT node, distance, path_order FROM graph_shortest_path
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = 'A' AND end_node = 'Z' AND weight_col = 'weight';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node on the path |
| `distance` | REAL | Cumulative distance from start |
| `path_order` | INTEGER | Position in path (0-indexed) |

Additional constraints: `start_node` (required), `end_node` (required).

### `graph_components`

Connected components via Union-Find with path compression.

```sql
SELECT node, component_id, component_size FROM graph_components
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `component_id` | INTEGER | Component index (0-based) |
| `component_size` | INTEGER | Number of nodes in this component |

### `graph_pagerank`

Iterative power method [PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) (Page, Brin, Motwani & Winograd, 1999) with configurable damping and iterations.

```sql
SELECT node, rank FROM graph_pagerank
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND damping = 0.85 AND iterations = 20;
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `rank` | REAL | PageRank score (sums to ~1.0) |

Additional constraints: `damping` (default 0.85), `iterations` (default 20).

---

## Centrality TVFs

### `graph_degree`

Degree centrality for all nodes in the graph.

```sql
SELECT node, in_degree, out_degree, degree, centrality FROM graph_degree
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `in_degree` | REAL | Count (or weighted sum) of incoming edges |
| `out_degree` | REAL | Count (or weighted sum) of outgoing edges |
| `degree` | REAL | Total degree (in + out) |
| `centrality` | REAL | Degree centrality (raw or normalized) |

Additional constraints: `normalized` (default 0; set to 1 to divide by N-1).

### `graph_betweenness`

Betweenness centrality via [Brandes' O(VE) algorithm](https://doi.org/10.1080/0022250X.2001.9990249) (Brandes, 2001). Identifies bridge nodes that lie on many shortest paths.

```sql
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both'
ORDER BY centrality DESC;
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `centrality` | REAL | Betweenness centrality score |

Additional constraints: `normalized` (default 0; set to 1 for values in [0, 1]).

!!! tip "When to use betweenness"
    Betweenness centrality is ideal for finding **bridge** concepts in knowledge graphs — nodes that connect otherwise separate clusters. In a GraphRAG workflow, these bridge nodes provide the most valuable context for retrieval.

### `graph_closeness`

Closeness centrality with [Wasserman-Faust](https://doi.org/10.1017/CBO9780511815478) normalization (Wasserman & Faust, 1994) for disconnected graphs.

```sql
SELECT node, centrality FROM graph_closeness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `centrality` | REAL | Closeness centrality score |

---

## Community Detection TVFs

### `graph_leiden`

Community detection via the [Leiden algorithm](https://arxiv.org/abs/1810.08473) (Traag, Waltman & van Eck, 2019). Produces well-connected communities with guaranteed connectivity — an improvement over Louvain.

```sql
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

| Output Column | Type | Description |
|---------------|------|-------------|
| `node` | TEXT | Node identifier |
| `community_id` | INTEGER | Community assignment (0-based, contiguous) |
| `modularity` | REAL | Global modularity score of the partition |

Additional constraints: `resolution` (default 1.0; higher values produce more, smaller communities).

!!! tip "When to use Leiden"
    Leiden community detection is used by Microsoft GraphRAG for hierarchical retrieval. Detect communities, compute supernode embeddings (mean of member vectors), then search supernodes first before drilling into the matching community.

---

## Graph Adjacency Virtual Table (`graph_adjacency`)

A persistent graph adjacency index backed by [Compressed Sparse Row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) arrays stored in shadow tables. The CSR format — originating from the Yale Sparse Matrix Package (Eisenstat et al., 1977) — provides O(1) neighbor access per node. muninn extends standard CSR with [blocked storage](https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf) (4,096-node blocks) for incremental updates. Provides pre-computed degree information and serves as a fast cache for graph algorithm TVFs.

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges',        -- source edge table (required)
    src_col='src',             -- source column name (required)
    dst_col='dst',             -- destination column name (required)
    weight_col='weight'        -- weight column (optional)
);
```

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node identifier |
| `node_idx` | INTEGER | Internal index (0-based, sequential) |
| `in_degree` | INTEGER | Number of incoming edges |
| `out_degree` | INTEGER | Number of outgoing edges |
| `weighted_in_degree` | REAL | Sum of incoming edge weights |
| `weighted_out_degree` | REAL | Sum of outgoing edge weights |

**Querying:**

```sql
-- Full degree sequence
SELECT node, in_degree, out_degree FROM g ORDER BY out_degree DESC;

-- Point lookup
SELECT * FROM g WHERE node = 'alice';
```

**Auto-rebuild:** Triggers on the edge table track INSERT/UPDATE/DELETE operations. The CSR cache is lazily rebuilt on the next query — incremental for small deltas (< 10%), full rebuild for large changes.

**Administrative commands:**

```sql
-- Force a full rebuild
INSERT INTO g(g) VALUES ('rebuild');

-- Force an incremental rebuild
INSERT INTO g(g) VALUES ('incremental_rebuild');
```

**Shadow tables** (auto-managed): `{name}_config`, `{name}_nodes`, `{name}_degree`, `{name}_csr_fwd`, `{name}_csr_rev`, `{name}_delta`.

!!! tip "When to use graph_adjacency"
    Create a `graph_adjacency` virtual table when you query graph algorithms repeatedly on the same edge table. The CSR cache eliminates redundant edge-table scans — the larger the graph and the more queries you run, the greater the speedup. For one-off queries on small graphs, the plain TVFs are simpler and equally fast.

---

## Graph Select TVF (`graph_select`)

A dbt-inspired node selection function for querying graph lineage — ancestors, descendants, closures, and set operations.

```sql
SELECT node, depth, direction FROM graph_select(
    'edges',      -- edge table name
    'src',        -- source column
    'dst',        -- destination column
    '+C+1'        -- selector expression
);
```

**Output columns:**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Selected node identifier |
| `depth` | INTEGER | Hop distance from the anchor node (0 for self) |
| `direction` | TEXT | `'self'`, `'ancestor'`, or `'descendant'` |

### Selector Expression Syntax

| Syntax | Meaning | Example |
|--------|---------|---------|
| `node` | The node itself | `C` → {C} |
| `+node` | Node and all ancestors | `+C` → {A, B, C} |
| `node+` | Node and all descendants | `C+` → {C, D, E, F} |
| `N+node` | Depth-limited ancestors | `1+E` → {C, Y, E} |
| `node+N` | Depth-limited descendants | `C+1` → {C, D, E} |
| `N+node+M` | Both directions, depth-limited | `1+C+1` → {A, B, C, D, E} |
| `+node+` | All ancestors and descendants | `+C+` → {A, B, C, D, E, F} |
| `@node` | Build closure (descendants + their ancestors) | `@C` → all connected nodes |
| `A B` | Union (space-separated) | `A B` → {A, B} |
| `A,B` | Intersection (comma-separated) | `+D,+E` → {A, B, C} |
| `not A` | Complement (all nodes except A's set) | `not C` → 7 nodes |

### Examples

```sql
-- What depends on module C? (all descendants)
SELECT node FROM graph_select('deps', 'src', 'dst', 'C+');

-- What does module C depend on? (all ancestors)
SELECT node FROM graph_select('deps', 'src', 'dst', '+C');

-- Full build closure: what must be rebuilt if C changes?
SELECT node FROM graph_select('deps', 'src', 'dst', '@C');

-- Common ancestors of D and E
SELECT node FROM graph_select('deps', 'src', 'dst', '+D,+E');

-- Everything within 2 hops of C in both directions
SELECT node FROM graph_select('deps', 'src', 'dst', '2+C+2');
```

!!! tip "When to use graph_select"
    Use `graph_select` for **dependency analysis** — build systems, data pipeline lineage, impact analysis, and dead code detection. The selector syntax is modeled after [dbt's node selection](https://docs.getdbt.com/reference/node-selection/syntax), making it familiar to data engineers.

---

## Node2Vec

### `node2vec_train()`

Learn vector embeddings from graph structure using biased random walks ([Grover & Leskovec, 2016](https://arxiv.org/abs/1607.00653)) and [Skip-gram with Negative Sampling (SGNS)](https://arxiv.org/abs/1310.4546) (Mikolov et al., 2013).

```sql
SELECT node2vec_train(
    edge_table,       -- name of edge table
    src_col,          -- source column name
    dst_col,          -- destination column name
    output_table,     -- HNSW table to store embeddings (must exist)
    dimensions,       -- embedding size (must match HNSW table)
    p,                -- return parameter (1.0 = uniform/DeepWalk)
    q,                -- in-out parameter (1.0 = uniform/DeepWalk)
    num_walks,        -- walks per node
    walk_length,      -- max steps per walk
    window_size,      -- SGNS context window
    negative_samples, -- negative samples per positive
    learning_rate,    -- initial learning rate (decays linearly)
    epochs            -- training epochs
);
-- Returns: number of nodes embedded
```

**p, q parameter guide:**

| Setting | Walk Behavior | Best For |
|---------|--------------|----------|
| p=1, q=1 | Uniform ([DeepWalk](https://arxiv.org/abs/1403.6652)) | General structural similarity |
| Low p (0.25) | BFS-like, stays local | Community/cluster detection |
| Low q (0.5) | DFS-like, explores far | Structural role similarity |
