# Plan: Scope-Partitioned Graph Adjacency

## Motivation

The `graph_adjacency` virtual table builds a single CSR over **all** edges in a source table. This works when the table contains one logical graph. But many real-world edge tables contain **multiple disjoint graphs** distinguished by scope columns:

```
events(uuid, parent_uuid, project_id, session_id, ...)
```

Here `(project_id, session_id)` partitions the edge space — edges within session S1 never reference nodes in session S2. Today, the only workaround is creating a separate `graph_adjacency` VT per scope combination, which is impractical when scopes are dynamic and numerous.

### Design Goal

Allow a single `graph_adjacency` VT to serve multiple disjoint subgraphs, partitioned by user-defined scope columns ("namespace"). Each namespace gets its own CSR partition, node index space, and degree cache. Queries specify a namespace to operate on a single subgraph without loading the entire table.

---

## Current State

### Shadow Tables (No Namespace Awareness)

| Table | Schema | Purpose |
|-------|--------|---------|
| `g_config` | `(key TEXT PK, value TEXT)` | KV config store |
| `g_nodes` | `(idx INTEGER PK, id TEXT UNIQUE)` | Global string ID ↔ integer index |
| `g_degree` | `(idx INTEGER PK, in_deg, out_deg, w_in_deg, w_out_deg)` | Degree sequence |
| `g_csr_fwd` | `(block_id INTEGER PK, offsets BLOB, targets BLOB, weights BLOB)` | Forward CSR blocks |
| `g_csr_rev` | `(block_id INTEGER PK, ...)` | Reverse CSR blocks |
| `g_delta` | `(rowid INTEGER PK, src TEXT, dst TEXT, weight REAL, op INTEGER)` | Change log |

Node indices are globally assigned in insertion order. CSR blocks cover 4096-node chunks (by index). Triggers on the source table feed the delta log.

### Key Limitation

Nodes from different scopes are interleaved across the global index space. A namespace filter returns scattered indices spanning all CSR blocks — you'd load every block regardless of scope.

---

## Design: Scope-Partitioned CSR

### Core Idea

Each unique combination of scope column values (a "namespace") gets:
- Its own **node index space** (0, 1, 2, ... local to that namespace)
- Its own **CSR blocks** (keyed by `(namespace_id, block_id)`)
- Its own **degree cache**
- Its own **delta partition**

When no scope columns are specified, the VT behaves exactly as it does today — a single implicit namespace covers all edges.

---

## Interface Design

### CREATE VIRTUAL TABLE

```sql
-- Without namespace (current behavior, unchanged):
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst',
    weight_col='weight'
);

-- With namespace (new):
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='events', src_col='uuid', dst_col='parent_uuid',
    namespace_cols='project_id,session_id'
);

-- With custom delimiter (new, optional):
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='events', src_col='uuid', dst_col='parent_uuid',
    namespace_cols='project_id,session_id',
    namespace_delimiter='/'
);
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `edge_table` | Yes | — | Source edge table name |
| `src_col` | Yes | — | Source node column |
| `dst_col` | Yes | — | Destination node column |
| `weight_col` | No | NULL | Edge weight column |
| `namespace_cols` | No | NULL | Comma-separated list of scope column names |
| `namespace_delimiter` | No | `\x1F` (ASCII Unit Separator) | Delimiter for composite namespace keys |

**Delimiter rationale:** The default `\x1F` (Unit Separator) is a non-printable ASCII control character that cannot appear in normal text data, eliminating collision risk. Users who prefer readable keys can set `namespace_delimiter='/'` or `namespace_delimiter=':'` — at the cost of ensuring their data doesn't contain the delimiter in scope column values.

All `namespace_cols` column names are validated through `id_validate()`.

### Virtual Table Schema

```sql
-- Without namespace (unchanged):
CREATE TABLE x(
    node TEXT,
    node_idx INTEGER,
    in_degree INTEGER,
    out_degree INTEGER,
    weighted_in_degree REAL,
    weighted_out_degree REAL,
    "g" HIDDEN                    -- command column
);

-- With namespace (extended):
CREATE TABLE x(
    node TEXT,                     -- col 0: string node ID
    node_idx INTEGER,              -- col 1: namespace-local integer index
    in_degree INTEGER,             -- col 2
    out_degree INTEGER,            -- col 3
    weighted_in_degree REAL,       -- col 4
    weighted_out_degree REAL,      -- col 5
    namespace TEXT,                -- col 6: composite namespace key (visible)
    namespace_id INTEGER,          -- col 7: integer namespace ID (visible)
    "g" HIDDEN                    -- col 8: command column
);
```

The `namespace` and `namespace_id` columns are only present when `namespace_cols` is specified at CREATE time. When present, they are **visible** (not hidden) so users can SELECT and filter on them.

### Query Interface

```sql
-- No namespace (unchanged):
SELECT * FROM g;                          -- full scan, all nodes
SELECT * FROM g WHERE node = 'alice';     -- point lookup

-- With namespace:
SELECT * FROM g WHERE namespace = 'proj1/sess1';              -- all nodes in scope
SELECT * FROM g WHERE namespace = 'proj1/sess1' AND node = 'uuid-abc';  -- point lookup in scope
SELECT * FROM g;                                               -- full scan across ALL namespaces

-- Namespace discovery:
SELECT DISTINCT namespace FROM g;         -- list all namespaces
SELECT namespace, COUNT(*) FROM g GROUP BY namespace;  -- node counts per scope
```

### Administrative Commands

```sql
-- Rebuild everything (all namespaces):
INSERT INTO g(g) VALUES('rebuild');

-- Rebuild a single namespace:
INSERT INTO g(g) VALUES('rebuild:proj1/sess1');

-- Incremental rebuild (auto-detects affected namespaces):
INSERT INTO g(g) VALUES('incremental_rebuild');
```

### TVF Integration

When centrality/community TVFs detect a `graph_adjacency` VT as their `edge_table`, they currently call `graph_data_load_from_adjacency()`. With namespace support:

```sql
-- Load only the subgraph for a specific namespace:
SELECT * FROM graph_betweenness(
    'g', 'uuid', 'parent_uuid',
    namespace => 'proj1/sess1'
);

SELECT * FROM graph_leiden(
    'g', 'uuid', 'parent_uuid',
    namespace => 'proj1/sess1'
);
```

**Without namespace param on a namespaced VT**: Error — "graph_adjacency 'g' has namespace_cols; specify namespace parameter to select a subgraph".

**Without namespace param on a non-namespaced VT**: Current behavior (load all edges).

---

## Shadow Table Schema Changes

### `g_config` — Add New Keys

| Key | Example | When Set |
|-----|---------|----------|
| `namespace_cols` | `project_id,session_id` | CREATE (if specified) |
| `namespace_delimiter` | `/` | CREATE (if specified, else `\x1F`) |

Existing keys (`edge_table`, `src_col`, `dst_col`, `weight_col`, `generation`, `block_size`) remain unchanged. `node_count` and `edge_count` move to per-namespace tracking in `g_namespace`.

### `g_namespace` — New Table

```sql
CREATE TABLE g_namespace (
    namespace_id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace_key TEXT UNIQUE NOT NULL,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0
);
```

Only created when `namespace_cols` is specified. Each unique combination of scope values gets a row. The `generation` is per-namespace, enabling independent rebuild tracking.

When no namespace_cols: this table is not created. The existing global `node_count`/`edge_count`/`generation` in `g_config` are used (current behavior).

### `g_nodes` — Add Namespace Column

```sql
-- Without namespace (unchanged):
CREATE TABLE g_nodes (idx INTEGER PRIMARY KEY, id TEXT UNIQUE);

-- With namespace:
CREATE TABLE g_nodes (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx INTEGER NOT NULL,
    id TEXT NOT NULL,
    PRIMARY KEY (namespace_id, idx)
);
CREATE UNIQUE INDEX g_nodes_id ON g_nodes(namespace_id, id);
```

`idx` is **local** to each namespace — every namespace starts at 0. The PK becomes `(namespace_id, idx)` and the uniqueness constraint becomes `(namespace_id, id)`.

When no namespace_cols: `namespace_id` is always 0 and the table behaves identically to before (a single namespace spanning all nodes).

### `g_degree` — Add Namespace Column

```sql
-- Without namespace (unchanged):
CREATE TABLE g_degree (idx INTEGER PRIMARY KEY, in_deg, out_deg, w_in_deg, w_out_deg);

-- With namespace:
CREATE TABLE g_degree (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx INTEGER NOT NULL,
    in_deg INTEGER,
    out_deg INTEGER,
    w_in_deg REAL,
    w_out_deg REAL,
    PRIMARY KEY (namespace_id, idx)
);
```

### `g_csr_fwd` / `g_csr_rev` — Add Namespace Column

```sql
-- Without namespace (unchanged):
CREATE TABLE g_csr_fwd (block_id INTEGER PRIMARY KEY, offsets BLOB, targets BLOB, weights BLOB);

-- With namespace:
CREATE TABLE g_csr_fwd (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id INTEGER NOT NULL,
    offsets BLOB,
    targets BLOB,
    weights BLOB,
    PRIMARY KEY (namespace_id, block_id)
);
```

Each namespace gets its own set of CSR blocks with **namespace-local** target indices. Block 0 of namespace 1 and block 0 of namespace 2 are independent.

### `g_delta` — Add Namespace Key

```sql
-- Without namespace (unchanged):
CREATE TABLE g_delta (rowid INTEGER PRIMARY KEY, src TEXT, dst TEXT, weight REAL, op INTEGER);

-- With namespace:
CREATE TABLE g_delta (
    rowid INTEGER PRIMARY KEY,
    namespace_key TEXT,
    src TEXT,
    dst TEXT,
    weight REAL,
    op INTEGER
);
CREATE INDEX g_delta_ns ON g_delta(namespace_key);
```

The trigger captures scope column values and concatenates them with the delimiter to produce `namespace_key`. This enables per-namespace incremental rebuild — only process deltas matching the namespace being rebuilt.

---

## Trigger Changes

### Current Triggers (No Namespace)

```sql
CREATE TRIGGER g_ai AFTER INSERT ON events BEGIN
    INSERT INTO g_delta(src, dst, weight, op)
    VALUES (NEW."uuid", NEW."parent_uuid", NULL, 1);
END;
```

### New Triggers (With Namespace)

```sql
CREATE TRIGGER g_ai AFTER INSERT ON events BEGIN
    INSERT INTO g_delta(namespace_key, src, dst, weight, op)
    VALUES (
        NEW."project_id" || '/' || NEW."session_id",
        NEW."uuid", NEW."parent_uuid", NULL, 1
    );
END;
```

The delimiter in the trigger SQL is set at CREATE time from the `namespace_delimiter` parameter. Each scope column is accessed via `NEW."col"` and concatenated.

For the DELETE and UPDATE triggers, the pattern is identical but using `OLD."col"` references.

---

## Rebuild Flow Changes

### Full Rebuild (Namespaced)

1. Query distinct namespace keys from source table:
   ```sql
   SELECT DISTINCT "project_id" || '/' || "session_id" AS ns
   FROM "events"
   ```
2. For each namespace key:
   a. `INSERT OR IGNORE INTO g_namespace(namespace_key) VALUES(?)`
   b. Load edges for this namespace:
      ```sql
      SELECT "uuid", "parent_uuid" FROM "events"
      WHERE "project_id" || '/' || "session_id" = ?
      ```
      (Or more efficiently, decompose the key back to individual column EQ predicates)
   c. Build CSR from loaded `GraphData`
   d. Store nodes, degrees, CSR blocks under this `namespace_id`
   e. Update `g_namespace` row with `node_count`, `edge_count`, `generation`
3. Clear delta log

**Optimization**: Rather than concatenating in SQL, use individual column predicates:
```sql
SELECT "uuid", "parent_uuid" FROM "events"
WHERE "project_id" = ?1 AND "session_id" = ?2
```
This leverages any existing indexes on the source table.

### Incremental Rebuild (Namespaced)

1. Group deltas by `namespace_key`:
   ```sql
   SELECT DISTINCT namespace_key FROM g_delta
   ```
2. For each affected namespace, run the existing incremental rebuild logic scoped to that namespace's shadow table partition
3. Clear processed deltas

### Full Rebuild (Non-Namespaced, Unchanged)

Existing behavior: load all edges, build single CSR, store under `namespace_id=0` (implicit). The `g_namespace` table does not exist. `node_count`/`edge_count`/`generation` remain in `g_config`.

---

## xBestIndex / xFilter Changes

### Current xBestIndex

Supports two modes:
- `idxNum=1`: Point lookup (`WHERE node = ?`)
- `idxNum=0`: Full scan

### New xBestIndex (With Namespace)

Uses a bitmask to track which filters are present:

| Bit | Filter | Effect |
|-----|--------|--------|
| 0 | `node = ?` | Point lookup within namespace |
| 1 | `namespace = ?` | Scope to single namespace |
| 2 | `namespace_id = ?` | Scope by integer ID |

Cost estimation:

| idxNum | Meaning | estimatedCost |
|--------|---------|---------------|
| `0b000` (0) | Full scan, all namespaces | 10000.0 |
| `0b010` (2) | Namespace scan | 100.0 |
| `0b011` (3) | Namespace + node point lookup | 1.0 |
| `0b100` (4) | Namespace_id scan | 100.0 |
| `0b101` (5) | Namespace_id + node point lookup | 1.0 |
| `0b001` (1) | Node point lookup, no namespace (ambiguous) | 500.0 |

### New xFilter

When namespace filter is present:
1. Resolve `namespace_key` → `namespace_id` (or use provided `namespace_id` directly)
2. Query scoped shadow tables:
   ```sql
   SELECT n.idx, n.id, COALESCE(d.in_deg, 0), ...
   FROM g_nodes n LEFT JOIN g_degree d
     ON n.namespace_id = d.namespace_id AND n.idx = d.idx
   WHERE n.namespace_id = ?
     AND n.id = ?  -- if node filter present
   ORDER BY n.idx
   ```

When no namespace filter on a namespaced VT (full scan across all namespaces):
```sql
SELECT ns.namespace_key, n.idx, n.id, COALESCE(d.in_deg, 0), ...
FROM g_nodes n
LEFT JOIN g_degree d ON n.namespace_id = d.namespace_id AND n.idx = d.idx
LEFT JOIN g_namespace ns ON n.namespace_id = ns.namespace_id
ORDER BY n.namespace_id, n.idx
```

---

## `graph_data_load_from_adjacency()` Changes

This function is the bridge used by centrality and community TVFs. It needs a namespace parameter:

```c
/* Current signature: */
int graph_data_load_from_adjacency(
    sqlite3 *db, const char *vtab_name, GraphData *g, char **pzErrMsg);

/* New signature: */
int graph_data_load_from_adjacency(
    sqlite3 *db, const char *vtab_name,
    const char *namespace_key,  /* NULL = load all (non-namespaced VT) */
    GraphData *g, char **pzErrMsg);
```

When `namespace_key` is provided:
1. Look up `namespace_id` from `g_namespace WHERE namespace_key = ?`
2. Check delta count for this namespace only:
   ```sql
   SELECT COUNT(*) FROM g_delta WHERE namespace_key = ?
   ```
3. If fresh, load from shadow tables filtered by `namespace_id`:
   ```sql
   SELECT id FROM g_nodes WHERE namespace_id = ? ORDER BY idx
   ```
   And load CSR blocks:
   ```sql
   SELECT block_id, offsets, targets, weights
   FROM g_csr_fwd WHERE namespace_id = ? ORDER BY block_id
   ```
4. If stale, fall back to loading from edge table with scope column predicates

When `namespace_key` is NULL on a non-namespaced VT: current behavior unchanged.

When `namespace_key` is NULL on a namespaced VT: return error.

---

## GraphLoadConfig Extension

To support scope filtering in the bulk-load path (used when CSR is stale or for non-adjacency TVFs):

```c
typedef struct {
    const char *edge_table;
    const char *src_col;
    const char *dst_col;
    const char *weight_col;
    const char *direction;
    const char *timestamp_col;
    sqlite3_value *time_start;
    sqlite3_value *time_end;

    /* Scope filtering (new) */
    int scope_count;                 /* 0 = no scope filter */
    const char **scope_cols;         /* validated column names */
    const char **scope_vals;         /* values to match (EQ) */
} GraphLoadConfig;
```

Generated SQL with scope:
```sql
SELECT "uuid", "parent_uuid" FROM "events"
WHERE "project_id" = ?1 AND "session_id" = ?2
  AND ("timestamp_col" >= ?3 OR ?3 IS NULL)
  AND ("timestamp_col" <= ?4 OR ?4 IS NULL)
```

Each `scope_cols[i]` passes through `id_validate()`. Values are bound as parameters (injection-safe).

---

## TVF Hidden Column Extension

All graph TVFs that use `graph_load.c` (centrality, community) gain scope hidden columns. Using the existing two-pass bitmask pattern from `graph_common.h`:

```sql
-- graph_degree with scope:
SELECT * FROM graph_degree(
    'events', 'uuid', 'parent_uuid',
    scope1_col => 'project_id', scope1_val => 'proj1',
    scope2_col => 'session_id', scope2_val => 'sess1'
);

-- Or when using a graph_adjacency VT:
SELECT * FROM graph_degree(
    'g', 'uuid', 'parent_uuid',
    namespace => 'proj1/sess1'
);
```

New hidden columns appended to each centrality/community TVF:

| Column | Type | Description |
|--------|------|-------------|
| `scope1_col` | TEXT | First scope column name |
| `scope1_val` | TEXT | First scope column value |
| `scope2_col` | TEXT | Second scope column name |
| `scope2_val` | TEXT | Second scope column value |
| `namespace` | TEXT | Shorthand for graph_adjacency namespace lookup |

The `namespace` column is syntactic sugar: when the `edge_table` is a `graph_adjacency` VT, `namespace` triggers `graph_data_load_from_adjacency()` with the namespace key. The `scope*` columns are used for direct edge table queries.

---

## Implementation Phases

### Phase 0: Groundwork — Extend `GraphLoadConfig` with Scope Filtering

**Files modified:** `graph_load.h`, `graph_load.c`

Add `scope_count`, `scope_cols[]`, `scope_vals[]` to `GraphLoadConfig`. Extend `graph_data_load()` SQL construction to append `AND "col" = ?N` for each scope column.

**Test:** Python test loading edges from a multi-scope table with scope filter.

**Risk:** Low. Additive change. Existing calls pass `scope_count=0`, zero behavior change.

### Phase 1: Namespace-Aware Shadow Tables

**Files modified:** `graph_adjacency.c`, `graph_adjacency.h`

- Extend `AdjParams` and `parse_adjacency_params()` with `namespace_cols` and `namespace_delimiter`
- Extend `AdjVtab` struct to carry `namespace_cols`, `namespace_col_count`, `namespace_delimiter`
- Conditionally create `g_namespace` table and namespace-aware schema variants of `g_nodes`, `g_degree`, `g_csr_fwd`, `g_csr_rev`, `g_delta`
- Store `namespace_cols` and `namespace_delimiter` in `g_config`

**Test:** Create VT with and without namespace_cols. Verify shadow table schemas.

**Risk:** Medium. Schema branching adds complexity. Must preserve backward compat.

### Phase 2: Namespace-Aware Triggers

**Files modified:** `graph_adjacency.c`

- Extend `install_triggers()` to include scope column capture in delta inserts
- Concatenate `NEW."col1" || delimiter || NEW."col2"` to produce `namespace_key`
- Index `g_delta(namespace_key)` for efficient per-namespace delta queries

**Test:** Insert/update/delete rows in source table. Verify delta entries have correct namespace_key.

**Risk:** Low. The trigger SQL construction is straightforward string formatting.

### Phase 3: Namespace-Aware Full Rebuild

**Files modified:** `graph_adjacency.c`

- New function `adj_full_rebuild_namespaced()`: queries distinct scope combos, loads per-namespace, stores per-namespace
- `adj_full_rebuild()` routes to namespaced path when `namespace_cols` is set
- Support `INSERT INTO g(g) VALUES('rebuild:ns_key')` for single-namespace rebuild

**Test:** Full rebuild on multi-scope table. Verify each namespace has correct node count, edge count, CSR blocks.

**Risk:** Medium. Most complex phase. The per-namespace SQL query construction needs the scope column decomposition logic.

### Phase 4: Namespace-Aware Incremental Rebuild

**Files modified:** `graph_adjacency.c`

- Group deltas by `namespace_key`
- For each affected namespace: load its CSR blocks, apply deltas, store back
- Unaffected namespaces are untouched

**Test:** Insert edges in one namespace. Verify only that namespace's blocks are rebuilt. Other namespaces' generation unchanged.

**Risk:** Medium. Incremental rebuild is already the most complex code path; adding namespace partitioning compounds this.

### Phase 5: Namespace-Aware Query Interface

**Files modified:** `graph_adjacency.c`

- Extend VT schema to include `namespace` and `namespace_id` columns when namespaced
- Extend `adj_xBestIndex()` with bitmask for namespace filters
- Extend `adj_xFilter()` to scope shadow table queries by namespace
- Extend `graph_data_load_from_adjacency()` with namespace parameter

**Test:** Point lookup with namespace filter. Full scan with and without namespace. Degree queries scoped to namespace match expected values.

**Risk:** Low-Medium. The xBestIndex bitmask pattern is well-established in `graph_common.h`.

### Phase 6: TVF Scope Integration

**Files modified:** `graph_centrality.c`, `graph_community.c`, `graph_common.h`

- Add `scope1_col`/`scope1_val`/`scope2_col`/`scope2_val`/`namespace` hidden columns to centrality and community TVFs
- Extend `graph_best_index_common()` bitmask to cover new columns
- In xFilter: populate `GraphLoadConfig.scope_*` from argv, or call `graph_data_load_from_adjacency()` with namespace

**Test:** Run betweenness/closeness/leiden on a namespaced table both directly (scope_cols) and via graph_adjacency VT (namespace).

**Risk:** Low. Follows exact same pattern as temporal column addition.

---

## Dependency Graph

```
Phase 0 (GraphLoadConfig scope) ──┬── Phase 1 (shadow table schema)
                                  │         │
                                  │   Phase 2 (triggers)
                                  │         │
                                  │   Phase 3 (full rebuild)
                                  │         │
                                  │   Phase 4 (incremental rebuild)
                                  │         │
                                  │   Phase 5 (query interface)
                                  │
                                  └── Phase 6 (TVF scope columns)
```

Phases 1–5 are sequential (each builds on the prior). Phase 0 and Phase 6 can proceed in parallel with Phases 1–5.

---

## Non-Namespaced Behavior: Zero Regression

When `namespace_cols` is NOT specified at CREATE time:

| Aspect | Behavior |
|--------|----------|
| `g_namespace` table | Not created |
| `g_nodes` schema | `(idx INTEGER PK, id TEXT UNIQUE)` — unchanged |
| `g_degree` schema | `(idx INTEGER PK, ...)` — unchanged |
| `g_csr_*` schema | `(block_id INTEGER PK, ...)` — unchanged |
| `g_delta` schema | `(rowid INTEGER PK, src, dst, weight, op)` — unchanged |
| VT output columns | 6 visible + 1 hidden — unchanged |
| Triggers | No namespace_key capture — unchanged |
| `graph_data_load_from_adjacency(db, name, NULL, &g, &err)` | Loads all nodes — unchanged |
| Full rebuild | Single pass, all edges — unchanged |
| Incremental rebuild | Single partition — unchanged |
| xBestIndex | `node = ?` or full scan — unchanged |

The implementation uses `if (vtab->namespace_col_count > 0)` guards at every branching point. Non-namespaced VTs never touch namespace-aware code paths.

---

## Open Questions

1. **Cross-namespace queries**: Should `graph_data_load_from_adjacency()` support loading multiple namespaces (e.g., `namespace IN ('a', 'b')`)? Deferred — start with single-namespace-or-all semantics.

2. **Namespace auto-discovery for TVFs**: When a TVF receives a `graph_adjacency` VT name and a `namespace` param, should it validate that the namespace exists before loading? Yes — return a clear error rather than an empty graph.

3. **Namespace deletion**: Should there be a command to drop a namespace's shadow data? e.g., `INSERT INTO g(g) VALUES('drop_namespace:proj1/sess1')`. Useful when sessions are archived. Deferred to a follow-up.

4. **Maximum scope columns**: The `scope1_col`/`scope2_col` TVF parameters support up to 2 scope dimensions. Should we support more? 2 covers the vast majority of cases. Users needing 3+ can pre-concatenate into a single column in their source table.
