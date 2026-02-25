# Phase 3: Components Shadow Table

**Date:** 2026-02-24
**Status:** Plan (not started)
**Depends on:** [Phase 1-2: GII Core + SSSP](./01_gii_sssp_session_kg.md)
**Downstream:** [Phase 4: Communities Shadow Table](./04_communities_shadow_tables.md) (warm-start seeding)

---

## 1. Overview

### What This Phase Delivers

A `{name}_components` shadow table on `graph_adjacency` that caches connected component
assignments. When the `features` parameter includes `'components'`, the adjacency VT
creates and maintains this shadow table alongside its CSR cache. The `graph_components`
TVF detects the shadow table and reads from it when fresh, avoiding both the SQL scan
and the Union-Find computation.

### Why Cache an O(V+E) Computation?

Union-Find with path compression and union-by-rank is nearly O(V+E) -- the inverse
Ackermann factor is negligible. The computation itself is cheap. What is *not* cheap is
the data loading path that precedes it:

1. **SQL scan cost.** The current `graph_components` TVF executes
   `SELECT src, dst FROM edge_table` and builds a string-keyed Union-Find from scratch.
   For a 100K-edge graph, this means 100K `sqlite3_column_text()` calls, 100K `strcmp()`
   lookups in a linear-scan array (`uf_find_or_add` is O(N) per call), and 200K `strdup()`
   allocations. The current `uf_find_or_add` is O(N) -- not hash-mapped -- making the
   full load path O(V*E) in the worst case.

2. **Redundant recomputation.** Components do not change unless edges change. A query
   that calls `graph_components` twice on the same unchanged graph pays the full cost
   twice. With a shadow table, the second call is a trivial `SELECT` over V rows.

3. **Downstream metadata.** The shadow table stores pre-computed `component_size` per
   node and summary statistics (`num_components`, `largest_component`) in `_config`.
   These are available without any computation at all.

4. **Leiden warm-start seeding (Phase 4).** Connected component IDs provide a natural
   initial partition for the Leiden algorithm. Instead of starting with V singleton
   communities, Leiden starts with K components (where K << V for connected graphs)
   and only needs to refine within each component. Cross-component edges do not exist
   by definition, so inter-component modularity is zero. This can reduce early Leiden
   iterations significantly.

---

## 2. Shadow Table Schema

### `{name}_components` Table

```sql
CREATE TABLE IF NOT EXISTS "{name}_components" (
    namespace_id INTEGER DEFAULT 0,
    node_idx     INTEGER,
    component_id INTEGER,            -- root index from Union-Find (canonical representative)
    component_size INTEGER,          -- number of nodes in this component
    generation   INTEGER,            -- G_adj at time of computation
    PRIMARY KEY (namespace_id, node_idx)
);
```

**Column semantics:**

| Column | Type | Description |
|--------|------|-------------|
| `namespace_id` | INTEGER | Scope partition key (Phase 1). Default 0 for unscoped graphs. |
| `node_idx` | INTEGER | Integer index into `_nodes` table. Foreign key to `{name}_nodes.idx`. |
| `component_id` | INTEGER | The canonical representative node index for this component. All nodes in the same component share the same `component_id`. Corresponds to the Union-Find root after full path compression. |
| `component_size` | INTEGER | Total number of nodes in this node's component. Denormalized for O(1) lookup. |
| `generation` | INTEGER | The `G_adj` generation counter at the time this row was computed. Used for staleness detection. |

### Summary Statistics in `{name}_config`

The rebuild path writes these keys into the existing `_config` KV table:

```sql
-- Per-namespace summary (namespace_id encoded in key for multi-namespace support)
INSERT OR REPLACE INTO "{name}_config"(key, value) VALUES
    ('components_generation',    '7'),     -- G_comp = G_adj at computation time
    ('num_components',           '42'),    -- total connected components
    ('largest_component',        '8931'),  -- node count of largest component
    ('smallest_component',       '1');     -- node count of smallest (often isolates)
```

For namespaced graphs (Phase 1), keys are prefixed: `'ns:{namespace_id}:components_generation'`, etc.

### Index

The primary key `(namespace_id, node_idx)` provides efficient lookup by node. An
additional covering index accelerates component-level queries:

```sql
CREATE INDEX IF NOT EXISTS "{name}_components_comp_idx"
    ON "{name}_components"(namespace_id, component_id);
```

This allows efficient queries like "list all nodes in component K" and "count components"
without a full table scan.

---

## 3. Union-Find Module

### Current Implementation (graph_tvf.c)

The existing Union-Find in `graph_tvf.c` (lines 1206-1277) operates on string IDs:

```c
typedef struct {
    char **ids;   /* node ID strings, owned */
    int *parent;  /* parent index (self = root) */
    int *rank;    /* union by rank */
    int count;
    int capacity;
} UnionFind;
```

Key problems:
- `uf_find_or_add()` uses **linear scan** (`strcmp` over all IDs) -- O(N) per call
- Owns string copies via `strdup()` -- unnecessary when CSR provides integer indices
- Cannot operate on CSR arrays directly -- requires SQL scan to populate

### New Implementation: CSR-Native Union-Find

**Recommendation: Option (b) -- write a new Union-Find that operates directly on
`CsrArray` integer indices.**

The CSR already has integer node indices (0..V-1) and adjacency information in
`offsets[]`/`targets[]`. A CSR-native Union-Find avoids all string operations and
runs in O(V + E * alpha(V)) time, where alpha is the inverse Ackermann function.

#### Function Signature

```c
/*
 * Compute connected components from forward and reverse CSR arrays.
 *
 * Runs Union-Find with path compression and union-by-rank over all edges
 * in both directions. Direction semantics: an edge u->v in the forward CSR
 * and v->u in the reverse CSR both connect u and v in the same component
 * (components are undirected by definition).
 *
 * Parameters:
 *   fwd            — forward CSR (out-edges)
 *   rev            — reverse CSR (in-edges), may be NULL for undirected
 *   component_out  — caller-allocated array of V ints, receives component ID per node
 *   num_components — receives total number of distinct components
 *
 * Returns 0 on success, -1 on error.
 * component_out[i] is the canonical representative (root) for node i.
 */
int components_from_csr(
    const CsrArray *fwd,
    const CsrArray *rev,
    int *component_out,
    int *num_components
);
```

#### Internal Structure

```c
/* Inline Union-Find on a pre-allocated int array (no heap allocation) */
static int uf_find_root(int *parent, int x) {
    /* Path halving: every other node points to its grandparent */
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static void uf_merge(int *parent, int *rank, int a, int b) {
    int ra = uf_find_root(parent, a);
    int rb = uf_find_root(parent, b);
    if (ra == rb) return;
    /* Union by rank */
    if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) rank[ra]++;
}

int components_from_csr(const CsrArray *fwd, const CsrArray *rev,
                        int *component_out, int *num_components) {
    int32_t V = fwd->node_count;
    int *rank = calloc(V, sizeof(int));
    if (!rank) return -1;

    /* Initialize: each node is its own parent */
    for (int32_t i = 0; i < V; i++)
        component_out[i] = i;

    /* Union over forward edges */
    for (int32_t u = 0; u < V; u++) {
        for (int32_t j = fwd->offsets[u]; j < fwd->offsets[u + 1]; j++) {
            uf_merge(component_out, rank, u, fwd->targets[j]);
        }
    }

    /* Union over reverse edges (if provided and distinct from forward) */
    if (rev && rev != fwd) {
        for (int32_t u = 0; u < rev->node_count; u++) {
            for (int32_t j = rev->offsets[u]; j < rev->offsets[u + 1]; j++) {
                uf_merge(component_out, rank, u, rev->targets[j]);
            }
        }
    }

    /* Final path compression pass: canonicalize all roots */
    int count = 0;
    for (int32_t i = 0; i < V; i++) {
        component_out[i] = uf_find_root(component_out, i);
        if (component_out[i] == i) count++;
    }

    *num_components = count;
    free(rank);
    return 0;
}
```

#### Why Not Reuse the Existing Union-Find?

| Aspect | Existing (graph_tvf.c) | New (CSR-native) |
|--------|----------------------|-------------------|
| Node identity | String IDs (`char**`) | Integer indices (`int[]`) |
| Lookup cost | O(N) linear scan per `find_or_add` | O(1) array index |
| Memory allocation | `strdup()` per node, dynamic resize | Single `calloc(V)` for rank |
| Data source | SQL `SELECT` statement | CSR `offsets[]`/`targets[]` arrays |
| String mapping | Inline (owns strings) | Deferred (uses `_nodes` table) |

The existing Union-Find remains in `graph_tvf.c` for the non-adjacency code path
(when `edge_table` is a raw SQL table). The new CSR-native version is used exclusively
when reading from the adjacency VT's shadow tables.

#### File Placement

Two options:
1. **New file `src/graph_components.c` + `src/graph_components.h`** -- clean separation
2. **Inline in `src/graph_adjacency.c`** -- avoids new compilation unit

**Recommendation: Option 1.** The components module is shared between:
- `graph_adjacency.c` (rebuild path writes `_components`)
- `graph_tvf.c` (TVF reads `_components` or calls `components_from_csr`)
- `graph_community.c` (Phase 4 reads component IDs for warm-start)

A separate file avoids circular includes and matches the existing pattern where
`graph_load.c`, `graph_csr.c`, `graph_centrality.c`, and `graph_community.c` are
each independent compilation units.

```
src/
├── graph_components.c    -- components_from_csr(), cache read/write helpers
├── graph_components.h    -- public API
├── graph_adjacency.c     -- calls components rebuild after CSR rebuild
├── graph_tvf.c           -- graph_components TVF reads from cache or computes
└── graph_community.c     -- (Phase 4) reads component IDs for Leiden seeding
```

---

## 4. Generation Counter Protocol

### Generation DAG

```
_delta (edge mutations via triggers)
    │
    ▼
_csr_fwd/_csr_rev + _nodes + _degree    →  G_adj (increments on rebuild)
    │
    ├──→ _sssp          G_sssp tracks G_adj  (Phase 2)
    ├──→ _components    G_comp tracks G_adj  (this phase)
    └──→ _communities   G_comm tracks G_adj  (Phase 4, optionally G_comp)
```

### Staleness Rules

```
G_adj   = generation counter in _config, key "generation"
G_comp  = generation counter in _config, key "components_generation"

FRESH:  G_comp == G_adj  →  _components shadow table is valid
STALE:  G_comp <  G_adj  →  _components must be recomputed from CSR
NEVER:  G_comp == 0      →  components have never been computed
```

### Read Path (TVF Query)

```
1. Read G_adj  from _config WHERE key='generation'
2. Read G_comp from _config WHERE key='components_generation'
3. If G_comp == G_adj:
     → Read from _components shadow table (O(V) row scan)
4. If G_comp < G_adj:
     → Load CSR from shadow tables
     → Run components_from_csr()
     → Write results to _components
     → Update G_comp = G_adj in _config
     → Return results
5. If G_adj == 0:
     → CSR not yet built; fall back to SQL scan + Union-Find
```

### Write Path (Adjacency Rebuild)

When `adj_full_rebuild()` or `adj_incremental_rebuild()` completes:

```
1. G_adj is incremented (already happens in current code)
2. If features includes 'components':
     a. Load rebuilt CSR (already in memory during rebuild)
     b. Run components_from_csr(fwd, rev, ...)
     c. REPLACE INTO _components for all nodes
     d. Update _config: components_generation = G_adj
     e. Update _config: num_components, largest_component, smallest_component
```

This is done inside the `SAVEPOINT adj_rebuild` transaction, so it is atomic with
the CSR rebuild. If the rebuild fails, the components cache rolls back too.

### Lazy vs Eager Rebuild

Two strategies for keeping `_components` in sync:

| Strategy | Rebuild Trigger | Latency Profile | Storage Overhead |
|----------|----------------|-----------------|------------------|
| **Eager** | During `adj_full_rebuild()` | Rebuild is slightly slower; queries are always fast | Always populated |
| **Lazy** | On first `graph_components` TVF query after rebuild | Rebuild is fast; first query pays recompute cost | Populated on demand |

**Recommendation: Eager.** Components are O(V+E) -- the same cost as building the
CSR itself. Adding components computation to the rebuild path increases rebuild time
by at most 2x (in practice much less, since CSR build dominates). The benefit is that
every subsequent TVF query hits the cache without a surprise latency spike.

If `features` does not include `'components'`, the eager rebuild path is skipped
entirely -- no shadow table, no computation, zero overhead.

---

## 5. Integration with `graph_components` TVF

### Detection Logic

The `graph_components` TVF currently receives `edge_table`, `src_col`, `dst_col` as
hidden parameters and always runs the SQL scan path. The integration adds adjacency
detection:

```c
/* In gc_filter(): */
const char *edge_table = (const char *)sqlite3_value_text(argv[0]);

if (is_graph_adjacency(db, edge_table)) {
    /* Check if _components shadow table exists and is fresh */
    int64_t g_adj  = config_get_int(db, edge_table, "generation", 0);
    int64_t g_comp = config_get_int(db, edge_table, "components_generation", 0);

    if (g_comp > 0 && g_comp == g_adj) {
        /* FRESH: read from shadow table */
        return gc_read_from_shadow(db, edge_table, namespace_id, &cur->results);
    } else {
        /* STALE or NEVER: load from adjacency, compute, optionally cache */
        return gc_compute_from_adjacency(db, edge_table, namespace_id, &cur->results);
    }
} else {
    /* Not an adjacency VT: current behavior (SQL scan + Union-Find) */
    return run_components(db, edge_table, src_col, dst_col, &cur->results);
}
```

### Shadow Table Read Path

```c
/*
 * Read pre-computed components from the _components shadow table.
 * Joins with _nodes to map node_idx back to string IDs for the TVF output.
 *
 * SQL:
 *   SELECT n.id, c.component_id, c.component_size
 *   FROM "{name}_components" c
 *   JOIN "{name}_nodes" n ON n.idx = c.node_idx
 *   WHERE c.namespace_id = ?1
 *   ORDER BY c.node_idx
 */
static int gc_read_from_shadow(
    sqlite3 *db,
    const char *vtab_name,
    int namespace_id,
    ComponentResults *results
);
```

### Compute-and-Cache Path

When the shadow table is stale (or does not exist because `features` omits
`'components'`), the TVF computes components via the adjacency VT's CSR:

```c
static int gc_compute_from_adjacency(
    sqlite3 *db,
    const char *vtab_name,
    int namespace_id,
    ComponentResults *results
) {
    /* 1. Load GraphData from adjacency shadow tables */
    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load_from_adjacency(db, vtab_name, &g, &errmsg);
    if (rc != SQLITE_OK) { /* handle error */ }

    /* 2. Build CSR */
    CsrArray fwd, rev;
    csr_build(&g, &fwd, &rev);

    /* 3. Run Union-Find on CSR */
    int *comp = malloc(fwd.node_count * sizeof(int));
    int num_comp;
    components_from_csr(&fwd, &rev, comp, &num_comp);

    /* 4. Compute component sizes */
    int *sizes = calloc(fwd.node_count, sizeof(int));
    for (int i = 0; i < fwd.node_count; i++)
        sizes[comp[i]]++;

    /* 5. Build results (string IDs from g.ids[]) */
    comp_results_init(results);
    for (int i = 0; i < g.node_count; i++) {
        comp_results_add(results, g.ids[i], comp[i], sizes[comp[i]]);
    }

    /* 6. Optionally write to _components if the shadow table exists */
    /* (only if features includes 'components') */

    free(comp);
    free(sizes);
    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
    return SQLITE_OK;
}
```

### Extended Hidden Columns

The TVF schema is extended with optional hidden columns for temporal and namespace
filtering. The new schema:

```sql
CREATE TABLE x(
    node TEXT,
    component_id INTEGER,
    component_size INTEGER,
    edge_table TEXT HIDDEN,      -- required
    src_col TEXT HIDDEN,         -- required
    dst_col TEXT HIDDEN,         -- required
    direction TEXT HIDDEN,       -- optional (default: 'both')
    timestamp_col TEXT HIDDEN,   -- optional (temporal filter)
    time_start TEXT HIDDEN,      -- optional (temporal lower bound)
    time_end TEXT HIDDEN,        -- optional (temporal upper bound)
    namespace INTEGER HIDDEN     -- optional (namespace_id for scoped queries)
);
```

New column indices:

```c
enum {
    GC_COL_NODE = 0,
    GC_COL_COMPONENT_ID,
    GC_COL_COMPONENT_SIZE,
    GC_COL_EDGE_TABLE,       /* hidden, required */
    GC_COL_SRC_COL,          /* hidden, required */
    GC_COL_DST_COL,          /* hidden, required */
    GC_COL_DIRECTION,        /* hidden, optional */
    GC_COL_TIMESTAMP_COL,    /* hidden, optional */
    GC_COL_TIME_START,       /* hidden, optional */
    GC_COL_TIME_END,         /* hidden, optional */
    GC_COL_NAMESPACE,        /* hidden, optional */
};
```

The `xBestIndex` implementation switches from manual constraint handling to
`graph_best_index_common()`, which already handles optional hidden columns with
contiguous `argvIndex` assignment and `idxNum` bitmask.

### Temporal Filtering

When `timestamp_col` is provided, the TVF cannot use the cached `_components` shadow
table (which was computed over the full time range). Instead, it falls back to
computing components on the temporally-filtered graph:

```
1. If timestamp_col is provided:
     → Cannot use _components cache (cache is for full graph)
     → Load graph via graph_data_load() with temporal WHERE clause
     → Run Union-Find on the filtered GraphData
     → Return results (do not cache — temporal queries are ephemeral)

2. If timestamp_col is NOT provided:
     → Use _components cache if fresh (via shadow table or adjacency detection)
```

This matches the behavior of `graph_degree`, `graph_betweenness`, and `graph_closeness`
in `graph_centrality.c`, which load the full graph from the adjacency VT but do not
cache temporally-filtered results.

---

## 6. Integration with Leiden (Phase 4 Preview)

### Why Components Matter for Leiden

The Leiden algorithm starts with an initial partition and iteratively refines it by
moving nodes between communities to maximize modularity. The default initial partition
assigns each node to its own singleton community (V communities for V nodes).

Connected components provide a strictly better initial partition:

1. **Guaranteed correctness.** Nodes in different components can never be in the same
   community (there are zero edges between components). Starting with component-level
   communities enforces this invariant from iteration 0.

2. **Reduced search space.** With K components instead of V singletons, the first
   Leiden iteration processes K communities instead of V. For a graph with one giant
   component and 1000 isolates, this reduces iteration 1 from V moves to K moves.

3. **Independent parallelism.** Each component's Leiden refinement is independent.
   While the current implementation is single-threaded, this structure enables future
   parallelization.

### Seeding Protocol (Phase 4)

Phase 4 will implement the following in `graph_community.c`:

```c
/*
 * If _components shadow table is fresh, use component IDs as initial partition.
 * Otherwise, fall back to singleton initialization.
 */
int *initial_partition = malloc(V * sizeof(int));

int64_t g_comp = config_get_int(db, vtab_name, "components_generation", 0);
int64_t g_adj  = config_get_int(db, vtab_name, "generation", 0);

if (g_comp > 0 && g_comp == g_adj) {
    /* Read component IDs from shadow table */
    read_component_ids(db, vtab_name, namespace_id, initial_partition, V);
} else {
    /* Singleton initialization */
    for (int i = 0; i < V; i++)
        initial_partition[i] = i;
}

leiden_run(graph, initial_partition, ...);
```

This is an optimization, not a hard dependency. Phase 4 works without Phase 3 --
it just starts with singletons. But when Phase 3 is complete, Leiden gets a better
starting point for free.

---

## 7. Implementation Steps

### Step 1: Add `components_from_csr()` Function

**File:** `src/graph_components.c` (new), `src/graph_components.h` (new)

- Implement `components_from_csr()` as described in Section 3
- Implement `components_compute_sizes()` helper that fills a `component_size[]` array
- No SQLite dependency in these functions -- pure algorithm on integer arrays
- Add to `TEST_LINK_SRC` in `scripts/generate_build.py` for test runner linkage

```c
/* graph_components.h */
#ifndef GRAPH_COMPONENTS_H
#define GRAPH_COMPONENTS_H

#include "graph_csr.h"

/*
 * Compute connected components from CSR arrays.
 * component_out: caller-allocated int[fwd->node_count], receives component ID per node.
 * num_components: receives count of distinct components.
 * Returns 0 on success, -1 on error.
 */
int components_from_csr(const CsrArray *fwd, const CsrArray *rev,
                        int *component_out, int *num_components);

/*
 * Compute component sizes from a component assignment array.
 * sizes_out: caller-allocated int[node_count], receives size of each node's component.
 * component: the component assignment from components_from_csr().
 * Also computes summary stats.
 */
void components_compute_sizes(const int *component, int node_count,
                              int *sizes_out, int *largest, int *smallest);

#endif /* GRAPH_COMPONENTS_H */
```

### Step 2: Add `_components` Shadow Table Creation

**File:** `src/graph_adjacency.c`

In `adjacency_create_shadow_tables()`, conditionally create the `_components` table
when the `features` parameter includes `'components'`:

```c
if (features & FEATURE_COMPONENTS) {
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_components\" ("
        "namespace_id INTEGER DEFAULT 0, "
        "node_idx INTEGER, "
        "component_id INTEGER, "
        "component_size INTEGER, "
        "generation INTEGER, "
        "PRIMARY KEY (namespace_id, node_idx))",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    sql = sqlite3_mprintf(
        "CREATE INDEX IF NOT EXISTS \"%w_components_comp_idx\" "
        "ON \"%w_components\"(namespace_id, component_id)",
        name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;
}
```

Also update `drop_shadow_tables()` to include `_components`:

```c
const char *suffixes[] = {
    "_config", "_nodes", "_degree", "_csr_fwd", "_csr_rev", "_delta",
    "_components"  /* Phase 3 */
};
```

### Step 3: Add Components Cache Population in Rebuild Path

**File:** `src/graph_adjacency.c`

In `adj_full_rebuild()`, after storing the CSR and degrees, compute and store components:

```c
/* After store_degrees() succeeds, inside SAVEPOINT adj_rebuild: */

if (vtab->features & FEATURE_COMPONENTS) {
    int *comp = malloc(fwd.node_count * sizeof(int));
    int num_comp;
    if (components_from_csr(&fwd, &rev, comp, &num_comp) == 0) {
        int *sizes = calloc(fwd.node_count, sizeof(int));
        int largest, smallest;
        components_compute_sizes(comp, fwd.node_count, sizes, &largest, &smallest);

        rc = store_components(vtab->db, vtab->vtab_name, 0 /* namespace_id */,
                              comp, sizes, fwd.node_count, vtab->generation);
        free(sizes);

        if (rc == SQLITE_OK) {
            config_set_int(vtab->db, vtab->vtab_name,
                           "components_generation", vtab->generation);
            config_set_int(vtab->db, vtab->vtab_name,
                           "num_components", num_comp);
            config_set_int(vtab->db, vtab->vtab_name,
                           "largest_component", largest);
            config_set_int(vtab->db, vtab->vtab_name,
                           "smallest_component", smallest);
        }
    }
    free(comp);
    if (rc != SQLITE_OK) goto rollback;
}
```

The `store_components()` helper:

```c
/*
 * Write component assignments to the _components shadow table.
 * Uses INSERT OR REPLACE for idempotency within the rebuild transaction.
 */
static int store_components(sqlite3 *db, const char *name, int namespace_id,
                            const int *component, const int *sizes,
                            int node_count, int64_t generation) {
    char *sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w_components\""
        "(namespace_id, node_idx, component_id, component_size, generation) "
        "VALUES (?1, ?2, ?3, ?4, ?5)", name);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    for (int i = 0; i < node_count; i++) {
        sqlite3_bind_int(stmt, 1, namespace_id);
        sqlite3_bind_int(stmt, 2, i);
        sqlite3_bind_int(stmt, 3, component[i]);
        sqlite3_bind_int(stmt, 4, sizes[component[i]]);
        sqlite3_bind_int64(stmt, 5, generation);
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) { sqlite3_finalize(stmt); return rc; }
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}
```

### Step 4: Add Components Cache Read in `graph_components` TVF `xFilter`

**File:** `src/graph_tvf.c`

Modify `gc_filter()` to detect adjacency VT and read from cache:

```c
static int gc_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
    /* ... existing parameter extraction ... */

    GraphVtab *vtab = (GraphVtab *)pCursor->pVtab;

    /* Try adjacency-aware path */
    if (is_graph_adjacency(vtab->db, edge_table)) {
        int64_t g_adj  = config_get_int(vtab->db, edge_table, "generation", 0);
        int64_t g_comp = config_get_int(vtab->db, edge_table,
                                         "components_generation", 0);

        if (!has_temporal && g_comp > 0 && g_comp == g_adj) {
            /* Cache hit: read from _components */
            return gc_read_from_shadow(vtab->db, edge_table,
                                       namespace_id, &cur->results);
        }

        /* Cache miss: compute from adjacency CSR */
        return gc_compute_from_adjacency(vtab->db, edge_table,
                                         namespace_id, &cur->results);
    }

    /* Fallback: raw SQL + Union-Find (current behavior) */
    return run_components(vtab->db, edge_table, src_col, dst_col, &cur->results);
}
```

### Step 5: Add Generation Check Logic

**File:** `src/graph_components.c` (or inline helpers in `graph_adjacency.c`)

```c
/*
 * Check if the _components shadow table is fresh for a given namespace.
 * Returns 1 if fresh (G_comp == G_adj), 0 if stale or nonexistent.
 */
int components_cache_is_fresh(sqlite3 *db, const char *vtab_name) {
    int64_t g_adj  = config_get_int(db, vtab_name, "generation", 0);
    int64_t g_comp = config_get_int(db, vtab_name, "components_generation", 0);
    return (g_comp > 0 && g_comp == g_adj);
}
```

### Step 6: Add Namespace Scoping

**File:** `src/graph_adjacency.c`, `src/graph_tvf.c`

All shadow table operations include `WHERE namespace_id = ?`:

```sql
-- Read:
SELECT n.id, c.component_id, c.component_size
FROM "{name}_components" c
JOIN "{name}_nodes" n ON n.idx = c.node_idx
WHERE c.namespace_id = ?1
ORDER BY c.node_idx;

-- Write (during rebuild):
DELETE FROM "{name}_components" WHERE namespace_id = ?1;
INSERT INTO "{name}_components"(...) VALUES (...);

-- Summary stats use namespaced keys:
components_generation             -- for namespace_id=0 (default)
ns:42:components_generation       -- for namespace_id=42
```

### Step 7: Add Temporal + Adjacency Detection to `graph_components` TVF

**File:** `src/graph_tvf.c`

Update `gc_connect()` to declare the extended schema with temporal and namespace
hidden columns. Update `gc_best_index()` to use `graph_best_index_common()`.
Update `gc_filter()` to parse the new optional parameters from `argv[]` using
the `idxNum` bitmask.

When temporal parameters are present, bypass the cache and compute from the
temporally-filtered graph via `graph_data_load()`.

### Step 8: Update Build System

**File:** `scripts/generate_build.py`

Add `src/graph_components.c` to the auto-discovered sources (or to `TEST_LINK_SRC`
if it needs to be linked into the test runner).

**File:** `test/test_main.c`

Add `extern void test_graph_components(void);` and call it from `main()`.

**File:** `test/test_graph_components.c` (new)

Unit tests for `components_from_csr()` covering:
- Single node (1 component)
- Disconnected pairs (V/2 components)
- Fully connected clique (1 component)
- Star graph (1 component)
- Two cliques connected by a bridge (1 component)
- Two isolated cliques (2 components)

---

## 8. Backward Compatibility

### Feature Flag Gating

| Scenario | `_components` table? | TVF behavior |
|----------|---------------------|-------------|
| `features` omits `'components'` | Not created | TVF works as today (SQL scan + Union-Find) |
| `features` includes `'components'` | Created and maintained | TVF reads from cache when fresh |
| `edge_table` is not a `graph_adjacency` VT | N/A | TVF works as today (SQL scan + Union-Find) |
| `edge_table` is adjacency VT, features omits `'components'` | Not created | TVF loads from adjacency CSR, computes on-the-fly, does not cache |

### API Stability

- The `graph_components` TVF output schema is unchanged: `node TEXT, component_id INTEGER, component_size INTEGER`
- The `component_id` values may differ numerically (Union-Find root indices vs shadow table values) but the grouping is identical: nodes in the same component get the same `component_id`
- New hidden columns (`direction`, `timestamp_col`, `time_start`, `time_end`, `namespace`) are optional and have no effect when omitted
- Existing queries that pass only `edge_table`, `src_col`, `dst_col` continue to work identically

### Upgrade Path

Existing `graph_adjacency` VTs (created before Phase 1 adds feature flags) have no
`features` config key. The absence of this key is treated as `features=''` (no
downstream caches). Users must `DROP` and re-`CREATE` the VT with `features='components'`
to enable the shadow table.

---

## 9. Verification Steps

### Unit Tests (`test/test_graph_components.c`)

```c
/* 1. Basic: components_from_csr matches expected output */
TEST(components_triangle) {
    /* 3-node triangle: 0->1, 1->2, 2->0 */
    /* All in one component */
    ASSERT_EQ_INT(num_components, 1);
    ASSERT_EQ_INT(component[0], component[1]);
    ASSERT_EQ_INT(component[1], component[2]);
}

/* 2. Disconnected: two isolated edges */
TEST(components_disconnected) {
    /* 0->1, 2->3 → 2 components */
    ASSERT_EQ_INT(num_components, 2);
    ASSERT_EQ_INT(component[0], component[1]);
    ASSERT_EQ_INT(component[2], component[3]);
    ASSERT(component[0] != component[2]);
}

/* 3. Isolated nodes: V nodes, 0 edges */
TEST(components_isolates) {
    /* Each node is its own component */
    ASSERT_EQ_INT(num_components, V);
}

/* 4. Component sizes are correct */
TEST(components_sizes) {
    /* 5-clique + 3-clique → sizes 5 and 3 */
    components_compute_sizes(comp, 8, sizes, &largest, &smallest);
    ASSERT_EQ_INT(largest, 5);
    ASSERT_EQ_INT(smallest, 3);
}
```

### Integration Tests (`pytests/test_graph_components.py`)

```python
def test_components_tvf_with_adjacency_matches_without(conn):
    """graph_components via adjacency VT produces same grouping as via raw table."""
    # Create edges table and adjacency VT
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("INSERT INTO edges VALUES ('a','b'),('b','c'),('d','e')")
    conn.execute("""CREATE VIRTUAL TABLE g USING graph_adjacency(
        edge_table='edges', src_col='src', dst_col='dst',
        features='components')""")

    # Components via raw table
    raw = conn.execute("""
        SELECT node, component_id FROM graph_components
        WHERE edge_table='edges' AND src_col='src' AND dst_col='dst'
    """).fetchall()

    # Components via adjacency VT
    adj = conn.execute("""
        SELECT node, component_id FROM graph_components
        WHERE edge_table='g' AND src_col='src' AND dst_col='dst'
    """).fetchall()

    # Same grouping (component_id values may differ, but grouping must match)
    raw_groups = group_by_component(raw)
    adj_groups = group_by_component(adj)
    assert raw_groups == adj_groups

def test_components_cache_hit_is_fast(conn):
    """Second call to graph_components is faster (reads from shadow table)."""
    # ... setup large graph ...
    t1 = time_components_query(conn, 'g')  # first call (may compute)
    t2 = time_components_query(conn, 'g')  # second call (cache hit)
    assert t2 < t1 * 0.5  # at least 2x faster

def test_components_stale_after_edge_insert(conn):
    """Adding an edge between two components merges them after rebuild."""
    # Initial: a-b and c-d are separate components
    conn.execute("INSERT INTO edges VALUES ('a','b'),('c','d')")
    conn.execute("INSERT INTO g(g) VALUES('rebuild')")
    result1 = get_components(conn, 'g')
    assert count_components(result1) == 2

    # Bridge the components
    conn.execute("INSERT INTO edges VALUES ('b','c')")
    conn.execute("INSERT INTO g(g) VALUES('rebuild')")
    result2 = get_components(conn, 'g')
    assert count_components(result2) == 1

def test_components_namespace_independent(conn):
    """Different namespaces have independent component assignments."""
    # ... Phase 1 prerequisite ...
    pass

def test_components_temporal_filter(conn):
    """Components with time filter differ from components without."""
    # ... edges with timestamps ...
    # Full range: 1 component (all connected)
    # Restricted range: 2 components (bridge edge excluded)
    pass
```

---

## 10. Complexity Analysis

### Time Complexity

| Operation | Without Cache | With Cache (fresh) | With Cache (stale) |
|-----------|--------------|-------------------|-------------------|
| Component query | O(V*E) SQL scan + O(N) linear UF lookup | O(V) shadow table read | O(V+E) CSR UF + O(V) write |
| Adjacency rebuild (no features) | unchanged | unchanged | unchanged |
| Adjacency rebuild (features='components') | N/A | N/A | +O(V+E) UF + O(V) INSERT |

Notes:
- "Without Cache" is O(V*E) because the current `uf_find_or_add()` does O(N) linear
  scan for each of 2E calls (one per edge endpoint). With hash-mapped GraphData this
  would be O(V+E), but the current code uses linear search.
- "With Cache (fresh)" is O(V) because it reads V rows from the `_components` table
  joined with `_nodes`. Both are indexed by primary key.
- "With Cache (stale)" is O(V+E) for the CSR-native Union-Find (array-indexed, no
  string operations) plus O(V) for writing V rows to the shadow table.

### Space Complexity

| Resource | Without Cache | With Cache |
|----------|--------------|-----------|
| Shadow table disk | 0 | O(V) per namespace (~20 bytes/row) |
| In-memory during computation | O(V+E) strings + UF arrays | O(V) int arrays (no strings) |
| `_config` entries | 0 | 4 keys per namespace |

For a 100K-node graph: `_components` stores 100K rows at ~20 bytes each = ~2 MB.
This is negligible compared to the CSR shadow tables which store O(E) edge data.

### Estimated Wall-Clock Impact

| Graph Size (V, E) | SQL scan + UF (current) | CSR UF (new, stale) | Shadow read (new, fresh) |
|-------------------|------------------------|--------------------|-----------------------|
| 1K nodes, 5K edges | ~2 ms | ~0.1 ms | ~0.5 ms |
| 10K nodes, 50K edges | ~50 ms | ~1 ms | ~3 ms |
| 100K nodes, 500K edges | ~5 sec (O(N) lookup dominates) | ~10 ms | ~25 ms |
| 1M nodes, 5M edges | impractical | ~100 ms | ~250 ms |

The current implementation's O(N) `uf_find_or_add` makes it quadratic in practice.
Even without caching, switching to CSR-native Union-Find provides a 50-500x speedup
on large graphs.

---

## 11. References

### Algorithms

- **Union-Find with path compression and union-by-rank:**
  Tarjan, R.E. (1975). "Efficiency of a good but not linear set union algorithm."
  *Journal of the ACM*, 22(2), 215-225.
  Nearly O(V+E) -- the inverse Ackermann factor alpha(V) is at most 4 for any
  practical input size (V < 10^80).

- **Path halving optimization:**
  Tarjan, R.E. & van Leeuwen, J. (1984). "Worst-case analysis of set union algorithms."
  *Journal of the ACM*, 31(2), 245-281.
  Path halving (setting parent to grandparent) is simpler than full path compression
  and has the same amortized complexity.

### SQLite Patterns

- **FTS5 shadow table pattern:**
  [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html).
  FTS5 uses shadow tables with generation counters (`_config` table) to track staleness.
  Our `_components` table follows the same pattern: a `_config` KV store holds generation
  counters, and shadow tables are rebuilt when the generation falls behind.

- **Virtual table eponymous-only pattern:**
  [The Virtual Table Mechanism of SQLite](https://sqlite.org/vtab.html).
  The `graph_components` TVF is eponymous-only (`xCreate = NULL`), meaning it exists
  as a built-in TVF without explicit `CREATE VIRTUAL TABLE`. It detects adjacency VTs
  by probing for `{name}_config` shadow tables at query time.

### Related Plan Documents

- [Phase 0: Gap Analysis](./00_gap_analysis.md) -- complete gap analysis and roadmap
- [Phase 1-2: GII Core + SSSP](./01_gii_sssp_session_kg.md) -- prerequisite: namespace support, feature flags, SSSP cache
- [Phase 4: Communities Shadow Table](./04_communities_shadow_tables.md) -- downstream consumer: Leiden warm-start from components

---

**Prev:** [Phase 1-2 — GII Core + SSSP + Session-Log KG](./01_gii_sssp_session_kg.md) | **Next:** [Phase 4 — Communities Shadow Table](./04_communities_shadow_tables.md)
