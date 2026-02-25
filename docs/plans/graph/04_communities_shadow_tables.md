# Phase 4: Communities Shadow Table with Warm-Start

**Date:** 2026-02-24
**Status:** Planned
**Depends on:** Phase 1 (Scoped Adjacency VT), Phase 3 (Components Shadow Table)
**Delivers:** Cached Leiden community detection partition in a shadow table, with warm-start from cached partition on incremental rebuild and optional component seeding on cold start.

---

## 1. Overview

### What This Phase Delivers

The Leiden algorithm (Traag et al., 2019) is the most computationally expensive graph analytic in the muninn extension. It iterates three phases -- local moving, refinement, aggregation -- until convergence, typically requiring 5-20 outer iterations. Each invocation of `graph_leiden` currently recomputes the partition from scratch, even when the underlying graph has not changed or has changed minimally.

This phase adds a `_communities` shadow table to the `graph_adjacency` virtual table, enabled by the `communities` feature flag. The shadow table caches the most recent Leiden partition, keyed by resolution parameter and scoped by namespace. Three access patterns are supported:

1. **Cache hit** -- the adjacency CSR has not changed since the last Leiden run and the requested resolution matches. The TVF reads directly from the shadow table. Cost: O(V) sequential read.

2. **Warm-start rebuild** -- the adjacency CSR has changed (new edges added, edges deleted) but the resolution matches. The cached partition seeds the Leiden algorithm: unchanged nodes retain their community assignment, changed nodes are reset to singletons. Cost: O(V'E' x iter) where V' is the set of changed nodes and their neighbors.

3. **Cold-start rebuild** -- either the cache is empty or the requested resolution differs from the cached resolution. The algorithm starts from singletons (or component IDs if the Phase 3 `_components` cache is available and fresh). Cost: O(VE x iter).

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Cold start (singletons) | O(VE x iter), iter in [5, 20] | O(V) for community[] + O(V) sum_tot |
| Cold start (component-seeded) | O(VE x iter), iter in [3, 15] (fewer iterations) | O(V) |
| Warm start (partial reset) | O(V'E' x iter), V' = changed neighborhood | O(V) |
| Cache hit (read from shadow) | O(V) sequential scan | O(V) result buffer |
| Shadow table write | O(V) INSERT/UPDATE | O(V) disk |

Where V' << V for incremental changes. The Dynamic Leiden paper (arXiv:2405.11658) demonstrates 1.1-1.4x speedup from warm-start on real-world networks with 1-10% edge churn.

### Warm-Start Justification

The Leiden algorithm's local moving phase (Phase 1) iterates over all nodes, evaluating the modularity gain of moving each node to neighboring communities. For a warm-started partition where 95% of nodes are in stable communities:

- Nodes whose neighborhoods have not changed will evaluate their current community assignment, find no improvement, and make zero moves in the first pass.
- Only nodes in changed neighborhoods will actively explore moves.
- The number of outer iterations is typically lower because the partition is already close to optimal.

This is the core insight of Dynamic Leiden: instead of restarting from V singletons, restart from V-V' stable assignments + V' singletons, where V' is the set of nodes whose local neighborhood changed.

---

## 2. Shadow Table Schema

### `{name}_communities` Table

```sql
CREATE TABLE IF NOT EXISTS "{name}_communities" (
    namespace_id  INTEGER DEFAULT 0,
    node_idx      INTEGER NOT NULL,
    community_id  INTEGER NOT NULL,
    PRIMARY KEY (namespace_id, node_idx)
);
```

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| `namespace_id` | INTEGER | Scope partition key (Phase 1). Default 0 for unscoped graphs. |
| `node_idx` | INTEGER | Index into `_nodes` table. Foreign key to `{name}_nodes.idx`. |
| `community_id` | INTEGER | Contiguous community label (0..K-1) from the Leiden partition. |

The resolution, modularity, generation, and community count are stored in the `_config` table rather than denormalized into every row. This keeps the row width minimal (3 integers per node) and avoids redundant storage of per-partition metadata.

### `{name}_config` Entries

```sql
-- Stored by the communities rebuild path:
INSERT OR REPLACE INTO "{name}_config"(key, value) VALUES
    ('communities_generation',  '<int64>'),   -- G_comm: generation when communities were computed
    ('communities_resolution',  '<double>'),  -- resolution parameter used for this partition
    ('communities_modularity',  '<double>'),  -- final modularity Q of the cached partition
    ('num_communities',         '<int>');      -- K: number of distinct communities
```

**Key semantics:**

| Config Key | Type | Description |
|------------|------|-------------|
| `communities_generation` | int64 | The adjacency generation (G_adj) at which this partition was computed. Stale when G_comm < G_adj. |
| `communities_resolution` | double | The gamma parameter used. Cache miss if requested resolution differs. |
| `communities_modularity` | double | Modularity Q of the cached partition. Informational and used for warm-start quality validation. |
| `num_communities` | int | Number of distinct communities K. Informational. |

### Resolution as Cache Key

The resolution parameter (gamma) fundamentally changes the Leiden partition. Higher resolution produces more, smaller communities; lower resolution produces fewer, larger communities. Two queries with different resolution values will produce entirely different partitions.

Therefore, the cache is valid **only** when the requested resolution matches the cached resolution. Comparison uses epsilon tolerance:

```c
static int resolution_matches(double cached, double requested) {
    return fabs(cached - requested) < 1e-10;
}
```

If the resolution does not match, the cache is fully invalidated and a cold-start rebuild is performed. The new partition (with the new resolution) replaces the cached partition entirely.

---

## 3. Warm-Start Design

### Modified Algorithm Entry Point

The current `run_leiden()` function initializes every node as a singleton:

```c
/* Current: always starts from singletons */
static double run_leiden(const GraphData *g, int *community,
                         double resolution, const char *direction);
```

The warm-start variant accepts an initial partition and a set of changed nodes:

```c
/*
 * Run Leiden with warm-start from a cached partition.
 *
 * @param g             Graph data (adjacency lists)
 * @param community     IN/OUT: initial partition on entry, final partition on exit.
 *                      community[i] = cached community ID for unchanged nodes,
 *                      community[i] = i (singleton) for changed nodes.
 * @param resolution    Modularity resolution parameter (gamma)
 * @param direction     "both", "forward", or "reverse"
 * @param changed_nodes Array of node indices whose neighborhoods changed
 * @param n_changed     Length of changed_nodes array
 * @return              Final modularity Q
 *
 * When n_changed == 0, this is equivalent to run_leiden() but skips
 * initialization (the cached partition is already loaded).
 * When changed_nodes == NULL, falls back to singleton initialization
 * (equivalent to cold start).
 */
static double run_leiden_warm(const GraphData *g, int *community,
                              double resolution, const char *direction,
                              const int *changed_nodes, int n_changed);
```

### Warm-Start Initialization

When the adjacency generation has advanced (G_comm < G_adj) but the resolution matches:

1. **Load cached partition** from `_communities` shadow table into `community[]` array.

2. **Identify changed nodes** from the `_delta` table. A node is "changed" if it appears as either `src` or `dst` in any delta operation since the last communities rebuild:

   ```sql
   SELECT DISTINCT n.idx FROM "{name}_delta" d
   JOIN "{name}_nodes" n ON n.id = d.src OR n.id = d.dst;
   ```

   The result is the set of directly affected nodes. Optionally, extend to 1-hop neighbors for better convergence:

   ```c
   /* Extend changed set to include 1-hop neighbors */
   static void extend_changed_to_neighbors(const GraphData *g,
                                            int *changed_bitmap, int N,
                                            const int *changed_nodes,
                                            int n_changed);
   ```

3. **Reset changed nodes to singletons** within the loaded partition:

   ```c
   /* Reset changed nodes to singleton communities.
    * Uses max_community+1..max_community+n_changed as new IDs
    * to avoid collisions with existing community IDs. */
   int max_comm = 0;
   for (int i = 0; i < N; i++)
       if (community[i] > max_comm) max_comm = community[i];

   for (int i = 0; i < n_changed; i++) {
       int v = changed_nodes[i];
       community[v] = max_comm + 1 + i;  /* unique singleton */
   }
   ```

4. **Run Leiden** on the partially-initialized `community[]` array. The local moving phase will quickly settle unchanged nodes (no better community available) and focus work on the changed neighborhoods.

### Handling New Nodes

When edges are added that reference nodes not present in the cached partition:

- New nodes (not in `_nodes` at the time of the last rebuild) are inherently "changed" -- they have no cached community assignment.
- After loading the cached partition, new nodes get singleton community IDs.
- The `_nodes` table is updated during adjacency rebuild (Phase 1), so by the time the communities warm-start runs, all node indices are available.

### Handling Deleted Nodes

When nodes lose all their edges (become isolated) due to deletions:

- The node still exists in `_nodes` but has an empty adjacency list.
- Leiden will assign it to a singleton community (it has no neighbors to join).
- The old community assignment in the cache is overwritten.

---

## 4. Component Seeding (Optional)

When the `_communities` cache is empty (first run, or after resolution change) and the Phase 3 `_components` cache is fresh, component IDs can seed the initial partition instead of singletons.

### Rationale

The Leiden algorithm cannot move nodes between disconnected components -- there are no edges to evaluate for cross-component modularity gain. Starting from component IDs:

- Gives Leiden a head start: it only needs to refine within components.
- Reduces the number of outer iterations (fewer merges needed from singletons to stable partition).
- Is free to compute: component IDs are already cached in `_components`.

### Implementation

```c
/*
 * Seed community[] from component cache when available.
 * Returns 1 if seeding was applied, 0 if not (caller should use singletons).
 */
static int seed_from_components(sqlite3 *db, const char *vtab_name,
                                int namespace_id, int *community, int N) {
    /* Check if _components exists and is fresh */
    int64_t G_comp = config_get_int(db, vtab_name, "components_generation", -1);
    int64_t G_adj  = config_get_int(db, vtab_name, "generation", 0);
    if (G_comp < G_adj)
        return 0;  /* stale or missing — fall back to singletons */

    /* Load component assignments */
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT node_idx, component_id FROM \"%w_components\" "
        "WHERE namespace_id = %d ORDER BY node_idx",
        vtab_name, namespace_id);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return 0;

    int loaded = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int idx = sqlite3_column_int(stmt, 0);
        int comp = sqlite3_column_int(stmt, 1);
        if (idx >= 0 && idx < N) {
            community[idx] = comp;
            loaded++;
        }
    }
    sqlite3_finalize(stmt);

    return (loaded == N) ? 1 : 0;
}
```

### When Component Seeding Applies

| `_communities` state | `_components` state | Initialization |
|----------------------|---------------------|----------------|
| Fresh (G_comm == G_adj, resolution matches) | Any | Cache hit -- no Leiden run |
| Stale (G_comm < G_adj, resolution matches) | Any | **Warm-start** from cached communities |
| Empty or resolution mismatch | Fresh (G_comp == G_adj) | **Component-seeded** cold start |
| Empty or resolution mismatch | Stale or missing | **Singleton** cold start |

Component seeding is strictly an optimization. It improves convergence speed but does not affect correctness. The Leiden algorithm will converge to the same quality partition regardless of initialization -- only the number of iterations differs.

---

## 5. Resolution Key Management

### Cache Validity Rules

The communities cache is valid when all three conditions hold:

1. `communities_generation` (G_comm) equals the current adjacency generation (G_adj).
2. `communities_resolution` matches the requested resolution (within epsilon 1e-10).
3. The `_communities` table is non-empty.

### Resolution Mismatch Behavior

When the user queries `graph_leiden` with a resolution different from the cached value:

1. The cache is treated as fully invalid (not warm-startable -- different resolution produces structurally different partitions).
2. A cold-start rebuild is performed with the new resolution.
3. The `_communities` table is cleared and repopulated with the new partition.
4. The `communities_resolution` config entry is updated to the new value.
5. `communities_generation` is set to the current G_adj.

This means the cache serves exactly one resolution value at a time per namespace. If a workload frequently alternates between two resolutions, each switch triggers a full recompute. This is an intentional simplification: storing multiple resolution-keyed partitions would complicate the schema, increase storage, and serve a rare use case.

### Config Read/Write Flow

```c
/* Read cached resolution */
static double config_get_double(sqlite3 *db, const char *name,
                                const char *key, double def) {
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT value FROM \"%w_config\" WHERE key='%w'", name, key);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return def;

    double result = def;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (val) result = strtod(val, NULL);
    }
    sqlite3_finalize(stmt);
    return result;
}

/* Write resolution with full precision */
static int config_set_double(sqlite3 *db, const char *name,
                             const char *key, double value) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", value);  /* round-trip precision */
    return config_set(db, name, key, buf);
}
```

The `%.17g` format ensures round-trip fidelity for IEEE 754 doubles, so a cached resolution of 1.0 will compare exactly equal to a requested 1.0 without epsilon issues from serialization.

---

## 6. Generation Counter Protocol

### Generation Counter Relationships

```
G_adj  = generation counter for the adjacency CSR (incremented on each rebuild)
G_comp = communities_generation config value for the components cache
G_comm = communities_generation config value for the communities cache
```

### Staleness Detection

```c
typedef enum {
    COMM_CACHE_HIT,         /* G_comm == G_adj AND resolution matches */
    COMM_CACHE_WARM_START,  /* G_comm < G_adj AND resolution matches */
    COMM_CACHE_COLD_START   /* G_comm missing OR resolution mismatch */
} CommCacheState;

static CommCacheState check_communities_cache(sqlite3 *db,
                                               const char *vtab_name,
                                               double requested_resolution) {
    int64_t G_adj  = config_get_int(db, vtab_name, "generation", 0);
    int64_t G_comm = config_get_int(db, vtab_name,
                                     "communities_generation", -1);

    if (G_comm < 0)
        return COMM_CACHE_COLD_START;  /* never computed */

    double cached_res = config_get_double(db, vtab_name,
                                           "communities_resolution", -1.0);
    if (!resolution_matches(cached_res, requested_resolution))
        return COMM_CACHE_COLD_START;  /* resolution mismatch */

    if (G_comm < G_adj)
        return COMM_CACHE_WARM_START;  /* stale but same resolution */

    return COMM_CACHE_HIT;  /* fresh */
}
```

### Generation Update Protocol

After a communities rebuild (warm-start or cold-start):

```c
/* After successful Leiden computation: */
config_set_int(db, vtab_name, "communities_generation", G_adj);
config_set_double(db, vtab_name, "communities_resolution", resolution);
config_set_double(db, vtab_name, "communities_modularity", Q);
config_set_int(db, vtab_name, "num_communities", K);
```

The generation is set to the current G_adj (not incremented independently). This ensures that a subsequent adjacency rebuild (which increments G_adj) will automatically mark the communities cache as stale.

---

## 7. Integration with `graph_leiden` TVF

### Decision Flow in `lei_filter()`

When the `graph_leiden` TVF receives a query:

```
Is edge_table a graph_adjacency VT?
├── NO → Current behavior: graph_data_load() + run_leiden() from scratch
└── YES → Check cache state:
         ├── CACHE_HIT → Read from _communities shadow table
         ├── WARM_START → Load cached partition + identify changed nodes
         │                + run_leiden_warm() + update shadow table
         └── COLD_START → Initialize (component-seeded or singletons)
                          + run_leiden() + write shadow table
```

### Modified `lei_filter()` Pseudocode

```c
static int lei_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    /* ... parse parameters (unchanged) ... */

    if (config.edge_table && is_graph_adjacency(vtab->db, config.edge_table)) {
        /* Check if communities feature is enabled */
        int64_t has_communities = config_get_int(
            vtab->db, config.edge_table, "has_communities", 0);

        if (has_communities) {
            int namespace_id = 0;  /* Phase 1 will provide this */
            CommCacheState state = check_communities_cache(
                vtab->db, config.edge_table, resolution);

            switch (state) {
            case COMM_CACHE_HIT:
                return lei_read_from_cache(cur, vtab->db,
                                           config.edge_table, namespace_id);

            case COMM_CACHE_WARM_START:
                return lei_warm_start(cur, vtab->db,
                                      config.edge_table, namespace_id,
                                      resolution, config.direction);

            case COMM_CACHE_COLD_START:
                return lei_cold_start(cur, vtab->db,
                                      config.edge_table, namespace_id,
                                      resolution, config.direction);
            }
        }

        /* Feature not enabled: load from CSR but don't cache */
        rc = graph_data_load_from_adjacency(vtab->db,
                                             config.edge_table, &g, &errmsg);
    } else {
        /* Not a graph_adjacency VT: load from SQL */
        rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    }

    /* ... current run_leiden() + result building (unchanged) ... */
}
```

### Cache Read Path

```c
static int lei_read_from_cache(LeidenCursor *cur, sqlite3 *db,
                               const char *vtab_name, int namespace_id) {
    double Q = config_get_double(db, vtab_name,
                                  "communities_modularity", 0.0);

    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT n.id, c.community_id FROM \"%w_communities\" c "
        "JOIN \"%w_nodes\" n ON n.idx = c.node_idx "
        "WHERE c.namespace_id = %d ORDER BY c.node_idx",
        vtab_name, vtab_name, namespace_id);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return SQLITE_ERROR;

    comr_init(&cur->results);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *node = (const char *)sqlite3_column_text(stmt, 0);
        int comm = sqlite3_column_int(stmt, 1);
        comr_add(&cur->results, node, comm, Q);
    }
    sqlite3_finalize(stmt);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}
```

### Cache Write Path

```c
static int store_communities(sqlite3 *db, const char *vtab_name,
                             int namespace_id, const GraphData *g,
                             const int *community, int N,
                             double resolution, double Q, int K) {
    /* Clear existing rows for this namespace */
    char *sql = sqlite3_mprintf(
        "DELETE FROM \"%w_communities\" WHERE namespace_id = %d",
        vtab_name, namespace_id);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    /* Insert new rows */
    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_communities\"(namespace_id, node_idx, community_id)"
        " VALUES (?, ?, ?)", vtab_name);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    for (int i = 0; i < N; i++) {
        sqlite3_bind_int(stmt, 1, namespace_id);
        sqlite3_bind_int(stmt, 2, i);
        sqlite3_bind_int(stmt, 3, community[i]);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);

    /* Update config metadata */
    int64_t G_adj = config_get_int(db, vtab_name, "generation", 0);
    config_set_int(db, vtab_name, "communities_generation", G_adj);
    config_set_double(db, vtab_name, "communities_resolution", resolution);
    config_set_double(db, vtab_name, "communities_modularity", Q);
    config_set_int(db, vtab_name, "num_communities", K);

    return SQLITE_OK;
}
```

### Namespace Filtering

When Phase 1 provides namespace support, the communities cache is independently maintained per namespace. Each namespace has:

- Its own rows in `_communities` (filtered by `namespace_id`).
- Its own generation/resolution/modularity entries in `_config` (keys prefixed with namespace, e.g., `communities_generation_0`, `communities_generation_1`).

Until Phase 1 is implemented, `namespace_id = 0` is used universally.

---

## 8. Implementation Steps

### Step 1: Add `run_leiden_warm()` Variant

**File:** `src/graph_community.c`

Add a new function alongside the existing `run_leiden()` that accepts an initial partition and changed node set. The core algorithm is identical -- only the initialization differs.

```c
static double run_leiden_warm(const GraphData *g, int *community,
                              double resolution, const char *direction,
                              const int *changed_nodes, int n_changed) {
    int N = g->node_count;
    if (N == 0) return 0.0;

    int use_both = direction && strcmp(direction, "both") == 0;

    double *k = (double *)malloc((size_t)N * sizeof(double));
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        k[i] = weighted_degree(g, i, use_both);
        m += k[i];
    }
    m /= 2.0;
    if (m <= 0.0) { free(k); return 0.0; }

    /*
     * Key difference from run_leiden():
     * community[] is PRE-INITIALIZED by the caller.
     * - Unchanged nodes: community[i] = cached community ID
     * - Changed nodes: community[i] = unique singleton ID
     * We do NOT reset to singletons here.
     */

    /* Compute initial sum_tot from the pre-initialized partition */
    int max_comm = 0;
    for (int i = 0; i < N; i++)
        if (community[i] > max_comm) max_comm = community[i];
    int alloc_size = max_comm + 1;
    /* Ensure enough room; sum_tot indexed by community ID */
    double *sum_tot = (double *)calloc((size_t)alloc_size, sizeof(double));
    for (int i = 0; i < N; i++)
        sum_tot[community[i]] += k[i];

    int *refined = (int *)malloc((size_t)N * sizeof(int));
    int max_iter = 100;

    for (int iter = 0; iter < max_iter; iter++) {
        int moves = leiden_local_moving(g, community, sum_tot, k, m,
                                        resolution, use_both);
        if (moves == 0) break;

        leiden_refinement(g, community, refined, k, m, resolution, use_both);
        memcpy(community, refined, (size_t)N * sizeof(int));

        int K = renumber_communities(community, N);
        /* Resize sum_tot if needed after renumbering */
        if (K > alloc_size) {
            sum_tot = (double *)realloc(sum_tot,
                                        (size_t)K * sizeof(double));
            alloc_size = K;
        }
        memset(sum_tot, 0, (size_t)alloc_size * sizeof(double));
        for (int i = 0; i < N; i++)
            sum_tot[community[i]] += k[i];
    }

    renumber_communities(community, N);
    double Q = compute_modularity(g, community, resolution, m, use_both);

    free(k);
    free(sum_tot);
    free(refined);
    return Q;
}
```

Note: `leiden_local_moving()` currently allocates `neigh_comms` with size `g->out[v].count + g->in[v].count + 1`, and `sum_tot` is accessed by community ID. With warm-start, community IDs can be larger than N (up to max_comm + n_changed). The `sum_tot` array must be allocated to accommodate the maximum community ID, not just N. This is handled in the warm-start initialization above.

### Step 2: Add `_communities` Shadow Table Creation

**File:** `src/graph_adjacency.c` (in `adjacency_create_shadow_tables()`)

When the `features` parameter includes `communities`, create the shadow table:

```c
if (has_feature(params->features, "communities")) {
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_communities\" "
        "(namespace_id INTEGER DEFAULT 0, "
        " node_idx INTEGER NOT NULL, "
        " community_id INTEGER NOT NULL, "
        " PRIMARY KEY (namespace_id, node_idx))",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    config_set(db, name, "has_communities", "1");
}
```

Also update `drop_shadow_tables()` to include `_communities`.

### Step 3: Add Communities Cache Population

**File:** `src/graph_community.c`

Add `store_communities()` function (shown in Section 7) that writes the partition to the shadow table and updates `_config` metadata.

### Step 4: Add Communities Cache Read in `lei_filter()`

**File:** `src/graph_community.c`

Add `lei_read_from_cache()` function (shown in Section 7) and integrate into the `lei_filter()` decision flow.

### Step 5: Add Warm-Start Logic

**File:** `src/graph_community.c`

Add `lei_warm_start()` function that:

1. Loads the graph via `graph_data_load_from_adjacency()`.
2. Loads the cached partition from `_communities`.
3. Identifies changed nodes from `_delta`.
4. Resets changed nodes to singletons.
5. Calls `run_leiden_warm()`.
6. Stores the updated partition via `store_communities()`.

```c
static int lei_warm_start(LeidenCursor *cur, sqlite3 *db,
                          const char *vtab_name, int namespace_id,
                          double resolution, const char *direction) {
    /* Load graph */
    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load_from_adjacency(db, vtab_name, &g, &errmsg);
    if (rc != SQLITE_OK) {
        cur->base.pVtab->zErrMsg = errmsg;
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    int *community = (int *)malloc((size_t)N * sizeof(int));

    /* Load cached partition */
    load_cached_communities(db, vtab_name, namespace_id, community, N);

    /* Identify changed nodes from delta */
    int *changed = NULL;
    int n_changed = 0;
    identify_changed_nodes(db, vtab_name, &g, &changed, &n_changed);

    /* Reset changed nodes to unique singletons */
    int max_comm = 0;
    for (int i = 0; i < N; i++)
        if (community[i] > max_comm) max_comm = community[i];
    for (int i = 0; i < n_changed; i++)
        community[changed[i]] = max_comm + 1 + i;

    /* Run warm-start Leiden */
    double Q = run_leiden_warm(&g, community, resolution, direction,
                               changed, n_changed);

    /* Count communities */
    int K = renumber_communities(community, N);

    /* Build result rows */
    comr_init(&cur->results);
    for (int i = 0; i < N; i++)
        comr_add(&cur->results, g.ids[i], community[i], Q);

    /* Store to cache */
    store_communities(db, vtab_name, namespace_id, &g, community, N,
                      resolution, Q, K);

    free(changed);
    free(community);
    graph_data_destroy(&g);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}
```

### Step 6: Add Component Seeding

**File:** `src/graph_community.c`

Add `seed_from_components()` function (shown in Section 4) and integrate into the cold-start path:

```c
static int lei_cold_start(LeidenCursor *cur, sqlite3 *db,
                          const char *vtab_name, int namespace_id,
                          double resolution, const char *direction) {
    /* Load graph */
    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load_from_adjacency(db, vtab_name, &g, &errmsg);
    if (rc != SQLITE_OK) { /* ... error handling ... */ }

    int N = g.node_count;
    int *community = (int *)malloc((size_t)N * sizeof(int));

    /* Try component seeding first */
    if (!seed_from_components(db, vtab_name, namespace_id, community, N)) {
        /* Fall back to singletons */
        for (int i = 0; i < N; i++)
            community[i] = i;
    }

    /* Run full Leiden (using run_leiden_warm with no changed nodes) */
    double Q = run_leiden_warm(&g, community, resolution, direction,
                               NULL, 0);
    int K = renumber_communities(community, N);

    /* Build results + store cache */
    comr_init(&cur->results);
    for (int i = 0; i < N; i++)
        comr_add(&cur->results, g.ids[i], community[i], Q);

    store_communities(db, vtab_name, namespace_id, &g, community, N,
                      resolution, Q, K);

    free(community);
    graph_data_destroy(&g);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}
```

### Step 7: Add Resolution Key Check

**File:** `src/graph_community.c`

Add `resolution_matches()` and `config_get_double()` / `config_set_double()` helpers (shown in Sections 2 and 5). These are used in `check_communities_cache()`.

### Step 8: Add Generation Check + Namespace Scoping

**File:** `src/graph_community.c`

Add `check_communities_cache()` function (shown in Section 6) and integrate into the top-level `lei_filter()` decision flow. Namespace scoping uses `namespace_id = 0` until Phase 1 provides dynamic namespace resolution.

### Step 9: Update Makefile

**File:** `Makefile`

No new source files are expected -- the changes are within existing `graph_community.c` and `graph_adjacency.c`. If helper functions for `config_get_double` / `config_set_double` are added to a shared location (e.g., `graph_adjacency.c` or a new `graph_config.c`), the Makefile's `SRC` list must be updated accordingly.

---

## 9. Backward Compatibility

### No Feature Flag = No Shadow Table

When the `graph_adjacency` VT is created without the `communities` feature:

- No `_communities` shadow table is created.
- No `has_communities` config entry exists.
- The `graph_leiden` TVF behaves exactly as it does today: loads via `graph_data_load_from_adjacency()` (if adjacency-aware) or `graph_data_load()`, runs `run_leiden()` from scratch, returns results.

### Non-Adjacency Edge Tables

When `graph_leiden` is called with an `edge_table` that is not a `graph_adjacency` VT:

- The TVF uses `graph_data_load()` to load the graph from the raw SQL table.
- No caching is attempted.
- `run_leiden()` (the original function) is called.
- This path is completely unchanged.

### Warm-Start Transparency

The warm-start produces the same partition quality as cold-start (within Leiden's inherent non-determinism from node visit order). The Leiden algorithm is not fully deterministic -- different node orderings in the local moving phase can produce different but equally valid partitions with similar modularity. Warm-start may produce a slightly different partition than cold-start for the same graph, but the modularity will be comparable.

Callers cannot distinguish cache-hit results from recomputed results. The output schema (`node TEXT, community_id INTEGER, modularity REAL`) is unchanged.

### Administrative Commands

The `rebuild` command on `graph_adjacency` should invalidate the communities cache by deleting the `communities_generation` config entry:

```c
/* In adj_full_rebuild(), after incrementing generation: */
config_set(db, vtab_name, "communities_generation", "-1");  /* force stale */
```

This ensures that a full adjacency rebuild triggers a communities recompute on the next `graph_leiden` invocation.

---

## 10. Verification Steps

### 10.1 Correctness: Warm-Start Quality

**Test:** Compare modularity of warm-started partition vs cold-started partition on the same graph.

```python
def test_warm_start_modularity(conn):
    """Warm-start modularity should be >= 95% of cold-start modularity."""
    # Build graph, run Leiden (cold start), record Q_cold
    # Add 5% new edges
    # Run Leiden again (warm start), record Q_warm
    # Assert Q_warm >= 0.95 * Q_cold
```

**Rationale:** Leiden is non-deterministic, so exact equality is not expected. A 95% threshold ensures warm-start does not produce a significantly degraded partition.

### 10.2 Performance: Warm-Start vs Cold-Start

**Test:** Measure wall time for warm-start vs cold-start on a graph with 10% edge changes.

```python
def test_warm_start_performance(conn):
    """Warm-start on 10% edge change should be faster than cold start."""
    # Build graph with 10K nodes, 50K edges
    # Run Leiden (cold start), record time T_cold
    # Add/remove 10% of edges
    # Force adjacency rebuild
    # Run Leiden (warm start), record time T_warm
    # Assert T_warm < T_cold  (expect 1.1-1.4x speedup)
```

### 10.3 Resolution Cache Invalidation

**Test:** Verify that changing resolution produces a different partition and invalidates cache.

```python
def test_resolution_invalidation(conn):
    """Different resolution produces different partition, cache invalidated."""
    # Run Leiden with resolution=1.0, record partition P1
    # Run Leiden with resolution=2.0, record partition P2
    # Assert P1 != P2 (different number of communities)
    # Run Leiden with resolution=1.0 again (should recompute, not cache hit)
    # Record partition P3
    # Assert P3 modularity close to P1 modularity
```

### 10.4 Staleness Detection

**Test:** Add edges, verify cache is marked stale, warm-start produces updated partition.

```python
def test_staleness_detection(conn):
    """Adding edges marks communities cache stale, warm-start updates it."""
    # Build graph, run Leiden, verify cache hit on re-query
    # Add new edges (triggers delta)
    # Rebuild adjacency CSR (increments G_adj)
    # Run Leiden again — should warm-start (G_comm < G_adj)
    # Verify new nodes are in the partition
    # Verify cache is now fresh (re-query is cache hit)
```

### 10.5 Component Seeding

**Test:** Cold start with component seed vs singletons -- compare iteration count.

```python
def test_component_seeding(conn):
    """Component-seeded cold start converges in fewer iterations."""
    # Build graph with 3 disconnected components
    # Compute components (populates _components cache)
    # Run Leiden with components feature enabled
    # Verify community boundaries align with component boundaries
    # Verify modularity > 0 (non-trivial partition within components)
```

### 10.6 Namespace Independence

**Test:** Different namespaces have independent community caches.

```python
def test_namespace_independence(conn):
    """Each namespace has its own community cache."""
    # Create graph_adjacency with namespace_cols='project_id'
    # Insert edges for project_id=1 (star graph)
    # Insert edges for project_id=2 (ring graph)
    # Run Leiden for namespace 1, verify 1 community
    # Run Leiden for namespace 2, verify different partition
    # Verify cache hit for namespace 1 does not return namespace 2 results
```

### 10.7 Empty and Degenerate Cases

```python
def test_empty_graph_communities(conn):
    """Empty graph produces empty communities cache."""

def test_single_node_communities(conn):
    """Single node graph produces one singleton community."""

def test_disconnected_nodes_communities(conn):
    """Nodes with no edges: each gets singleton community."""
```

---

## 11. Complexity Analysis

### Time Complexity

| Operation | Cold Start | Warm Start | Cache Hit |
|-----------|-----------|-----------|----------|
| Graph load | O(V+E) from CSR | O(V+E) from CSR | O(0) skipped |
| Initialization | O(V) singletons | O(V) load + O(V') reset | O(0) skipped |
| Local moving (per iter) | O(VE) | O(V'E') where V'=changed | O(0) skipped |
| Refinement (per iter) | O(VE) | O(V'E') | O(0) skipped |
| Iterations | 5-20 typical | 2-10 typical (fewer) | 0 |
| Result read | O(V) from community[] | O(V) from community[] | O(V) from SQLite |
| Cache write | O(V) INSERT | O(V) INSERT | O(0) skipped |
| **Total** | **O(VE x iter)** | **O(V'E' x iter')** | **O(V)** |

Where:
- V' = nodes in changed neighborhoods (typically 1-5% of V for incremental changes)
- E' = edges incident to V' (proportionally small)
- iter' < iter because the partition starts closer to optimal

### Space Complexity

| Resource | Per Namespace | Notes |
|----------|--------------|-------|
| `_communities` rows | O(V) | 3 integers per node (~12 bytes/row + B-tree overhead) |
| `_config` entries | O(1) | 4 key-value pairs |
| In-memory community[] | O(V) | During computation only |
| In-memory sum_tot[] | O(max_comm) | ≤ O(V) |
| In-memory changed[] | O(V') | For warm-start only |

### Disk Overhead Estimate

For a graph with V = 100,000 nodes:

| Item | Size |
|------|------|
| `_communities` table | ~1.2 MB (12 bytes/row + SQLite page overhead) |
| `_config` entries | ~200 bytes |
| **Total** | **~1.2 MB per namespace** |

This is negligible compared to the CSR storage (which stores O(V+E) integers and doubles).

---

## 12. References

- **Leiden Algorithm (Traag et al., 2019)** -- The original Leiden paper describing the three-phase local moving / refinement / aggregation algorithm with guaranteed convergence to well-connected communities.
  [https://www.nature.com/articles/s41598-019-41695-z](https://www.nature.com/articles/s41598-019-41695-z)

- **Dynamic Leiden (arXiv:2405.11658, 2024)** -- Extension of Leiden for dynamic graphs. Demonstrates warm-start from previous partition gives 1.1-1.4x speedup by focusing re-evaluation on nodes in changed neighborhoods.
  [https://arxiv.org/html/2405.11658v1](https://arxiv.org/html/2405.11658v1)

- **MV4PG: Materialized Views for Property Graphs (arXiv:2411.18847, 2024)** -- Formalization of materialized view maintenance for graph analytics, including incremental update strategies for derived properties (communities, centrality).
  [https://arxiv.org/html/2411.18847v1](https://arxiv.org/html/2411.18847v1)

- **SuiteSparse:GraphBLAS (Davis, 2019)** -- The GraphBLAS delta-merge pattern that inspired the adjacency VT's delta log + CSR rebuild approach, which the communities cache builds upon.
  [https://dl.acm.org/doi/10.1145/3322125](https://dl.acm.org/doi/10.1145/3322125)

- **Modularity and Community Structure (Newman, 2006)** -- Foundational definition of modularity Q used by the Leiden algorithm's objective function.
  [https://doi.org/10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103)
