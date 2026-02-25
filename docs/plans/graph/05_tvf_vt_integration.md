# Phase 5: TVF/VT Integration

**Date:** 2026-02-24
**Status:** Not started
**Depends on:** Phase 1 (scoped adjacency), Phase 2 (SSSP), Phase 3 (components), Phase 4 (communities)
**Scope:** Feature flags, unified rebuild DAG, adjacency+temporal awareness for all 10 TVFs

---

## 1. Overview

Phase 5 is the integration layer that ties Phases 1-4 together. It delivers three things:

1. **Feature flags on `CREATE VIRTUAL TABLE`** — the `features='...'` parameter controls
   which downstream shadow tables are created (`_sssp`, `_components`, `_communities`).
   No features means CSR-only, minimal trigger overhead.

2. **Unified rebuild DAG** — a single `INSERT INTO g(g) VALUES('rebuild')` cascades
   through all enabled shadow tables in dependency order, using generation counters for
   staleness detection.

3. **Adjacency + temporal awareness for all 10 TVFs** — every TVF detects when its
   `edge_table` is a `graph_adjacency` VT, reads from CSR cache instead of issuing SQL,
   and accepts `timestamp_col`, `time_start`, `time_end`, and `namespace` hidden columns.

### Current State

Four TVFs (degree, betweenness, closeness, leiden) already detect `graph_adjacency` via
`is_graph_adjacency()` and load from CSR via `graph_data_load_from_adjacency()`. They also
accept temporal parameters. The remaining six TVFs (BFS, DFS, shortest_path, components,
pagerank, graph_select) use independent SQL loading with no adjacency detection, no temporal
support, and in some cases redundant data structures (Union-Find in components, `PRAdjList`
with O(N) lookup in pagerank).

### Target State

All 10 TVFs:
- Detect `graph_adjacency` and read from CSR cache when available
- Accept temporal hidden columns for time-bounded graph views
- Accept a `namespace` hidden column for scoped queries
- Leverage shadow table caches (SSSP, components, communities) when fresh

---

## 2. Feature Flags Architecture

### 2.1 Bitmask Definition

```c
/* graph_adjacency.h — feature flag constants */
#define ADJ_FEAT_SSSP        0x01
#define ADJ_FEAT_COMPONENTS  0x02
#define ADJ_FEAT_COMMUNITIES 0x04
```

### 2.2 AdjVtab Struct Addition

The existing `AdjVtab` struct gains a `features` field:

```c
typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;
    char *namespace_cols;    /* Phase 1: comma-separated scope columns */
    int features;            /* Phase 5: bitmask of ADJ_FEAT_* flags */
    int64_t generation;      /* increments on each rebuild */
} AdjVtab;
```

### 2.3 Parsing in xCreate

The `features` parameter is a comma-separated string of flag names. Parsing happens
in `parse_adjacency_params()`:

```c
/* In parse_adjacency_params() — add to the existing parameter loop: */
} else if (strncmp(arg, "features=", 9) == 0) {
    const char *val = strip_quotes(arg + 9, buf, (int)sizeof(buf));
    params->features = parse_features(val);
}
```

The `parse_features()` function:

```c
static int parse_features(const char *spec) {
    int flags = 0;
    const char *p = spec;
    while (*p) {
        /* Skip whitespace and commas */
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;

        if (strncmp(p, "sssp", 4) == 0 && (p[4] == ',' || p[4] == '\0' || p[4] == ' ')) {
            flags |= ADJ_FEAT_SSSP;
            p += 4;
        } else if (strncmp(p, "components", 10) == 0 &&
                   (p[10] == ',' || p[10] == '\0' || p[10] == ' ')) {
            flags |= ADJ_FEAT_COMPONENTS;
            p += 10;
        } else if (strncmp(p, "communities", 11) == 0 &&
                   (p[11] == ',' || p[11] == '\0' || p[11] == ' ')) {
            flags |= ADJ_FEAT_COMMUNITIES;
            p += 11;
        } else {
            return -1; /* unknown feature — caller reports error */
        }
    }
    return flags;
}
```

### 2.4 Feature Validation

No implicit dependencies are enforced. Each feature flag is independent:

- `features='sssp'` -- creates `_sssp` only
- `features='components'` -- creates `_components` only
- `features='communities'` -- creates `_communities` only
- `features='communities,components'` -- creates both; Leiden can optionally seed from components
- `features='sssp,components,communities'` -- full-featured mode
- `features=''` or omitted -- CSR-only (lean mode)

If `communities` is enabled and `components` is also enabled, the Leiden algorithm
can seed its initial partition from component IDs. This is an optimization, not a
requirement. The Leiden algorithm works correctly without component seeding.

### 2.5 Persistence

The features bitmask is stored in `_config`:

```sql
INSERT OR REPLACE INTO "{name}_config"(key, value) VALUES ('features', '7');
```

On `xConnect` (reconnect after close), the bitmask is restored from `_config`. This
ensures that dropping/re-opening the connection preserves feature state.

---

## 3. Unified Shadow Table Creation

### 3.1 In xCreate

After parsing parameters and creating the core shadow tables (`_config`, `_nodes`,
`_degree`, `_csr_fwd`, `_csr_rev`, `_delta`), create feature-gated tables:

```c
static int adjacency_create_feature_tables(sqlite3 *db, const char *name, int features) {
    int rc;

    if (features & ADJ_FEAT_SSSP) {
        /* Phase 2 schema: all-pairs SSSP cache */
        char *sql = sqlite3_mprintf(
            "CREATE TABLE IF NOT EXISTS \"%w_sssp\" "
            "(namespace_id INTEGER DEFAULT 0, "
            " source_idx INTEGER, target_idx INTEGER, "
            " distance REAL, "
            " PRIMARY KEY (namespace_id, source_idx, target_idx))",
            name);
        rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) return rc;
    }

    if (features & ADJ_FEAT_COMPONENTS) {
        /* Phase 3 schema: connected component assignments */
        char *sql = sqlite3_mprintf(
            "CREATE TABLE IF NOT EXISTS \"%w_components\" "
            "(namespace_id INTEGER DEFAULT 0, "
            " idx INTEGER, component_id INTEGER, "
            " PRIMARY KEY (namespace_id, idx))",
            name);
        rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) return rc;
    }

    if (features & ADJ_FEAT_COMMUNITIES) {
        /* Phase 4 schema: Leiden community assignments */
        char *sql = sqlite3_mprintf(
            "CREATE TABLE IF NOT EXISTS \"%w_communities\" "
            "(namespace_id INTEGER DEFAULT 0, "
            " idx INTEGER, community_id INTEGER, "
            " modularity REAL, "
            " PRIMARY KEY (namespace_id, idx))",
            name);
        rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) return rc;
    }

    return SQLITE_OK;
}
```

### 3.2 In xDestroy

Drop all feature shadow tables alongside the core tables:

```c
static int drop_feature_tables(sqlite3 *db, const char *name) {
    const char *suffixes[] = {"_sssp", "_components", "_communities"};
    for (int i = 0; i < 3; i++) {
        char *sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w%s\"", name, suffixes[i]);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
    }
    return SQLITE_OK;
}
```

The `drop_shadow_tables()` function is extended to call `drop_feature_tables()` after
dropping the core tables. This ensures clean teardown regardless of which features were
enabled.

---

## 4. Unified Rebuild DAG

### 4.1 Generation Counters

Each level of the DAG has its own generation counter stored in `_config`:

| Config Key | Tracks | Description |
|-----------|--------|-------------|
| `generation` | CSR rebuild | Existing counter — incremented on each `rebuild` or `incremental_rebuild` |
| `gen_sssp` | SSSP computation | Incremented when `_sssp` is recomputed |
| `gen_components` | Components computation | Incremented when `_components` is recomputed |
| `gen_communities` | Communities computation | Incremented when `_communities` is recomputed |
| `gen_adj_at_sssp` | Snapshot of `generation` when SSSP was last computed | Staleness check: stale if `generation > gen_adj_at_sssp` |
| `gen_adj_at_components` | Snapshot of `generation` when components was last computed | Staleness check |
| `gen_adj_at_communities` | Snapshot of `generation` when communities was last computed | Staleness check |

A downstream cache is **fresh** when `gen_adj_at_X == generation`. It is **stale** when
`gen_adj_at_X < generation`.

### 4.2 Full Rebuild Command

`INSERT INTO g(g) VALUES('rebuild')` triggers the full DAG:

```
Step 1: Always
  ├── Load edges from source table
  ├── Build GraphData
  ├── Build CSR (fwd + rev)
  ├── Store blocked CSR in _csr_fwd, _csr_rev
  ├── Store _nodes, _degree
  └── Increment generation (G_adj)

Step 2: If features & ADJ_FEAT_COMPONENTS
  ├── Run Union-Find on CSR (components_from_csr())
  ├── Write results to _components
  ├── Set gen_components = gen_components + 1
  └── Set gen_adj_at_components = G_adj

Step 3: If features & ADJ_FEAT_COMMUNITIES
  ├── Load previous partition from _communities (for warm-start)
  ├── If ADJ_FEAT_COMPONENTS enabled: seed from _components
  ├── Run Leiden (run_leiden_warm() if previous partition exists)
  ├── Write results to _communities
  ├── Set gen_communities = gen_communities + 1
  └── Set gen_adj_at_communities = G_adj

Step 4: If features & ADJ_FEAT_SSSP
  ├── Run all-pairs SSSP on CSR (sssp_all_pairs_from_csr())
  ├── Write results to _sssp
  ├── Set gen_sssp = gen_sssp + 1
  └── Set gen_adj_at_sssp = G_adj
```

SSSP runs last because it is the most expensive (O(V*E) or O(V^2 log V)) and does not
feed into other shadow tables. Components runs before communities because Leiden can
optionally seed from component IDs.

### 4.3 Incremental Rebuild Command

`INSERT INTO g(g) VALUES('incremental_rebuild')` follows the same DAG but:

1. CSR is rebuilt via `csr_apply_delta()` instead of full reload
2. Components are recomputed from the merged CSR (incremental Union-Find is not implemented)
3. Communities use warm-start from the previous partition
4. SSSP is fully recomputed (incremental all-pairs SSSP is not practical)

### 4.4 Rebuild Implementation Sketch

```c
static int adj_rebuild_full(AdjVtab *vtab) {
    sqlite3 *db = vtab->db;
    const char *name = vtab->vtab_name;
    int features = vtab->features;

    /* Step 1: Rebuild CSR (existing implementation) */
    int rc = rebuild_csr_full(vtab);  /* existing function */
    if (rc != SQLITE_OK) return rc;

    vtab->generation++;
    config_set_int(db, name, "generation", vtab->generation);

    /* Step 2: Components */
    if (features & ADJ_FEAT_COMPONENTS) {
        rc = rebuild_components(vtab);
        if (rc != SQLITE_OK) return rc;
        config_set_int(db, name, "gen_adj_at_components", vtab->generation);
    }

    /* Step 3: Communities */
    if (features & ADJ_FEAT_COMMUNITIES) {
        rc = rebuild_communities(vtab);
        if (rc != SQLITE_OK) return rc;
        config_set_int(db, name, "gen_adj_at_communities", vtab->generation);
    }

    /* Step 4: SSSP */
    if (features & ADJ_FEAT_SSSP) {
        rc = rebuild_sssp(vtab);
        if (rc != SQLITE_OK) return rc;
        config_set_int(db, name, "gen_adj_at_sssp", vtab->generation);
    }

    return SQLITE_OK;
}
```

---

## 5. Base TVF Upgrades: graph_bfs

### 5.1 Current Implementation

In `graph_tvf.c`, `run_bfs()` uses **lazy SQL per-node**: for each node in the BFS queue,
it executes `SELECT dst FROM edges WHERE src = ?`. It uses `StrHashSet` for visited
tracking and a `Queue` for the BFS frontier.

- No temporal filtering
- No adjacency detection
- No namespace support
- O(1) SQL prepare per node expansion (but each node requires a round-trip to SQLite)

### 5.2 New Schema

Add hidden columns for temporal and namespace support:

```c
/* New column layout for graph_bfs / graph_dfs */
enum {
    GTRAV_COL_NODE = 0,
    GTRAV_COL_DEPTH,
    GTRAV_COL_PARENT,
    GTRAV_COL_EDGE_TABLE,      /* hidden */
    GTRAV_COL_SRC_COL,         /* hidden */
    GTRAV_COL_DST_COL,         /* hidden */
    GTRAV_COL_START_NODE,      /* hidden */
    GTRAV_COL_MAX_DEPTH,       /* hidden */
    GTRAV_COL_DIRECTION,       /* hidden */
    GTRAV_COL_TIMESTAMP_COL,   /* hidden — new */
    GTRAV_COL_TIME_START,      /* hidden — new */
    GTRAV_COL_TIME_END,        /* hidden — new */
    GTRAV_COL_NAMESPACE,       /* hidden — new */
};
```

The vtab declaration becomes:

```c
sqlite3_declare_vtab(db,
    "CREATE TABLE x("
    "  node TEXT, depth INTEGER, parent TEXT,"
    "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
    "  start_node TEXT HIDDEN, max_depth INTEGER HIDDEN, direction TEXT HIDDEN,"
    "  timestamp_col TEXT HIDDEN, time_start HIDDEN, time_end HIDDEN,"
    "  namespace TEXT HIDDEN"
    ")");
```

### 5.3 xBestIndex Update

Replace the current naive argvIndex assignment with `graph_best_index_common()`:

```c
static int gtrav_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    /* Required: edge_table, src_col, dst_col, start_node (bits 0-3) */
    return graph_best_index_common(pInfo, GTRAV_COL_EDGE_TABLE, GTRAV_COL_NAMESPACE,
                                   0x0F, 1000.0);
}
```

### 5.4 xFilter: Adjacency Detection Path

```c
static int gtrav_filter(...) {
    /* ... parse argv into edge_table, src_col, dst_col, start_node,
           max_depth, direction, timestamp_col, time_start, time_end,
           namespace ... */

    if (edge_table && is_graph_adjacency(vtab->db, edge_table)) {
        /* Fast path: load GraphData from CSR cache */
        GraphData g;
        graph_data_init(&g);
        char *errmsg = NULL;
        int rc = graph_data_load_from_adjacency(vtab->db, edge_table, &g, &errmsg);
        if (rc != SQLITE_OK) {
            vtab->base.zErrMsg = errmsg;
            graph_data_destroy(&g);
            return SQLITE_ERROR;
        }

        /* BFS on in-memory adjacency lists */
        rc = run_bfs_on_graphdata(&g, start_node, max_depth, direction, &cur->results);
        graph_data_destroy(&g);
        if (rc != SQLITE_OK) {
            vtab->base.zErrMsg = sqlite3_mprintf("graph_bfs: traversal failed");
            return SQLITE_ERROR;
        }
    } else {
        /* Slow path: existing lazy SQL approach (backward compatible) */
        GraphLoadConfig config;
        memset(&config, 0, sizeof(config));
        config.edge_table = edge_table;
        config.src_col = src_col;
        config.dst_col = dst_col;
        config.direction = direction;
        config.timestamp_col = timestamp_col;
        config.time_start = time_start_val;
        config.time_end = time_end_val;

        /* When temporal params are set, use graph_data_load for filtered loading */
        if (timestamp_col) {
            GraphData g;
            graph_data_init(&g);
            char *errmsg = NULL;
            int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
            if (rc != SQLITE_OK) { /* handle error */ }

            rc = run_bfs_on_graphdata(&g, start_node, max_depth, direction, &cur->results);
            graph_data_destroy(&g);
        } else {
            /* Original lazy SQL path — no temporal filter */
            rc = run_bfs(vtab->db, edge_table, src_col, dst_col,
                         start_node, max_depth, direction, &cur->results);
        }
    }
    /* ... */
}
```

### 5.5 New: run_bfs_on_graphdata()

A new function that runs BFS on an in-memory `GraphData` structure instead of issuing
per-node SQL. This is shared by the adjacency-aware and temporal-filter paths:

```c
static int run_bfs_on_graphdata(const GraphData *g, const char *start_node,
                                 int max_depth, const char *direction,
                                 TraversalResults *results) {
    tr_init(results, 64);
    int start_idx = graph_data_find(g, start_node);
    if (start_idx < 0) return SQLITE_OK;  /* start node not found */

    int use_out = !direction || strcmp(direction, "reverse") != 0;
    int use_in  = direction && (strcmp(direction, "reverse") == 0 ||
                                strcmp(direction, "both") == 0);

    int N = g->node_count;
    int *visited = (int *)calloc((size_t)N, sizeof(int));
    int *queue = (int *)malloc((size_t)N * sizeof(int));
    int *depth_arr = (int *)malloc((size_t)N * sizeof(int));
    int *parent_arr = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) parent_arr[i] = -1;

    int qhead = 0, qtail = 0;
    visited[start_idx] = 1;
    queue[qtail] = start_idx;
    depth_arr[qtail] = 0;
    qtail++;

    while (qhead < qtail) {
        int v = queue[qhead];
        int d = depth_arr[qhead];
        qhead++;

        tr_push(results, g->ids[v], d,
                parent_arr[v] >= 0 ? g->ids[parent_arr[v]] : NULL);

        if (d >= max_depth) continue;

        /* Expand forward adjacency */
        if (use_out) {
            for (int e = 0; e < g->out[v].count; e++) {
                int w = g->out[v].edges[e].target;
                if (!visited[w]) {
                    visited[w] = 1;
                    parent_arr[w] = v;
                    queue[qtail] = w;
                    depth_arr[qtail] = d + 1;
                    qtail++;
                }
            }
        }

        /* Expand reverse adjacency */
        if (use_in) {
            for (int e = 0; e < g->in[v].count; e++) {
                int w = g->in[v].edges[e].target;
                if (!visited[w]) {
                    visited[w] = 1;
                    parent_arr[w] = v;
                    queue[qtail] = w;
                    depth_arr[qtail] = d + 1;
                    qtail++;
                }
            }
        }
    }

    free(visited);
    free(queue);
    free(depth_arr);
    free(parent_arr);
    return SQLITE_OK;
}
```

**Performance benefit:** The adjacency-aware path avoids N separate SQL queries. For a
graph with 10K nodes and average degree 10, this eliminates ~10K `sqlite3_step()` calls
and replaces them with direct array access.

---

## 6. Base TVF Upgrades: graph_dfs

### 6.1 Current Implementation

`run_dfs()` in `graph_tvf.c` is structurally identical to `run_bfs()` but uses a stack
instead of a queue. Same lazy SQL per-node approach, same `StrHashSet` for visited,
same lack of temporal/adjacency support.

### 6.2 Changes

Identical to BFS (Section 5) in terms of:
- New hidden columns (same enum, shared `gtrav_connect` already handles both)
- xBestIndex update (shared `gtrav_best_index`)
- xFilter adjacency detection (shared `gtrav_filter`)

New function: `run_dfs_on_graphdata()` — same as `run_bfs_on_graphdata()` but uses
a LIFO stack:

```c
static int run_dfs_on_graphdata(const GraphData *g, const char *start_node,
                                 int max_depth, const char *direction,
                                 TraversalResults *results) {
    /* Same setup as BFS */
    /* ... */

    /* DFS: use top-of-stack instead of head-of-queue */
    int stack_top = 0;
    visited[start_idx] = 1;
    stack[stack_top] = start_idx;
    depth_arr[stack_top] = 0;
    stack_top++;

    while (stack_top > 0) {
        stack_top--;
        int v = stack[stack_top];
        int d = depth_arr[stack_top];

        tr_push(results, g->ids[v], d,
                parent_arr[v] >= 0 ? g->ids[parent_arr[v]] : NULL);

        if (d >= max_depth) continue;

        /* Push neighbors (reverse order for consistent DFS ordering) */
        /* ... expand out[] and optionally in[] ... */
    }

    /* ... cleanup ... */
    return SQLITE_OK;
}
```

Since BFS and DFS share `gtrav_connect` and `gtrav_filter` (differentiated by
`vtab->is_dfs`), the xFilter function selects `run_bfs_on_graphdata()` or
`run_dfs_on_graphdata()` based on the flag.

---

## 7. Base TVF Upgrades: graph_shortest_path

### 7.1 Current Implementation

Two independent lazy SQL functions:
- `run_shortest_path_bfs()` — unweighted BFS with parent tracking, per-node SQL
- `run_shortest_path_dijkstra()` — weighted Dijkstra with linear-scan PQ, per-node SQL

Both use `StrHashSet` for visited/settled tracking and `sqlite3_mprintf`-based string
management throughout.

### 7.2 New Schema

Add hidden columns for temporal, namespace, and weight support via `graph_best_index_common()`:

```c
enum {
    GSP_COL_NODE = 0,
    GSP_COL_DISTANCE,
    GSP_COL_PATH_ORDER,
    GSP_COL_EDGE_TABLE,       /* hidden */
    GSP_COL_SRC_COL,          /* hidden */
    GSP_COL_DST_COL,          /* hidden */
    GSP_COL_START_NODE,       /* hidden */
    GSP_COL_END_NODE,         /* hidden */
    GSP_COL_WEIGHT_COL,       /* hidden */
    GSP_COL_TIMESTAMP_COL,    /* hidden — new */
    GSP_COL_TIME_START,       /* hidden — new */
    GSP_COL_TIME_END,         /* hidden — new */
    GSP_COL_NAMESPACE,        /* hidden — new */
};
```

### 7.3 Adjacency Detection + SSSP Cache Shortcut

When the edge table is a `graph_adjacency` VT:

1. Load `GraphData` via `graph_data_load_from_adjacency()`
2. Check if `ADJ_FEAT_SSSP` is enabled and the `_sssp` cache is fresh
3. If fresh: single-pair distance is an O(1) lookup into `_sssp`
4. If stale or not enabled: run Dijkstra/BFS on in-memory adjacency lists

```c
/* In gsp_filter: */
if (edge_table && is_graph_adjacency(vtab->db, edge_table)) {
    /* Try SSSP cache first */
    int64_t gen_adj = config_get_int(vtab->db, edge_table, "generation", 0);
    int64_t gen_at_sssp = config_get_int(vtab->db, edge_table, "gen_adj_at_sssp", -1);

    if (gen_at_sssp == gen_adj) {
        /* Cache is fresh — O(1) lookup */
        int rc = sssp_cache_lookup(vtab->db, edge_table, start_node, end_node,
                                   &cur->results);
        if (rc == SQLITE_OK && cur->results.count > 0) {
            cur->eof = 0;
            return SQLITE_OK;
        }
        /* Fall through to in-memory path if lookup fails */
    }

    /* In-memory Dijkstra on GraphData */
    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load_from_adjacency(vtab->db, edge_table, &g, &errmsg);
    if (rc != SQLITE_OK) { /* handle error */ }

    rc = run_shortest_path_on_graphdata(&g, start_node, end_node,
                                         weight_col != NULL, &cur->results);
    graph_data_destroy(&g);
}
```

### 7.4 New: run_shortest_path_on_graphdata()

Runs Dijkstra (weighted) or BFS (unweighted) on an in-memory `GraphData`, using the
existing `DoublePQ` from `graph_centrality.c` (extracted to a shared header) or a
reimplemented min-heap:

```c
static int run_shortest_path_on_graphdata(const GraphData *g,
                                           const char *start_node,
                                           const char *end_node,
                                           int weighted,
                                           PathResults *results) {
    pr_init(results, 16);
    int start_idx = graph_data_find(g, start_node);
    int end_idx = graph_data_find(g, end_node);
    if (start_idx < 0 || end_idx < 0) return SQLITE_OK;

    int N = g->node_count;
    double *dist = (double *)malloc((size_t)N * sizeof(double));
    int *parent = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) { dist[i] = -1.0; parent[i] = -1; }
    dist[start_idx] = 0.0;

    if (weighted) {
        /* Dijkstra with binary heap */
        /* ... use DoublePQ, expand via g->out[], track parent[] ... */
    } else {
        /* BFS */
        /* ... use int queue, expand via g->out[], track parent[] ... */
    }

    /* Trace path from end_idx back to start_idx via parent[] */
    if (dist[end_idx] >= 0) {
        /* ... build path, reverse, push to results ... */
    }

    free(dist);
    free(parent);
    return SQLITE_OK;
}
```

---

## 8. Base TVF Upgrades: graph_components

### 8.1 Current Implementation

`run_components()` in `graph_tvf.c` has its own complete implementation:
- Own SQL loading: `SELECT src, dst FROM edges`
- Own `UnionFind` with O(N) linear scan for node lookup (`uf_find_or_add`)
- No temporal support, no adjacency detection, no weight support

### 8.2 Three-Tier Strategy

When the TVF is invoked, it follows one of three paths:

**Path A: Adjacency VT with `FEAT_COMPONENTS` enabled and cache is fresh**
```
_components shadow table → read directly → return results
```

**Path B: Adjacency VT (with or without `FEAT_COMPONENTS`), cache stale or absent**
```
graph_data_load_from_adjacency() → Union-Find on GraphData → return results
If FEAT_COMPONENTS: also write results to _components, update generation
```

**Path C: Plain table (no adjacency VT)**
```
graph_data_load() → Union-Find on GraphData → return results
(or existing run_components() for backward compatibility)
```

### 8.3 New Schema

Add hidden columns:

```c
enum {
    GC_COL_NODE = 0,
    GC_COL_COMPONENT_ID,
    GC_COL_COMPONENT_SIZE,
    GC_COL_EDGE_TABLE,       /* hidden */
    GC_COL_SRC_COL,          /* hidden */
    GC_COL_DST_COL,          /* hidden */
    GC_COL_TIMESTAMP_COL,    /* hidden — new */
    GC_COL_TIME_START,       /* hidden — new */
    GC_COL_TIME_END,         /* hidden — new */
    GC_COL_NAMESPACE,        /* hidden — new */
};
```

### 8.4 Cache Read Logic

```c
static int gc_filter(...) {
    /* ... parse argv ... */

    if (edge_table && is_graph_adjacency(vtab->db, edge_table)) {
        int64_t gen_adj = config_get_int(vtab->db, edge_table, "generation", 0);
        int64_t gen_at_comp = config_get_int(vtab->db, edge_table,
                                              "gen_adj_at_components", -1);

        if (gen_at_comp == gen_adj) {
            /* Cache fresh — read from _components shadow table */
            return gc_read_from_shadow(vtab->db, edge_table, &cur->results);
        }

        /* Cache stale — recompute from CSR */
        GraphData g;
        graph_data_init(&g);
        char *errmsg = NULL;
        int rc = graph_data_load_from_adjacency(vtab->db, edge_table, &g, &errmsg);
        if (rc != SQLITE_OK) { /* handle error */ }

        rc = run_components_on_graphdata(&g, &cur->results);
        graph_data_destroy(&g);
        return rc;
    }

    /* Fallback: original SQL path with temporal support */
    /* ... */
}
```

### 8.5 New: run_components_on_graphdata()

Replaces the SQL-based `run_components()` with Union-Find on `GraphData`:

```c
static int run_components_on_graphdata(const GraphData *g,
                                        ComponentResults *results) {
    comp_results_init(results);
    int N = g->node_count;
    if (N == 0) return SQLITE_OK;

    /* Union-Find with integer indices (O(1) lookup via GraphData hash map) */
    int *parent = (int *)malloc((size_t)N * sizeof(int));
    int *rank = (int *)calloc((size_t)N, sizeof(int));
    for (int i = 0; i < N; i++) parent[i] = i;

    /* Process all edges */
    for (int i = 0; i < N; i++) {
        for (int e = 0; e < g->out[i].count; e++) {
            int j = g->out[i].edges[e].target;
            /* Union i and j */
            int ri = uf_find_int(parent, i);
            int rj = uf_find_int(parent, j);
            if (ri != rj) {
                if (rank[ri] < rank[rj]) { int t = ri; ri = rj; rj = t; }
                parent[rj] = ri;
                if (rank[ri] == rank[rj]) rank[ri]++;
            }
        }
    }

    /* Count component sizes */
    int *comp_size = (int *)calloc((size_t)N, sizeof(int));
    for (int i = 0; i < N; i++) {
        comp_size[uf_find_int(parent, i)]++;
    }

    /* Build results */
    for (int i = 0; i < N; i++) {
        int root = uf_find_int(parent, i);
        comp_results_add(results, g->ids[i], root, comp_size[root]);
    }

    free(parent);
    free(rank);
    free(comp_size);
    return SQLITE_OK;
}
```

**Performance benefit:** The existing `uf_find_or_add()` uses O(N) linear scan for
every node lookup. The GraphData-based version uses O(1) hash-map lookup via
`graph_data_find_or_add()`, then Union-Find operates on integer indices directly.

---

## 9. Base TVF Upgrades: graph_pagerank

### 9.1 Current Implementation

`run_pagerank()` in `graph_tvf.c` uses a custom `PRAdjList` structure with:
- `pr_adj_find_or_add()` — O(N) linear scan for node lookup
- Parallel arrays: `ids[]`, `out_edges[][]`, `out_count[]`, `out_cap[]`
- No temporal support, no adjacency detection

### 9.2 New Schema

Add hidden columns:

```c
enum {
    GPR_COL_NODE = 0,
    GPR_COL_RANK,
    GPR_COL_EDGE_TABLE,      /* hidden */
    GPR_COL_SRC_COL,         /* hidden */
    GPR_COL_DST_COL,         /* hidden */
    GPR_COL_DAMPING,         /* hidden */
    GPR_COL_ITERATIONS,      /* hidden */
    GPR_COL_TIMESTAMP_COL,   /* hidden — new */
    GPR_COL_TIME_START,      /* hidden — new */
    GPR_COL_TIME_END,        /* hidden — new */
    GPR_COL_NAMESPACE,       /* hidden — new */
};
```

### 9.3 Adjacency Detection Path

```c
static int gpr_filter(...) {
    /* ... parse argv ... */

    if (edge_table && is_graph_adjacency(vtab->db, edge_table)) {
        GraphData g;
        graph_data_init(&g);
        char *errmsg = NULL;
        int rc = graph_data_load_from_adjacency(vtab->db, edge_table, &g, &errmsg);
        if (rc != SQLITE_OK) { /* handle error */ }

        rc = run_pagerank_on_graphdata(&g, damping, iterations, &cur->results);
        graph_data_destroy(&g);
        return rc;
    }

    /* Fallback: temporal-aware graph_data_load or original SQL path */
    if (timestamp_col) {
        GraphLoadConfig config = { /* ... */ };
        GraphData g;
        graph_data_init(&g);
        int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
        /* ... run_pagerank_on_graphdata ... */
    } else {
        /* Original path (backward compatible) */
        rc = run_pagerank(vtab->db, edge_table, src_col, dst_col,
                          damping, iterations, &cur->results);
    }
}
```

### 9.4 New: run_pagerank_on_graphdata()

Replaces the `PRAdjList`-based `run_pagerank()`:

```c
static int run_pagerank_on_graphdata(const GraphData *g, double damping,
                                      int iterations, PRResults *results) {
    pr_results_init(results);
    int N = g->node_count;
    if (N == 0) return SQLITE_OK;

    double *rank_cur = (double *)malloc((size_t)N * sizeof(double));
    double *rank_new = (double *)malloc((size_t)N * sizeof(double));

    double init_rank = 1.0 / N;
    for (int i = 0; i < N; i++) rank_cur[i] = init_rank;

    double teleport = (1.0 - damping) / N;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N; i++) rank_new[i] = teleport;

        for (int i = 0; i < N; i++) {
            int out_deg = g->out[i].count;
            if (out_deg == 0) {
                double share = damping * rank_cur[i] / N;
                for (int j = 0; j < N; j++) rank_new[j] += share;
            } else {
                double share = damping * rank_cur[i] / out_deg;
                for (int e = 0; e < out_deg; e++) {
                    rank_new[g->out[i].edges[e].target] += share;
                }
            }
        }

        double *tmp = rank_cur; rank_cur = rank_new; rank_new = tmp;
    }

    for (int i = 0; i < N; i++) {
        pr_results_add(results, g->ids[i], rank_cur[i]);
    }

    free(rank_cur);
    free(rank_new);
    return SQLITE_OK;
}
```

**Performance benefit:** Eliminates O(N) linear scan per node lookup (`pr_adj_find_or_add`).
GraphData's hash map provides O(1) amortized lookup. For a 100K-node graph, this turns
~100K linear scans (each up to 100K comparisons) into ~100K hash probes.

### 9.5 Future: PageRank Shadow Table

A `_pagerank` shadow table could be added in a future phase as a new feature flag
(`ADJ_FEAT_PAGERANK`). This is not in scope for Phase 5 because PageRank is cheap
to recompute (O(iterations * E), typically 20 iterations) and does not feed into
other algorithms.

---

## 10. Base TVF Upgrades: graph_select

### 10.1 Current Implementation

`graph_select_tvf.c` uses `graph_data_load()` to build a `GraphData` structure, then
evaluates the selector AST against it. It does NOT call `is_graph_adjacency()` or
`graph_data_load_from_adjacency()`.

### 10.2 Changes

Add adjacency detection in `gs_filter()`:

```c
/* In gs_filter(), replace: */
int rc = graph_data_load(vtab->db, &config, &cur->graph, &load_err);

/* With: */
int rc;
if (is_graph_adjacency(vtab->db, edge_table)) {
    rc = graph_data_load_from_adjacency(vtab->db, edge_table, &cur->graph, &load_err);
} else {
    rc = graph_data_load(vtab->db, &config, &cur->graph, &load_err);
}
```

### 10.3 Namespace Support

When a `namespace` hidden column is provided, it is passed to
`graph_data_load_from_adjacency()` (Phase 1 extends this function to accept namespace
filtering). For plain tables, namespace is ignored.

### 10.4 New Hidden Columns

The `graph_select` TVF gains additional hidden columns:

```c
enum {
    GS_COL_NODE = 0,
    GS_COL_DEPTH,
    GS_COL_DIRECTION,
    GS_COL_EDGE_TABLE,       /* hidden */
    GS_COL_SRC_COL,          /* hidden */
    GS_COL_DST_COL,          /* hidden */
    GS_COL_SELECTOR,         /* hidden */
    GS_COL_TIMESTAMP_COL,    /* hidden — new */
    GS_COL_TIME_START,       /* hidden — new */
    GS_COL_TIME_END,         /* hidden — new */
    GS_COL_NAMESPACE,        /* hidden — new */
};
```

---

## 11. Hidden Column Pattern

### 11.1 Consistent Layout

All TVFs follow a consistent pattern for hidden columns. The general order is:

```
[output columns...],
edge_table TEXT HIDDEN,
src_col TEXT HIDDEN,
dst_col TEXT HIDDEN,
[algorithm-specific hidden params...],
direction TEXT HIDDEN,      /* where applicable */
timestamp_col TEXT HIDDEN,
time_start HIDDEN,
time_end HIDDEN,
namespace TEXT HIDDEN
```

### 11.2 Summary Table

| TVF | Algorithm-Specific Params | Direction | Temporal | Namespace |
|-----|--------------------------|:---------:|:--------:|:---------:|
| `graph_bfs` | `start_node`, `max_depth` | Yes | Yes | Yes |
| `graph_dfs` | `start_node`, `max_depth` | Yes | Yes | Yes |
| `graph_shortest_path` | `start_node`, `end_node`, `weight_col` | No | Yes | Yes |
| `graph_components` | (none) | No | Yes | Yes |
| `graph_pagerank` | `damping`, `iterations` | No | Yes | Yes |
| `graph_degree` | `weight_col`, `normalized` | Yes | Yes (existing) | Yes |
| `graph_betweenness` | `weight_col`, `normalized`, `auto_approx` | Yes | Yes (existing) | Yes |
| `graph_closeness` | `weight_col`, `normalized` | Yes | Yes (existing) | Yes |
| `graph_leiden` | `weight_col`, `resolution` | Yes | Yes (existing) | Yes |
| `graph_select` | `selector` | No | Yes | Yes |

### 11.3 xBestIndex Standardization

All TVFs use `graph_best_index_common()` for consistent two-pass argvIndex assignment.
The currently non-conforming TVFs are:

- `gtrav_best_index` (BFS/DFS) — currently uses naive sequential argvIndex
- `gsp_best_index` (shortest_path) — currently uses naive sequential argvIndex
- `gc_best_index` (components) — uses its own bitmask logic
- `gpr_best_index` (pagerank) — uses its own bitmask logic

All four must be migrated to `graph_best_index_common()`. The centrality and community
TVFs already use it and serve as the reference implementation.

### 11.4 Namespace Hidden Column

The `namespace` column accepts a TEXT value that identifies the namespace within a
scoped `graph_adjacency` VT (Phase 1). When set:

- `graph_data_load_from_adjacency()` loads only edges within that namespace
- TVFs on plain tables ignore the namespace (no-op)
- The namespace value format matches the composite key defined in Phase 1
  (e.g., `"project_42:session_7"`)

---

## 12. Implementation Steps

### Step 1: Add features bitmask to AdjVtab

- Add `int features;` to `AdjVtab` struct
- Add `int features;` to `AdjParams` struct
- Parse `features='...'` in `parse_adjacency_params()`
- Implement `parse_features()` function
- Store features in `_config` on xCreate
- Restore features from `_config` on xConnect
- Error on unknown feature names

**Files:** `graph_adjacency.c`, `graph_adjacency.h`

### Step 2: Add feature-gated shadow table creation/destruction

- Implement `adjacency_create_feature_tables()`
- Call it from `adj_xCreate()` after `adjacency_create_shadow_tables()`
- Implement `drop_feature_tables()`
- Call it from `adj_xDestroy()` before `drop_shadow_tables()`

**Files:** `graph_adjacency.c`

### Step 3: Add unified rebuild DAG with generation cascade

- Add generation counter config keys: `gen_sssp`, `gen_components`, `gen_communities`,
  `gen_adj_at_sssp`, `gen_adj_at_components`, `gen_adj_at_communities`
- Implement `rebuild_components()`, `rebuild_communities()`, `rebuild_sssp()`
  (wrappers that call Phase 2/3/4 functions and write results to shadow tables)
- Modify `adj_rebuild_full()` to cascade through the DAG
- Modify `adj_rebuild_incremental()` similarly

**Files:** `graph_adjacency.c`
**Depends on:** Phase 2 (`graph_sssp.h`), Phase 3 (`components_from_csr()`), Phase 4 (`run_leiden_warm()`)

### Step 4: Add temporal + namespace hidden columns to BFS/DFS

- Extend `gtrav_connect` vtab declaration with new hidden columns
- Migrate `gtrav_best_index` from naive argvIndex to `graph_best_index_common()`
- Update `gtrav_filter` to parse new hidden columns from bitmask

**Files:** `graph_tvf.c`

### Step 5: Add adjacency detection to BFS

- Implement `run_bfs_on_graphdata()`
- Add `is_graph_adjacency()` check in `gtrav_filter` for BFS path
- When adjacency detected: load GraphData, run BFS on in-memory adjacency
- When temporal params set and NOT adjacency: use `graph_data_load()` with temporal config
- When neither: use existing `run_bfs()` (backward compatible)

**Files:** `graph_tvf.c`

### Step 6: Add adjacency detection to DFS

- Implement `run_dfs_on_graphdata()`
- Same xFilter logic as BFS, selecting DFS function via `vtab->is_dfs`

**Files:** `graph_tvf.c`

### Step 7: Add adjacency detection to shortest_path

- Extend `gsp_connect` vtab declaration with temporal + namespace hidden columns
- Migrate `gsp_best_index` to `graph_best_index_common()`
- Implement `run_shortest_path_on_graphdata()`
- Add SSSP cache shortcut: check `gen_adj_at_sssp == generation`, if fresh do O(1) lookup
- Update `gsp_filter` with adjacency detection and temporal support

**Files:** `graph_tvf.c`

### Step 8: Add adjacency detection to components

- Extend `gc_connect` vtab declaration with temporal + namespace hidden columns
- Migrate `gc_best_index` to `graph_best_index_common()`
- Implement `run_components_on_graphdata()`
- Add `_components` cache read: check `gen_adj_at_components == generation`
- Update `gc_filter` with three-tier strategy (cache/CSR/SQL)

**Files:** `graph_tvf.c`

### Step 9: Add adjacency detection to pagerank

- Extend `gpr_connect` vtab declaration with temporal + namespace hidden columns
- Migrate `gpr_best_index` to `graph_best_index_common()`
- Implement `run_pagerank_on_graphdata()`
- Update `gpr_filter` with adjacency detection and temporal support

**Files:** `graph_tvf.c`

### Step 10: Add adjacency detection to graph_select

- Add `#include "graph_adjacency.h"` to `graph_select_tvf.c`
- Add temporal + namespace hidden columns to `gs_connect` declaration
- Update `gs_best_index` required mask for new column count
- Add `is_graph_adjacency()` check in `gs_filter`
- When adjacency: use `graph_data_load_from_adjacency()`

**Files:** `graph_select_tvf.c`

### Step 11: Add namespace hidden column to all TVFs

- Add namespace handling to the adjacency path in each TVF's xFilter
- Pass namespace to `graph_data_load_from_adjacency()` (Phase 1 API extension)
- Ignore namespace when loading from plain tables

**Files:** `graph_tvf.c`, `graph_centrality.c`, `graph_community.c`, `graph_select_tvf.c`
**Depends on:** Phase 1 (namespace support in `graph_data_load_from_adjacency`)

### Step 12: Update all xBestIndex implementations for new hidden columns

- Verify all 10 TVFs use `graph_best_index_common()`
- Verify argvIndex is contiguous (no gaps) for all column combinations
- Verify required_mask correctly identifies mandatory columns
- Verify optional temporal columns do not break backward compatibility

**Files:** `graph_tvf.c`, `graph_centrality.c`, `graph_community.c`, `graph_select_tvf.c`

### Step 13: Update Makefile, tests

- Add `graph_adjacency.h` include dependency to `graph_tvf.c` and `graph_select_tvf.c`
  in the Makefile
- Add C unit tests for:
  - Feature flag parsing: `test_parse_features()`
  - `run_bfs_on_graphdata()`, `run_dfs_on_graphdata()`
  - `run_shortest_path_on_graphdata()`
  - `run_components_on_graphdata()`
  - `run_pagerank_on_graphdata()`
- Add Python integration tests (see Section 13)

**Files:** `Makefile`, `test/test_graph_tvf.c` (new), `pytests/test_graph_integration.py` (new)

---

## 13. Verification Steps

### 13.1 Per-TVF Verification Matrix

For each of the 10 TVFs, all five test paths must pass:

| Test | Description | Path |
|------|-------------|------|
| **(a)** Backward compatibility | TVF on plain edge table, no temporal, no adjacency | Original SQL path |
| **(b)** Adjacency VT | TVF with `edge_table` pointing to `graph_adjacency` VT | CSR load path |
| **(c)** Temporal | TVF with `timestamp_col`, `time_start`, `time_end` on plain table | `graph_data_load()` with temporal config |
| **(d)** Namespace | TVF with `namespace` on scoped adjacency VT | Namespace-filtered CSR load |
| **(e)** Equivalence | Compare results from paths (a), (b), (c) on identical data | All must produce identical results |

### 13.2 Example Test: graph_bfs

```python
def test_bfs_backward_compat(conn):
    """(a) BFS on plain table — existing behavior unchanged."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("INSERT INTO edges VALUES('A','B'),('B','C'),('C','D')")
    rows = conn.execute("""
        SELECT node, depth, parent FROM graph_bfs
        WHERE edge_table='edges' AND src_col='src' AND dst_col='dst'
          AND start_node='A' AND max_depth=10
    """).fetchall()
    assert [r[0] for r in rows] == ['A', 'B', 'C', 'D']

def test_bfs_adjacency_vt(conn):
    """(b) BFS on graph_adjacency VT — uses CSR cache."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("INSERT INTO edges VALUES('A','B'),('B','C'),('C','D')")
    conn.execute("""
        CREATE VIRTUAL TABLE g USING graph_adjacency(
            edge_table='edges', src_col='src', dst_col='dst'
        )
    """)
    conn.execute("INSERT INTO g(g) VALUES('rebuild')")
    rows = conn.execute("""
        SELECT node, depth, parent FROM graph_bfs
        WHERE edge_table='g' AND src_col='src' AND dst_col='dst'
          AND start_node='A' AND max_depth=10
    """).fetchall()
    assert [r[0] for r in rows] == ['A', 'B', 'C', 'D']

def test_bfs_temporal(conn):
    """(c) BFS with temporal filter on plain table."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT, ts INTEGER)")
    conn.execute("INSERT INTO edges VALUES('A','B',1),('B','C',5),('C','D',10)")
    rows = conn.execute("""
        SELECT node, depth FROM graph_bfs
        WHERE edge_table='edges' AND src_col='src' AND dst_col='dst'
          AND start_node='A' AND max_depth=10
          AND timestamp_col='ts' AND time_start=1 AND time_end=6
    """).fetchall()
    # Only edges with ts in [1, 6]: A->B (ts=1), B->C (ts=5)
    assert [r[0] for r in rows] == ['A', 'B', 'C']

def test_bfs_equivalence(conn):
    """(e) Results from plain table and adjacency VT must match."""
    # ... setup both paths, compare sorted results ...
```

### 13.3 Feature Flag Tests

```python
def test_features_empty(conn):
    """features='' creates only core shadow tables."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("""
        CREATE VIRTUAL TABLE g USING graph_adjacency(
            edge_table='edges', src_col='src', dst_col='dst'
        )
    """)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'g_%'"
    ).fetchall()}
    assert tables == {'g_config', 'g_nodes', 'g_degree', 'g_csr_fwd', 'g_csr_rev', 'g_delta'}

def test_features_components_only(conn):
    """features='components' creates _components but not _sssp or _communities."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("""
        CREATE VIRTUAL TABLE g USING graph_adjacency(
            edge_table='edges', src_col='src', dst_col='dst',
            features='components'
        )
    """)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'g_%'"
    ).fetchall()}
    assert 'g_components' in tables
    assert 'g_sssp' not in tables
    assert 'g_communities' not in tables

def test_features_all(conn):
    """features='sssp,components,communities' creates all shadow tables."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("""
        CREATE VIRTUAL TABLE g USING graph_adjacency(
            edge_table='edges', src_col='src', dst_col='dst',
            features='sssp,components,communities'
        )
    """)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'g_%'"
    ).fetchall()}
    assert 'g_sssp' in tables
    assert 'g_components' in tables
    assert 'g_communities' in tables
```

### 13.4 Rebuild DAG Test

```python
def test_rebuild_cascades(conn):
    """Single rebuild populates all downstream caches."""
    conn.execute("CREATE TABLE edges(src TEXT, dst TEXT)")
    conn.execute("INSERT INTO edges VALUES('A','B'),('B','C'),('A','C')")
    conn.execute("""
        CREATE VIRTUAL TABLE g USING graph_adjacency(
            edge_table='edges', src_col='src', dst_col='dst',
            features='components,communities'
        )
    """)
    conn.execute("INSERT INTO g(g) VALUES('rebuild')")

    # Components populated
    comp_rows = conn.execute("SELECT COUNT(*) FROM g_components").fetchone()[0]
    assert comp_rows == 3

    # Communities populated
    comm_rows = conn.execute("SELECT COUNT(*) FROM g_communities").fetchone()[0]
    assert comm_rows == 3

    # Generation counters consistent
    gen_adj = int(conn.execute(
        "SELECT value FROM g_config WHERE key='generation'"
    ).fetchone()[0])
    gen_at_comp = int(conn.execute(
        "SELECT value FROM g_config WHERE key='gen_adj_at_components'"
    ).fetchone()[0])
    gen_at_comm = int(conn.execute(
        "SELECT value FROM g_config WHERE key='gen_adj_at_communities'"
    ).fetchone()[0])
    assert gen_at_comp == gen_adj
    assert gen_at_comm == gen_adj
```

### 13.5 Performance Verification

Measure each TVF with and without adjacency VT on a benchmark graph (10K nodes, 50K edges):

| Metric | Without VT (SQL path) | With VT (CSR path) | Expected Speedup |
|--------|----------------------|--------------------|----|
| BFS (full traversal) | Baseline | ~5-20x | Eliminates N SQL queries |
| DFS (full traversal) | Baseline | ~5-20x | Same as BFS |
| Shortest path (single pair) | Baseline | ~5-20x (compute) or ~1000x (cache) | SSSP cache is O(1) |
| Components | Baseline | ~2-5x (compute) or ~100x (cache) | Eliminates O(N) lookup |
| PageRank (20 iterations) | Baseline | ~10-50x | Eliminates O(N) lookup per node |

---

## 14. References

### SQLite Virtual Tables
- [The Virtual Table Mechanism of SQLite](https://sqlite.org/vtab.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html) — shadow table pattern reference

### Graph Algorithms
- [PecanPy: Fast Node2Vec (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/19/3377/6184859) — CSR-based graph algorithm optimization
- [Brandes (2001): Betweenness Centrality](https://doi.org/10.1080/0022250X.2001.9990249) — all-pairs SSSP for betweenness
- [Leiden Algorithm (Traag et al., 2019)](https://www.nature.com/articles/s41598-019-41695-z) — community detection
- [Dynamic Leiden (2024)](https://arxiv.org/html/2405.11658v1) — warm-start from cached partition

### Graph Storage
- [GRainDB: Predefined Joins (VLDB 2022)](https://arxiv.org/abs/2108.10540) — trigger-based adjacency
- [DuckPGQ: SQL/PGQ in DuckDB (CIDR 2023)](https://www.cidrdb.org/cidr2023/papers/p66-wolde.pdf) — on-demand CSR, feature flag inspiration
- [SuiteSparse:GraphBLAS](https://dl.acm.org/doi/10.1145/3322125) — delta merge pattern

---

**Prev:** [Phase 4 — Communities Shadow Table](./04_communities_shadow_tables.md) | **Next:** [Phase 6 — Benchmarks](./06_benchmarks.md)
