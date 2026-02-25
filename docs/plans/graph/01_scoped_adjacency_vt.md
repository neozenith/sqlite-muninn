# Phase 1: Scoped Adjacency Virtual Table

**Date:** 2026-02-24
**Status:** Not started
**Depends on:** None (foundational phase)
**Blocks:** Phases 2, 3, 4, 5, 6

---

## 1. Overview

This phase adds namespace/scope support to the existing `graph_adjacency` virtual table.
A single VT instance partitions its CSR, node index space, degree cache, and delta log
by a composite namespace key derived from user-specified scope columns on the source
edge table.

### Why This Is Mandatory

Real-world edge tables contain multiple disjoint graphs keyed by scope columns
(e.g., `project_id`, `session_id`, `tenant_id`). Without namespace support, users must
either:

1. Create a separate `graph_adjacency` VT per scope combination (impractical for dynamic scopes), or
2. Rebuild the entire CSR when any scope's edges change (wasteful for large multi-tenant tables).

Namespace support makes each scope's CSR independent: inserts to scope A trigger rebuilds
only for scope A. Queries filter by namespace and read only the relevant CSR partition.
All downstream shadow tables (SSSP, components, communities in Phases 2-4) inherit the
namespace partitioning from this phase.

No SQLite graph extension supports namespace-scoped adjacency indexes. This is a
differentiating feature for muninn.

### Deliverables

1. `namespace_cols` parameter on `CREATE VIRTUAL TABLE`
2. `{name}_namespace` shadow table (namespace registry)
3. All existing shadow tables keyed by `namespace_id`
4. Triggers that capture scope column values and route deltas to the correct namespace
5. Per-namespace full rebuild and incremental rebuild
6. Namespace filter on xBestIndex/xFilter (hidden column)
7. Updated `graph_data_load_from_adjacency()` accepting an optional namespace key
8. Backward compatibility: omitting `namespace_cols` is identical to current behavior

---

## 2. Design

### 2.1 SQL Syntax

```sql
-- Scoped adjacency: partition by project_id and session_id
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges',
    src_col='src',
    dst_col='dst',
    weight_col='weight',
    namespace_cols='project_id,session_id'
);

-- Unscoped (backward compatible): equivalent to current behavior
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges',
    src_col='src',
    dst_col='dst'
);
```

The `namespace_cols` parameter is a comma-separated list of column names from the edge
table. Each column must pass `id_validate()`. The order matters: it defines the
composite key ordering for the namespace hash.

### 2.2 Namespace Key Computation

A composite scope combination (e.g., `project_id=42, session_id='abc'`) is mapped to
a single `namespace_id INTEGER` via:

1. **Text representation:** Concatenate the TEXT values of each scope column with a
   NUL byte separator: `"42\0abc"`. SQLite coerces all values to TEXT for this purpose.
2. **Hash:** Compute DJB2 hash of the concatenated byte string (reusing the existing
   `graph_str_hash()` from `graph_common.h`, extended to handle embedded NUL bytes).
3. **Registry lookup:** Look up the hash in `{name}_namespace`. If not found, INSERT
   a new row with the next available `namespace_id` (auto-increment).
4. **Collision handling:** The registry stores the full composite key text alongside the
   hash. On lookup, compare the stored key to resolve collisions (linear probe in the
   registry table, which is a B-tree and handles this naturally via the `scope_key` UNIQUE
   constraint).

The `namespace_id` is a dense integer starting at 0. When `namespace_cols` is omitted,
there is a single implicit namespace with `namespace_id = 0` and `scope_key = ''`.

### 2.3 Shadow Table Schema Changes

All shadow tables gain a `namespace_id` column as part of their primary key. The
namespace registry is a new shadow table.

### 2.4 Data Flow

```
Edge INSERT into source table
    │
    ▼
Trigger fires: captures NEW.src, NEW.dst, NEW.weight, scope col values
    │
    ▼
INSERT INTO {name}_delta(namespace_id, src, dst, weight, op, scope_key)
    │                     ▲
    │      namespace_id resolved at query time (not trigger time)
    │      trigger stores raw scope_key; namespace_id resolved during rebuild
    ▼
adj_ensure_fresh() called on next query
    │
    ├── delta count > 0 → rebuild (full or incremental)
    │   │
    │   ├── Group deltas by scope_key → resolve to namespace_id
    │   │
    │   ├── Per-namespace: load CSR blocks, apply deltas, store
    │   │
    │   └── Update per-namespace generation in _config
    │
    └── delta count = 0 → serve from shadow tables
        │
        └── Filter by namespace_id in _nodes JOIN _degree query
```

**Key design decision:** Triggers store the raw `scope_key` (concatenated scope column
values) in the delta table rather than resolving `namespace_id` at trigger time. This
avoids the trigger needing to call a UDF or query the namespace registry. Resolution
happens during rebuild, which is the only time `namespace_id` matters.

---

## 3. Implementation Steps

### Step 1: Parse `namespace_cols` in xCreate/xConnect

**File:** `src/graph_adjacency.c`, lines 66-144 (`AdjParams` and `parse_adjacency_params`)

Add `namespace_cols` to the `AdjParams` struct and parse it in `parse_adjacency_params()`.
Split on commas, validate each column name with `id_validate()`, and store as a
dynamically-allocated array of strings.

```c
/* In AdjParams (line ~67) */
typedef struct {
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;
    char **namespace_cols;   /* array of column name strings, NULL if unscoped */
    int namespace_col_count; /* 0 if unscoped */
} AdjParams;
```

In the parsing loop (line ~91), add:

```c
} else if (strncmp(arg, "namespace_cols=", 15) == 0) {
    const char *val = strip_quotes(arg + 15, buf, (int)sizeof(buf));
    /* Parse comma-separated list into params->namespace_cols[] */
    /* Each column name must pass id_validate() */
}
```

Store parsed namespace columns in `AdjVtab` (see Step 4 for struct changes).

### Step 2: Create `{name}_namespace` Shadow Table

**File:** `src/graph_adjacency.c`, function `adjacency_create_shadow_tables()` (line ~150)

Add the namespace registry table:

```sql
CREATE TABLE IF NOT EXISTS "{name}_namespace" (
    namespace_id INTEGER PRIMARY KEY,
    scope_key    TEXT UNIQUE NOT NULL,
    scope_hash   INTEGER NOT NULL
);
```

When `namespace_cols` is omitted, insert the default namespace during xCreate:

```sql
INSERT INTO "{name}_namespace"(namespace_id, scope_key, scope_hash)
VALUES (0, '', 0);
```

Also add `"namespace"` to the shadow name list in `adj_xShadowName()` (line ~1401),
the `drop_shadow_tables()` suffix list (line ~210), and the `adj_xRename()` suffix
list (line ~1378).

### Step 3: Add `namespace_id` to Shadow Tables

**File:** `src/graph_adjacency.c`, function `adjacency_create_shadow_tables()` (line ~150)

Modify all shadow table schemas:

**`_nodes`** (currently line ~164):
```sql
CREATE TABLE IF NOT EXISTS "{name}_nodes" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    id           TEXT NOT NULL,
    PRIMARY KEY (namespace_id, idx)
);
CREATE INDEX IF NOT EXISTS "{name}_nodes_id"
    ON "{name}_nodes"(namespace_id, id);
```

The `id TEXT UNIQUE` constraint becomes a composite unique constraint on
`(namespace_id, id)` via the index. Different namespaces may contain the same
node ID with different `idx` values, since each namespace has an independent
node index space.

**`_degree`** (currently line ~173):
```sql
CREATE TABLE IF NOT EXISTS "{name}_degree" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    in_deg       INTEGER,
    out_deg      INTEGER,
    w_in_deg     REAL,
    w_out_deg    REAL,
    PRIMARY KEY (namespace_id, idx)
);
```

**`_csr_fwd`** (currently line ~183):
```sql
CREATE TABLE IF NOT EXISTS "{name}_csr_fwd" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);
```

**`_csr_rev`** (currently line ~192):
```sql
CREATE TABLE IF NOT EXISTS "{name}_csr_rev" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);
```

**`_delta`** (currently line ~201):
```sql
CREATE TABLE IF NOT EXISTS "{name}_delta" (
    rowid     INTEGER PRIMARY KEY,
    scope_key TEXT NOT NULL DEFAULT '',
    src       TEXT,
    dst       TEXT,
    weight    REAL,
    op        INTEGER
);
CREATE INDEX IF NOT EXISTS "{name}_delta_scope"
    ON "{name}_delta"(scope_key);
```

The delta table uses `scope_key` (raw text) rather than `namespace_id` because
triggers cannot resolve namespace IDs without a UDF. The scope_key is resolved
to a namespace_id during rebuild.

**`_config`** (currently line ~155):

The config table remains namespace-global for parameters like `edge_table`, `src_col`,
`dst_col`, `weight_col`, `namespace_cols`. Per-namespace metadata uses a composite
key convention:

```
key = "ns:{namespace_id}:generation"
key = "ns:{namespace_id}:node_count"
key = "ns:{namespace_id}:edge_count"
key = "ns:{namespace_id}:block_size"
```

Global keys remain unchanged: `edge_table`, `src_col`, `dst_col`, `weight_col`,
`namespace_cols`, `generation` (global generation counter).

### Step 4: Modify Triggers to Capture Scope Column Values

**File:** `src/graph_adjacency.c`, function `install_triggers()` (line ~223)

The trigger must capture scope column values from the `NEW` row and concatenate them
into a `scope_key` stored in the delta table.

When `namespace_cols` is NULL (unscoped), triggers are identical to current behavior
with `scope_key = ''`.

When `namespace_cols = ['project_id', 'session_id']`, the AFTER INSERT trigger becomes:

```sql
CREATE TRIGGER IF NOT EXISTS "{name}_ai" AFTER INSERT ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(NEW."project_id" AS TEXT) || X'00' || CAST(NEW."session_id" AS TEXT),
        NEW."src",
        NEW."dst",
        NEW."weight",
        1
    );
END;
```

The `X'00'` (NUL byte) separator prevents ambiguity: `('1', '23')` and `('12', '3')`
produce different scope_keys (`'1\x0023'` vs `'12\x003'`).

The `install_triggers()` function signature changes to accept the namespace column list:

```c
static int install_triggers(sqlite3 *db, const char *vtab_name,
                            const char *edge_table, const char *src_col,
                            const char *dst_col, const char *weight_col,
                            const char **namespace_cols, int namespace_col_count);
```

The scope_key expression is built dynamically by iterating `namespace_cols`:

```c
/* Build scope_key expression for trigger SQL */
char *scope_expr;
if (namespace_col_count == 0) {
    scope_expr = sqlite3_mprintf("''");
} else {
    /* Start with first column */
    scope_expr = sqlite3_mprintf("CAST(NEW.\"%w\" AS TEXT)", namespace_cols[0]);
    for (int i = 1; i < namespace_col_count; i++) {
        char *prev = scope_expr;
        scope_expr = sqlite3_mprintf("%s || X'00' || CAST(NEW.\"%w\" AS TEXT)",
                                     prev, namespace_cols[i]);
        sqlite3_free(prev);
    }
}
```

The same pattern applies to AFTER DELETE (using `OLD.`) and AFTER UPDATE (both `OLD.`
and `NEW.`).

### Step 5: Modify Full Rebuild to Iterate Per-Namespace

**File:** `src/graph_adjacency.c`, function `adj_full_rebuild()` (line ~565)

The full rebuild becomes a two-level operation:

1. **Outer loop:** Iterate over distinct `scope_key` values in `_delta`, plus all
   existing `namespace_id` values in `_namespace`. This covers both namespaces that
   have pending deltas and namespaces that need rebuilding.
2. **Inner loop (per namespace):** Load edges from the source table filtered by the
   scope columns matching this namespace's scope values, build CSR, store in shadow
   tables with the appropriate `namespace_id`.

New function: `adj_full_rebuild_namespace()` — rebuilds a single namespace.

```c
static int adj_full_rebuild_namespace(AdjVtab *vtab, int64_t namespace_id,
                                      const char *scope_key);
```

This function:
1. Builds a `WHERE` clause from the scope_key: parse the NUL-separated values
   back into individual column values, generate
   `WHERE "project_id" = ?1 AND "session_id" = ?2`.
2. Executes `SELECT src, dst [, weight] FROM edge_table WHERE <scope_filter>`
3. Builds GraphData, CSR, stores in shadow tables with `namespace_id` prefix
4. Updates per-namespace config keys

The top-level `adj_full_rebuild()` becomes:

```c
static int adj_full_rebuild(AdjVtab *vtab) {
    if (vtab->namespace_col_count == 0) {
        /* Unscoped: single namespace, same as current behavior */
        return adj_full_rebuild_namespace(vtab, 0, "");
    }
    /* Scoped: iterate all namespaces */
    /* 1. Collect all known namespace_ids from _namespace table */
    /* 2. Collect any new scope_keys from _delta not yet in _namespace */
    /* 3. Register new scope_keys → new namespace_ids */
    /* 4. For each namespace_id: rebuild */
    /* 5. Clear delta log */
    /* 6. Increment global generation */
}
```

### Step 6: Modify Incremental Rebuild to Rebuild Only Affected Namespaces

**File:** `src/graph_adjacency.c`, function `adj_incremental_rebuild()` (line ~721)

The incremental rebuild gains a namespace-grouping step:

1. **Group deltas by scope_key:** `SELECT DISTINCT scope_key FROM _delta`
2. **Resolve each scope_key to namespace_id:** look up in `_namespace`, auto-register
   if new
3. **Per affected namespace:** load that namespace's CSR blocks, apply deltas for
   that namespace only, store updated blocks
4. **Unaffected namespaces:** zero I/O (no blocks loaded, no blocks written)

New function: `adj_incremental_rebuild_namespace()`:

```c
static int adj_incremental_rebuild_namespace(AdjVtab *vtab, int64_t namespace_id,
                                              const char *scope_key);
```

The delta filtering adds a `WHERE scope_key = ?` clause to the delta scan:

```sql
SELECT src, dst, weight, op FROM "{name}_delta" WHERE scope_key = ?
```

Node registry loading becomes namespace-scoped:

```sql
SELECT idx, id FROM "{name}_nodes" WHERE namespace_id = ? ORDER BY idx
```

The threshold for incremental vs full rebuild is evaluated per-namespace:

```c
int64_t ns_delta = delta_count_for_namespace(vtab->db, vtab->vtab_name, scope_key);
int64_t ns_edges = config_get_int(vtab->db, vtab->vtab_name,
                                   ns_edge_count_key, 0);
int64_t threshold = ns_edges / 10;
if (threshold < 10) threshold = 10;

if (ns_delta <= threshold)
    adj_incremental_rebuild_namespace(vtab, namespace_id, scope_key);
else
    adj_full_rebuild_namespace(vtab, namespace_id, scope_key);
```

### Step 7: Add Namespace Filter to xBestIndex/xFilter

**File:** `src/graph_adjacency.c`, functions `adj_xBestIndex()` (line ~1200) and
`adj_xFilter()` (line ~1230)

Add a hidden `namespace` column to the VT schema. The column enum becomes:

```c
enum {
    ADJ_COL_NODE = 0,
    ADJ_COL_NODE_IDX,
    ADJ_COL_IN_DEGREE,
    ADJ_COL_OUT_DEGREE,
    ADJ_COL_W_IN_DEGREE,
    ADJ_COL_W_OUT_DEGREE,
    ADJ_COL_NAMESPACE,  /* NEW: hidden, for filtering */
    ADJ_COL_COMMAND,    /* hidden: same name as table, for command pattern */
    ADJ_NUM_COLS
};
```

The vtab schema declaration (line ~1050) becomes:

```c
char *schema = sqlite3_mprintf(
    "CREATE TABLE x("
    "node TEXT, node_idx INTEGER, "
    "in_degree INTEGER, out_degree INTEGER, "
    "weighted_in_degree REAL, weighted_out_degree REAL, "
    "namespace HIDDEN, "
    "\"%w\" HIDDEN)",
    argv[2]);
```

**xBestIndex changes:**

The `idxNum` bitmask encoding becomes:

| Bit | Meaning |
|-----|---------|
| 0 (0x01) | node = ? (point lookup) |
| 1 (0x02) | namespace = ? (namespace filter) |

```c
static int adj_xBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    int has_node = -1;
    int has_namespace = -1;

    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        if (pInfo->aConstraint[i].op != SQLITE_INDEX_CONSTRAINT_EQ) continue;

        if (pInfo->aConstraint[i].iColumn == ADJ_COL_NODE)
            has_node = i;
        else if (pInfo->aConstraint[i].iColumn == ADJ_COL_NAMESPACE)
            has_namespace = i;
    }

    int idx_num = 0;
    int argv_idx = 1;

    if (has_node >= 0) {
        pInfo->aConstraintUsage[has_node].argvIndex = argv_idx++;
        pInfo->aConstraintUsage[has_node].omit = 1;
        idx_num |= 0x01;
    }
    if (has_namespace >= 0) {
        pInfo->aConstraintUsage[has_namespace].argvIndex = argv_idx++;
        pInfo->aConstraintUsage[has_namespace].omit = 1;
        idx_num |= 0x02;
    }

    pInfo->idxNum = idx_num;

    /* Cost estimation */
    if (idx_num & 0x01) {
        pInfo->estimatedCost = 1.0;
        pInfo->estimatedRows = 1;
    } else if (idx_num & 0x02) {
        pInfo->estimatedCost = 100.0;   /* single namespace scan */
        pInfo->estimatedRows = 100;
    } else {
        pInfo->estimatedCost = 1000.0;  /* all namespaces */
        pInfo->estimatedRows = 1000;
    }
    return SQLITE_OK;
}
```

**xFilter changes:**

Parse `idxNum` to determine which filters are active, then build the appropriate SQL:

```c
static int adj_xFilter(sqlite3_vtab_cursor *pCursor, int idxNum,
                       const char *idxStr, int argc, sqlite3_value **argv) {
    /* ... existing cleanup ... */

    int has_node = (idxNum & 0x01);
    int has_namespace = (idxNum & 0x02);

    /* Determine argv positions */
    int node_argv = -1, ns_argv = -1;
    int pos = 0;
    if (has_node) node_argv = pos++;
    if (has_namespace) ns_argv = pos++;

    /* Resolve namespace_id from namespace filter value */
    int64_t namespace_id = 0; /* default: unscoped */
    if (has_namespace && ns_argv >= 0 && ns_argv < argc) {
        /* Look up namespace_id from scope_key or direct namespace_id */
        /* Accept either INTEGER (direct namespace_id) or TEXT (scope_key) */
        if (sqlite3_value_type(argv[ns_argv]) == SQLITE_INTEGER) {
            namespace_id = sqlite3_value_int64(argv[ns_argv]);
        } else {
            const char *scope_key = (const char *)sqlite3_value_text(argv[ns_argv]);
            namespace_id = resolve_namespace_id(vtab->db, vtab->vtab_name, scope_key);
        }
    }

    /* Build query with namespace filter */
    if (has_node) {
        sql = sqlite3_mprintf(
            "SELECT n.idx, n.id, COALESCE(d.in_deg,0), COALESCE(d.out_deg,0), "
            "COALESCE(d.w_in_deg,0.0), COALESCE(d.w_out_deg,0.0), n.namespace_id "
            "FROM \"%w_nodes\" n "
            "LEFT JOIN \"%w_degree\" d ON n.namespace_id=d.namespace_id AND n.idx=d.idx "
            "WHERE n.namespace_id=?1 AND n.id=?2",
            vtab->vtab_name, vtab->vtab_name);
    } else if (has_namespace) {
        sql = sqlite3_mprintf(
            "SELECT n.idx, n.id, COALESCE(d.in_deg,0), COALESCE(d.out_deg,0), "
            "COALESCE(d.w_in_deg,0.0), COALESCE(d.w_out_deg,0.0), n.namespace_id "
            "FROM \"%w_nodes\" n "
            "LEFT JOIN \"%w_degree\" d ON n.namespace_id=d.namespace_id AND n.idx=d.idx "
            "WHERE n.namespace_id=?1 ORDER BY n.idx",
            vtab->vtab_name, vtab->vtab_name);
    } else {
        sql = sqlite3_mprintf(
            "SELECT n.idx, n.id, COALESCE(d.in_deg,0), COALESCE(d.out_deg,0), "
            "COALESCE(d.w_in_deg,0.0), COALESCE(d.w_out_deg,0.0), n.namespace_id "
            "FROM \"%w_nodes\" n "
            "LEFT JOIN \"%w_degree\" d ON n.namespace_id=d.namespace_id AND n.idx=d.idx "
            "ORDER BY n.namespace_id, n.idx",
            vtab->vtab_name, vtab->vtab_name);
    }
    /* Bind parameters... */
}
```

The `adj_xColumn` function (line ~1293) gains a case for `ADJ_COL_NAMESPACE`:

```c
case ADJ_COL_NAMESPACE:
    sqlite3_result_value(ctx, sqlite3_column_value(cur->stmt, 6));
    break;
```

### Step 8: Update `graph_data_load_from_adjacency()`

**File:** `src/graph_adjacency.h` (line ~39) and `src/graph_adjacency.c` (line ~1532)

The function signature changes to accept an optional namespace key:

```c
/* New signature */
int graph_data_load_from_adjacency(sqlite3 *db, const char *vtab_name,
                                   const char *namespace_key,
                                   GraphData *g, char **pzErrMsg);
```

When `namespace_key` is NULL, the function loads namespace_id=0 (backward compatible
for unscoped VTs and for callers that don't need scoping).

When `namespace_key` is non-NULL:
1. Resolve `namespace_key` to `namespace_id` via `_namespace` table lookup
2. Filter all shadow table reads by `namespace_id`
3. If stale (deltas exist for this scope_key), fall back to loading from the source
   edge table with appropriate `WHERE` clause on scope columns

The stale check changes from global delta count to per-scope delta count:

```c
/* Check staleness for this specific namespace */
int64_t dc;
if (namespace_key) {
    dc = delta_count_for_scope(db, vtab_name, namespace_key);
} else {
    dc = delta_count(db, vtab_name);
}
```

The `load_graph_from_shadow()` helper (line ~1458) gains a `namespace_id` parameter
and filters all its queries:

```c
static int load_graph_from_shadow(sqlite3 *db, const char *name,
                                  int64_t namespace_id,
                                  GraphData *g, char **pzErrMsg);
```

### Step 9: Update `is_graph_adjacency()` and All TVFs That Call It

**File:** `src/graph_adjacency.c` (line ~1414), `src/graph_centrality.c`, `src/graph_community.c`

The `is_graph_adjacency()` function itself does not change its signature. It still
returns 1 if the name corresponds to a graph_adjacency VT.

However, every call site that follows the pattern:

```c
if (config.edge_table && is_graph_adjacency(vtab->db, config.edge_table)) {
    rc = graph_data_load_from_adjacency(vtab->db, config.edge_table, &g, &errmsg);
} else {
    rc = graph_data_load(vtab->db, &config, &g, &errmsg);
}
```

Must be updated to pass a namespace key. This requires the TVFs to accept a new
optional hidden parameter `namespace` that propagates through to the load call.

**Affected call sites:**

| File | Line | TVF |
|------|------|-----|
| `src/graph_centrality.c` | ~476 | `graph_degree` |
| `src/graph_centrality.c` | ~710 | `graph_betweenness` |
| `src/graph_centrality.c` | ~1024 | `graph_closeness` |
| `src/graph_community.c` | ~549 | `graph_leiden` |

Each TVF gains an optional hidden `namespace` parameter column. The pattern becomes:

```c
const char *namespace_key = graph_safe_text(argv_namespace);  /* NULL if not provided */

if (config.edge_table && is_graph_adjacency(vtab->db, config.edge_table)) {
    rc = graph_data_load_from_adjacency(vtab->db, config.edge_table,
                                         namespace_key, &g, &errmsg);
} else {
    rc = graph_data_load(vtab->db, &config, &g, &errmsg);
}
```

When `namespace_key` is NULL and the VT has multiple namespaces, the load function
returns an error instructing the caller to specify a namespace. This prevents silent
cross-namespace graph merging.

---

## 4. AdjVtab Struct Changes

**File:** `src/graph_adjacency.c`, lines 33-42

Current struct:

```c
typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;
    int64_t generation;
} AdjVtab;
```

New struct:

```c
typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;      /* virtual table name (for shadow table prefixes) */
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;     /* NULL if unweighted */
    int64_t generation;   /* global generation counter */

    /* Namespace support */
    char **namespace_cols;     /* array of scope column names, NULL if unscoped */
    int namespace_col_count;   /* 0 if unscoped */
} AdjVtab;
```

The `adj_xDisconnect()` function (line ~1155) must free `namespace_cols`:

```c
static int adj_xDisconnect(sqlite3_vtab *pVTab) {
    AdjVtab *vtab = (AdjVtab *)pVTab;
    sqlite3_free(vtab->vtab_name);
    sqlite3_free(vtab->edge_table);
    sqlite3_free(vtab->src_col);
    sqlite3_free(vtab->dst_col);
    sqlite3_free(vtab->weight_col);
    for (int i = 0; i < vtab->namespace_col_count; i++)
        sqlite3_free(vtab->namespace_cols[i]);
    sqlite3_free(vtab->namespace_cols);
    sqlite3_free(vtab);
    return SQLITE_OK;
}
```

---

## 5. Shadow Table Schemas (Complete)

All `CREATE TABLE` statements for the new schema. These replace the current statements
in `adjacency_create_shadow_tables()`.

```sql
-- Config: global KV store (unchanged schema, new key conventions)
CREATE TABLE IF NOT EXISTS "{name}_config" (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Namespace registry (NEW)
CREATE TABLE IF NOT EXISTS "{name}_namespace" (
    namespace_id INTEGER PRIMARY KEY,
    scope_key    TEXT UNIQUE NOT NULL,
    scope_hash   INTEGER NOT NULL
);

-- Node registry: scoped string ID <-> integer index
CREATE TABLE IF NOT EXISTS "{name}_nodes" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    id           TEXT NOT NULL,
    PRIMARY KEY (namespace_id, idx)
);
CREATE INDEX IF NOT EXISTS "{name}_nodes_id"
    ON "{name}_nodes"(namespace_id, id);

-- Degree sequence: scoped
CREATE TABLE IF NOT EXISTS "{name}_degree" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    in_deg       INTEGER,
    out_deg      INTEGER,
    w_in_deg     REAL,
    w_out_deg    REAL,
    PRIMARY KEY (namespace_id, idx)
);

-- Forward CSR: scoped blocked storage
CREATE TABLE IF NOT EXISTS "{name}_csr_fwd" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);

-- Reverse CSR: scoped blocked storage
CREATE TABLE IF NOT EXISTS "{name}_csr_rev" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);

-- Delta log: stores raw scope_key for later resolution
CREATE TABLE IF NOT EXISTS "{name}_delta" (
    rowid     INTEGER PRIMARY KEY,
    scope_key TEXT NOT NULL DEFAULT '',
    src       TEXT,
    dst       TEXT,
    weight    REAL,
    op        INTEGER
);
CREATE INDEX IF NOT EXISTS "{name}_delta_scope"
    ON "{name}_delta"(scope_key);
```

---

## 6. Trigger SQL

### 6.1 Unscoped (namespace_cols omitted)

Identical to current behavior, with the addition of `scope_key = ''`:

```sql
-- AFTER INSERT
CREATE TRIGGER IF NOT EXISTS "{name}_ai" AFTER INSERT ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', NEW."{src_col}", NEW."{dst_col}", NEW."{weight_expr}", 1);
END;

-- AFTER DELETE
CREATE TRIGGER IF NOT EXISTS "{name}_ad" AFTER DELETE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', OLD."{src_col}", OLD."{dst_col}", OLD."{weight_expr}", 2);
END;

-- AFTER UPDATE
CREATE TRIGGER IF NOT EXISTS "{name}_au" AFTER UPDATE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', OLD."{src_col}", OLD."{dst_col}", OLD."{weight_expr}", 2);
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', NEW."{src_col}", NEW."{dst_col}", NEW."{weight_expr}", 1);
END;
```

### 6.2 Scoped (namespace_cols = 'project_id,session_id')

The scope_key expression concatenates scope column values with NUL separators:

```sql
-- AFTER INSERT
CREATE TRIGGER IF NOT EXISTS "{name}_ai" AFTER INSERT ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(NEW."project_id" AS TEXT) || X'00' || CAST(NEW."session_id" AS TEXT),
        NEW."{src_col}",
        NEW."{dst_col}",
        NEW."{weight_expr}",
        1
    );
END;

-- AFTER DELETE
CREATE TRIGGER IF NOT EXISTS "{name}_ad" AFTER DELETE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(OLD."project_id" AS TEXT) || X'00' || CAST(OLD."session_id" AS TEXT),
        OLD."{src_col}",
        OLD."{dst_col}",
        OLD."{weight_expr}",
        2
    );
END;

-- AFTER UPDATE
CREATE TRIGGER IF NOT EXISTS "{name}_au" AFTER UPDATE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(OLD."project_id" AS TEXT) || X'00' || CAST(OLD."session_id" AS TEXT),
        OLD."{src_col}",
        OLD."{dst_col}",
        OLD."{weight_expr}",
        2
    );
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(NEW."project_id" AS TEXT) || X'00' || CAST(NEW."session_id" AS TEXT),
        NEW."{src_col}",
        NEW."{dst_col}",
        NEW."{weight_expr}",
        1
    );
END;
```

### 6.3 Trigger Generation Code

The `install_triggers()` function builds the scope_key expression dynamically:

```c
static char *build_scope_key_expr(const char **namespace_cols, int count,
                                  const char *row_prefix) {
    /* row_prefix is "NEW" or "OLD" */
    if (count == 0)
        return sqlite3_mprintf("''");

    char *expr = sqlite3_mprintf("CAST(%s.\"%w\" AS TEXT)", row_prefix, namespace_cols[0]);
    for (int i = 1; i < count; i++) {
        char *prev = expr;
        expr = sqlite3_mprintf("%s || X'00' || CAST(%s.\"%w\" AS TEXT)",
                               prev, row_prefix, namespace_cols[i]);
        sqlite3_free(prev);
    }
    return expr;
}
```

---

## 7. Namespace Key Computation

### 7.1 Hash Function

Extend the existing DJB2 hash (`graph_str_hash()` in `graph_common.h`) to handle
byte strings with embedded NUL bytes:

```c
/* DJB2 hash for byte string of known length (handles embedded NUL) */
static GRAPH_UNUSED unsigned int graph_bytes_hash(const unsigned char *data, int len) {
    unsigned int h = 5381;
    for (int i = 0; i < len; i++)
        h = h * 33 + data[i];
    return h;
}
```

For scope_key strings containing NUL separators, use `graph_bytes_hash()` with the
full byte length. SQLite stores BLOBish TEXT with embedded NULs correctly; use
`sqlite3_column_bytes()` to get the true length.

### 7.2 Auto-Registration

When a delta rebuild encounters a scope_key not yet in `_namespace`, it auto-registers:

```c
static int64_t resolve_or_register_namespace(sqlite3 *db, const char *vtab_name,
                                              const char *scope_key, int scope_key_len) {
    /* 1. Look up existing */
    sqlite3_stmt *stmt;
    char *sql = sqlite3_mprintf(
        "SELECT namespace_id FROM \"%w_namespace\" WHERE scope_key = ?",
        vtab_name);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return -1;

    sqlite3_bind_text(stmt, 1, scope_key, scope_key_len, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t ns_id = sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return ns_id;
    }
    sqlite3_finalize(stmt);

    /* 2. Register new namespace */
    unsigned int hash = graph_bytes_hash((const unsigned char *)scope_key, scope_key_len);
    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_namespace\"(scope_key, scope_hash) VALUES (?, ?)",
        vtab_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return -1;

    sqlite3_bind_text(stmt, 1, scope_key, scope_key_len, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, (int)hash);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) return -1;
    return sqlite3_last_insert_rowid(db);
}
```

The `namespace_id` is the SQLite-generated `rowid` from the auto-increment primary key,
making it dense and monotonically increasing.

### 7.3 Scope Key Decomposition

To generate the `WHERE` clause for per-namespace edge loading, decompose the scope_key
back into individual column values by splitting on NUL bytes:

```c
/* Split scope_key by NUL separator into an array of string pointers.
 * Pointers point into the scope_key buffer (not copied).
 * Returns count of segments. */
static int decompose_scope_key(const char *scope_key, int scope_key_len,
                                const char **parts, int max_parts) {
    int count = 0;
    const char *p = scope_key;
    const char *end = scope_key + scope_key_len;

    while (p < end && count < max_parts) {
        parts[count++] = p;
        /* Advance past this segment and its NUL separator */
        while (p < end && *p != '\0') p++;
        p++; /* skip NUL separator */
    }
    return count;
}
```

Then generate the WHERE clause:

```c
/* WHERE "project_id" = ?1 AND "session_id" = ?2 */
static char *build_scope_where(const char **namespace_cols, int count) {
    if (count == 0) return sqlite3_mprintf("1=1");

    char *where = sqlite3_mprintf("\"%w\" = ?%d", namespace_cols[0], 1);
    for (int i = 1; i < count; i++) {
        char *prev = where;
        where = sqlite3_mprintf("%s AND \"%w\" = ?%d", prev, namespace_cols[i], i + 1);
        sqlite3_free(prev);
    }
    return where;
}
```

---

## 8. xBestIndex/xFilter Changes

### 8.1 Current Pattern

The existing `adj_xBestIndex` (line ~1200) uses a simple boolean for node point lookup:

- `idxNum = 0`: full scan
- `idxNum = 1`: node = ? (point lookup, argvIndex=1)

### 8.2 New Pattern

A bitmask encoding supports both filters independently:

| idxNum Bits | Filter | argvIndex |
|-------------|--------|-----------|
| None (0x00) | Full scan (all namespaces, all nodes) | -- |
| Bit 0 (0x01) | `node = ?` | Sequential |
| Bit 1 (0x02) | `namespace = ?` | Sequential |
| Both (0x03) | `node = ?` AND `namespace = ?` | Sequential |

**argvIndex assignment:** Uses the two-pass pattern from `graph_best_index_common()`
in `graph_common.h` (line ~62). Arguments are assigned sequentially in column order:
if both `node` (col 0) and `namespace` (col 6) have EQ constraints, `node` gets
argvIndex=1 and `namespace` gets argvIndex=2.

### 8.3 Cost Estimation

| idxNum | estimatedCost | estimatedRows | Rationale |
|--------|--------------|---------------|-----------|
| 0x00 | 1000.0 | 1000 | Full scan across all namespaces |
| 0x01 | 1.0 | 1 | Point lookup (single node, could be any namespace) |
| 0x02 | 100.0 | 100 | Single namespace scan |
| 0x03 | 1.0 | 1 | Namespace + node (most specific) |

### 8.4 xFilter SQL Generation

The shadow table JOIN query in xFilter must adapt to the constraint combination.
All four cases are shown in Step 7 above. The key addition is that when `namespace`
is constrained, the JOIN condition includes `n.namespace_id = d.namespace_id` and
the WHERE includes `n.namespace_id = ?`.

When `namespace` is not constrained on a multi-namespace VT, the query returns
results from all namespaces. The `namespace` output column lets the caller
distinguish which namespace each row belongs to.

---

## 9. Backward Compatibility

When `namespace_cols` is omitted from `CREATE VIRTUAL TABLE`:

| Aspect | Behavior |
|--------|----------|
| `_namespace` table | Created with single row: `(0, '', 0)` |
| `_nodes` PK | `(0, idx)` — always namespace_id=0 |
| `_degree` PK | `(0, idx)` — always namespace_id=0 |
| `_csr_fwd` PK | `(0, block_id)` — always namespace_id=0 |
| `_csr_rev` PK | `(0, block_id)` — always namespace_id=0 |
| `_delta.scope_key` | Always `''` |
| Triggers | `scope_key` expression is `''` (literal empty string) |
| `adj_full_rebuild` | Single iteration over namespace_id=0 |
| `adj_incremental_rebuild` | Single iteration over namespace_id=0 |
| xFilter without namespace constraint | Returns all nodes (same as current) |
| `graph_data_load_from_adjacency(db, name, NULL, &g, &err)` | Loads namespace_id=0 |
| All existing TVF calls | Pass NULL for namespace_key (loads namespace_id=0) |

**Migration path for existing databases:** When xConnect detects an existing VT
(shadow tables already exist), it checks whether `_namespace` table exists. If not,
it runs a one-time migration:

1. Create `_namespace` table with default row `(0, '', 0)`
2. `ALTER TABLE "{name}_nodes" ADD COLUMN namespace_id INTEGER NOT NULL DEFAULT 0`
   (SQLite supports ADD COLUMN; the DEFAULT fills existing rows)
3. Same for `_degree`, `_csr_fwd`, `_csr_rev`
4. `ALTER TABLE "{name}_delta" ADD COLUMN scope_key TEXT NOT NULL DEFAULT ''`
5. Recreate primary keys (requires table rebuild in SQLite, done via
   `CREATE TABLE new ... AS SELECT ... ; DROP old; ALTER TABLE new RENAME TO old`)

Alternatively, since this is a pre-1.0 extension, document that existing VTs must
be dropped and recreated. The simpler approach is preferred:

```c
/* In adj_init() for xConnect path */
if (!namespace_table_exists(db, argv[2])) {
    /* Pre-namespace VT: drop and recreate shadow tables */
    drop_shadow_tables(db, argv[2]);
    adjacency_create_shadow_tables(db, argv[2]);
    /* Force full rebuild on next query */
    config_set_int(db, argv[2], "generation", 0);
}
```

---

## 10. Verification Steps

### Test 1: Scoped VT Creation and Basic Operation

```sql
-- Setup
CREATE TABLE edges (
    src TEXT, dst TEXT, weight REAL,
    project_id INTEGER, session_id TEXT
);

CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst',
    weight_col='weight', namespace_cols='project_id,session_id'
);

-- Insert edges into two different scopes
INSERT INTO edges VALUES ('A', 'B', 1.0, 1, 'alpha');
INSERT INTO edges VALUES ('B', 'C', 2.0, 1, 'alpha');
INSERT INTO edges VALUES ('X', 'Y', 3.0, 2, 'beta');

-- Verify namespace registry
SELECT * FROM g_namespace;
-- Expected: two rows (namespace_id=1, scope_key='1\x00alpha'),
--                    (namespace_id=2, scope_key='2\x00beta')
-- (plus namespace_id=0 default if present)

-- Query specific namespace
SELECT node, in_degree, out_degree FROM g WHERE namespace = 1;
-- Expected: A(0,1), B(1,1), C(1,0)

SELECT node, in_degree, out_degree FROM g WHERE namespace = 2;
-- Expected: X(0,1), Y(1,0)

-- Query all namespaces
SELECT namespace, node, out_degree FROM g;
-- Expected: 6 rows total (3 from ns1, 2 from ns2, plus potential default)
```

### Test 2: Independent CSR Blocks Per Namespace

```sql
-- After Test 1, verify CSR blocks are partitioned
SELECT namespace_id, COUNT(*) as blocks FROM g_csr_fwd GROUP BY namespace_id;
-- Expected: two groups, each with their own block count

-- Verify node indices are independent per namespace
SELECT namespace_id, idx, id FROM g_nodes ORDER BY namespace_id, idx;
-- Expected: namespace 1 has idx 0,1,2 for A,B,C
--           namespace 2 has idx 0,1 for X,Y
```

### Test 3: Scoped Incremental Rebuild

```sql
-- Insert into only one scope
INSERT INTO edges VALUES ('C', 'D', 1.5, 1, 'alpha');

-- This should only trigger rebuild for namespace_id=1
-- Verify by checking delta table
SELECT scope_key, src, dst FROM g_delta;
-- Expected: one row with scope_key matching project_id=1, session_id='alpha'

-- Force incremental rebuild
INSERT INTO g(g) VALUES('incremental_rebuild');

-- Verify namespace 1 has new node D
SELECT node, out_degree FROM g WHERE namespace = 1;
-- Expected: A(1), B(1), C(1), D(0)

-- Verify namespace 2 unchanged
SELECT node, out_degree FROM g WHERE namespace = 2;
-- Expected: X(1), Y(0)
```

### Test 4: TVF with Namespace Filter

```sql
-- Centrality on specific namespace
SELECT node, centrality FROM graph_degree(
    'g', 'src', 'dst', NULL, NULL, NULL, NULL, NULL, 1
) ORDER BY centrality DESC;
-- (last hidden param is namespace)
-- Expected: only nodes from namespace 1
```

### Test 5: Backward Compatibility (No namespace_cols)

```sql
-- Create unscoped VT
CREATE TABLE simple_edges (src TEXT, dst TEXT);
INSERT INTO simple_edges VALUES ('P', 'Q'), ('Q', 'R');

CREATE VIRTUAL TABLE g2 USING graph_adjacency(
    edge_table='simple_edges', src_col='src', dst_col='dst'
);

-- Verify works exactly as before
SELECT node, in_degree, out_degree FROM g2;
-- Expected: P(0,1), Q(1,1), R(1,0)

-- Verify _namespace has single default row
SELECT * FROM g2_namespace;
-- Expected: (0, '', 0)
```

### Test 6: Cross-Namespace Protection

```sql
-- Attempting to load from multi-namespace VT without specifying namespace
-- should return an error (not silently merge all namespaces)
-- This is verified by calling graph_data_load_from_adjacency with NULL namespace_key
-- on a VT that has namespace_col_count > 0 and multiple registered namespaces
```

### Test 7: Scope Key Collision Resistance

```sql
-- Insert edges with scope values that could collide without NUL separator
INSERT INTO edges VALUES ('M', 'N', 1.0, 12, '3');   -- scope_key: '12\x003'
INSERT INTO edges VALUES ('O', 'P', 1.0, 1, '23');   -- scope_key: '1\x0023'

-- Verify these are distinct namespaces
SELECT COUNT(*) FROM g_namespace WHERE scope_key IN (
    CAST(12 AS TEXT) || X'00' || '3',
    '1' || X'00' || '23'
);
-- Expected: 2 (distinct entries)
```

### C Unit Tests

Add to `test/test_graph_adjacency.c`:

1. `test_namespace_parse` — verify `namespace_cols` parameter parsing with 0, 1, and 3 columns
2. `test_scope_key_build` — verify NUL-separated scope_key construction
3. `test_scope_key_decompose` — verify round-trip: build -> decompose -> match original values
4. `test_namespace_hash` — verify `graph_bytes_hash()` produces different hashes for NUL-containing variants

### Python Integration Tests

Add to `pytests/test_graph_adjacency.py`:

1. `test_scoped_adjacency_creation` — create VT with namespace_cols, verify shadow tables
2. `test_scoped_insert_and_query` — insert edges with different scopes, verify filtered results
3. `test_scoped_incremental_rebuild` — insert into one scope, verify only that scope rebuilds
4. `test_unscoped_backward_compat` — verify current behavior is unchanged
5. `test_namespace_tvf_integration` — verify centrality/community TVFs accept namespace param

---

## 11. References

### SQLite Virtual Tables
- [The Virtual Table Mechanism of SQLite](https://sqlite.org/vtab.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html) (command pattern, shadow tables)
- [xBestIndex Method](https://sqlite.org/vtab.html#xbestindex) (idxNum bitmask)
- [xShadowName](https://sqlite.org/vtab.html#xshadowname) (shadow table registration)

### Graph Storage Approaches
- [GRainDB: Predefined Joins (VLDB 2022)](https://arxiv.org/abs/2108.10540) — trigger-based graph acceleration
- [A+ Indexes: Lightweight Adjacency Lists (Kuzu/Waterloo)](https://arxiv.org/abs/2004.00130) — blocked columnar storage inspiration
- [DuckPGQ: SQL/PGQ in DuckDB (CIDR 2023)](https://www.cidrdb.org/cidr2023/papers/p66-wolde.pdf) — schema-based namespace separation
- [SuiteSparse:GraphBLAS Algorithm 1000](https://dl.acm.org/doi/10.1145/3322125) — delta+merge pattern

### Namespace/Projection Prior Art
- [Neo4j GDS: Graph Projections](https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/) — label-based graph partitioning
- [MV4PG: Materialized Views for Property Graphs (2024)](https://arxiv.org/html/2411.18847v1) — scoped graph views

### Competitor Projects (None Support Scoped Adjacency)
- [GraphQLite (Rust, Cypher)](https://github.com/colliery-io/graphqlite) — in-memory, no namespace support
- [sqlite-graph (C99, Cypher alpha)](https://github.com/agentflare-ai/sqlite-graph) — no namespace support
- [simple-graph (pure SQL)](https://github.com/dpapathanasiou/simple-graph) — no namespace support
