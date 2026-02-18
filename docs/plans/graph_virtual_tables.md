# Graph Virtual Tables — Persistent Adjacency & Cascading Algorithm Indexes

**Status:** Draft Plan
**Date:** 2026-02-18

## Motivation

Today, every muninn graph algorithm (`graph_betweenness`, `graph_closeness`,
`graph_leiden`, `graph_degree`) independently calls `graph_data_load()`, which
does a full `SELECT src, dst FROM edge_table` and builds an in-memory adjacency
structure from scratch. Running three algorithms on the same edge table means
three full table scans and three adjacency list constructions.

This is analogous to doing a full-text scan on every query instead of using an
FTS index. SQLite already has the pattern for persistent indexes over existing
tables: **shadow tables** (used by FTS5, R-Tree, and muninn's own HNSW).

The key insight: **a persistent `graph_adjacency` virtual table is the graph
equivalent of an FTS index** — it maintains a pre-built, query-ready structure
(CSR) that stays synchronised with the source edge table via triggers. Multiple
algorithm virtual tables can then cascade off this single adjacency index.

### Current Data Flow (wasteful)

```
edge_table ──SELECT──→ graph_data_load() ──→ betweenness (O(VE))
edge_table ──SELECT──→ graph_data_load() ──→ closeness   (O(V²))
edge_table ──SELECT──→ graph_data_load() ──→ Leiden       (O(VE) × iterations)
edge_table ──SELECT──→ graph_data_load() ──→ degree       (O(V+E))
```

### Target Data Flow (cascading virtual tables)

```
edge_table
    │
    │ (triggers → dirty flag / delta log)
    ▼
graph_adjacency VT  ← shadow tables: CSR + node registry + degree sequence
    │
    ├──→ graph_betweenness VT  ← shadow tables: cached (node, centrality)
    ├──→ graph_closeness VT    ← shadow tables: cached (node, centrality)
    ├──→ graph_degree VT       ← trivial: reads degree sequence from adjacency
    ├──→ graph_leiden VT       ← shadow tables: cached (node, community_id, modularity)
    ├──→ graph_select TVF      ← reads CSR for selector evaluation
    └──→ graph_pagerank VT     ← shadow tables: cached (node, rank)
```

## System Boundaries: Virtual Table Primitives

### What Is a Virtual Table?

A virtual table is a C struct (`sqlite3_module`) with ~20 callback methods that
SQLite invokes at well-defined points. The virtual table implementation controls
what data is returned (via `xFilter`/`xNext`/`xColumn`) and how writes are
handled (via `xUpdate`). All state is stored in **shadow tables** — regular
SQLite tables that the virtual table creates and manages.

### Inputs, Triggers, and Outputs

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SQLite Database File                            │
│                                                                     │
│  ┌──────────────┐     AFTER INSERT/     ┌─────────────────────┐    │
│  │  edge_table   │────UPDATE/DELETE────→│  _adjacency_delta    │    │
│  │  (user table) │     (triggers)       │  (shadow table)      │    │
│  └──────────────┘                       └─────────┬───────────┘    │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  graph_adjacency virtual table                               │   │
│  │                                                              │   │
│  │  Shadow tables (regular SQLite B-tree tables):               │   │
│  │  ├── _csr_fwd     BLOB: forward offsets[] + targets[]        │   │
│  │  ├── _csr_rev     BLOB: reverse offsets[] + targets[]        │   │
│  │  ├── _nodes       (idx INTEGER PK, id TEXT)                  │   │
│  │  ├── _degree      (idx INTEGER PK, in_deg, out_deg, w_deg)  │   │
│  │  ├── _config      (key TEXT PK, value)                       │   │
│  │  └── _delta       (src TEXT, dst TEXT, weight REAL, op INT)  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         │  All shadow tables go through the SAME:                   │
│         │  • B-tree storage engine                                  │
│         │  • Page cache                                             │
│         │  • WAL / rollback journal                                 │
│         │  • VFS layer (could be disk, memory, encryption, S3...)   │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    VFS Layer                                  │   │
│  │  Default: unix/win32 (disk files)                            │   │
│  │  Custom:  in-memory, encryption (SEE), compression (ZIPVFS), │   │
│  │           network (S3, etc.)                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### The Virtual Table Lifecycle

| Event | SQLite Calls | Your Code Does |
|-------|-------------|----------------|
| `CREATE VIRTUAL TABLE g USING graph_adjacency(...)` | `xCreate` | Create shadow tables, install triggers on edge table, do initial CSR build |
| Connection reopened / table accessed | `xConnect` | Re-attach to existing shadow tables (do NOT re-create them) |
| `SELECT * FROM g` | `xBestIndex` → `xFilter` → `xNext`/`xEof`/`xColumn` | Check dirty flag, rebuild CSR if needed, iterate results |
| `INSERT INTO g(g) VALUES('rebuild')` | `xUpdate` (command pattern) | Full CSR rebuild from edge table |
| `INSERT INTO g(g) VALUES('incremental_rebuild')` | `xUpdate` (command pattern) | Merge delta into existing CSR |
| `DROP TABLE g` | `xDestroy` | Drop all shadow tables, remove triggers |
| `ALTER TABLE g RENAME TO g2` | `xRename` | Rename shadow tables `g_*` → `g2_*` |
| Transaction commit/rollback | `xSync`/`xCommit`/`xRollback` | Shadow table writes are auto-transactional; only needed for in-memory state sync |

### The Command Pattern (via xUpdate)

FTS5 established a convention: `INSERT INTO vtab(vtab) VALUES('command')` is
routed through `xUpdate`. The virtual table detects the table-name column is
non-NULL and dispatches to command handlers instead of doing a normal insert.

```sql
-- These are administrative commands, NOT data inserts:
INSERT INTO my_graph(my_graph) VALUES('rebuild');              -- full rebuild
INSERT INTO my_graph(my_graph) VALUES('incremental_rebuild');  -- merge delta
INSERT INTO my_graph(my_graph) VALUES('integrity_check');      -- verify CSR consistency
INSERT INTO my_graph(my_graph) VALUES('stats');                -- return build stats
```

### Shadow Tables Are Regular Tables

Shadow tables are **first-class SQLite B-tree tables**. This means:

- **BLOBs work** — CSR arrays (offsets, targets, weights) can be stored as
  raw binary BLOBs. A CSR for 1M edges = ~4MB of int32 arrays = one BLOB column.
- **Indexes work** — `CREATE INDEX` on shadow table columns for fast lookups.
- **WAL works** — concurrent readers see consistent snapshots of the CSR.
- **Transactions work** — shadow table writes within `xUpdate` are automatically
  part of the ambient SQLite transaction. If the transaction rolls back, shadow
  table changes roll back too. No extra code needed.
- **VFS is transparent** — if the database uses an encryption VFS, shadow table
  BLOBs are encrypted. If it uses an in-memory VFS, shadow tables are in memory.
  The virtual table implementation is VFS-agnostic.

### BLOB Packing for CSR Arrays

CSR consists of three arrays. Packing them into BLOBs:

```c
/* Forward CSR: one row in _csr_fwd shadow table */
typedef struct {
    int32_t node_count;           /* V */
    int32_t edge_count;           /* E */
    int32_t offsets[V + 1];       /* offsets[i] = start of node i's neighbors */
    int32_t targets[E];           /* neighbor indices, grouped by source */
    /* optional: double weights[E] for weighted graphs */
} CsrBlob;  /* stored as a single BLOB: sqlite3_result_blob() */
```

**Important:** Do NOT store the entire CSR as a single giant BLOB. SQLite's
page-level copy-on-write means modifying a 100MB BLOB rewrites all pages.
Instead, chunk into blocks:

```sql
CREATE TABLE my_graph_csr_fwd (
    block_id INTEGER PRIMARY KEY,  -- block of ~4096 nodes
    offsets  BLOB,                 -- int32[] for this block's offset slice
    targets  BLOB,                 -- int32[] for this block's edge targets
    weights  BLOB                  -- double[] (NULL if unweighted)
);
```

This enables **block-level incremental rebuilds** (Phase 3: Blocked CSR) —
only rewrite the blocks whose nodes have changed edges.

### xBestIndex / xFilter: How Queries Flow

When a user writes `SELECT * FROM my_graph WHERE node = 'X'`, SQLite calls:

1. **`xBestIndex`** — receives constraints (e.g., `node = ?`). You report
   estimated cost and which constraints you'll handle. Return `idxNum` as a
   bitmask encoding your chosen strategy. `LIMIT`/`OFFSET` arrive as special
   constraint operators (73/74) — you can handle pagination natively.

2. **`xFilter`** — receives the strategy (`idxNum`) and constraint values.
   You check the dirty flag, rebuild if needed, then position the cursor.

3. **`xNext`/`xEof`/`xColumn`** — iterate rows. For `graph_adjacency`,
   this iterates the node registry returning adjacency info. For algorithm
   VTs, this iterates cached results.

### What Cascading Virtual Tables Look Like

A cascading VT reads from another VT's shadow tables instead of the raw edge
table. The dependency chain is managed through dirty flags:

```
edge_table triggers → adjacency._delta (dirty flag on adjacency)
adjacency rebuild   → clears adjacency dirty flag
                    → sets dirty flag on betweenness, closeness, leiden, etc.
betweenness query   → checks own dirty flag
                    → if dirty: reads from adjacency CSR, recomputes, caches
                    → if clean: returns cached results
```

The dirty flag propagation is simple: each downstream VT stores a
`last_adjacency_generation` counter. The adjacency VT increments its
generation on every rebuild. On query, if `my_generation < adjacency_generation`,
results are stale → recompute.

---

## Design: `graph_adjacency` Virtual Table

### SQL Interface

```sql
-- Create an adjacency index over an existing edge table
CREATE VIRTUAL TABLE my_graph USING graph_adjacency(
    edge_table='edges',
    src_col='source',
    dst_col='target',
    weight_col='weight'        -- optional, NULL for unweighted
);

-- The virtual table exposes the node registry + degree sequence
SELECT node, in_degree, out_degree, weighted_degree
FROM my_graph;

-- Filter to specific node
SELECT * FROM my_graph WHERE node = 'critical_node';

-- Administrative commands
INSERT INTO my_graph(my_graph) VALUES('rebuild');
INSERT INTO my_graph(my_graph) VALUES('incremental_rebuild');
INSERT INTO my_graph(my_graph) VALUES('stats');
```

### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node string ID |
| `node_idx` | INTEGER | Internal integer index (0-based) |
| `in_degree` | INTEGER | Count of incoming edges |
| `out_degree` | INTEGER | Count of outgoing edges |
| `weighted_in_degree` | REAL | Sum of incoming edge weights |
| `weighted_out_degree` | REAL | Sum of outgoing edge weights |

### Shadow Tables

| Shadow Table | Schema | Purpose |
|-------------|--------|---------|
| `{name}_nodes` | `(idx INTEGER PK, id TEXT UNIQUE)` | Node registry: string ↔ integer index mapping |
| `{name}_degree` | `(idx INTEGER PK, in_deg INT, out_deg INT, w_in_deg REAL, w_out_deg REAL)` | Pre-computed degree sequence |
| `{name}_csr_fwd` | `(block_id INTEGER PK, offsets BLOB, targets BLOB, weights BLOB)` | Forward CSR (outgoing edges), chunked by block |
| `{name}_csr_rev` | `(block_id INTEGER PK, offsets BLOB, targets BLOB, weights BLOB)` | Reverse CSR (incoming edges), chunked by block |
| `{name}_delta` | `(rowid INTEGER PK, src TEXT, dst TEXT, weight REAL, op INTEGER)` | Pending changes (1=INSERT, 2=DELETE) |
| `{name}_config` | `(key TEXT PK, value TEXT)` | Metadata: edge_table, src_col, dst_col, weight_col, generation, node_count, edge_count, block_size, built_at |

### Triggers on Source Edge Table

Installed in `xCreate`, removed in `xDestroy`:

```sql
-- After INSERT: log the new edge
CREATE TRIGGER {name}_ai AFTER INSERT ON {edge_table} BEGIN
    INSERT INTO {name}_delta(src, dst, weight, op)
    VALUES (NEW.{src_col}, NEW.{dst_col}, NEW.{weight_col}, 1);
END;

-- After DELETE: log the removed edge
CREATE TRIGGER {name}_ad AFTER DELETE ON {edge_table} BEGIN
    INSERT INTO {name}_delta(src, dst, weight, op)
    VALUES (OLD.{src_col}, OLD.{dst_col}, OLD.{weight_col}, 2);
END;

-- After UPDATE: log delete of old + insert of new
CREATE TRIGGER {name}_au AFTER UPDATE ON {edge_table} BEGIN
    INSERT INTO {name}_delta(src, dst, weight, op)
    VALUES (OLD.{src_col}, OLD.{dst_col}, OLD.{weight_col}, 2);
    INSERT INTO {name}_delta(src, dst, weight, op)
    VALUES (NEW.{src_col}, NEW.{dst_col}, NEW.{weight_col}, 1);
END;
```

### CSR Build Process

**Full rebuild** (`'rebuild'` command or first build):

```
1. SELECT DISTINCT src, dst FROM edge_table
   → Build node registry (string → idx) in _nodes
2. Count edges per source node → build offsets[] array
3. Second pass: fill targets[] array (sorted by target idx within each source)
4. Pack into BLOBs, write to _csr_fwd (chunked by block)
5. Repeat for reverse CSR → _csr_rev
6. Compute degree sequence → _degree
7. Clear _delta, increment generation counter
```

**Time complexity:** O(E log E) for sorting targets within each source block.
**Space:** O(V + E) in shadow tables.

### Architecture

```
src/graph_adjacency.c  ── Virtual table implementation (xCreate, xConnect,
    │                      xBestIndex, xFilter, xUpdate, xDestroy)
    ├── src/graph_adjacency.h  ── Registration function + CsrReader API
    ├── src/graph_csr.c        ── CSR build, serialize/deserialize, merge
    ├── src/graph_csr.h        ── CSR types and blob packing
    └── graph_load.c           ── Reuse existing graph loading for initial build
```

---

## Cascading Algorithm Virtual Tables

Cascading algorithm virtual tables (betweenness, closeness, Leiden, PageRank,
SSSP, components, degree) are designed in a separate specification:

**See:** `docs/plans/graph_algorithm_virtual_tables.md`

The core concept: each algorithm VT depends on `graph_adjacency`, reads from
its CSR shadow tables, and caches its own results with generation-counter
staleness detection. The adjacency VT must be solid — with full rebuild, delta
merge, and blocked CSR working correctly and quantitatively benchmarked —
before layering algorithm caching on top.

---

## Incremental Rebuild Strategy

### The Spectrum of Rebuild Strategies

| Strategy | Write Cost | Rebuild Cost | Scan Quality | Complexity |
|----------|-----------|-------------|-------------|------------|
| **Full rebuild** | 0 (no tracking) | O(E) always | Perfect | Trivial |
| **Dirty flag only** | O(1) per write | O(E) when dirty | Perfect | Low |
| **Delta log + merge** | O(1) per write | O(E + P log P) | Perfect after merge | Medium |
| **Blocked CSR + delta** | O(1) per write | O(affected blocks) | Perfect within blocks | Medium-High |
| **PCSR (packed memory)** | O(log² E) per edge | 0 (always ready) | ~2× degraded (gaps) | High |

### Recommended: Three-Phase Incremental Strategy

#### Phase A: Cached Full Rebuild (dirty flag)

The simplest useful improvement over today's load-and-discard:

- Triggers on the edge table set a `dirty` flag in `_config`
- On query: if dirty, full rebuild from edge table → CSR. If clean, use cache.
- `INSERT INTO g(g) VALUES('rebuild')` forces a rebuild

This alone eliminates redundant rebuilds when running multiple algorithms
back-to-back.

#### Phase B: Delta Log + Merge

Triggers append to `_delta` instead of just setting a dirty flag. On query:

```
if delta is empty:
    use cached CSR (zero cost)
else if delta_count < threshold (e.g., 10% of edge_count):
    incremental_rebuild: merge delta into existing CSR
else:
    full rebuild (delta is too large, merge overhead not worth it)
```

**The merge algorithm** (GraphBLAS-style):

```
1. Read existing _csr_fwd BLOBs → in-memory CSR arrays
2. Read _delta rows, separate into inserts[] and deletes[]
3. For each delete: mark target in CSR as tombstone (-1)
4. For each insert: append to per-source insert buffer
5. For each source node with changes:
   a. Compact: remove tombstones from existing neighbors
   b. Merge-sort: merge new inserts with existing neighbors
   c. Update offsets
6. Repack into BLOBs, write back to shadow tables
7. Update _degree for affected nodes
8. Clear _delta, increment generation
```

**Time complexity:** O(E + P log P) where P = delta size. When P << E, the
merge is dominated by the O(E) CSR read — still a full scan, but no SQL query
against the edge table.

**Key advantage over full rebuild:** The merge reads from shadow table BLOBs
(sequential binary data) instead of running `SELECT src, dst FROM edge_table`
(SQL parsing, B-tree traversal, type conversion). For a 1M-edge graph, the
BLOB read is ~4MB sequential I/O vs. 1M B-tree lookups.

The `'incremental_rebuild'` command forces a delta merge regardless of threshold:

```sql
INSERT INTO my_graph(my_graph) VALUES('incremental_rebuild');
```

#### Phase C: Blocked CSR + Block-Level Merge

Partition nodes into blocks of ~4096. Each block's CSR is stored as a separate
row in `_csr_fwd`. When the delta only affects nodes in a few blocks, only
those blocks are rebuilt:

```
1. Read _delta, group changes by block_id (= node_idx / block_size)
2. For each affected block:
   a. Read that block's BLOB from _csr_fwd
   b. Apply inserts/deletes for nodes in this block
   c. Repack and write back
3. For unaffected blocks: no I/O at all
```

This reduces the merge cost from O(E) to O(affected_blocks × block_size) when
changes are localized. For a 1M-edge graph with 100 changed edges affecting 5
blocks: ~20K edges processed instead of 1M.

**Trade-off:** Block boundaries add a small indirection overhead during graph
algorithm execution. When a BFS crosses from block 3 to block 7, the target
lookup requires reading a different BLOB. In practice, SQLite's page cache
makes this fast (blocks fit in a few pages each).

---

## Implementation Plan

### Phase 1: `graph_adjacency` with Cached Full Rebuild

The foundation. A virtual table that builds a CSR from an edge table and
caches it in shadow tables. Queries check a dirty flag and rebuild if needed.

**Files to create:**
- `src/graph_adjacency.c` — virtual table implementation
- `src/graph_adjacency.h` — registration function + CsrReader API
- `src/graph_csr.c` — CSR build, BLOB serialization/deserialization
- `src/graph_csr.h` — CSR types and blob packing
- `test/test_graph_csr.c` — C unit tests for CSR build + round-trip
- `pytests/test_graph_adjacency.py` — Python integration tests

**Files to modify:**
- `src/muninn.c` — register `graph_adjacency` module
- `Makefile` — add new source files

**Scope:**
- `CREATE VIRTUAL TABLE` creates shadow tables + triggers on edge table
- Full CSR build from edge table (forward + reverse)
- Pre-computed degree sequence in `_degree` shadow table
- Dirty flag: triggers set dirty, queries check + rebuild if dirty
- `'rebuild'` command via xUpdate (command pattern)
- `DROP TABLE` cleans up shadow tables + removes triggers
- `xRename` renames shadow tables + triggers
- Expose node registry + degree via `SELECT`
- `xBestIndex` supports `node = ?` constraint for O(1) lookup
- Single monolithic CSR BLOB per direction (blocked CSR comes in Phase 3)

**Estimated:** ~800 lines of C (vtab glue ~300, CSR build/serialize ~300,
tests ~200)

### Phase 2: Delta Log + Incremental Merge

Add change tracking so that small edge table mutations don't require a full
CSR rebuild.

**Scope:**
- `_delta` shadow table with INSERT/DELETE tracking
- Triggers append to `_delta` (replacing dirty-flag-only triggers from Phase 1)
- `'incremental_rebuild'` command: merge delta into existing CSR BLOBs
- Auto-threshold: if `delta_count < 10% of edge_count`, merge; otherwise
  fall back to full rebuild
- On-query auto-merge: check delta count in `xFilter`, merge if stale
- The merge algorithm (GraphBLAS-style):
  1. Read existing CSR BLOBs → in-memory arrays
  2. Apply tombstones for deletes, insert buffers for inserts
  3. Compact + merge-sort per source node
  4. Repack BLOBs, update `_degree` for affected nodes
  5. Clear `_delta`, increment generation counter

**Estimated:** ~400 lines (merge algorithm ~250, trigger changes ~50,
threshold logic ~100)

### Phase 3: Blocked CSR + Block-Level Rebuild

Partition the CSR into blocks so that incremental merges only rewrite the
affected blocks. This is the most impactful optimisation for large graphs
with localised changes.

**Scope:**
- Partition node ID space into blocks of ~4096 nodes
- Each block = one row in `_csr_fwd` / `_csr_rev` shadow tables
- `'rebuild'` rewrites all blocks (full rebuild)
- `'incremental_rebuild'` groups delta by block, rewrites only affected blocks
- Block-aware CSR reader API for algorithm consumption:
  - Random access: given node index, compute `block_id = idx / block_size`,
    read that block's BLOB, index into local offset
  - Sequential scan: iterate blocks in order for full-graph algorithms
- Unaffected blocks: zero I/O during incremental merge
- For a 1M-edge graph with 100 changed edges in 5 blocks: ~20K edges
  processed instead of 1M

**Migration from Phase 1/2:** The monolithic CSR BLOB from Phase 1 is
replaced by per-block BLOBs. Phase 2's merge logic is adapted to operate
on individual blocks. The shadow table schema changes from one row to
V/block_size rows, but the trigger and delta tracking remain the same.

**Estimated:** ~400 lines (block partitioning ~150, block-level merge ~150,
block-aware reader ~100)

### Phase 4: Migrate Existing TVFs to Use `graph_adjacency`

Make existing algorithm TVFs optionally read from a `graph_adjacency` VT
instead of calling `graph_data_load()` on every invocation.

**Scope:**
- Modify `graph_betweenness`, `graph_closeness`, `graph_leiden`,
  `graph_degree`, `graph_pagerank` TVFs to accept a `graph_adjacency`
  name as the first parameter
- Backward-compatible: old `(edge_table, src, dst, ...)` syntax still works
- Detection: if first arg matches a registered `graph_adjacency` VT name,
  read CSR from shadow tables; otherwise, fall back to `graph_data_load()`
- Modify `graph_select` (from `graph_search_syntax.md`) to also accept
  a `graph_adjacency` name
- Expose a C API (`CsrReader`) for algorithms to consume blocked CSR:
  `csr_reader_open()`, `csr_reader_neighbors()`, `csr_reader_close()`

**Estimated:** ~200 lines per TVF modification (~1200 total for 6 TVFs)

### Phase 5: C-Level Benchmark Suite

Quantitatively measure time, memory, and disk usage across all approaches.
Each approach remains available (not replaced) so benchmarks are always
reproducible and users can choose the right trade-off.

**Benchmark matrix:**

| Approach | Build Method | What's Measured |
|----------|-------------|-----------------|
| **TVF only** (baseline) | `graph_data_load()` per query | Time per query, peak RSS, no disk overhead |
| **CSR + full rebuild** (Phase 1) | Full edge table scan → CSR BLOBs | Rebuild time, query time, shadow table disk size |
| **CSR + delta merge** (Phase 2) | Delta accumulation → merge | Write overhead (trigger cost), merge time vs full rebuild, break-even delta % |
| **Blocked CSR** (Phase 3) | Block-level merge | Merge time for N changes in B blocks vs full merge, block read overhead during algorithms |

**Measurements per benchmark:**

| Metric | How Measured |
|--------|-------------|
| **Wall time** | `clock_gettime(CLOCK_MONOTONIC)` around each operation |
| **Peak memory** | `getrusage(RUSAGE_SELF).ru_maxrss` before/after |
| **Disk usage** | `SELECT SUM(pgsize) FROM dbstat WHERE name LIKE '{vtab}_%'` for shadow tables |
| **Trigger overhead** | Time for N inserts with vs without triggers installed |
| **Cache hit ratio** | Count of full rebuilds vs cached reads over a workload |

**Test workloads:**

| Workload | V | E | Pattern |
|----------|---|---|---------|
| Small | 500 | 2,000 | Random Erdos-Renyi |
| Medium | 10,000 | 50,000 | Power-law (Barabasi-Albert) |
| Large | 100,000 | 500,000 | Power-law |
| Incremental | 10,000 | 50,000 + 100 inserts | Delta merge vs full rebuild |
| Localised | 100,000 | 500,000 + 100 inserts in 5 blocks | Blocked merge vs full merge |

**Files to create:**
- `test/bench_graph_adjacency.c` — C benchmark harness
- `benchmarks/scripts/bench_adjacency.py` — Python benchmark wrapper
  (uses same JSONL output format as existing benchmarks)

**Output:** JSONL results in `benchmarks/results/` for comparison.
Columns: approach, workload, operation, wall_time_ms, peak_rss_kb,
disk_bytes, edge_count, delta_count.

**Estimated:** ~500 lines of C (benchmark harness ~300, workload generators
~200) + ~200 lines of Python wrapper

### Future: Cascading Algorithm Virtual Tables

Once the `graph_adjacency` foundation is solid and quantitatively validated,
the next step is cascading algorithm virtual tables that cache expensive
computations (betweenness, closeness, Leiden, PageRank) with
generation-counter-based staleness detection.

**See:** `docs/plans/graph_algorithm_virtual_tables.md`

---

## Risks & Considerations

### BLOB Size and Page-Level Copy-on-Write

SQLite stores BLOBs across multiple B-tree pages (default 4096 bytes each).
Modifying a BLOB rewrites all pages that contain it. For a 1M-edge CSR:
- `targets[]` = 4MB = ~1000 pages
- Full BLOB rewrite on every rebuild = ~1000 page writes

**Mitigation:** Blocked CSR (Phase 3) limits rewrites to affected blocks.
Each block's BLOB is ~16KB (4096 nodes × 4 bytes) = ~4 pages.

### Trigger Overhead on Write-Heavy Workloads

Each edge INSERT fires a trigger that INSERTs into `_delta`. For bulk loads
(e.g., 1M edges), this doubles the write volume.

**Mitigation options:**
1. Drop triggers before bulk load, rebuild after: `DROP TRIGGER g_ai; ... INSERT 1M rows ...; INSERT INTO g(g) VALUES('rebuild');`
2. Deferred trigger execution (SQLite's `PRAGMA defer_foreign_keys` doesn't
   apply to regular triggers, but the delta approach naturally batches)
3. Document that bulk loads should use `'rebuild'` instead of trigger-based
   incremental tracking

### Memory During Rebuild

The CSR build itself requires O(V + E) memory. For the same large graphs
that blow out `graph_data_load()`, the rebuild will have the same peak memory.

**Mitigation:** The CSR build can be done in a streaming two-pass fashion:
- Pass 1: `SELECT src, COUNT(*) FROM edges GROUP BY src` → build offsets[]
- Pass 2: `SELECT src, dst FROM edges ORDER BY src` → fill targets[]

This avoids building the full adjacency list in memory. The working set is
O(V) for offsets + O(block_size) for the current block's targets.

### Cascading Rebuild Cost (Future: Algorithm VTs)

When cascading algorithm VTs are added (see `graph_algorithm_virtual_tables.md`),
an edge table change will invalidate `graph_adjacency`, which in turn
invalidates all downstream VTs. A query on `graph_betweenness` would trigger
a full O(VE) recomputation even if only one edge changed.

**Mitigation:** The generation counter makes invalidation explicit. Users
control when rebuilds happen via the `'rebuild'` command pattern. Lazy rebuild
means the cost is only paid when results are actually queried. The blocked
CSR from Phase 3 helps here: if only one block changed, algorithms that
operate per-block can limit their recomputation scope.

### Concurrent Access

SQLite's WAL mode allows concurrent reads. A long-running betweenness
computation reading from adjacency shadow tables won't block other readers.
However, a rebuild (which writes to shadow tables) will block under WAL mode
until readers finish.

### Shadow Table Protection

With `SQLITE_DBCONFIG_DEFENSIVE` enabled, shadow tables become read-only to
user SQL. Users cannot accidentally corrupt the CSR by writing directly to
shadow tables. The virtual table's `xShadowName` method declares which suffixes
are protected: `_nodes`, `_degree`, `_csr_fwd`, `_csr_rev`, `_delta`, `_config`.

### Backward Compatibility

Existing TVF syntax `graph_betweenness('edges', 'src', 'dst', ...)` must
continue to work. The adjacency-based syntax is an enhancement, not a
replacement. Detection logic: if the first argument matches a registered
`graph_adjacency` VT name, use CSR; otherwise, fall back to
`graph_data_load()`.

---

## Comparison: Academic Approaches

| Approach | Used By | Fit for SQLite Extension |
|----------|---------|------------------------|
| **Index-free adjacency** (pointer chains) | Neo4j | No — requires custom storage engine |
| **CSR in shadow tables** (our approach) | Kuzu (similar), GRainDB | **Yes** — uses existing SQLite B-tree |
| **On-demand CSR** (no persistence) | DuckPGQ | Partial — what we do today |
| **PCSR** (packed memory arrays) | Academic | No — 2× scan degradation, high complexity |
| **BACH** (LSM-tree graph) | Academic | No — storage engine inside storage engine |
| **Delta + merge** (our incremental strategy) | GraphBLAS | **Yes** — simple, preserves scan quality |
| **Blocked CSR** (chunked) | Kuzu NodeGroups | **Yes** — maps well to shadow table rows |

## References

### SQLite Virtual Tables
- [The Virtual Table Mechanism of SQLite](https://sqlite.org/vtab.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html)
- [sqlite3_module API](https://www.sqlite.org/c3ref/module.html)
- [sqlite3_index_info](https://www.sqlite.org/c3ref/index_info.html)

### Graph Storage
- [GRainDB: Predefined Joins (VLDB 2022)](https://arxiv.org/abs/2108.10540)
- [A+ Indexes: Lightweight Adjacency Lists (Kuzu/Waterloo)](https://arxiv.org/abs/2004.00130)
- [Kuzu Database Internals](https://docs.kuzudb.com/developer-guide/database-internal/)
- [DuckPGQ: SQL/PGQ in DuckDB (CIDR 2023)](https://www.cidrdb.org/cidr2023/papers/p66-wolde.pdf)

### Incremental CSR
- [Packed CSR (Wheatman & Xu, 2018)](https://itshelenxu.github.io/files/papers/pcsr.pdf)
- [BACH: Bridging Adjacency List and CSR (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p1509-miao.pdf)
- [LSMGraph: Multi-Level CSR (SIGMOD 2024)](https://arxiv.org/html/2411.06392v1)
- [SuiteSparse:GraphBLAS Algorithm 1000](https://dl.acm.org/doi/10.1145/3322125)

### Graph Algorithms
- [Leiden Algorithm (Traag et al., 2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [Dynamic Leiden (2024)](https://arxiv.org/html/2405.11658v1)
- [Neo4j GDS: Graph Projections](https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/)
- [PecanPy: Fast Node2Vec (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/19/3377/6184859)

### Graph Database Design
- [Neo4j: Index-Free Adjacency](https://neo4j.com/blog/cypher-and-gql/native-vs-non-native-graph-technology/)
- [MV4PG: Materialized Views for Property Graphs (2024)](https://arxiv.org/html/2411.18847v1)
