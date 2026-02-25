# Phase 2: SSSP Shadow Tables

**Date:** 2026-02-24
**Status:** Plan (not started)
**Depends on:** [Phase 1: Scoped Adjacency VT](./01_scoped_adjacency_vt.md)
**Depended on by:** [Phase 5: TVF/VT Integration](./05_tvf_vt_integration.md)

---

## 1. Overview

This phase delivers two things:

1. **A shared SSSP module** (`graph_sssp.c` / `graph_sssp.h`) extracted from the
   static functions currently buried in `graph_centrality.c`. This gives betweenness,
   closeness, and any future algorithm a single, well-tested SSSP implementation.

2. **An optional `_sssp` shadow table** in the `graph_adjacency` virtual table,
   activated by `features='sssp'`. When enabled, the adjacency VT caches all-pairs
   shortest-path distances and path counts as packed BLOBs. A generation counter
   tracks staleness relative to the CSR adjacency, so betweenness and closeness
   TVFs can read from cache when fresh and recompute when stale.

### Why This Matters

Betweenness centrality (Brandes 2001) runs all-pairs SSSP: O(VE) unweighted,
O(VE + V^2 log V) weighted. Closeness centrality also runs all-pairs SSSP: O(V^2)
unweighted. When both are queried on the same unchanged graph, SSSP is computed
twice. Each invocation allocates and discards V arrays of size V.

Extracting SSSP into a shared module eliminates code duplication. Caching the
all-pairs distance matrix (when the graph is small enough) eliminates redundant
computation across TVF calls.

### Complexity Analysis

| Algorithm | SSSP Runs | Per-Run Cost (unweighted) | Per-Run Cost (weighted) | Total |
|-----------|-----------|---------------------------|-------------------------|-------|
| Betweenness (exact) | V | O(V + E) BFS | O(E + V log V) Dijkstra | O(VE) / O(VE + V^2 log V) |
| Betweenness (approx) | sqrt(V) | O(V + E) | O(E + V log V) | O(sqrt(V) * E) |
| Closeness | V | O(V + E) BFS | O(E + V log V) Dijkstra | O(VE) / O(VE + V^2 log V) |
| **Both (no cache)** | **2V** | — | — | **2x total** |
| **Both (with cache)** | **V (first call) + 0 (second call)** | — | — | **1x total** |

For a graph with V=5000, E=50000: each all-pairs SSSP takes ~5000 BFS runs at
O(55000) each = ~275M operations. Caching saves 275M operations on every subsequent
betweenness or closeness query.

---

## 2. Architecture Decision: What to Cache

### Option A: Cache All-Pairs dist[] Matrix

Store `dist[source][v]` and `sigma[source][v]` for all V sources as packed
double BLOBs in the `_sssp` shadow table. Total storage: O(V^2) for distances
plus O(V^2) for sigma (path counts).

**Pros:**
- Closeness becomes O(V) — just sum the cached dist row per source
- Betweenness avoids recomputing dist[] and sigma[] — only pred[] and
  back-propagation need live computation
- Second query on the same graph is instantaneous for closeness

**Cons:**
- O(V^2) storage: infeasible above ~10K nodes (see Section 8)
- Rebuild cost is still O(VE) — caching doesn't reduce first-query time
- sigma[] is only needed by betweenness, wasted for closeness-only workloads

### Option B: Extract Shared Module Only (No Caching)

Move `sssp_bfs`, `sssp_dijkstra`, and supporting types into `graph_sssp.c`.
Both TVFs call the shared functions. No shadow table, no generation tracking.

**Pros:**
- Zero overhead — pure code reuse
- No storage cost, no staleness checking
- Simpler implementation

**Cons:**
- Every betweenness/closeness query still recomputes from scratch
- No amortization of repeated queries on the same graph

### Option C: Cache Per-Source SSSP On Demand

Only cache SSSP results for sources that have been queried. Populate lazily.

**Pros:**
- Amortized cost — only cache what is actually queried
- Lower storage than full all-pairs

**Cons:**
- Betweenness and closeness both need ALL sources — this degenerates to
  Option A for the primary use case
- Complex lazy population logic with partial staleness

### Recommendation

**Option B as baseline, Option A gated behind `features='sssp'` with a node
count threshold.**

The shared SSSP module (Option B) is always built. It provides clean code
separation with zero overhead. The `_sssp` shadow table (Option A) is only
created when:

1. The user explicitly enables `features='sssp'` in the `CREATE VIRTUAL TABLE`
2. The graph has V <= `max_sssp_nodes` (default: 10000)

When V exceeds the threshold, the shadow table exists but is not populated.
The TVFs fall back to live computation using the shared module. This prevents
accidental multi-GB shadow tables on large graphs.

Option C is rejected because the primary consumers (betweenness and closeness)
both require all-pairs SSSP, making lazy per-source caching equivalent to
full caching with extra bookkeeping.

---

## 3. New File: graph_sssp.c / graph_sssp.h

### Types to Extract from graph_centrality.c

The following types and functions are currently `static` in `graph_centrality.c`
(lines 96-335). They must be moved to the new module:

```c
/* From graph_centrality.c lines 103-165 */
typedef struct { int node; double dist; } DPQEntry;
typedef struct { DPQEntry *entries; int size; int capacity; } DoublePQ;

/* From graph_centrality.c lines 177-205 */
typedef struct { int *items; int count; int capacity; } IntList;

/* From graph_centrality.c line 168 */
static int double_eq(double a, double b);
```

### graph_sssp.h — Public API

```c
/*
 * graph_sssp.h -- Single-source shortest paths for graph algorithms
 *
 * Provides BFS (unweighted) and Dijkstra (weighted) SSSP with full
 * Brandes working-set output: dist[], sigma[], pred[], and BFS order
 * stack. Used by betweenness, closeness, and SSSP shadow table cache.
 */
#ifndef GRAPH_SSSP_H
#define GRAPH_SSSP_H

#include "graph_load.h"

/* ── Double-precision priority queue (Dijkstra) ─────────── */

typedef struct {
    int node;
    double dist;
} SsspPQEntry;

typedef struct {
    SsspPQEntry *entries;
    int size;
    int capacity;
} SsspPQ;

void sssp_pq_init(SsspPQ *pq, int capacity);
void sssp_pq_destroy(SsspPQ *pq);
void sssp_pq_push(SsspPQ *pq, int node, double dist);
SsspPQEntry sssp_pq_pop(SsspPQ *pq);

/* ── Predecessor list (dynamic int array) ────────────────── */

typedef struct {
    int *items;
    int count;
    int capacity;
} SsspPredList;

void sssp_pred_init(SsspPredList *l);
void sssp_pred_push(SsspPredList *l, int val);
void sssp_pred_clear(SsspPredList *l);
void sssp_pred_destroy(SsspPredList *l);

/* ── Epsilon comparison for tie detection ────────────────── */

int sssp_double_eq(double a, double b);

/* ── SSSP working set ────────────────────────────────────── */

/*
 * Caller-allocated working arrays for SSSP. Reusable across
 * multiple source invocations (betweenness/closeness loop).
 *
 * All arrays are of size node_count. Caller is responsible for
 * allocation and deallocation.
 */
typedef struct {
    int node_count;
    double *dist;           /* [V] shortest distance from source */
    double *sigma;          /* [V] shortest-path count (NULL ok for closeness-only) */
    SsspPredList *pred;     /* [V] predecessor lists (NULL ok for closeness-only) */
    int *stack;             /* [V] BFS/Dijkstra order (NULL ok for closeness-only) */
    int stack_size;         /* number of entries in stack[] after SSSP */
} SsspWorkingSet;

/*
 * Allocate a working set for V nodes.
 * If needs_brandes is true, allocates sigma[], pred[], and stack[].
 * If needs_brandes is false, only allocates dist[].
 * Returns 0 on success, -1 on allocation failure.
 */
int sssp_working_set_init(SsspWorkingSet *ws, int node_count, int needs_brandes);

/* Free all memory owned by the working set. */
void sssp_working_set_destroy(SsspWorkingSet *ws);

/* ── Core SSSP functions ─────────────────────────────────── */

/*
 * BFS-based SSSP for unweighted graphs.
 *
 * Sets ws->dist[v] = shortest distance from source.
 * If ws->sigma is non-NULL, sets shortest-path counts.
 * If ws->pred is non-NULL, fills predecessor lists.
 * If ws->stack is non-NULL, fills BFS order stack.
 *
 * direction: "forward", "reverse", or "both" (NULL = "forward").
 */
void sssp_bfs(const GraphData *g, int source,
              SsspWorkingSet *ws, const char *direction);

/*
 * Dijkstra-based SSSP for weighted graphs.
 * Same interface and semantics as sssp_bfs.
 */
void sssp_dijkstra(const GraphData *g, int source,
                   SsspWorkingSet *ws, const char *direction);

/*
 * Run the appropriate SSSP variant based on g->has_weights.
 * Convenience wrapper that dispatches to sssp_bfs or sssp_dijkstra.
 */
void sssp_run(const GraphData *g, int source,
              SsspWorkingSet *ws, const char *direction);

#endif /* GRAPH_SSSP_H */
```

### Key Design Decisions

1. **Renamed types.** `DoublePQ` becomes `SsspPQ`, `IntList` becomes
   `SsspPredList`. The `sssp_` prefix prevents collisions with the existing
   `PriorityQueue` in `priority_queue.h` (used by HNSW).

2. **SsspWorkingSet with optional fields.** Closeness only needs `dist[]`.
   By making `sigma`, `pred`, and `stack` optional (NULL-checked in the
   SSSP functions), closeness avoids allocating O(V) arrays it never reads.
   This directly addresses the current waste identified in the gap analysis.

3. **Ownership semantics.** The caller owns the `SsspWorkingSet` and can
   reuse it across the V iterations of the all-pairs loop. This matches the
   current pattern in `bet_filter` and `clo_filter` where arrays are
   `malloc`'d once and reused.

---

## 4. Shadow Table Schema

When `features='sssp'` is enabled in the `CREATE VIRTUAL TABLE` statement,
the adjacency VT creates the following additional shadow table:

### `{name}_sssp` Table

```sql
CREATE TABLE IF NOT EXISTS "{name}_sssp" (
    namespace_id  INTEGER NOT NULL,
    source_idx    INTEGER NOT NULL,
    distances     BLOB NOT NULL,        -- double[V], little-endian
    sigma         BLOB,                 -- double[V], little-endian (NULL for closeness-only)
    PRIMARY KEY (namespace_id, source_idx)
);
```

**Column semantics:**

| Column | Type | Description |
|--------|------|-------------|
| `namespace_id` | INTEGER | Scope partition from Phase 1. 0 for non-scoped VTs |
| `source_idx` | INTEGER | Node index of the SSSP source (0..V-1) |
| `distances` | BLOB | `double[V]` packed little-endian. `dist[i]` = shortest distance from `source_idx` to node `i`. -1.0 = unreachable |
| `sigma` | BLOB | `double[V]` packed little-endian. `sigma[i]` = number of shortest paths. NULL if only closeness has been computed |

**What is NOT cached:**

- **Predecessor lists (`pred[]`).** These are O(VE) worst case in total. A dense
  graph with V=5000, E=500000 could have millions of predecessor entries. The
  storage cost is prohibitive and predecessor lists are only needed by
  betweenness back-propagation, not closeness. Betweenness reconstructs `pred[]`
  on-the-fly from the cached `dist[]` (see Section 6).

- **Stack order.** The BFS/Dijkstra settlement order can be reconstructed from
  `dist[]` by sorting nodes by non-decreasing distance. This costs O(V log V)
  per source but avoids storing V^2 integers.

### BLOB Encoding

All values are stored as raw `double` (IEEE 754 binary64) in platform-native
byte order. SQLite stores BLOBs verbatim, and the extension always runs on the
same platform that wrote the data. No byte-swapping is needed.

```c
/* Write dist[] to BLOB */
sqlite3_bind_blob(stmt, col, ws->dist,
                  ws->node_count * (int)sizeof(double), SQLITE_TRANSIENT);

/* Read dist[] from BLOB */
const double *cached_dist = (const double *)sqlite3_column_blob(stmt, col);
int n_doubles = sqlite3_column_bytes(stmt, col) / (int)sizeof(double);
```

---

## 5. Generation Counter Protocol

### Generation Storage

The adjacency VT's `_config` shadow table stores generation counters:

```sql
-- Written by Phase 1 rebuild:
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_adj', '42');

-- Written by Phase 2 SSSP rebuild:
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_sssp', '42');
```

The `generation_adj` counter increments on every CSR rebuild (full or
incremental). The `generation_sssp` counter records which `generation_adj`
the SSSP cache was computed against.

### Staleness Check

```
On SSSP cache read:
    G_adj  = SELECT value FROM {name}_config WHERE key = 'generation_adj'
    G_sssp = SELECT value FROM {name}_config WHERE key = 'generation_sssp'

    if G_sssp == G_adj:
        cache is FRESH → read from _sssp
    else:
        cache is STALE → recompute, write to _sssp, set G_sssp = G_adj
```

### Namespace Scoping

When Phase 1 namespace support is present, each namespace has independent
generation tracking:

```sql
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_sssp:' || namespace_id, generation_adj_value);
```

The staleness check includes the namespace qualifier:

```c
char *key = sqlite3_mprintf("generation_sssp:%lld", namespace_id);
/* ... query _config for this key ... */
```

### Rebuild Triggers

SSSP rebuild occurs in these scenarios:

1. **Explicit rebuild:** `INSERT INTO g(g) VALUES('rebuild')` triggers a full
   CSR rebuild (incrementing `generation_adj`), which implicitly invalidates
   the SSSP cache. The next TVF query detects staleness and recomputes.

2. **TVF query on stale cache:** When `bet_filter` or `clo_filter` detects
   `G_sssp < G_adj`, it recomputes all-pairs SSSP in-line and writes back
   to the `_sssp` shadow table before returning results.

3. **Proactive SSSP rebuild:** A new command `INSERT INTO g(g) VALUES('rebuild_sssp')`
   explicitly rebuilds the SSSP cache. Useful after a series of edge insertions
   when the user knows queries are coming.

---

## 6. Integration with Betweenness TVF

### Current Flow (bet_filter, lines 645-837)

```
1. Parse argv → GraphLoadConfig
2. Load graph → GraphData (from adjacency VT or raw SQL)
3. Allocate: CB[V], dist[V], sigma[V], delta[V], stack[V], pred[V]
4. For each source s in {0..V-1} (or sampled subset):
   a. sssp_bfs/dijkstra(g, s, dist, sigma, pred, stack, &stack_size, direction)
   b. Backward accumulation: walk stack[] in reverse, accumulate delta via pred[]
   c. CB[w] += delta[w]
5. Normalize, build results
```

### Modified Flow (with SSSP cache)

```
1. Parse argv → GraphLoadConfig
2. Load graph → GraphData (from adjacency VT or raw SQL)
3. Check if edge_table is a graph_adjacency VT with 'sssp' feature
4. IF sssp cache exists AND is fresh:
   a. Allocate: CB[V], delta[V], stack[V], pred[V]
   b. For each source s:
      i.   Read dist[V] and sigma[V] from _sssp shadow table
      ii.  Reconstruct stack[] by sorting reachable nodes by dist[] (ascending)
      iii. Compute pred[v] on-the-fly from dist[] and graph adjacency:
           For each edge (v → w): if dist[w] == dist[v] + weight(v,w)
                                   then v is a predecessor of w
      iv.  Backward accumulation (unchanged)
      v.   CB[w] += delta[w]
   c. Normalize, build results
5. ELSE (no cache or stale):
   a. Original flow (steps 3-5 from current implementation)
   b. IF graph_adjacency VT with 'sssp' feature AND V <= max_sssp_nodes:
      Write dist[] and sigma[] to _sssp shadow table after computation
      Update generation_sssp = generation_adj
```

### Predecessor Reconstruction

When reading from cache, predecessors must be reconstructed from `dist[]`
and the graph's adjacency lists. This avoids caching O(VE) predecessor data.

```c
/*
 * Reconstruct pred[] for source s from cached dist[] and graph adjacency.
 * For each node w, scan its incoming edges. If dist[v] + weight(v,w) == dist[w],
 * then v is a predecessor of w on a shortest path from s.
 *
 * Cost: O(E) per source — same as the original SSSP traversal.
 */
static void reconstruct_predecessors(
    const GraphData *g, const double *dist,
    SsspPredList *pred, int N, const char *direction
) {
    int use_out = !direction || strcmp(direction, "reverse") != 0;
    int use_in  = direction && (strcmp(direction, "reverse") == 0 ||
                                strcmp(direction, "both") == 0);

    for (int v = 0; v < N; v++) {
        if (dist[v] < 0) continue;  /* unreachable */

        for (int pass = 0; pass < 2; pass++) {
            const GraphAdjList *adj = (pass == 0)
                ? (use_out ? &g->out[v] : NULL)
                : (use_in  ? &g->in[v]  : NULL);
            if (!adj) continue;

            for (int e = 0; e < adj->count; e++) {
                int w = adj->edges[e].target;
                if (dist[w] < 0) continue;
                double edge_dist = g->has_weights ? adj->edges[e].weight : 1.0;
                if (sssp_double_eq(dist[w], dist[v] + edge_dist)) {
                    /* Deduplicate: check if v is already last pred of w */
                    if (pred[w].count == 0 ||
                        pred[w].items[pred[w].count - 1] != v) {
                        sssp_pred_push(&pred[w], v);
                    }
                }
            }
        }
    }
}
```

**Note:** Reconstructing predecessors from `dist[]` costs O(E) per source,
which is the same asymptotic cost as the original BFS/Dijkstra. The savings
come from avoiding the priority queue / BFS queue overhead and from reading
`sigma[]` directly from cache (no accumulation needed).

For betweenness with the SSSP cache, the per-source cost drops from
O(V + E) (full SSSP) to O(E) (predecessor scan + back-propagation). The
V term is eliminated because there is no queue management.

### Stack Reconstruction

The Brandes back-propagation requires nodes in reverse BFS/Dijkstra order
(non-decreasing distance from source). This can be reconstructed from `dist[]`:

```c
/*
 * Reconstruct the BFS order stack from cached dist[].
 * Nodes are sorted by dist[] in non-decreasing order.
 * Unreachable nodes (dist < 0) are excluded.
 *
 * Uses a simple insertion sort since V is bounded by max_sssp_nodes (10K).
 * For V=10K, O(V^2) insertion sort takes ~0.1ms — negligible vs I/O.
 */
static int reconstruct_stack(const double *dist, int N, int *stack) {
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (dist[i] >= 0)
            stack[count++] = i;
    }
    /* Sort by non-decreasing distance */
    for (int i = 1; i < count; i++) {
        int key = stack[i];
        double key_dist = dist[key];
        int j = i - 1;
        while (j >= 0 && dist[stack[j]] > key_dist) {
            stack[j + 1] = stack[j];
            j--;
        }
        stack[j + 1] = key;
    }
    return count;
}
```

---

## 7. Integration with Closeness TVF

### Current Flow (clo_filter, lines 963-1101)

```
1. Parse argv → GraphLoadConfig
2. Load graph → GraphData
3. Allocate: dist[V], sigma[V], stack[V], pred[V], closeness[V]
4. For each source s in {0..V-1}:
   a. sssp_bfs/dijkstra(g, s, dist, sigma, pred, stack, &stack_size, direction)
   b. Sum dist[i] for reachable nodes, count reachable
   c. closeness[s] = reachable / sum_dist  (with Wasserman-Faust normalization)
5. Build results
```

**Current waste:** `sigma[V]` (8V bytes) and `pred[V]` (allocated as V
`SsspPredList` structs, each with a dynamic array) are allocated and populated
but never read. For V=10000, this wastes ~80KB for sigma and ~160KB+ for
pred structures.

### Modified Flow (with SSSP cache)

```
1. Parse argv → GraphLoadConfig
2. Load graph → GraphData
3. Check if edge_table is a graph_adjacency VT with 'sssp' feature
4. IF sssp cache exists AND is fresh:
   a. Allocate: closeness[V]
   b. For each source s:
      i.   Read dist[V] from _sssp shadow table (skip sigma — not needed)
      ii.  Sum dist[i] for reachable nodes, count reachable
      iii. closeness[s] = reachable / sum_dist
   c. Build results
   NOTE: Zero SSSP computation. Cost is O(V^2) reads from shadow table.
5. ELSE IF no cache, but shared SSSP module available:
   a. Allocate: closeness[V]
   b. Init SsspWorkingSet with needs_brandes=0  ← KEY CHANGE
      (only allocates dist[], skips sigma/pred/stack)
   c. For each source s:
      i.   sssp_run(g, s, &ws, direction)
      ii.  Sum dist[i], count reachable, compute closeness[s]
   d. IF graph_adjacency VT with 'sssp' feature AND V <= max_sssp_nodes:
      Write dist[] rows to _sssp (sigma=NULL for each row)
      Update generation_sssp
   e. Build results
```

### Memory Savings (No Cache Path)

Even without the shadow table, the shared module improves closeness:

| Allocation | Before (graph_centrality.c) | After (graph_sssp.h) |
|------------|---------------------------|---------------------|
| `dist[V]` | Yes | Yes |
| `sigma[V]` | Yes (wasted) | No (`needs_brandes=0`) |
| `pred[V]` (V IntList structs) | Yes (wasted) | No (`needs_brandes=0`) |
| `stack[V]` | Yes (wasted) | No (`needs_brandes=0`) |
| **Total per-source overhead** | **8V + 8V + 24V + 4V = 44V bytes** | **8V bytes** |

For V=10000: **440KB wasted per query** drops to 0KB.

---

## 8. Size Limits and Feasibility

### Storage Requirements Table

The `_sssp` shadow table stores two BLOBs per source node: `distances`
(always) and `sigma` (only when betweenness has been computed).

| V | dist matrix | sigma matrix | Total (both) | Total (dist only) | Feasible? |
|---:|------------:|-------------:|-------------:|------------------:|-----------|
| 100 | 80 KB | 80 KB | 160 KB | 80 KB | Trivial |
| 500 | 2 MB | 2 MB | 4 MB | 2 MB | Trivial |
| 1,000 | 8 MB | 8 MB | 16 MB | 8 MB | Yes |
| 2,000 | 32 MB | 32 MB | 64 MB | 32 MB | Yes |
| 5,000 | 200 MB | 200 MB | 400 MB | 200 MB | Marginal |
| 10,000 | 800 MB | 800 MB | 1.6 GB | 800 MB | Limit |
| 20,000 | 3.2 GB | 3.2 GB | 6.4 GB | 3.2 GB | Infeasible |
| 50,000 | 20 GB | 20 GB | 40 GB | 20 GB | Infeasible |

**Formula:** `V * V * sizeof(double) = V^2 * 8 bytes` per matrix.

### Configurable Threshold

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst',
    features='sssp',
    max_sssp_nodes=5000   -- override default 10000
);
```

The `max_sssp_nodes` parameter (default: 10000) controls the upper bound.

**Behavior when V > max_sssp_nodes:**

- The `_sssp` shadow table is created (schema is fixed at VT creation time)
- The `_sssp` shadow table remains empty
- The shared SSSP module is used for live computation (Option B behavior)
- No error is raised — the feature silently degrades to uncached mode
- The `_config` table stores `sssp_skipped_reason = 'node_count_exceeds_threshold'`
  so the user can diagnose why caching is not active

**Behavior when V <= max_sssp_nodes:**

- SSSP is cached on first betweenness or closeness query
- Subsequent queries read from cache if generation is fresh
- `INSERT INTO g(g) VALUES('rebuild_sssp')` forces explicit recomputation

### Disk vs Memory

The shadow table is on-disk (SQLite B-tree). SSSP results are read row-by-row
during TVF execution, not loaded into memory all at once. Peak memory during
betweenness with cache is:

```
Per-source: dist[V] + sigma[V] + pred[V] + stack[V] + delta[V]
          = 8V + 8V + ~24V + 4V + 8V = ~52V bytes
```

This is the same as the current implementation. The shadow table does not
increase peak memory — it only adds disk I/O (which SQLite page cache
amortizes).

---

## 9. Implementation Steps

### Step 1: Create graph_sssp.h

Create `src/graph_sssp.h` with the public API defined in Section 3.

**File:** `src/graph_sssp.h`

### Step 2: Create graph_sssp.c

Create `src/graph_sssp.c` containing the implementations moved from
`graph_centrality.c`:

- `SsspPQ`: `sssp_pq_init`, `sssp_pq_destroy`, `sssp_pq_push`, `sssp_pq_pop`
  (adapted from `dpq_init`, `dpq_destroy`, `dpq_push`, `dpq_pop`)
- `SsspPredList`: `sssp_pred_init`, `sssp_pred_push`, `sssp_pred_clear`,
  `sssp_pred_destroy` (adapted from `intlist_init`, `intlist_push`,
  `intlist_clear`, `intlist_destroy`)
- `sssp_double_eq` (adapted from `double_eq`)
- `sssp_working_set_init`, `sssp_working_set_destroy`
- `sssp_bfs` (adapted from static `sssp_bfs`, modified to use `SsspWorkingSet`
  and NULL-check `sigma`/`pred`/`stack`)
- `sssp_dijkstra` (adapted, same modifications)
- `sssp_run` (new dispatcher)

**File:** `src/graph_sssp.c`

The file must begin with:

```c
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_sssp.h"
#include "graph_load.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
```

### Step 3: Update graph_centrality.c

Remove all static SSSP types and functions from `graph_centrality.c`:

- Delete lines 96-335 (DoublePQ, IntList, double_eq, sssp_bfs, sssp_dijkstra)
- Add `#include "graph_sssp.h"` at the top
- Update `bet_filter`:
  - Replace `IntList *pred` with `SsspPredList *pred`
  - Replace `intlist_*` calls with `sssp_pred_*` calls
  - Replace `double *sigma` + `int *stack` with `SsspWorkingSet ws`
  - Replace `sssp_bfs`/`sssp_dijkstra` calls with `sssp_run(&g, s, &ws, direction)`
- Update `clo_filter`:
  - Use `SsspWorkingSet ws` with `needs_brandes=0`
  - Remove allocation of `sigma`, `pred`, `stack` (no longer needed)
  - Replace SSSP calls with `sssp_run(&g, s, &ws, direction)`

**File:** `src/graph_centrality.c`

### Step 4: Update build configuration

Add `graph_sssp.c` to the source file list in `scripts/generate_build.py`
(which drives the Makefile's `SRC` and `TEST_LINK_SRC` variables).

Add `graph_sssp.h` to `HEADERS`.

**File:** `scripts/generate_build.py`

### Step 5: Add _sssp shadow table creation

In `graph_adjacency.c`, extend `adjacency_create_shadow_tables()` to
conditionally create the `_sssp` table when the `features` parameter
includes `'sssp'`. This depends on Phase 1 adding `features` parsing.

```c
/* In adjacency_create_shadow_tables(), after existing shadow tables: */
if (params->features & FEATURE_SSSP) {
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_sssp\" ("
        "  namespace_id INTEGER NOT NULL,"
        "  source_idx   INTEGER NOT NULL,"
        "  distances    BLOB NOT NULL,"
        "  sigma        BLOB,"
        "  PRIMARY KEY (namespace_id, source_idx)"
        ")", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;
}
```

**File:** `src/graph_adjacency.c`

### Step 6: Add SSSP cache population

Create a new static function in `graph_adjacency.c` that populates the
`_sssp` shadow table for a given namespace:

```c
/*
 * Populate _sssp shadow table for all sources in the given namespace.
 * Called by rebuild_sssp command or lazily on first TVF cache miss.
 *
 * Writes V rows, each with dist[V] and optionally sigma[V] as BLOBs.
 */
static int sssp_cache_populate(
    sqlite3 *db, const char *vtab_name,
    int64_t namespace_id, const GraphData *g,
    const char *direction, int include_sigma
);
```

**File:** `src/graph_adjacency.c`

### Step 7: Add SSSP cache read in bet_filter and clo_filter

Add helper functions that check for and read from the SSSP cache:

```c
/*
 * Check if SSSP cache is available and fresh for the given adjacency VT.
 * Returns 1 if cache hit, 0 if miss. On hit, *gen_sssp is set.
 */
static int sssp_cache_check(
    sqlite3 *db, const char *vtab_name,
    int64_t namespace_id, int64_t *gen_adj, int64_t *gen_sssp
);

/*
 * Read one row of cached SSSP data for source_idx.
 * Copies dist[] into caller-provided buffer. Optionally reads sigma[].
 * Returns SQLITE_OK on success, SQLITE_DONE if row not found.
 */
static int sssp_cache_read_row(
    sqlite3 *db, const char *vtab_name,
    int64_t namespace_id, int source_idx,
    double *dist_out, int dist_size,
    double *sigma_out, int sigma_size  /* sigma_out may be NULL */
);
```

These helpers are called from `bet_filter` and `clo_filter` in
`graph_centrality.c`, gated by the `is_graph_adjacency()` check that
already exists.

**File:** `src/graph_centrality.c`

### Step 8: Add generation check logic

Add generation read/write functions that query `_config`:

```c
/* Read a generation counter from _config. Returns -1 if not found. */
static int64_t read_generation(
    sqlite3 *db, const char *vtab_name, const char *key
);

/* Write a generation counter to _config. */
static int write_generation(
    sqlite3 *db, const char *vtab_name, const char *key, int64_t value
);
```

The `rebuild` command in `graph_adjacency.c` already increments
`generation_adj`. Step 8 adds the corresponding reads in the TVF code
and writes in the SSSP population code.

**File:** `src/graph_adjacency.c` (write), `src/graph_centrality.c` (read)

### Step 9: Add namespace_id scoping

All SSSP operations (check, populate, read) include `namespace_id` in
their WHERE clauses. When Phase 1 is not yet implemented (namespace_id is
always 0), this is a no-op filter on the primary key.

```sql
-- Read cache for namespace 0, source 42:
SELECT distances, sigma FROM "{name}_sssp"
    WHERE namespace_id = 0 AND source_idx = 42;

-- Invalidate cache for namespace 3:
DELETE FROM "{name}_sssp" WHERE namespace_id = 3;
```

**Files:** `src/graph_adjacency.c`, `src/graph_centrality.c`

---

## 10. Verification Steps

### C Unit Tests

1. **sssp_bfs correctness after extraction.** Build a small graph (5 nodes,
   directed), run `sssp_bfs` via the new module, assert `dist[]`, `sigma[]`,
   and `pred[]` match expected values. Test with `needs_brandes=1` and
   `needs_brandes=0` (verifying NULL sigma/pred don't crash).

2. **sssp_dijkstra correctness after extraction.** Same graph with weights,
   verify Dijkstra produces correct distances and path counts.

3. **SsspPQ ordering.** Push entries in random order, verify `sssp_pq_pop`
   returns them in non-decreasing distance order.

4. **SsspWorkingSet lifecycle.** Init with `needs_brandes=1`, verify all
   fields non-NULL. Init with `needs_brandes=0`, verify `sigma`, `pred`,
   `stack` are NULL and `dist` is non-NULL. Destroy, verify no leaks (ASan).

5. **sssp_double_eq edge cases.** Test tie detection at various magnitudes
   and near-zero values.

**File:** `test/test_graph_sssp.c`

### Python Integration Tests

6. **Betweenness with SSSP cache matches without cache.** Create a graph
   adjacency VT with `features='sssp'`. Run betweenness. Then create the same
   graph as a plain edge table. Run betweenness. Assert identical results
   (within floating-point tolerance).

7. **Closeness with SSSP cache matches without cache.** Same comparison
   as test 6 but for closeness.

8. **Performance: second call is faster.** Run betweenness twice on the same
   adjacency VT. Measure wall time. Assert the second call is measurably
   faster (cache hit avoids SSSP computation). Use a graph large enough
   that SSSP is non-trivial (V >= 500).

9. **Generation staleness.** Create adjacency VT with `features='sssp'`.
   Run betweenness (populates cache). Insert a new edge. Run `rebuild`.
   Run betweenness again. Verify the cache was invalidated (generation
   mismatch) and results reflect the new edge.

10. **Namespace scoping.** Create adjacency VT with `features='sssp'` and
    `namespace_cols='project_id'`. Populate two namespaces. Run betweenness
    on namespace A. Modify namespace B's edges. Verify namespace A's SSSP
    cache is still fresh (independent generations).

11. **V > max_sssp_nodes.** Create an adjacency VT with `max_sssp_nodes=10`
    and a graph with 50 nodes. Run betweenness. Verify `_sssp` table is
    empty (caching skipped). Verify results are still correct (live computation).

12. **Closeness does not allocate sigma.** This is a behavioral test: run
    closeness on a plain edge table (no VT, no cache). Verify correct results.
    The memory savings are verified by ASan (no leaks of unused sigma/pred
    arrays).

**File:** `pytests/test_sssp_cache.py`

---

## 11. References

1. **Brandes, U. (2001).** A faster algorithm for betweenness centrality.
   *Journal of Mathematical Sociology*, 25(2), 163-177.
   DOI: [10.1080/0022250X.2001.9990249](https://doi.org/10.1080/0022250X.2001.9990249)
   -- The foundational algorithm for betweenness centrality. Defines the
   SSSP working set (dist, sigma, pred, stack) and backward accumulation.

2. **Geisberger, R., Sanders, P., & Schultes, D. (2008).** Better approximation
   of betweenness centrality. *Proceedings of ALENEX 2008*, 90-100.
   DOI: [10.1137/1.9781611972887.9](https://doi.org/10.1137/1.9781611972887.9)
   -- Improved sampling strategies for approximate betweenness. Relevant
   to the `auto_approx_threshold` parameter already in `bet_filter`.

3. **Riondato, M. & Kornaropoulos, E.M. (2016).** Fast approximation of
   betweenness centrality through sampling. *Data Mining and Knowledge
   Discovery*, 30, 438-475.
   DOI: [10.1007/s10618-015-0423-0](https://doi.org/10.1007/s10618-015-0423-0)
   -- Sample-based approximation with probabilistic guarantees. Shows
   that O(V / epsilon^2) samples suffice for (1+epsilon) approximation.

4. **Davis, T.A. (2019).** Algorithm 1000: SuiteSparse:GraphBLAS: Graph
   algorithms in the language of sparse linear algebra. *ACM TOMS*, 45(4).
   DOI: [10.1145/3322125](https://doi.org/10.1145/3322125)
   -- Generation counter pattern for lazy recomputation of derived
   structures. Our `generation_adj` / `generation_sssp` protocol
   follows this design.

5. **Wasserman, S. & Faust, K. (1994).** *Social Network Analysis: Methods
   and Applications.* Cambridge University Press.
   -- Defines the normalized closeness centrality formula used in
   `clo_filter`: C(v) = (reachable / sum_dist) * (reachable / (N-1)).

6. **Eppstein, D. & Wang, J. (2004).** Fast approximation of centrality.
   *Journal of Graph Algorithms and Applications*, 8(1), 39-45.
   DOI: [10.7155/jgaa.00081](https://doi.org/10.7155/jgaa.00081)
   -- Demonstrates that O(V / epsilon^2 * log V) random SSSP runs
   approximate closeness centrality. Relevant to future approximate
   closeness support.

---

**Prev:** [Phase 1 — Scoped Adjacency VT](./01_scoped_adjacency_vt.md) | **Next:** [Phase 3 — Components Shadow Table](./03_components_shadow_tables.md)
