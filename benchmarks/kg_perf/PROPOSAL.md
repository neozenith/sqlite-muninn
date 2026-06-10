# Muninn-side proposal: filtered-KG query speedups

This doc ranks the data-structure additions to `sqlite-muninn` that the `kg_perf`
hill-climb harness has empirically justified, with the measured upper-bound speedup
each one would unlock.

## Setup

- Workload: `viz/frontend/public/demos/sessions_demo.db` (122 951 events, 4 590
  chunks, 43 267 entities, 10 849 canonical nodes, 3 054 edges, 8 projects)
- Query shape: events → (project + time) filter → chunks → entities → entity_clusters
  → top-K seeds by centrality → BFS expand → min-degree prune
- Filters: 4 widths spanning full corpus → narrow (project + 7 days)
- Queries: 4 shapes covering degree, node-betweenness, edge-betweenness with
  varying top-K, depth, min-degree
- Repetitions: 5 timed runs per (strategy × workload), 1 warmup, p50 wall-ms

## Headline numbers (p50 ms)

| Filter / Query | baseline | sql_subset | chunk_canonical | topk_cache (warm) |
|---|---:|---:|---:|---:|
| 7d × node_bw | 82 | 52 (1.6×) | 5 (17×) | 0.01 (6700×) |
| 7d × edge_bw | 93 | 53 (1.8×) | 5 (20×) | 0.01 (6500×) |
| 30d × node_bw | 94 | 66 (1.4×) | 10 (9×) | 0.02 (5200×) |
| 30d × edge_bw | 106 | 68 (1.6×) | 10 (10×) | 0.02 (5500×) |
| project × node_bw | 90 | 70 (1.3×) | 29 (3×) | 0.02 (3900×) |
| full × node_bw | 103 | 95 (1.1×) | 32 (3×) | 0.03 (4100×) |

## Findings

### 1. The filter chain — not Brandes — is the bottleneck

The original `kg/payload.py` implementation runs `graph_node_betweenness` on the
full edge table, then post-filters the results in Python by the `allowed_canonicals`
set built from a 4-way join (events × chunks × entities × clusters). At a 3K-edge
graph, Brandes runs in single-digit ms. The 4-way join is what consumes ~50ms.

`sql_subset` (induced Brandes via SQL temp table) gave only 1.0–1.8×. The big jump
came from `chunk_canonical` (denormalized provenance index) which delivered 2.3–20×.
The earlier hypothesis — that a `node_filter_table=` argument on `graph_node_betweenness`
would be the leverage point — was wrong at this scale.

### 2. Induced-subgraph centrality is also semantically more correct

Across all narrow-filter rows, `seed_jaccard` between baseline and sql_subset/chunk_canonical
is 0.20–0.50, i.e. the strategies disagree on 50–80% of their top-K. The induced
answer is what the dashboard "top-K most central in this filter window" actually
wants — running Brandes on the full graph and post-filtering produces "global central
nodes that happen to be in this window," which is a different question. **The
optimization is also a correctness fix, not a pure perf change.**

### 3. K-core is the wrong drop-in for `min_degree` pruning

K-core's iterative peeling cascades — on sparse session-KG subgraphs it routinely
wipes out the entire result on `min_degree=3` queries (0 nodes returned vs 24–145
from single-pass prune). K-core is a useful primitive, but as a *separate* feature
("show me the densely-connected core"), not a swap-in for the existing prune.
Iter 3 was a deliberate negative result.

### 4. Result-level caching dominates everything once warm

`topk_cache` keyed on `(filter, query, edge_generation)` reaches 0.01–0.10ms on
warm reads — 500×–6700× over baseline. The dashboard query pattern (user clicks
the same project/window repeatedly) is the natural fit. The signature must include
an edge-generation counter so cached entries auto-invalidate on edge mutation; this
is exactly what GII's `_config.generation` is designed for.

## Recommended muninn changes, ranked

### A. `chunk_canonical` as a maintained shadow table (highest leverage)

Add to `gii.c` (or a new sibling `provenance.c`) a shadow table:

```sql
CREATE TABLE _gii_provenance (
    namespace_id INTEGER NOT NULL,
    chunk_id     INTEGER NOT NULL,
    canonical    TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    PRIMARY KEY (namespace_id, chunk_id, canonical)
);
CREATE INDEX _gii_provenance_proj_ts ON _gii_provenance(namespace_id, project_id, timestamp);
CREATE INDEX _gii_provenance_canonical ON _gii_provenance(namespace_id, canonical);
```

Maintained by triggers on `event_message_chunks`, `entities`, `entity_clusters`.
Resolves the 4-way join into a single indexed scan.

**Measured upper bound:** 6×–20× on filtered queries (5–32ms saved per query).
**Implementation cost:** moderate — three trigger sets and a shadow table; the
denormalization rule is straightforward.

### B. Top-K result cache integrated with GII generation counter

A new TVF `graph_topk_centrality` that:
1. Reads namespace + filter + query parameters as hidden columns
2. Computes `signature = hash(filter ‖ query ‖ G_adj)`
3. Looks up `_gii_topk_cache(signature, ...)` — O(1) on hit
4. On miss: runs the centrality TVF, BFS expand, prune; stores the result

Cache is invalidated automatically when `G_adj` increments (edge mutation), via
the same generation-DAG protocol the existing GII plan describes.

**Measured upper bound:** 500×–6700× on warm reads; ~equal to chunk_canonical on
cold reads. **Implementation cost:** moderate — the storage is a single shadow
table; the new TVF is mostly orchestration.

### C. Skip filter-aware Brandes (DEFER, not justified at current scale)

The original "`node_filter_table=` argument to `graph_node_betweenness`" proposal
was not justified at the 3K-edge scale of `sessions_demo.db`. The sql_subset
strategy (which is the SQL-only equivalent of that change) only gave 1.0–1.8×
speedup. Reconsider this when corpora grow past ~50K edges and Brandes itself
becomes the bottleneck.

### D. Skip k-core integration entirely

K-core is the wrong abstraction for the `min_degree` query parameter. If a future
visualization mode ("show me the densely-connected core") wants k-core, expose it
as its own TVF, not a replacement for the existing prune.

## Iteration sequencing for muninn implementation

1. Land **(A)** first — biggest single win, simplest data structure, exercises
   the existing GII trigger / shadow-table machinery.
2. Land **(B)** on top — once **(A)** is in, the cache table piggybacks on the
   same generation-counter wiring.
3. Re-benchmark on a larger corpus before deciding **(C)**. The harness already
   supports adding new workloads under `kg_perf/workload.py`.
4. **(D)** is a "do not implement" recommendation.

## How to reproduce

```bash
uv run --no-sync -m benchmarks.kg_perf manifest             # list permutations
uv run --no-sync -m benchmarks.kg_perf run-all --strategy baseline
uv run --no-sync -m benchmarks.kg_perf run-all --strategy chunk_canonical
uv run --no-sync -m benchmarks.kg_perf run-all --strategy topk_cache
uv run --no-sync -m benchmarks.kg_perf compare              # speedup + Jaccard table
```

Raw measurements live in `benchmarks/kg_perf/results/*.jsonl`; they accumulate
across runs, and `compare` always picks the most-recent record per (strategy,
workload).
