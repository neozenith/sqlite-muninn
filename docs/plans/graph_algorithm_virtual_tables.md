# Graph Algorithm Virtual Tables — Cascading Cached Computation

**Status:** Draft Plan (depends on `graph_virtual_tables.md` Phases 1-5)
**Date:** 2026-02-18
**Prerequisite:** `graph_adjacency` virtual table with blocked CSR +
incremental rebuild, quantitatively benchmarked.

## Motivation

Once `graph_adjacency` provides a persistent, incrementally-maintained CSR
index over an edge table, the next leverage point is caching expensive
algorithm results. Today, running `graph_betweenness` on a 5,000-node graph
takes O(VE) = 250M edge relaxations — multiple seconds. If the graph hasn't
changed, that's wasted compute.

The cascading pattern: each algorithm virtual table reads from
`graph_adjacency`'s CSR shadow tables, runs its computation, and caches
results in its own shadow tables. A generation counter detects staleness.

### Target Architecture

```
edge_table
    │ (triggers → delta log)
    ▼
graph_adjacency VT  ← CSR + node registry + degree (generation counter G)
    │
    ├──→ graph_sssp VT           ← O(V²) all-pairs distances + sigma
    │        │
    │   ┌────┴────┐
    │   ▼         ▼
    │  graph_betweenness VT    graph_closeness VT
    │
    ├──→ graph_leiden VT         ← community partition + modularity
    │        ▲
    │        │ (optional seed)
    │   graph_components VT      ← component IDs
    │
    ├──→ graph_pagerank VT       ← rank scores
    │
    ├──→ graph_degree            ← trivial: reads _degree from adjacency
    │
    └──→ graph_select TVF        ← no cache (evaluates selectors on live CSR)
```

## Staleness Detection: Generation Counter Protocol

Each `graph_adjacency` VT maintains a monotonically increasing `generation`
counter in its `_config` shadow table. The counter is incremented on every
rebuild (full or incremental).

Each downstream algorithm VT stores `last_adjacency_gen` in its own `_config`.
On query:

```c
int adjacency_gen = read_config(adjacency_vtab, "generation");
int my_gen = read_config(my_vtab, "last_adjacency_gen");
if (my_gen < adjacency_gen) {
    /* Stale: recompute from adjacency CSR, cache results */
    recompute_and_cache();
    write_config(my_vtab, "last_adjacency_gen", adjacency_gen);
} else {
    /* Fresh: serve cached results directly */
}
```

### Multi-Level Cascading

For chains like `adjacency → sssp → betweenness`, each level checks
the generation of its immediate parent. If SSSP is stale, it rebuilds
from adjacency. If betweenness is stale but SSSP is fresh, betweenness
reads from SSSP's cache.

## Algorithm Dependency Map

Based on code analysis of `graph_centrality.c`, `graph_community.c`,
`graph_tvf.c`, and `node2vec.c`:

### What Each Algorithm Actually Needs

| Algorithm | From Adjacency | From Degree | From SSSP | From Components |
|-----------|---------------|-------------|-----------|-----------------|
| **Betweenness** | Forward + reverse CSR | — | Produces SSSP (sigma, pred, dist) | — |
| **Closeness** | Forward + reverse CSR | — | Only dist[] from SSSP | — |
| **Leiden** | Forward + reverse CSR | weighted_degree (hot path) | — | Optional: seed partition |
| **PageRank** | Forward CSR only | out_degree (dangling nodes) | — | — |
| **Degree** | — | Direct read from `_degree` | — | — |
| **graph_select** | Forward + reverse CSR | — | — | — |
| **Node2Vec** | Forward CSR (undirected) | — | — | — |

### Key Finding: Leiden Does NOT Depend on Betweenness

Leiden is purely modularity-based. Its inputs are:
- `k_i` — node weighted degree (from `_degree` shadow table)
- `sum_in` — intra-community edge weight sum (computed from adjacency)
- `sum_tot` — community total degree (from adjacency + `_degree`)
- `2m` — total edge weight (from `_config`)

The modularity gain formula:
`ΔQ = [sum_in + k_i→C] / 2m - γ[(sum_tot + k_i) / 2m]²`

No betweenness, SSSP, or path-counting information is used.

### Shared SSSP Opportunity

Betweenness runs all-pairs SSSP: O(VE) exact. Closeness also runs
all-pairs SSSP: O(V²) unweighted. Both are the most expensive operations
in the extension. Sharing a single SSSP pass eliminates redundant work.

Betweenness needs from each SSSP: `dist[]`, `sigma[]`, `pred[]`
Closeness needs from each SSSP: `dist[]` only

A shared `graph_sssp` VT could persist `(src_idx, dst_idx, distance, sigma)`
in a shadow table. This is O(V²) rows — feasible for V < 10K (~100M rows
at V=10K, likely too large beyond that).

## Design: Individual Algorithm Virtual Tables

### `graph_betweenness`

```sql
CREATE VIRTUAL TABLE my_betweenness USING graph_betweenness(
    adjacency='my_graph',
    direction='both',
    approximate=0,               -- 0 = exact, 1 = sampled (sqrt(V) sources)
    normalized=1                 -- normalize by (V-1)(V-2)
);

SELECT node, centrality FROM my_betweenness;
SELECT node, centrality FROM my_betweenness WHERE node = 'X';

INSERT INTO my_betweenness(my_betweenness) VALUES('rebuild');
```

**Shadow tables:**
- `{name}_results` — `(node_idx INTEGER PK, centrality REAL)`
- `{name}_config` — adjacency name, direction, approximate, normalized,
  last_adjacency_gen

**Complexity:** O(VE) exact, O(E√V) approximate.

### `graph_closeness`

Same pattern as betweenness. If a `graph_sssp` VT exists on the same
adjacency, closeness reads from it instead of running its own SSSP.

**Shadow tables:**
- `{name}_results` — `(node_idx INTEGER PK, centrality REAL)`
- `{name}_config` — adjacency name, direction, last_adjacency_gen

**Complexity:** O(V²) unweighted, O(V(E + V log V)) weighted.

### `graph_sssp` (Optional Shared Cache)

```sql
CREATE VIRTUAL TABLE my_sssp USING graph_sssp(adjacency='my_graph');
```

**Shadow tables:**
- `{name}_distances` — `(src_idx INT, dst_idx INT, distance REAL, sigma REAL, PRIMARY KEY(src_idx, dst_idx))`
- `{name}_config` — adjacency name, last_adjacency_gen

**Complexity:** O(V × E) to build. O(V²) storage.

### `graph_leiden`

```sql
CREATE VIRTUAL TABLE my_communities USING graph_leiden(
    adjacency='my_graph',
    direction='both',
    resolution=1.0,
    max_iterations=100
);

SELECT node, community_id, modularity FROM my_communities;

INSERT INTO my_communities(my_communities) VALUES('rebuild');
```

**Shadow tables:**
- `{name}_results` — `(node_idx INTEGER PK, community_id INTEGER)`
- `{name}_config` — modularity score, last_adjacency_gen, iterations_run,
  resolution

**Component seeding:** If a `graph_components` VT exists on the same
adjacency, Leiden seeds its initial partition with component IDs instead
of singleton communities. This reduces Phase 1 iterations on disconnected
graphs because nodes in separate components can never be merged.

**Partition seeding on rebuild:** When `'rebuild'` is called and cached
results exist, the previous partition is used as the starting point
(following the Dynamic Leiden pattern). This gives 1.1-1.4× speedup over
cold-start recomputation.

### `graph_components`

```sql
CREATE VIRTUAL TABLE my_components USING graph_components(
    adjacency='my_graph'
);

SELECT node, component_id, component_size FROM my_components;
```

**Shadow tables:**
- `{name}_results` — `(node_idx INTEGER PK, component_id INTEGER, component_size INTEGER)`
- `{name}_config` — last_adjacency_gen

**Complexity:** O(V + E) via Union-Find. Very cheap — strong candidate
for early implementation.

### `graph_pagerank`

```sql
CREATE VIRTUAL TABLE my_pagerank USING graph_pagerank(
    adjacency='my_graph',
    damping=0.85,
    iterations=100,
    tolerance=1e-6
);

SELECT node, rank FROM my_pagerank;
```

**Shadow tables:**
- `{name}_results` — `(node_idx INTEGER PK, rank REAL)`
- `{name}_config` — damping, iterations, tolerance, last_adjacency_gen

**Complexity:** O(E × iterations). Typically 20-100 iterations.

### `graph_degree` (No Separate VT Needed)

Degree centrality is already computed and stored in `graph_adjacency`'s
`_degree` shadow table. No separate virtual table needed:

```sql
-- Degree centrality is a direct read from the adjacency VT
SELECT node, in_degree, out_degree, weighted_in_degree, weighted_out_degree
FROM my_graph;
```

If API consistency requires a separate `graph_degree` TVF, its `xFilter`
simply queries `{adjacency}_degree JOIN {adjacency}_nodes`.

### `graph_select` (No Cache)

`graph_select` evaluates selector expressions against the live CSR. It is
a TVF, not a cached VT — each invocation evaluates the expression fresh.
See `docs/plans/graph_search_syntax.md`.

```sql
SELECT node FROM graph_select('my_graph', '+critical_node');
SELECT node FROM graph_select('my_graph', 'not @main_pipeline');
```

## Key Refactors Required

### Extract SSSP into Shared Module

Currently `sssp_bfs` and `sssp_dijkstra` are static functions in
`graph_centrality.c` that operate on `GraphData` adjacency lists. To
support both the existing TVFs and the new VTs (which read from CSR):

1. Extract into `src/graph_sssp.c` / `src/graph_sssp.h`
2. Provide two backends: one for `GraphData` (backward compat), one for
   CSR arrays (new VTs)
3. Shared interface: `sssp_run(backend, source, dist, sigma, pred, stack)`

### Closeness: Remove Unused Allocations

`graph_centrality.c` closeness implementation allocates `sigma[]` and
`pred[]` arrays that it never reads. These are passed to `sssp_bfs` /
`sssp_dijkstra` because of the shared function signature, but closeness
only uses `dist[]`. A "no predecessors" mode in the SSSP functions would
eliminate O(V + E) wasted memory per source.

### Node2Vec: Use GraphData / CSR

`node2vec.c` has its own `Graph` struct with O(N) linear-scan node lookup.
It should use `graph_load.c`'s `GraphData` (or better, read from
`graph_adjacency`'s CSR). This eliminates O(N²) graph loading for Node2Vec.

## Cascading Blocked Incremental Rebuild

The most ambitious goal: when `graph_adjacency` does a blocked incremental
rebuild (only rewriting affected blocks), downstream algorithm VTs could
limit their recomputation to the affected region of the graph.

This is a hard problem — betweenness centrality is fundamentally global
(a single edge change can affect betweenness scores everywhere). But some
algorithms have locality properties:

| Algorithm | Locality | Incremental Feasibility |
|-----------|----------|------------------------|
| **Degree** | Perfect — only affected nodes change | Trivial: update `_degree` rows for changed nodes |
| **Components** | High — Union-Find can handle incremental edge additions | Medium: edge deletion requires re-scan of affected component |
| **PageRank** | Medium — power method converges faster from previous state | Warm-start: use cached ranks as initial vector |
| **Leiden** | Medium — Dynamic Leiden only re-processes affected vertices | Partition seeding + local re-processing |
| **Closeness** | Low — single edge change can shift all distances | Must recompute all-pairs SSSP |
| **Betweenness** | None — globally coupled | Must recompute fully |

**Recommendation:** Start with warm-start approaches (PageRank from cached
ranks, Leiden from cached partition) before attempting true incremental
algorithms. The benchmark suite from `graph_virtual_tables.md` Phase 5 will
quantify whether warm-start is sufficient or if true incremental algorithms
are needed.

## Implementation Phases

### Phase A: Components + Degree VTs (lowest complexity, immediate value)

- `graph_components` VT: O(V+E) Union-Find, trivial caching
- Degree: already available from `graph_adjacency._degree`
- These establish the cascading VT pattern with minimal risk

### Phase B: PageRank VT (medium complexity)

- Warm-start from cached ranks
- Forward CSR only (simplest algorithm to port to CSR reader)

### Phase C: Leiden VT (medium complexity)

- Partition seeding from cached results
- Component seeding from `graph_components` VT
- Requires adjacency + degree (both from `graph_adjacency`)

### Phase D: Betweenness + Closeness VTs (highest complexity)

- Shared SSSP refactor
- Optional `graph_sssp` shared cache VT
- Approximate betweenness (sampled sources) as a faster alternative

### Phase E: Benchmarks for Algorithm VTs

- Extend the benchmark suite from `graph_virtual_tables.md` Phase 5
- Measure: cached hit latency, recomputation time, warm-start speedup
- Compare: VT-cached vs fresh TVF invocation

## Risks & Considerations

### Storage Growth

Each algorithm VT adds O(V) rows of cached results. For a 100K-node graph:
- betweenness: ~800KB (100K × 8 bytes)
- closeness: ~800KB
- Leiden: ~400KB (100K × 4 bytes)
- PageRank: ~800KB
- components: ~800KB
- SSSP (if used): O(V²) = ~40GB for 100K nodes — NOT feasible at this scale

Total (without SSSP): ~3.6MB. Acceptable.

### Non-Determinism in Leiden

Leiden community IDs can vary between runs (the algorithm's local-moving
phase visits nodes in insertion order, and tie-breaking is order-dependent).
Caching makes community IDs stable across sessions — this is actually a
benefit, not a risk. However, `'rebuild'` should document that community
IDs may change.

### SSSP Cache Size Limits

The all-pairs SSSP cache is O(V²). Recommended threshold: only create
`graph_sssp` VT for graphs with V < 10,000. Beyond that, betweenness and
closeness should run their own per-source SSSP without caching intermediate
results.

### Backward Compatibility

The existing TVF syntax (`graph_betweenness('edges', 'src', 'dst', ...)`)
must continue to work alongside the new VT syntax. The VT form is an
enhancement for users who want persistent caching; the TVF form remains
for ad-hoc queries.

## References

- [Leiden Algorithm (Traag et al., 2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [Dynamic Leiden (2024)](https://arxiv.org/html/2405.11658v1)
- [Neo4j GDS: Graph Projections](https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/)
- [PecanPy: Fast Node2Vec (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/19/3377/6184859)
- [SuiteSparse:GraphBLAS Algorithm 1000](https://dl.acm.org/doi/10.1145/3322125)
- [Fast Approximation of Betweenness Centrality (2014)](https://dl.acm.org/doi/10.1145/2556195.2556224)
- Prerequisite spec: `docs/plans/graph_virtual_tables.md`
