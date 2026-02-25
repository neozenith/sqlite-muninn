# Graph Subsystem Gap Analysis

**Date:** 2026-02-24
**Status:** Active plan (distilled from `backlog/`)
**Scope:** All graph virtual tables, TVFs, shadow tables, and benchmarks

---

## Executive Summary

The muninn graph subsystem has a solid foundation: a `graph_adjacency` virtual table
with blocked CSR, delta merge, and trigger-based change tracking. Nine TVFs cover
traversal (BFS, DFS, shortest path), components, PageRank, centrality (degree,
betweenness, closeness), community detection (Leiden), and selector evaluation.

Three centrality/community TVFs already detect a `graph_adjacency` VT and read from
its CSR cache via `graph_data_load_from_adjacency()`. The remaining five base TVFs
(BFS, DFS, shortest path, components, PageRank) do **not** leverage the adjacency VT.

The gap is threefold:

1. **No namespace/scope support** — the adjacency VT serves a single graph per instance
2. **No downstream cached computation** — SSSP, components, and communities are
   recomputed from scratch on every TVF call
3. **No unified architecture** — the backlog proposed separate VTs per algorithm, but
   the correct design is a single VT with feature flags and a DAG of shadow tables

---

## Current State: What Is Implemented

### Source Files and Modules

| Module | File(s) | Status | Description |
|--------|---------|--------|-------------|
| **graph_adjacency** | `graph_adjacency.c/.h` | ✅ Phase 1-3 done | Blocked CSR VT with dirty flag, delta merge, blocked CSR |
| **graph_csr** | `graph_csr.c/.h` | ✅ Complete | CSR build, serialize/deserialize, delta merge, block ops |
| **graph_load** | `graph_load.c/.h` | ✅ Complete | Shared graph loading with hash-map, temporal WHERE clauses |
| **graph_tvf** | `graph_tvf.c/.h` | ✅ Base complete | BFS, DFS, shortest_path (lazy SQL), components (UnionFind), PageRank |
| **graph_centrality** | `graph_centrality.c/.h` | ✅ Complete | degree, betweenness, closeness — with temporal + adjacency VT detection |
| **graph_community** | `graph_community.c/.h` | ✅ Complete | Leiden — with temporal + adjacency VT detection |
| **graph_select** | `graph_select_tvf.c/.h` | ✅ Complete | dbt-style selector DSL parser + evaluator |
| **graph_selector** | `graph_selector_parse.c/.h`, `graph_selector_eval.c/.h` | ✅ Complete | AST parser + bit-vector evaluation |
| **id_validate** | `id_validate.c/.h` | ✅ Complete | SQL injection prevention for dynamic identifiers |
| **graph_common** | `graph_common.h` | ✅ Complete | Shared xBestIndex helper, hash function, safe_text |

### Registered Modules (muninn.c)

```
hnsw_index           — HNSW vector virtual table
graph_bfs            — BFS traversal TVF
graph_dfs            — DFS traversal TVF
graph_shortest_path  — Dijkstra/BFS shortest path TVF
graph_components     — Connected components TVF
graph_pagerank       — PageRank TVF
graph_degree         — Degree centrality TVF (adjacency-aware)
graph_betweenness    — Betweenness centrality TVF (adjacency-aware)
graph_closeness      — Closeness centrality TVF (adjacency-aware)
graph_leiden         — Leiden community detection TVF (adjacency-aware)
graph_adjacency      — Persistent CSR adjacency VT
graph_select         — dbt-style selector TVF
node2vec_train       — Node2Vec scalar function
muninn_embed/etc     — GGUF embedding functions (via llama.cpp)
```

### graph_adjacency Shadow Tables (Current)

```sql
{name}_config    — (key TEXT PK, value TEXT) — metadata KV store
{name}_nodes     — (idx INTEGER PK, id TEXT UNIQUE) — node registry
{name}_degree    — (idx INTEGER PK, in_deg, out_deg, w_in_deg, w_out_deg)
{name}_csr_fwd   — (block_id INTEGER PK, offsets BLOB, targets BLOB, weights BLOB)
{name}_csr_rev   — (block_id INTEGER PK, offsets BLOB, targets BLOB, weights BLOB)
{name}_delta     — (rowid INTEGER PK, src TEXT, dst TEXT, weight REAL, op INTEGER)
```

### Adjacency-Awareness Matrix

| TVF | Uses `is_graph_adjacency()`? | Uses `graph_data_load_from_adjacency()`? | Temporal? |
|-----|:-:|:-:|:-:|
| `graph_degree` | ✅ Yes | ✅ Yes | ✅ Yes |
| `graph_betweenness` | ✅ Yes | ✅ Yes | ✅ Yes |
| `graph_closeness` | ✅ Yes | ✅ Yes | ✅ Yes |
| `graph_leiden` | ✅ Yes | ✅ Yes | ✅ Yes |
| `graph_bfs` | ❌ No | ❌ No (lazy SQL) | ❌ No |
| `graph_dfs` | ❌ No | ❌ No (lazy SQL) | ❌ No |
| `graph_shortest_path` | ❌ No | ❌ No (lazy SQL) | ❌ No |
| `graph_components` | ❌ No | ❌ No (own SQL + UnionFind) | ❌ No |
| `graph_pagerank` | ❌ No | ❌ No (own SQL + PRAdjList) | ❌ No |
| `graph_select` | ❌ No | ❌ No (uses graph_data_load) | ❌ No |

---

## Gap 1: Scoped/Namespaced CSR (MANDATORY)

**Problem:** `graph_adjacency` builds one CSR over all edges. Real-world edge tables
contain multiple disjoint graphs keyed by scope columns (e.g., `project_id, session_id`).
Creating a separate VT per scope combination is impractical for dynamic scopes.

**Requirement:** A single `graph_adjacency` VT must partition its CSR, node index space,
degree cache, and delta log by namespace. Each namespace gets independent CSR blocks.
Queries filter by namespace. All downstream shadow tables (SSSP, components, communities)
must also be namespace-scoped.

**Current state:** Zero namespace awareness in any shadow table schema.

**Detailed plan:** [Phase 1: Scoped Adjacency VT](./01_scoped_adjacency_vt.md)

---

## Gap 2: SSSP Shadow Table (MANDATORY)

**Problem:** Betweenness centrality runs all-pairs SSSP: O(VE) exact. Closeness also
runs all-pairs SSSP: O(V²) unweighted. Running both on the same graph computes SSSP
twice. The `sssp_bfs` and `sssp_dijkstra` functions are static in `graph_centrality.c`
and cannot be shared.

**Requirement:** Extract SSSP into a shared module. Cache all-pairs SSSP results in
a shadow table of the adjacency VT when the `sssp` feature flag is enabled. Provide
generation-counter-based staleness detection so betweenness and closeness read from
cache when fresh.

**Current state:** SSSP is re-computed from scratch on every betweenness/closeness
TVF invocation. No shared module. No caching.

**Detailed plan:** [Phase 2: SSSP Shadow Tables](./02_sssp_shadow_tables.md)

---

## Gap 3: Components Shadow Table (MANDATORY)

**Problem:** `graph_components` in `graph_tvf.c` has its own SQL loading and UnionFind
implementation. It does not use `graph_load.c`, does not detect `graph_adjacency`, has
no temporal support, and recomputes from scratch every time.

**Requirement:** Cache connected component assignments in a shadow table of the adjacency
VT when the `components` feature flag is enabled. Union-Find runs O(V+E) — trivially
cheap to cache. The TVF must read from cache when fresh and recompute when stale.

**Current state:** Fully independent implementation with no caching or adjacency awareness.

**Detailed plan:** [Phase 3: Components Shadow Table](./03_components_shadow_tables.md)

---

## Gap 4: Communities Shadow Table (MANDATORY)

**Problem:** Leiden community detection is O(VE × iterations). Running it on the same
unchanged graph recomputes the same partition. The Dynamic Leiden variant (arXiv:2405.11658)
shows that seeding from a previous partition gives 1.1-1.4× speedup.

**Requirement:** Cache Leiden partition results in a shadow table of the adjacency VT
when the `communities` feature flag is enabled. Support warm-start from cached partition
on incremental rebuild. Optionally seed initial partition from component IDs.

**Current state:** `graph_leiden` recomputes from scratch each invocation. It does
detect `graph_adjacency` for loading but does not cache results.

**Detailed plan:** [Phase 4: Communities Shadow Table](./04_communities_shadow_tables.md)

---

## Gap 5: Unified VT Architecture (MANDATORY)

**Problem:** The backlog proposed separate VTs per algorithm (graph_betweenness VT,
graph_closeness VT, graph_components VT, etc.). This creates N separate CREATE
statements, N separate shadow table namespaces, and N independent staleness checks.

**Requirement:** A single `graph_adjacency` VT with feature flags that enable/disable
downstream shadow tables. The VT maintains a shared `_delta` table and a DAG of
incrementally-built cached artifacts. Feature flags allow users to trade insertion
throughput for query latency.

**Design:**

```sql
-- Full-featured adjacency with all downstream caches:
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst',
    weight_col='weight',
    namespace_cols='project_id,session_id',
    features='sssp,components,communities'
);

-- Lean adjacency for insertion-heavy workloads:
CREATE VIRTUAL TABLE g_lean USING graph_adjacency(
    edge_table='edges', src_col='src', dst_col='dst'
    -- no features = CSR only, minimal trigger overhead
);
```

**Shadow table DAG with generation counters:**

```
_delta (shared change log)
    │
    ▼
_csr_fwd/_csr_rev + _nodes + _degree  (generation G_adj)
    │
    ├──→ _sssp       (generation G_sssp, tracks G_adj)
    │        │
    │   (betweenness/closeness TVFs read from _sssp when fresh)
    │
    ├──→ _components  (generation G_comp, tracks G_adj)
    │        │
    │   (components TVF reads from _components when fresh)
    │   (Leiden can seed from _components)
    │
    └──→ _communities (generation G_comm, tracks G_adj, optionally G_comp)
             │
        (leiden TVF reads from _communities when fresh)
```

**Current state:** No feature flags. No downstream shadow tables. No generation DAG.

**Detailed plan:** [Phase 5: TVF/VT Integration](./05_tvf_vt_integration.md)

---

## Gap 6: Base TVF Adjacency + Temporal Awareness (MANDATORY)

**Problem:** The five base TVFs (BFS, DFS, shortest_path, components, PageRank) do not
detect `graph_adjacency` and do not support temporal filtering. They duplicate SQL
loading logic that `graph_load.c` already handles.

**Requirement:** All base TVFs must:
1. Detect when `edge_table` is a `graph_adjacency` VT and read from CSR cache
2. Accept `timestamp_col`, `time_start`, `time_end` parameters
3. Leverage the components shadow table when available (for `graph_components`)

**Current state:** Zero adjacency awareness or temporal support in base TVFs.

**Detailed plan:** Covered in [Phase 5: TVF/VT Integration](./05_tvf_vt_integration.md)

---

## Gap 7: Benchmark Suite (MANDATORY)

**Problem:** Existing benchmarks cover graph TVF query times and some VT build metrics
(in `benchmarks/charts/graph_vt_*.json`), but there is no systematic benchmark comparing:
- TVF-only (baseline) vs VT-cached performance across all algorithms
- Scoped vs non-scoped rebuild times
- Trigger overhead per insertion
- Cache hit vs miss latency
- Warm-start (Leiden/PageRank) vs cold-start recomputation

**Requirement:** A C-level benchmark harness and Python wrapper measuring wall time,
peak RSS, disk usage, and cache hit ratio across all approaches and workload sizes.

**Current state:** Partial benchmarks exist in `benchmarks/results/` and `benchmarks/charts/`.
No systematic comparison of VT+shadow-table permutations.

**Detailed plan:** [Phase 6: Benchmarks](./06_benchmarks.md)

---

## Research: Storage Approaches

### Our Approach: Blocked CSR in SQLite Shadow Tables

We store CSR arrays as BLOBs in shadow tables, chunked by 4096-node blocks. Delta
changes accumulate in `_delta` via triggers. When the delta exceeds a threshold (or
on explicit `rebuild`), the CSR is rebuilt from scratch or incrementally merged.

This is closest to the **Kuzu NodeGroups** approach (blocked columnar storage) and
the **GraphBLAS delta+merge** pattern.

### Academic Approaches Evaluated

| Approach | Paper | Write Cost | Read Quality | SQLite Fit | Decision |
|----------|-------|-----------|-------------|------------|----------|
| **Blocked CSR + delta** (ours) | GraphBLAS (2019) | O(1) per write (trigger) | Perfect after merge | ✅ Native B-tree BLOBs | **Current approach** |
| **PCSR** (packed memory arrays) | [Wheatman & Xu 2018](https://itshelenxu.github.io/files/papers/pcsr.pdf) | O(log² E) per edge | ~2× degraded (gaps) | ❌ Requires custom allocator | Rejected |
| **BACH** (LSM-tree graph) | [Miao et al., VLDB 2025](https://www.vldb.org/pvldb/vol18/p1509-miao.pdf) | O(log N) amortized | Excellent (compaction → CSR) | ❌ Storage engine inside storage engine | Rejected |
| **LSMGraph** (multi-level CSR) | [arXiv:2411.06392](https://arxiv.org/html/2411.06392v1) | O(log N) amortized | Good (version-controlled) | ❌ Requires LSM-tree primitives | Rejected |
| **GRainDB** (predefined joins) | [VLDB 2022](https://arxiv.org/abs/2108.10540) | Low (sidecar index) | Excellent | ⚠️ Needs storage engine hooks | Inspiration for trigger approach |
| **A+ Indexes** (Kuzu) | [arXiv:2004.00130](https://arxiv.org/abs/2004.00130) | Low (group-based) | Excellent | ⚠️ Columnar storage | Inspiration for blocked CSR |
| **DuckPGQ** (SQL/PGQ) | [CIDR 2023](https://www.cidrdb.org/cidr2023/papers/p66-wolde.pdf) | On-demand CSR | Perfect | ⚠️ DuckDB-specific | Query syntax inspiration |
| **RapidStore** (2025) | [arXiv:2507.00839](https://arxiv.org/html/2507.00839v1) | ART + buffer blocks | High concurrent | ❌ Custom engine | Not applicable |
| **GraphCSR** (degree-equalized) | [VLDB 2025](https://www.vldb.org/pvldb/vol18/p4255-gan.pdf) | N/A (static) | Excellent for analytics | ⚠️ Static graphs only | Potential future optimization |

### Why LSM Trees Are Not Appropriate Here

BACH and LSMGraph are compelling systems, but their core insight — using LSM-tree
compaction to progressively transform adjacency lists into CSR — requires control over
the storage engine. SQLite's B-tree is not an LSM tree, and implementing an LSM tree
inside SQLite shadow tables would be building a storage engine inside a storage engine.

Our blocked CSR + delta merge achieves the same goal (efficient reads via CSR, efficient
writes via delta log) using SQLite's native B-tree as the storage layer. The delta log
is functionally equivalent to an LSM "memtable" that flushes into CSR blocks during
rebuild. The key difference: we flush explicitly (via `rebuild` command or auto-threshold)
rather than via background compaction, which is the correct model for SQLite's
single-writer architecture.

**Verdict:** Our current approach is well-suited to SQLite's constraints. LSM trees
should not be reconsidered unless we move away from SQLite's storage engine entirely.

### Competitor Implementations

| Project | Language | Graph Storage | Graph Algorithms | Namespace Support |
|---------|----------|---------------|------------------|-------------------|
| **muninn (this)** | C11 | Blocked CSR in shadow tables | 9 TVFs + VT cache (planned) | Planned |
| **GraphQLite** | Rust | In-memory (SQLite persistence) | 15 (Cypher-based) | None |
| **sqlite-graph** | C99 | In-memory + SQLite persistence | Cypher parser (alpha) | None |
| **simple-graph** | SQL | Raw SQLite tables + CTEs | Path traversal only | None |
| **Neo4j GDS** | Java | In-memory CSR projection | 60+ algorithms | Label-based |
| **DuckPGQ** | C++ | On-demand CSR | SQL/PGQ queries | Schema-based |
| **Kuzu** (archived) | C++ | Columnar NodeGroups | Cypher queries | Label-based |

Key finding: **No SQLite graph extension supports namespace-scoped adjacency indexes**.
This would be a differentiating feature.

### References

#### Graph Storage
- [GRainDB: Predefined Joins (VLDB 2022)](https://arxiv.org/abs/2108.10540)
- [A+ Indexes: Lightweight Adjacency Lists (Kuzu/Waterloo)](https://arxiv.org/abs/2004.00130)
- [DuckPGQ: SQL/PGQ in DuckDB (CIDR 2023)](https://www.cidrdb.org/cidr2023/papers/p66-wolde.pdf)
- [Packed CSR (Wheatman & Xu, 2018)](https://itshelenxu.github.io/files/papers/pcsr.pdf)
- [BACH: Bridging Adjacency List and CSR (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p1509-miao.pdf)
- [LSMGraph: Multi-Level CSR (2024)](https://arxiv.org/html/2411.06392v1)
- [GraphCSR: Degree-Equalized CSR (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p4255-gan.pdf)
- [RapidStore: Dynamic Graph Storage (2025)](https://arxiv.org/html/2507.00839v1)

#### Graph Algorithms
- [SuiteSparse:GraphBLAS Algorithm 1000](https://dl.acm.org/doi/10.1145/3322125)
- [Leiden Algorithm (Traag et al., 2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [Dynamic Leiden (2024)](https://arxiv.org/html/2405.11658v1)
- [Fast Approximation of Betweenness Centrality (2014)](https://dl.acm.org/doi/10.1145/2556195.2556224)
- [PecanPy: Fast Node2Vec (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/19/3377/6184859)

#### Graph Database Design
- [Neo4j GDS: Graph Projections](https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/)
- [MV4PG: Materialized Views for Property Graphs (2024)](https://arxiv.org/html/2411.18847v1)
- [Graphiti / Zep: Temporal Knowledge Graph (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)

#### SQLite Virtual Tables
- [The Virtual Table Mechanism of SQLite](https://sqlite.org/vtab.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html)

#### Competitor Projects
- [GraphQLite (Rust, Cypher)](https://github.com/colliery-io/graphqlite)
- [sqlite-graph (C99, Cypher alpha)](https://github.com/agentflare-ai/sqlite-graph)
- [simple-graph (pure SQL)](https://github.com/dpapathanasiou/simple-graph)

---

## Implementation Roadmap

| Phase | Document | Depends On | Core Deliverable |
|-------|----------|------------|------------------|
| **1** | [Scoped Adjacency VT](./01_scoped_adjacency_vt.md) | — | Namespace-aware shadow tables, triggers, rebuild, query |
| **2** | [SSSP Shadow Tables](./02_sssp_shadow_tables.md) | Phase 1 | Shared SSSP module, cached all-pairs distances |
| **3** | [Components Shadow Table](./03_components_shadow_tables.md) | Phase 1 | Cached Union-Find, O(V+E) |
| **4** | [Communities Shadow Table](./04_communities_shadow_tables.md) | Phase 1, 3 | Cached Leiden partition, warm-start |
| **5** | [TVF/VT Integration](./05_tvf_vt_integration.md) | Phase 1-4 | All TVFs leverage shadow tables, unified feature flags |
| **6** | [Benchmarks](./06_benchmarks.md) | Phase 1-5 | Systematic VT vs non-VT comparison |

```
Phase 1 ─────────┬──→ Phase 2 ──→ Phase 5
(scoped CSR)     ├──→ Phase 3 ──→ Phase 4 ──→ Phase 5
                 └──→ Phase 6 (can start after Phase 1, iterate with each phase)
```

---

## Escalators-Not-Stairs Checklist

Every requirement below is **mandatory**. None may be silently downgraded to optional,
fallback, or "skip with warning".

| # | Requirement | Status | Verification |
|---|------------|--------|-------------|
| 1 | Scoped/namespaced CSR | ❌ Not started | `CREATE VT ... namespace_cols='a,b'` creates partitioned shadow tables |
| 2 | SSSP shadow table with generation counter | ❌ Not started | `SELECT * FROM g WHERE features LIKE '%sssp%'` shows cached distances |
| 3 | Components shadow table | ❌ Not started | `graph_components` TVF reads from `_components` shadow table when fresh |
| 4 | Communities shadow table with warm-start | ❌ Not started | Leiden rebuild from cached partition faster than cold start |
| 5 | Single VT with feature flags | ❌ Not started | `features='sssp,components,communities'` parameter accepted |
| 6 | All base TVFs detect graph_adjacency | ❌ Not started | BFS/DFS/SP/components/PageRank use CSR cache |
| 7 | All base TVFs support temporal params | ❌ Not started | `timestamp_col`, `time_start`, `time_end` accepted |
| 8 | Benchmark VT vs non-VT permutations | ❌ Not started | JSONL results comparing all approaches × all workloads |
| 9 | Research papers linked in plan docs | ✅ This document | Every referenced paper has a URL |
| 10 | Prior art evaluated against our approach | ✅ This document | Storage approach table with decision rationale |
