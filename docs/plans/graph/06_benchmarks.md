# Phase 6: Systematic Graph Benchmark Suite

**Date:** 2026-02-24
**Status:** Plan (not started)
**Depends on:** Phases 1-5
**Prerequisite code:** `graph_adjacency` VT with namespace support, SSSP/components/communities
shadow tables, all TVFs upgraded with adjacency detection and temporal awareness

---

## 1. Overview

### Why Systematic Benchmarks Are Mandatory

Phases 1-5 introduce significant new machinery: namespace-scoped CSR, three tiers of
shadow table caching (SSSP, components, communities), trigger-based change tracking,
delta merge strategies, warm-start Leiden, and adjacency-aware TVFs. Each feature has
an engineering cost (disk, memory, trigger latency, rebuild time) and an expected
payoff (faster queries, cached results, incremental updates). Without rigorous
measurement, we cannot:

1. **Quantify the value of each cached shadow table** --- betweenness with a warm SSSP
   cache should be 10-100x faster than recomputing all-pairs shortest paths, but the
   actual speedup depends on graph size, density, and structure.

2. **Identify crossover points where caching pays off** --- for a 100-node graph, the
   overhead of maintaining shadow tables may exceed the query-time savings. For a
   10K-node graph, the savings should dominate. The benchmark suite finds the exact
   crossover.

3. **Validate that the CSR path is always faster than SQL scan for in-memory queries**
   --- the adjacency-aware TVFs (Phase 5) should never be slower than the SQL-scan
   baseline. If they are, the detection logic or CSR loading has a bug.

4. **Measure the overhead cost of features** --- triggers add per-INSERT latency, shadow
   tables consume disk, namespace partitioning adds memory. Users need concrete numbers
   to choose the right feature flags for their workload.

5. **Establish regression baselines** --- future refactors must not degrade performance.
   JSONL results serve as the regression oracle.

### Relationship to Existing Benchmarks

The harness at `benchmarks/harness/` already provides the Treatment ABC pattern, JSONL
accumulation in `benchmarks/results/`, graph generation utilities (`generate_erdos_renyi`,
`generate_barabasi_albert`), and chart infrastructure (`ChartSpec` + aggregator +
renderer). The existing `graph_vt` category covers five approaches (tvf, csr,
csr_full_rebuild, csr_incremental, csr_blocked) for four algorithms (degree,
betweenness, closeness, leiden) at four workload sizes.

Phase 6 extends this foundation with:
- All nine algorithms (adding BFS, DFS, shortest_path, components, PageRank)
- Shadow table cache hit/miss measurements (SSSP, components, communities)
- Namespace-scoped benchmarks (1/10/100/1000 namespaces)
- Warm-start Leiden vs cold-start comparison
- SSSP disk/time feasibility measurements at scale
- Power-law and real-world graph topologies alongside Erdos-Renyi

---

## 2. Benchmark Matrix

### Axes

```
Algorithm:   {bfs, dfs, shortest_path, components, pagerank,
              degree, betweenness, closeness, leiden}

Mode:        {tvf_only, adjacency_vt, adjacency_vt_cached}

Size:        {100, 1_000, 10_000, 100_000}

Graph type:  {erdos_renyi, barabasi_albert, power_law_cluster, real_world}

Scope:       {unscoped, 1_namespace, 10_namespaces, 100_namespaces}
```

### Mode Definitions

| Mode | Description | Shadow Tables Used |
|------|-------------|--------------------|
| `tvf_only` | TVF scans edge table via SQL on every call. No VT, no CSR. Baseline. | None |
| `adjacency_vt` | TVF detects `graph_adjacency` VT, loads from CSR cache. No downstream caches. | `_csr_fwd`, `_csr_rev`, `_nodes`, `_degree` |
| `adjacency_vt_cached` | TVF reads from downstream shadow table cache when generation is fresh. Full feature flags enabled. | All of the above + `_sssp` and/or `_components` and/or `_communities` (algorithm-dependent) |

### Algorithm-Mode Applicability

Not every algorithm uses every mode. The `adjacency_vt_cached` mode only applies to
algorithms that have a downstream shadow table:

| Algorithm | tvf_only | adjacency_vt | adjacency_vt_cached | Cache Source |
|-----------|:--------:|:------------:|:-------------------:|:------------:|
| bfs | Y | Y | -- | (no cache) |
| dfs | Y | Y | -- | (no cache) |
| shortest_path | Y | Y | -- | (no cache; SSSP is all-pairs, SP is point-to-point) |
| components | Y | Y | Y | `_components` |
| pagerank | Y | Y | -- | (no cache) |
| degree | Y | Y | -- | (computed from CSR directly) |
| betweenness | Y | Y | Y | `_sssp` |
| closeness | Y | Y | Y | `_sssp` |
| leiden | Y | Y | Y | `_communities` |

### Total Permutations

Applicable permutations (excluding inapplicable algorithm-mode combinations):

| Category | Algorithms | Modes | Sizes | Graph Types | Scopes | Subtotal |
|----------|-----------|-------|-------|-------------|--------|----------|
| Traversal (bfs, dfs, sp) | 3 | 2 | 4 | 4 | 4 | 384 |
| Components | 1 | 3 | 4 | 4 | 4 | 192 |
| PageRank | 1 | 2 | 4 | 4 | 4 | 128 |
| Degree | 1 | 2 | 4 | 4 | 4 | 128 |
| Betweenness | 1 | 3 | 4 | 4 | 4 | 192 |
| Closeness | 1 | 3 | 4 | 4 | 4 | 192 |
| Leiden | 1 | 3 | 4 | 4 | 4 | 192 |
| **Total** | **9** | | | | | **1,408** |

With 3 repetitions per measurement for statistical robustness: **4,224 runs**.

### Size Constraints

Betweenness and closeness are O(VE) and O(V^2) respectively. At 100K nodes these are
prohibitively expensive for the `tvf_only` and `adjacency_vt` modes. The matrix applies
algorithm-specific size caps:

| Algorithm | Max Size (tvf_only) | Max Size (adjacency_vt) | Max Size (adjacency_vt_cached) |
|-----------|:-------------------:|:-----------------------:|:------------------------------:|
| bfs, dfs, shortest_path | 100K | 100K | -- |
| components | 100K | 100K | 100K |
| pagerank | 100K | 100K | -- |
| degree | 100K | 100K | -- |
| betweenness | 10K | 10K | 10K |
| closeness | 10K | 10K | 10K |
| leiden | 100K | 100K | 100K |

---

## 3. Graph Generation

All graph generators are deterministic (seeded RNG) for reproducibility.

### 3.1 Erdos-Renyi Random Graphs

Uses the existing `generate_erdos_renyi()` from `benchmarks/harness/common.py`.
Controllable density via `avg_degree` parameter.

```
Parameters: n_nodes, avg_degree=5, weighted=True, seed=42
Properties: Uniform degree distribution, no clustering
Use case:   Baseline --- tests algorithms on graphs with no structural bias
```

### 3.2 Barabasi-Albert Power-Law Graphs

Uses the existing `generate_barabasi_albert()` from `benchmarks/harness/common.py`.
Preferential attachment produces power-law (scale-free) degree distribution.

```
Parameters: n_nodes, m=3, weighted=True, seed=42
Properties: Power-law degree distribution, hub-and-spoke structure
Use case:   Realistic social/citation network topology
```

### 3.3 Power-Law Cluster Graphs

A new generator combining Barabasi-Albert preferential attachment with Watts-Strogatz
local clustering. This produces graphs with both hubs (power-law degree) and community
structure (high clustering coefficient).

```
Parameters: n_nodes, m=3, p_triangle=0.3, weighted=True, seed=42
Properties: Power-law degree + high clustering coefficient
Use case:   Tests community detection (Leiden) on realistic structure
```

Implementation: After standard Barabasi-Albert attachment, for each new edge (u, v),
with probability `p_triangle`, also connect u to a random neighbor of v (triangle
closure). This is the Holme-Kim model.

### 3.4 Real-World Graph

The Wealth of Nations knowledge graph from `benchmarks/kg/3300_chunks.db`. This is
the same graph used in the demo database builder. Entity-to-entity edges derived from
relation extraction provide a naturally structured graph with ~500-2000 nodes depending
on extraction quality.

```
Source:     benchmarks/kg/3300_chunks.db (entity relations)
Properties: Variable density, natural community structure, weighted by co-occurrence
Use case:   Ground truth for realistic workloads
```

### 3.5 Scoped Graph Generation

For namespace benchmarks, a base graph of total size N is partitioned into K namespaces:

```
Strategy: For each edge (u, v, w) in the base graph, assign it to namespace
          k = hash(u, v) % K. Each namespace gets approximately N/K nodes
          and E/K edges, but with natural variation from the hash distribution.
```

This produces K independent subgraphs within the same edge table, each tagged with a
`namespace` column. The `graph_adjacency` VT is created with `namespace_cols='namespace'`.

---

## 4. Benchmark Scripts (Treatment Classes)

All benchmarks are implemented as Treatment subclasses in the existing harness framework,
registered in `registry.py`, and executable via the CLI:

```bash
uv run --directory . -m benchmarks.harness benchmark --id <permutation_id>
```

### 4.1 Treatment Classes

| File | Class | Category | Description |
|------|-------|----------|-------------|
| `treatments/graph_vt_tvf_baseline.py` | `GraphVtTvfBaselineTreatment` | `graph_vt_phase6` | TVF-only baseline for all 9 algorithms |
| `treatments/graph_vt_adjacency.py` | `GraphVtAdjacencyTreatment` | `graph_vt_phase6` | Adjacency-aware TVF path (CSR-loaded, no downstream cache) |
| `treatments/graph_vt_cached.py` | `GraphVtCachedTreatment` | `graph_vt_phase6` | Full shadow table cached path |
| `treatments/graph_vt_overhead.py` | `GraphVtOverheadTreatment` | `graph_vt_overhead` | Trigger and rebuild overhead isolation |
| `treatments/graph_vt_scope.py` | `GraphVtScopeTreatment` | `graph_vt_scope` | Namespace scaling measurements |
| `treatments/graph_vt_warmstart.py` | `GraphVtWarmstartTreatment` | `graph_vt_warmstart` | Leiden warm-start vs cold-start comparison |

### 4.2 Common Harness Module

`benchmarks/harness/graph_bench_common.py` provides shared utilities for Phase 6:

```python
# Graph generation (extends existing common.py generators)
def generate_power_law_cluster(n_nodes, m, p_triangle, weighted, seed) -> (edges, adj)
def load_real_world_graph(kg_db_path) -> (edges, adj)
def partition_into_namespaces(edges, n_namespaces) -> list[(namespace, edges)]

# Measurement helpers
def measure_shadow_table_disk(conn, vtab_name, shadow_suffix) -> int
def measure_cache_generation(conn, vtab_name) -> int
def measure_peak_rss_delta(fn) -> (result, rss_delta_kb)

# Scoped graph setup
def setup_scoped_edge_table(conn, edges_by_namespace) -> str
def create_scoped_adjacency_vt(conn, edge_table, namespaces, features) -> str
```

### 4.3 Permutation ID Convention

Following the existing naming pattern in the harness:

```
{category}_{algorithm}_{mode}_{size}_{graph_type}[_scope{N}]

Examples:
  graph_vt_phase6_betweenness_tvf_only_10000_barabasi_albert
  graph_vt_phase6_leiden_adjacency_vt_cached_1000_erdos_renyi
  graph_vt_scope_leiden_adjacency_vt_cached_10000_erdos_renyi_scope100
  graph_vt_warmstart_leiden_warm_10000_barabasi_albert
  graph_vt_overhead_trigger_10000_erdos_renyi
```

---

## 5. JSONL Output Format

### 5.1 Schema

Each benchmark run produces one JSONL line. The schema extends the existing `graph_vt`
format with additional fields for Phase 6 measurements.

```json
{
  "permutation_id": "graph_vt_phase6_betweenness_adjacency_vt_cached_10000_power_law",
  "category": "graph_vt_phase6",
  "wall_time_setup_ms": 245.3,
  "wall_time_run_ms": 142.5,
  "peak_rss_mb": 256.4,
  "rss_delta_mb": 18.2,
  "db_size_bytes": 2097152,
  "timestamp": "2026-02-24T12:00:00.000000+00:00",
  "platform": "darwin-arm64",
  "python_version": "3.12.8",

  "algorithm": "betweenness",
  "mode": "adjacency_vt_cached",
  "n_nodes": 10000,
  "n_edges": 99876,
  "avg_degree": 10,
  "graph_type": "power_law",
  "scope": {
    "namespaces": 1,
    "scope_type": "unscoped"
  },

  "metrics": {
    "query_time_ms": 142.5,
    "build_time_ms": 34.2,
    "rebuild_time_ms": null,
    "cache_hit": true,
    "cache_generation": 3,
    "cache_stale": false,
    "shadow_table_disk_bytes": {
      "_csr_fwd": 524288,
      "_csr_rev": 524288,
      "_nodes": 81920,
      "_degree": 40960,
      "_sssp": 819200,
      "_components": null,
      "_communities": null
    },
    "total_shadow_disk_bytes": 1990656,
    "trigger_overhead_us_per_insert": 8.3,
    "trigger_batch_size": 1000
  },

  "repetition": 1,
  "n_repetitions": 3,

  "system": {
    "os": "darwin",
    "arch": "arm64",
    "cpu": "Apple M1",
    "ram_gb": 16,
    "sqlite_version": "3.45.0"
  }
}
```

### 5.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `permutation_id` | string | Unique slug identifying this exact benchmark configuration |
| `category` | string | One of: `graph_vt_phase6`, `graph_vt_overhead`, `graph_vt_scope`, `graph_vt_warmstart` |
| `algorithm` | string | One of the 9 algorithm names |
| `mode` | string | `tvf_only`, `adjacency_vt`, or `adjacency_vt_cached` |
| `n_nodes` | int | Number of nodes in the graph |
| `n_edges` | int | Actual number of edges inserted |
| `avg_degree` | float | Average degree (n_edges / n_nodes) |
| `graph_type` | string | `erdos_renyi`, `barabasi_albert`, `power_law_cluster`, `real_world` |
| `scope.namespaces` | int | Number of namespaces (1 = unscoped) |
| `scope.scope_type` | string | `unscoped`, `scoped` |
| `metrics.query_time_ms` | float | Wall time for the algorithm TVF call |
| `metrics.build_time_ms` | float | Wall time for initial CSR build (null if tvf_only) |
| `metrics.rebuild_time_ms` | float | Wall time for incremental rebuild (null if not applicable) |
| `metrics.cache_hit` | bool | Whether the downstream shadow table was fresh |
| `metrics.cache_generation` | int | Generation counter value at query time |
| `metrics.shadow_table_disk_bytes` | object | Per-shadow-table disk usage via `dbstat` |
| `metrics.trigger_overhead_us_per_insert` | float | Microseconds of overhead per INSERT from triggers |
| `repetition` | int | 1-indexed repetition number |
| `n_repetitions` | int | Total repetitions (always 3) |
| `system` | object | Platform identification for reproducibility |

### 5.3 File Naming Convention

JSONL files accumulate in `benchmarks/results/` following the existing convention:

```
{permutation_id}.jsonl

Examples:
  graph_vt_phase6_betweenness_adjacency_vt_cached_10000_power_law.jsonl
  graph_vt_scope_leiden_adjacency_vt_cached_10000_erdos_renyi_scope100.jsonl
  graph_vt_warmstart_leiden_warm_10000_barabasi_albert.jsonl
```

---

## 6. Specific Benchmark Scenarios

### 6a. TVF-Only vs CSR-Loaded (Validates Phase 5 Adjacency Detection)

**Goal:** Prove that the adjacency-aware TVF path is faster than the SQL-scan baseline
for every algorithm at every size above the crossover point.

**Method:**
1. Generate graph at size N with type T
2. Mode `tvf_only`: call each TVF against the raw edge table
3. Mode `adjacency_vt`: create `graph_adjacency` VT, call each TVF against the VT name
4. Compare `query_time_ms` between modes

**Expected results:**

| Size | Expected Speedup (CSR over SQL scan) |
|------|--------------------------------------|
| 100 | 1.0-1.5x (overhead dominates at small N) |
| 1K | 2-5x |
| 10K | 5-10x |
| 100K | 10-50x |

**Failure criteria:** If `adjacency_vt` is slower than `tvf_only` at N >= 1K for any
algorithm, the adjacency detection or CSR loading path has a performance bug.

### 6b. Cache Hit vs Cache Miss (Validates Phases 2-4 Shadow Tables)

**Goal:** Measure the speedup from reading cached shadow table results vs recomputing
from scratch.

**Method for betweenness/closeness (SSSP cache):**
1. Create `graph_adjacency` VT with `features='sssp'`
2. First call: `SELECT * FROM graph_betweenness(...)` --- this is a cache miss, builds `_sssp`
3. Second call: same query --- this is a cache hit, reads from `_sssp`
4. Compare `query_time_ms` between call 1 and call 2

**Method for components:**
1. Create `graph_adjacency` VT with `features='components'`
2. First call: `SELECT * FROM graph_components(...)` --- builds `_components`
3. Second call: same query --- reads from `_components`

**Method for leiden:**
1. Create `graph_adjacency` VT with `features='communities'`
2. First call: `SELECT * FROM graph_leiden(...)` --- builds `_communities`
3. Second call: same query --- reads from `_communities`

**Expected results:**

| Algorithm | Size | Cache Miss (ms) | Cache Hit (ms) | Speedup |
|-----------|------|-----------------|----------------|---------|
| betweenness | 1K | ~50 | ~0.5 | 100x |
| betweenness | 5K | ~1,000 | ~2 | 500x |
| betweenness | 10K | ~5,000 | ~10 | 500x |
| closeness | 1K | ~50 | ~0.5 | 100x |
| components | 1K | ~2 | ~0.1 | 20x |
| components | 10K | ~20 | ~0.5 | 40x |
| leiden | 1K | ~5 | ~0.2 | 25x |
| leiden | 10K | ~100 | ~2 | 50x |

**Failure criteria:** If cache hit is not at least 10x faster than cache miss for
betweenness/closeness at N >= 1K, the SSSP caching is not working correctly.

### 6c. Trigger Overhead (Validates Phase 1 Delta Approach)

**Goal:** Measure the per-INSERT cost of the auto-installed triggers that track changes
for the `graph_adjacency` VT.

**Method:**
1. Create edge table and `graph_adjacency` VT (triggers auto-installed)
2. INSERT a batch of N edges, measure wall time (T_with)
3. DROP all triggers
4. INSERT the same batch again, measure wall time (T_without)
5. Per-INSERT overhead = (T_with - T_without) / N

**Workloads:**

| Batch Size | Graph Size | Edge Table Rows |
|------------|-----------|-----------------|
| 100 | 1K | 5K |
| 1,000 | 10K | 50K |
| 10,000 | 100K | 500K |
| 100,000 | 100K | 1M |

**Expected results:**

| Batch Size | Per-INSERT Overhead |
|------------|---------------------|
| 100 | < 50 us |
| 1,000 | < 20 us |
| 10,000 | < 10 us |
| 100,000 | < 10 us |

**Failure criteria:** If per-INSERT overhead exceeds 50 us at any batch size, the
trigger implementation needs optimization (e.g., batch delta coalescing).

### 6d. Rebuild Times (Validates Phase 1 Blocked CSR)

**Goal:** Compare full rebuild vs incremental rebuild at various delta sizes to find the
crossover where incremental is cheaper.

**Method:**
1. Create graph of size N, build `graph_adjacency` VT
2. Insert delta_pct% new edges (1%, 5%, 10%, 20%, 50%)
3. Measure full rebuild time: `INSERT INTO g(g) VALUES ('rebuild')`
4. Reset, re-insert same delta
5. Measure incremental rebuild time: `INSERT INTO g(g) VALUES ('incremental_rebuild')`
6. Also measure blocked incremental (mutations concentrated in 1 block)

**Expected results:**

| Delta % | Full Rebuild | Incremental | Blocked Incremental | Cheapest |
|---------|-------------|-------------|---------------------|----------|
| 1% | 1.0x | 0.1x | 0.05x | blocked |
| 5% | 1.0x | 0.3x | 0.15x | blocked |
| 10% | 1.0x | 0.5x | 0.3x | blocked |
| 20% | 1.0x | 0.8x | 0.6x | incremental (marginal) |
| 50% | 1.0x | 1.1x | 1.0x | full |

The crossover where incremental becomes more expensive than full rebuild should be
near delta = 20% of total edges.

**Failure criteria:** If incremental rebuild is slower than full rebuild at delta < 10%,
the delta merge implementation has a bug or excessive overhead.

### 6e. Namespace Scaling (Validates Phase 1 Scoping)

**Goal:** Verify that namespace-scoped rebuilds scale linearly with the number of
affected namespaces, and that querying a single namespace is independent of the total
number of namespaces.

**Method:**
1. Generate a base graph of 100K edges
2. Partition into K namespaces (1, 10, 100, 1000)
3. Create `graph_adjacency` VT with `namespace_cols='namespace'`
4. Measure: rebuild time for all namespaces
5. Measure: rebuild time for 1 namespace (after mutating only that namespace)
6. Measure: query time for 1 namespace via TVF with namespace filter

**Expected results:**

| K Namespaces | Full Rebuild (ms) | Single NS Rebuild (ms) | Single NS Query (ms) |
|-------------|-------------------|------------------------|---------------------|
| 1 | T_base | T_base | Q_base |
| 10 | ~1.0x T_base | ~0.1x T_base | ~0.1x Q_base |
| 100 | ~1.0-1.2x T_base | ~0.01x T_base | ~0.01x Q_base |
| 1000 | ~1.2-1.5x T_base | ~0.001x T_base | ~0.001x Q_base |

**Failure criteria:** If single-namespace rebuild time does not decrease proportionally
to 1/K, the scoped rebuild is touching all namespaces instead of only the dirty one.

### 6f. Warm-Start Leiden (Validates Phase 4)

**Goal:** Measure the speedup from seeding Leiden with a cached partition vs starting
from singletons, after incremental graph changes.

**Method:**
1. Generate graph, run Leiden to completion, cache partition in `_communities`
2. Mutate graph by delta_pct% (add/remove edges)
3. Cold start: run Leiden from singleton partition
4. Warm start: run Leiden seeded from cached `_communities` partition
5. Component-seeded: run Leiden seeded from `_components` (if available)

**Delta sizes:** 1%, 5%, 10%, 20%

**Expected results (from Dynamic Leiden paper, arXiv:2405.11658):**

| Delta % | Cold Start (ms) | Warm Start (ms) | Speedup | Quality (modularity) |
|---------|----------------|-----------------|---------|---------------------|
| 1% | T_cold | ~0.7x T_cold | 1.4x | identical |
| 5% | T_cold | ~0.75x T_cold | 1.3x | identical |
| 10% | T_cold | ~0.8x T_cold | 1.2x | < 0.01 modularity loss |
| 20% | T_cold | ~0.9x T_cold | 1.1x | < 0.02 modularity loss |

The warm-start speedup is modest (1.1-1.4x) because Leiden's inner loop dominates.
The primary value is avoiding convergence to a different partition when changes are small.

**Also measure:** Modularity of warm-start vs cold-start partitions to verify that
warm start does not degrade community quality.

**Failure criteria:** If warm start is slower than cold start at any delta size, or if
modularity degrades by more than 0.05, the warm-start seeding is counterproductive.

### 6g. SSSP Feasibility (Validates Phase 2 Size Limits)

**Goal:** Establish the practical size limits for caching all-pairs SSSP results in the
`_sssp` shadow table.

**Method:**
1. Generate graph at sizes V = 500, 1K, 2K, 5K, 10K, 20K
2. Create `graph_adjacency` VT with `features='sssp'`
3. Trigger SSSP build (first betweenness query)
4. Measure: build time, `_sssp` shadow table disk size, peak RSS during build

**Expected storage (dense SSSP matrix):**

The SSSP cache stores dist(u, v) and sigma(u, v) for all pairs. Storage is O(V^2).

| V | Pairs (V^2) | Estimated Disk (float32 dist + uint32 sigma per pair) | Build Time |
|---|------------|-------------------------------------------------------|-----------|
| 500 | 250K | ~2 MB | < 1s |
| 1K | 1M | ~8 MB | ~2s |
| 2K | 4M | ~32 MB | ~10s |
| 5K | 25M | ~200 MB | ~60s |
| 10K | 100M | ~800 MB | ~300s |
| 20K | 400M | ~3.2 GB | ~1200s |

**Decision threshold:** If disk exceeds 500 MB or build time exceeds 60s, the SSSP
cache should emit a warning and fall back to on-demand computation. This determines
the `SSSP_MAX_NODES` constant for Phase 2.

**Failure criteria:** If actual disk usage is more than 2x the estimated value, the
SSSP serialization format is inefficient and needs compaction (e.g., sparse storage
for disconnected pairs).

---

## 7. Visualization

### 7.1 Chart Specifications

Charts are defined as `ChartSpec` instances in
`benchmarks/harness/analysis/charts_graph_vt_phase6.py`, following the existing pattern
in `charts_graph_vt.py`.

#### Bar Charts: Algorithm x Mode Comparison

One chart per algorithm showing query time across the three modes at each size.
Total: 9 charts.

```python
ChartSpec(
    name="graph_vt_p6_{algorithm}",
    title="{Algorithm} Query Time by Mode",
    sources=["graph_vt_phase6_*.jsonl"],
    filters={"algorithm": "{algorithm}"},
    x_field="n_nodes",
    y_field="metrics.query_time_ms",
    group_fields=["mode"],
    variant_fields=[],
    repeat_fields=["mode", "n_nodes", "graph_type"],
    y_label="Query Time (ms)",
    x_label="Graph Size (nodes)",
    log_x=True,
    log_y=True,
)
```

#### Line Charts: Performance Scaling Curves

One chart per mode showing all algorithms' query times as the graph grows.
Total: 3 charts.

```python
ChartSpec(
    name="graph_vt_p6_scaling_{mode}",
    title="Algorithm Scaling ({Mode})",
    sources=["graph_vt_phase6_*.jsonl"],
    filters={"mode": "{mode}"},
    x_field="n_nodes",
    y_field="metrics.query_time_ms",
    group_fields=["algorithm"],
    ...
)
```

#### Heatmap: Full Benchmark Matrix

Single heatmap with algorithms on the y-axis, sizes on the x-axis, and color intensity
representing the speedup ratio (adjacency_vt_cached / tvf_only). Green = large speedup,
red = regression.

#### Scatter: Trigger Overhead vs Edge Count

X-axis: number of edges in edge table. Y-axis: per-INSERT trigger overhead (us).
One series per graph type.

#### Bar: Cache Hit vs Miss

Paired bar chart showing cache-miss time and cache-hit time side by side for each
cached algorithm at each size.

#### Line: Namespace Scaling

X-axis: number of namespaces (log scale). Y-axis: single-namespace rebuild time (ms).
Expected: linear decrease.

#### Bar: Warm-Start vs Cold-Start Leiden

Grouped bar chart: cold start, warm start, and component-seeded start at each delta
percentage. Secondary y-axis: modularity score.

#### Bar: SSSP Feasibility

X-axis: graph size (V). Two y-axes: disk usage (MB, left) and build time (s, right).
Clear visual of where the feasibility threshold lies.

### 7.2 Output

All charts render to `benchmarks/charts/graph_vt_p6_*.png` via matplotlib.

---

## 8. Implementation Steps

### Step 1: graph_bench_common.py

Create `benchmarks/harness/graph_bench_common.py` with:
- `generate_power_law_cluster()` (Holme-Kim model)
- `load_real_world_graph()` (loads from KG chunk database)
- `partition_into_namespaces()`
- `measure_shadow_table_disk()` (per-table breakdown)
- `measure_cache_generation()`

**Depends on:** Nothing (pure Python + existing common.py utilities)

### Step 2: GraphVtTvfBaselineTreatment

Create `benchmarks/harness/treatments/graph_vt_tvf_baseline.py`:
- Extends Treatment ABC
- Generates graph, runs all 9 algorithms against raw edge table via SQL
- Records per-algorithm `query_time_ms`
- Category: `graph_vt_phase6`

**Depends on:** Step 1

### Step 3: GraphVtAdjacencyTreatment

Create `benchmarks/harness/treatments/graph_vt_adjacency.py`:
- Creates `graph_adjacency` VT (no feature flags)
- Runs all 9 algorithms against VT name
- Records per-algorithm `query_time_ms` + `build_time_ms` + `disk_bytes`

**Depends on:** Step 1, Phase 5 implementation (all TVFs adjacency-aware)

### Step 4: GraphVtCachedTreatment

Create `benchmarks/harness/treatments/graph_vt_cached.py`:
- Creates `graph_adjacency` VT with `features='sssp,components,communities'`
- Runs applicable algorithms twice (miss then hit)
- Records cache miss/hit times, generation counters, per-shadow-table disk

**Depends on:** Step 1, Phases 2-4 implementation (shadow tables exist)

### Step 5: GraphVtOverheadTreatment

Create `benchmarks/harness/treatments/graph_vt_overhead.py`:
- Tests trigger overhead (with/without triggers, various batch sizes)
- Tests rebuild times (full vs incremental vs blocked at various delta percentages)
- Category: `graph_vt_overhead`

**Depends on:** Step 1, Phase 1 implementation (scoped adjacency with triggers)

### Step 6: GraphVtScopeTreatment

Create `benchmarks/harness/treatments/graph_vt_scope.py`:
- Partitions graph into N namespaces
- Measures full rebuild, single-namespace rebuild, single-namespace query
- Category: `graph_vt_scope`

**Depends on:** Step 1, Phase 1 implementation (namespace support)

### Step 7: GraphVtWarmstartTreatment

Create `benchmarks/harness/treatments/graph_vt_warmstart.py`:
- Runs Leiden cold vs warm vs component-seeded after delta mutations
- Records times and modularity scores
- Category: `graph_vt_warmstart`

**Depends on:** Step 1, Phase 4 implementation (warm-start Leiden)

### Step 8: Register permutations in registry.py

Add `_graph_vt_phase6_permutations()`, `_graph_vt_overhead_permutations()`,
`_graph_vt_scope_permutations()`, `_graph_vt_warmstart_permutations()` functions to
`benchmarks/harness/registry.py`. Wire into `all_permutations()`.

**Depends on:** Steps 2-7

### Step 9: Chart definitions

Create `benchmarks/harness/analysis/charts_graph_vt_phase6.py` with all ChartSpec
instances from Section 7. Register in the analysis pipeline.

**Depends on:** Step 8

### Step 10: Run full matrix and collect results

Execute all permutations via `make -C benchmarks graph-vt-phase6-all`. Collect JSONL.
Generate charts. Analyze results.

**Depends on:** Steps 1-9

---

## 9. Makefile Targets

Add the following targets to `benchmarks/Makefile`:

```makefile
# ── Phase 6 Graph VT Benchmarks ─────────────────────────────────

benchmark-graph-vt-phase6: $(EXTENSION)      ## Run all missing Phase 6 graph VT benchmarks
	$(BENCH) manifest --missing --category graph_vt_phase6 --commands | sh

benchmark-graph-vt-overhead: $(EXTENSION)    ## Run all missing trigger/rebuild overhead benchmarks
	$(BENCH) manifest --missing --category graph_vt_overhead --commands | sh

benchmark-graph-vt-scope: $(EXTENSION)       ## Run all missing namespace scaling benchmarks
	$(BENCH) manifest --missing --category graph_vt_scope --commands | sh

benchmark-graph-vt-warmstart: $(EXTENSION)   ## Run all missing Leiden warm-start benchmarks
	$(BENCH) manifest --missing --category graph_vt_warmstart --commands | sh

benchmark-graph-vt-phase6-all: $(EXTENSION)  ## Run ALL missing Phase 6 benchmarks
	$(BENCH) manifest --missing --category graph_vt_phase6 --commands | sh
	$(BENCH) manifest --missing --category graph_vt_overhead --commands | sh
	$(BENCH) manifest --missing --category graph_vt_scope --commands | sh
	$(BENCH) manifest --missing --category graph_vt_warmstart --commands | sh

benchmark-graph-vt-phase6-charts:            ## Generate Phase 6 charts from results
	$(BENCH) analyse --charts graph_vt_p6
```

Also add these targets to the `.PHONY` list:

```makefile
.PHONY: ... benchmark-graph-vt-phase6 benchmark-graph-vt-overhead \
        benchmark-graph-vt-scope benchmark-graph-vt-warmstart \
        benchmark-graph-vt-phase6-all benchmark-graph-vt-phase6-charts
```

---

## 10. Verification Criteria

### 10.1 Coverage Requirements

| Requirement | Criterion |
|-------------|-----------|
| All 9 algorithms measured | Every algorithm appears in at least 3 modes x 4 sizes of JSONL output |
| Statistical robustness | Every permutation has 3 repetitions with mean and stddev reported |
| All graph types covered | Each algorithm measured on erdos_renyi, barabasi_albert, power_law_cluster, and real_world |
| Namespace scaling measured | At least 4 namespace counts (1, 10, 100, 1000) for 2 algorithms |
| Warm-start measured | Leiden cold/warm/component-seeded at 4 delta percentages |
| SSSP feasibility measured | At least 6 graph sizes from V=500 to V=20K |

### 10.2 Reproducibility

- All graph generators use deterministic seeds (default `seed=42`)
- System metadata (OS, arch, CPU, RAM, SQLite version) recorded in every JSONL line
- JSONL files are append-only; re-running a permutation appends rather than overwrites
- Charts auto-generated from JSONL without manual data munging

### 10.3 JSONL Validation

A validation script checks every JSONL line against the schema:

```python
REQUIRED_FIELDS = [
    "permutation_id", "category", "wall_time_setup_ms", "wall_time_run_ms",
    "peak_rss_mb", "timestamp", "platform", "algorithm", "mode",
    "n_nodes", "n_edges", "graph_type",
]

def validate_jsonl_line(record: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    if record.get("n_repetitions", 3) != 3:
        errors.append(f"Expected 3 repetitions, got {record.get('n_repetitions')}")
    return errors
```

### 10.4 Regression Detection

After each full benchmark run, compare against the previous run's results:

```python
REGRESSION_THRESHOLD = 2.0  # Flag if any metric is >2x slower

def check_regressions(current: list[dict], previous: list[dict]) -> list[str]:
    """Return list of regression warnings."""
    warnings = []
    prev_by_id = {r["permutation_id"]: r for r in previous}
    for curr in current:
        pid = curr["permutation_id"]
        if pid in prev_by_id:
            prev_time = prev_by_id[pid].get("metrics", {}).get("query_time_ms", 0)
            curr_time = curr.get("metrics", {}).get("query_time_ms", 0)
            if prev_time > 0 and curr_time > REGRESSION_THRESHOLD * prev_time:
                warnings.append(
                    f"REGRESSION: {pid} went from {prev_time:.1f}ms to {curr_time:.1f}ms "
                    f"({curr_time/prev_time:.1f}x slower)"
                )
    return warnings
```

---

## 11. Success Metrics

The Phase 6 benchmark suite is complete when all of the following are demonstrated:

| # | Metric | Target | Measured By |
|---|--------|--------|-------------|
| 1 | CSR path faster than SQL scan | For ALL 9 algorithms at N >= 1K | Scenario 6a |
| 2 | Cache hit faster than cache miss | For ALL 4 cached algorithms (betweenness, closeness, components, leiden) at N >= 1K | Scenario 6b |
| 3 | Cache hit speedup for SSSP-backed algorithms | >= 10x for betweenness/closeness at N >= 1K | Scenario 6b |
| 4 | Trigger overhead within budget | < 50 us per INSERT at all batch sizes | Scenario 6c |
| 5 | Incremental rebuild faster than full rebuild | When delta < 20% of total edges | Scenario 6d |
| 6 | Namespace query scales with partition size | Single-NS query time ~ O(E/K) where K = number of namespaces | Scenario 6e |
| 7 | Warm-start Leiden faster than cold start | For delta <= 10%, speedup >= 1.1x | Scenario 6f |
| 8 | Warm-start Leiden preserves quality | Modularity loss < 0.05 for delta <= 20% | Scenario 6f |
| 9 | SSSP feasibility boundary identified | Clear V_max where disk > 500 MB or time > 60s | Scenario 6g |
| 10 | All results documented | JSONL output for every permutation, charts generated, no manual steps | All scenarios |
| 11 | Regression baseline established | Previous-run comparison available for all permutations | Section 10.4 |

---

## 12. References

### Existing Benchmark Infrastructure

- Treatment ABC pattern: `benchmarks/harness/treatments/base.py`
- Graph VT treatment: `benchmarks/harness/treatments/graph_vt.py`
- Graph generation: `benchmarks/harness/common.py` (`generate_erdos_renyi`, `generate_barabasi_albert`)
- JSONL I/O: `benchmarks/harness/common.py` (`write_jsonl`, `read_jsonl`)
- Chart specifications: `benchmarks/harness/analysis/charts_graph_vt.py`
- Registry: `benchmarks/harness/registry.py`
- CLI: `benchmarks/harness/cli.py`
- Makefile: `benchmarks/Makefile`

### Existing Benchmark Results

- Graph VT results (5 approaches x 4 workloads): `benchmarks/results/graph_vt_*.jsonl`
- Graph traversal results (muninn vs graphqlite): `benchmarks/results/graph_{muninn,graphqlite}_*.jsonl`
- Centrality results: `benchmarks/results/centrality_muninn_*.jsonl`

### Algorithm Complexity References

- BFS/DFS: O(V+E) --- Cormen et al., CLRS 4th edition
- Dijkstra (binary heap): O((V+E) log V) --- Cormen et al., CLRS 4th edition
- Union-Find components: O(V+E * alpha(V)) --- Tarjan, 1975
- PageRank: O((V+E) * iterations) --- Page et al., 1999
- Brandes betweenness: O(VE) --- [Brandes, 2001](https://doi.org/10.1080/0022250X.2001.9990249)
- Closeness centrality: O(V^2) unweighted, O(V(V+E) log V) weighted --- [Freeman, 1978](https://doi.org/10.1016/0378-8733(78)90021-7)
- Leiden: O(VE * iterations) --- [Traag et al., 2019](https://www.nature.com/articles/s41598-019-41695-z)
- Dynamic Leiden warm-start: [arXiv:2405.11658](https://arxiv.org/html/2405.11658v1)

### Graph Generation Models

- Erdos-Renyi: [Erdos & Renyi, 1959](https://doi.org/10.5486/PMD.1959.6.3-4.12)
- Barabasi-Albert: [Barabasi & Albert, 1999](https://doi.org/10.1126/science.286.5439.509)
- Holme-Kim (power-law cluster): [Holme & Kim, 2002](https://doi.org/10.1103/PhysRevE.65.026107)

### PecanPy Benchmark Methodology

- [PecanPy: CSR vs non-CSR comparison (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/19/3377/6184859)

---

**Prev:** [Phase 5 — TVF/VT Integration](./05_tvf_vt_integration.md) | **Back to:** [Phase 0 — Gap Analysis](./00_gap_analysis.md)
