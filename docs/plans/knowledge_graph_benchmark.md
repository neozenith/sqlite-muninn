# Knowledge Graph Benchmark — Gap Analysis

A gap analysis of remaining work to complete the **combined vector search + graph traversal** benchmark (the GraphRAG pattern), exercising all five of muninn's subsystems together.

**Status:** Gap analysis. Updated 2026-02-12. All C prerequisites are complete. Remaining work is benchmarks, documentation, and the KG pipeline.

**Previous status:** This document was originally a planning document. The graph algorithm prerequisites (centrality, community detection) have been fully implemented since the plan was written. This revision converts it to a gap analysis tracking what remains.

---

## Table of Contents

1. [Implementation Status Summary](#implementation-status-summary)
2. [Gap 1: Benchmark Suite — Missing Operations](#gap-1-benchmark-suite--missing-operations)
3. [Gap 2: Knowledge Graph Pipeline — Not Started](#gap-2-knowledge-graph-pipeline--not-started)
4. [Gap 3: Documentation — Not Updated for New Subsystems](#gap-3-documentation--not-updated-for-new-subsystems)
5. [Gap 4: Comparison Baselines — Incomplete](#gap-4-comparison-baselines--incomplete)
6. [Resolved Prerequisites](#resolved-prerequisites)
7. [Remaining Open Questions](#remaining-open-questions)
8. [Original Plan Reference](#original-plan-reference)

---

## Implementation Status Summary

### C Extension — COMPLETE

All five subsystems are fully implemented with tests:

| Subsystem | TVF/Function | C Source | Python Tests | C Tests | Status |
|-----------|-------------|---------|-------------|---------|--------|
| HNSW vector index | `hnsw_index` | `hnsw_vtab.c`, `hnsw_algo.c` | `test_hnsw_vtab.py` | `test_hnsw_algo.c` | **Done** |
| Graph traversal | `graph_bfs`, `graph_dfs`, `graph_shortest_path`, `graph_components`, `graph_pagerank` | `graph_tvf.c` | `test_graph_tvf.py` | — | **Done** |
| Centrality | `graph_degree`, `graph_betweenness`, `graph_closeness` | `graph_centrality.c` | `test_graph_centrality.py` | — | **Done** |
| Community detection | `graph_leiden` | `graph_community.c` | `test_graph_community.py` | — | **Done** |
| Node2Vec | `node2vec_train()` | `node2vec.c` | `test_node2vec.py` | — | **Done** |

**Note:** The original plan called for Louvain first, then Leiden as an upgrade. Implementation went directly to Leiden (superior algorithm with connectivity guarantees), which is what Microsoft GraphRAG uses. The `graph_louvain` TVF name referenced in the original plan should be read as `graph_leiden`.

### Primitive Benchmarks — MOSTLY COMPLETE

| Benchmark | Script | Results | Charts | Status |
|-----------|--------|---------|--------|--------|
| VSS search (muninn vs sqlite-vector vs sqlite-vec vs vectorlite) | `benchmark_vss.py` | 23 JSONL files | 30+ charts | **Done** |
| Graph traversal (BFS, DFS, SP, components, PageRank) | `benchmark_graph.py` | 44 JSONL files | 30+ charts | **Done** |
| Centrality timing (degree, betweenness, closeness) | — | — | — | **Missing** |
| Community detection timing (Leiden) | — | — | — | **Missing** |
| Node2Vec training time + quality | — | — | — | **Missing** |
| FTS5/BM25 keyword search | — | — | — | **Missing** |

### Knowledge Graph Pipeline — NOT STARTED

| Component | Script | Status |
|-----------|--------|--------|
| Entity extraction (GLiNER + SVO + FTS5) | `kg_extract.py` | **Not started** |
| Entity coalescing | `kg_coalesce.py` | **Not started** |
| Gutenberg catalog + download | `kg_gutenberg.py` | **Not started** |
| KG benchmark runner | `benchmark_kg.py` | **Not started** |
| KG benchmark analysis | `benchmark_kg_analyze.py` | **Not started** |
| Ground truth queries | — | **Not started** |

### Documentation — STALE

| Document | Status | Issue |
|----------|--------|-------|
| `docs/plans/graph_algorithms.md` | **Stale** | Says "Plan only. Not implemented." — but all algorithms are implemented |
| `docs/index.md` | **Stale** | Missing centrality, community detection, Leiden from features list |
| `mkdocs.yml` nav | **Incomplete** | No pages for centrality, community, Node2Vec, API reference, or examples |
| `docs/benchmarks.md` | **Stale** | No mention of centrality/community/Node2Vec benchmarks |
| `README.md` | **Partially stale** | Features section missing centrality + community; API section missing these TVFs |
| `CLAUDE.md` | **Stale** | "What We Already Have" table doesn't list centrality/community/Leiden |
| `skills/muninn/SKILL.md` | **OK** | Already mentions centrality + Leiden in description |

---

## Gap 1: Benchmark Suite — Missing Operations

### 1a. Centrality + Community Not in Graph Benchmarks

`benchmark_graph.py` defines `ALL_OPERATIONS = ["bfs", "dfs", "shortest_path", "components", "pagerank"]`. The three centrality TVFs and Leiden community detection are **not benchmarked**.

**Work needed:**
- Add `degree`, `betweenness`, `closeness`, `leiden` to `ALL_OPERATIONS` in `benchmark_graph.py`
- Add engine-level wrappers to call these TVFs (similar to existing `run_bfs()`, `run_pagerank()`, etc.)
- For Leiden: capture modularity score and community count in results
- For centrality: capture computation time at varying graph sizes
- Update `benchmark_graph_analyze.py` to produce charts for these operations
- Add corresponding Makefile targets or extend existing ones

**Estimated effort:** Medium — the framework exists, just needs new operations wired in.

### 1b. Node2Vec Not Benchmarked

Node2Vec is fully implemented and integration-tested (Karate Club graph in `test_node2vec.py`), but there are no benchmarks measuring:

- Training time vs graph size (N nodes, E edges)
- Embedding quality vs p,q hyperparameters
- Embedding dimensionality trade-offs (32 vs 64 vs 128 vs 384)
- Node2Vec embeddings for retrieval quality

**Work needed:**
- New script `benchmark_node2vec.py` or extend `benchmark_graph.py` with Node2Vec operations
- Sweep p,q values: `[0.25, 0.5, 1.0, 2.0, 4.0]` x `[0.25, 0.5, 1.0, 2.0, 4.0]`
- Measure: training time, embedding variance, cluster separation (silhouette score)

**Estimated effort:** Medium — standalone script, no complex dependencies.

### 1c. FTS5/BM25 Not Benchmarked

Despite extensive planning docs discussing FTS5 for concept discovery and hybrid retrieval (BM25 + VSS fusion via Reciprocal Rank Fusion), there is zero FTS5 usage in any benchmark.

**Work needed:**
- This is a KG benchmark concern (Gap 2), not a primitive benchmark issue
- FTS5 is a built-in SQLite feature — no extension needed
- Benchmark BM25 entry point vs HNSW entry point for graph expansion

**Estimated effort:** Part of the KG benchmark (Gap 2).

### 1d. Recursive CTE Baseline Incomplete

`benchmark_graph.py` mentions "cte — Recursive CTEs in plain SQLite (baseline)" in docs/comments but the CTE engine is not fully wired into the benchmark suite (only muninn and graphqlite engines are active).

**Work needed:**
- Verify CTE baseline implementation status in `benchmark_graph.py`
- If incomplete, implement CTE versions of BFS, DFS, shortest_path, components
- CTE PageRank is impractical (no iteration control in pure SQL) — skip or note as limitation

**Estimated effort:** Small — CTEs are straightforward SQL.

---

## Gap 2: Knowledge Graph Pipeline — Not Started

This is the largest remaining gap. All C primitives exist, but the Python pipeline to construct a knowledge graph and run the GraphRAG-style benchmark is entirely unbuilt.

### 2a. Entity Extraction Pipeline

**Files needed:**
- `benchmarks/scripts/kg_extract.py` — Main extraction pipeline (GLiNER + SVO + FTS5)
- `benchmarks/scripts/kg_coalesce.py` — Entity resolution via HNSW-based blocking
- `benchmarks/scripts/kg_gutenberg.py` — Project Gutenberg catalog + download + caching

**Python dependencies needed:**
- `gliner` — Zero-shot NER
- `spacy` + `en_core_web_lg` (or `en_core_web_md`) — Dependency parsing + SVO
- `textacy` — SVO triple extraction helper
- `sentence-transformers` — Already in benchmark deps
- `requests` — Already available

**Output:** Cached SQLite databases in `benchmarks/kg/{gutenberg_id}.db`, one per book.

### 2b. KG Benchmark Runner

**Files needed:**
- `benchmarks/scripts/benchmark_kg.py` — Runs the 6-phase benchmark workflow:
  - Phase A: Build KG (load cached .db, insert into HNSW, Node2Vec)
  - Phase B: GraphRAG retrieval (VSS → BFS expansion → context assembly)
  - Phase C: Graph analytics (betweenness, PageRank, Leiden on the KG)
  - Phase D: Hierarchical retrieval (Leiden → supernodes → multi-level search)
  - Phase E: Node2Vec hyperparameter sweep
  - Phase F: Temporal KG queries (bi-temporal schema pattern)
- `benchmarks/scripts/benchmark_kg_analyze.py` — Aggregate JSONL results into charts

### 2c. Ground Truth Queries

**Needed:** 50-100 retrieval questions about Wealth of Nations content with:
- Human-annotated relevant passages per question
- Relevance judgments at passage and entity level
- Ground-truth bridge concepts (for betweenness centrality validation)

**Approach options:**
- LLM-generated questions with human validation
- Manual curation of ~50 seed questions
- Could use the Wealth of Nations' own table of contents as a question source

### 2d. Makefile Targets

**Needed in `benchmarks/Makefile`:**
```makefile
kg-extract:           ## Extract KG from Wealth of Nations (reference)
kg-extract-random:    ## Extract KG from a random Gutenberg economics book
kg-extract-book:      ## Extract KG from a specific book (BOOK_ID=...)
kg-coalesce:          ## Entity resolution + dedup (all cached KGs)
kg:                   ## Run KG benchmark on all cached KGs
kg-analyze:           ## Analyze KG results -> charts
```

---

## Gap 3: Documentation — Not Updated for New Subsystems

### 3a. README.md Gaps

The README is partially stale:

| Section | Issue |
|---------|-------|
| **Features** list | Missing "Centrality Measures" and "Community Detection" as separate bullet points |
| **Mermaid diagram** | Only shows HNSW, Graph TVFs, Node2Vec — no centrality or community box |
| **Quick Start SQL** | No examples of `graph_degree`, `graph_betweenness`, `graph_closeness`, `graph_leiden` |
| **API Reference** | Missing `graph_degree`, `graph_betweenness`, `graph_closeness`, `graph_leiden` sections |
| **Examples table** | No example for centrality or community detection use cases |
| **Research References** | Missing Leiden (Traag 2019), Brandes (2001) betweenness |

### 3b. mkdocs Site Gaps

The documentation site (`mkdocs.yml` nav) currently has only 4 pages:
1. Home (`index.md`)
2. Benchmarks overview (`benchmarks.md`)
3. VSS benchmarks (`benchmarks/vss.md`)
4. Graph benchmarks (`benchmarks/graph.md`)

**Missing pages:**

| Page | Content | Priority |
|------|---------|----------|
| **API Reference** | Complete TVF reference for all 11 functions | **P1** |
| **Getting Started / Installation** | Python, Node.js, C installation guides | **P1** |
| **Examples** | Port the 5 examples from `examples/` into docs | P2 |
| **Centrality & Community** | Dedicated page explaining centrality + Leiden with use cases | P2 |
| **Node2Vec Guide** | How to use Node2Vec, p/q tuning, integration with HNSW | P2 |
| **GraphRAG Cookbook** | End-to-end tutorial: VSS → graph expansion → centrality ranking | P2 |
| **Benchmarks (Centrality)** | Centrality benchmark results (once Gap 1a is resolved) | P3 |
| **Benchmarks (KG)** | Knowledge graph benchmark results (once Gap 2 is resolved) | P3 |
| **Competitive Landscape** | Summary of `docs/plans/competitive_landscape.md` for public docs | P3 |

### 3c. docs/index.md Gaps

The MkDocs landing page is minimal — it's a subset of the README. It lists only 5 features and a basic Quick Start. Missing:
- Centrality + community detection features
- Link to API reference
- Installation instructions
- Links to examples

### 3d. Plan Documents — Stale Status Lines

| Document | Current Status Line | Should Be |
|----------|-------------------|-----------|
| `docs/plans/graph_algorithms.md` | "Plan only. Not implemented." | "**Completed.** All algorithms implemented (2026-02-xx)." |
| `docs/plans/knowledge_graph_benchmark.md` | N/A (this document) | Updated to gap analysis (this revision) |

### 3e. CLAUDE.md Gaps

The `CLAUDE.md` "What We Already Have" table (referenced from `knowledge_graph_benchmark.md` section) and the architecture section need updating:

| Section | Issue |
|---------|-------|
| **Architecture diagram** | Shows 3 subsystems, should show 5 (add centrality + community) |
| **Module Layering** | Already correct — lists all modules including `graph_centrality.c` and `graph_community.c` |

### 3f. Skills Gaps

`skills/muninn/SKILL.md` is already well-maintained — it mentions degree/betweenness/closeness centrality and Leiden. However:

| Issue | Detail |
|-------|--------|
| **SKILL.md description** | Says "Three subsystems" — should say "Five subsystems" |
| **Available TVFs table** | Correctly lists all 9 TVFs |
| **Cookbooks** | `cookbook-sql.md` and `cookbook-python.md` may not cover centrality/community examples |

---

## Gap 4: Comparison Baselines — Incomplete

From the competitive landscape analysis (`docs/plans/competitive_landscape.md`), several competitive comparison benchmarks are planned but not implemented:

| Comparison | Status | Notes |
|-----------|--------|-------|
| muninn HNSW vs sqlite-vector | **Done** | Primary VSS benchmark |
| muninn HNSW vs sqlite-vec | **Partial** | Framework supports it; verify results exist |
| muninn HNSW vs vectorlite | **Partial** | Framework supports it; verify results exist |
| muninn graph TVFs vs GraphQLite | **Partial** | Depends on `HAS_GRAPHQLITE` flag; may have gaps |
| muninn graph TVFs vs recursive CTEs | **Incomplete** | Engine defined but not fully wired (Gap 1d) |
| muninn integrated vs sqlite-vec + GraphQLite (separate tools) | **Not started** | Key competitive benchmark for the "single extension" value prop |
| VSS-only vs VSS+Graph (GraphRAG value) | **Not started** | Part of KG benchmark (Gap 2) |
| BM25+Graph vs VSS+Graph | **Not started** | Part of KG benchmark (Gap 2) |

---

## Resolved Prerequisites

These items from the original plan's "Prerequisites" section are now complete:

| Prerequisite | Original Status | Current Status |
|-------------|----------------|----------------|
| Primitive VSS benchmarks | Required | **Done** — 23 JSONL result files, 30+ charts |
| Primitive graph benchmarks | Required | **Done** — 44 JSONL result files, 30+ charts |
| Betweenness centrality TVF | P1 | **Done** — `graph_betweenness` in `graph_centrality.c` |
| Louvain/Leiden community detection | P2/P3 | **Done** — `graph_leiden` in `graph_community.c` (skipped Louvain, went directly to Leiden) |
| Degree centrality TVF | P4 | **Done** — `graph_degree` in `graph_centrality.c` |
| Closeness centrality TVF | P4 | **Done** — `graph_closeness` in `graph_centrality.c` |
| Python + Node.js wrappers | Not in original plan | **Done** — `sqlite_muninn/` and `index.mjs`/`index.cjs` |
| CI/CD pipeline | Not in original plan | **Done** — Multi-platform builds, tests, docs deployment |
| Examples | Not in original plan | **Done** — 5 examples covering all subsystems |

---

## Remaining Open Questions

These questions from the original plan are **still open** (not resolved by implementation):

1. **Entity extraction quality**: How reliable is GLiNER zero-shot NER on 18th-century economic text? The manual seed list of ~50 key entities remains essential.

2. **Ground truth annotation**: Who annotates the 50-100 retrieval questions? Could use LLM-generated questions with human validation.

3. **Node2Vec hyperparameters**: What (p, q) values best capture the KG's conceptual structure? The benchmark should sweep these.

4. ~~**Louvain vs Leiden**~~ **Resolved**: Went directly to Leiden. No Louvain needed.

5. **Betweenness centrality scalability**: Brandes' O(VE) is fine for ~3K nodes / ~10K edges. At 100K+ nodes, approximate betweenness (random sampling) would be needed.

6. **FTS5 + HNSW fusion**: What's the best way to combine BM25 scores and vector distances? Reciprocal Rank Fusion is the starting point.

7. **REBEL vs SVO extraction**: Typed relations (REBEL, 1.6 GB) vs noisy SVO extraction (spaCy, 40 MB). Unclear if typed relations improve retrieval enough.

8. **Comparison fairness**: How to fairly compare "muninn integrated" vs "separate tools" (sqlite-vec + GraphQLite)?

9. **Entity coalescing threshold**: Cosine similarity threshold for entity merging. Too low → over-merging, too high → fragmentation.

10. **Graph density impact**: Sparse (high-confidence edges only) vs dense (including co-occurrence edges) — different retrieval characteristics.

11. **Random book text quality**: NER/SVO pipeline performance on texts from different eras (1776-1920s).

12. **Cross-book concept alignment**: Can Node2Vec find structurally equivalent concepts across different books?

13. **Book length normalization**: Small books (Communist Manifesto, ~30 pages) vs large (WoN, ~400 pages).

14. **Temporal edge simulation**: Static text doesn't have temporal data — simulate via progressive agent sessions?

15. **Gutendex API reliability**: Third-party API; fallback to offline CSV catalog?

---

## Original Plan Reference

The original research, entity extraction approaches, HuggingFace model recommendations, dataset descriptions, benchmark workflow (Phases A-F), metrics definitions, and key references have been preserved below for reference. These sections are unchanged from the original plan.

<details>
<summary><strong>Click to expand: Original Research & Plan Details</strong></summary>

### Vision

Current benchmarks test each subsystem in isolation:

- **Vector search benchmarks** compare HNSW vs brute-force at varying N and dimension
- **Graph traversal benchmarks** compare TVFs vs CTEs vs GraphQLite on synthetic topologies

The knowledge graph benchmark tests the *composition* — the workflow where a vector similarity search provides an entry point into a graph, and graph traversal expands the context. This is the core pattern behind GraphRAG, where:

1. **VSS entry point**: Query vector -> HNSW search -> find nearest graph node
2. **Graph expansion**: From that node -> BFS/DFS to explore k-hop neighbors
3. **Context assembly**: Collect text from traversed nodes as retrieval context
4. **Graph analytics**: Betweenness centrality identifies bridge concepts; PageRank finds authoritative nodes
5. **Hierarchical retrieval** (advanced): Leiden communities -> supernode embeddings -> multi-level search

muninn is uniquely positioned here — it's the only SQLite extension combining HNSW + graph TVFs + centrality + community detection + Node2Vec in a single shared library. The benchmark demonstrates whether this integration provides measurable advantages over stitching separate tools together.

### Research: State of the Art

#### Vector-Seeded Graph Traversal (The GraphRAG Pattern)

The core insight behind GraphRAG is that **vector similarity search alone misses relational context**. A query about "How does the division of labour affect wages?" might find passages about wages but miss causally-linked concepts like "productivity" or "market price" that are connected via graph edges but not semantically similar to the query.

The pattern works in stages:

```
Query "How does division of labour affect wages?"
  │
  ├─ [1] VSS Entry Point ──────── HNSW search → nearest passage nodes
  │                                (finds: "wages", "labour", "price")
  │
  ├─ [2] Graph Expansion ──────── BFS 2-hop from entry points
  │                                (discovers: "productivity", "market price",
  │                                 "rent", "profit" via CAUSES/COMPOSED_OF edges)
  │
  ├─ [3] Centrality Ranking ───── Betweenness centrality scores bridge nodes
  │                                (ranks "market price" highest — it bridges
  │                                 labour/wages cluster to rent/profit cluster)
  │
  └─ [4] Context Assembly ─────── Collect passage text from traversed+ranked nodes
                                   (richer context than VSS alone)
```

**Key research systems:**

- **Microsoft GraphRAG (2024)**: Extracts KG from documents via LLM, builds community hierarchy (Leiden algorithm), generates community summaries, retrieval via summary search → drill into communities. The gold standard for hierarchical GraphRAG.
- **HybridRAG (2024)**: Combines vector DB + graph DB with joint scoring (vector similarity + graph distance). Exactly what muninn enables in a single SQLite extension.
- **NaviX (VLDB 2025)**: Native dual indexing in the DB kernel — vector index + graph index with pruning strategies that leverage both simultaneously. The research frontier.
- **Deep GraphRAG (2025)**: Multi-hop reasoning via graph-guided evidence chains starting from VSS results.

#### Entity Coalescing and Synonym Merging

A critical challenge in KG construction: "division of labour", "division of labor", and "the labour is divided" all refer to the same concept. Without merging, the graph becomes fragmented.

**Coalescing Pipeline:**

```
Raw Entities  →  Blocking  →  Matching  →  Merging  →  Canonical Graph
(many dupes)    (group by    (pairwise    (resolve     (clean nodes)
                 similarity)  comparison)  clusters)
```

- **Stage 1 — Blocking**: HNSW-based embedding similarity (cosine < 0.3) or token overlap (>50%)
- **Stage 2 — Matching**: Cascade: exact match → fuzzy (Jaro-Winkler > 0.85) → embedding similarity → WordNet synsets
- **Stage 3 — Merging**: Leiden clustering on match graph to prevent over-merging, select canonical form

#### Temporal Knowledge Graphs

Bi-temporal model (valid time + transaction time) per edge. No C extension changes needed — pure schema pattern on edge tables plus application-level query construction. See original plan for full schema and SQL examples.

### Entity Extraction Approaches

1. **GLiNER zero-shot NER** — Custom entity types at inference time. Recommended: `urchade/gliner_small-v2.1` (350 MB).
2. **spaCy SVO extraction** — Dependency parse → (subject, verb, object) triples → graph edges.
3. **FTS5/BM25 concept discovery** — Zero-model approach using SQLite's built-in FTS5 to identify important terms.

**Hybrid Pipeline**: All three combined → union → coalesce → edge construction → output SQLite KG.

### HuggingFace Model Recommendations

| Tier | Entity Extraction | Relation Extraction | Node Embeddings | Total Size |
|------|------------------|-------------------|-----------------|------------|
| Tier 1 (Minimal) | GLiNER small | spaCy SVO | MiniLM-L6-v2 | < 500 MB |
| Tier 2 (Better) | GLiNER medium | REBEL | BGE-base-en-v1.5 | < 2.5 GB |
| Tier 3 (Fastest) | spaCy small | spaCy SVO | potion-base-8M | < 100 MB |

### Dataset: Economics Texts from Project Gutenberg

Primary: Wealth of Nations (Gutenberg #3300, ~2,500 passage chunks). Additional: random economics books from Gutendex API for generalization testing. See original plan for full catalog, download pipeline, and entity/relation type definitions.

### Benchmark Workflow (Phases A-F)

- **Phase A**: Build KG in muninn (HNSW + edges + Node2Vec + FTS5)
- **Phase B**: GraphRAG-style retrieval (VSS → BFS expansion → context assembly)
- **Phase C**: Graph analytics (betweenness, PageRank, Leiden on the KG)
- **Phase D**: Hierarchical retrieval (Leiden → supernodes → multi-level search)
- **Phase E**: Node2Vec hyperparameter sweep (p,q grid)
- **Phase F**: Temporal KG queries (bi-temporal schema pattern)

### Metrics

- **Retrieval Quality**: Context recall, precision, hop efficiency, coverage, bridge discovery
- **Graph Analytics**: Betweenness/PageRank/Leiden time, modularity, centrality correlation
- **End-to-End**: Total latency, VSS time, expansion time, Node2Vec training time, memory
- **Temporal**: Snapshot creation time, temporal vs current traversal ratio, storage overhead
- **Baselines**: VSS-only, Graph-only, BM25+Graph, BM25-only, Centrality-guided vs uniform, Separate tools

### Key References

- Microsoft GraphRAG (2024), Deep GraphRAG (2025), HybridRAG (2024), NaviX (VLDB 2025)
- Multi-Scale Node Embeddings (2024), Node2Vec (Grover & Leskovec 2016)
- Louvain (Blondel 2008), Leiden (Traag 2019), Brandes Betweenness (2001)
- GLiNER (NAACL 2024), REBEL (ACL 2021)
- Vesper-Memory (2025), Zep/Graphiti (arXiv:2501.13956)

### Implementation Sketch — File Structure

```
benchmarks/
  scripts/
    benchmark_kg.py            # KG benchmark runner (works on any economics text)
    benchmark_kg_analyze.py    # KG benchmark analysis + charts
    kg_extract.py              # Entity/relation extraction pipeline
    kg_coalesce.py             # Entity resolution + synonym merging
    kg_gutenberg.py            # Project Gutenberg catalog + download + caching
  kg/                          # Cached knowledge graphs (one SQLite DB per book)
  texts/                       # Cached plain text downloads
  results/kg_*.jsonl           # KG benchmark results (includes gutenberg_id)
  vectors/kg_*.npy             # Pre-computed entity embeddings per book
```

</details>
