# New Graph Algorithms Plan

New graph algorithm TVFs to add to the muninn extension: community detection (Louvain/Leiden), centrality measures (betweenness, closeness, degree), and supporting research.

**Status:** Plan only. Not implemented. These are prerequisites for the [Knowledge Graph Benchmark](knowledge_graph_benchmark.md).

---

## Table of Contents

1. [Overview](#overview)
2. [What We Already Have](#what-we-already-have)
3. [Community Detection: Louvain and Leiden](#community-detection-louvain-and-leiden)
4. [Graph Centrality Measures](#graph-centrality-measures)
5. [Node2Vec Deep Dive and Alternatives](#node2vec-deep-dive-and-alternatives)
6. [Knowledge Graph Embeddings (KGE)](#knowledge-graph-embeddings-kge)
7. [Implementation Plan](#implementation-plan)
8. [Key References](#key-references)
9. [Open Questions](#open-questions)

---

## Overview

The muninn extension currently implements BFS, DFS, shortest path, connected components, PageRank, and Node2Vec. The next tier of graph algorithms — community detection and centrality measures — are needed for:

- **GraphRAG workflows**: Centrality-guided retrieval and hierarchical community-based search (see [Knowledge Graph Benchmark](knowledge_graph_benchmark.md))
- **Standalone graph analytics**: These algorithms are independently useful for any graph stored in SQLite
- **Competitive parity**: GraphQLite (Rust, alpha) already offers Louvain and betweenness centrality. Adding them keeps muninn's feature set comprehensive.

### What We Already Have

| Capability | TVF/Function | Status |
|-----------|-------------|--------|
| BFS traversal | `graph_bfs` | Done |
| DFS traversal | `graph_dfs` | Done |
| Shortest path (Dijkstra) | `graph_shortest_path` | Done |
| Connected components | `graph_components` (Union-Find) | Done |
| PageRank | `graph_pagerank` (power method, configurable damping/iterations) | Done |
| Node2Vec embeddings | `node2vec_train()` (biased walks + SGNS) | Done |

### What's Missing

| Capability | Proposed TVF | Priority | Est. Lines of C |
|-----------|-------------|----------|-----------------|
| Community detection | `graph_louvain` | **P1** | ~300 |
| Leiden refinement | `graph_leiden` | P2 | ~100 additional |
| Betweenness centrality | `graph_betweenness` | **P1** | ~200 |
| Closeness centrality | `graph_closeness` | P3 | ~80 |
| Degree centrality | `graph_degree` | P3 | ~30 |

---

## Community Detection: Louvain and Leiden

### What is Louvain?

The **Louvain algorithm** (Blondel et al., 2008) detects communities in large networks by optimizing **modularity** — a measure of how densely connected nodes within a community are compared to random connections.

**Algorithm (two phases, iterated):**

1. **Local moving**: Each node is moved to the neighboring community that maximizes modularity gain. Repeat until no improvement.
2. **Aggregation**: Collapse each community into a single supernode. Build a new graph of supernodes. Return to phase 1.

**Modularity** Q measures the fraction of edges within communities minus the expected fraction if edges were random:

```
Q = (1/2m) Σ [A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)
```

where `A_ij` is the adjacency matrix, `k_i` is the degree of node i, `m` is total edges, and δ is 1 when nodes i,j are in the same community.

**Complexity**: O(n log n) — fast enough for million-node graphs.

**Problem**: Louvain can produce **badly connected or even disconnected communities**. In experiments, up to 25% of communities are badly connected and up to 16% are disconnected (Traag et al., 2019). This happens because the greedy local moving phase can trap nodes in communities they shouldn't belong to.

### What is Leiden?

The **Leiden algorithm** (Traag, Waltman & van Eck, 2019) fixes Louvain's connectivity guarantees by adding a **refinement phase**:

1. **Local moving** (same as Louvain)
2. **Refinement**: Communities may be split to guarantee all communities are **well-connected**. Only nodes well-connected within their community are retained.
3. **Aggregation** (based on the *refined* partition, not the local-moving partition)

**Key improvements:**
- **Guaranteed connectivity**: All identified communities are both well-separated and well-connected
- **Faster convergence**: Actually runs faster than Louvain in practice
- **Better partitions**: Higher modularity scores at convergence
- **Subset optimality**: When converged, all vertices are optimally assigned

### Difference from Connected Components

Connected components (what we have now via `graph_components`) only finds **disconnected subgraphs** — nodes with no path between them. Louvain/Leiden finds **densely-connected clusters within a single connected component**. In most real-world graphs, there's one giant connected component, so `graph_components` returns a single cluster for nearly all nodes. Community detection is the tool that finds meaningful structure within that giant component.

### Proposed TVF Interface

```sql
-- Louvain community detection
SELECT node, community_id, modularity
FROM graph_louvain
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND weight_col = 'weight'     -- optional; unweighted if omitted
  AND resolution = 1.0;         -- optional; higher = smaller communities

-- Leiden (upgrade path)
SELECT node, community_id, modularity
FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

### Implementation Sketch (~300 lines of C for Louvain)

1. **Initialize**: Each node in its own community. Build adjacency list + degree array.
2. **Local moving phase**:
   - For each node, compute modularity gain `ΔQ` for moving to each neighbor's community
   - `ΔQ = [Σ_in + k_i_in] / 2m - [(Σ_tot + k_i) / 2m]² - [Σ_in / 2m - (Σ_tot / 2m)² - (k_i / 2m)²]`
   - Move node to community with maximum positive ΔQ
   - Repeat until no moves improve modularity
3. **Aggregation phase**: Collapse communities into supernodes, aggregate edge weights. Return to step 2.
4. **Termination**: When no aggregation produces improvement.

**Data structures needed:**
- Community assignment array: `int64_t community[N]`
- Community internal weight: `double sigma_in[N]`
- Community total degree: `double sigma_tot[N]`
- Adjacency list: reuse existing graph loading from `graph_tvf.c`

**Upgrade path to Leiden**: Add a refinement step between local moving and aggregation (~100 more lines). Communities are checked for internal connectivity and split if needed.

### Use Cases Beyond GraphRAG

- **Code architecture analysis**: Detect module clusters in call graphs
- **Infrastructure mapping**: Find clusters of tightly-coupled services
- **Session log analysis**: Group related agent interactions into coherent episodes

---

## Graph Centrality Measures

### Betweenness Centrality

**What it is**: Measures how often a node lies on the shortest path between other node pairs. Nodes with high betweenness are "bridge" concepts connecting otherwise separate clusters.

```
BC(v) = Σ_{s≠v≠t} [σ_st(v) / σ_st]
```

where σ_st is the total number of shortest paths from s to t, and σ_st(v) is the number of those paths passing through v.

**Why it matters**: Bridge nodes are structurally important — they connect communities that would otherwise be isolated. In knowledge graphs, these are the concepts needed for cross-topic retrieval. In infrastructure graphs, these are single points of failure.

**Algorithm**: Brandes' algorithm computes betweenness for all nodes in O(VE) time (unweighted) or O(VE + V² log V) (weighted).

### Proposed TVF Interface

```sql
SELECT node, centrality
FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND weight_col = 'weight'     -- optional
  AND normalized = 1;           -- optional; normalize by (N-1)(N-2)
```

### Implementation Sketch (~200 lines of C)

Brandes' algorithm:

1. For each source node s:
   a. Run BFS (unweighted) or Dijkstra (weighted) from s
   b. Track: shortest path count `σ[t]`, predecessor list `P[t]`, distance `d[t]`
2. Backward accumulation (from farthest nodes back to s):
   a. For each node w in reverse BFS order: `δ[v] += (σ[v]/σ[w]) × (1 + δ[w])` for each predecessor v of w
   b. `BC[w] += δ[w]` (except for source s)
3. Normalize: divide by `(N-1)(N-2)` for undirected or `(N-1)(N-2)/2` for directed

**Scalability**: O(VE) means ~3K nodes / ~10K edges runs in milliseconds. For >100K nodes, approximate betweenness (sample k source nodes instead of all) brings it to O(kE).

### Other Centrality Measures

| Measure | Formula | Meaning | Complexity | Implementation |
|---------|---------|---------|-----------|---------------|
| **Degree centrality** | `deg(v) / (N-1)` | How many direct connections | O(V+E) | Trivial — count edges per node |
| **Closeness centrality** | `(N-1) / Σ d(v,u)` | Average distance to all other nodes | O(V(V+E)) | BFS from each node |
| **Eigenvector centrality** | `x_v ∝ Σ A_{vu} × x_u` | Connected to other well-connected nodes | O(V+E) per iteration | Power iteration (like PageRank but undirected) |
| **PageRank** | Power iteration with damping | Random walk stationary distribution | O(V+E) per iteration | **Already implemented** |

### Proposed Combined TVF (Nice-to-Have)

```sql
-- All centrality measures in one pass
SELECT node, degree, closeness, betweenness
FROM graph_centrality
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

This would compute degree (trivial), closeness, and betweenness in a single graph load. The BFS traversals needed for closeness and betweenness overlap, so a combined pass saves I/O.

---

## Node2Vec Deep Dive and Alternatives

### How Node2Vec Works (Already in muninn)

Node2Vec (Grover & Leskovec, 2016) learns vector embeddings for graph nodes using a two-step process:

1. **Biased random walks**: Simulate walks on the graph where the bias is controlled by two parameters:
   - **p** (return parameter): Controls likelihood of returning to the previous node. Low p → stay local (BFS-like). High p → explore (DFS-like).
   - **q** (in-out parameter): Controls preference for inward vs outward nodes. Low q → explore outward (DFS-like). High q → stay close (BFS-like).

2. **Skip-gram with Negative Sampling (SGNS)**: Treat random walk sequences as "sentences" and node IDs as "words". Train a Word2Vec-style model to predict context nodes from target nodes.

**What the parameters capture:**
- **p=1, q=1**: Equivalent to DeepWalk (uniform random walks). Captures a balance of structural and community information.
- **Low p, high q**: BFS-like walks. Captures **homophily** — nodes in the same community get similar embeddings.
- **High p, low q**: DFS-like walks. Captures **structural equivalence** — nodes with similar structural roles (e.g., all "hub" nodes) get similar embeddings, even if they're in different communities.

**Our implementation** (`src/node2vec.c`):
- Full biased random walk with second-order Markov chain
- Skip-gram with negative sampling using frequency^0.75 distribution
- Sigmoid lookup table (1000 entries) for speed
- L2-normalized output embeddings (ready for cosine similarity in HNSW)
- Linear learning rate decay
- Supports undirected graphs, ~10K nodes max (linear node lookup)

### Alternatives to Node2Vec

| Method | Type | Advantages | Disadvantages | Best For |
|--------|------|-----------|---------------|----------|
| **DeepWalk** (2014) | Shallow / random walks | Simplest implementation, uniform walks | No bias control (p,q); less expressive | Simple graphs where you don't need homophily/structural tuning |
| **LINE** (2015) | Shallow / edge sampling | Explicitly optimizes 1st+2nd order proximity; scales to millions | Doesn't capture higher-order structure | Very large graphs; when memory is limited |
| **GraphSAGE** (2017) | Deep / GNN | **Inductive** — generalizes to unseen nodes; uses node features | Requires node feature vectors as input; heavier to train | Dynamic graphs with new nodes; when node features exist |
| **GAT** (2018) | Deep / GNN | Attention mechanism highlights important neighbors; best quality | Heavy compute (GPU needed); not easily portable to C | When quality matters more than portability |
| **GCN** (2017) | Deep / GNN | Simple, strong baseline for node classification | Fixed aggregation (mean), not inductive | Static graphs with labels for supervised learning |
| **TransE** (2013) | KGE / translational | Specifically designed for (head, relation, tail) triples | Requires typed relations; can't model symmetric relations | Knowledge graphs with typed edges (not our case) |
| **RotatE** (2019) | KGE / rotational | Handles symmetry, antisymmetry, inversion, composition | More complex; still requires typed relations | Complex relation patterns in KGs |

**Why Node2Vec is the right choice for muninn:**

1. **Zero-dependency**: Runs in pure C with no ML framework. GraphSAGE/GAT require PyTorch/TensorFlow.
2. **No node features needed**: Node2Vec works from graph topology alone. GraphSAGE needs feature vectors.
3. **Tunable**: The p,q parameters let us control what the embeddings capture.
4. **Compatible with HNSW**: Output embeddings go directly into our HNSW index for similarity search.
5. **Fast for small graphs**: For ~3K nodes, Node2Vec trains in seconds. GNNs would be overkill.

**When to consider alternatives**: If muninn ever needs to support graphs >100K nodes or inductive settings (embedding new nodes without retraining), GraphSAGE would be worth investigating. For now, Node2Vec is ideal.

---

## Knowledge Graph Embeddings (KGE)

KGE methods differ from Node2Vec in that they explicitly model **typed relations** between entities:

- **TransE**: Models relations as translations: `head + relation ≈ tail` in vector space. "England" + "TRADES_WITH" ≈ "Holland". Simple but can't model symmetric relations (if A trades with B, B trades with A — but TransE would need two different vectors).
- **RotatE**: Models relations as rotations in complex vector space. Handles symmetry, antisymmetry, inversion, and composition patterns.
- **ComplEx**: Uses complex-valued embeddings with Hermitian dot product. Good for symmetric and antisymmetric relations.

**Relevance to our benchmark**: Knowledge graphs with typed relations (CAUSES, COMPOSED_OF, etc.) could benefit from KGE methods for relation-aware retrieval. However, they require a training pipeline that's harder to implement in pure C. The pragmatic approach: use Node2Vec for structure-only embeddings stored in HNSW, and handle relation types at the query level (e.g., filter BFS expansion by relation type).

---

## Implementation Plan

### Phase 1: Betweenness Centrality (Priority 1)

**TVF**: `graph_betweenness(edge_table, src_col, dst_col, weight_col, normalized)`

**Output**: `(node TEXT, centrality REAL)`

**Steps:**
1. Add `src/graph_centrality.c` with Brandes' algorithm
2. Register `graph_betweenness` TVF in `graph_tvf.c`
3. Add C unit tests in `test/test_graph_centrality.c`
4. Add Python integration tests in `pytests/`
5. Verify on known graphs with hand-computed betweenness values

### Phase 2: Louvain Community Detection (Priority 1)

**TVF**: `graph_louvain(edge_table, src_col, dst_col, weight_col, resolution)`

**Output**: `(node TEXT, community_id INTEGER, modularity REAL)`

**Steps:**
1. Add `src/graph_community.c` with Louvain algorithm
2. Register `graph_louvain` TVF in `graph_tvf.c`
3. Add C unit tests — verify on Zachary's Karate Club (known community structure)
4. Add Python integration tests
5. Verify modularity scores match reference implementations

### Phase 3: Leiden Upgrade (Priority 2)

**TVF**: `graph_leiden(edge_table, src_col, dst_col, weight_col, resolution)`

**Steps:**
1. Add refinement phase to `src/graph_community.c`
2. Register `graph_leiden` TVF
3. Test: verify all communities are connected (the guarantee Leiden provides over Louvain)

### Phase 4: Degree/Closeness Centrality (Priority 3)

**TVFs**: `graph_degree`, `graph_closeness`

**Steps:**
1. Add to `src/graph_centrality.c` alongside betweenness
2. Degree is trivial; closeness reuses BFS from betweenness

---

## Key References

### Algorithms

- **Blondel et al. (2008)** — Louvain: Fast modularity optimization. O(n log n).
- **Traag, Waltman & van Eck (2019)** — Leiden: Fixes Louvain's connectivity guarantees via refinement. Used by Microsoft GraphRAG.
- **Brandes (2001)** — Betweenness centrality: O(VE) algorithm for all-nodes betweenness.
- **Grover & Leskovec (2016)** — Node2Vec: Biased random walks + Skip-gram for graph embeddings. Already implemented.

### Systems

- **Neo4j Graph Data Science**: The reference implementation for graph algorithms at scale. Includes all algorithms listed here plus many more.
- **GraphQLite** (colliery-io): Rust-based SQLite graph extension with Cypher + Louvain + betweenness. Alpha-stage. The closest graph-side competitor.
- **NetworkX**: Python reference implementations of all these algorithms. Useful for validation.

---

## Open Questions

1. **Louvain vs Leiden as default**: Start with Louvain (simpler) and add Leiden later? Or implement Leiden directly since it's strictly better and only ~100 lines more?

2. **Resolution parameter**: Louvain/Leiden's `resolution` parameter controls community granularity. What default? Standard is 1.0, but for small graphs (~3K nodes) a lower value may produce more useful communities.

3. **Approximate betweenness**: Brandes' O(VE) is fine for ~10K nodes. At what graph size should we offer approximate betweenness (random source sampling)? Threshold: ~50K nodes?

4. **Combined centrality TVF**: Is a single `graph_centrality` TVF returning all measures preferable to separate TVFs? Pros: single graph load. Cons: can't skip expensive measures you don't need.

5. **Weighted vs unweighted**: All algorithms have weighted variants. Should weight support be mandatory from the start, or added as a follow-up?

6. **Directed graph handling**: Betweenness and community detection behave differently on directed vs undirected graphs. The existing TVFs use a `direction` parameter. Should the new TVFs follow the same pattern?
