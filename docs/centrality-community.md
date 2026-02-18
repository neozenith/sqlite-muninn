# Centrality & Community Detection

muninn provides four TVFs for analyzing graph structure: three centrality measures and Leiden community detection. All operate on **any** existing SQLite edge table — no special schema required.

## When to Use What

| Question | Use |
|----------|-----|
| Which nodes have the most connections? | `graph_degree` |
| Which nodes bridge separate clusters? | `graph_betweenness` |
| Which nodes can reach others most quickly? | `graph_closeness` |
| What clusters exist in the graph? | `graph_leiden` |

## Setup

All examples use this edge table:

```sql
.load ./muninn

CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0);

-- Two clusters connected by a bridge
INSERT INTO edges VALUES
    ('alice', 'bob', 1.0), ('alice', 'carol', 1.0), ('bob', 'carol', 1.0),
    ('bob', 'dave', 1.0), ('carol', 'dave', 1.0),
    ('dave', 'eve', 1.0),  -- bridge edge
    ('eve', 'frank', 1.0), ('eve', 'grace', 1.0), ('frank', 'grace', 1.0);
```

---

## Degree Centrality

Counts incoming, outgoing, and total edges per node. The simplest centrality measure — fast and useful as a first pass.

```sql
SELECT node, in_degree, out_degree, degree, centrality
FROM graph_degree
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

**With normalization** (divides by N-1, producing values in [0, 1]):

```sql
SELECT node, centrality FROM graph_degree
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND normalized = 1
ORDER BY centrality DESC;
```

**With weights** (sums edge weights instead of counting):

```sql
SELECT node, degree FROM graph_degree
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND weight_col = 'weight';
```

### Output Columns

| Column | Description |
|--------|-------------|
| `node` | Node identifier |
| `in_degree` | Count (or weighted sum) of incoming edges |
| `out_degree` | Count (or weighted sum) of outgoing edges |
| `degree` | Total degree (in + out) |
| `centrality` | Raw degree, or normalized if `normalized = 1` |

---

## Betweenness Centrality

Identifies **bridge nodes** that sit on many shortest paths between other nodes. Computed via [Brandes' algorithm](https://doi.org/10.1080/0022250X.2001.9990249) (Brandes, 2001) in O(VE) time.

```sql
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both'
ORDER BY centrality DESC;
```

In the example graph, `dave` has the highest betweenness because it's the sole bridge between the two clusters.

!!! tip "GraphRAG Application"
    In knowledge graphs, high-betweenness nodes connect otherwise separate topic clusters. These are the most valuable context nodes for retrieval — they provide cross-domain links that improve answer quality.

**Normalized** (values in [0, 1], scaled by `2/((N-1)(N-2))` for undirected):

```sql
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both' AND normalized = 1;
```

### Direction Modes

| `direction` | Behavior |
|-------------|----------|
| `'forward'` (default) | Follow edges src → dst only |
| `'reverse'` | Follow edges dst → src only |
| `'both'` | Treat edges as undirected |

---

## Closeness Centrality

Measures how quickly a node can reach all other nodes. Nodes with high closeness are "close to everything" in the graph.

```sql
SELECT node, centrality FROM graph_closeness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both'
ORDER BY centrality DESC;
```

Uses **[Wasserman-Faust](https://doi.org/10.1017/CBO9780511815478) normalization** (Wasserman & Faust, 1994) for disconnected graphs: if a node can only reach R out of N-1 other nodes, the score is weighted by `(R/(N-1))^2` to avoid inflating scores for small components.

---

## Temporal Filtering

All centrality TVFs support temporal filtering — compute centrality only on edges within a time window:

```sql
-- Add timestamps to edges
CREATE TABLE events (src TEXT, dst TEXT, ts TEXT);
INSERT INTO events VALUES ('alice', 'bob', '2026-01-15T10:00:00');
INSERT INTO events VALUES ('bob', 'carol', '2026-02-01T14:30:00');

-- Centrality only for January events
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'events' AND src_col = 'src' AND dst_col = 'dst'
  AND timestamp_col = 'ts'
  AND time_start = '2026-01-01T00:00:00'
  AND time_end = '2026-01-31T23:59:59'
  AND direction = 'both';
```

---

## Leiden Community Detection

The [Leiden algorithm](https://arxiv.org/abs/1810.08473) (Traag, Waltman & van Eck, 2019) partitions a graph into **well-connected communities**. It improves on Louvain by guaranteeing that communities are internally connected — no "phantom" communities that fall apart on inspection.

```sql
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

### Resolution Parameter

The `resolution` parameter controls community granularity:

| Resolution | Effect |
|-----------|--------|
| `< 1.0` | Fewer, larger communities |
| `1.0` (default) | Standard modularity optimization |
| `> 1.0` | More, smaller communities |

```sql
-- Fine-grained communities
SELECT node, community_id FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND resolution = 2.0;
```

### Output Columns

| Column | Description |
|--------|-------------|
| `node` | Node identifier |
| `community_id` | Community assignment (0-based, contiguous) |
| `modularity` | Global modularity score of the partition (same for all rows) |

### Weighted Graphs

Leiden respects edge weights — stronger connections are more likely to end up in the same community:

```sql
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND weight_col = 'weight';
```

!!! tip "Microsoft GraphRAG Pattern"
    Microsoft's GraphRAG uses Leiden for hierarchical retrieval: detect communities, compute a summary embedding for each community, then search community embeddings first before drilling into individual nodes. This provides efficient "global search" over large knowledge graphs.

---

## Combining Centrality + Community

A common pattern: detect communities, then find the most important nodes within each community.

```sql
-- Step 1: Get communities
CREATE TEMP TABLE node_communities AS
SELECT node, community_id FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';

-- Step 2: Get betweenness centrality
CREATE TEMP TABLE node_centrality AS
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both';

-- Step 3: Top node per community
SELECT nc.community_id, nc.node, ncent.centrality
FROM node_communities nc
JOIN node_centrality ncent ON nc.node = ncent.node
WHERE ncent.centrality = (
    SELECT MAX(ncent2.centrality)
    FROM node_communities nc2
    JOIN node_centrality ncent2 ON nc2.node = ncent2.node
    WHERE nc2.community_id = nc.community_id
)
ORDER BY nc.community_id;
```

This identifies the **representative node** for each community — useful for summarization, sampling, or hierarchical search.

## References

- Brandes, U. (2001). [A Faster Algorithm for Betweenness Centrality](https://doi.org/10.1080/0022250X.2001.9990249). *Journal of Mathematical Sociology*, 25(2), 163–177.
- Wasserman, S. & Faust, K. (1994). [Social Network Analysis: Methods and Applications](https://doi.org/10.1017/CBO9780511815478). *Cambridge University Press*.
- Traag, V. A., Waltman, L. & van Eck, N. J. (2019). [From Louvain to Leiden: guaranteeing well-connected communities](https://arxiv.org/abs/1810.08473). *Scientific Reports*, 9(1), 5233.

## Further Reading

- [API Reference](api.md) — Full parameter details for all TVFs
- [GraphRAG Cookbook](graphrag-cookbook.md) — End-to-end pipeline using centrality + community detection
- [Node2Vec Guide](node2vec.md) — Learn structural embeddings from graph topology
