# Centrality and Community

Five TVFs for structural graph analysis on any existing edge table: four centrality measures (`graph_degree`, `graph_node_betweenness`, `graph_edge_betweenness`, `graph_closeness`) and Leiden community detection (`graph_leiden`). All support weighted, directed, and temporally filtered inputs through the [shared constraint syntax](api.md#graph-tvf-constraint-syntax).

## When to use what

| Question | TVF |
|----------|-----|
| Which nodes have the most connections? | [`graph_degree`](api.md#graph_degree) |
| Which **nodes** bridge separate clusters? | [`graph_node_betweenness`](api.md#graph_node_betweenness) |
| Which **edges** hold the graph together? | [`graph_edge_betweenness`](api.md#graph_edge_betweenness) |
| Which nodes can reach everyone fastest? | [`graph_closeness`](api.md#graph_closeness) |
| What clusters exist? | [`graph_leiden`](api.md#graph_leiden) |
| What's the top node in each cluster? | [Leiden + betweenness, joined](#combining-centrality-with-communities) |

## Setup — a two-cluster graph

Every example below uses this tiny graph — two triangles joined by a single bridge edge (`dave → eve`):

```sql
.load ./muninn

CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0);

INSERT INTO edges VALUES
  ('alice', 'bob',   1.0), ('alice', 'carol', 1.0), ('bob',   'carol', 1.0),
  ('bob',   'dave',  1.0), ('carol', 'dave',  1.0),
  ('dave',  'eve',   1.0),   -- bridge
  ('eve',   'frank', 1.0), ('eve',   'grace', 1.0), ('frank', 'grace', 1.0);
```

---

## Degree centrality

Cheapest centrality — in/out/total edge counts per node. Run this first as a sanity check before spending time on betweenness or closeness.

```sql
SELECT node, in_degree, out_degree, degree
  FROM graph_degree
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both';
```

```text
node    in_degree  out_degree  degree
------  ---------  ----------  ------
alice   2.0        2.0         4.0
bob     3.0        3.0         6.0
carol   3.0        3.0         6.0
dave    3.0        3.0         6.0
eve     3.0        3.0         6.0
frank   2.0        2.0         4.0
grace   2.0        2.0         4.0
```

**Weighted**: pass `weight_col = 'weight'` and degrees become sums of edge weights.

**Normalized** (values in `[0, 1]`, scaled by `N - 1`):

```sql
SELECT node, round(centrality, 3) AS c FROM graph_degree
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND normalized = 1
  ORDER BY c DESC;
```

---

## Node betweenness

Brandes' O(VE) algorithm. Identifies **bridge nodes** — those sitting on many shortest paths. In the example graph, `dave` and `eve` are the two endpoints of the only bridge, so they dominate:

```sql
SELECT node, round(centrality, 3) AS c FROM graph_node_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND normalized = 1
  ORDER BY c DESC;
```

```text
node    c
------  -----
dave    0.600
eve     0.600
bob     0.133
carol   0.133
frank   0.133
grace   0.133
alice   0.000
```

### Performance on large graphs

Exact betweenness is O(VE) — slow on anything over ~50k nodes. The `auto_approx_threshold` constraint switches to source-sampling when the graph exceeds the threshold:

```sql
-- Approximate on graphs larger than 10k nodes (samples ceil(√N) sources)
SELECT node, centrality FROM graph_node_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND auto_approx_threshold = 10000
  ORDER BY centrality DESC LIMIT 20;
```

Default threshold is 50,000. Set it lower on graphs you know are large.

!!! tip "GraphRAG signal"
    Bridge nodes are the most valuable retrieval context in knowledge graphs — they connect otherwise disjoint topic clusters. Betweenness is typically worth caching as a regular table and recomputing on a schedule rather than on every query.

---

## Edge betweenness

Same algorithm, per-edge output. Returns one row per edge present in the graph.

```sql
SELECT src, dst, round(centrality, 3) AS c FROM graph_edge_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
  ORDER BY c DESC LIMIT 3;
```

```text
src     dst     c
------  ------  ------
dave    eve     12.000
bob     dave    5.000
carol   dave    5.000
```

The `dave → eve` edge has by far the highest score — removing it would split the graph. This is the signal used by the **Girvan-Newman** hierarchical clustering algorithm and by muninn's [entity resolution](entity-resolution.md) cascade (for bridge-edge removal before Leiden).

---

## Closeness centrality

Inverse of total shortest-path distance — high closeness means a node can reach every other node in few hops. muninn uses **Wasserman-Faust** normalization by default, which handles disconnected graphs gracefully: if a node can only reach R of the N−1 other nodes, its score is scaled by `(R / (N−1))²`.

```sql
SELECT node, round(centrality, 3) AS c FROM graph_closeness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
  ORDER BY c DESC;
```

`normalized = 1` is the default (unlike degree and betweenness). Pass `normalized = 0` for unscaled scores.

---

## Direction modes

All centrality TVFs accept `direction`:

| Value | Behavior |
|-------|----------|
| `'forward'` | Follow edges src → dst only |
| `'reverse'` | Follow edges dst → src only |
| `'both'` (default for most) | Treat edges as undirected |

For undirected graphs stored with one row per edge, use `'both'`. For DAGs or directed graphs where edge direction carries meaning, pick `'forward'` or `'reverse'` deliberately.

---

## Temporal filtering

All centrality TVFs accept optional `timestamp_col`, `time_start`, `time_end`. When all three are supplied, the TVF loads only edges whose timestamp falls within the window — useful for time-sliced social or activity graphs.

```sql
CREATE TABLE events (src TEXT, dst TEXT, ts TEXT);
INSERT INTO events VALUES
  ('alice', 'bob',   '2026-01-15T10:00:00'),
  ('bob',   'carol', '2026-02-01T14:30:00'),
  ('carol', 'dave',  '2026-03-03T09:15:00');

-- Betweenness over January-only edges
SELECT node, centrality FROM graph_node_betweenness
  WHERE edge_table = 'events' AND src_col = 'src' AND dst_col = 'dst'
    AND timestamp_col = 'ts'
    AND time_start = '2026-01-01T00:00:00'
    AND time_end   = '2026-01-31T23:59:59'
    AND direction = 'both';
```

Timestamps are compared as strings (ISO 8601 is the safe choice).

---

## Leiden community detection

The Leiden algorithm (Traag, Waltman & van Eck, 2019) partitions a graph into communities by maximizing modularity, with a guarantee that every community is internally well-connected — unlike Louvain, which can produce phantom communities that split apart on inspection.

```sql
SELECT node, community_id, round(modularity, 3) AS mod
  FROM graph_leiden
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both';
```

```text
node    community_id  mod
------  ------------  -----
alice   0             0.432
bob     0             0.432
carol   0             0.432
dave    0             0.432
eve     1             0.432
frank   1             0.432
grace   1             0.432
```

### Resolution parameter

`resolution` controls the granularity of the partition:

| Resolution | Effect |
|-----------|--------|
| `< 1.0` | Fewer, larger communities |
| `1.0` (default) | Standard modularity |
| `> 1.0` | More, smaller communities |

```sql
-- Finer partitioning
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND resolution = 2.5;
```

Sweep a few values (0.5, 1.0, 2.0) and pick the partition that matches your domain understanding — modularity alone is not sufficient for choosing a resolution.

### Weighted communities

```sql
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND weight_col = 'weight';
```

Strong edges are more likely to keep endpoints in the same community.

!!! tip "Microsoft GraphRAG pattern"
    Microsoft's [GraphRAG](https://microsoft.github.io/graphrag/) uses Leiden for hierarchical retrieval: detect communities, compute a summary embedding per community (mean of member vectors, or an LLM-generated label), search supernodes first, drill into the matching community. muninn supplies the building blocks — see [`muninn_label_groups`](api.md#muninn_label_groups) for the labeling step.

---

## Combining centrality with communities

A common pattern: detect communities, then pick the most important node inside each.

```sql
WITH node_comm AS (
  SELECT node, community_id FROM graph_leiden
    WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
      AND direction = 'both'
),
node_cent AS (
  SELECT node, centrality FROM graph_node_betweenness
    WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
      AND direction = 'both' AND normalized = 1
)
SELECT nc.community_id, nc.node, round(cent.centrality, 3) AS c
  FROM node_comm nc
  JOIN node_cent cent USING (node)
  ORDER BY nc.community_id, c DESC;
```

`dave` and `eve` will emerge as representatives of their respective communities — both sit on the bridge, which gives them maximum betweenness among their neighbors. These "community representatives" are a good context unit for LLM summarization ([`muninn_summarize`](api.md#muninn_summarize)) or prompt-grounding.

## Where to go next

- [Entity Resolution](entity-resolution.md) — uses `graph_edge_betweenness` + `graph_leiden` as part of a deduplication cascade
- [Node2Vec](node2vec.md) — learn structural embeddings that *encode* community and centrality signal
- [GraphRAG Cookbook](graphrag-cookbook.md) — full retrieval pipeline built on these primitives
- [API Reference — Centrality](api.md#centrality) — every constraint and default

## References

- Brandes, U. (2001). [A Faster Algorithm for Betweenness Centrality](https://doi.org/10.1080/0022250X.2001.9990249). *Journal of Mathematical Sociology*, 25(2), 163–177.
- Wasserman, S. & Faust, K. (1994). *Social Network Analysis: Methods and Applications*. Cambridge University Press.
- Traag, V. A., Waltman, L. & van Eck, N. J. (2019). [From Louvain to Leiden: guaranteeing well-connected communities](https://arxiv.org/abs/1810.08473). *Scientific Reports*, 9(1), 5233.
