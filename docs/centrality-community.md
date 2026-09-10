| How well separated is a grouping? | [`graph_conductance`](api.md#graph_conductance) |
# Centrality and Community

Six TVFs for structural graph analysis on any existing edge table: four centrality measures (`graph_degree`, `graph_node_betweenness`, `graph_edge_betweenness`, `graph_closeness`), Leiden community detection (`graph_leiden`), and conductance scoring of any grouping (`graph_conductance`). All support weighted, directed, and temporally filtered inputs through the [shared constraint syntax](api.md#graph-tvf-constraint-syntax).

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

## Conductance: scoring a partition

`graph_leiden` reports a global `modularity`, which is only defined for the partition Leiden itself found and is not comparable across partitions of different granularity. [`graph_conductance`](api.md#graph_conductance) scores **any** node-to-group membership, one row per group, on a fixed scale: `phi = 0` is a closed group, `phi = 1` is a group where every edge leaves. Every other graph TVF derives a labelling; this one scores one.

```text
internal(S) = summed weight of edges with both endpoints in S
cut(S)      = summed weight of edges with exactly one endpoint in S
vol(S)      = 2 * internal(S) + cut(S)
phi(S)      = cut(S) / min(vol(S), vol(V) - vol(S))
```

### Score a membership you already have

The membership table can come from anywhere: an org chart, a manual labelling, a different algorithm.

```sql
CREATE TABLE assignment (node TEXT, team TEXT);
INSERT INTO assignment VALUES
  ('alice','left'), ('bob','left'),  ('carol','left'), ('dave','left'),
  ('eve','right'),  ('frank','right'), ('grace','right');

SELECT group_id, size, internal, cut, vol, round(phi, 3) AS phi
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
    AND membership_table = 'assignment'
    AND group_col = 'team' AND member_col = 'node'
  ORDER BY phi;
```

```text
group_id  size  internal  cut  vol   phi
--------  ----  --------  ---  ----  -----
left      4     5.0       1.0  11.0  0.143
right     3     3.0       1.0  7.0   0.143
```

The single bridge edge is the whole cut on both sides. `left` has the larger volume, so its denominator is the *complement's* volume (`18 - 11 = 7`), which is why both groups land on the same `phi`.

### Score the partition Leiden found

`graph_leiden` already emits `(node, community_id)` rows, so the composition is plain SQL:

```sql
CREATE TEMP TABLE discovered AS
  SELECT node, community_id FROM graph_leiden
    WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
      AND direction = 'both' AND resolution = 1.0;

SELECT group_id, size, internal, cut, vol, round(phi, 3) AS phi
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
    AND membership_table = 'discovered'
    AND group_col = 'community_id' AND member_col = 'node'
  ORDER BY phi;
```

```text
group_id  size  internal  cut  vol   phi
--------  ----  --------  ---  ----  -----
0         4     5.0       1.0  11.0  0.143
1         3     3.0       1.0  7.0   0.143
```

`group_id` is TEXT verbatim from `group_col`, so Leiden's INTEGER IDs come back as `'0'` and `'1'`.

### Compare a declared membership against a discovered one

Both sides are scored by the same metric on the same scale, which is the point modularity cannot serve:

```sql
SELECT 'declared' AS source, round(AVG(phi), 3) AS mean_phi, COUNT(*) AS groups
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND membership_table = 'assignment'
    AND group_col = 'team' AND member_col = 'node'
UNION ALL
SELECT 'leiden', round(AVG(phi), 3), COUNT(*)
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND membership_table = 'discovered'
    AND group_col = 'community_id' AND member_col = 'node';
```

```text
source    mean_phi  groups
--------  --------  ------
declared  0.143     2
leiden    0.143     2
```

### Apply a volume floor

Small groups are degenerate: a singleton with no internal edges scores `phi = 1.0` regardless of quality, and a group that swallows a whole component scores `phi = 0.0`. Both are arithmetically correct, and both are noise. Filter on `vol` rather than trusting `phi` alone:

```sql
SELECT group_id, size, vol, round(phi, 3) AS phi
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND membership_table = 'discovered'
    AND group_col = 'community_id' AND member_col = 'node'
    AND vol >= 10
  ORDER BY phi DESC;
```

```text
group_id  size  vol   phi
--------  ----  ----  -----
0         4     11.0  0.143
```

Weighted (`weight_col`) and temporally filtered (`timestamp_col`, `time_start`, `time_end`) inputs are inherited from the shared constraint set with no extra surface. Nodes in the graph that have no membership row are ungrouped: their edges still count toward the `cut` of the grouped endpoint, so a partial labelling is scored against the whole graph, not against itself.

!!! tip "Network community profile"
    Leskovec, Lang & Mahoney (2010) characterise a graph by plotting the best conductance found at each group size. Sweep `resolution` in `graph_leiden`, score each partition with `graph_conductance`, and plot `MIN(phi)` grouped by `size` to reproduce that profile in SQL.

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
- [API Reference — `graph_conductance`](api.md#graph_conductance) — output semantics for ungrouped nodes and degenerate groups

## References

- Brandes, U. (2001). [A Faster Algorithm for Betweenness Centrality](https://doi.org/10.1080/0022250X.2001.9990249). *Journal of Mathematical Sociology*, 25(2), 163–177.
- Wasserman, S. & Faust, K. (1994). *Social Network Analysis: Methods and Applications*. Cambridge University Press.
- Traag, V. A., Waltman, L. & van Eck, N. J. (2019). [From Louvain to Leiden: guaranteeing well-connected communities](https://arxiv.org/abs/1810.08473). *Scientific Reports*, 9(1), 5233.
- Kannan, R., Vempala, S. & Vetta, A. (2004). [On Clusterings: Good, Bad and Spectral](https://doi.org/10.1145/990308.990313). *Journal of the ACM*, 51(3), 497–515.
- Leskovec, J., Lang, K. J. & Mahoney, M. W. (2010). [Empirical Comparison of Algorithms for Network Community Detection](https://doi.org/10.1145/1772690.1772755). *WWW 2010*, 631–640.
