# Node2Vec

Learn vector embeddings from graph topology using biased random walks (Grover & Leskovec, 2016) and skip-gram with negative sampling (Mikolov et al., 2013). The output is written directly into an existing `hnsw_index`, turning structural similarity into an ordinary KNN query.

## Concept in three steps

1. **Biased random walks** — for each node, generate `num_walks` walks of up to `walk_length` steps. Walks are biased by two parameters (`p`, `q`) that trade off BFS-like vs DFS-like behavior.
2. **Skip-gram training** — treat walks like sentences, slide a context window of size `window`, and train per-node vectors so that nodes appearing close together in walks have close vectors.
3. **Store in HNSW** — each node ends up with one L2-normalized vector written to the provided HNSW table at `rowid = node_index + 1`.

## Signature

```sql
node2vec_train(
    edge_table TEXT,
    src_col TEXT,
    dst_col TEXT,
    output_table TEXT,     -- must already exist as hnsw_index, matching 'dimensions'
    dimensions INTEGER,
    p REAL, q REAL,
    num_walks INTEGER,
    walk_length INTEGER,
    window INTEGER,
    neg_samples INTEGER,
    learning_rate REAL,
    epochs INTEGER
) -> INTEGER               -- count of nodes embedded
```

All 12 arguments are positional and required. Edges are treated as undirected regardless of `src`/`dst` direction.

## Minimal recipe

```sql
.load ./muninn

CREATE TABLE edges (src TEXT, dst TEXT);
INSERT INTO edges VALUES
  ('a', 'b'), ('b', 'c'), ('c', 'd'),
  ('a', 'e'), ('e', 'f'), ('f', 'd');

CREATE VIRTUAL TABLE node_emb USING hnsw_index(dimensions=64, metric='cosine');

SELECT node2vec_train(
  'edges', 'src', 'dst', 'node_emb',
  64,                -- dimensions (match HNSW)
  1.0, 1.0,          -- p, q  (DeepWalk defaults)
  10, 40,            -- num_walks, walk_length
  5, 5,              -- window, neg_samples
  0.025, 3           -- learning_rate, epochs
);
```

```text
6        -- number of nodes embedded
```

## `p` and `q` — what they actually do

After each walk step from node `t` to node `v`, the next step's transition probability is scaled by:

- `1/p` — for returning to `t`
- `1.0` — for neighbors of `v` that are also neighbors of `t` (stay local)
- `1/q` — for neighbors of `v` that are far from `t` (explore outward)

| Setting | Behavior | Captures | When to use |
|---------|----------|----------|-------------|
| `p=1, q=1` | Uniform (DeepWalk) | General structural similarity | Default; start here |
| `p=0.25, q=1` | BFS-like, stays local | Community / neighborhood | "Find me nodes in the same cluster" |
| `p=1, q=0.5` | DFS-like, explores far | Structural role (hubs, bridges) | "Find me nodes in a similar graph *role*" |
| `p=0.5, q=2.0` | Very local | Tight cliques | "Find me nodes in the same dense subgraph" |

Start with `p=q=1` unless you have a specific structural hypothesis. Changing `p` and `q` only matters at scale (> ~500 nodes); for small graphs the differences are in the noise.

## Other parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_walks` | 10 | More walks → better coverage, linearly more compute |
| `walk_length` | 80 | Longer walks capture global structure |
| `window` | 5 | Skip-gram context — larger relates more distant walk neighbors |
| `neg_samples` | 5 | Standard SGNS value |
| `learning_rate` | 0.025 | Decays linearly over epochs |
| `epochs` | 5 | Training passes over the full walk corpus |
| `dimensions` | — | **Must equal** the `dimensions` declared on `output_table` |

### Dimension sizing

| Dimensions | Use case |
|-----------|----------|
| 16–32 | < 1,000 nodes, quick experiments |
| 64 | Default for most graphs |
| 128 | > 10,000 nodes or when precision matters |
| 256+ | Rarely worth it — diminishing returns |

## Querying structural similarity

After training, the HNSW index contains one embedding per node. muninn assigns `rowid = node_index + 1`, where `node_index` is the order the node was first seen while scanning the edge table.

To find nodes structurally similar to a known node:

```sql
-- Look up a node's index via the HNSW shadow table
WITH a_vec AS (
  SELECT vector FROM node_emb_nodes WHERE id = (
    SELECT node_idx + 1 FROM g WHERE node = 'a'
  )
)
SELECT ne.rowid, round(ne.distance, 4) AS d FROM node_emb ne, a_vec
  WHERE ne.vector MATCH a_vec.vector AND ne.k = 5;
```

This assumes a `graph_adjacency` virtual table `g` over the same edge table — its `node_idx` is the canonical node index. Without `graph_adjacency`, maintain your own mapping (e.g. a table `nodes(name TEXT PRIMARY KEY, idx INTEGER)`) populated in the same insertion order used for edges.

```text
rowid  d
-----  ------
1      0.0000
4      0.0832       -- 'd' is structurally similar to 'a'
...
```

!!! note "Rowid mapping caveat"
    `node2vec_train` does not expose the node-to-rowid map directly. The safest approach is to create a `graph_adjacency` vtable over the same edge table and use its `node_idx` column — it uses the same canonical ordering. See [API Reference — `graph_adjacency`](api.md#graph_adjacency).

## Exporting embeddings to Python

```python
import sqlite3, struct

db = sqlite3.connect("graph.db")
db.enable_load_extension(True)
db.load_extension("./muninn")

dim = 64
for rowid, blob in db.execute("SELECT rowid, vector FROM node_emb_nodes"):
    vec = struct.unpack(f"{dim}f", blob)
    print(f"Node idx {rowid - 1}: first 3 dims = {vec[:3]}")
```

The shadow table `{name}_nodes` is stable and safe to read directly.

## Use-case patterns

### Structural similarity ranking

Given a seed node from a vector search over content embeddings, boost recall by also retrieving structurally similar nodes from Node2Vec space. The two embeddings capture **different** similarity axes.

### Community supernodes

After running Leiden, compute the mean Node2Vec vector per community as a supernode embedding. Search supernodes first, drill into the matching community. See [GraphRAG Cookbook](graphrag-cookbook.md) for the full pattern.

### Role detection

Low-`q` walks capture **roles** rather than community — a low-q Node2Vec index groups all hub-like nodes together regardless of which cluster they're in. Useful for finding peripheral-role nodes, bridge-role nodes, etc.

## See also

- [API Reference — `node2vec_train`](api.md#node2vec_train) — exact parameter spec
- [Centrality and Community](centrality-community.md) — complementary signal from Leiden / betweenness
- [GraphRAG Cookbook](graphrag-cookbook.md) — full pipeline using Node2Vec for structural expansion

## References

- Grover, A. & Leskovec, J. (2016). [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653). *KDD '16*.
- Perozzi, B., Al-Rfou, R. & Skiena, S. (2014). [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652). *KDD '14*.
- Mikolov, T. et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). *NeurIPS 2013*.
