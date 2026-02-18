# Node2Vec Guide

[Node2Vec](https://arxiv.org/abs/1607.00653) (Grover & Leskovec, 2016) learns vector embeddings from graph structure. Each node gets a dense vector that captures its structural role and neighborhood — nodes with similar graph positions end up close in vector space. The embeddings are stored directly in an HNSW index for similarity search.

## How It Works

1. **Random walks**: For each node, generate multiple biased random walks through the graph
2. **Skip-gram training**: Treat walks like sentences and train word2vec-style ([SGNS](https://arxiv.org/abs/1310.4546); Mikolov et al., 2013) to predict context nodes from center nodes
3. **Store in HNSW**: Write the learned embeddings into an existing `hnsw_index` virtual table

The result: you can query "which nodes are structurally similar to node X?" using vector similarity search.

## Basic Usage

```sql
.load ./muninn

-- Create the edge table
CREATE TABLE edges (src TEXT, dst TEXT);
INSERT INTO edges VALUES ('a', 'b'), ('b', 'c'), ('c', 'd'),
                         ('a', 'e'), ('e', 'f'), ('f', 'd');

-- Create an HNSW index for the embeddings
CREATE VIRTUAL TABLE node_emb USING hnsw_index(
    dimensions=64, metric='cosine'
);

-- Train Node2Vec (returns the number of nodes embedded)
SELECT node2vec_train(
    'edges',        -- edge table
    'src',          -- source column
    'dst',          -- destination column
    'node_emb',     -- HNSW table to store embeddings
    64,             -- embedding dimensions (must match HNSW table)
    1.0,            -- p: return parameter
    1.0,            -- q: in-out parameter
    10,             -- num_walks: walks per node
    80,             -- walk_length: max steps per walk
    5,              -- window_size: SGNS context window
    5,              -- negative_samples: negatives per positive
    0.025,          -- learning_rate: initial LR (decays linearly)
    5               -- epochs: training epochs
);
```

## Understanding p and q

The `p` and `q` parameters control the bias of random walks, which determines **what kind of structure** the embeddings capture.

### Return Parameter (p)

Controls the likelihood of returning to the previous node:

- **Low p (e.g., 0.25)**: Walks tend to backtrack, staying local — BFS-like behavior
- **High p (e.g., 4.0)**: Walks avoid backtracking, exploring further

### In-Out Parameter (q)

Controls whether walks move inward (toward the previous node's neighborhood) or outward:

- **Low q (e.g., 0.5)**: Walks explore outward, DFS-like — captures structural roles
- **High q (e.g., 2.0)**: Walks stay close to the start, capturing community structure

### Parameter Guide

| Setting | Walk Behavior | Best For | Example Use Case |
|---------|--------------|----------|-----------------|
| p=1, q=1 | Uniform random ([DeepWalk](https://arxiv.org/abs/1403.6652)) | General similarity | Default starting point |
| p=0.25, q=1 | BFS-like, local | Community/cluster similarity | Finding nodes in the same neighborhood |
| p=1, q=0.5 | DFS-like, exploratory | Structural role similarity | Finding hubs, bridges, periphery nodes |
| p=0.5, q=2.0 | Very local, community-focused | Tight community detection | Identifying cliques |

!!! tip "Start with p=1, q=1"
    The uniform setting (DeepWalk) is a good default. Only tune p and q if you have a specific structural hypothesis to test. For community detection, try low p. For structural role detection (finding all hub-like nodes), try low q.

## Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_walks` | 10 | Walks per node. More walks = better coverage, slower training |
| `walk_length` | 80 | Max steps per walk. Longer walks capture more global structure |
| `window_size` | 5 | SGNS context window. Larger windows relate more distant walk neighbors |
| `negative_samples` | 5 | Negative samples per positive. Standard value for SGNS |
| `learning_rate` | 0.025 | Initial learning rate. Decays linearly during training |
| `epochs` | 5 | Training passes over all walks |
| `dimensions` | — | Embedding size. Must match the HNSW table's `dimensions` parameter |

### Dimension Sizing

Embedding dimensionality is a quality vs. cost trade-off:

| Dimensions | Use Case |
|-----------|----------|
| 16–32 | Small graphs (< 1K nodes), quick experiments |
| 64 | Good default for most graphs |
| 128 | Large graphs (> 10K nodes) or when precision matters |
| 256+ | Rarely needed; diminishing returns |

## Integration with HNSW

After training, the HNSW index contains one embedding per node. You can search it like any other HNSW index:

```sql
-- Find nodes structurally similar to node 'alice'
-- Step 1: Get alice's embedding
SELECT vector FROM node_emb_nodes WHERE rowid = (
    SELECT rowid FROM node_emb WHERE rowid = CAST('alice' AS INTEGER)
);

-- Step 2: Use it as a query vector
SELECT rowid, distance FROM node_emb
WHERE vector MATCH ?alice_embedding AND k = 5;
```

### Retrieving Embeddings in Python

```python
import struct

# Get all embeddings
rows = db.execute("SELECT rowid, vector FROM node_emb_nodes").fetchall()
for rowid, blob in rows:
    dim = 64
    vector = struct.unpack(f'{dim}f', blob)
    print(f"Node {rowid}: first 3 dims = {vector[:3]}")
```

## Use Cases

### Similarity Search on Graph Structure

Find nodes with similar structural positions without needing attribute data:

```sql
-- Train on a citation graph
SELECT node2vec_train('citations', 'citing', 'cited', 'paper_emb', 64,
                      1.0, 1.0, 10, 80, 5, 5, 0.025, 5);

-- Find papers with similar citation patterns to paper #42
SELECT rowid, distance FROM paper_emb
WHERE vector MATCH (SELECT vector FROM paper_emb_nodes WHERE rowid = 42)
  AND k = 10;
```

### Cluster Detection with Community + Node2Vec

Combine Leiden communities with Node2Vec embeddings:

```sql
-- Detect communities
CREATE TEMP TABLE communities AS
SELECT node, community_id FROM graph_leiden
WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';

-- Train embeddings
SELECT node2vec_train('edges', 'src', 'dst', 'node_emb', 64,
                      0.25, 1.0, 10, 80, 5, 5, 0.025, 5);

-- Now you can: compute mean embedding per community for hierarchical search
```

### Enriching Vector Search with Graph Context

If you have both content embeddings (from a language model) and graph embeddings (from Node2Vec), you can combine them for hybrid retrieval:

1. Search content embeddings for semantic relevance
2. For top results, find their graph neighbors via BFS
3. Re-rank using centrality scores

See the [GraphRAG Cookbook](graphrag-cookbook.md) for a full worked example.

## References

- Grover, A. & Leskovec, J. (2016). [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653). *KDD '16*.
- Perozzi, B., Al-Rfou, R. & Skiena, S. (2014). [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652). *KDD '14*.
- Mikolov, T. et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). *NeurIPS 2013*.

## Further Reading

- [API Reference](api.md#node2vec) — Full parameter reference
- [Centrality & Community Guide](centrality-community.md) — Combine with centrality analysis
- [GraphRAG Cookbook](graphrag-cookbook.md) — End-to-end pipeline using Node2Vec
