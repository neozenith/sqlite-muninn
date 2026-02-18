# Pure SQL Cookbook

All examples assume muninn is loaded via `.load ./muninn` in the SQLite CLI.

## Interactive Vector Search Session

```sql
-- Create an HNSW index
CREATE VIRTUAL TABLE demo USING hnsw_index(
    dimensions=4, metric='l2'
);

-- Insert some vectors (using zeroblob + custom function or from application code)
-- In practice, vectors are inserted from Python/Node.js/C via blob parameters

-- Search (query vector provided as blob parameter from application)
SELECT rowid, distance FROM demo WHERE vector MATCH ? AND k = 5;
```

## Graph Analysis Pipeline

```sql
-- 1. Create an edge table
CREATE TABLE edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight REAL DEFAULT 1.0
);

-- 2. Insert edges
INSERT INTO edges VALUES ('alice', 'bob', 1.0);
INSERT INTO edges VALUES ('bob', 'carol', 1.0);
INSERT INTO edges VALUES ('carol', 'alice', 1.0);
INSERT INTO edges VALUES ('dave', 'eve', 1.0);

-- 3. Find connected components
SELECT * FROM graph_components('edges', 'source', 'target');
-- Returns: alice->0, bob->0, carol->0, dave->1, eve->1

-- 4. Compute PageRank
SELECT * FROM graph_pagerank('edges', 'source', 'target');

-- 5. BFS from alice
SELECT * FROM graph_bfs('edges', 'source', 'target', 'alice');

-- 6. Shortest path from alice to carol
SELECT * FROM graph_shortest_path('edges', 'source', 'target', 'alice', 'carol');
```

## Centrality Analysis

```sql
-- Degree centrality
SELECT * FROM graph_degree('edges', 'source', 'target');

-- Betweenness centrality (identifies bridges/brokers)
SELECT * FROM graph_betweenness('edges', 'source', 'target');

-- Closeness centrality (identifies well-connected nodes)
SELECT * FROM graph_closeness('edges', 'source', 'target');
```

## Community Detection

```sql
-- Leiden algorithm for community detection
SELECT * FROM graph_leiden('edges', 'source', 'target');
```

## Graph Adjacency Cache

```sql
-- Create a persistent adjacency index over the edge table
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges', src_col='source', dst_col='target', weight_col='weight'
);

-- Query pre-computed degrees (reads CSR cache, not edge table)
SELECT node, in_degree, out_degree, weighted_in_degree, weighted_out_degree
FROM g ORDER BY out_degree DESC;

-- Point lookup
SELECT * FROM g WHERE node = 'alice';

-- Algorithm TVFs can read from the adjacency cache
SELECT node, centrality FROM graph_degree('g', 'source', 'target', 'weight');

-- Edge mutations automatically mark the cache as dirty
INSERT INTO edges VALUES ('alice', 'frank', 1.5);
-- Next query on g will auto-rebuild (incremental if delta is small)

-- Force a manual rebuild
INSERT INTO g(g) VALUES ('rebuild');
```

## Graph Select: Lineage Queries

```sql
-- Build a dependency DAG
CREATE TABLE deps (src TEXT, dst TEXT);
INSERT INTO deps VALUES ('A', 'C');
INSERT INTO deps VALUES ('B', 'C');
INSERT INTO deps VALUES ('C', 'D');
INSERT INTO deps VALUES ('C', 'E');
INSERT INTO deps VALUES ('E', 'F');

-- What depends on C? (descendants)
SELECT node, depth FROM graph_select('deps', 'src', 'dst', 'C+');
-- C(0), D(1), E(1), F(2)

-- What does C depend on? (ancestors)
SELECT node, depth FROM graph_select('deps', 'src', 'dst', '+C');
-- A(1), B(1), C(0)

-- Build closure: what must rebuild if C changes?
SELECT node FROM graph_select('deps', 'src', 'dst', '@C');

-- Common ancestors of D and E (intersection)
SELECT node FROM graph_select('deps', 'src', 'dst', '+D,+E');
-- A, B, C

-- Depth-limited: 1 hop in each direction from C
SELECT node FROM graph_select('deps', 'src', 'dst', '1+C+1');

-- Everything NOT downstream of C
SELECT node FROM graph_select('deps', 'src', 'dst', 'not C+');
```

## Node2Vec: Graph Embeddings to Vector Search

```sql
-- Create HNSW index for the embeddings
CREATE VIRTUAL TABLE node_emb USING hnsw_index(
    dimensions=64, metric='cosine'
);

-- Train Node2Vec on the edge table → embeddings stored in HNSW index
SELECT node2vec_train('edges', 'source', 'target', 'node_emb', 64);

-- Now search for nodes similar to a given node's embedding
-- (retrieve the embedding first, then use it as a query)
```
