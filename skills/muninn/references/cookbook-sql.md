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
