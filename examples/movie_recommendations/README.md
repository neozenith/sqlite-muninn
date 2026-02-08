# Movie Recommendations — Full Pipeline

Learn movie embeddings from co-viewing patterns and find similar movies.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Node2Vec training | `SELECT node2vec_train('edges', 'src', 'dst', 'hnsw_table', dim, p, q, ...)` |
| All 13 parameters | dimensions, p, q, walks, length, window, negatives, lr, epochs |
| HNSW KNN on learned embeddings | `WHERE vector MATCH ? AND k = 8` |
| Rowid mapping | Replicating C code's first-seen node ordering |

## Data

15 movies in 3 genre clusters with co-preference edges ("users who liked X also liked Y"):

- **Sci-fi** (5): The Matrix, Inception, Interstellar, Blade Runner, Arrival
- **Action** (5): Die Hard, Mad Max, John Wick, Gladiator, The Dark Knight
- **Comedy** (5): Superbad, The Hangover, Bridesmaids, Step Brothers, Anchorman

Two cross-genre bridges connect the clusters weakly (The Dark Knight ↔ The Matrix, Superbad ↔ John Wick).

## The Rowid Mapping Problem

`node2vec_train()` assigns embedding rowids based on the order nodes are first encountered while iterating `SELECT src, dst FROM edge_table`. The Python script must replicate this ordering to map between movie names and HNSW rowids.

See `node2vec.c:72-92` (`graph_node_index()`) and `node2vec.c:559` (`rowid = i + 1`).

## Node2Vec Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| p | 0.5 | Low p = BFS-like walks, stay in local neighborhood |
| q | 0.5 | Low q = favor exploring within community structure |
| num_walks | 10 | Walks per node for training corpus |
| walk_length | 40 | Steps per walk |
| dimensions | 32 | Embedding size |
| epochs | 5 | Training passes over walk corpus |

## Run

```bash
make all
python examples/movie_recommendations/example.py
```

## Expected Output

- Node2Vec embeds all 15 movies
- KNN for "The Matrix" returns mostly sci-fi movies
- KNN for "Die Hard" returns mostly action movies
