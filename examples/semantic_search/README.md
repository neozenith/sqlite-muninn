# Semantic Search — Document Similarity with HNSW

Find similar documents using vector embeddings and cosine similarity.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Create HNSW index | `CREATE VIRTUAL TABLE ... USING hnsw_index(dimensions=8, metric='cosine')` |
| Insert vectors | `INSERT INTO idx (rowid, vector) VALUES (?, ?)` |
| KNN search | `SELECT rowid, distance FROM idx WHERE vector MATCH ? AND k = 5` |
| Point lookup | `SELECT vector FROM idx WHERE rowid = 3` |
| Delete | `DELETE FROM idx WHERE rowid = 2` |

## Data

12 tech articles across 3 topics with hand-crafted 8-dimensional vectors:

- **AI articles** (4) — high values in AI/ML/NLP dimensions
- **Web articles** (4) — high values in Web/Frontend/Backend dimensions
- **Database articles** (4) — high values in DB/SQL dimensions

## Run

```bash
# From sqlite-vec-graph/ after building:
make all
python examples/semantic_search/example.py
```

## Expected Output

- KNN search with an AI-biased query returns AI articles first
- Point lookup retrieves the exact stored vector
- After deleting a document, it no longer appears in search results
