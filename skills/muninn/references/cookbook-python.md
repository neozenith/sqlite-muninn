# Python Cookbook

## Setup

```python
import sqlite3
import struct
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)
```

## Semantic Search over Documents

```python
# 1. Create index matching your embedding model's dimension
db.execute("""
    CREATE VIRTUAL TABLE doc_embeddings USING hnsw_index(
        dimensions=384, metric='cosine'
    )
""")

# 2. Insert document embeddings
dim = 384
for doc_id, embedding in documents:
    blob = struct.pack(f'{dim}f', *embedding)
    db.execute(
        "INSERT INTO doc_embeddings(rowid, vector) VALUES (?, ?)",
        (doc_id, blob)
    )

# 3. Search by query embedding
query_blob = struct.pack(f'{dim}f', *query_embedding)
results = db.execute(
    "SELECT rowid, distance FROM doc_embeddings WHERE vector MATCH ? AND k = 10",
    (query_blob,)
).fetchall()
```

## Batch Vector Insert

```python
dim = 128
data = [(i, struct.pack(f'{dim}f', *vec)) for i, vec in enumerate(vectors)]
db.executemany(
    "INSERT INTO my_index(rowid, vector) VALUES (?, ?)",
    data
)
db.commit()
```

## Knowledge Graph + Vector Hybrid

```python
# Store entities with embeddings
db.execute("""
    CREATE VIRTUAL TABLE entity_embeddings USING hnsw_index(
        dimensions=128, metric='cosine'
    )
""")

# Store relationships in a regular edge table
db.execute("""
    CREATE TABLE relationships (
        source TEXT,
        target TEXT,
        relation TEXT
    )
""")

# Find similar entities via vector search
similar = db.execute(
    "SELECT rowid, distance FROM entity_embeddings WHERE vector MATCH ? AND k = 5",
    (query_blob,)
).fetchall()

# Explore their graph neighborhood via BFS
for entity_id, _ in similar:
    neighbors = db.execute(
        "SELECT * FROM graph_bfs('relationships', 'source', 'target', ?)",
        (str(entity_id),)
    ).fetchall()
```

## Node2Vec to Clustering

```python
import numpy as np

# Generate graph embeddings
db.execute("""
    CREATE VIRTUAL TABLE node_emb USING hnsw_index(
        dimensions=64, metric='cosine'
    )
""")
db.execute("""
    SELECT node2vec_train('edges', 'source', 'target', 'node_emb', 64)
""")

# Extract embeddings as numpy array
rows = db.execute("SELECT rowid, vector FROM node_emb_nodes").fetchall()
ids = [r[0] for r in rows]
vectors = np.array([struct.unpack('64f', r[1]) for r in rows])

# Cluster with scikit-learn
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=5).fit_predict(vectors)
```
