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

## Text-to-Vector with sqlite-lembed (Local GGUF)

Skip manual vector encoding entirely. sqlite-lembed runs a GGUF embedding model
directly in SQLite — combined with muninn, you get text-in, semantic-search-out.

```python
import sqlite3
import sqlite_lembed
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
sqlite_lembed.load(db)
db.enable_load_extension(False)

# Register a GGUF model (downloaded from HuggingFace, ~36 MB for MiniLM)
db.execute("""
    INSERT INTO temp.lembed_models(name, model)
    SELECT 'MiniLM', lembed_model_from_file('models/all-MiniLM-L6-v2.Q8_0.gguf')
""")

# Create HNSW index matching the model's output dimension
db.execute("""
    CREATE VIRTUAL TABLE doc_embeddings USING hnsw_index(
        dimensions=384, metric='cosine'
    )
""")

# Embed and index all documents in one statement
db.execute("""
    INSERT INTO doc_embeddings(rowid, vector)
    SELECT id, lembed('MiniLM', content) FROM documents
""")

# Semantic search — embed the query inline, no struct.pack needed
results = db.execute("""
    SELECT v.rowid, v.distance, d.content
    FROM doc_embeddings v
    JOIN documents d ON d.id = v.rowid
    WHERE v.vector MATCH lembed('MiniLM', 'search query text')
      AND k = 10
""").fetchall()
```

## Remote API Embedding with sqlite-rembed

For OpenAI, Ollama, Nomic, or other remote embedding APIs:

```python
import sqlite_rembed

sqlite_rembed.load(db)

# Register an OpenAI client (reads OPENAI_API_KEY from environment)
db.execute("""
    INSERT INTO temp.rembed_clients(name, options)
    VALUES ('text-embedding-3-small', 'openai')
""")

# Create HNSW index for OpenAI's 1536-dimensional output
db.execute("""
    CREATE VIRTUAL TABLE api_embeddings USING hnsw_index(
        dimensions=1536, metric='cosine'
    )
""")

# Embed and insert (one API call per row)
for doc_id, content in documents:
    embedding = db.execute(
        "SELECT rembed('text-embedding-3-small', ?)", (content,)
    ).fetchone()[0]
    db.execute(
        "INSERT INTO api_embeddings(rowid, vector) VALUES (?, ?)",
        (doc_id, embedding)
    )
```

## Auto-Embed Trigger Pattern

Automatically embed new rows using a TEMP trigger. Works with both lembed and rembed:

```python
# Create the trigger (requires lembed model already registered)
db.execute("""
    CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
    BEGIN
      INSERT INTO doc_embeddings(rowid, vector)
        VALUES (NEW.id, lembed('MiniLM', NEW.content));
    END
""")

# Now just insert text — embedding + indexing happens automatically
db.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (99, "New document text"))

# Re-embed on update
db.execute("""
    CREATE TEMP TRIGGER auto_reembed AFTER UPDATE OF content ON documents
    BEGIN
      DELETE FROM doc_embeddings WHERE rowid = NEW.id;
      INSERT INTO doc_embeddings(rowid, vector)
        VALUES (NEW.id, lembed('MiniLM', NEW.content));
    END
""")
```

## Semantic Search over Documents (Manual Encoding)

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
