# Text Embeddings Guide

muninn stores and searches vectors, but it does not generate embeddings from text. This guide shows how to get text embeddings into your HNSW index — from the simplest Python approach to a fully SQL-native workflow using GGUF models.

## Approaches

| Approach | Embedding Source | Language | Batch Speed | Dependencies |
|----------|-----------------|----------|-------------|--------------|
| [Python (sentence-transformers)](#python-sentence-transformers) | PyTorch model | Python | Fast (batched) | `sentence-transformers` |
| [SQL-native (sqlite-lembed)](#sql-native-sqlite-lembed) | GGUF model file | Pure SQL | Moderate (per-row) | `sqlite-lembed` extension |
| [SQL-native (sqlite-rembed)](#sql-native-sqlite-rembed) | Remote API | Pure SQL | Network-bound | `sqlite-rembed` extension |

---

## Python (sentence-transformers)

The most common approach. Generate embeddings in Python, insert the raw float32 blobs into muninn.

```python
import sqlite3
import struct
from sentence_transformers import SentenceTransformer

# Load model (downloads on first run, cached afterward)
model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()  # 384

# Connect and load muninn
db = sqlite3.connect("mydata.db")
db.enable_load_extension(True)
db.load_extension("./muninn")

# Create HNSW index
db.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_vectors
    USING hnsw_index(dimensions={dim}, metric='cosine')
""")

# Source data
documents = [
    (1, "The quick brown fox jumps over the lazy dog"),
    (2, "A fast runner sprints across the field"),
    (3, "SQLite is a lightweight embedded database"),
    (4, "Vector search finds similar items by distance"),
]

# Embed and insert
texts = [text for _, text in documents]
vectors = model.encode(texts, normalize_embeddings=True)

for (doc_id, _text), vec in zip(documents, vectors):
    blob = struct.pack(f"{dim}f", *vec.tolist())
    db.execute(
        "INSERT INTO doc_vectors(rowid, vector) VALUES (?, ?)",
        (doc_id, blob),
    )
db.commit()

# Search
query_vec = model.encode("fast animal", normalize_embeddings=True)
query_blob = struct.pack(f"{dim}f", *query_vec.tolist())

results = db.execute(
    "SELECT rowid, distance FROM doc_vectors WHERE vector MATCH ? AND k = 3",
    (query_blob,),
).fetchall()

for rowid, distance in results:
    print(f"  rowid={rowid}  distance={distance:.4f}")
```

!!! tip "Batch embedding is fast"
    `model.encode(texts)` processes a list of strings in a single forward pass with batching. This is significantly faster than encoding one string at a time. Always batch when inserting many rows.

---

## SQL-Native (sqlite-lembed)

Skip Python entirely. [sqlite-lembed](https://github.com/asg017/sqlite-lembed) is a companion SQLite extension that runs GGUF embedding models directly from SQL. Combined with muninn, you get a **text-in, semantic-search-out** workflow in pure SQL.

### What is a GGUF Model?

GGUF is a binary format for machine learning models that bundles weights, tokenizer vocabulary, and configuration in a single file. Models come in quantized variants (Q4, Q8, F16, F32) that trade file size for precision. For embedding tasks, Q8_0 is near-lossless and much smaller than full precision.

### Install

=== "pip"

    ```bash
    pip install sqlite-lembed
    ```

=== "npm"

    ```bash
    npm install sqlite-lembed
    ```

=== "Pre-built binary"

    Download from [GitHub Releases](https://github.com/asg017/sqlite-lembed/releases) (macOS and Linux).

### Download a Model

```bash
mkdir -p models

# all-MiniLM-L6-v2 — 36 MB, 384 dimensions, English
curl -L -o models/all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

Other models to consider:

| Model | Dims | Quantization | File Size | Notes |
|-------|------|-------------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Q8_0 | 36 MB | Smallest, fast, English |
| `nomic-embed-text-v1.5` | 768 | Q4_K_M | 84 MB | Long context (8192 tokens), Matryoshka |
| `BGE-small-en-v1.5` | 384 | Q8_0 | 37 MB | Strong English retrieval |
| `BGE-M3` | 1024 | Q4_K_M | 438 MB | Multilingual (100+ languages) |

Find more GGUF embedding models on [HuggingFace](https://huggingface.co/models?search=gguf+embedding) or the [curated collection](https://huggingface.co/collections/ChristianAzinn/embedding-ggufs-6615e9f216917dfdc6773fa3).

### Full SQL Workflow

```sql
-- Load both extensions
.load ./muninn
.load lembed0

-- Register the GGUF model (session-scoped)
INSERT INTO temp.lembed_models(name, model)
  SELECT 'MiniLM', lembed_model_from_file('models/all-MiniLM-L6-v2.Q8_0.gguf');

-- Create an HNSW index
CREATE VIRTUAL TABLE doc_vectors USING hnsw_index(
    dimensions=384, metric='cosine'
);

-- Source table for documents
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL
);

-- Insert documents
INSERT INTO documents(id, content) VALUES
    (1, 'The quick brown fox jumps over the lazy dog'),
    (2, 'A fast runner sprints across the field'),
    (3, 'SQLite is a lightweight embedded database'),
    (4, 'Vector search finds similar items by distance');

-- Embed and index all documents
INSERT INTO doc_vectors(rowid, vector)
  SELECT id, lembed('MiniLM', content) FROM documents;

-- Semantic search — embed the query inline
SELECT d.content, v.distance
FROM doc_vectors v
JOIN documents d ON d.id = v.rowid
WHERE v.vector MATCH lembed('MiniLM', 'fast animal')
  AND k = 5
ORDER BY v.distance;
```

### Auto-Embed with Triggers

Use a trigger to automatically embed new documents on insert:

```sql
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO doc_vectors(rowid, vector)
    VALUES (NEW.id, lembed('MiniLM', NEW.content));
END;

-- Now just insert text — the trigger handles embedding
INSERT INTO documents(content) VALUES ('Neural networks learn from data');
INSERT INTO documents(content) VALUES ('Graph databases store relationships');
```

!!! warning "Use TEMP triggers"
    Persistent triggers (`CREATE TRIGGER` without `TEMP`) store the trigger SQL in the database schema. When the database is later opened **without** sqlite-lembed loaded, SQLite cannot compile the trigger body and schema loading fails. Use `TEMP` triggers unless your application guarantees both extensions are loaded before every `sqlite3_open`.

### Update Workflow

When document content changes, re-embed and update the index:

```sql
CREATE TEMP TRIGGER auto_reembed AFTER UPDATE OF content ON documents
BEGIN
  DELETE FROM doc_vectors WHERE rowid = NEW.id;
  INSERT INTO doc_vectors(rowid, vector)
    VALUES (NEW.id, lembed('MiniLM', NEW.content));
END;
```

---

## SQL-Native (sqlite-rembed)

[sqlite-rembed](https://github.com/asg017/sqlite-rembed) generates embeddings by calling remote APIs — OpenAI, Nomic, Cohere, Jina, or local servers like Ollama.

### Install and Configure

```bash
pip install sqlite-rembed
```

=== "OpenAI"

    ```sql
    .load rembed0
    .load ./muninn

    INSERT INTO temp.rembed_clients(name, options) VALUES ('openai', 'openai');
    -- Reads OPENAI_API_KEY from environment

    SELECT rembed('openai', 'hello world');
    ```

=== "Ollama (local)"

    ```sql
    .load rembed0
    .load ./muninn

    INSERT INTO temp.rembed_clients(name, options)
      VALUES ('nomic', rembed_client_options('ollama', 'nomic-embed-text'));
    -- Requires: ollama serve + ollama pull nomic-embed-text

    SELECT rembed('nomic', 'hello world');
    ```

=== "Nomic API"

    ```sql
    .load rembed0
    .load ./muninn

    INSERT INTO temp.rembed_clients(name, options)
      VALUES ('nomic', rembed_client_options('nomic', 'nomic-embed-text-v1.5'));
    -- Reads NOMIC_API_KEY from environment

    SELECT rembed('nomic', 'hello world');
    ```

The `rembed()` function returns the same raw float32 blob format — it works identically with muninn's HNSW index.

!!! note "One HTTP call per row"
    sqlite-rembed does not batch requests. Each `rembed()` call makes one HTTP round-trip. For bulk embedding, use the Python approach or sqlite-lembed with a local GGUF model.

---

## Combining with GraphRAG

All three approaches produce the same float32 blob format. Once embeddings are in the HNSW index, the full [GraphRAG Cookbook](graphrag-cookbook.md) pipeline works regardless of how the embeddings were generated:

```sql
-- 1. Vector similarity search (seeding)
SELECT rowid, distance FROM doc_vectors
WHERE vector MATCH ?query_embedding AND k = 5;

-- 2. Graph expansion from seed nodes
SELECT node, depth FROM graph_bfs
WHERE edge_table = 'relationships' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = ?seed_node AND max_depth = 2;

-- 3. Centrality ranking
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'relationships' AND src_col = 'src' AND dst_col = 'dst'
  AND direction = 'both';
```

See [GraphRAG Cookbook](graphrag-cookbook.md) for the complete end-to-end tutorial.

---

## Vector Format Reference

All embedding methods must produce vectors in muninn's expected format:

| Property | Value |
|----------|-------|
| Encoding | Raw little-endian IEEE 754 float32 |
| Size | `4 bytes x dimensions` |
| Header | None |
| Normalization | Recommended for cosine metric (not enforced) |

=== "Python"

    ```python
    import struct
    blob = struct.pack(f"{dim}f", *values)
    ```

=== "Python (NumPy)"

    ```python
    import numpy as np
    blob = np.array(values, dtype=np.float32).tobytes()
    ```

=== "JavaScript"

    ```javascript
    const blob = Buffer.from(new Float32Array(values).buffer);
    ```

=== "C"

    ```c
    float vec[384];
    sqlite3_bind_blob(stmt, 1, vec, sizeof(vec), SQLITE_STATIC);
    ```

---

## Choosing a Model

| Priority | Recommended Model | Dims | Why |
|----------|------------------|------|-----|
| Smallest / fastest | all-MiniLM-L6-v2 | 384 | 22M params, sub-ms inference |
| Best general English | nomic-embed-text-v1.5 | 768 | Long context, Matryoshka support |
| Multilingual | BGE-M3 | 1024 | 100+ languages |
| High quality English | mxbai-embed-large-v1 | 1024 | Top MTEB scores |

!!! tip "Matryoshka embeddings"
    Models like nomic-embed-text-v1.5 support **Matryoshka Representation Learning** — you can truncate the output vector to a shorter dimension (e.g., 128 instead of 768) for faster search with minimal quality loss. Just take the first N values and re-normalize.

## Runnable Example

A complete end-to-end example is in [`examples/text_embeddings/`](https://github.com/neozenith/sqlite-muninn/tree/main/examples/text_embeddings). It demonstrates the full text-in, semantic-search-out workflow:

| Feature | How |
|---------|-----|
| Local GGUF embedding | `lembed('MiniLM', text)` via sqlite-lembed |
| OpenAI API embedding | `rembed('text-embedding-3-small', text)` via sqlite-rembed |
| Embed + insert in one SQL statement | `INSERT INTO idx(rowid, vector) SELECT id, lembed(...) FROM docs` |
| Auto-embed trigger | `CREATE TEMP TRIGGER ... lembed(...)` on INSERT |
| KNN semantic search | `WHERE vector MATCH lembed('MiniLM', 'query') AND k = 3` |
| Auto-download model | GGUF model downloaded to `models/` on first run |

```bash
make all
pip install sqlite-lembed              # local GGUF (no API key needed)
python examples/text_embeddings/example.py
```

The GGUF model file (all-MiniLM-L6-v2, 36 MB) is **downloaded automatically** on first run. Set `GGUF_MODEL_PATH` to use a custom model, or `OPENAI_API_KEY` to enable the rembed section:

```bash
# Both local and remote embedding
export OPENAI_API_KEY="sk-..."
python examples/text_embeddings/example.py
```

Each section runs conditionally — if sqlite-lembed or sqlite-rembed is not installed, or if `OPENAI_API_KEY` is empty, the corresponding section is skipped with a warning.

## Next Steps

- [Getting Started](getting-started.md) — Build and load the extension
- [API Reference](api.md) — HNSW index parameters and query syntax
- [GraphRAG Cookbook](graphrag-cookbook.md) — Full pipeline combining vector search with graph traversal
- [Node2Vec Guide](node2vec.md) — Learn structural embeddings from graph topology
