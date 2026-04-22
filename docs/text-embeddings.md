# Text Embeddings

How to go from raw text to a populated `hnsw_index`. muninn ships its own GGUF embedder (`muninn_embed`) that runs inside SQLite with optional Metal GPU acceleration — this is the preferred path. Python-side and remote-API embedders are also supported for workflows that already have embedding code.

## The three paths

| Path | Embedder lives | Best for |
|------|---------------|----------|
| [Native `muninn_embed`](#path-1-native-muninn_embed-preferred) | In muninn, via llama.cpp | New projects, SQL-native pipelines, macOS with Metal |
| [Python batch (sentence-transformers)](#path-2-python-batch-via-sentence-transformers) | Your application | Existing ML code, very large bulk ingestion |
| [Remote API (sqlite-rembed)](#path-3-remote-api-via-sqlite-rembed) | OpenAI / Nomic / Cohere / Ollama | API-driven pipelines, models muninn doesn't run locally |

All three produce the same `float32` blob — once vectors are in the HNSW index, the downstream graph, centrality, and retrieval code is identical regardless of how they got there.

---

## Path 1 — Native `muninn_embed` (preferred)

### Loading a GGUF model

muninn stores models in a session-scoped virtual table called `temp.muninn_models`. Register a model once per connection:

```sql
.load ./muninn

INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

SELECT name, dim FROM temp.muninn_models;
```

```text
name    dim
------  ---
MiniLM  384
```

On macOS, all model layers are offloaded to the Metal GPU by default. Override with `MUNINN_GPU_LAYERS=0` for CPU-only. See [Getting Started](getting-started.md#metal-gpu-acceleration-macos).

### Downloading a model

Pick one based on your language coverage, quality ceiling, and file-size budget.

```bash
mkdir -p models

# English, tiny & fast
curl -L -o models/all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

| Model | Dims | Quant | File | Strengths |
|-------|------|-------|------|-----------|
| all-MiniLM-L6-v2 | 384 | Q8_0 | 36 MB | Smallest, fast, English only |
| nomic-embed-text-v1.5 | 768 | Q4_K_M | 84 MB | Long context (8192 tok), Matryoshka |
| BGE-small-en-v1.5 | 384 | Q8_0 | 37 MB | Strong English retrieval |
| BGE-M3 | 1024 | Q4_K_M | 438 MB | 100+ languages |
| Qwen3-Embedding-8B | 4096 | Q4_K_M | 4.7 GB | State-of-the-art retrieval quality |

Find more GGUF embedding models on [HuggingFace](https://huggingface.co/models?search=gguf+embedding) or the [curated collection](https://huggingface.co/collections/ChristianAzinn/embedding-ggufs-6615e9f216917dfdc6773fa3).

Model pooling (MEAN for BERT-family, LAST for Qwen3, etc.) is read from the GGUF metadata — muninn never hardcodes it.

### Embed + index in a single statement

```sql
CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT);

INSERT INTO documents(content) VALUES
  ('The quick brown fox jumps over the lazy dog'),
  ('A fast runner sprints across the field'),
  ('SQLite is a lightweight embedded database'),
  ('Vector search finds similar items by distance'),
  ('Neural networks learn patterns from data');

CREATE VIRTUAL TABLE docs_vec USING hnsw_index(
  dimensions=384, metric='cosine'
);

INSERT INTO docs_vec(rowid, vector)
  SELECT id, muninn_embed('MiniLM', content) FROM documents;
```

### Semantic search — embed the query inline

```sql
SELECT d.content, round(v.distance, 4) AS dist
  FROM docs_vec v JOIN documents d ON d.id = v.rowid
  WHERE v.vector MATCH muninn_embed('MiniLM', 'fast animal')
    AND k = 3
  ORDER BY v.distance;
```

```text
content                                              dist
---------------------------------------------------  ------
A fast runner sprints across the field               0.3881
The quick brown fox jumps over the lazy dog          0.5217
Neural networks learn patterns from data             0.7402
```

### Auto-embed new rows with a trigger

```sql
CREATE TEMP TRIGGER docs_auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO docs_vec(rowid, vector)
    VALUES (NEW.id, muninn_embed('MiniLM', NEW.content));
END;

INSERT INTO documents(content) VALUES ('Graph databases store relationships');
-- The trigger embedded and indexed the new row automatically.
```

!!! warning "Use `TEMP` triggers for model-backed embedding"
    Persistent triggers (`CREATE TRIGGER` without `TEMP`) are written into the database schema. Opening the database later **without** muninn loaded (or without that model registered) causes schema compilation to fail. `TEMP` triggers live only for the session and avoid this trap.

### Re-embed on update

```sql
CREATE TEMP TRIGGER docs_auto_reembed AFTER UPDATE OF content ON documents
BEGIN
  DELETE FROM docs_vec WHERE rowid = NEW.id;
  INSERT INTO docs_vec(rowid, vector)
    VALUES (NEW.id, muninn_embed('MiniLM', NEW.content));
END;
```

### Unloading a model

```sql
DELETE FROM temp.muninn_models WHERE name = 'MiniLM';
```

### Performance notes

- Single-text embedding throughput on M1 Pro (Metal, MiniLM, 384 dim): ~5,000 embeds/sec
- `muninn_embed` does not batch internally — one call per row. For 100k+ document bulk ingestion, Path 2 (Python batch) can be 3–5× faster.
- CPU fallback (`MUNINN_GPU_LAYERS=0`) is ~3× slower on Apple Silicon, but works identically.
- Multiple SQLite connections can call `muninn_embed` concurrently; the registry is thread-safe, each connection gets its own compute context.

---

## Path 2 — Python batch via sentence-transformers

Best for bulk ingestion where Python can batch thousands of texts into a single GPU forward pass.

```python
import sqlite3, struct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()   # 384

db = sqlite3.connect("mydata.db")
db.enable_load_extension(True)
db.load_extension("./muninn")
db.enable_load_extension(False)

db.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec
    USING hnsw_index(dimensions={dim}, metric='cosine')
""")

documents = [
    (1, "The quick brown fox jumps over the lazy dog"),
    (2, "A fast runner sprints across the field"),
    (3, "SQLite is a lightweight embedded database"),
]

texts = [t for _, t in documents]
vectors = model.encode(texts, normalize_embeddings=True)

db.executemany(
    "INSERT INTO docs_vec(rowid, vector) VALUES (?, ?)",
    [(i, struct.pack(f"{dim}f", *vec.tolist()))
     for (i, _), vec in zip(documents, vectors)],
)
db.commit()

# Query-time: embed the query in Python, pass the blob
query_vec = model.encode("fast animal", normalize_embeddings=True)
query_blob = struct.pack(f"{dim}f", *query_vec.tolist())

for rowid, distance in db.execute(
    "SELECT rowid, distance FROM docs_vec "
    "WHERE vector MATCH ? AND k = 3", (query_blob,)
):
    print(rowid, f"{distance:.4f}")
```

!!! tip "Batch encoding is the win"
    `model.encode(texts)` processes many strings in a single forward pass — the speedup vs. per-row embedding scales roughly with batch size. For large ingestion, batch on the Python side even if you use `muninn_embed` at query time.

---

## Path 3 — Remote API via sqlite-rembed

For OpenAI / Nomic / Cohere / Jina / Ollama. [sqlite-rembed](https://github.com/asg017/sqlite-rembed) adds a `rembed()` scalar that makes one HTTP call per row. Useful for API-only models muninn can't run locally (`text-embedding-3-large`, Cohere Embed v3, etc.).

```sql
.load rembed0
.load ./muninn

INSERT INTO temp.rembed_clients(name, options) VALUES ('openai', 'openai');
-- Reads OPENAI_API_KEY from the environment

CREATE VIRTUAL TABLE docs_vec USING hnsw_index(dimensions=1536, metric='cosine');

INSERT INTO docs_vec(rowid, vector)
  SELECT id, rembed('openai', content) FROM documents;
```

!!! note "One HTTP call per row"
    `rembed()` does not batch. Each row is one round-trip to the provider. For thousands of rows, run Path 2 in Python (which can batch provider APIs) and then insert the blobs.

---

## Vector format reference

Every embedding path — muninn, Python, remote — must produce **the same blob format**:

| Property | Value |
|----------|-------|
| Encoding | Raw little-endian IEEE 754 `float32` array |
| Size | `4 × dimensions` bytes |
| Header | none |
| Normalization | Recommended for `metric='cosine'`; required if you want distances in `[0, 2]` |

=== "Python (struct)"
    ```python
    import struct
    blob = struct.pack(f"{dim}f", *values)
    ```

=== "Python (NumPy)"
    ```python
    import numpy as np
    blob = np.asarray(values, dtype=np.float32).tobytes()
    ```

=== "Node.js"
    ```javascript
    const blob = Buffer.from(new Float32Array(values).buffer);
    ```

=== "C"
    ```c
    float vec[384];
    sqlite3_bind_blob(stmt, 1, vec, sizeof(vec), SQLITE_STATIC);
    ```

=== "Rust"
    ```rust
    let bytes: Vec<u8> = values.iter()
        .flat_map(|v: &f32| v.to_le_bytes())
        .collect();
    ```

## Choosing a model

| Priority | Model | Dim | Why |
|----------|-------|-----|-----|
| Smallest / fastest | all-MiniLM-L6-v2 | 384 | 22M params, sub-ms on Metal |
| Best general English | nomic-embed-text-v1.5 | 768 | Long context, Matryoshka-truncatable |
| Multilingual | BGE-M3 | 1024 | 100+ languages |
| Retrieval quality ceiling | Qwen3-Embedding-8B | 4096 | Top MTEB, 4.7 GB file |

!!! tip "Matryoshka embeddings"
    nomic-embed-text-v1.5 and some BGE models support Matryoshka Representation Learning — you can truncate the output to a shorter dimension (e.g. 128 from 768) for faster search with minimal quality loss. Truncate, then re-normalize.

## Combining with graph retrieval

Once embeddings are in the index, the full retrieval pipeline is path-agnostic:

```sql
-- Phase 1: vector seed
SELECT rowid FROM docs_vec
  WHERE vector MATCH muninn_embed('MiniLM', 'find close matches') AND k = 5;

-- Phase 2: graph expansion from seeds
SELECT node FROM graph_bfs
  WHERE edge_table = 'relationships' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = ?seed AND max_depth = 2;

-- Phase 3: centrality ranking
SELECT node, centrality FROM graph_node_betweenness
  WHERE edge_table = 'relationships' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both';
```

See [GraphRAG Cookbook](graphrag-cookbook.md) for the full pipeline.

## See also

- [API Reference — `muninn_embed`](api.md#muninn_embed)
- [API Reference — `hnsw_index`](api.md#hnsw_index)
- [Chat and Extraction](chat-and-extraction.md) — same GGUF infrastructure, chat-side
- [GraphRAG Cookbook](graphrag-cookbook.md) — end-to-end pipeline
