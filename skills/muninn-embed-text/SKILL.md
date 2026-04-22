---
name: muninn-embed-text
description: >
  Generates text embeddings inside SQLite via muninn_embed() backed by local
  GGUF models (BERT, MiniLM, nomic-embed-text, BGE-M3, Qwen3-Embedding) through
  llama.cpp with Metal GPU acceleration on macOS. Covers muninn_embed_model
  registration, the temp.muninn_models virtual table, composing with hnsw_index
  for semantic search, and auto-embed TEMP triggers. Use when the user mentions
  "text embedding", "semantic search", "sentence embedding", "muninn_embed",
  "GGUF embedding model", "MiniLM", "nomic-embed", "BGE-M3", "Qwen3 embedding",
  "embed model in SQLite", "text-in semantic-search-out", or wants to embed
  text directly in SQL.
license: MIT
---

# muninn-embed-text — GGUF text embeddings in SQLite

`muninn_embed()` runs llama.cpp under the hood to produce L2-normalized float32 embeddings from raw text, in SQL. Output is the exact blob format accepted by `hnsw_index` (see [muninn-vector-search](../muninn-vector-search/SKILL.md)).

## Step 1 — Download a GGUF embedding model

```bash
mkdir -p models
curl -L -o models/all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

| Model | Dims | File | Use case |
|-------|------|------|----------|
| all-MiniLM-L6-v2 Q8_0 | 384 | 36 MB | Fast default for English |
| nomic-embed-text-v1.5 Q4_K_M | 768 | 84 MB | Long context, multilingual |
| BGE-M3 Q4_K_M | 1024 | 438 MB | 100+ languages |
| Qwen3-Embedding-8B Q4_K_M | 4096 | 4.7 GB | SOTA retrieval quality |

## Step 2 — Register the model (session-scoped)

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

Pooling is read from the GGUF metadata (BERT → MEAN, Qwen3 → LAST) — never hardcode. Models are registered per-connection and in `temp.*`; re-register after reconnecting.

## Step 3 — Embed text inline in SQL

```sql
SELECT length(muninn_embed('MiniLM', 'hello, world')) AS bytes;
```

```text
bytes
-----
1536        -- 384 floats × 4 bytes
```

`muninn_embed(model_name, text)` returns a `BLOB` with byte length `4 × dim`, already L2-normalized. That blob goes directly into `hnsw_index` or into any other column.

## End-to-end — semantic search pipeline

```sql
.load ./muninn

INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT);
INSERT INTO documents(content) VALUES
  ('The quick brown fox jumps over the lazy dog'),
  ('A fast runner sprints across the field'),
  ('SQLite is a lightweight embedded database'),
  ('Vector search finds similar items by distance');

CREATE VIRTUAL TABLE docs USING hnsw_index(dimensions=384, metric='cosine');

INSERT INTO docs(rowid, vector)
  SELECT id, muninn_embed('MiniLM', content) FROM documents;

SELECT d.content, round(v.distance, 4) AS dist
  FROM docs v JOIN documents d ON d.id = v.rowid
  WHERE v.vector MATCH muninn_embed('MiniLM', 'fast animal') AND k = 2;
```

```text
content                                        dist
---------------------------------------------  ------
A fast runner sprints across the field         0.3881
The quick brown fox jumps over the lazy dog    0.5217
```

## Auto-embed triggers

Use `TEMP` triggers only — persistent triggers fail when the DB is reopened without the extension.

```sql
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO docs(rowid, vector)
    VALUES (NEW.id, muninn_embed('MiniLM', NEW.content));
END;

CREATE TEMP TRIGGER auto_reembed AFTER UPDATE OF content ON documents
BEGIN
  DELETE FROM docs WHERE rowid = NEW.id;
  INSERT INTO docs(rowid, vector)
    VALUES (NEW.id, muninn_embed('MiniLM', NEW.content));
END;

-- Now text goes in, embeddings materialize automatically
INSERT INTO documents(content) VALUES ('Neural networks learn from data');
```

## Runtime variants

### Python

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)
db.enable_load_extension(False)

db.execute("""
    INSERT INTO temp.muninn_models(name, model)
    SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf')
""")

# Embed and retrieve as bytes
blob = db.execute("SELECT muninn_embed('MiniLM', ?)", ('hello',)).fetchone()[0]
print(len(blob))   # 1536
```

### Node.js

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);

db.exec(`
  INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');
`);

const blob = db.prepare("SELECT muninn_embed('MiniLM', ?) AS v")
               .get("hello world").v;
console.log(blob.byteLength);   // 1536
```

### WASM

```javascript
// Model file must be present in the virtual filesystem first
sqlite3.FS.writeFile("/models/miniLM.gguf", gguf_bytes);
db.exec(`
  INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('/models/miniLM.gguf');
`);
const blob = db.selectValue("SELECT muninn_embed('MiniLM', 'hello')");
```

WASM runs CPU-only — expect 5–10× slower embedding vs. Metal on macOS.

## Performance notes

- On macOS, embeddings run on Metal GPU by default (`MUNINN_GPU_LAYERS=99`). Single-text throughput ≈ 5k embeddings/sec for MiniLM on M1.
- No batched variant exists — `muninn_embed` takes one text at a time. For throughput, open multiple connections and embed in parallel.
- `muninn_token_count('MiniLM', text)` is cheaper than generating an embedding when you only need to chunk.

## Related functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `muninn_embed_model(path, n_ctx?)` | opaque handle | Load a GGUF file — only useful as INSERT value into `temp.muninn_models` |
| `muninn_embed(name, text)` | BLOB | Embed text with a registered model |
| `muninn_model_dim(name)` | INTEGER | Query dimension of a registered model |
| `muninn_tokenize(name, text)` | JSON array | Inspect tokenization |
| `muninn_token_count(name, text)` | INTEGER | Count tokens cheaply |

## See also

- [muninn-vector-search](../muninn-vector-search/SKILL.md) — the HNSW index that consumes embeddings
- [muninn-chat-extract](../muninn-chat-extract/SKILL.md) — GGUF chat models for NER/RE/summarization
- [muninn-graphrag](../muninn-graphrag/SKILL.md) — composite retrieval pipeline
- [text-embeddings.md](../../docs/text-embeddings.md) — full reference
