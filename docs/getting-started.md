# Getting Started

This page walks through installing muninn, loading the extension, and running a smoke test that exercises each major capability. Every code block is copy-paste runnable.

## 1. Install

Pick one of the four methods below. All four produce the same loadable library — the difference is only where the file lives on disk.

=== "Python (pip)"
    ```bash
    pip install sqlite-muninn
    ```

    ```python
    import sqlite3, sqlite_muninn
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_muninn.load(db)           # resolves the bundled .so/.dylib/.dll
    db.enable_load_extension(False)
    ```

    !!! warning "macOS system Python cannot load extensions"
        Apple's `/usr/bin/python3` is compiled with `SQLITE_OMIT_LOAD_EXTENSION`. Use Homebrew Python (`brew install python`) or install `pysqlite3-binary` as a drop-in replacement.

=== "Node.js (npm)"
    ```bash
    npm install sqlite-muninn better-sqlite3
    ```

    ```javascript
    import Database from "better-sqlite3";
    import { load } from "sqlite-muninn";

    const db = new Database(":memory:");
    load(db);
    ```

=== "SQLite CLI — prebuilt binary"
    Download the prebuilt shared library for your platform from [GitHub Releases](https://github.com/neozenith/sqlite-muninn/releases) and place it anywhere on your filesystem.

    ```bash
    sqlite3
    sqlite> .load /path/to/muninn        # no file extension needed
    ```

=== "From source"
    Requires a C11 compiler, CMake, Python 3.11+ (for the build driver), and SQLite headers.

    ```bash
    # Prerequisites
    brew install sqlite                     # macOS
    sudo apt-get install libsqlite3-dev     # Debian / Ubuntu

    # Clone with submodules (includes vendored llama.cpp and yyjson)
    git clone --recurse-submodules https://github.com/neozenith/sqlite-muninn.git
    cd sqlite-muninn
    make all                                # → muninn.dylib / .so / .dll
    ```

    The Makefile auto-detects macOS and enables Metal GPU acceleration. On Apple Silicon the first build of `vendor/llama.cpp` takes ~3 minutes.

## 2. Verify the install

Load the extension and check every major subsystem registered. This is a 20-line end-to-end smoke test.

```sql
-- sqlite3 shell, or equivalent via a host language
.load ./muninn

-- HNSW vector index
CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=4, metric='l2');
INSERT INTO vec(rowid, vector) VALUES (1, X'0000803F000000000000000000000000');
SELECT rowid, distance FROM vec
  WHERE vector MATCH X'0000803F000000000000000000000000' AND k = 1;

-- Graph TVF (works on any edge table)
CREATE TABLE edges (src TEXT, dst TEXT);
INSERT INTO edges VALUES ('a', 'b'), ('b', 'c');
SELECT node, depth FROM graph_bfs
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'a' AND max_depth = 5;
```

```text
rowid  distance
-----  --------
1      0.0

node   depth
-----  -----
a      0
b      1
c      2
```

If both queries return rows, the extension is loaded correctly.

## 3. Download a GGUF model (optional, enables `muninn_embed` / `muninn_chat`)

muninn can embed text and run chat/extraction with local GGUF models through llama.cpp. Pull a small embedding model to smoke-test the ML surface:

```bash
mkdir -p models
curl -L -o models/all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

| Model | Dims | File | Notes |
|-------|------|------|-------|
| all-MiniLM-L6-v2 Q8_0 | 384 | 36 MB | English, fast, good default |
| nomic-embed-text-v1.5 Q4_K_M | 768 | 84 MB | Long context (8192 tokens), multilingual |
| BGE-M3 Q4_K_M | 1024 | 438 MB | 100+ languages |
| Qwen3-Embedding-8B Q4_K_M | 4096 | 4.7 GB | State-of-the-art retrieval quality |

### Smoke-test the embedding path

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

```sql
SELECT length(muninn_embed('MiniLM', 'hello, world')) AS blob_bytes;
```

```text
blob_bytes
----------
1536          -- 384 floats × 4 bytes
```

If the `blob_bytes` value equals `dim × 4`, llama.cpp is fully wired up.

### Metal GPU acceleration (macOS)

On macOS, muninn offloads all layers to Metal by default (`MUNINN_DEFAULT_GPU_LAYERS=99` compile-time flag). Set an env var before loading to override:

```bash
export MUNINN_GPU_LAYERS=0      # CPU-only
export MUNINN_LOG_LEVEL=warn    # surface llama.cpp warnings (default: silent)
sqlite3 -cmd ".load ./muninn"
```

On Linux, CPU is the default; there is no CUDA support in the current build.

## 4. A guided tour of the capability surfaces

### 4.1 Vector search with a GGUF embedder

```sql
.load ./muninn

INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

CREATE VIRTUAL TABLE docs USING hnsw_index(dimensions=384, metric='cosine');

CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT);
INSERT INTO documents(content) VALUES
  ('The quick brown fox jumps over the lazy dog'),
  ('A fast runner sprints across the field'),
  ('SQLite is a lightweight embedded database'),
  ('Vector search finds similar items by distance');

INSERT INTO docs(rowid, vector)
  SELECT id, muninn_embed('MiniLM', content) FROM documents;

SELECT d.content, round(v.distance, 4) AS dist
  FROM docs v JOIN documents d ON d.id = v.rowid
  WHERE v.vector MATCH muninn_embed('MiniLM', 'fast animal') AND k = 2;
```

```text
content                                              dist
---------------------------------------------------  ------
A fast runner sprints across the field               0.3881
The quick brown fox jumps over the lazy dog          0.5217
```

See [Text Embeddings](text-embeddings.md) for Python-side embedders, remote API embedders, auto-embed triggers, and model selection.

### 4.2 Graph traversal on any edge table

Graph TVFs do not need special tables — they run on any table with a source and destination column.

```sql
CREATE TABLE friendships (user_a TEXT, user_b TEXT);
INSERT INTO friendships VALUES
  ('alice', 'bob'), ('bob', 'carol'), ('carol', 'dave'),
  ('dave', 'eve'),  ('alice', 'eve'), ('eve',  'frank');

-- All friends of alice within 2 hops
SELECT node, depth FROM graph_bfs
  WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b'
    AND start_node = 'alice' AND max_depth = 2 AND direction = 'both';
```

```text
node    depth
------  -----
alice   0
bob     1
eve     1
carol   2
dave    2
frank   2
```

### 4.3 Centrality and community detection

```sql
-- Bridge users (highest node betweenness)
SELECT node, round(centrality, 3) AS c FROM graph_node_betweenness
  WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b'
    AND direction = 'both' AND normalized = 1
  ORDER BY c DESC LIMIT 3;

-- Community partition (Leiden)
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b';
```

See [Centrality and Community](centrality-community.md) for parameter tuning.

### 4.4 Learn structural embeddings with Node2Vec

```sql
CREATE VIRTUAL TABLE user_emb USING hnsw_index(dimensions=32, metric='cosine');

SELECT node2vec_train(
  'friendships', 'user_a', 'user_b', 'user_emb',
  32,                    -- dimensions (must match HNSW)
  1.0, 1.0,              -- p, q (DeepWalk defaults)
  10, 40,                -- num_walks, walk_length
  5, 5,                  -- window_size, negative_samples
  0.025, 3               -- learning_rate, epochs
);
```

Returns the number of nodes embedded. See [Node2Vec](node2vec.md) for p/q tuning.

### 4.5 Extract structured data from text

```sql
INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');

SELECT json_extract(
  muninn_extract_entities('Qwen3.5-4B',
    'Elon Musk founded Tesla in 2003 in Palo Alto.',
    'person,organization,date,location'),
  '$.entities'
) AS entities;
```

```text
entities
------------------------------------------------------------------------
[{"text":"Elon Musk","type":"person","score":0.98},
 {"text":"Tesla","type":"organization","score":0.97},
 {"text":"2003","type":"date","score":0.95},
 {"text":"Palo Alto","type":"location","score":0.94}]
```

See [Chat and Extraction](chat-and-extraction.md) for batch variants, relation extraction, summarization, and unsupervised (open-label) mode.

## 5. Vector blob format

Vectors are raw little-endian `float32` arrays with no header — `4 × dimensions` bytes per vector. See [Text Embeddings → Vector format reference](text-embeddings.md#vector-format-reference) for per-language `struct.pack` / `Float32Array` / C snippets.

## 6. Where to go next

| Goal | Page |
|------|------|
| Understand every SQL symbol | [API Reference](api.md) |
| Embed text and run semantic search | [Text Embeddings](text-embeddings.md) |
| Named entity / relation extraction | [Chat and Extraction](chat-and-extraction.md) |
| Dedupe entities end-to-end | [Entity Resolution](entity-resolution.md) |
| Centrality, Leiden, PageRank | [Centrality and Community](centrality-community.md) |
| dbt-style lineage queries | [Graph Select](graph-select.md) |
| Learn structural node embeddings | [Node2Vec](node2vec.md) |
| Combine everything into retrieval | [GraphRAG Cookbook](graphrag-cookbook.md) |
| How the extension is built internally | [Architecture](architecture.md) |

## Common pitfalls

- **`unable to open shared library` when loading on macOS.** System Python cannot load extensions. Use Homebrew Python or `pysqlite3-binary`.
- **`Segmentation fault` in tests.** The ASan-instrumented build (`make debug`) cannot be loaded from a non-ASan Python. Use `make all` for integration from Python.
- **`unable to find model 'X'`.** Model names in `muninn_embed` / `muninn_chat` refer to the `name` column in `temp.muninn_models` / `temp.muninn_chat_models`. Models are session-scoped — re-register after reconnecting.
- **CMake hangs on Apple Silicon during first build.** llama.cpp's `GGML_NATIVE=ON` triggers a hanging SVE feature probe. muninn's Makefile passes `-DGGML_NATIVE=OFF` automatically; if you bypass the Makefile, pass it manually.
