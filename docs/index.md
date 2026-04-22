# sqlite-muninn

<div align="center">
    <img src="assets/muninn_logo_transparent.png" alt="Muninn raven logo" width="360"/>
</div>

This project aims to build **agentic memory** and **knowledge graph** primitives for sqlite as a native C extension and also made available for Python, Node.JS and WASM. 

It is **an advanced collection of knowledge graph primitives** like Vector Similarity Search, HNSW Indexes, Graph database, Community Detection, Node2Vec capabilities and loading GGUF models via llama.cpp integration.

```text
Huginn and Muninn fly each day over the wide world.
I fear for Huginn that he may not return,
yet I worry more for Muninn.

- Poetic Edda (Grimnismal, stanza 20)
```

_Odin fears losing Memory more than Thought._

Huginn and Muninn are the two ravens of Odin and their names translate to _Thoughts_ and _Memory_.


| Package Index | Published Version | Downloads | 
|---|---|---|
| PyPI | [![PyPI](https://img.shields.io/pypi/v/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) | [![PyPI Downloads](https://img.shields.io/pypi/dm/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) |
| npm | [![npm](https://img.shields.io/npm/v/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) | [![npm Downloads](https://img.shields.io/npm/dm/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) |

## Quick Start

```sql
.load ./muninn

-- 1. Load an embedding model (one-time, session-scoped)
INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

-- 2. Create an HNSW index sized to the model
CREATE VIRTUAL TABLE docs USING hnsw_index(dimensions=384, metric='cosine');

-- 3. Embed and index text in one statement
INSERT INTO docs(rowid, vector) VALUES
  (1, muninn_embed('MiniLM', 'The quick brown fox jumps over the lazy dog')),
  (2, muninn_embed('MiniLM', 'SQLite is a lightweight embedded database')),
  (3, muninn_embed('MiniLM', 'Vector search finds similar items by distance'));

-- 4. Semantic KNN — embed the query inline
SELECT rowid, round(distance, 4) AS dist FROM docs
  WHERE vector MATCH muninn_embed('MiniLM', 'find close matches') AND k = 2;
```

```text
rowid  dist
-----  ------
3      0.1823
2      0.3104
```

| Package | Version | Downloads |
|---------|---------|-----------|
| PyPI    | [![PyPI](https://img.shields.io/pypi/v/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) | [![PyPI Downloads](https://img.shields.io/pypi/dm/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) |
| npm     | [![npm](https://img.shields.io/npm/v/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) | [![npm Downloads](https://img.shields.io/npm/dm/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) |

## Status

!!! warning "Pre-release software"
    APIs may change between minor versions. Shadow-table layouts are versioned via each virtual table's `_config` table; upgrades may require a rebuild. Pin an exact version in production.

| Platform | Build | GPU | Status |
|----------|-------|-----|--------|
| macOS (Apple Silicon) | Full | Metal, all layers by default (`MUNINN_GPU_LAYERS=99`) | Supported |
| macOS (Intel) | Full | CPU + Accelerate framework | Supported |
| Linux (x86_64) | Full | CPU, optional BLAS | Supported |
| Linux (ARM64) | Full | CPU | Supported |
| Windows | Full | CPU | Experimental — contributions welcome |
| WASM | Full | CPU only | Supported |

Every platform ships the same SQL surface — HNSW, graph TVFs, centrality, Leiden, node2vec, `graph_select`, and the full GGUF LLM family (`muninn_embed`, `muninn_chat`, `muninn_extract_*`, `muninn_extract_er`, `muninn_label_groups`).

## Install

=== "Python"
    ```bash
    pip install sqlite-muninn
    ```
    ```python
    import sqlite3, sqlite_muninn
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_muninn.load(db)
    ```

=== "Node.js"
    ```bash
    npm install sqlite-muninn
    ```
    ```javascript
    import Database from "better-sqlite3";
    import { load } from "sqlite-muninn";
    const db = new Database(":memory:");
    load(db);
    ```

=== "SQLite CLI"
    ```bash
    # Download the prebuilt binary from GitHub Releases, or build from source:
    brew install sqlite && make all          # macOS
    sudo apt install libsqlite3-dev && make  # Linux
    sqlite3
    sqlite> .load ./muninn
    ```

=== "From source"
    ```bash
    git clone --recurse-submodules https://github.com/neozenith/sqlite-muninn.git
    cd sqlite-muninn
    make all            # builds muninn.dylib / muninn.so / muninn.dll
    make test           # C unit tests
    make test-python    # Python integration tests
    ```

See [Getting Started](getting-started.md) for per-platform prerequisites, verification, and common pitfalls.

## What muninn provides

A single `.load` registers three capability surfaces:

### Vector search

| Symbol | Kind | Purpose |
|--------|------|---------|
| [`hnsw_index`](api.md#hnsw_index) | Virtual table | Hierarchical Navigable Small World KNN index |
| [`muninn_embed`](api.md#muninn_embed) | Scalar | Embed text with a local GGUF model |
| [`muninn_embed_model`](api.md#muninn_embed_model) | Scalar | Register a GGUF embedding model |
| [`muninn_model_dim`](api.md#muninn_model_dim) | Scalar | Query model embedding dimension |
| [`muninn_models`](api.md#muninn_models) | Virtual table | Embedding model lifecycle |

### Graph algorithms

| Symbol | Kind | Purpose |
|--------|------|---------|
| [`graph_bfs`, `graph_dfs`](api.md#graph_bfs-graph_dfs) | TVF | Breadth/depth-first traversal |
| [`graph_shortest_path`](api.md#graph_shortest_path) | TVF | Unweighted BFS / weighted Dijkstra |
| [`graph_components`](api.md#graph_components) | TVF | Connected components (union-find) |
| [`graph_pagerank`](api.md#graph_pagerank) | TVF | Iterative PageRank |
| [`graph_degree`, `graph_node_betweenness`, `graph_edge_betweenness`, `graph_closeness`](api.md#centrality) | TVF | Centrality measures |
| [`graph_leiden`](api.md#graph_leiden) | TVF | Leiden community detection |
| [`graph_select`](graph-select.md) | TVF | dbt-style node selector DSL |
| [`graph_adjacency`](api.md#graph_adjacency) | Virtual table | Persistent CSR adjacency cache with delta triggers |
| [`node2vec_train`](api.md#node2vec_train) | Scalar | Learn structural embeddings from graph topology |

### GGUF LLM inference

| Symbol | Kind | Purpose |
|--------|------|---------|
| [`muninn_chat`](api.md#muninn_chat) | Scalar | Free-form generation with optional GBNF grammar |
| [`muninn_chat_model`](api.md#muninn_chat_model) | Scalar | Register a GGUF chat model |
| [`muninn_chat_models`](api.md#muninn_chat_models) | Virtual table | Chat model lifecycle |
| [`muninn_extract_entities`](api.md#muninn_extract_entities), [`muninn_extract_relations`](api.md#muninn_extract_relations), [`muninn_extract_ner_re`](api.md#muninn_extract_ner_re) | Scalar | Grammar-constrained NER / RE / combined (supervised + unsupervised) |
| [`muninn_extract_entities_batch`](api.md#muninn_extract_entities_batch), [`muninn_extract_ner_re_batch`](api.md#muninn_extract_ner_re_batch) | Scalar | Multi-sequence batch variants |
| [`muninn_summarize`](api.md#muninn_summarize) | Scalar | Abstractive summarization |
| [`muninn_extract_er`](api.md#muninn_extract_er) | Scalar | End-to-end entity resolution (KNN → scoring → LLM → Leiden) |
| [`muninn_label_groups`](api.md#muninn_label_groups) | TVF | LLM-powered concise labels for arbitrary groupings |
| [`muninn_tokenize`, `muninn_tokenize_text`, `muninn_token_count`](api.md#tokenizers) | Scalar | Tokenization against any loaded model |

## By task

| I want to… | Read |
|------------|------|
| Install and verify the extension | [Getting Started](getting-started.md) |
| Embed text and run KNN search | [Text Embeddings](text-embeddings.md) |
| Extract entities, relations, or summaries from text | [Chat and Extraction](chat-and-extraction.md) |
| Deduplicate and link entities | [Entity Resolution](entity-resolution.md) |
| Run PageRank / Leiden / betweenness on an edge table | [Centrality and Community](centrality-community.md) |
| Query dbt-style graph lineage | [Graph Select](graph-select.md) |
| Learn structural node embeddings | [Node2Vec](node2vec.md) |
| Combine all of the above into a retrieval pipeline | [GraphRAG Cookbook](graphrag-cookbook.md) |
| Understand the internal architecture | [Architecture](architecture.md) |
| Look up any SQL symbol | [API Reference](api.md) |
| See performance data | [Benchmarks](benchmarks.md) |


## License and links

- [GitHub](https://github.com/neozenith/sqlite-muninn)
- [PyPI](https://pypi.org/project/sqlite-muninn/) · [npm](https://www.npmjs.com/package/sqlite-muninn)
- Licensed under the MIT License.
