# GGUF ML Integration — From Composed Embeddings to Native Inference

Captured: 2026-02-20. Distilled from research session on embedding model formats, C inference libraries, and the GGUF ecosystem.

**Status:** Phase 0 ready for immediate documentation. Phases 1-3 are architectural plan.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Phase 0 — Compose with sqlite-lembed Today](#2-phase-0--compose-with-sqlite-lembed-today)
3. [Phase 1 — Vendor llama.cpp for GGUF Inference](#3-phase-1--vendor-llamacpp-for-gguf-inference)
4. [Phase 2 — Text-Aware Virtual Tables](#4-phase-2--text-aware-virtual-tables)
5. [Phase 3 — Composite Search Operations](#5-phase-3--composite-search-operations)
6. [GGUF Feature Surface](#6-gguf-feature-surface)
7. [Model Recommendations](#7-model-recommendations)
8. [Quantization Quality Reference](#8-quantization-quality-reference)
9. [Build System Integration](#9-build-system-integration)
10. [Research: Format Comparison](#10-research-format-comparison)
11. [Prior Art](#11-prior-art)
12. [Open Questions](#12-open-questions)

---

## 1. Vision

The primitives in this extension — HNSW vector search, graph traversal TVFs, centrality measures, community detection, Node2Vec — are individually robust and stand on their own. The transformative value is in their **composition**.

Consider a `knowledge_graph_search()` that:

1. **FTS5 + HNSW** — retrieves candidate documents via full-text and semantic similarity
2. **Reranker model** — applies a cross-encoder to score candidates precisely
3. **Graph seeding** — top-ranked nodes become starting points for graph traversal
4. **Sibling expansion** — BFS/DFS traverses related nodes via typed edges
5. **Centrality ranking** — PageRank or betweenness scores prioritize structurally important results
6. **Return** — a rich, context-aware result set that no single primitive could produce

This composition is only possible when we control all the primitives in a single extension. Bolting separate extensions together cannot achieve the shared-memory, zero-copy data flow that makes this performant.

### Identity Evolution

The project identity evolves from "zero-dependency C11 SQLite extension" to **"single-dependency C11 SQLite extension powered by llama.cpp"**. llama.cpp becomes the one key dependency, unlocking the full GGUF model ecosystem: embeddings, reranking, entity extraction, classification, and eventually multimodal inference — all from SQL.

The inspiration is sqlite-lembed, but extended beyond embeddings to the **full GGUF feature set for general ML in the database**.

---

## 2. Phase 0 — Compose with sqlite-lembed Today

**Effort:** Documentation only. Zero code changes to the extension.

Users can skip Python for embedding generation today by loading sqlite-lembed alongside this extension. This is worth documenting immediately as a supported workflow.

### Prerequisites

```bash
# Install sqlite-lembed (macOS/Linux)
pip install sqlite-lembed
# Or download from: https://github.com/asg017/sqlite-lembed/releases

# Download a GGUF embedding model
curl -L -o all-MiniLM-L6-v2.Q8_0.gguf \
  https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf
```

### Usage Pattern

```sql
-- Load both extensions
.load ./muninn
.load lembed0

-- Register the GGUF model (session-scoped)
INSERT INTO temp.lembed_models(name, model)
  SELECT 'MiniLM', lembed_model_from_file('all-MiniLM-L6-v2.Q8_0.gguf');

-- Create an HNSW index
CREATE VIRTUAL TABLE doc_vectors USING hnsw_index(
  dim=384, metric=cosine, max_elements=100000
);

-- Source table
CREATE TABLE documents(id INTEGER PRIMARY KEY, content TEXT);

-- Auto-embed via trigger
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO doc_vectors(rowid, vector)
    VALUES (NEW.id, lembed('MiniLM', NEW.content));
END;

-- Insert text — trigger handles embedding
INSERT INTO documents(content) VALUES ('The quick brown fox jumps over the lazy dog');
INSERT INTO documents(content) VALUES ('A lazy dog sleeps in the sun');

-- Semantic search — embed query inline
SELECT d.content, v.distance
FROM doc_vectors v
JOIN documents d ON d.id = v.rowid
WHERE v.vector MATCH lembed('MiniLM', 'fast animal')
ORDER BY v.distance
LIMIT 10;
```

### Blob Format Compatibility

Both extensions use the identical blob format: raw little-endian IEEE 754 float32 values, no header, `4 bytes x dim`. The output of `lembed()` inserts directly into `hnsw_index` with zero conversion.

### Trigger Considerations

| Trigger Type | Behavior |
|---|---|
| `CREATE TEMP TRIGGER` | Works perfectly. Extension only needed during the session. |
| `CREATE TRIGGER` (persistent) | Trigger body is compiled on `sqlite3_open()`. If lembed is not loaded before opening the database, schema loading fails. Requires `sqlite3_auto_extension()` in application code. |

**Recommendation:** Use TEMP triggers for interactive/ad-hoc usage. For applications, load both extensions via `sqlite3_auto_extension()` before opening any database file.

### Why sqlite-lembed Works in Triggers

sqlite-lembed registers its functions with `SQLITE_INNOCUOUS` (not `SQLITE_DIRECTONLY`), which explicitly permits use inside triggers, views, and generated columns. This was a deliberate design choice.

### Remote Embeddings (sqlite-rembed)

For API-based embedding (OpenAI, Nomic, Cohere, Ollama), sqlite-rembed provides the same pattern:

```sql
.load rembed0

INSERT INTO temp.rembed_clients(name, options) VALUES ('openai', 'openai');

-- Same trigger pattern, different embedding source
CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
BEGIN
  INSERT INTO doc_vectors(rowid, vector)
    VALUES (NEW.id, rembed('openai', NEW.content));
END;
```

### Phase 0 Limitations

- **No batch embedding** — each `lembed()` call is an independent inference pass
- **Two extensions to load** — users must manage both `.dylib`/`.so` files
- **CPU-only** in pre-built lembed binaries
- **Model management** is external — user must download and path GGUF files

---

## 3. Phase 1 — Vendor llama.cpp for GGUF Inference

**Effort:** Medium. New C wrapper code + build system changes.

### Architecture

```
src/
├── muninn.c              ← existing entry point, adds embed_register_functions()
├── hnsw_vtab.c           ← existing
├── hnsw_algo.c           ← existing
├── embed_gguf.c          ← NEW: llama.cpp wrapper (pure C11)
├── embed_gguf.h          ← NEW: internal API
└── ...

vendor/
└── llama.cpp/            ← git submodule, pinned to release tag
```

### C API Wrapper

The wrapper exposes a minimal C11 interface over llama.cpp's `extern "C"` API:

```c
// embed_gguf.h
typedef struct EmbedModel EmbedModel;

EmbedModel *embed_model_load(const char *gguf_path, int n_ctx);
void        embed_model_free(EmbedModel *model);
int         embed_model_dim(const EmbedModel *model);

// text in → float32 blob out. Returns dimension count, or -1 on error.
int embed_text(EmbedModel *model, const char *text, float *out, int max_dim);
```

### SQL Functions Registered

| Function | Signature | Returns | Purpose |
|----------|-----------|---------|---------|
| `muninn_embed(model, text)` | `(TEXT, TEXT) → BLOB` | float32 blob | Generate embedding |
| `muninn_embed_model(path)` | `(TEXT) → POINTER` | opaque model handle | Load GGUF file |
| `muninn_tokenize(model, text)` | `(TEXT, TEXT) → TEXT` | JSON array | Debug tokenization |
| `muninn_model_dim(model)` | `(TEXT) → INT` | dimension count | Query model metadata |

### Model Registration

Following sqlite-lembed's pattern:

```sql
-- Virtual table for loaded models (session-scoped)
INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

-- Generate embeddings
SELECT muninn_embed('MiniLM', 'hello world');
```

### Compilation

The C11 code compiles normally. Linking requires the C++ standard library for llama.cpp internals:

```makefile
# macOS
$(CC) -std=c11 -shared -o muninn.dylib src/*.c \
    $(LLAMA_LIBS) -I vendor/llama.cpp/include \
    -lc++ -framework Accelerate

# Linux
$(CC) -std=c11 -shared -o muninn.so src/*.c \
    $(LLAMA_LIBS) -I vendor/llama.cpp/include \
    -lstdc++ -lm -lpthread
```

### Binary Size Impact

| Build | Size |
|-------|------|
| Current muninn (no llama.cpp) | ~200 KB |
| With llama.cpp (CPU-only, MinSizeRel, stripped) | ~10-25 MB |
| With llama.cpp (CPU-only, Release -O2, stripped) | ~15-30 MB |

### What Phase 1 Enables

- Single extension — one `.load ./muninn` gives everything
- Internal batch embedding (build multi-sequence batches in C)
- Tokenization and model metadata from SQL
- Foundation for Phase 2 and 3

---

## 4. Phase 2 — Text-Aware Virtual Tables

**Effort:** Medium-high. New virtual table module.

### Proposed SQL Interface

```sql
-- Create a text-aware HNSW index
CREATE VIRTUAL TABLE docs USING hnsw_text(
  model='models/all-MiniLM-L6-v2.Q8_0.gguf',
  dim=384,
  metric=cosine,
  max_elements=100000
);

-- Insert text — embedding is automatic
INSERT INTO docs(rowid, content) VALUES (1, 'The quick brown fox');
INSERT INTO docs(rowid, content) VALUES (2, 'A lazy dog sleeps');

-- Search by text — query embedding is automatic
SELECT rowid, distance FROM docs
WHERE content MATCH 'fast animal'
LIMIT 10;

-- Matryoshka: use a lower dimension for faster search
CREATE VIRTUAL TABLE docs_compact USING hnsw_text(
  model='models/nomic-embed-text-v1.5.Q4_K_M.gguf',
  dim=128,  -- Matryoshka truncation from 768
  metric=cosine
);
```

### Shadow Tables

Extends the existing `_config`, `_nodes`, `_edges` pattern:

| Table | Contents |
|-------|----------|
| `docs_config` | dim, metric, M, efConstruction + model_path, model_dim |
| `docs_nodes` | rowid → vector blob (existing) |
| `docs_edges` | HNSW neighbor lists (existing) |
| `docs_content` | rowid → original text (NEW) |

### xUpdate Behavior

On `INSERT`: tokenize text → run inference → normalize → HNSW insert (existing path).
On `MATCH`: tokenize query → run inference → normalize → HNSW search (existing path).

The embedding step is invisible to the user — text goes in, semantic search comes out.

---

## 5. Phase 3 — Composite Search Operations

**Effort:** High. Novel SQL functions combining multiple subsystems.

### knowledge_graph_search()

```sql
SELECT * FROM knowledge_graph_search(
  query='economic impact of climate change',
  model='nomic-embed',
  top_k=5,           -- HNSW recall candidates
  reranker='jina-reranker',  -- optional cross-encoder
  expand_hops=2,     -- BFS hops from seed nodes
  edge_table='edges',
  rank_by='pagerank' -- centrality scoring
);
```

Internal pipeline:

1. `muninn_embed('nomic-embed', query)` → query vector
2. HNSW search → top_k candidate rowids
3. (Optional) reranker cross-scores query against candidate texts → re-sort
4. Seed top results as graph starting nodes
5. BFS/DFS expansion via `graph_tvf` internals → sibling nodes
6. Centrality scoring via `graph_centrality` internals → ranked result set
7. Return combined results with distance, rank, hop-depth

### entity_resolve()

```sql
SELECT * FROM entity_resolve(
  entity_table='entities',
  name_column='name',
  embedding_model='MiniLM',
  similarity_threshold=0.85,
  community_algorithm='leiden'
);
```

Combines: embedding similarity (HNSW) + community detection (Leiden) to cluster duplicate entities — a core knowledge graph construction primitive.

### rerank()

```sql
SELECT rowid, rerank_score FROM rerank(
  model='jina-reranker-v3',
  query='fast animal',
  candidate_table='search_results',
  text_column='content'
)
ORDER BY rerank_score DESC
LIMIT 10;
```

Uses llama.cpp's `--pooling rank` mode with GGUF cross-encoder models.

---

## 6. GGUF Feature Surface

The GGUF model format via llama.cpp unlocks capabilities far beyond text embeddings:

### Available Today (Mature in llama.cpp)

| Capability | Example Models | Use in Extension |
|---|---|---|
| **Text embeddings** | all-MiniLM-L6-v2, nomic-embed-text-v1.5, BGE, GTE, E5 | `muninn_embed()`, `hnsw_text` |
| **Multilingual embeddings** | BGE-M3, Qwen3-Embedding, jina-v3, LaBSE | Same API, different model file |
| **Code embeddings** | nomic-embed-code, jina-v4-text-code | Code search indexes |
| **Cross-encoder reranking** | jina-reranker-v3, Qwen3-Reranker | `rerank()` TVF |
| **Matryoshka dimensions** | nomic-v1.5, Qwen3, jina-v3 | Truncate to any dim for size/speed tradeoff |
| **Quantization** | Q2 through F32 on all models | Smaller files, faster inference |

### Emerging (Requires llama.cpp Patches)

| Capability | Status | Use in Extension |
|---|---|---|
| **Multimodal embeddings** | jina-v4 (requires Jina's llama.cpp fork) | Image+text search |
| **ColBERT multi-vector** | BGE-M3 supports it; GGUF impl incomplete | Fine-grained retrieval |

### Not Yet Available in GGUF

| Capability | Alternative |
|---|---|
| Named Entity Recognition | Use generative LLM GGUF models with structured output |
| Text classification | Embedding + external classifier head |
| Standalone CLIP/SigLIP | Only via jina-v4's integrated vision encoder |

---

## 7. Model Recommendations

### By Use Case

| Use Case | Model | Quant | File Size | Dims | Max Context |
|----------|-------|-------|-----------|------|-------------|
| Tiny/embedded/mobile | all-MiniLM-L6-v2 | Q8_0 | 36 MB | 384 | 256 |
| General purpose | nomic-embed-text-v1.5 | Q4_K_M | 84 MB | 128-768 (MRL) | 8192 |
| Multilingual | BGE-M3 | Q4_K_M | 438 MB | 1024 | 8192 |
| Multilingual + Matryoshka | Qwen3-Embedding-0.6B | Q4_K_M | ~420 MB | 32-1024 (MRL) | 32K |
| Code search | nomic-embed-code | Q4_K_M | 4.08 GB | — | — |
| Reranking | jina-reranker-v3 | Q8_0 | 640 MB | — | — |
| High quality (English) | mxbai-embed-large-v1 | Q4_K_M | 216 MB | 1024 | 512 |

### Matryoshka Memory Impact

For HNSW indexes where memory scales as ~(dim x 4 + 356) bytes/node:

| Dim | Bytes/Node | Vectors in 8 GB |
|-----|-----------|----------------|
| 64 | 612 | ~13.7M |
| 128 | 868 | ~9.6M |
| 256 | 1,380 | ~6.1M |
| 384 | 1,892 | ~4.4M |
| 768 | 3,428 | ~2.4M |
| 1024 | 4,452 | ~1.9M |

Using Matryoshka (e.g., nomic-embed-text-v1.5 at dim=128 instead of 768) gives a **4x capacity increase** with the same model file.

---

## 8. Quantization Quality Reference

Published MSE measurements from nomic-embed-text-v1.5 GGUF against FP32 reference:

| Quantization | File Size | MSE vs FP32 | Verdict |
|---|---|---|---|
| F32 | 548 MB | baseline | Reference |
| F16 | 274 MB | 4.21e-10 | Identical |
| Q8_0 | 146 MB | 5.79e-06 | Indistinguishable |
| Q6_K | 113 MB | 5.58e-05 | Extremely low loss |
| Q5_K_M | 99.6 MB | 6.55e-05 | Very low loss |
| **Q4_K_M** | **84 MB** | **2.42e-04** | **Recommended sweet spot** |
| Q4_0 | 77.8 MB | 6.32e-04 | Noticeable but usable |
| Q2_K | 49.4 MB | 2.33e-03 | Significant degradation |

**Takeaway:** Q8_0 is effectively lossless for retrieval. Q4_K_M offers the best size/quality tradeoff and should be the default recommendation.

---

## 9. Build System Integration

### Directory Layout

```
sqlite-vector-graph/
├── vendor/
│   └── llama.cpp/            ← git submodule, pinned to release tag (e.g., b4464)
├── src/
│   ├── muninn.c              ← adds embed_register_functions() call
│   ├── embed_gguf.c          ← NEW: llama.cpp C11 wrapper
│   ├── embed_gguf.h          ← NEW: internal API
│   └── ...                   ← existing source files
└── Makefile                  ← updated with llama.cpp build targets
```

### Makefile Targets

```makefile
LLAMA_DIR     = vendor/llama.cpp
LLAMA_BUILD   = $(LLAMA_DIR)/build
LLAMA_LIBS    = $(LLAMA_BUILD)/src/libllama.a \
                $(LLAMA_BUILD)/ggml/src/libggml-base.a \
                $(LLAMA_BUILD)/ggml/src/ggml-cpu/libggml-cpu.a
LLAMA_INCLUDE = -I$(LLAMA_DIR)/include

# One-time build of llama.cpp static libraries
$(LLAMA_LIBS):
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) \
	    -DBUILD_SHARED_LIBS=OFF \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	    -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
	    -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
	    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
	    -DLLAMA_BUILD_SERVER=OFF \
	    -DCMAKE_BUILD_TYPE=MinSizeRel
	cmake --build $(LLAMA_BUILD) --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Extension with GGUF support
muninn.dylib: src/*.c $(LLAMA_LIBS)
	$(CC) -std=c11 -shared -fPIC -O2 -o $@ src/*.c \
	    $(LLAMA_INCLUDE) $(LLAMA_LIBS) \
	    $(if $(findstring Darwin,$(shell uname)),-lc++ -framework Accelerate,-lstdc++ -lm -lpthread)
```

### API Stability Note

llama.cpp renames public API functions frequently. Key renames (2024-2025):

| Old | New | When |
|-----|-----|------|
| `llama_load_model_from_file` | `llama_model_load_from_file` | PR #10784 |
| `llama_new_context_with_model` | `llama_init_from_model` | PR #10784 |
| `llama_n_embd` | `llama_model_n_embd` | PR #10784 |
| `llama_kv_cache_clear` | removed | PR #15472 |

**Mitigation:** Pin the submodule to a specific tagged release. Test against that version in CI. Update deliberately.

---

## 10. Research: Format Comparison

Summary of the model format research that informed this plan:

| Format | Pure C API? | Runtime Size | Tokenizer? | Model Size (MiniLM) | Verdict |
|---|---|---|---|---|---|
| **GGUF/ggml** | Yes (via llama.cpp extern C) | < 1 MB (ggml) / 10-15 MB (llama.cpp) | Yes (in GGUF file) | 36 MB (Q8_0) | **Selected** |
| ONNX | C99 via ORT | 15-30 MB | No | ~90 MB (FP32) | Too heavy, no tokenizer |
| SafeTensors | Single-header C reader | 0 MB (reader only) | No | ~90 MB (FP32) | Interesting for pure-C path but requires hand-written model arch |
| TFLite | Pure C | 3-10 MB | No | Conversion needed | Weak model ecosystem |
| TorchScript | C++ only | 200+ MB | No | — | Far too heavy |
| OpenVINO | C API | 100+ MB | No | — | Intel-only, too heavy |

### Why GGUF Won

1. **Tokenizer embedded in the model file** — no separate tokenizer dependency
2. **4-bit quantization** — 14-36 MB embedding models vs 90 MB+ for ONNX/SafeTensors FP32
3. **Proven in SQLite context** — sqlite-lembed demonstrates the exact integration pattern
4. **Broadest model ecosystem** — 20+ embedding models, rerankers, multilingual, code
5. **Single-file models** — one `.gguf` file contains weights + tokenizer + metadata
6. **Active ecosystem** — llama.cpp is the dominant open-source inference project

### Why Not Pure C (SafeTensors + Custom Transformer)

The antirez/gte-pure-C project proved a zero-dependency pure C path is feasible (~700 lines for one model architecture). However:

- Each model architecture variant requires its own C implementation
- No quantization without porting ggml's quantization kernels
- No tokenizer — must implement WordPiece/BPE/SentencePiece separately
- Model files are 5-6x larger without quantization
- Cannot support the full GGUF ecosystem (rerankers, code models, multilingual)

The llama.cpp dependency buys vastly more capability for the same integration effort.

---

## 11. Prior Art

| Project | Approach | Relation to This Plan |
|---|---|---|
| **sqlite-lembed** (Alex Garcia) | C wrapper around llama.cpp for `lembed()` scalar function. GGUF models. Apache-2.0/MIT. | Direct inspiration. Phase 0 composes with it. Phase 1 internalizes the same pattern. |
| **sqlite-rembed** (Alex Garcia) | Rust extension calling remote embedding APIs (OpenAI, Ollama, etc.) | Phase 0 alternative for API-based embedding. |
| **sqlite-ai** (SQLite Cloud) | C extension wrapping llama.cpp for text generation + embeddings. | Similar approach but broader scope (chat + generation). Elastic License (not OSS). |
| **antirez/gte-pure-C** | ~700 lines of pure C implementing GTE-Small BERT forward pass. Loads SafeTensors weights. | Proved pure-C embedding inference is feasible. We chose GGUF for the broader ecosystem. |
| **bert.cpp / embeddings.cpp** | ggml-based BERT inference with built-in tokenizers. | Predecessors to llama.cpp's BERT support. Now largely superseded. |
| **PostgresML pgml.embed()** | Rust PostgreSQL extension running HuggingFace models inside the database. | Same "ML in the database" philosophy, but for PostgreSQL. AGPL licensed. |
| **SQL Server 2025** | ONNX Runtime + tokenizers-cpp for `AI_GENERATE_EMBEDDINGS()`. | Microsoft's approach. Commercial, Windows-only, ONNX-based. |

---

## 12. Open Questions

1. **Feature flag or always-on?** Should llama.cpp be a compile-time option (`#ifdef MUNINN_GGUF`) or always included? A feature flag keeps the minimal build small (~200 KB) but complicates the build matrix.

2. **Model management UX.** Should models be registered via virtual table (sqlite-lembed pattern), configuration pragma, or path in `CREATE VIRTUAL TABLE`? The virtual table pattern is proven but requires a separate registration step.

3. **Batch embedding.** llama.cpp supports multi-sequence batches. Should `hnsw_text` INSERT automatically batch when given multiple rows, or should there be an explicit `muninn_embed_batch()` function?

4. **Metal/GPU support.** The plan disables GPU backends for simplicity. Should there be a separate build target for Metal (macOS) or CUDA (Linux) acceleration? This would dramatically speed up embedding generation for bulk operations.

5. **llama.cpp version policy.** How often do we update the pinned llama.cpp version? The API changes frequently. Options: (a) pin to LTS-like releases, (b) update quarterly, (c) maintain a compatibility shim.

6. **Reranker integration.** Reranking requires `LLAMA_POOLING_TYPE_RANK` which produces a single score, not a vector. The SQL interface for reranking is fundamentally different from embedding. Should it be a separate TVF or a mode of the same function?

7. **Chunking.** sqlite-lembed provides `lembed_chunks()` for splitting long text into model-sized pieces. Do we replicate this, or leave chunking to the application layer?

8. **Amalgamation.** The current extension supports single-file amalgamation. Can llama.cpp be amalgamated, or does the GGUF build require separate compilation?
