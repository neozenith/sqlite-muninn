# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A C11 SQLite extension (`muninn`) combining HNSW vector search, graph traversal TVFs, centrality measures, Leiden community detection, Node2Vec embeddings, GGUF embedding/reranking, and GGUF LLM chat/extraction — in a single shared library. Loaded via `.load ./muninn` in SQLite. Uses llama.cpp (vendored) for ML inference and yyjson (vendored) for JSON processing.

## Build & Test Commands

```bash
make all            # Build muninn.dylib/.so/.dll (Metal GPU enabled on macOS)
make debug          # Build with ASan + UBSan (for debugging)
make test           # Run C unit tests (builds test_runner, then executes)
make test-python    # Run Python integration tests (requires .venv with pytest)
make test-all       # Both C and Python tests
make clean          # Remove muninn extension and test_runner binaries
```

**Prerequisites:** SQLite dev headers (`brew install sqlite` on macOS, `libsqlite3-dev` on Linux) and a C11 compiler.

**Python tests** expect a virtualenv at `.venv/` with pytest. The `conftest.py` auto-builds the extension before running tests.

**Benchmarks:** `make -C benchmarks help` to see all benchmark targets.

## Architecture

### Extension Entry Point

`src/muninn.c` — The `sqlite3_muninn_init` function registers eight subsystems with SQLite:

1. **`hnsw_register_module`** — the `hnsw_index` virtual table (vector similarity search)
2. **`graph_register_tvfs`** — graph traversal table-valued functions (BFS, DFS, shortest path, components, PageRank)
3. **`centrality_register_tvfs`** — centrality measures (`graph_degree`, `graph_betweenness`, `graph_closeness`)
4. **`community_register_tvfs`** — community detection (`graph_leiden`)
5. **`node2vec_register_functions`** — the `node2vec_train()` scalar function
6. **`gii_register_module`** — Graph Incremental Index (`USING gii(...)` virtual table)
7. **`embed_register_functions`** — GGUF embedding/reranking (`muninn_embed()`, `muninn_embed_model()`, `muninn_models` VT)
8. **`chat_register_functions`** — GGUF LLM chat/extraction (`muninn_chat()`, `muninn_extract_*()`, `muninn_chat_models` VT)

### Module Layering

```
hnsw_vtab.c  ── SQLite virtual table glue (xCreate, xFilter, xUpdate, shadow tables)
    └── hnsw_algo.c  ── Core HNSW algorithm (insert, search, delete, neighbor selection)
            ├── vec_math.c  ── SIMD-accelerated distance functions (L2, cosine, inner product)
            └── priority_queue.c  ── Binary min-heap for beam search

graph_tvf.c  ── Graph TVFs (BFS, DFS, shortest path, components, PageRank)
    └── id_validate.c  ── SQL identifier validation (anti-injection for dynamic table/column names)

graph_centrality.c  ── Centrality TVFs (degree, betweenness, closeness)
    └── graph_load.c  ── Shared graph loading: hash-map node lookup, weighted edges, temporal filtering

graph_community.c  ── Community detection TVFs (Leiden algorithm)
    └── graph_load.c

gii.c  ── Graph Incremental Index (delta cascade: _delta → _sssp_delta → _comp_delta → _comm_delta)

node2vec.c  ── Node2Vec random walks + Skip-gram with Negative Sampling

llama_embed.c  ── GGUF embedding/reranking via llama.cpp (Metal GPU accelerated)
    └── Uses llama.h pooled embeddings, model-native pooling (BERT→MEAN, Qwen3→LAST)

llama_chat.c  ── GGUF LLM chat/extraction via llama.cpp (Metal GPU accelerated)
    ├── GBNF grammar-constrained generation (NER, RE, NER+RE grammars)
    ├── Batch multi-sequence inference via llama_batch
    └── yyjson for JSON validation/minification of grammar-constrained output
```

### Build System

`scripts/generate_build.py` is the **single source of truth** for build configuration:
- Source/header discovery and dependency-sorted ordering
- Platform detection (Darwin/Linux/Windows) with Metal GPU, BLAS, linker flags
- CMake flags for llama.cpp (base + platform overrides)
- Queried by Makefile via `$(shell uv run scripts/generate_build.py query VAR)`
- Also generates: Windows `.bat` build script, amalgamation (`dist/muninn.c`), npm sub-packages

### Vendored Dependencies

- **llama.cpp** (`vendor/llama.cpp`, git submodule): ML inference engine, built as static libraries via CMake
  - macOS: Metal GPU enabled (`GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`), ~2.5x speedup
  - `MUNINN_DEFAULT_GPU_LAYERS=99` on macOS (all layers to GPU), overridable via `MUNINN_GPU_LAYERS` env var
- **yyjson** (`vendor/yyjson`, v0.10.0): JSON validation/minification in llama_embed.c and llama_chat.c

### Key Design Patterns

- `hnsw_vtab.c` is the SQLite integration layer managing shadow tables (`_config`, `_nodes`, `_edges`); delegates to pure-algorithmic `hnsw_algo.c` (SQLite-free, testable in isolation)
- **Graph TVFs** operate on *any* existing SQLite table with source/destination columns — no HNSW required. Table/column names validated via `id_validate.c` to prevent SQL injection
- **Node2Vec** bridges both subsystems: reads graph structure from an edge table, writes embeddings into an HNSW virtual table
- **llama_chat.c** uses embedded GBNF grammars for guaranteed well-formed JSON output from LLM extraction functions. Grammar sampler rejects invalid tokens during generation
- **Batch inference** (`muninn_extract_ner_re_batch`) uses `llama_batch` multi-sequence processing. Each prompt gets a unique `seq_id` in the KV cache

### HNSW Node Storage

Open-addressing hash table (`nodes` in `HnswIndex`) mapping `int64_t` IDs to `HnswNode` structs. Each node stores its vector, level, per-level neighbor lists, and a soft-delete flag. Auto-resizes with power-of-2 capacity.

### Testing Structure

- **C unit tests** (`test/`): Custom framework in `test_common.h` with `ASSERT`, `ASSERT_EQ_INT`, `ASSERT_EQ_FLOAT`, `RUN_TEST`, `TEST` macros. Add new suites via `test/test_<module>.c` + extern in `test_main.c` + Makefile `TEST_SRC`.
- **Python integration tests** (`pytests/`): pytest-based via `sqlite3.load_extension()`. The `conn` fixture provides fresh in-memory SQLite with muninn loaded.

### Vector Format

Vectors are passed as raw `float32` blobs (little-endian, `sizeof(float) * dim` bytes). In Python: `struct.pack(f'{dim}f', *values)`.
