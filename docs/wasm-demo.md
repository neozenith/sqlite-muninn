# WASM Demo

An interactive demo of the **muninn** SQLite extension running entirely in your browser via WebAssembly.

<a href="../examples/wasm/" target="_blank" class="md-button md-button--primary">
  Launch WASM Demo
</a>

## What It Does

The demo loads a pre-built **Wealth of Nations** knowledge graph (~3,300 text chunks) into an in-browser SQLite instance compiled to WASM, then lets you explore it through three synchronized panels:

| Panel | Technology | muninn Subsystem |
|-------|-----------|-----------------|
| **FTS Results** (left) | SQLite FTS5 | Full-text search |
| **Embedding Space** (center) | Deck.GL 3D + Transformers.js | HNSW vector search with pre-calculated UMAP coordinates |
| **Knowledge Graph** (right) | Cytoscape.js | Graph BFS traversal via `graph_bfs` TVF |

## How It Works

1. **WASM + SQLite** — The muninn C extension is compiled alongside SQLite into a single `.wasm` binary using Emscripten. All five subsystems (HNSW, graph TVFs, centrality, community detection, Node2Vec) are available.

2. **In-browser embeddings** — [Transformers.js](https://huggingface.co/docs/transformers.js) loads the `Xenova/all-MiniLM-L6-v2` model (384-dim, fp32) to generate query embeddings directly in the browser.

3. **Single CTE query** — A search triggers a CTE pipeline: vector similarity search (HNSW) finds the closest entity, then `graph_bfs` expands 1-hop neighbors, all scored by cosine similarity.

!!! note "First Load"
    The first search downloads the ~90MB fp32 embedding model. Subsequent searches use the browser cache and are near-instant.

## Building Locally

```bash
# Build the WASM module (requires Emscripten SDK)
make -C wasm build

# Build docs with the demo included
make docs-build

# Or serve locally
make docs-serve
```

The demo is also available standalone via `make -C wasm dev` on port 8300.
