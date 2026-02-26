# Demo Database Builder

8-phase pipeline that generates self-contained SQLite demo databases for the viz app. Each database contains chunks, FTS5 index, HNSW vector index, NER entities, relations, entity resolution, UMAP projections, and Node2Vec embeddings.

## CLI Usage

```bash
# List available books and embedding models
uv run -m benchmarks.demo_builder list-books
uv run -m benchmarks.demo_builder list-models

# Show build status matrix (all book × model permutations)
uv run -m benchmarks.demo_builder manifest

# Generate runnable build commands for missing permutations
uv run -m benchmarks.demo_builder manifest --missing --commands

# Build a single demo database
uv run -m benchmarks.demo_builder build --book-id 3300 --embedding-model MiniLM

# Regenerate manifest.json from existing built DBs
uv run -m benchmarks.demo_builder write-manifest

# Override output folder (default: viz/frontend/public/demos)
uv run -m benchmarks.demo_builder build --output-folder /tmp/demos --book-id 3300 --embedding-model MiniLM
```

## Build Phases

| # | Phase | Description |
|---|-------|-------------|
| 1 | chunks+fts+embeddings | Read raw text, split into model-aware chunks, build FTS5, compute/load embeddings, insert into HNSW |
| 2 | ner | Extract named entities with GLiNER |
| 3 | relations | Extract relations with GLiREL |
| 4 | entity_embeddings | Compute entity name embeddings |
| 5 | umap | UMAP 2D+3D projections for chunks and entities |
| 6 | entity_resolution | HNSW blocking + Jaro-Winkler + Leiden clustering |
| 7 | node2vec | Node2Vec random walks + Skip-gram embeddings |
| 8 | metadata+validation | Write meta table, validate all tables |

## Database Schema

```mermaid
erDiagram
    chunks {
        int chunk_id PK
        text text
    }
    chunks_fts {
        text text "FTS5 content=chunks"
    }
    chunks_vec {
        int rowid PK
        blob vector "HNSW float32"
    }
    entities {
        int entity_id PK
        int chunk_id FK
        text name
        text entity_type
        int start_char
        int end_char
    }
    relations {
        int relation_id PK
        int chunk_id FK
        text src
        text rel_type
        text dst
        real score
    }
    entity_clusters {
        text name
        int cluster_id
    }
    nodes {
        int node_id PK
        text name
        text entity_type
        int mention_count
    }
    edges {
        int edge_id PK
        text src
        text dst
        text rel_type
        real weight
    }
    entities_vec {
        int rowid PK
        blob vector "HNSW float32"
    }
    node2vec_emb {
        int rowid PK
        blob vector "HNSW float32"
    }
    chunks_vec_umap {
        int chunk_id PK
        real x2d
        real y2d
        real x3d
        real y3d
        real z3d
    }
    entities_vec_umap {
        int entity_id PK
        real x2d
        real y2d
        real x3d
        real y3d
        real z3d
    }
    meta {
        text key PK
        text value
    }

    chunks ||--o{ entities : "chunk_id"
    chunks ||--o{ relations : "chunk_id"
    entities }o--o{ entity_clusters : "name"
    entity_clusters }o--|| nodes : "cluster → node"
    nodes ||--o{ edges : "src/dst"
```

## Prerequisites

```bash
# Build the muninn extension
make all

# Download Gutenberg texts
uv run -m benchmarks.harness prep texts

# Install ML dependencies
uv pip install gliner glirel spacy sentence-transformers umap-learn "numpy>=2.0,<2.4"
python -m spacy download en_core_web_lg
```

## Model-Aware Chunking

Chunk sizes are determined by the **tightest constraint** across all models in the pipeline. Each chunk passes through embedding, NER (GLiNER), and RE (GLiREL) models:

| Model | Max | Unit | Constraint |
|-------|-----|------|------------|
| MiniLM | 256 | subword tokens | 768 chars |
| NomicEmbed | 8,192 | subword tokens | 4,096 chars |
| GLiNER medium-v2.1 | 384 | word tokens | ~1,920 chars |
| GLiREL large-v0 | 384 | word tokens | ~1,920 chars |

The effective chunk size is `min(embedding_model_chars, NER_RE_cap)`:

| Embedding Model | Model Chars | NER/RE Cap | Effective |
|-----------------|-------------|------------|-----------|
| MiniLM | 768 | 1,920 | **768** (embedding is tighter) |
| NomicEmbed | 4,096 | 1,920 | **1,920** (NER/RE is tighter) |

Without this cap, NomicEmbed chunks (~725 word tokens) would be silently truncated by GLiNER/GLiREL at 384 tokens, losing entities and relations in the second half of each chunk.

Chunks overlap by ~10% and snap to sentence boundaries to avoid splitting mid-sentence.

## Output

Built databases are written to `viz/frontend/public/demos/` by default with a `manifest.json` index file:

```json
{
  "databases": [
    {
      "id": "3300_MiniLM",
      "book_id": 3300,
      "model": "MiniLM",
      "dim": 384,
      "file": "3300_MiniLM.db",
      "size_bytes": 32059392,
      "label": "Book 3300 + MiniLM (384d)"
    }
  ]
}
```
