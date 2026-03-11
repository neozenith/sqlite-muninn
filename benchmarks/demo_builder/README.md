# Demo Database Builder

10-phase pipeline that generates self-contained SQLite demo databases for the viz app. Each database contains chunks, FTS5 index, HNSW vector index, NER entities, relations, entity resolution, UMAP projections, and Node2Vec embeddings.

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

## Build Pipeline DAG

The pipeline is a directed acyclic graph with two parallel tracks branching from `chunks`:

```mermaid
flowchart TB
    TEXT[("Source Text<br/>Gutenberg / custom")]

    subgraph VEC ["Vector Path  ·  embedding + projection"]
        P1["1 · chunks<br/> chunks · chunks_fts"]
        P2["2 · chunks_embeddings<br/> chunks_vec HNSW"]
        P3["3 · chunks_umap<br/> chunks_vec_umap"]
    end

    subgraph NLP ["NLP Extraction  ·  NER + RE"]
        P4["4 · ner<br/> entities · ner_chunks_log"]
        P5["5 · relations<br/> relations · re_chunks_log"]
    end

    subgraph KG ["Graph Pipeline  ·  resolution + structural embeddings"]
        P6["6 · entity_embeddings<br/> entities_vec HNSW · entity_vec_map"]
        P7["7 · entities_umap<br/> entities_vec_umap"]
        P8["8 · entity_resolution<br/> entity_clusters · nodes · edges"]
        P9["9 · node2vec<br/> node2vec_emb HNSW"]
    end

    P10["10 · metadata<br/> meta"]

    TEXT --> P1
    P1  --> P2
    P2  --> P3
    P1  --> P4
    P4  --> P5
    P4  --> P6
    P6  --> P7
    P5  --> P8
    P6  --> P8
    P8  --> P9
    P3  --> P10
    P7  --> P10
    P9  --> P10

    classDef vec  fill:#1e3a5f,color:#e8f4fd,stroke:#3b82f6
    classDef nlp  fill:#3b1f4a,color:#f3e8ff,stroke:#a855f7
    classDef kg   fill:#14382a,color:#d1fae5,stroke:#10b981
    classDef term fill:#3d2000,color:#fef3c7,stroke:#f59e0b

    class P1,P2,P3 vec
    class P4,P5 nlp
    class P6,P7,P8,P9 kg
    class P10 term
```

- **Phase 1 forks**: `chunks` feeds both `chunks_embeddings` (vector path) and `ner` (NLP path) independently
- **UMAP phases are independent**: `chunks_umap` depends only on `chunks_embeddings`; `entities_umap` depends only on `entity_embeddings`
- **Phase 8 joins**: `entity_resolution` depends on both `relations` (P5) and `entity_embeddings` (P6)

## Build Phases

| # | Phase | Depends on | Description |
|---|-------|------------|-------------|
| 1 | **chunks** | source text | Split text into model-aware chunks, build FTS5 |
| 2 | **chunks_embeddings** | chunks | SentenceTransformer → HNSW vector index |
| 3 | **chunks_umap** | chunks_embeddings | UMAP 2D+3D projections for chunks |
| 4 | **ner** | chunks | Extract named entities (GLiNER2 / GLiNER / muninn) |
| 5 | **relations** | ner | Extract relations (GLiNER2 / GLiREL / muninn) |
| 6 | **entity_embeddings** | ner | SentenceTransformer → entity HNSW index |
| 7 | **entities_umap** | entity_embeddings | UMAP 2D+3D projections for entities |
| 8 | **entity_resolution** | relations + entity_embeddings | HNSW blocking + Jaro-Winkler + Leiden clustering |
| 9 | **node2vec** | entity_resolution | Node2Vec random walks + Skip-gram embeddings |
| 10 | **metadata** | chunks_umap + entities_umap + node2vec | Write meta table, validate all tables |

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
        blob vector "HNSW 768d cosine"
    }
    chunks_vec_umap {
        int id PK
        real x2d
        real y2d
        real x3d
        real y3d
        real z3d
    }
    entities {
        int entity_id PK
        text name
        text entity_type
        text source "gliner2 | gliner | muninn"
        int chunk_id FK
        real confidence
    }
    ner_chunks_log {
        int chunk_id PK
        text processed_at
    }
    relations {
        int relation_id PK
        text src
        text dst
        text rel_type
        real weight
        int chunk_id
        text source "gliner2 | glirel | muninn"
    }
    re_chunks_log {
        int chunk_id PK
        text processed_at
    }
    entity_vec_map {
        int rowid PK
        text name
    }
    entities_vec {
        int rowid PK
        blob vector "HNSW 768d cosine"
    }
    entities_vec_umap {
        int id PK
        real x2d
        real y2d
        real x3d
        real y3d
        real z3d
    }
    entity_clusters {
        text name PK
        text canonical
    }
    nodes {
        int node_id PK
        text name UK
        text entity_type
        int mention_count
    }
    edges {
        text src PK "composite PK"
        text dst PK "composite PK"
        text rel_type PK "composite PK"
        real weight
    }
    node2vec_emb {
        int rowid PK
        blob vector "HNSW 64d cosine"
    }
    meta {
        text key PK
        text value
    }

    chunks ||--|| chunks_vec : "chunk_id = rowid"
    chunks ||--|| chunks_vec_umap : "chunk_id = id"
    chunks ||--o{ entities : "chunk_id"
    chunks ||--o{ relations : "chunk_id"
    entity_vec_map ||--|| entities_vec : "rowid"
    entity_vec_map ||--|| entities_vec_umap : "rowid = id"
    entities }o--o{ entity_clusters : "name"
    entity_clusters }o--|| nodes : "canonical = name"
    nodes ||--o{ edges : "name = src/dst"
    nodes ||--|| node2vec_emb : "node_id = rowid"
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
