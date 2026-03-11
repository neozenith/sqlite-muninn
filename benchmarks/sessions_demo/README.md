# Sessions Demo Builder

11-phase pipeline that builds a sessions demo SQLite database from Claude Code JSONL session logs (`~/.claude/projects/**/*.jsonl`). The output DB is fully compatible with the `viz/` frontend: it produces the same tables as `demo_builder` (chunks, entities, relations, nodes, edges, UMAP projections, node2vec embeddings) and is registered in `manifest.json` alongside other demo DBs.

## CLI Usage

```bash
# Full incremental build (only stale phases run)
uv run -m benchmarks.sessions_demo build

# Show build status and pending work without running anything
uv run -m benchmarks.sessions_demo build --status

# Custom output location
uv run -m benchmarks.sessions_demo --output-folder /tmp/demos build

# Cache management (ingest schema only, no ML)
uv run -m benchmarks.sessions_demo cache init
uv run -m benchmarks.sessions_demo cache update
uv run -m benchmarks.sessions_demo cache rebuild
uv run -m benchmarks.sessions_demo cache clear
uv run -m benchmarks.sessions_demo cache status

# Verbose logging
uv run -m benchmarks.sessions_demo -v build
```

## Build Pipeline DAG

The pipeline is a directed acyclic graph, not a linear sequence. The key structural points are:

- **Phase 2 fans out 3-way**: `chunks` feeds `chunks_vec` (vector path), `ner`, and `relations` (NLP paths) independently — all three can run in parallel
- **UMAP phases are independent**: `chunks_vec_umap` depends only on `chunks_vec`; `entities_vec_umap` depends only on `entity_embeddings` — neither depends on the other
- **Phase 9 joins**: `entity_resolution` depends on both `relations` (P6) and `entity_embeddings` (P7)

```mermaid
flowchart TB
    JSONL[("JSONL Files<br/>~/.claude/projects/**/*.jsonl")]

    subgraph PRE ["Pre-KG  ·  always incremental"]
        P1["1 · ingest<br/> events · event_edges · projects · sessions"]
        P2["2 · chunks<br/> event_message_chunks · chunks · chunks_fts"]
        P3["3 · embeddings<br/> chunks_vec  768d HNSW"]
        P4["4 · chunks_vec_umap<br/> chunks_vec_umap"]
    end

    subgraph NLP ["NLP Extraction  ·  incremental via log tables"]
        P5["5 · ner<br/> entities · ner_chunks_log"]
        P6["6 · relations<br/> relations · re_chunks_log"]
    end

    subgraph KG ["Graph Pipeline  ·  self-managing rebuild"]
        P7["7 · entity_embeddings<br/> entities_vec  768d HNSW · entity_vec_map"]
        P8["8 · entities_vec_umap<br/> entities_vec_umap"]
        P9["9 · entity_resolution<br/> entity_clusters · nodes · edges"]
        P10["10 · node2vec<br/> node2vec_emb HNSW"]
    end

    P11["11 · metadata<br/> meta"]

    JSONL --> P1
    P1  --> P2
    P2  --> P3
    P3  --> P4
    P2  --> P5
    P2  --> P6
    P5  --> P7
    P7  --> P8
    P6  --> P9
    P7  --> P9
    P9  --> P10
    P4  --> P11
    P8  --> P11
    P10 --> P11

    classDef pre   fill:#1e3a5f,color:#e8f4fd,stroke:#3b82f6
    classDef nlp   fill:#3b1f4a,color:#f3e8ff,stroke:#a855f7
    classDef kg    fill:#14382a,color:#d1fae5,stroke:#10b981
    classDef term  fill:#3d2000,color:#fef3c7,stroke:#f59e0b

    class P1,P2,P3,P4 pre
    class P5,P6 nlp
    class P7,P8,P9,P10 kg
    class P11 term
```

## Build Phases

Each phase tracks its own staleness with `is_stale(conn)` — only stale phases execute. Up-to-date phases restore their context fields from the DB and are skipped entirely.

| # | Phase | Depends on | Staleness check | Outputs |
|---|-------|------------|-----------------|---------|
| 1 | **ingest** | JSONL files | Files changed on disk since `source_files.mtime` | `events`, `event_edges`, `events_fts`, `projects`, `sessions` |
| 2 | **chunks** | ingest | Events with content `NOT IN event_message_chunks` | `event_message_chunks`, `chunks`, `chunks_fts` |
| 3 | **embeddings** | chunks | `chunk_id NOT IN chunks_vec_nodes` | `chunks_vec` (HNSW 768d) |
| 4 | **chunks_vec_umap** | embeddings | `chunks_vec_umap` count ≠ `chunks_vec_nodes` count | `chunks_vec_umap`, `*_chunks_umap*.joblib` |
| 5 | **ner** | chunks | Chunks `NOT IN ner_chunks_log` | `entities`, `ner_chunks_log` |
| 6 | **relations** | chunks | Chunks `NOT IN re_chunks_log` | `relations`, `re_chunks_log` |
| 7 | **entity_embeddings** | ner | Entity names `NOT IN entity_vec_map` | `entities_vec` (HNSW 768d), `entity_vec_map` |
| 8 | **entities_vec_umap** | entity_embeddings | `entities_vec_umap` count ≠ `entity_vec_map` count | `entities_vec_umap`, `*_entities_umap*.joblib` |
| 9 | **entity_resolution** | relations + entity_embeddings | `entity_clusters` count < distinct entity names | `entity_clusters`, `nodes`, `edges` |
| 10 | **node2vec** | entity_resolution | `node2vec_emb` count ≠ `nodes` count | `node2vec_emb` (HNSW) |
| 11 | **metadata** | all | Always re-runs (cheap count aggregation) | `meta` |

### Incrementality model

| Phase group | Strategy |
|-------------|----------|
| ingest, chunks, embeddings | Fully incremental — each run only processes new data |
| chunks_vec_umap, entities_vec_umap | Fit-once + `transform()` — independent joblib models, each reused for new vectors |
| ner, relations | Fully incremental via `*_chunks_log` tracking tables |
| entity_embeddings | Incremental — embeds only entity names not yet in `entity_vec_map` |
| entity_resolution, node2vec | Self-managing full rebuild — drops and recreates their own tables |
| metadata | Always re-runs (trivial `SELECT count(*)` aggregation) |

## Database Schema

The output DB is split into two logical layers: the **session layer** (events, chunks, embeddings) and the **KG layer** (entities, relations, graph, UMAP, node2vec).

### Session Layer

```mermaid
erDiagram
    source_files {
        int id PK
        text filepath UK
        real mtime
        int size_bytes
        int line_count
        text last_ingested_at
        text project_id
        text session_id
        text file_type "main_session | subagent | agent_root"
    }
    projects {
        int id PK
        text project_id UK
        text first_activity
        text last_activity
        int session_count
        int event_count
    }
    sessions {
        int id PK
        text session_id
        text project_id
        text first_timestamp
        text last_timestamp
        int event_count
        int subagent_count
        int total_input_tokens
        int total_output_tokens
        int total_cache_read_tokens
        int total_cache_creation_tokens
        real total_cost_usd
    }
    events {
        int id PK
        text uuid
        text parent_uuid
        text fqn_id "project::session::uuid"
        text event_type
        text timestamp
        text timestamp_local
        text session_id
        text project_id
        int is_sidechain
        text agent_id
        text agent_slug
        text message_role
        int is_meta
        text first_content_block_type
        text message_content
        text message_content_json
        text model_id
        int input_tokens
        int output_tokens
        int cache_read_tokens
        int cache_creation_tokens
        int cache_5m_tokens
        int source_file_id FK
        int line_number
        text raw_json
    }
    event_edges {
        int id PK
        text project_id
        text session_id
        text event_uuid
        text parent_event_uuid
        text fqn_src "project::session::parent_uuid"
        text fqn_dst "project::session::uuid"
        int source_file_id FK
    }
    event_message_chunks {
        int chunk_id PK
        int event_id FK
        text text
        int chunk_offset
    }
    chunks {
        int chunk_id PK
        text text
    }

    source_files ||--o{ events : "source_file_id"
    source_files ||--o{ event_edges : "source_file_id"
    projects ||--o{ sessions : "project_id"
    sessions ||--o{ events : "session_id"
    events ||--o{ event_edges : "uuid"
    events ||--o{ event_message_chunks : "event_id"
    event_message_chunks ||--|| chunks : "chunk_id"
```

FTS5 virtual tables (`events_fts`, `event_message_chunks_fts`, `chunks_fts`) mirror their content tables for full-text search. Internal tracking tables (`cache_metadata`, `_build_progress`) manage incremental build state.

### KG Layer

```mermaid
erDiagram
    chunks {
        int chunk_id PK
        text text
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

## Fully Qualified Names (FQN)

Event UUIDs are only unique within a session. To enable cross-session graph traversal, every event and edge gets a globally unique FQN:

```
{project_id}::{session_id}::{event_uuid}
```

- `events.fqn_id` — globally unique node identifier
- `event_edges.fqn_src` — parent node FQN (edge source)
- `event_edges.fqn_dst` — child node FQN (edge destination)

All FQN columns are indexed for fast graph lookups.

## Chunking Strategy

Chunk size is constrained by the smallest model window in the pipeline:

| Model | Max Tokens | Token type | Max chars | Used by |
|-------|-----------|------------|-----------|---------|
| NomicEmbed v1.5 (GGUF) | 2,048 | subword | 1,500 (truncated) | embeddings |
| GLiNER medium-v2.1 | 384 | word | ~1,920 | ner |
| GLiREL large-v0 | 384 | word | ~1,920 | relations |

Chunks are split at **1,920 chars** by the chunks phase. Before embedding, text is truncated to **1,500 chars** (`EMBED_MAX_CHARS`) — Claude Code session logs are code-heavy and NomicEmbed's subword tokenizer encodes code at ~1.3 tokens/char, which can push 1,920-char chunks past the 2,048-token context window.

## Prerequisites

```bash
# Build the muninn C extension (includes embed_gguf + hnsw subsystems)
make all

# Install Python ML dependencies
uv pip install gliner glirel sentence-transformers umap-learn joblib numpy

# Download spaCy model for NER fallback
python -m spacy download en_core_web_sm

# GGUF model must exist at:
# models/nomic-embed-text-v1.5.Q8_0.gguf
```

## Output

Built databases are written to `viz/frontend/public/demos/` by default, where the viz frontend auto-discovers them via `manifest.json`. The DB is registered with ID `sessions_demo` and label `Claude Code Sessions (768d)`.

```
viz/frontend/public/demos/
├── manifest.json          ← updated after each build
├── sessions_demo.db       ← the built database
├── sessions_demo_umap2d.joblib  ← saved UMAP reducer (reused for incremental runs)
└── sessions_demo_umap3d.joblib
```
