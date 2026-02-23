# Plan: Restore KG Demo Database Builder

## Problem

The `wasm/assets/3300.db` demo database was originally built by the legacy script
`benchmarks/scripts/kg_extract.py` + `benchmarks/scripts/kg_coalesce.py`, invoked via
`make -C benchmarks kg-demo`. Those scripts were deleted in commit `2410c77` (2026-02-21)
when the benchmark harness refactor decomposed them into Treatment subclasses. The harness
treatments (`kg_extract.py`, `kg_resolve.py`, `kg_graphrag.py`) are mostly **stubs** — they
define the Treatment ABC interface but return hardcoded zeros. No Makefile target currently
rebuilds the demo database.

Both `wasm/` and `viz/` depend on this pre-built database for their KG visualisations.

## Current State

### What Still Exists (assets we can reuse)

| Asset | Path | Status |
|-------|------|--------|
| Gutenberg text | `benchmarks/texts/gutenberg_3300.txt` | 2.4 MB, present |
| Chunks DB | `benchmarks/kg/3300_chunks.db` | 1,850 chunks, present |
| Cached MiniLM vectors | `benchmarks/vectors/MiniLM_wealth_of_nations.npy` | 384-dim, present |
| Cached Nomic vectors | `benchmarks/vectors/NomicEmbed_wealth_of_nations_docs.npy` | 768-dim, present |
| GGUF models | `models/*.gguf` | MiniLM (25 MB), Nomic (146 MB), BGE-Large (358 MB) |
| Stale demo DB | `wasm/assets/3300.db` | 12.8 MB, still loadable but was built by deleted scripts |
| Viz KG service | `viz/server/services/kg.py` | 766 lines, fully functional, expects the full schema |
| WASM demo | `wasm/script.js` | 1,020 lines, expects UMAP tables + FTS + HNSW + graph |
| Harness chunker | `benchmarks/harness/prep/kg_chunks.py` | Working, creates `text_chunks` table |

### What's Missing (deleted / never implemented)

| Capability | Was In | Harness Replacement | Status |
|------------|--------|---------------------|--------|
| ML-based NER extraction | `kg_extract.py` (legacy) | `treatments/kg_extract.py` | ML adapters stubbed |
| Relation extraction | `kg_extract.py` (legacy) | `treatments/kg_re.py` | Crude entity-pair proxy |
| Embedding into HNSW | `kg_extract.py` (legacy) | — | Not in any harness file |
| UMAP projection | `kg_extract.py` (legacy) | `viz/server/services/embeddings.py` | Viz has UMAPProjector but not in build pipeline |
| Entity resolution | `kg_coalesce.py` (legacy) | `treatments/kg_resolve.py` | Stub (returns zeros) |
| Clean graph build | `kg_coalesce.py` (legacy) | — | Not ported |
| Node2Vec training | `kg_coalesce.py` (legacy) | — | Not ported |
| GraphRAG query demo | `kg_coalesce.py` (legacy) | `treatments/kg_graphrag.py` | Stub (returns zeros) |
| `make kg-demo` target | `benchmarks/Makefile` | — | Deleted |

### Target Schema (what `wasm/script.js` and `viz/server/services/kg.py` expect)

```
chunks (chunk_id, text)                          — 1,850 rows
chunks_fts (FTS5, content=chunks)                — full-text index
chunks_vec (HNSW 384-dim cosine)                 — 1,850 embeddings
chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d)   — 1,850 projections
entities (entity_id, name, entity_type, source, chunk_id, confidence) — ~11K rows
entities_vec (HNSW 384-dim cosine)               — unique entity embeddings
entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) — entity projections
entity_vec_map (rowid, name)                     — maps HNSW rowid → entity name
entity_clusters (name, canonical)                — synonym resolution
nodes (node_id, name, entity_type, mention_count) — canonical entities
edges (src, dst, rel_type, weight)               — coalesced relation graph
node2vec_emb (HNSW 64-dim cosine)                — structural graph embeddings
meta (key, value)                                — provenance metadata
```

## Design Decisions

### Embedding Model: MiniLM (384-dim)

The legacy pipeline used `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Keep it.

1. The WASM demo uses `Xenova/all-MiniLM-L6-v2` in-browser — switching would break client-side search
2. 384-dim keeps the demo DB under 15 MB (important for WASM download)
3. The cached `.npy` vectors already exist
4. Embeddings are external to SQLite — Python precomputes them, WASM computes live via Transformers.js, viz computes live via sentence-transformers

### KG Pipeline: Real-World ML Models, Not FTS Keyword Hacks

All models run externally in Python. The demo DB is a pre-built artifact — no live model
inference is needed at demo time. We have the full scope of any Python package, ML model,
or GGUF model for the build step.

## Research: Modern KG Pipeline Options

### NER Models (Named Entity Recognition)

| Model | Params | Approach | Custom Labels? | Quality | Install |
|-------|--------|----------|---------------|---------|---------|
| **GLiNER medium v2.1** | 209M | Bi-encoder, zero-shot | Yes, at inference | High | `pip install gliner` |
| GLiNER2 large v1 | 340M | Multi-task (NER+RE+classify) | Yes, schema API | High | `pip install gliner2` |
| spaCy `en_core_web_lg` | 560M | Pipeline NER | No (17 fixed types) | Medium | `pip install spacy` |

**GLiNER** is the standout: zero-shot NER where you define entity types at runtime via
natural language labels (e.g. "person", "economic concept", "commodity", "institution").
No fine-tuning, no fixed vocabulary. The medium model (~600 MB) is the quality/speed sweet spot.

### Relation Extraction Models

| Model | Params | Approach | Custom Labels? | Quality | Install |
|-------|--------|----------|---------------|---------|---------|
| **GLiREL large v0** | ~400M | Bi-encoder, zero-shot RE | Yes, at inference | High | `pip install glirel` |
| REBEL large | 406M | Seq2seq, 200+ Wikidata types | No (fixed vocab) | Medium | `pip install transformers` |
| ReLiK CIE | varies | Retriever+Reader, EL+RE | No (Wikidata) | High | `pip install relik` |
| textacy SVO | 0 | Dependency parse rules | N/A | Low | `pip install textacy` |

**GLiREL** is GLiNER's companion for relations — same zero-shot approach where you define
relation labels at runtime (e.g. "regulates", "trades_with", "produces"). One forward pass,
works with pre-extracted GLiNER entities.

> Note: GLiREL is CC BY-NC-SA 4.0 (non-commercial). Fine for a demo/research project.

### Entity Resolution

| Approach | Method | Quality | Install |
|----------|--------|---------|---------|
| **HNSW blocking + Jaro-Winkler + Leiden** | Embed names → KNN candidates → string sim → cluster | Good | muninn extension + pure Python |
| Splink | Fellegi-Sunter probabilistic matching | Excellent | `pip install splink` |
| KGGen iterative clustering | LLM-as-judge clustering | Best | `pip install kg-gen` (needs Ollama) |

For the demo, the existing approach (HNSW blocking → Jaro-Winkler → Leiden) is solid and
exercises muninn's own subsystems. No external dependency needed.

### Full Pipeline Packages (for reference)

| Package | NER | RE | Entity Resolution | Requires LLM? |
|---------|-----|-----|-------------------|---------------|
| Microsoft GraphRAG | LLM prompt | LLM prompt | LLM summarisation | Yes (expensive) |
| KGGen (NeurIPS 2025) | LLM prompt | LLM prompt | Iterative LLM clustering | Yes (Ollama OK) |
| LightRAG | LLM prompt | LLM prompt | Deduplication | Yes |
| **GLiNER + GLiREL** | Encoder model | Encoder model | Manual/heuristic | **No** |

**Decision: GLiNER + GLiREL for NER+RE.** No LLM required, runs on CPU, zero-shot with
custom domain labels, fast encoder models. Entity resolution via muninn's own HNSW + Leiden.

This is a genuine, published, peer-reviewed KG extraction pipeline — not a keyword hack.

## Implementation Plan

### Approach: Single `build_demo_db.py` Script

Rather than trying to complete the half-finished harness treatments (which are designed for
benchmarking with setup/run/teardown lifecycle and JSONL metrics output), create a dedicated
**demo database builder script** at `benchmarks/scripts/build_demo_db.py`. This script's
sole purpose is producing a self-contained `.db` file for the WASM and viz demos.

The harness treatments can be completed separately as a benchmark concern; the demo builder
is a demo/docs concern with different priorities (correctness > metrics, simplicity > generality).

### Phase 1: Core Tables (chunks + FTS + embeddings)

**File:** `benchmarks/scripts/build_demo_db.py`

1. **Create database** at `wasm/assets/{book_id}.db`
2. **Import chunks** from `benchmarks/kg/{book_id}_chunks.db` → `chunks` table
   - Rename `text_chunks.id` → `chunks.chunk_id`, `text_chunks.text` → `chunks.text`
   - Or re-chunk from `benchmarks/texts/gutenberg_{book_id}.txt` (256-word window, 50 overlap)
3. **Build FTS5 index** — `chunks_fts` content-sync table on `chunks`
4. **Load cached chunk embeddings** from `benchmarks/vectors/MiniLM_wealth_of_nations.npy`
   - These are pre-computed 384-dim float32 vectors, one per chunk
   - Requires `make -C benchmarks prep-vectors` to have been run first
5. **Insert into `chunks_vec`** HNSW virtual table (384-dim, cosine, M=16, ef_construction=200)
   - Use `pack_vector()` to convert numpy arrays to float32 BLOBs
6. **Write initial meta** — book_id, embedding_model, num_chunks

**Dependencies:** sqlite3, numpy. Muninn extension for HNSW virtual table.

### Phase 2: Entity Extraction (GLiNER zero-shot NER)

7. **Load GLiNER** model: `urchade/gliner_medium-v2.1` (209M params, ~600 MB download, cached by HF)
8. **Define domain entity types** for Wealth of Nations:
   ```python
   labels = [
       "person",              # Adam Smith, David Ricardo, etc.
       "organization",        # East India Company, Parliament, etc.
       "location",            # England, Scotland, America, etc.
       "economic concept",    # division of labour, supply and demand, etc.
       "commodity",           # gold, silver, corn, wool, etc.
       "institution",         # government, church, university, etc.
       "legal concept",       # statute, bounty, monopoly, etc.
       "occupation",          # labourer, merchant, manufacturer, etc.
   ]
   ```
9. **Batch extract** entities from all 1,850 chunks:
   - `model.predict_entities(chunk_text, labels, threshold=0.3)`
   - Batch size 32 (matches legacy GLiNER config)
   - Each entity: `(name, entity_type, chunk_id, confidence_score)`
10. **Insert into `entities`** table with `source='gliner'`

**Expected yield:** ~5,000–15,000 entity mentions across 1,850 chunks (depends on threshold).

### Phase 3: Relation Extraction (GLiREL zero-shot RE)

11. **Load GLiREL** model: `jackboyla/glirel-large-v0` (~400M params)
12. **Define domain relation types:**
    ```python
    relation_labels = [
        "produces",          # country produces commodity
        "trades_with",       # entity trades with entity
        "regulates",         # institution regulates activity
        "employs",           # organization employs person/occupation
        "located_in",        # entity located in place
        "influences",        # concept influences concept
        "part_of",           # entity is part of entity
        "opposes",           # policy opposes policy
    ]
    ```
13. **Extract relations** per chunk, using GLiNER entities as input spans:
    - `model.predict_relations(tokens, relation_labels, ner=gliner_entities, threshold=0.5)`
    - Each relation: `(src_entity, relation_type, dst_entity, chunk_id, confidence)`
14. **Insert into `relations`** table with `source='glirel'`

**Expected yield:** ~1,000–5,000 relations (one or more per chunk that has 2+ entities).

### Phase 4: Entity Embeddings

15. **Collect unique entity names** from `entities` table (deduplicated by name)
16. **Embed** using same MiniLM model as chunks (load from sentence-transformers or cached)
17. **Insert into `entities_vec`** HNSW table (384-dim, cosine)
18. **Populate `entity_vec_map`** — maps HNSW rowid → entity name string

### Phase 5: UMAP Dimensionality Reduction

19. **Extract all vectors** from `chunks_vec_nodes` and `entities_vec_nodes` shadow tables
20. **Compute UMAP 2D + 3D** projections:
    - cosine metric, n_neighbors=15, min_dist=0.1, random_state=42
    - Fit on chunks (larger set), transform entities into same space
21. **Insert into `chunks_vec_umap`** (id, x2d, y2d, x3d, y3d, z3d)
22. **Insert into `entities_vec_umap`** (id, x2d, y2d, x3d, y3d, z3d)

**Dependencies:** `umap-learn`, numpy.

### Phase 6: Entity Resolution (Coalescing)

Uses muninn's own subsystems — this is the showcase for the extension.

23. **HNSW blocking** — for each unique entity, KNN search on `entities_vec` to find
    candidate match pairs within cosine distance < 0.4. This is O(N·K·logN) vs O(N^2) pairwise.
24. **Matching cascade** — score each candidate pair via:
    - Exact match (score=1.0)
    - Case-insensitive exact (score=0.9)
    - Jaro-Winkler similarity (pure Python, no dependency)
    - Cosine similarity (1.0 - HNSW distance)
    - Combined weighted score; accept pairs above threshold 0.5
25. **Leiden clustering** — `SELECT * FROM graph_leiden(...)` on the match-pair graph
    to cluster synonyms into canonical groups (requires muninn extension)
26. **Populate `entity_clusters`** — (name → canonical) mapping, canonical = highest
    mention-count member of each cluster
27. **Build clean graph:**
    - `nodes` table: one row per canonical entity, aggregated mention_count + entity_type
    - `edges` table: deduplicated relations using canonical names, aggregated weights

### Phase 7: Node2Vec Structural Embeddings

28. **Train Node2Vec** via `SELECT node2vec_train(...)` SQL function (requires muninn):
    - Source: `edges` table (src, dst columns)
    - Target: `node2vec_emb` HNSW virtual table (64-dim, cosine)
    - Parameters: p=0.5, q=0.5, walks=10, walk_length=40, window=5, neg=5, epochs=5

### Phase 8: Metadata + Validation

29. **Write `meta` table:**
    - book_id, text_file, strategies (e.g. "gliner+glirel")
    - total_entities, total_relations, num_chunks
    - embedding_model, ner_model, re_model
    - build_timestamp
30. **Validation pass** — query each expected table, log row counts, assert non-empty
31. **VACUUM** — compact the database file

### CLI Interface

```
python benchmarks/scripts/build_demo_db.py \
    --book-id 3300 \
    --output wasm/assets/3300.db \
    --embedding-model MiniLM \
    [--force]
```

### Makefile Target

Add to root `Makefile` (or `benchmarks/Makefile`):

```makefile
demo-db: extension                          ## Build the KG demo database for wasm/viz
	.venv/bin/python benchmarks/scripts/build_demo_db.py \
		--book-id 3300 \
		--output wasm/assets/3300.db \
		--embedding-model MiniLM
```

## Dependency Matrix

| Phase | Python Packages | Muninn Extension? |
|-------|----------------|-------------------|
| 1 — Chunks + FTS + chunk embeddings | numpy, sentence-transformers | Yes (HNSW) |
| 2 — NER (GLiNER) | gliner | No |
| 3 — RE (GLiREL) | glirel, spacy | No |
| 4 — Entity embeddings | numpy, sentence-transformers | Yes (HNSW) |
| 5 — UMAP projections | umap-learn | No |
| 6 — Entity resolution | — (pure Python + SQL) | Yes (Leiden) |
| 7 — Node2Vec | — | Yes (node2vec_train) |
| 8 — Meta + validation | — | No |

Install all build dependencies:

```bash
uv run -m spacy download en_core_web_lg
```

## Execution Order

All 8 phases run sequentially. Every phase is mandatory. If any phase fails, the
script aborts with a clear error — a partial database is not a valid demo database.

**Prerequisites before running:**
- `make all` — builds the muninn extension (required for HNSW, Leiden, Node2Vec)
- `make -C benchmarks prep-vectors` — precomputes the MiniLM `.npy` embeddings
- `uv run -m spacy download en_core_web_lg` — spaCy model for GLiREL tokenization

The output database exercises all five muninn subsystems:
1. HNSW (chunk + entity vector search)
2. Graph TVFs (BFS expansion in GraphRAG queries)
3. Centrality (degree/betweenness/closeness on the entity graph)
4. Community detection (Leiden for entity resolution AND topic clustering)
5. Node2Vec (structural graph embeddings)

## File Layout

```
benchmarks/scripts/build_demo_db.py    — Main builder script (~500-700 lines)
wasm/assets/3300.db                    — Output demo database (~13-15 MB)
```

## Success Criteria

The measure of success is these two targets passing:

```bash
make -C wasm ci    # prettier format + Playwright E2E tests + video conversion
make -C viz ci     # ruff + eslint + mypy + tsc + pytest + vitest + Playwright E2E + videos
```

Both CI targets run Playwright E2E tests that load the demo database, exercise the
search UI (FTS, HNSW, graph), and verify the visualisations render. If the database
schema is incomplete or tables are empty, these tests fail.

## Resolved Questions

1. **UMAP reproducibility:** Not a concern — UMAP is purely for visualising relative
   spacing of embedding results. `random_state=42` is fine.

2. **GLiREL license:** CC BY-NC-SA 4.0. Fine for this research/demo project.

3. **DB size budget:** No constraint. Even hundreds of MB is fine — this is a localhost demo.

4. **viz/ database path:** Builder outputs to `wasm/assets/3300.db`. Add a
   `--copy-to viz/server/data/3300.db` flag so viz points to the same built artifact.

5. **HuggingFace model cache:** `HF_HOME=.hf_cache` is exported in the environment.
   All HuggingFace model downloads (GLiNER, GLiREL, sentence-transformers) will cache
   to `.hf_cache/` in the project root.
