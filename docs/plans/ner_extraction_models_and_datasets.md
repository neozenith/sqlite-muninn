# NER Entity & Relation Extraction Models and Benchmark Datasets

Captured: 2026-02-16. Split from `kg_extraction_benchmark_spec.md`. Updated: 2026-02-18 with implementation plan.

This document catalogues the NER/RE models under evaluation, the benchmark datasets used to measure extraction quality, and the concrete implementation plan for cross-model comparison on Wealth of Nations text.

---

## Table of Contents

1. [Entity Extraction Models](#1-entity-extraction-models)
2. [Benchmark Datasets](#2-benchmark-datasets)
3. [Implementation Plan](#3-implementation-plan)
4. [Per-Model Results Database](#4-per-model-results-database)
5. [Cross-Model Comparison](#5-cross-model-comparison)
6. [Quality Inspection Workflow](#6-quality-inspection-workflow)
7. [Makefile Integration](#7-makefile-integration)
8. [References](#references)

---

## 1. Entity Extraction Models

### 1a. GLiNER Family (all use `gliner` Python package)

All models share the same API: `GLiNER.from_pretrained(model_id)` → `model.predict_entities(text, labels, threshold=0.5)`.

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License |
|----------|--------|------|-------------|-------------|---------|
| `urchade/gliner_small-v2.1` | 166M | 611 MB | 50-100 | ~52 | Apache-2.0 |
| `urchade/gliner_medium-v2.1` | 209M | 781 MB | 80-160 | ~56 | Apache-2.0 |
| `urchade/gliner_large-v2.1` | 459M | 1.78 GB | 300-500 | 60.9 | Apache-2.0 |
| `urchade/gliner_multi-v2.1` | 209M | 1.16 GB | 100-200 | ~54 | Apache-2.0 |
| `knowledgator/modern-gliner-bi-large-v1.0` | ~530M | 2.12 GB | 300-500 | >60.9 | Apache-2.0 |
| `knowledgator/gliner-multitask-large-v0.5` | ~440M | 1.76 GB | 300-500 | >60.9 | Apache-2.0 |

**Notes:**
- `modern-gliner-bi-large` uses ModernBERT backbone (8192 token context, 4x faster than DeBERTa). Requires dev `transformers`: `uv add --group benchmark "transformers @ git+https://github.com/huggingface/transformers.git"`
- `gliner-multitask-large` supports 7 tasks (NER, RE, classification, summarization, sentiment, QA) via prompt tuning
- `gliner_multi-v2.1` covers 20+ languages including CJK, Arabic, Hindi, Finnish

### 1b. Competitors (GLiNER-compatible API)

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License | Notes |
|----------|--------|------|-------------|-------------|---------|-------|
| `numind/NuNerZero` | ~400M | 1.8 GB | 300-500 | ~64 | MIT | Labels MUST be lowercase. Use `merge_entities()` post-processing for adjacent spans. |

**Loading code is identical to GLiNER:**
```python
from gliner import GLiNER
model = GLiNER.from_pretrained("numind/NuNerZero")
entities = model.predict_entities(text, [l.lower() for l in labels])
```

### 1c. Competitors (different API — generative)

| Model ID | Params | Disk | CPU ms/chunk | Zero-Shot F1 | License | Notes |
|----------|--------|------|-------------|-------------|---------|-------|
| `dyyyyyyyy/GNER-T5-base` | 248M | ~1.0 GB | 2,000-5,000 | 59.5 | MIT | Seq2seq, outputs BIO text. 10-50x slower than encoder models. |
| `dyyyyyyyy/GNER-T5-large` | 783M | ~3.1 GB | 5,000-15,000 | 63.5 | MIT | Include with caution — ~8-25 min for 100 chunks. |

**Loading code (different from GLiNER):**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-base")
# Requires instruction-formatted input, outputs BIO-tagged text that must be parsed
```

### 1d. Excluded Models

| Model | Reason |
|-------|--------|
| SpanMarker | Not zero-shot — fixed predefined label set, cannot accept arbitrary entity types |
| GNER-T5-xl (3B) | Too large/slow for CPU benchmarking |
| GNER-T5-xxl (11B) | Far too large for 16GB RAM |
| UniNER-7B/13B | Requires GPU, 7B+ parameters |

### 1e. Installation

```bash
# Core (covers all GLiNER family + NuNER)
uv add --group benchmark gliner

# For modern-gliner-bi-large (ModernBERT support)
uv add --group benchmark "transformers @ git+https://github.com/huggingface/transformers.git"

# For GNER-T5 (usually already installed via gliner deps)
uv add --group benchmark torch transformers
```

---

## 2. Benchmark Datasets

### 2a. Entity Extraction (NER)

| Dataset | Size | Types | Domain | Access | Priority |
|---------|------|-------|--------|--------|----------|
| **CoNLL-2003** | 22K sentences | 4 (PER/LOC/ORG/MISC) | Reuters news | HuggingFace `datasets` | **P0** — gold standard, required |
| **CrossNER** | Small per domain | Domain-specific | Politics, science, music, literature, AI | GitHub | **P1** — tests cross-domain generalization |
| **WNUT-17** | Small | 6 types | Social media (noisy) | HuggingFace | P2 — tests on novel entities |
| **Few-NERD** | 188K sentences | 66 fine-grained | Wikipedia | HuggingFace | P3 — large, run subset only |

**Rationale:** CoNLL-2003 is required for comparability (every NER paper reports against it). CrossNER tests domain transfer (relevant to our Gutenberg economics texts). The others are optional stretch targets.

### 2b. Relation Extraction

| Dataset | Size | Level | Relations | Access | Priority |
|---------|------|-------|-----------|--------|----------|
| **DocRED** | 5K docs, 63K triples | Document | 96 types | GitHub | **P1** — document-level, matches chunked pipeline |
| **WebNLG** | 17K triple sets | Text-to-triple | ~450 DBpedia props | Free | P2 — full KG extraction |
| **TACRED** | 106K sentences | Sentence | 42 types | LDC/mirrors | P3 — sentence-level only |

### 2c. Standard Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| **Entity-level micro F1** (strict) | NER | Span boundaries AND type must match exactly |
| **Precision** | NER | Fraction of predicted entities that are correct |
| **Recall** | NER | Fraction of gold entities that were found |
| **Triple F1** (strict) | RE | Subject + predicate + object must all match |
| **Inference time** | All | Wall-clock ms per chunk |
| **Cost** | LLM | USD per 1K chunks (from token usage × pricing) |

---

## 3. Implementation Plan

### 3a. Architecture Overview

All models extract from the **same shared chunks**. A chunks-only DB (`3300_chunks.db`) is created as a checkpoint immediately after chunking, before any extraction strategies run. This is a lightweight, stable, shareable reference point that every model DB can ATTACH to resolve `chunk_id → text`.

```
benchmarks/kg/3300_chunks.db                   ← checkpoint: chunks + FTS5 only (~200 KB)
                 │
                 │  ATTACH 'kg/3300_chunks.db' AS chunks_db
                 │
                 ├── benchmarks/kg/extractions/3300_fts5.db
                 ├── benchmarks/kg/extractions/3300_gliner_small-v2.1.db
                 ├── benchmarks/kg/extractions/3300_gliner_medium-v2.1.db
                 ├── benchmarks/kg/extractions/3300_gliner_large-v2.1.db
                 ├── benchmarks/kg/extractions/3300_nunerzer0.db
                 ├── benchmarks/kg/extractions/3300_gner-t5-base.db
                 └── ...

benchmarks/kg/3300.db                          ← full pipeline: chunks + entities + embeddings + coalescing
benchmarks/results/kg_extraction.jsonl         ← aggregate metrics (append-only)
```

**Why a chunks DB checkpoint?** A wholesale copy of the SQLite file at the right moment gives us a single source of truth for the input text. Model DBs contain only extraction results (entities, relations, timing) and ATTACH the chunks DB for inspection queries. No duplication, no drift.

### 3a-i. Change to `kg_extract.py`

After chunking completes (and the FTS5 index is rebuilt), `kg_extract.py` creates a copy of the DB file:

```python
import shutil

# After insert_chunks() and FTS5 rebuild, before any extraction strategies run:
chunks_db_path = KG_DIR / f"{book_id}_chunks.db"
if not chunks_db_path.exists():
    # Checkpoint: vacuum into a clean copy with just chunks + FTS5
    conn.execute("VACUUM")
    conn.close()
    shutil.copy2(db_path, chunks_db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)
    log.info("Chunks checkpoint: %s (%d KB)", chunks_db_path.name, chunks_db_path.stat().st_size // 1024)
```

**Idempotent:** Only copies if the chunks DB doesn't already exist. Re-running `kg_extract.py` on the same book skips the copy. To regenerate, delete `3300_chunks.db` first.

### 3b. New Script: `benchmarks/scripts/kg_ner_benchmark.py`

Single-model extraction runner. Processes one model at a time for clear isolation.

```bash
# Run a specific model
python benchmarks/scripts/kg_ner_benchmark.py --book-id 3300 --model gliner:urchade/gliner_small-v2.1
python benchmarks/scripts/kg_ner_benchmark.py --book-id 3300 --model gliner:numind/NuNerZero
python benchmarks/scripts/kg_ner_benchmark.py --book-id 3300 --model gner:dyyyyyyyy/GNER-T5-base
python benchmarks/scripts/kg_ner_benchmark.py --book-id 3300 --model fts5

# Re-extract (drops and recreates the model DB)
python benchmarks/scripts/kg_ner_benchmark.py --book-id 3300 --model gliner:urchade/gliner_small-v2.1 --force
```

**Implementation steps:**

1. **Parse `--model` argument** into `(family, model_id)` tuple. Families: `gliner`, `gner`, `fts5`, `spacy`.
2. **Verify chunks DB** exists at `benchmarks/kg/{book_id}_chunks.db` (error if not — run `kg_extract.py` first).
3. **Read chunks** from the chunks DB into memory (list of `(chunk_id, text)` tuples).
4. **Create model DB** at `benchmarks/kg/extractions/{book_id}_{safe_model_name}.db`.
5. **Run extraction**, recording wall-clock time per batch and total.
6. **Build co-occurrence relations** (reuse `build_cooccurrence_edges()` logic from `kg_extract.py`).
7. **Write summary metrics** to `benchmarks/results/kg_extraction.jsonl`.
8. **Print summary** to stdout: entity count, relation count, type distribution, timing.

### 3c. Model Adapter Pattern

Each model family implements the same interface:

```python
def extract(model_id: str, chunks: list[str], entity_types: list[str]) -> list[ExtractionResult]:
    """Returns a list of ExtractionResult, one per chunk."""
    ...

@dataclass
class EntityMention:
    name: str
    entity_type: str
    confidence: float
    start_char: int | None = None   # character offset within chunk (if available)
    end_char: int | None = None

@dataclass
class RelationMention:
    src: str
    dst: str
    rel_type: str
    confidence: float

@dataclass
class ExtractionResult:
    chunk_id: int
    entities: list[EntityMention]
    relations: list[RelationMention]     # only populated by RE-capable models
    elapsed_ms: float
```

**Adapter implementations:**

| Family | Adapter | Entity extraction | Relation extraction |
|--------|---------|-------------------|---------------------|
| `gliner` | `GLiNERAdapter` | `model.predict_entities()` | Co-occurrence only (post-hoc) |
| `gner` | `GNERAdapter` | Seq2seq BIO parsing | Co-occurrence only (post-hoc) |
| `fts5` | `FTS5Adapter` | Seed term matching via `fts5vocab` | Co-occurrence (built-in) |
| `spacy` | `SpacyAdapter` | `doc.ents` | SVO triples via `textacy` |

### 3d. Shared Entity Types

All models use the same entity type list for fair comparison:

```python
ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "economic concept",
    "commodity",
    "institution",
]
```

This matches the existing `GLINER_ENTITY_TYPES` in `kg_extract.py`. For models that support custom types (GLiNER, NuNER, LLMs), these are passed directly. For spaCy (fixed types like `PERSON`, `ORG`, `GPE`), a mapping normalises the output types.

### 3e. Implementation Order

| Step | What | Validates | Est. Effort |
|------|------|-----------|-------------|
| 0 | Update `kg_extract.py` to emit `_chunks.db` checkpoint | Shared chunks DB creation | Tiny — add `shutil.copy2` after chunking |
| 1 | Script skeleton + FTS5 adapter | Model DB creation, ATTACH pattern, metrics output | Small — reuses existing `extract_fts5_concepts()` |
| 2 | GLiNER adapter (small model) | Core adapter pattern, batch prediction | Small — reuses existing `extract_gliner_entities()` |
| 3 | Run GLiNER small + medium + large | Three model DBs, first cross-comparison | Just running the script 3 times |
| 4 | NuNerZero adapter | Labels-must-be-lowercase edge case | Small — same GLiNER API |
| 5 | GNER-T5 adapter | Seq2seq BIO parsing, different API surface | Medium — need BIO parser |
| 6 | spaCy adapter | SVO relation extraction | Small — reuses `extract_spacy_svo()` |
| 7 | Comparison script | Cross-model analysis | Medium |

---

## 4. Per-Model Results Database

### 4a. Schema

Each model DB contains **only extraction results** — no chunk text. Chunk text lives in the shared `{book_id}_chunks.db` and is accessed via ATTACH when needed for inspection.

**Model DB schema:**

```sql
-- Metadata: what model produced this, when, configuration
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- Keys: model_family, model_id, book_id, entity_types, extraction_timestamp,
--        total_time_s, chunk_count

-- Extracted entities — every mention, linked to source chunk by chunk_id
CREATE TABLE entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT,
    chunk_id INTEGER NOT NULL,       -- FK into chunks_db.chunks(chunk_id) via ATTACH
    confidence REAL DEFAULT 1.0,
    start_char INTEGER,              -- character offset within chunk (NULL if unavailable)
    end_char INTEGER
);

-- Extracted relations
CREATE TABLE relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel_type TEXT,
    weight REAL DEFAULT 1.0,
    chunk_id INTEGER,
    source TEXT NOT NULL              -- 'model' (direct extraction) or 'cooccurrence' (post-hoc)
);

-- Per-chunk timing for performance analysis
CREATE TABLE chunk_timing (
    chunk_id INTEGER PRIMARY KEY,
    elapsed_ms REAL NOT NULL
);

-- Indexes
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_chunk ON entities(chunk_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_relations_src ON relations(src);
CREATE INDEX idx_relations_dst ON relations(dst);
```

**Chunks DB schema** (created by `kg_extract.py` checkpoint):

```sql
-- This is what's inside 3300_chunks.db — a copy of 3300.db taken after chunking
CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
```

### 4b. What's Different from the Main `3300.db` Schema

| Field | Main DB | Model DB | Why |
|-------|---------|----------|-----|
| `chunks` table | Present | Not present — ATTACH `_chunks.db` instead | Single source of truth, no duplication |
| `entities.source` | `'gliner'`, `'spacy_ner'`, `'fts5'` | Not needed — entire DB is one model | Simpler queries |
| `entities.start_char` / `end_char` | Not present | Added | Enables exact span highlighting in inspection |
| `chunk_timing` | Not present | Added | Per-chunk latency for performance profiling |
| `meta` table | Shared across all strategies | Model-specific config | Know exactly what produced this data |
| HNSW tables | Present (embeddings) | Not present | Embeddings are a separate concern; keep model DBs focused on extraction |

### 4c. Meta Table Contents

```sql
INSERT INTO meta VALUES ('model_family', 'gliner');
INSERT INTO meta VALUES ('model_id', 'urchade/gliner_small-v2.1');
INSERT INTO meta VALUES ('book_id', '3300');
INSERT INTO meta VALUES ('entity_types', '["person","organization","location","economic concept","commodity","institution"]');
INSERT INTO meta VALUES ('extraction_timestamp', '2026-02-18T10:30:00Z');
INSERT INTO meta VALUES ('total_time_s', '45.2');
INSERT INTO meta VALUES ('chunk_count', '1850');
INSERT INTO meta VALUES ('entity_count', '8234');
INSERT INTO meta VALUES ('relation_count', '2100');
INSERT INTO meta VALUES ('threshold', '0.3');           -- model-specific param
```

---

## 5. Cross-Model Comparison

### 5a. The Comparison Problem

We have **no gold-standard annotations** for Wealth of Nations. The standard NER datasets (CoNLL-2003, etc.) use different text and different entity types. So we cannot compute F1 against a ground truth.

Instead, we use three complementary comparison strategies:

### 5b. Strategy 1 — Volume and Distribution Comparison

Automated, runs across all model DBs. Answers: "What does each model see?"

```sql
-- Entity volume per model (query each DB separately, aggregate in Python)
SELECT COUNT(*) as total_entities,
       COUNT(DISTINCT name) as unique_entities,
       COUNT(DISTINCT chunk_id) as chunks_with_entities
FROM entities;

-- Entity type distribution
SELECT entity_type, COUNT(*) as count, COUNT(DISTINCT name) as unique_names
FROM entities
GROUP BY entity_type
ORDER BY count DESC;

-- Top entities per model
SELECT name, entity_type, COUNT(*) as mentions, AVG(confidence) as avg_conf
FROM entities
GROUP BY name, entity_type
ORDER BY mentions DESC
LIMIT 30;
```

**Output:** A comparison table like:

| Metric | FTS5 | GLiNER-S | GLiNER-M | GLiNER-L | NuNER | GNER-T5 | spaCy |
|--------|------|----------|----------|----------|-------|---------|-------|
| Total entities | 11,641 | ? | ? | ? | ? | ? | ? |
| Unique entities | 58 | ? | ? | ? | ? | ? | ? |
| Chunks with entities | 1,850 | ? | ? | ? | ? | ? | ? |
| Relations | 3,048 | ? | ? | ? | ? | ? | ? |
| Avg confidence | 1.0 | ? | ? | ? | ? | ? | ? |
| Ms/chunk (avg) | <1 | ? | ? | ? | ? | ? | ? |

### 5c. Strategy 2 — Inter-Model Agreement

Automated. Answers: "Do models agree on what's an entity?"

For each pair of models, compute overlap on the same chunk:

```sql
-- Attach two model DBs (from a scratch in-memory connection)
ATTACH 'benchmarks/kg/extractions/3300_gliner_small-v2.1.db' AS a;
ATTACH 'benchmarks/kg/extractions/3300_gliner_large-v2.1.db' AS b;

-- Entity overlap: names found by both models in the same chunk
SELECT COUNT(DISTINCT a_ent.name) as overlap
FROM a.entities a_ent
JOIN b.entities b_ent
  ON a_ent.chunk_id = b_ent.chunk_id
  AND LOWER(a_ent.name) = LOWER(b_ent.name);
```

**Agreement metrics:**
- **Jaccard index** per chunk: `|A ∩ B| / |A ∪ B|` — how similar are the entity sets?
- **Consensus entities**: Names extracted by ≥ N models — high-confidence "true" entities.
- **Model-unique entities**: Names found by only one model — candidates for false positives or unique recall.

### 5d. Strategy 3 — Human Spot-Check (Random Sampling)

Manual, but the most informative for building intuition. See [Section 6](#6-quality-inspection-workflow) below.

### 5e. New Script: `benchmarks/scripts/kg_ner_compare.py`

```bash
# Generate comparison report across all model DBs for a book
python benchmarks/scripts/kg_ner_compare.py --book-id 3300

# Compare specific models
python benchmarks/scripts/kg_ner_compare.py --book-id 3300 --models gliner_small-v2.1,gliner_large-v2.1,nunerzer0

# Output formats
python benchmarks/scripts/kg_ner_compare.py --book-id 3300 --format table    # stdout (default)
python benchmarks/scripts/kg_ner_compare.py --book-id 3300 --format jsonl    # append to results
```

**Output includes:**
1. Volume comparison table (as above)
2. Entity type distribution per model
3. Top-30 entities per model with mention counts
4. Pairwise Jaccard agreement matrix
5. Consensus entities (found by ≥ 3 models)
6. Model-unique entities (found by exactly 1 model)

---

## 6. Quality Inspection Workflow

### 6a. Random Sampling Queries

Open any model DB in `sqlite3`, ATTACH the chunks DB, and run these queries to build intuition:

```sql
-- First: attach the shared chunks DB
ATTACH 'benchmarks/kg/3300_chunks.db' AS chunks_db;
```

**Sample entities with their source text:**
```sql
-- 10 random entities with the chunk text they were extracted from
SELECT e.entity_id, e.name, e.entity_type, e.confidence,
       e.chunk_id, SUBSTR(c.text, 1, 200) as chunk_preview
FROM entities e
JOIN chunks_db.chunks c ON e.chunk_id = c.chunk_id
ORDER BY RANDOM()
LIMIT 10;
```

**Sample relations with context:**
```sql
-- 10 random relations with the chunk they co-occurred in
SELECT r.src, r.rel_type, r.dst, r.weight,
       r.chunk_id, SUBSTR(c.text, 1, 200) as chunk_preview
FROM relations r
LEFT JOIN chunks_db.chunks c ON r.chunk_id = c.chunk_id
ORDER BY RANDOM()
LIMIT 10;
```

**Entities from a specific chunk (deep dive):**
```sql
-- Pick a random chunk and see everything extracted from it
SELECT chunk_id, text FROM chunks_db.chunks ORDER BY RANDOM() LIMIT 1;

-- Then with that chunk_id:
SELECT name, entity_type, confidence FROM entities WHERE chunk_id = ?;
SELECT src, rel_type, dst, weight FROM relations WHERE chunk_id = ?;
```

**High-confidence vs low-confidence comparison:**
```sql
-- Entities the model is most confident about
SELECT name, entity_type, confidence, chunk_id
FROM entities ORDER BY confidence DESC LIMIT 10;

-- Entities the model is least confident about (likely noise)
SELECT name, entity_type, confidence, chunk_id
FROM entities WHERE confidence < 0.5 ORDER BY confidence ASC LIMIT 10;
```

**FTS5 search for specific topics (uses the chunks DB's FTS5 index):**
```sql
-- Find chunks about a specific topic, then see what entities were extracted
SELECT e.name, e.entity_type, e.confidence, SUBSTR(c.text, 1, 150) as preview
FROM chunks_db.chunks_fts f
JOIN chunks_db.chunks c ON c.chunk_id = f.rowid
JOIN entities e ON e.chunk_id = c.chunk_id
WHERE chunks_db.chunks_fts MATCH 'division of labour'
ORDER BY c.chunk_id, e.confidence DESC;
```

### 6b. Side-by-Side Comparison Query

For comparing two models on the same chunk, ATTACH both model DBs plus the chunks DB:

```sql
ATTACH 'benchmarks/kg/3300_chunks.db' AS chunks_db;
ATTACH 'benchmarks/kg/extractions/3300_gliner_small-v2.1.db' AS model_a;
ATTACH 'benchmarks/kg/extractions/3300_gliner_large-v2.1.db' AS model_b;

-- Pick a random chunk and show its text
SELECT chunk_id, SUBSTR(text, 1, 300) FROM chunks_db.chunks ORDER BY RANDOM() LIMIT 1;

-- See what each model extracted from that chunk (e.g. chunk_id = 42)
SELECT 'gliner_small' as model, name, entity_type, confidence
FROM model_a.entities WHERE chunk_id = 42
UNION ALL
SELECT 'gliner_large' as model, name, entity_type, confidence
FROM model_b.entities WHERE chunk_id = 42
ORDER BY model, confidence DESC;
```

### 6c. Optional: Annotation Helper Script

For building ground truth on a small sample (10-20 chunks):

```bash
# Present random chunks and all model extractions for human annotation
python benchmarks/scripts/kg_ner_annotate.py --book-id 3300 --sample-size 10
```

This would:
1. Pick N random chunk_ids
2. For each chunk, show the text and extractions from every model
3. Prompt for human judgment: correct / incorrect / partial per entity
4. Save annotations to `benchmarks/kg/annotations/3300_gold.json`
5. Compute per-model precision/recall against the annotated sample

**Priority:** P2 — valuable but not blocking. The random sampling queries (6a) deliver 80% of the insight with 0% tooling.

---

## 7. Makefile Integration

### 7a. New Targets in `benchmarks/Makefile`

```makefile
# ── NER Benchmark ───────────────────────────────────────────────
MODELS ?= fts5 \
          gliner:urchade/gliner_small-v2.1 \
          gliner:urchade/gliner_medium-v2.1 \
          gliner:urchade/gliner_large-v2.1 \
          gliner:numind/NuNerZero

kg-ner-extract: kg/$(BOOK_ID).db                ## Run single model extraction (MODEL=gliner:urchade/gliner_small-v2.1)
	$(PYTHON) scripts/kg_ner_benchmark.py --book-id $(BOOK_ID) --model $(MODEL)

kg-ner-extract-all: kg/$(BOOK_ID).db            ## Run all NER models on a book
	@for model in $(MODELS); do \
	    echo "=== Extracting: $$model ==="; \
	    $(PYTHON) scripts/kg_ner_benchmark.py --book-id $(BOOK_ID) --model $$model; \
	done

kg-ner-compare: kg/extractions/                 ## Compare all model results for a book
	$(PYTHON) scripts/kg_ner_compare.py --book-id $(BOOK_ID)

kg-ner-clean:                                    ## Remove all extraction databases
	rm -rf kg/extractions/
```

### 7b. Directory Structure After Running

```
benchmarks/
├── kg/
│   ├── 3300.db                            # full pipeline: chunks + entities + embeddings + coalescing
│   ├── 3300_chunks.db                     # checkpoint: chunks + FTS5 only (created by kg_extract.py)
│   ├── extractions/                       # per-model results (entities + relations only, no chunks)
│   │   ├── 3300_fts5.db
│   │   ├── 3300_gliner_small-v2.1.db
│   │   ├── 3300_gliner_medium-v2.1.db
│   │   ├── 3300_gliner_large-v2.1.db
│   │   ├── 3300_nunerzer0.db
│   │   └── ...
│   └── annotations/                       # optional human annotations
│       └── 3300_gold.json
├── results/
│   ├── kg_extraction.jsonl                # per-model metrics (append-only)
│   └── kg_graphrag.jsonl                  # existing GraphRAG metrics
└── scripts/
    ├── kg_extract.py                      # existing: full pipeline (updated to emit _chunks.db)
    ├── kg_coalesce.py                     # existing: entity resolution
    ├── kg_ner_benchmark.py                # NEW: single-model extraction
    └── kg_ner_compare.py                  # NEW: cross-model analysis
```

---

## References

### Entity Extraction
- GLiNER (Zaratiana et al., NAACL 2024)
- NuNER-Zero (NuMind, 2024)
- GNER (Ding et al., ACL 2024 Findings)
- SpanMarker (Aarsen, 2023)

### Knowledge Graph Quality (datasets)
- KGGen / MINE benchmark (NeurIPS 2025) — fact recovery from KGs
- Text2KGBench (ISWC 2023) — ontology-driven KG extraction
- DocRED (Yao et al., ACL 2019) — document-level relation extraction
