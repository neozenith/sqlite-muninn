# Text Embeddings — Semantic Search with muninn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/text_embeddings/example.ipynb)

Zero-dependency end-to-end example: load GGUF embedding models, embed documents,
build HNSW indices, and perform semantic similarity search — all inside a single
SQLite extension. Compares three models side-by-side.

## Models

| Model | Params | Dim | Quantization | Size | Context | Architecture |
|-------|--------|-----|-------------|------|---------|--------------|
| **all-MiniLM-L6-v2** | 22M | 384 | Q8_0 | 23 MB | 512 | BERT (mean pooling) |
| **nomic-embed-text-v1.5** | 137M | 768 | Q8_0 | 146 MB | 8K | BERT (mean pooling) |


All models are auto-downloaded to `models/` on first run. 

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Load GGUF model | `INSERT INTO temp.muninn_models(name, model) SELECT 'name', muninn_embed_model('path.gguf')` |
| Query dimension | `SELECT muninn_model_dim('MiniLM')` |
| Generate embedding | `SELECT muninn_embed('MiniLM', 'hello world')` |
| Prefixed embedding | `SELECT muninn_embed('NomicEmbed', 'search_query: hello world')` |
| Bulk embed + index | `INSERT INTO idx(rowid, vector) SELECT id, muninn_embed('MiniLM', content) FROM docs` |
| KNN semantic search | `WHERE vector MATCH muninn_embed('MiniLM', 'query') AND k = 3` |
| Auto-embed trigger | `CREATE TEMP TRIGGER ... muninn_embed(...)` on INSERT |
| Multi-model compare | Side-by-side search results from all models |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
# Run the example (models auto-download on first run)
python examples/text_embeddings/example.py

# See llama.cpp internals (model loading, tensor info, etc.)
MUNINN_LOG_LEVEL=verbose python examples/text_embeddings/example.py
```

## Sections

| # | Section | What It Shows |
|---|---------|---------------|
| 1 | **Model Loading & Inspection** | Load all models, query dimensions, show prefixes |
| 2 | **Embed & Build HNSW Indices** | Per-model HNSW indices with bulk INSERT...SELECT |
| 3 | **Comparative Semantic Search** | Side-by-side KNN results for each query across all models |
| 4 | **Auto-Embed Trigger** | TEMP trigger that auto-embeds new rows on INSERT |

## Data

8 sentences across 4 topics:

- **Nature:** fox in the forest, wolves and bears
- **AI/ML:** neural networks, gradient descent
- **Food:** Italian pasta
- **Space:** Mars rover, stars and galaxies

3 semantic queries test cross-topic retrieval:

- "animals in the wild" — should match nature documents
- "machine learning and artificial intelligence" — should match AI documents
- "outer space exploration" — should match space documents

## Model Architecture Notes

### BERT vs Decoder embedding models

muninn automatically handles both architectures through GGUF metadata:

- **BERT models** (MiniLM, Nomic): bidirectional attention, **mean pooling** — the
  embedding is the average of all token hidden states.

The pooling type is baked into each GGUF file's metadata, so muninn reads it
automatically. No configuration needed.

### Task instruction prefixes

Some models use prefixes to specialize embeddings for different tasks:

| Model | Document prefix | Query prefix |
|-------|----------------|--------------|
| MiniLM | *(none)* | *(none)* |
| nomic-embed-text-v1.5 | `search_document: ` | `search_query: ` |

The example handles prefixes via `ModelConfig.doc_prefix` / `query_prefix`.

## Example Output

```sh
=== muninn Text Embeddings Example ===

  Project root:  .
  Extension:     ./build/muninn
INFO: Model MiniLM found: ./models/all-MiniLM-L6-v2.Q8_0.gguf (25.0 MB)
INFO: Model NomicEmbed found: ./models/nomic-embed-text-v1.5.Q8_0.gguf (146.1 MB)

  Models ready: ['MiniLM', 'NomicEmbed']
  Loaded muninn extension (HNSW + GGUF embedding).

  Created documents table with 8 rows.

============================================================
Section 1: Model Loading & Inspection
============================================================

  MiniLM: dim=384, file=all-MiniLM-L6-v2.Q8_0.gguf

  NomicEmbed: dim=768, file=nomic-embed-text-v1.5.Q8_0.gguf
    doc prefix: 'search_document: '
    query prefix: 'search_query: '

  All loaded models: [('MiniLM', 384), ('NomicEmbed', 768)]

============================================================
Section 2: Embed Documents & Build HNSW Indices
============================================================

  MiniLM: created HNSW index (dim=384)
  MiniLM: embedded and indexed 8 documents
  MiniLM: verified vector size = 1536 bytes (384 floats)

  NomicEmbed: created HNSW index (dim=768)
  NomicEmbed: embedded and indexed 8 documents
  NomicEmbed: verified vector size = 3072 bytes (768 floats)

============================================================
Section 3: Comparative Semantic Search
============================================================

  Query: "animals in the wild"
  ──────────────────────────────────────────────────────

    MiniLM (dim=384):
      #6   dist=0.4151  Wolves and bears roam the dense woodland trails
      #1   dist=0.5264  The quick brown fox jumps over the lazy dog in the forest
      #4   dist=0.9143  The Mars rover collected soil samples from the crater

    NomicEmbed (dim=768):
      #6   dist=0.3564  Wolves and bears roam the dense woodland trails
      #1   dist=0.3728  The quick brown fox jumps over the lazy dog in the forest
      #2   dist=0.4477  A neural network learns patterns from large datasets

  Query: "machine learning and artificial intelligence"
  ──────────────────────────────────────────────────────

    MiniLM (dim=384):
      #2   dist=0.5631  A neural network learns patterns from large datasets
      #7   dist=0.7358  Gradient descent optimizes the loss function during training
      #5   dist=0.8395  SQLite is the most widely deployed database engine in the world

    NomicEmbed (dim=768):
      #2   dist=0.3018  A neural network learns patterns from large datasets
      #7   dist=0.4072  Gradient descent optimizes the loss function during training
      #5   dist=0.4730  SQLite is the most widely deployed database engine in the world

  Query: "outer space exploration"
  ──────────────────────────────────────────────────────

    MiniLM (dim=384):
      #8   dist=0.7220  Stars and galaxies fill the observable universe
      #4   dist=0.7655  The Mars rover collected soil samples from the crater
      #2   dist=0.8278  A neural network learns patterns from large datasets

    NomicEmbed (dim=768):
      #8   dist=0.4264  Stars and galaxies fill the observable universe
      #4   dist=0.4682  The Mars rover collected soil samples from the crater
      #5   dist=0.5448  SQLite is the most widely deployed database engine in the world

============================================================
Section 4: Auto-Embed Trigger (using MiniLM)
============================================================

  Created TEMP trigger for auto-embedding on INSERT.
  Inserted doc #100: 'Black holes warp spacetime near the event horizon'

  Search for "phenomena in space" top-3:
    #8    dist=0.5579  Stars and galaxies fill the observable universe
    #100  dist=0.7544  Black holes warp spacetime near the event horizon
    #4    dist=0.8466  The Mars rover collected soil samples from the crater

  Trigger-inserted document found in results.

============================================================
Done. All sections completed successfully.
============================================================
```
