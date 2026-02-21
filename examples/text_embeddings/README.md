# Text Embeddings — Semantic Search with Real Models

End-to-end text-in, semantic-search-out using muninn's HNSW index with real embedding models.

## What This Demonstrates

| Feature | How |
|---------|-----|
| Local GGUF embedding | `lembed('MiniLM', text)` via sqlite-lembed |
| OpenAI API embedding | `rembed('text-embedding-3-small', text)` via sqlite-rembed |
| Embed + insert in one SQL statement | `INSERT INTO idx(rowid, vector) SELECT id, lembed(...) FROM docs` |
| Auto-embed trigger | `CREATE TEMP TRIGGER ... lembed(...)` on INSERT |
| KNN semantic search | `WHERE vector MATCH lembed('MiniLM', 'query') AND k = 3` |

## Prerequisites

Build the muninn extension and install at least one embedding extension:

```bash
make all
pip install sqlite-lembed    # Local GGUF models (no API key needed)
pip install sqlite-rembed    # Remote API (requires OPENAI_API_KEY)
```

The GGUF model file (`all-MiniLM-L6-v2.Q8_0.gguf`, 36 MB) is **downloaded automatically** on first run into `models/`. No manual setup needed.

## Run

```bash
# lembed only (local, no API key needed — model auto-downloads)
python examples/text_embeddings/example.py

# Both lembed and rembed
export OPENAI_API_KEY="sk-..."
python examples/text_embeddings/example.py

# Use a different GGUF model file (skips auto-download)
GGUF_MODEL_PATH=/path/to/custom-model.gguf python examples/text_embeddings/example.py
```

## Data

8 sentences across 4 topics, embedded with real models:

- **Nature:** fox in the forest, wolves and bears
- **AI/ML:** neural networks, gradient descent
- **Food:** Italian pasta
- **Space:** Mars rover, stars and galaxies

3 semantic queries test cross-topic retrieval:

- "animals in the wild" — should match nature documents
- "machine learning and artificial intelligence" — should match AI documents
- "outer space exploration" — should match space documents

## Sections

The example runs sections conditionally based on available dependencies:

| Section | Requires | Auto-Setup | Skipped When |
|---------|----------|-----------|-------------|
| **lembed** | `pip install sqlite-lembed` | GGUF model auto-downloaded to `models/` | Package not installed, or download fails |
| **rembed** | `pip install sqlite-rembed` + `OPENAI_API_KEY` | None | Package not installed, or API key empty |

Warnings are logged for each skipped section with install instructions.
When `GGUF_MODEL_PATH` is set to a custom path, auto-download is skipped and the file must exist.

## Expected Output (first run, lembed only)

```
=== Text Embeddings Example ===

  Project root:  /path/to/sqlite-muninn
  Extension:     /path/to/sqlite-muninn/muninn
INFO: GGUF model not found at /path/to/sqlite-muninn/models/all-MiniLM-L6-v2.Q8_0.gguf — downloading...
  Downloading all-MiniLM-L6-v2.Q8_0.gguf: 36.2/36.2 MB (100%)
INFO: Downloaded model to .../models/all-MiniLM-L6-v2.Q8_0.gguf (36.2 MB)
WARNING: OPENAI_API_KEY is not set or empty. Skipping rembed examples.
  Loaded muninn extension.

  Created documents table with 8 rows.

============================================================
Section: sqlite-lembed (local GGUF embedding)
============================================================

  Loaded sqlite-lembed extension.
  Registered model: all-MiniLM-L6-v2.Q8_0.gguf
  Created HNSW index: dim=384, metric=cosine
  Embedded and indexed 8 documents.

  --- Semantic Search (lembed) ---

  Query: "animals in the wild"
    #1   dist=0.5678  The quick brown fox jumps over the lazy dog in the forest
    #6   dist=0.6012  Wolves and bears roam the dense woodland trails
    ...
```

On subsequent runs, the model is found immediately:

```
INFO: GGUF model found: .../models/all-MiniLM-L6-v2.Q8_0.gguf
```
