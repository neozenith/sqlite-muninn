# LLM Extract — Structured NER & RE with muninn

Zero-dependency end-to-end example: load a GGUF chat model, extract named
entities, extract relations, and run summarisation — all via SQL functions
inside a single SQLite extension. No Python ML libraries needed.

## Models

| Model | Params | Quantization | Size | Context | License |
|-------|--------|-------------|------|---------|---------|
| **Qwen3-4B** (default) | 4B | Q4_K_M | ~2.5 GB | 32K | Apache 2.0 |
| Qwen3-8B | 8B | Q4_K_M | ~5.0 GB | 32K | Apache 2.0 |
| Phi-4-mini | 3.8B | Q4_K_M | ~2.5 GB | 128K | MIT |
| Llama-3.2-1B | 1B | Q4_K_M | ~0.7 GB | 128K | Llama 3.2 |

The default model (Qwen3-4B) is auto-downloaded to `models/` on first run.
Edit `DEFAULT_MODEL` in `example.py` to switch models.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Load GGUF chat model | `INSERT INTO temp.muninn_chat_models(name, model) SELECT 'name', muninn_chat_model('path.gguf')` |
| List loaded models | `SELECT name, n_ctx FROM muninn_chat_models` |
| Plain chat completion | `SELECT muninn_chat('model', 'What is 2+2?')` |
| Named entity recognition | `SELECT muninn_extract_entities('model', text, 'person,org,location')` |
| Relation extraction | `SELECT muninn_extract_relations('model', text, entities_json)` |
| Text summarisation | `SELECT muninn_summarize('model', text, 64)` |
| Bulk NER→RE pipeline | CTE chaining `muninn_extract_entities` → `muninn_extract_relations` in pure SQL |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
# Run the example (model auto-downloads on first run, ~2.5 GB)
python examples/llm_extract/example.py
```

## Sections

| # | Section | What It Shows |
|---|---------|---------------|
| 1 | **Model Loading** | Load a GGUF chat model into the `muninn_chat_models` registry, inspect context window |
| 2 | **Plain Chat** | Free-form chat completion without grammar constraints |
| 3 | **Named Entity Recognition** | Extract person, organization, location, date entities with GBNF grammar-constrained JSON output |
| 4 | **Relation Extraction** | Chain NER entities into relation extraction, producing subject→predicate→object triples |
| 5 | **Summarisation** | Generate concise labels for entity communities (the GraphRAG community naming use case) |
| 6 | **Bulk SQL Pipeline** | NER→RE chained entirely in SQL via CTE — no Python in the loop |

## Data

5 sample documents covering corporate, financial, scientific, and business domains:

- "Alice Smith founded ACME Corporation in New York City in 1987."
- "Bob Jones, CEO of TechStart, announced a partnership with ACME Corporation."
- "The European Central Bank raised interest rates to combat inflation in the eurozone."
- "Dr. Marie Curie discovered radium at the University of Paris in 1898."
- "Amazon acquired Whole Foods for $13.7 billion, reshaping the grocery industry."

## How It Works

### Grammar-Constrained Generation

`muninn_extract_entities` and `muninn_extract_relations` use GBNF grammars
embedded in the C extension to constrain LLM output to valid JSON. This means
the model *cannot* produce malformed output — every response is guaranteed to
parse as the expected schema.

### The Bulk SQL Pipeline (Section 6)

The key insight: since NER and RE are scalar SQL functions, they compose in
standard SQL without any application code:

```sql
WITH ner AS (
    SELECT id, content,
           muninn_extract_entities('model', content, 'person,organization,location') AS entities_json
    FROM documents
)
SELECT id,
       muninn_extract_relations('model', content, entities_json) AS relations_json
FROM ner
```

This runs NER on every document, then feeds the entity JSON directly into
relation extraction — all in a single query. No Python loop, no intermediate
storage, no serialisation overhead.

## Testing

```bash
# Registration and error handling tests (no model needed)
uv run -m pytest pytests/test_chat_gguf.py -v

# Full integration tests with a real model
MUNINN_CHAT_MODEL=models/Qwen3-4B-Q4_K_M.gguf uv run -m pytest pytests/test_chat_gguf.py -v
```
