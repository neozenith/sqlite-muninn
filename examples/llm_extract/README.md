# LLM Extract — Comparing muninn GGUF vs GLiNER2

Consolidated benchmark comparing structured information extraction across
multiple GGUF chat models (via muninn SQL functions) and GLiNER2 (205M span
extraction model) on 5 curated documents.

## Models

### GGUF Chat Models (via muninn)

| Model | Params | Quant | Size | Context | License |
|-------|--------|-------|------|---------|---------|
| **Qwen3-4B** | 4B | Q4_K_M | ~2.5 GB | 32K | Apache 2.0 |
| **Qwen3-8B** | 8B | Q4_K_M | ~5.0 GB | 32K | Apache 2.0 |
| **Gemma-3-4B** | 3.8B | Q4_K_M | ~2.5 GB | 8K | Gemma |

### Comparison Baseline

| Model | Params | Size | Type |
|-------|--------|------|------|
| **GLiNER2** | 205M | ~400 MB | Span extraction (NER + RE) |

## Sections

| # | Section | What It Compares |
|---|---------|------------------|
| 1 | **Model Loading** | Load all GGUF models + GLiNER2, measure load times |
| 2 | **Chat Completion** | Free-form text generation via `muninn_chat()` (GGUF only) |
| 3 | **Summarisation** | Text summarisation via `muninn_summarize()` (GGUF only) |
| 4 | **NER Comparison** | `muninn_extract_entities()` per model vs GLiNER2 `batch_extract_entities()` |
| 5 | **RE Comparison** | `muninn_extract_relations()` per model vs GLiNER2 `batch_extract_relations()` |
| 6 | **Combined NER+RE** | `muninn_extract_ner_re()` (1 LLM call/doc) vs GLiNER2 (2 batch calls) |
| 7 | **CTE Pipeline** | SQL CTE chaining NER→RE (2 LLM calls/doc) vs GLiNER2 combined |
| 8 | **Summary Tables** | Side-by-side timing, speedup, and extraction count tables |

## Prerequisites

```bash
make all            # Build muninn extension
uv add gliner2      # Install GLiNER2 for comparison baseline
```

## Run

```bash
# Models auto-download on first run (~10 GB total for all 3 GGUF models)
uv run examples/llm_extract/example.py
```

## Data

5 curated documents covering corporate, financial, scientific, and business domains:

1. "Alice Smith founded ACME Corporation in New York City in 1987."
2. "Bob Jones, CEO of TechStart, announced a partnership with ACME Corporation."
3. "The European Central Bank raised interest rates to combat inflation in the eurozone."
4. "Dr. Marie Curie discovered radium at the University of Paris in 1898."
5. "Amazon acquired Whole Foods for $13.7 billion, reshaping the grocery industry."

## Key Architectural Differences

### Grammar-Constrained Generation (muninn)

`muninn_extract_entities`, `muninn_extract_relations`, and `muninn_extract_ner_re` use
GBNF grammars embedded in the C extension to constrain LLM output at the **token level**.
The grammar sampler rejects invalid tokens during generation, so output is guaranteed to
be well-formed JSON matching the expected schema — no post-processing needed.

### Span Extraction (GLiNER2)

GLiNER2 uses a 205M-param DeBERTa-based model trained for token classification.
Entities are extracted as spans with start/end positions. Relations are extracted
as head-tail pairs. No generative LLM is involved.

### RE Input Asymmetry

- **muninn** `muninn_extract_relations()` requires prior NER output — it chains off
  extracted entities to find relations between them.
- **GLiNER2** `batch_extract_relations()` operates directly on raw text — no entity
  dependency. This makes the standalone RE comparison (Section 5) structurally asymmetric.

### The CTE Pipeline (Section 7)

Since NER and RE are scalar SQL functions, they compose in standard SQL:

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

This runs NER on every document, then feeds the entity JSON directly into relation
extraction — all in a single query. No Python loop, no intermediate storage.

## Testing

```bash
# Registration and error handling tests (no model needed)
uv run -m pytest pytests/test_chat_gguf.py -v

# Full integration tests with a real model
MUNINN_CHAT_MODEL=models/Qwen3-4B-Q4_K_M.gguf uv run -m pytest pytests/test_chat_gguf.py -v
```

# Results

```text
  Absolute Timings (5 documents)
  ---------------------------------------------------------------------------------
                       Qwen3.5-4B      Qwen3.5-9B      Gemma-3-4B         GLiNER2
  ---------------------------------------------------------------------------------
         NER only          18.57s          34.78s          19.08s           0.39s
          RE only          23.20s          34.72s          21.22s           0.42s
  Combined NER+RE          38.71s          54.17s          40.82s           0.81s
     CTE Pipeline          61.57s          93.81s          54.85s           0.81s
  ---------------------------------------------------------------------------------

  Per-Document Timings
  ---------------------------------------------------------------------------------
                       Qwen3.5-4B      Qwen3.5-9B      Gemma-3-4B         GLiNER2
  ---------------------------------------------------------------------------------
         NER only      3.714s/doc      6.956s/doc      3.815s/doc      0.078s/doc
          RE only      4.639s/doc      6.945s/doc      4.244s/doc      0.084s/doc
  Combined NER+RE      7.741s/doc     10.835s/doc      8.165s/doc      0.161s/doc
     CTE Pipeline     12.314s/doc     18.762s/doc     10.970s/doc      0.161s/doc
  ---------------------------------------------------------------------------------

  Speedup (vs slowest per metric)
  ---------------------------------------------------------------------------------
                       Qwen3.5-4B      Qwen3.5-9B      Gemma-3-4B         GLiNER2
  ---------------------------------------------------------------------------------
         NER only            1.9x            1.0x            1.8x           89.6x
          RE only            1.5x            1.0x            1.6x           82.9x
  Combined NER+RE            1.4x            1.0x            1.3x           67.1x
     CTE Pipeline            1.5x            1.0x            1.7x          116.3x
```

So whilst the `muninn_extract_entities` and `muninn_extract_relations` is available, there are smaller task focused models that perform better and faster than throwing a whole LLM at the task.