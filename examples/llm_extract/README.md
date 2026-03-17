# LLM Extract — Comparing muninn GGUF vs GLiNER2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_extract/example.ipynb)

Consolidated benchmark comparing structured information extraction across
multiple GGUF chat models (via muninn SQL functions) and GLiNER2 (205M span
extraction model) on 5 curated documents.

## Models

### GGUF Chat Models (via muninn)

| Model | Params | Quant | Size | Context | License |
|-------|--------|-------|------|---------|---------|
| **Qwen3.5-0.8B** | 0.8B | Q4_K_M | ~0.5 GB | 8K | Apache 2.0 |
| **Qwen3.5-2B** | 2B | Q4_K_M | ~1.3 GB | 8K | Apache 2.0 |
| **Qwen3.5-4B** | 4B | Q4_K_M | ~2.7 GB | 8K | Apache 2.0 |
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
# Models auto-download on first run (~7 GB total for all 4 GGUF models)
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
  -------------------------------------------------------------------------------------------------
                     Qwen3.5-0.8B      Qwen3.5-2B      Qwen3.5-4B      Gemma-3-4B         GLiNER2
  -------------------------------------------------------------------------------------------------
         NER only         705.60s          17.14s          21.34s          21.03s           0.32s
          RE only          13.32s          19.87s          23.86s          22.22s           0.44s
  Combined NER+RE        1866.08s          66.62s          41.32s          40.38s           0.76s
     CTE Pipeline        1450.48s          53.35s          67.57s          60.35s           0.76s
  -------------------------------------------------------------------------------------------------

  Per-Document Timings
  -------------------------------------------------------------------------------------------------
                     Qwen3.5-0.8B      Qwen3.5-2B      Qwen3.5-4B      Gemma-3-4B         GLiNER2
  -------------------------------------------------------------------------------------------------
         NER only    141.121s/doc      3.428s/doc      4.268s/doc      4.205s/doc      0.064s/doc
          RE only      2.664s/doc      3.974s/doc      4.773s/doc      4.444s/doc      0.088s/doc
  Combined NER+RE    373.216s/doc     13.324s/doc      8.264s/doc      8.076s/doc      0.153s/doc
     CTE Pipeline    290.095s/doc     10.669s/doc     13.514s/doc     12.071s/doc      0.153s/doc
  -------------------------------------------------------------------------------------------------

  Speedup (vs slowest per metric)
  -------------------------------------------------------------------------------------------------
                     Qwen3.5-0.8B      Qwen3.5-2B      Qwen3.5-4B      Gemma-3-4B         GLiNER2
  -------------------------------------------------------------------------------------------------
         NER only            1.0x           41.2x           33.1x           33.6x         2195.9x
          RE only            1.8x            1.2x            1.0x            1.1x           54.0x
  Combined NER+RE            1.0x           28.0x           45.2x           46.2x         2445.0x
     CTE Pipeline            1.0x           27.2x           21.5x           24.0x         1900.5x
  -------------------------------------------------------------------------------------------------
```

The Qwen3.5-0.8B model is pathologically slow for structured extraction — the grammar constraint prevents malformed JSON but can't stop the tiny model from generating excessive repetitive tokens, ballooning NER to ~141s/doc. The 2B+ models are ~3-5s/doc which is reasonable for LLM extraction but still 50-80x slower than GLiNER2 on these short documents. So whilst `muninn_extract_entities` and `muninn_extract_relations` are available, there are smaller task-focused models that perform better and faster than throwing a whole LLM at the task.