# LLM Extract — llama.cpp Chat Extensions Plan

> Extends the muninn SQLite extension with chat / completion inference capabilities
> built on the already-vendored `vendor/llama.cpp`. Adds `chat_gguf.c` alongside
> the existing `embed_gguf.c`, and layers Python convenience adapters on top.

---

## Motivation

`embed_gguf.c` uses llama.cpp's **embedding path** (pooled hidden states). The
same vendored binary also exposes a **completion path** (autoregressive sampling)
that is untapped. Completion + GBNF grammar-constrained decoding enables:

- **Structured NER** — `{"entities": [{"text": "...", "type": "...", "score": 0.9}]}`
- **Structured RE** — `{"relations": [{"head": "...", "rel": "...", "tail": "..."}]}`
- **LLM-CER** — cluster-level entity resolution judgment
- **Community naming** — summarise a Leiden community's nodes into a human label

All without any new external dependencies (llama.cpp is already vendored).

---

## Architecture Overview

```
muninn.c
├── embed_gguf.c          (existing) — embedding path, pooled hidden states
└── chat_gguf.c           (NEW)      — completion path, GBNF grammar support
    ├── muninn_chat(model, prompt [, grammar])           — scalar SQL function
    ├── muninn_extract_entities(model, text, labels)     — NER convenience wrapper
    ├── muninn_extract_relations(model, text, ents_json) — RE convenience wrapper
    └── muninn_summarize(model, text [, max_tokens])     — summarisation wrapper
```

Python adapters (no C changes needed):
```
benchmarks/harness/treatments/
├── kg_ner_adapters.py    — add LlmNerAdapter(model_path, grammar)
├── kg_re_adapters.py     — add LlmREAdapter(model_path, grammar)
└── kg_resolve.py         — add LlmERAdapter(model_path) alongside Leiden stage

benchmarks/demo_builder/phases/
└── community_naming.py   (NEW) — PhaseCommunitySummarisation
```

---

## Phase 1 — `chat_gguf.c`: Core Completion Infrastructure

### 1.1 Model Registry Extension

`embed_gguf.c` already maintains a `g_models[MAX_MODELS]` array with embedding
contexts. Chat models require a **different context configuration**:

| Parameter | Embedding context | Chat context |
|---|---|---|
| `llama_context_params.embeddings` | `true` | `false` |
| `llama_context_params.n_ctx` | model default | `ctx_len` (e.g. 4096) |
| `llama_sampler_chain` | N/A | nucleus + temperature |
| `llama_batch` | multi-seq pooled | single-seq autoregressive |

Options:
- **Separate registry** (`g_chat_models[]`) — clean separation, no cross-contamination
- **Shared registry with type flag** — reduces total model slots

Recommendation: **separate registry** (`g_chat_models[MAX_CHAT_MODELS]` with
`MAX_CHAT_MODELS 8`). Chat models are much larger than embedding models and will
rarely coexist with many embedding models.

### 1.2 Model Loading — `muninn_chat_model(path)`

Mirrors `muninn_embed_model()` from `embed_gguf.c`. Returns an opaque handle stored
in the registry:

```sql
INSERT INTO temp.muninn_models(name, model)
SELECT 'qwen3-8b', muninn_chat_model('models/qwen3-8b-q4_k_m.gguf');
```

Loading parameters:
- `llama_model_params.n_gpu_layers = -1` (offload all layers; falls back to CPU)
- `llama_context_params.n_ctx = 4096` (default; configurable via optional arg)
- Sampler chain: temperature=0.0, top-p=1.0 (deterministic for extraction tasks)

### 1.3 Core SQL Function — `muninn_chat(model, prompt [, grammar_gbnf])`

```sql
-- Plain completion
SELECT muninn_chat('qwen3-8b', 'Extract entities from: "Alice works at ACME."');

-- Grammar-constrained JSON
SELECT muninn_chat('qwen3-8b',
    'Extract entities as JSON from: "Alice works at ACME."',
    '{"type": "object", "properties": {"entities": {"type": "array", ...}}}'
);
```

**Grammar parameter:** Accepts a JSON Schema string. `chat_gguf.c` converts it to
GBNF internally using llama.cpp's `json_schema_to_grammar()` utility (available
since llama.cpp b3000+). This means callers always provide JSON Schema (portable,
readable) rather than raw GBNF.

**Implementation notes:**
- Prompt is formatted using the model's built-in chat template via
  `llama_chat_apply_template()` — no hardcoded ChatML or Llama templates in C
- Max new tokens: configurable via a 4th argument (default 512 for extraction tasks)
- Output is NULL-terminated UTF-8 text returned as `SQLITE_TEXT`

### 1.4 Convenience SQL Functions

These are thin wrappers over `muninn_chat()` with pre-baked JSON Schema grammars:

#### `muninn_extract_entities(model, text, labels_csv)`

```sql
SELECT muninn_extract_entities(
    'qwen3-8b',
    'Alice founded ACME Corp in 1987 in New York.',
    'person,organization,location,date'
);
-- → '{"entities":[{"text":"Alice","type":"person","score":0.97},...]}'
```

Built-in JSON Schema (compiled to GBNF internally):
```json
{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text":  {"type": "string"},
          "type":  {"type": "string"},
          "score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["text", "type"]
      }
    }
  },
  "required": ["entities"]
}
```

#### `muninn_extract_relations(model, text, entities_json)`

```sql
SELECT muninn_extract_relations(
    'qwen3-8b',
    'Alice founded ACME Corp in 1987.',
    '[{"text":"Alice","type":"person"},{"text":"ACME Corp","type":"organization"}]'
);
-- → '{"relations":[{"head":"Alice","rel":"founded","tail":"ACME Corp","score":0.91}]}'
```

Entities JSON is passed back into the prompt to constrain which spans the model
considers — prevents hallucinating entity text not present in the input.

#### `muninn_summarize(model, text [, max_tokens])`

```sql
SELECT muninn_summarize('qwen3-8b',
    'Community nodes: Alice (person), ACME Corp (organization), New York (location). '
    'Relations: Alice founded ACME Corp; ACME Corp located_in New York.',
    64
);
-- → 'Alice founded ACME Corp in New York.'
```

Used by the community naming phase (see Phase 4).

---

## Phase 2 — Python Adapters for the Benchmark Harness

New adapters implementing the existing ABCs in `kg_ner_adapters.py` and
`kg_re_adapters.py`. These use `llama-cpp-python` (Python binding) rather than the
SQL functions, for tighter integration with the batch processing pipeline.

> **Dependency:** `uv add llama-cpp-python` — Python bindings to llama.cpp.
> This is separate from the vendored C submodule; the Python package includes its
> own compiled binary. Accept the redundancy for now (C extension = SQL queries,
> Python binding = harness adapter batch inference).

### 2.1 `LlmNerAdapter`

```python
class LlmNerAdapter(NerModelAdapter):
    """NER via local GGUF chat model with JSON Schema grammar."""

    def __init__(self, model_path: str, ctx_len: int = 4096) -> None: ...

    def predict(self, texts: list[str], labels: list[str]) -> list[list[EntityMention]]: ...
```

Batch strategy: process texts individually (no batching — chat models are
autoregressive). For bulk pipelines, run with `n_parallel=4` via llama.cpp's
parallel decoding if hardware allows.

Prompt template (system + user, ChatML):
```
System: You are a precise named entity recognition system. Extract entities of the
        specified types. Respond only with valid JSON matching the schema.

User:   Extract entities of types: {labels_csv}
        Text: {text}
```

### 2.2 `LlmREAdapter`

```python
class LlmREAdapter(ReModelAdapter):
    """Relation extraction via local GGUF chat model."""

    def __init__(self, model_path: str, relation_types: list[str], ctx_len: int = 4096) -> None: ...
```

Key advantage over GLiREL: prompt can include the **full paragraph context** (not
just within-sentence token spans). Cross-sentence relations are naturally handled.

Prompt includes the entity list extracted in the NER phase — the model is
constrained to only emit relations between those spans, preventing hallucination of
new entity text.

### 2.3 `LlmERAdapter`

Implements the **LLM-CER** (in-context clustering) pattern from [arXiv 2506.02509](https://arxiv.org/html/2506.02509v1).

```python
class LlmERAdapter:
    """LLM-based cluster judgment for entity resolution.

    Replaces/augments the Jaro-Winkler + Leiden stages in kg_resolve.py.
    Only invoked for candidate pairs where cosine similarity is in the
    ambiguous range (0.4 – 0.7). Clear matches (> 0.7) and clear mismatches
    (< 0.4) are handled by the existing threshold logic.
    """

    def should_merge(self, candidates: list[str]) -> list[list[str]]: ...
    """Return list of merge groups from a set of candidate entity name strings."""
```

Prompt presents the candidate cluster to the model:
```
Given these candidate entity mentions, group those that refer to the same
real-world entity. Return JSON: {"groups": [["mention1", "mention2"], ...]}.

Candidates: ["Alice Smith", "A. Smith", "Alice S.", "Bob Jones"]
```

**Cost gating:** Only call the LLM when `0.4 < cosine_similarity < 0.7` — the
region where string similarity and embedding similarity disagree. This bounds the
number of LLM calls to the ambiguous fraction of the candidate pairs.

### 2.4 Model Catalog

Full catalog of confirmed GGUF chat models for use in `benchmarks/harness/prep/gguf_models.py`
and the `examples/llm_extract/` example. All URLs verified against official HuggingFace repos.

#### Tier 1 — Recommended for NER/RE extraction tasks

| Name | Params | File | Size | Context | License | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-4B** | 4B | `qwen3-4b-q4_k_m.gguf` | ~2.5 GB | 32K / 128K | Apache 2.0 | **Default example model.** Best sub-5B quality; tool-calling, JSON output; `/no_think` disables chain-of-thought |
| **Qwen3-8B** | 8B | `qwen3-8b-q4_k_m.gguf` | ~5.0 GB | 32K / 128K | Apache 2.0 | Stronger extraction; fits 8 GB unified memory with Q4_K_M |
| **Qwen3.5-9B** | 9B | `Qwen3.5-9B-Q4_K_M.gguf` | ~5.7 GB | 262K | Apache 2.0 | Released 2026-03-02; 262K native context; beats much larger models on benchmarks; no official Qwen/ GGUF yet — using unsloth (YC S24, trusted quantizer, fixes upstream model bugs) until official repo appears |
| **Gemma-3-4B-IT** | 4B | `gemma-3-4b-it-qat-q4_0.gguf` | ~2.6 GB | 128K | Gemma | QAT quantization — near-bfloat16 quality at Q4_0 size; official Google GGUF |
| **Phi-4-mini** | 3.8B | `phi-4-mini-instruct-q4_k_m.gguf` | ~2.5 GB | 128K | MIT | Exceptional structured output at 3.8B; MIT license |

#### Tier 2 — Alternatives and edge cases

| Name | Params | File | Size | Context | License | Notes |
|---|---|---|---|---|---|---|
| **Llama-3.2-3B** | 3B | `llama-3.2-3b-instruct-q4_k_m.gguf` | ~1.9 GB | 128K | Llama 3.2 | Lightest viable full-quality option |
| **Llama-3.1-8B** | 8B | `meta-llama-3.1-8b-instruct-q4_k_m.gguf` | ~4.9 GB | 128K | Llama 3.1 | Most widely deployed 8B; users may already have it from Ollama/LM Studio |
| **Llama-3.2-1B** | 1B | `llama-3.2-1b-instruct-q4_k_m.gguf` | ~0.7 GB | 128K | Llama 3.2 | Sanity-check / CI use only; instruction-following is basic |
| **SmolLM2-1.7B** | 1.7B | `smollm2-1.7b-instruct-q4_k_m.gguf` | ~1.1 GB | 8K | Apache 2.0 | Ultra-lightweight; 8K context limit |

#### GGUF download URLs

```python
GGUF_CHAT_MODELS: list[dict[str, str]] = [
    # ── Tier 1 ──────────────────────────────────────────────────────────
    {
        "name": "Qwen3-4B",
        "filename": "qwen3-4b-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf",
        "params": "4B",
        "size_gb": "2.5",
        "ctx_len": "32768",
        "license": "Apache-2.0",
        "task": "chat",
    },
    {
        "name": "Qwen3-8B",
        "filename": "qwen3-8b-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-q4_k_m.gguf",
        "params": "8B",
        "size_gb": "5.0",
        "ctx_len": "32768",
        "license": "Apache-2.0",
        "task": "chat",
    },
    {
        "name": "Qwen3.5-9B",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
        "params": "9B",
        "size_gb": "5.7",
        "ctx_len": "262144",
        "license": "Apache-2.0",
        "task": "chat",
    },
    {
        "name": "Gemma-3-4B-IT",
        "filename": "gemma-3-4b-it-qat-q4_0.gguf",
        "url": "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/gemma-3-4b-it-qat-q4_0.gguf",
        "params": "4B",
        "size_gb": "2.6",
        "ctx_len": "131072",
        "license": "Gemma",
        "task": "chat",
    },
    {
        "name": "Phi-4-mini",
        "filename": "phi-4-mini-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        "params": "3.8B",
        "size_gb": "2.5",
        "ctx_len": "131072",
        "license": "MIT",
        "task": "chat",
    },
    # ── Tier 2 ──────────────────────────────────────────────────────────
    {
        "name": "Llama-3.2-3B",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "params": "3B",
        "size_gb": "1.9",
        "ctx_len": "131072",
        "license": "Llama-3.2",
        "task": "chat",
    },
    {
        "name": "Llama-3.1-8B",
        "filename": "meta-llama-3.1-8b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "params": "8B",
        "size_gb": "4.9",
        "ctx_len": "131072",
        "license": "Llama-3.1",
        "task": "chat",
    },
    {
        "name": "Llama-3.2-1B",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf",
        "params": "1B",
        "size_gb": "0.7",
        "ctx_len": "131072",
        "license": "Llama-3.2",
        "task": "chat",
    },
    {
        "name": "SmolLM2-1.7B",
        "filename": "smollm2-1.7b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf",
        "params": "1.7B",
        "size_gb": "1.1",
        "ctx_len": "8192",
        "license": "Apache-2.0",
        "task": "chat",
    },
]
```

---

## Phase 3 — Benchmark Harness Integration

### 3.1 New Permutations

The existing `NER_ADAPTERS` dict in `kg_ner_adapters.py` uses factory lambdas:

```python
NER_ADAPTERS: dict[str, Callable[[], NerModelAdapter]] = {
    # existing ...
    "llm-qwen3-4b": lambda: LlmNerAdapter(GGUF_CHAT_MODELS_DIR / "qwen3-4b-q4_k_m.gguf"),
    "llm-qwen3-8b": lambda: LlmNerAdapter(GGUF_CHAT_MODELS_DIR / "qwen3-8b-q4_k_m.gguf"),
}
```

Similarly for `RE_ADAPTERS`. These new slugs are automatically picked up by the
registry → benchmark harness → chart pipeline with **zero other changes**.

### 3.2 Benchmark Metrics

LLM-based adapters produce the same `entity_micro_f1` and `triple_f1` metrics —
the harness is model-agnostic. New metrics to add for LLM-specific analysis:

| Metric | Key | Description |
|---|---|---|
| Tokens used | `llm_tokens_in` / `llm_tokens_out` | Cost proxy |
| Latency per chunk | `llm_ms_per_chunk` | Speed comparison vs encoder models |
| Grammar violations | `llm_parse_failures` | Should be 0 with GBNF; non-zero = bug |

---

## Phase 4 — Community Summarisation

### 4.1 New Phase: `PhaseCommunitySummarisation`

```
benchmarks/demo_builder/phases/community_naming.py
benchmarks/sessions_demo/phases/community_naming.py  (or reuse via duck typing)
```

Runs after `PhaseEntityResolution` (which produces Leiden community assignments).
For each community with ≥ 3 members:

1. Query the `entity_clusters` + `entities` tables for member names and types
2. Query `relations` for edges within the community
3. Format as a text summary prompt
4. Call `muninn_summarize()` SQL function (Phase 1) or `LlmSummarizer` Python class
5. Write result to `community_labels` table

```sql
CREATE TABLE community_labels (
    community_id INTEGER PRIMARY KEY,
    label        TEXT NOT NULL,    -- e.g. "Alice's Work at ACME Corp"
    summary      TEXT,             -- longer description (optional)
    model        TEXT NOT NULL,    -- which GGUF generated this
    generated_at TEXT NOT NULL
);
```

### 4.2 Viz Integration

The `community_labels` table is surfaced in the KG Pipeline Explorer to replace
the current numeric community IDs (`community_0`, `community_1`) with
human-readable names. The viz backend's graph endpoint includes `community_label`
in node metadata if the table exists.

### 4.3 Community Size Threshold

- Communities with 1–2 members: skip (no meaningful summarisation possible)
- Communities with 3–10 members: compact list prompt (`~200 tokens`)
- Communities with > 10 members: top-10 by entity frequency + `"and N others"`

---

## Phase 5 — Paid API Provider Adapters

Same adapter ABCs, different backends. Add to `kg_ner_adapters.py`:

```python
class AnthropicNerAdapter(NerModelAdapter):
    """NER via Anthropic Claude API (tool_use / structured output)."""
    def __init__(self, model: str = "claude-sonnet-4-6") -> None: ...

class OpenAiNerAdapter(NerModelAdapter):
    """NER via OpenAI API (json_schema response format)."""
    def __init__(self, model: str = "gpt-4.1-mini") -> None: ...

class GeminiNerAdapter(NerModelAdapter):
    """NER via Google Gemini API (responseSchema parameter)."""
    def __init__(self, model: str = "gemini-2.0-flash") -> None: ...
```

These allow the 97-permutation benchmark to extend to ~130+ permutations covering
the full local-vs-API cost/quality tradeoff.

---

## Implementation Sequence

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | `chat_gguf.c` + SQL functions | `embed_gguf.c` pattern (already exists) |
| 2 | Python adapters (`LlmNerAdapter`, `LlmREAdapter`) | Phase 1 optional; can use `llama-cpp-python` standalone |
| 2b | `LlmERAdapter` | Phase 2 + `kg_resolve.py` understanding |
| 3 | Harness integration + new permutations | Phase 2 |
| 4 | `PhaseCommunitySummarisation` | Phase 1 (SQL path) or Phase 2 (Python path) |
| 5 | API provider adapters | Phase 2 (same ABCs) |

Phase 2 can start before Phase 1 is complete — `llama-cpp-python` is an independent
Python package that can drive the same GGUF models. Phase 1 (the C SQL function) is
valuable for the viz layer where direct SQL access is preferred.

---

## Model-Specific Implementation Notes

### Qwen3 thinking tokens

Qwen3 models (4B, 8B) support a dual thinking/non-thinking mode controlled by
the system prompt:

- **`/no_think`** in the system prompt disables chain-of-thought entirely —
  essential for `muninn_extract_entities` / `muninn_extract_relations` where the
  output must be clean JSON immediately.
- Without `/no_think`, the model may emit a `<think>...</think>` block before the
  JSON response. `chat_gguf.c` must strip everything up to and including the
  closing `</think>` token (ID 151668) before returning the result to SQLite.

Recommendation: always set `/no_think` in the system prompt for all extraction
convenience functions. Expose thinking mode only through the raw `muninn_chat()`
interface for callers who want it.

### Gemma 3 quantization format

Google's official Gemma 3 GGUFs use **QAT Q4_0** (quantization-aware training),
not the community-standard Q4_K_M. The QAT process bakes quantization into the
training loop, so Q4_0 quality is nearly identical to bfloat16. Use the official
`google/gemma-3-*-it-qat-q4_0-gguf` repo rather than community re-quantizations.

### Default context length

For per-document NER/RE tasks, 8K tokens is sufficient and keeps KV-cache memory
low. Default `n_ctx = 8192` in `chat_gguf.c`; document how users can override to
32K or 128K for long-document summarisation use cases.

---

## GBNF Grammar Notes

llama.cpp ships `json_schema_to_grammar()` in `common/json-schema-to-grammar.cpp`.
This converts JSON Schema → GBNF automatically. The C wrapper in `chat_gguf.c`
calls this utility, meaning callers never write raw GBNF — they provide portable
JSON Schema strings.

**Guarantees:**
- 100% well-formed JSON output (invalid tokens masked during sampling)
- No hallucinated keys or types
- Still allows the model to choose values (it is NOT a fill-in-the-blank template)

**Limitation:** Schema complexity affects generation speed. Keep schemas shallow
(max 2–3 levels of nesting) for extraction tasks.

---

## Example (`examples/llm_extract/`)

Alongside Phase 1 delivery, create `examples/llm_extract/example.py` and
`examples/llm_extract/README.md` following the same pattern as
`examples/text_embeddings/`. The example must:

- Require **no dependencies beyond the muninn extension** (`make all`)
- **Auto-download** the GGUF chat model to `models/` on first run (same
  `urllib.request` progress-bar pattern as `text_embeddings`)
- Use **Qwen3-4B Q4_K_M** (~2.5 GB) as the default model (Tier 1, Apache 2.0,
  best sub-5B quality); larger models (Qwen3-8B, Gemma-3-4B, Phi-4-mini)
  commented out with download size noted

### Sections

| # | Section | Key SQL |
|---|---------|---------|
| 1 | **Model Loading** | `muninn_chat_model()` + verify in `muninn_models` |
| 2 | **Plain Chat** | `muninn_chat(model, prompt)` — no grammar |
| 3 | **NER** | `muninn_extract_entities(model, text, labels_csv)` — parse + display JSON |
| 4 | **Relation Extraction** | `muninn_extract_relations(model, text, entities_json)` chaining off Section 3 |
| 5 | **Summarisation** | `muninn_summarize(model, context, 64)` — simulate community label generation |
| 6 | **Bulk SQL Pipeline** | `INSERT...SELECT` over a documents table; CTE NER→RE chain |

Section 6 is the most important to demonstrate: NER and RE chained entirely in SQL
with no Python orchestration, showing the core advantage of the scalar-function design:

```sql
WITH ner AS (
    SELECT id, content,
           muninn_extract_entities('Qwen3-4B', content, 'person,organization,location') AS entities_json
    FROM documents
)
SELECT id,
       muninn_extract_relations('Qwen3-4B', content, entities_json) AS relations_json
FROM ner;
```

### Run

```bash
# Model auto-downloads on first run (~2.4 GB)
python examples/llm_extract/example.py

# Enable llama.cpp internals logging
MUNINN_LOG_LEVEL=verbose python examples/llm_extract/example.py
```

---

## Related Plans

- `docs/plans/kg/01_ner_model_adapters.md` — existing NER adapter design
- `docs/plans/kg/02_relation_extraction.md` — existing RE adapter design
- `docs/plans/kg/03_entity_resolution.md` — existing ER pipeline design
- `docs/plans/llm_kg_research.md` — research snapshot + GGUF model catalog
- `docs/plans/gliner2_re_upgrade.md` — near-term GLiREL speed fix
- `docs/plans/gguf_ml_integration.md` — Phase 1 (embedding) retrospective
