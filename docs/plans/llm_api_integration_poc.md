# LLM API Integration POC

Captured: 2026-02-16. Split from `kg_extraction_benchmark_spec.md`.

A standalone proof-of-concept for calling LLM APIs (cloud and local Ollama) to perform entity extraction from reference text. The focus is on **API integration and performance metrics** (token counting, cost, latency) rather than the specific extraction task.

---

## Table of Contents

1. [Provider Configuration](#1-provider-configuration)
2. [Token Usage Tracking](#2-token-usage-tracking)
3. [Pricing Lookup Table](#3-pricing-lookup-table)
4. [Structured Output (JSON Schema)](#4-structured-output-json-schema)
5. [POC Validation Plan](#5-poc-validation-plan)
6. [POC Script](#6-poc-script)

---

## 1. Provider Configuration

| Provider | Env Var | Package | Install | Import |
|----------|---------|---------|---------|--------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` | `pip install anthropic` | `from anthropic import Anthropic` |
| OpenAI | `OPENAI_API_KEY` | `openai` | `pip install openai` | `from openai import OpenAI` |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY` — if both set, `GOOGLE_API_KEY` wins) | `google-genai` | `pip install google-genai` | `from google import genai` |
| Ollama | `OLLAMA_HOST` (default: `http://127.0.0.1:11434`) | `ollama` | `pip install ollama` | `import ollama` |

**Note:** `google-generativeai` is deprecated since Aug 2025. Use `google-genai`.

---

## 2. Token Usage Tracking

Each SDK reports token usage differently:

```python
# Anthropic
usage = {
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
}

# OpenAI
usage = {
    "input_tokens": response.usage.prompt_tokens,
    "output_tokens": response.usage.completion_tokens,
}

# Gemini
usage = {
    "input_tokens": response.usage_metadata.prompt_token_count,
    "output_tokens": response.usage_metadata.candidates_token_count,
}

# Ollama
usage = {
    "input_tokens": response["prompt_eval_count"],
    "output_tokens": response["eval_count"],
    "eval_duration_ns": response["eval_duration"],  # for tokens/sec
}
```

---

## 3. Pricing Lookup Table (Feb 2026)

All prices per 1M tokens.

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|------------|
| Google | Gemini 2.0 Flash-Lite | 0.075 | 0.30 |
| Google | Gemini 2.5 Flash-Lite | 0.10 | 0.40 |
| OpenAI | GPT-4o-mini | 0.15 | 0.60 |
| Google | Gemini 2.5 Flash | 0.30 | 2.50 |
| Anthropic | Claude Haiku 4.5 | 1.00 | 5.00 |
| OpenAI | GPT-4o | 2.50 | 10.00 |
| Anthropic | Claude Sonnet 4.5 | 3.00 | 15.00 |
| Ollama | (any local model) | 0.00 | 0.00 |

**Cost calculation:**
```python
cost = (usage["input_tokens"] / 1_000_000) * price_input + \
       (usage["output_tokens"] / 1_000_000) * price_output
```

---

## 4. Structured Output (JSON Schema)

All four providers support constrained JSON decoding. Use Pydantic for schema definition:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    entity_type: str
    confidence: float

class Relation(BaseModel):
    source: str
    target: str
    relation_type: str

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

| Provider | How to enforce schema |
|----------|---------------------|
| Anthropic | `client.messages.parse(output_format=ExtractionResult)` |
| OpenAI | `client.beta.chat.completions.parse(response_format=ExtractionResult)` |
| Gemini | `config={"response_mime_type": "application/json", "response_json_schema": ExtractionResult.model_json_schema()}` |
| Ollama | `format=ExtractionResult.model_json_schema()` + `options={"temperature": 0}` |

---

## 5. POC Validation Plan

Before building the full benchmark suite, validate each integration path with a small proof-of-concept.

### 5a. POC Scope

- **Text:** 10 chunks from Wealth of Nations (already cached)
- **Entity types:** `["person", "organization", "location", "economic concept", "commodity"]`
- **Gold standard:** Manually annotate the 10 chunks (~30 min)
- **Measure:** F1 score, time, cost per provider

### 5b. POC Order

| Step | Integration | Validates | Risk |
|------|------------|-----------|------|
| 1 | GLiNER small (existing) | Baseline NER quality | None (already works) |
| 2 | Ollama + NuExtract | Local LLM structured extraction | Server setup, prompt format |
| 3 | Ollama + Triplex | Local LLM KG triple extraction | Different output schema |
| 4 | OpenAI GPT-4o-mini | Cheapest API provider | API key, network |
| 5 | Anthropic Claude Haiku | Second cheapest API | API key, structured output |
| 6 | Google Gemini Flash-Lite | Cheapest overall | New SDK (`google-genai`) |

### 5c. POC Success Criteria

- Each integration can process 10 chunks and return valid structured JSON
- Token usage is captured for cost calculation
- Wall-clock time is recorded
- Output can be scored against gold-standard annotations
- No integration requires more than 20 lines of provider-specific code

---

## 6. POC Script

**Location:** `benchmarks/scripts/kg_extraction_poc.py`

CLI interface:
```bash
# Run specific provider
python benchmarks/scripts/kg_extraction_poc.py --provider gliner --model urchade/gliner_small-v2.1
python benchmarks/scripts/kg_extraction_poc.py --provider ollama --model nuextract
python benchmarks/scripts/kg_extraction_poc.py --provider openai --model gpt-4o-mini
python benchmarks/scripts/kg_extraction_poc.py --provider anthropic --model claude-haiku-4-5-20251001
python benchmarks/scripts/kg_extraction_poc.py --provider gemini --model gemini-2.0-flash-lite

# Run all available providers
python benchmarks/scripts/kg_extraction_poc.py --all

# Compare results
python benchmarks/scripts/kg_extraction_poc.py --compare
```
