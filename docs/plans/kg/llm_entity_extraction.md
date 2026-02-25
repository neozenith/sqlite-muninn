# LLM Entity Extraction — Ollama Local Models

Captured: 2026-02-16. Split from `kg_extraction_benchmark_spec.md`.

This document covers the Ollama-hosted local models purpose-built for entity and KG triple extraction, including setup, prompt formats, and the broader model matrix for NER benchmarking.

---

## Table of Contents

1. [CLI Commands](#1-cli-commands)
2. [Model Details](#2-model-details)
3. [NuExtract Prompt Format](#3-nuextract-prompt-format)
4. [Integration Architecture](#4-integration-architecture)
5. [Full Ollama Model Matrix for NER Benchmarking](#5-full-ollama-model-matrix-for-ner-benchmarking)

---

## 1. CLI Commands

```bash
# Start server (if not running as a system service)
ollama serve

# NuExtract — purpose-built for structured extraction (Phi-3 fine-tune)
ollama pull nuextract                      # 2.2 GB default (Q4_0)
ollama pull nuextract:3.8b-q8_0           # 4.1 GB higher quality
ollama pull sroecker/nuextract-tiny-v1.5   # 494M ultra-light variant

# Triplex — purpose-built for KG triple extraction (Phi-3 fine-tune)
ollama pull sciphi/triplex                 # 2.4 GB default (3.8B)
ollama pull sciphi/triplex:1.5b            # 1.1 GB tiny variant (32K context)
```

---

## 2. Model Details

| Model | Base | Params | Q4 Size | Context | Purpose | License |
|-------|------|--------|---------|---------|---------|---------|
| `nuextract` | Phi-3-mini | 3.8B | 2.2 GB | 4K | Structured data extraction | MIT |
| `nuextract:3.8b-q8_0` | Phi-3-mini | 3.8B | 4.1 GB | 4K | Higher quality extraction | MIT |
| `sroecker/nuextract-tiny-v1.5` | Qwen 2.5 | 0.5B | ~0.4 GB | 4K | Ultra-light extraction | MIT |
| `sciphi/triplex` | Phi-3-mini | 3.8B | 2.4 GB | 4K | KG triple extraction | CC-BY-NC-SA-4.0* |
| `sciphi/triplex:1.5b` | — | 1.5B | 1.1 GB | 32K | Tiny triple extraction | CC-BY-NC-SA-4.0* |

*Triplex: NC license waived for orgs under $5M annual revenue.

---

## 3. NuExtract Prompt Format

NuExtract uses a specific template:
```
<|input|>
{text to extract from}
<|output|>
```

**Critical:** Always set `temperature: 0` for extraction. NuExtract is purely extractive — all output text should be present in the input.

---

## 4. Integration Architecture

```
┌────────────────────┐     HTTP      ┌──────────────────┐
│  Python benchmark  │ ──────────→   │  ollama serve    │
│  (ollama SDK)      │ ←──────────   │  (localhost:11434)│
│                    │   JSON        │  loads GGUF model │
└────────────────────┘               └──────────────────┘
```

The Python SDK is a thin HTTP client. It **cannot** load models directly — the server must be running. Configure via `OLLAMA_HOST` env var.

---

## 5. Full Ollama Model Matrix for NER Benchmarking

Beyond the purpose-built extraction models, these general-purpose models are worth benchmarking for entity extraction quality:

| Model | Ollama Tag | Params | Q4 Size | Good at Extraction? |
|-------|-----------|--------|---------|-------------------|
| NuExtract | `nuextract` | 3.8B | 2.2 GB | Best — purpose-built |
| Triplex | `sciphi/triplex` | 3.8B | 2.4 GB | Best for KG triples |
| Qwen 2.5 7B | `qwen2.5:7b` | 7B | ~4.5 GB | Strong JSON adherence |
| Llama 3.2 3B | `llama3.2:3b` | 3B | ~2.0 GB | Decent, good for person entities |
| Gemma 2 9B | `gemma2:9b` | 9B | ~5.2 GB | Highest accuracy, tight on RAM |
| Phi-3.5 mini | `phi3.5` | 3.8B | ~2.3 GB | Good instruction following |
| Mistral 7B | `mistral:7b` | 7B | ~4.1 GB | Strong unique entity extraction |
