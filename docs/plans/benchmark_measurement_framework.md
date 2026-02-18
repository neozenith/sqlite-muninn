# Benchmark Measurement Framework

Captured: 2026-02-16. Split from `kg_extraction_benchmark_spec.md`.

This document defines the JSONL schemas, results storage conventions, and metric definitions used to capture benchmark data for the muninn KG pipeline. Results feed into mkdocs benchmark charts.

---

## Table of Contents

1. [Per-Model Extraction Metrics](#1-per-model-extraction-metrics)
2. [Per-Coalescing Metrics](#2-per-coalescing-metrics)
3. [Results Storage](#3-results-storage)
4. [Integration with mkdocs Benchmark Charts](#4-integration-with-mkdocs-benchmark-charts)

---

## 1. Per-Model Extraction Metrics

Every extraction benchmark run records:

```python
{
    "model_id": "urchade/gliner_small-v2.1",
    "model_type": "gliner",           # gliner | gner | llm_api | ollama
    "dataset": "conll2003",
    "chunk_count": 100,
    "timestamp": "2026-02-16T12:00:00Z",

    # Quality
    "entity_precision": 0.82,
    "entity_recall": 0.75,
    "entity_f1": 0.78,

    # Performance
    "total_time_s": 12.5,
    "avg_ms_per_chunk": 125.0,
    "peak_memory_mb": 1800,

    # Cost (LLM only)
    "input_tokens": 50000,
    "output_tokens": 12000,
    "cost_usd": 0.045,

    # Model metadata
    "params_millions": 166,
    "quantization": null,             # or "q4_0", "q8_0" for Ollama
}
```

---

## 2. Per-Coalescing Metrics

```python
{
    "method": "hnsw_blocking+jaro_winkler+leiden",
    "dataset": "dblp_acm",
    "blocking_threshold": 0.4,

    # Quality
    "pairwise_precision": 0.91,
    "pairwise_recall": 0.87,
    "pairwise_f1": 0.89,
    "bcubed_f1": 0.85,

    # Graph quality
    "nodes_before": 5000,
    "nodes_after": 3200,
    "edges_before": 12000,
    "edges_after": 8500,
    "singleton_ratio": 0.12,
    "connected_components": 45,

    # Performance
    "blocking_time_s": 2.1,
    "matching_time_s": 0.8,
    "clustering_time_s": 0.3,
    "total_time_s": 3.2,
}
```

---

## 3. Results Storage

All results accumulate in JSONL files:

```
benchmarks/results/
  kg_extraction.jsonl      # NER model benchmarks
  kg_coalescing.jsonl      # Entity resolution benchmarks
  kg_llm_extraction.jsonl  # LLM API extraction benchmarks
  kg_graphrag.jsonl        # End-to-end GraphRAG query benchmarks (existing)
```

Each line is a self-contained JSON record with a `timestamp` field, allowing append-only accumulation and time-series analysis.

---

## 4. Integration with mkdocs Benchmark Charts

Benchmark results from the JSONL files are visualised in the project's mkdocs documentation site. Key chart types:

### Extraction Quality Comparison
- **X-axis:** Model ID
- **Y-axis:** Entity F1 score
- **Grouping:** By model type (gliner, gner, llm_api, ollama)
- **Source:** `kg_extraction.jsonl` + `kg_llm_extraction.jsonl`

### Cost vs Quality Trade-off
- **X-axis:** Cost USD per 1K chunks
- **Y-axis:** Entity F1 score
- **Bubble size:** Inference time (ms/chunk)
- **Source:** `kg_llm_extraction.jsonl`

### Coalescing Pipeline Performance
- **X-axis:** Dataset
- **Y-axis:** Pairwise F1 / B-Cubed F1
- **Secondary axis:** Node reduction ratio (`nodes_after / nodes_before`)
- **Source:** `kg_coalescing.jsonl`

### Time-Series Tracking
- **X-axis:** Timestamp
- **Y-axis:** F1 score or latency
- **Purpose:** Track regressions and improvements across code changes
- **Source:** All JSONL files
