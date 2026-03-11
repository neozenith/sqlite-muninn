# Phase 3: Entity Resolution

## Goal

Implement the ER pipeline in `kg_resolve.py` for both KG coalescing and Febrl benchmark evaluation.

## Mode 1: KG Coalescing (`_run_kg_coalesce`)

Same algorithm as `demo_builder/phases/entity_resolution.py`:

1. Load entities from NER extraction output
2. Embed entity names with SentenceTransformer
3. HNSW blocking — insert embeddings, KNN search for candidate pairs (cosine distance < 0.4)
4. Matching cascade — exact → case-insensitive → Jaro-Winkler → combined score
5. Leiden clustering via `graph_leiden` TVF
6. Metrics: nodes_before/after, singleton_ratio, timing

Requires: `requires_muninn = True` (HNSW + Leiden TVFs)

## Mode 2: ER Benchmark (`_run_er_dataset`)

1. Load Febrl parquet data
2. Ground truth: `rec-N-dup-M` IDs → records with same N are true matches
3. Block using name field embeddings via HNSW
4. Match via Jaro-Winkler on name fields
5. Cluster via Leiden
6. Evaluate: `bcubed_f1` and pairwise F1 from `kg_metrics.py`

## Offline Preparation

```bash
# Download SentenceTransformer models used by entity resolution
uv run -m benchmarks.harness prep kg-models --model all-MiniLM-L6-v2
uv run -m benchmarks.harness prep kg-models --model nomic-embed-text-v1.5
```

### SentenceTransformer offline loading pattern

SentenceTransformer models are self-contained — no separate backbone download needed.
Passing the local snapshot path directly bypasses all hub lookup:

```python
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Resolve to local snapshot path
path = snapshot_download("nomic-ai/nomic-embed-text-v1.5", local_files_only=True)
# Local path bypasses HF hub entirely — no offline_mode() patch required
model = SentenceTransformer(path, trust_remote_code=True)
```

This is in contrast to GLiNER/GLiREL which need `offline_mode()` because they call
`AutoTokenizer.from_pretrained(repo_id)` (a hub URL, not a local path) internally.

## Files

- **MODIFY**: `benchmarks/harness/treatments/kg_resolve.py`
- `benchmarks/harness/prep/kg_models.py` — includes ST models in registry
- `benchmarks/demo_builder/phases/entity_embeddings.py` — same pattern for demo_builder
