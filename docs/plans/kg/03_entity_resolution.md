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

## Files

- **MODIFY**: `benchmarks/harness/treatments/kg_resolve.py`
