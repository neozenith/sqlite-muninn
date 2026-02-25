# Phase 5: Analysis & Charts

## Status

Charts are **already implemented** in `benchmarks/harness/analysis/charts_kg.py`. This phase verifies they work once data flows through the pipeline.

## Existing Chart Specs

1. **kg_extraction_speed** — NER speed by model (ms/chunk)
2. **kg_extraction_entity_count** — Entity count by model
3. **ner_entity_f1_by_model** — Entity F1 by NER model (grouped by dataset)
4. **ner_precision_recall** — Precision vs recall scatter
5. **ner_speed_vs_f1** — Speed vs quality Pareto frontier
6. **re_triple_f1_by_model** — Triple F1 by RE model
7. **re_speed_vs_f1** — RE speed vs quality
8. **er_pairwise_f1** — ER pairwise F1 by dataset
9. **graphrag_retrieval_quality** — Passage recall by entry+expansion

## Additional Charts (if needed)

- **ER B-Cubed F1** — `bcubed_f1` metric (add if ER pipeline produces it)
- **NER per-dataset breakdown** — heatmap of model x dataset F1 values
