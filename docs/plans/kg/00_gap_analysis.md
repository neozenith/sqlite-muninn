# KG Benchmark Suite — Gap Analysis

## Overview

The KG benchmark suite benchmarks NER, RE, ER, and GraphRAG pipelines using muninn as infrastructure. The Treatment wrappers, metrics functions, dataset prep, and registry wiring are complete. The gap is in **model adapter implementations** and **pipeline logic**.

## What's Done

| Component | File | Status |
|-----------|------|--------|
| `NerModelAdapter` ABC | `kg_types.py` | Complete |
| `EntityMention` dataclass | `kg_types.py` | Complete |
| NER adapters (7) | `kg_ner_adapters.py` | Complete: GLiNER (3), NuNerZero, GNER-T5 (2), spaCy |
| `KGNerExtractionTreatment` | `kg_extract.py` | Complete — gutenberg + NER dataset modes, F1 computation |
| RE adapters (3) | `kg_re_adapters.py` | Complete: GLiREL, spaCy SVO, EntityPair |
| `KGRelationExtractionTreatment` | `kg_re.py` | Complete — NER+RE pipeline, triple F1 |
| `KGEntityResolutionTreatment` | `kg_resolve.py` | Complete — HNSW blocking + Jaro-Winkler + Leiden |
| `KGGraphRAGTreatment` | `kg_graphrag.py` | Complete — VSS/BM25 entry + BFS expansion |
| `kg_metrics.py` | All 3 functions | Complete: `entity_micro_f1`, `triple_f1`, `bcubed_f1` |
| Dataset prep | `prep/kg_datasets.py` | Complete: 14 datasets |
| Registry | `registry.py` | Complete |
| Constants | `common.py` | Complete — 7 NER models, 10 datasets |
| Charts | `analysis/charts_kg.py` | Complete — 11 chart specs |

## Implementation Roadmap

1. **Phase 1**: NER model adapters — DONE
2. **Phase 2**: RE model adapters — DONE
3. **Phase 3**: ER pipeline — DONE
4. **Phase 4**: GraphRAG retrieval — DONE
5. **Phase 5**: Charts — DONE (verify after data flows)
