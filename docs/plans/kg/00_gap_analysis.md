# KG Benchmark Suite — Gap Analysis

## Overview

The KG benchmark suite benchmarks NER, RE, ER, and GraphRAG pipelines using muninn as infrastructure. The Treatment wrappers, metrics functions, dataset prep, and registry wiring are complete. The gap is in **model adapter implementations** and **pipeline logic**.

## What's Done

| Component | File | Status |
|-----------|------|--------|
| `NerModelAdapter` ABC | `kg_extract.py:34-54` | Complete |
| `FTS5Adapter` | `kg_extract.py:56-85` | Complete |
| `EntityMention` dataclass | `kg_extract.py:24-31` | Complete |
| `KGNerExtractionTreatment` | `kg_extract.py:148-343` | Complete |
| `KGRelationExtractionTreatment` wrapper | `kg_re.py:50-192` | Wrapper complete, uses entity-pair proxy |
| `KGEntityResolutionTreatment` wrapper | `kg_resolve.py:18-123` | Wrapper complete, `_run_*` return zeroed metrics |
| `KGGraphRAGTreatment` wrapper | `kg_graphrag.py:18-91` | Wrapper complete, `run()` returns zeroed metrics |
| `kg_metrics.py` | All 3 functions | Complete: `entity_micro_f1`, `triple_f1`, `bcubed_f1` |
| Dataset prep | `prep/kg_datasets.py` | Complete: 14 datasets |
| Registry | `registry.py:133-194` | Complete |
| Constants | `common.py:341-524` | Complete |
| Charts | `analysis/charts_kg.py` | Complete |

## What's Stub/Missing

| Component | Gap | Phase |
|-----------|-----|-------|
| 7 of 8 `NER_ADAPTERS` entries are `None` | GLiNER (3), NuNerZero, GNER-T5 (2), spaCy | Phase 1 |
| RE model integration | No GLiREL/spaCy SVO adapter | Phase 2 |
| ER pipeline (`kg_resolve.py`) | `_run_kg_coalesce()` and `_run_er_dataset()` empty | Phase 3 |
| GraphRAG pipeline (`kg_graphrag.py`) | `run()` empty | Phase 4 |
| NER dataset label mapping | Hardcoded labels → need dataset-specific extraction | Phase 1 |

## Implementation Roadmap

1. **Phase 1**: NER model adapters (`kg_ner_adapters.py`)
2. **Phase 2**: RE model adapters (`kg_re_adapters.py`) + treatment update
3. **Phase 3**: ER pipeline in `kg_resolve.py`
4. **Phase 4**: GraphRAG retrieval in `kg_graphrag.py`
5. **Phase 5**: Charts already done — verify after data flows through
