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

## Offline Readiness

All KG benchmark model adapters use a two-phase offline workflow:

| Phase | When | Command |
|-------|------|---------|
| **Online (prep)** | Once, while connected | `uv run -m benchmarks.harness prep kg-models` |
| **Offline (benchmark)** | Any time after prep | `uv --offline run -m benchmarks.harness benchmark --id <id>` |

### Model categories

| Type | Models | Backbone needed? |
|------|--------|-----------------|
| GLiNER (3 variants) | `urchade/gliner_{small,medium,large}-v2.1` | Yes — `deberta-v3-base` / `deberta-v3-large` |
| NuNerZero | `numind/NuNerZero` | Yes — read from `gliner_config.json:model_name` |
| GNER-T5 (2 variants) | `dyyyyyyyy/GNER-T5-{base,large}` | No — T5 is self-contained |
| GLiREL | `jackboyla/glirel-large-v0` | Yes — `deberta-v3-large` |
| SentenceTransformer | MiniLM, nomic-embed-text-v1.5 | No — local snapshot path bypasses hub |
| spaCy | `en_core_web_lg` | N/A — installed via `spacy download` |

### Verification commands

```bash
# All models (KG harness, sessions_demo, demo_builder)
uv run -m benchmarks.harness prep kg-models --status
```

### Key implementation details

- **`offline_mode()` in `benchmarks.demo_builder.common`**: patches `huggingface_hub.constants.HF_HUB_OFFLINE = True`
  so `is_offline_mode()` returns True for the entire call tree (including transformers' `AutoTokenizer`/`AutoModel`).
  Setting `os.environ["HF_HUB_OFFLINE"]` after import is a no-op — the constant is frozen at import time.
- **GLiNER/GLiREL backbone issue**: their snapshots ship weights only; backbone tokenizer/model repos
  must be downloaded separately. `_read_backbone()` in `common.py` reads the backbone repo ID from the model config.
- **SentenceTransformer**: passing a local snapshot path directly bypasses all hub lookup — no `offline_mode()` needed.
