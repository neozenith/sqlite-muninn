# Phase 1: NER Model Adapters

## Goal

Implement 7 NER model adapters conforming to `NerModelAdapter` ABC, wire them into the `NER_ADAPTERS` dict, and fix dataset-specific label mapping.

## Adapter Implementations

### GLiNERAdapter (covers 3 model slugs)

- Uses `gliner.GLiNER.from_pretrained(model_id)` + `predict_entities(text, labels, threshold=0.3)`
- Returns dicts with `text`, `label`, `start` (char offset), `end` (char offset), `score`
- Maps directly to `EntityMention`
- Single class parameterised by model_id for small/medium/large

### NuNerZeroAdapter

- Same GLiNER API via `gliner.GLiNER.from_pretrained("numind/NuNerZero")`
- **Key difference**: labels must be lowercased before passing to `predict_entities()`
- Maps output labels back to original case via a label lookup

### GNERAdapter (covers T5-base and T5-large)

- Uses `transformers` AutoTokenizer + AutoModelForSeq2SeqLM
- Instruction-formatted input: wraps text + entity type list
- Parses structured text output back to `EntityMention` spans
- 10-50x slower than encoder models

### SpaCyAdapter

- Uses `spacy.load("en_core_web_lg")`
- Fixed entity types mapped: PERSON→person, ORG→organization, GPE→location, LOC→location, etc.
- Ignores the `labels` parameter (fixed pipeline)

## Label Mapping Strategy

The `_run_ner_dataset()` method extracts unique entity labels from gold annotations (`entities.jsonl`) and passes those as the `labels` parameter to adapters. This replaces the previously hardcoded label list.

## Offline Preparation

All NER adapters use `snapshot_download(repo_id, local_files_only=True)` at load time. Run once while online:

```bash
# Download all KG models (covers all NER adapters)
uv run -m benchmarks.harness prep kg-models

# Download a specific NER adapter only
uv run -m benchmarks.harness prep kg-models --model gliner_medium-v2.1
uv run -m benchmarks.harness prep kg-models --model numind_NuNerZero
uv run -m benchmarks.harness prep kg-models --model gner-t5-base
```

### Per-adapter notes

| Adapter | Backbone | Offline pattern |
|---------|----------|-----------------|
| `GLiNERAdapter` (small/medium) | `deberta-v3-base` | `snapshot_download` + `offline_mode()` |
| `GLiNERAdapter` (large) | `deberta-v3-large` | `snapshot_download` + `offline_mode()` |
| `NuNerZeroAdapter` | read from `gliner_config.json` | `snapshot_download` + `offline_mode()` |
| `GNERAdapter` (T5 base/large) | none — self-contained | `snapshot_download(local_files_only=True)` only |
| `SpaCyAdapter` | n/a — spaCy package | `spacy download en_core_web_lg` (installed once) |

`offline_mode()` is required for GLiNER-based adapters because `GLiNER.from_pretrained()` calls
`AutoTokenizer.from_pretrained(config.model_name)` without `local_files_only`. The patch forces the
entire `transformers` loading chain into offline mode for the duration of the `with` block.

## Files

- `benchmarks/harness/treatments/kg_types.py` — shared `EntityMention` and `NerModelAdapter` ABC
- `benchmarks/harness/treatments/kg_ner_adapters.py` — adapter implementations
- `benchmarks/harness/treatments/kg_extract.py` — `NER_ADAPTERS` dict + treatment
- `benchmarks/harness/prep/kg_models.py` — `prep_kg_models()` for model cache prep
- `benchmarks/demo_builder/common.py` — `offline_mode()`, `_read_backbone()` shared utilities
