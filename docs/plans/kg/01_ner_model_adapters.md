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

## Files

- `benchmarks/harness/treatments/kg_types.py` — shared `EntityMention` and `NerModelAdapter` ABC
- `benchmarks/harness/treatments/kg_ner_adapters.py` — adapter implementations
- `benchmarks/harness/treatments/kg_extract.py` — `NER_ADAPTERS` dict + treatment
