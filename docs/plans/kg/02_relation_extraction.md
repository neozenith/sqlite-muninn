# Phase 2: Relation Extraction

## Goal

Replace the entity-pair proxy in `KGRelationExtractionTreatment` with real RE model adapters: GLiREL and spaCy SVO.

## RE Adapter ABC

```python
class ReModelAdapter(ABC):
    def load(self) -> None: ...
    def extract_relations(self, text: str, entities: list[EntityMention]) -> list[RelationMention]: ...
    @property
    def model_id(self) -> str: ...
```

## Adapter Implementations

### GLiRELAdapter

- `snapshot_download("jackboyla/glirel-large-v0")` + `_from_pretrained()` workaround
- spaCy tokenization for GLiREL's token-based input
- `char_span_to_token_span()` for mapping NER char offsets to token spans
- `predict_relations(tokens, labels, threshold=0.5, ner=ner_spans)`
- Reference: `demo_builder/phases/re.py`

### SpaCySVOAdapter

- Dependency parsing: `nsubj→VERB→dobj` patterns
- Lightweight baseline for RE

### EntityPairAdapter

- Existing entity-pair proxy logic, renamed as explicit baseline

## Treatment Changes

- `KGRelationExtractionTreatment` gets an `RE_ADAPTERS` dict
- `setup()` loads NER adapter + RE adapter
- `run()` calls NER first, then passes entities to RE adapter

## Offline Preparation

```bash
# Download GLiREL and its backbone
uv run -m benchmarks.harness prep kg-models --model glirel

# Check cache status
uv run -m benchmarks.harness prep kg-models --status
```

### GLiRELAdapter offline loading pattern

```python
# 1. Resolve to local cache path — raises EnvironmentError if not cached
glirel_dir = snapshot_download("jackboyla/glirel-large-v0", local_files_only=True)

# 2. Patch HF_HUB_OFFLINE for the entire _from_pretrained() call tree.
#    GLiREL's TransformerWordEmbeddings calls AutoModel.from_pretrained(model_name)
#    without local_files_only, so the global patch is required.
with offline_mode():
    model = GLiREL._from_pretrained(model_id=glirel_dir, ..., local_files_only=True)
```

The `_from_pretrained` workaround is required because `GLiREL.from_pretrained()` was broken by
huggingface_hub >= 1.0 (missing `proxies`/`resume_download` kwargs). The `_from_pretrained` class
method bypasses the broken hub code path.

## Files

- **NEW**: `benchmarks/harness/treatments/kg_re_adapters.py`
- **MODIFY**: `benchmarks/harness/treatments/kg_re.py`
- `benchmarks/harness/prep/kg_models.py` — includes `glirel` slug in registry
- `benchmarks/demo_builder/common.py` — `offline_mode()` shared utility
