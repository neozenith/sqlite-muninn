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

## Files

- **NEW**: `benchmarks/harness/treatments/kg_re_adapters.py`
- **MODIFY**: `benchmarks/harness/treatments/kg_re.py`
