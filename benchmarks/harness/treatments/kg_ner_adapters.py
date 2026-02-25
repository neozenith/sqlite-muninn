"""NER model adapter implementations for the KG benchmark suite.

Each adapter wraps a different NER model behind the NerModelAdapter ABC,
normalizing to the common extract(text, labels) -> list[EntityMention] interface.

Adapters:
- GLiNERAdapter: Zero-shot NER via GLiNER (small/medium/large variants)
- NuNerZeroAdapter: Zero-shot NER via NuNerZero (labels must be lowercase)
- GNERAdapter: Generative NER via GNER-T5 (seq2seq, slower, higher quality)
- SpaCyAdapter: Statistical NER via spaCy en_core_web_lg pipeline
"""

import logging
import re

import spacy
from gliner import GLiNER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from benchmarks.harness.treatments.kg_types import EntityMention, NerModelAdapter

log = logging.getLogger(__name__)

# ── spaCy entity type mapping ────────────────────────────────────

# Maps spaCy fine-grained types to coarser labels used in benchmarks.
# The mapping is intentionally broad — some datasets use lowercase, some uppercase.
SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "person",
    "PER": "person",
    "NORP": "group",
    "FAC": "location",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "work_of_art",
    "LAW": "law",
    "LANGUAGE": "language",
}


class GLiNERAdapter(NerModelAdapter):
    """Zero-shot NER via GLiNER encoder models.

    Supports urchade/gliner_small-v2.1, gliner_medium-v2.1, gliner_large-v2.1.
    GLiNER accepts arbitrary entity labels at inference time — no fine-tuning needed.
    """

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._model: GLiNER | None = None

    def load(self):
        log.info("Loading GLiNER model: %s", self._model_id)
        self._model = GLiNER.from_pretrained(self._model_id)

    def extract(self, text, labels):
        assert self._model is not None, "load() must be called before extract()"
        entities = self._model.predict_entities(text, labels, threshold=0.3)
        return [
            EntityMention(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e["score"],
            )
            for e in entities
        ]

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_type(self):
        return "gliner"


class NuNerZeroAdapter(NerModelAdapter):
    """Zero-shot NER via NuNerZero (numind/NuNerZero).

    Uses the same GLiNER API but requires labels to be lowercased.
    The adapter lowercases labels before prediction and maps output labels
    back to their original case.
    """

    _MODEL_ID = "numind/NuNerZero"

    def __init__(self):
        self._model: GLiNER | None = None

    def load(self):
        log.info("Loading NuNerZero model: %s", self._MODEL_ID)
        self._model = GLiNER.from_pretrained(self._MODEL_ID)

    def extract(self, text, labels):
        assert self._model is not None, "load() must be called before extract()"
        # NuNerZero requires lowercase labels
        lower_to_original = {label.lower(): label for label in labels}
        lowered_labels = list(lower_to_original.keys())

        entities = self._model.predict_entities(text, lowered_labels, threshold=0.3)
        return [
            EntityMention(
                text=e["text"],
                label=lower_to_original.get(e["label"], e["label"]),
                start=e["start"],
                end=e["end"],
                score=e["score"],
            )
            for e in entities
        ]

    @property
    def model_id(self):
        return self._MODEL_ID

    @property
    def model_type(self):
        return "nuner"


class GNERAdapter(NerModelAdapter):
    """Generative NER via GNER-T5 (seq2seq instruction-following model).

    GNER takes instruction-formatted text as input and generates entity mentions
    as structured text output. Requires parsing the output back to EntityMention spans.

    Significantly slower (10-50x) than encoder-based models like GLiNER.
    Supports dyyyyyyyy/GNER-T5-base and dyyyyyyyy/GNER-T5-large.
    """

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None

    def load(self):
        log.info("Loading GNER model: %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)

    def extract(self, text, labels):
        assert self._tokenizer is not None, "load() must be called before extract()"
        assert self._model is not None, "load() must be called before extract()"

        # Build instruction prompt in the GNER format
        label_str = ", ".join(labels)
        instruction = f"Please extract entities of type [{label_str}] from the following text: {text}"

        inputs = self._tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512)
        outputs = self._model.generate(**inputs, max_new_tokens=256)
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_gner_output(decoded, text, labels)

    def _parse_gner_output(self, output: str, original_text: str, labels: list[str]) -> list[EntityMention]:
        """Parse GNER structured output into EntityMention spans.

        GNER outputs text like: "Adam Smith [person], political economy [concept]"
        We extract entity text + label, then find their positions in the original text.
        """
        mentions = []
        # Pattern: "entity text [label]" separated by commas
        pattern = re.compile(r"([^,\[\]]+?)\s*\[([^\]]+)\]")

        for match in pattern.finditer(output):
            entity_text = match.group(1).strip()
            entity_label = match.group(2).strip()

            # Validate that the label is in the expected set (case-insensitive)
            label_lower_map = {lbl.lower(): lbl for lbl in labels}
            matched_label = label_lower_map.get(entity_label.lower(), entity_label)

            # Find the entity span in the original text
            start = original_text.find(entity_text)
            if start == -1:
                # Try case-insensitive search
                start = original_text.lower().find(entity_text.lower())
            if start == -1:
                continue  # Entity not found in source text, skip

            end = start + len(entity_text)
            mentions.append(
                EntityMention(
                    text=original_text[start:end],
                    label=matched_label,
                    start=start,
                    end=end,
                    score=1.0,
                )
            )

        return mentions

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_type(self):
        return "gner"


class SpaCyAdapter(NerModelAdapter):
    """Statistical NER via spaCy en_core_web_lg pipeline.

    spaCy has fixed entity types (PERSON, ORG, GPE, etc.) determined at training time.
    The labels parameter is ignored — we extract all entities spaCy recognizes and
    map them to the standardized label vocabulary via SPACY_LABEL_MAP.
    """

    def __init__(self):
        self._nlp: spacy.language.Language | None = None

    def load(self):
        log.info("Loading spaCy en_core_web_lg pipeline")
        self._nlp = spacy.load("en_core_web_lg")

    def extract(self, text, labels):
        assert self._nlp is not None, "load() must be called before extract()"
        doc = self._nlp(text)
        mentions = []
        for ent in doc.ents:
            mapped_label = SPACY_LABEL_MAP.get(ent.label_, ent.label_.lower())
            mentions.append(
                EntityMention(
                    text=ent.text,
                    label=mapped_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=1.0,
                )
            )
        return mentions

    @property
    def model_id(self):
        return "en_core_web_lg"

    @property
    def model_type(self):
        return "spacy"
