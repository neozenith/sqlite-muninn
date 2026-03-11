"""RE model adapter implementations for the KG benchmark suite.

Each adapter wraps a different relation extraction model behind the ReModelAdapter ABC,
normalizing to extract_relations(text, entities) -> list[RelationMention].

Adapters:
- GLiRELAdapter: Zero-shot RE via GLiREL (token-level spans, spaCy tokenization)
- SpaCySVOAdapter: Dependency-based SVO triple extraction
- EntityPairAdapter: Baseline that emits all entity pairs as "related_to"
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import spacy
from glirel import GLiREL
from huggingface_hub import snapshot_download
from llama_cpp import Llama

from benchmarks.demo_builder.common import offline_mode
from benchmarks.harness.common import GGUF_MODELS_DIR
from benchmarks.harness.treatments.kg_types import EntityMention

log = logging.getLogger(__name__)


@dataclass
class RelationMention:
    """A single relation extracted between two entities."""

    subject: str
    predicate: str
    object: str
    score: float = 1.0


class ReModelAdapter(ABC):
    """Common interface for all relation extraction models."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def extract_relations(self, text: str, entities: list[EntityMention]) -> list[RelationMention]:
        """Extract relations between entities found in text."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string."""


def _char_span_to_token_span(doc: spacy.tokens.Doc, char_start: int, char_end: int) -> tuple[int, int] | None:
    """Convert character offsets to spaCy token indices (inclusive start, exclusive end)."""
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span is None:
        return None
    return (span.start, span.end)


# ── Default relation labels for GLiREL ────────────────────────────

GLIREL_DEFAULT_LABELS = [
    "produces",
    "trades_with",
    "regulates",
    "employs",
    "located_in",
    "influences",
    "part_of",
    "opposes",
    "founded_by",
    "member_of",
    "works_for",
    "born_in",
    "capital_of",
    "subsidiary_of",
]


class GLiRELAdapter(ReModelAdapter):
    """Zero-shot RE via GLiREL large model.

    Uses spaCy tokenization to convert NER character spans to token-level spans
    required by GLiREL. Maps GLiREL output positions back to entity names via
    span overlap matching.

    Reference: benchmarks/demo_builder/phases/re.py
    """

    def __init__(self, relation_labels: list[str] | None = None):
        self._re_model: GLiREL | None = None
        self._nlp: spacy.language.Language | None = None
        self._relation_labels = relation_labels or GLIREL_DEFAULT_LABELS

    def load(self):
        log.info("Loading GLiREL large-v0...")
        glirel_dir = snapshot_download("jackboyla/glirel-large-v0", local_files_only=True)
        with offline_mode():
            self._re_model = GLiREL._from_pretrained(
                model_id=glirel_dir,
                revision=None,
                cache_dir=None,
                force_download=False,
                proxies=None,
                resume_download=False,
                local_files_only=True,
                token=None,
            )
        log.info("Loading spaCy en_core_web_lg for tokenization...")
        self._nlp = spacy.load("en_core_web_lg")

    def extract_relations(self, text, entities):
        assert self._re_model is not None, "load() must be called before extract_relations()"
        assert self._nlp is not None, "load() must be called before extract_relations()"

        if len(entities) < 2:
            return []

        # Tokenize with spaCy
        doc = self._nlp(text)
        tokens = [token.text for token in doc]

        # Convert entity char spans to token-level spans for GLiREL
        ner_spans = []
        for ent in entities:
            token_span = _char_span_to_token_span(doc, ent.start, ent.end)
            if token_span is None:
                continue
            ner_spans.append([token_span[0], token_span[1], ent.label, ent.text])

        if len(ner_spans) < 2:
            return []

        # Build position -> entity name lookup for mapping GLiREL output back to NER entities.
        span_to_name: dict[tuple[int, int], str] = {}
        for span in ner_spans:
            span_to_name[(span[0], span[1])] = span[3]

        def _find_entity(pos: list[int]) -> str | None:
            """Map GLiREL head_pos/tail_pos to NER entity name via span overlap."""
            r_start, r_end = pos[0], pos[1]
            best_name = None
            best_overlap = 0
            for (s, e), found_name in span_to_name.items():
                overlap = max(0, min(e, r_end) - max(s, r_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = found_name
            return best_name

        # Extract relations
        relations = self._re_model.predict_relations(
            tokens, self._relation_labels, threshold=0.5, ner=ner_spans, top_k=10
        )

        mentions = []
        for rel in relations:
            head = _find_entity(rel["head_pos"])
            tail = _find_entity(rel["tail_pos"])
            if head is None or tail is None or head == tail:
                continue
            mentions.append(
                RelationMention(
                    subject=head,
                    predicate=rel["label"],
                    object=tail,
                    score=rel.get("score", 1.0),
                )
            )

        return mentions

    @property
    def model_id(self):
        return "glirel-large-v0"


class SpaCySVOAdapter(ReModelAdapter):
    """Dependency-based SVO triple extraction via spaCy.

    For each sentence, finds nsubj->VERB->dobj dependency patterns and
    emits (subject_text, verb_lemma, object_text) triples.
    A lightweight statistical baseline for relation extraction.
    """

    def __init__(self):
        self._nlp: spacy.language.Language | None = None

    def load(self):
        log.info("Loading spaCy en_core_web_lg for SVO extraction")
        self._nlp = spacy.load("en_core_web_lg")

    def extract_relations(self, text, entities):
        assert self._nlp is not None, "load() must be called before extract_relations()"

        doc = self._nlp(text)
        # Build entity text lookup for matching SVO spans to NER entities
        entity_texts = {ent.text.lower() for ent in entities}

        mentions = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subject (nsubj or nsubjpass)
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                # Find object (dobj or attr)
                objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "pobj")]

                for subj in subjects:
                    subj_text = _get_span_text(subj)
                    for obj in objects:
                        obj_text = _get_span_text(obj)
                        # Only emit if at least one side matches a known entity
                        if subj_text.lower() in entity_texts or obj_text.lower() in entity_texts:
                            mentions.append(
                                RelationMention(
                                    subject=subj_text,
                                    predicate=token.lemma_,
                                    object=obj_text,
                                    score=1.0,
                                )
                            )

        return mentions

    @property
    def model_id(self):
        return "spacy_svo"


class EntityPairAdapter(ReModelAdapter):
    """Baseline RE adapter: emits all entity pairs as "related_to".

    This is the original entity-pair proxy from the initial kg_re.py implementation,
    preserved as a named baseline for comparison with real RE models.
    """

    def load(self):
        pass  # No model to load

    def extract_relations(self, text, entities):
        mentions = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                mentions.append(
                    RelationMention(
                        subject=e1.text,
                        predicate="related_to",
                        object=e2.text,
                        score=1.0,
                    )
                )
        return mentions

    @property
    def model_id(self):
        return "entity_pair_proxy"


class LlmREAdapter(ReModelAdapter):
    """LLM-based relation extraction via llama-cpp-python with structured JSON output.

    Uses a GGUF chat model to extract relations between provided entities via
    prompted generation with JSON schema grammar constraints.
    """

    def __init__(self, model_path: str, relation_types: list[str] | None = None, ctx_len: int = 4096):
        self._model_path = model_path
        self._relation_types = relation_types or GLIREL_DEFAULT_LABELS
        self._ctx_len = ctx_len
        self._model: Llama | None = None

    def load(self):
        log.info("Loading LLM RE model: %s (ctx=%d)", self._model_path, self._ctx_len)
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self._ctx_len,
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",
        )

    def extract_relations(self, text, entities):
        assert self._model is not None, "load() must be called before extract_relations()"

        if len(entities) < 2:
            return []

        # Build entities JSON for the prompt
        entities_json = json.dumps(
            [{"text": e.text, "type": e.label, "start": e.start, "end": e.end} for e in entities]
        )
        relation_types_csv = ", ".join(self._relation_types)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise relation extraction system. "
                    "Extract relations between provided entities. /no_think"
                ),
            },
            {
                "role": "user",
                "content": (f"Relation types: {relation_types_csv}\nEntities: {entities_json}\nText: {text}"),
            },
        ]

        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "head": {"type": "string"},
                                "rel": {"type": "string"},
                                "tail": {"type": "string"},
                                "score": {"type": "number"},
                            },
                            "required": ["head", "rel", "tail"],
                        },
                    }
                },
                "required": ["relations"],
            },
        }

        response = self._model.create_chat_completion(
            messages=messages,
            response_format=response_format,
            max_tokens=512,
            temperature=0.0,
        )

        content = response["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            log.warning("LLM RE returned invalid JSON: %.200s", content)
            return []

        # Build entity name set for validation
        entity_names = {e.text for e in entities}

        mentions = []
        for rel in parsed.get("relations", []):
            head = rel.get("head", "")
            predicate = rel.get("rel", "")
            tail = rel.get("tail", "")
            score = float(rel.get("score", 1.0))

            if not head or not predicate or not tail:
                continue
            if head == tail:
                continue

            # Only emit relations where at least one side is a known entity
            if head not in entity_names and tail not in entity_names:
                continue

            mentions.append(
                RelationMention(
                    subject=head,
                    predicate=predicate,
                    object=tail,
                    score=score,
                )
            )

        return mentions

    @property
    def model_id(self):
        return Path(self._model_path).stem


def _get_span_text(token: spacy.tokens.Token) -> str:
    """Get the full span text for a token including its subtree compound modifiers."""
    # Include compound modifiers to get full noun phrases
    compounds = [child for child in token.children if child.dep_ == "compound"]
    if compounds:
        # Sort by position and include the head token
        all_tokens = sorted(compounds + [token], key=lambda t: t.i)
        return " ".join(t.text for t in all_tokens)
    return str(token.text)


# RE model slug -> adapter factory callable
RE_ADAPTERS: dict[str, type[ReModelAdapter] | Callable[[], ReModelAdapter]] = {
    "glirel": GLiRELAdapter,
    "spacy_svo": SpaCySVOAdapter,
    "entity_pair": EntityPairAdapter,
    "llm-qwen3-4b": lambda: LlmREAdapter(str(GGUF_MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf")),
    "llm-qwen3-8b": lambda: LlmREAdapter(str(GGUF_MODELS_DIR / "Qwen3-8B-Q4_K_M.gguf")),
    "llm-phi4-mini": lambda: LlmREAdapter(str(GGUF_MODELS_DIR / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")),
    "llm-gemma3-4b": lambda: LlmREAdapter(str(GGUF_MODELS_DIR / "google_gemma-3-4b-it-Q4_K_M.gguf")),
}
