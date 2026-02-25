"""KG relation extraction treatment.

Benchmarks relation extraction models on RE benchmark datasets.
Runs NER first to extract entities, then passes entities to an RE model adapter
to extract typed relations. Computes triple-level F1 against gold-standard triples.

Source: docs/plans/kg/02_relation_extraction.md
"""

import json
import logging
import time

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.treatments.base import Treatment
from benchmarks.harness.treatments.kg_extract import NER_ADAPTERS
from benchmarks.harness.treatments.kg_metrics import triple_f1
from benchmarks.harness.treatments.kg_re_adapters import RE_ADAPTERS

log = logging.getLogger(__name__)


def _load_re_dataset(dataset_name: str) -> tuple[list[dict], list[dict]]:
    """Load a prepped RE benchmark dataset.

    Returns:
        (texts, triples) where:
        - texts: list of {"id": int, "text": str}
        - triples: list of {"text_id": int, "subject": str, "predicate": str, "object": str}
    """
    dataset_dir = KG_DIR / "re" / dataset_name
    texts_path = dataset_dir / "texts.jsonl"
    triples_path = dataset_dir / "triples.jsonl"

    if not texts_path.exists():
        log.warning("RE dataset texts not found: %s — run 'prep kg-re' first", texts_path)
        return [], []

    texts = [json.loads(line) for line in texts_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    triples = []
    if triples_path.exists():
        triples = [
            json.loads(line) for line in triples_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()
        ]

    return texts, triples


# Map model_slug to the RE adapter it should use.
# Models that have a dedicated RE adapter use it; others fall back to entity_pair proxy.
_MODEL_TO_RE_ADAPTER: dict[str, str] = {
    "fts5": "entity_pair",
    "gliner_small-v2.1": "glirel",
    "spacy_en_core_web_lg": "spacy_svo",
}


class KGRelationExtractionTreatment(Treatment):
    """Single KG relation extraction benchmark permutation.

    Runs NER to extract entities, then RE adapter to extract typed relations.
    Computes triple-level F1 against gold-standard triples when available.
    """

    def __init__(self, model_slug: str, dataset: str):
        self._model_slug = model_slug
        self._dataset = dataset
        self._ner_adapter = None
        self._re_adapter = None

    @property
    def requires_muninn(self) -> bool:
        return False

    @property
    def category(self):
        return "kg-re"

    @property
    def permutation_id(self):
        return f"kg-re_{self._model_slug}_{self._dataset}"

    @property
    def label(self):
        return f"KG RE: {self._model_slug} / {self._dataset}"

    @property
    def sort_key(self):
        return (self._dataset, self._model_slug)

    def params_dict(self):
        re_slug = _MODEL_TO_RE_ADAPTER.get(self._model_slug, "entity_pair")
        return {
            "model_slug": self._model_slug,
            "dataset": self._dataset,
            "re_adapter": re_slug,
        }

    def setup(self, conn, db_path):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predicted_triples (
                id INTEGER PRIMARY KEY,
                text_id INTEGER,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                score REAL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_timing (
                text_id INTEGER PRIMARY KEY,
                ner_time_ms REAL,
                re_time_ms REAL,
                total_time_ms REAL
            )
        """)
        conn.commit()

        # Load NER adapter
        ner_factory = NER_ADAPTERS[self._model_slug]
        self._ner_adapter = ner_factory()
        self._ner_adapter.load()

        # Load RE adapter
        re_slug = _MODEL_TO_RE_ADAPTER.get(self._model_slug, "entity_pair")
        re_factory = RE_ADAPTERS[re_slug]
        self._re_adapter = re_factory()
        self._re_adapter.load()

        return {
            "model_slug": self._model_slug,
            "dataset": self._dataset,
            "re_adapter": re_slug,
        }

    def run(self, conn):
        texts, gold_triples = _load_re_dataset(self._dataset)

        if not texts:
            return {
                "total_time_s": 0,
                "avg_ms_per_text": 0,
                "n_texts": 0,
                "n_triples": 0,
                "triple_precision": 0.0,
                "triple_recall": 0.0,
                "triple_f1": 0.0,
            }

        # Build gold triple lookup: text_id -> list of (subject, predicate, object)
        gold_by_text: dict[int, list[tuple[str, str, str]]] = {}
        for tr in gold_triples:
            tid = tr["text_id"]
            gold_by_text.setdefault(tid, []).append((tr["subject"], tr["predicate"], tr["object"]))

        # Extract labels from gold entities if available, else use defaults
        labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]
        total_triples = 0
        text_times = []
        all_predicted_triples: list[tuple[str, str, str]] = []
        all_gold_triples: list[tuple[str, str, str]] = []

        for text_entry in texts:
            text_id = text_entry["id"]
            text = text_entry["text"]

            # Phase 1: NER extraction
            t0 = time.perf_counter()
            entities = self._ner_adapter.extract(text, labels)
            ner_ms = (time.perf_counter() - t0) * 1000

            # Phase 2: Relation extraction
            t1 = time.perf_counter()
            relations = self._re_adapter.extract_relations(text, entities)
            re_ms = (time.perf_counter() - t1) * 1000

            total_ms = ner_ms + re_ms
            text_times.append(total_ms)

            # Store predicted triples
            predicted = []
            for rel in relations:
                triple = (rel.subject, rel.predicate, rel.object)
                predicted.append(triple)
                conn.execute(
                    "INSERT INTO predicted_triples(text_id, subject, predicate, object, score) VALUES (?,?,?,?,?)",
                    (text_id, rel.subject, rel.predicate, rel.object, rel.score),
                )

            conn.execute(
                "INSERT OR IGNORE INTO text_timing(text_id, ner_time_ms, re_time_ms, total_time_ms) VALUES (?,?,?,?)",
                (text_id, ner_ms, re_ms, total_ms),
            )
            total_triples += len(predicted)

            all_predicted_triples.extend(predicted)
            gold_text_triples = gold_by_text.get(text_id, [])
            all_gold_triples.extend(gold_text_triples)

        conn.commit()
        total_time = sum(text_times) / 1000

        metrics = {
            "total_time_s": round(total_time, 3),
            "avg_ms_per_text": round(sum(text_times) / len(text_times), 3) if text_times else 0,
            "n_texts": len(texts),
            "n_triples": total_triples,
        }

        # Compute triple F1 against gold labels
        f1_result = triple_f1(all_predicted_triples, all_gold_triples)
        metrics["triple_precision"] = f1_result["precision"]
        metrics["triple_recall"] = f1_result["recall"]
        metrics["triple_f1"] = f1_result["f1"]

        return metrics

    def teardown(self, conn):
        self._ner_adapter = None
        self._re_adapter = None
