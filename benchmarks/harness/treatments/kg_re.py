"""KG relation extraction treatment.

Benchmarks relation extraction models on RE benchmark datasets.
Uses the same NER adapter pattern as kg_extract, treating extracted entity pairs
as crude relation proxies. More sophisticated RE adapters can be added later.

When gold triples are available, computes triple-level F1 via kg_metrics.triple_f1().

Source: docs/plans/ner_extraction_models_and_datasets.md
"""

import json
import logging
import time

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.treatments.base import Treatment
from benchmarks.harness.treatments.kg_extract import NER_ADAPTERS, FTS5Adapter
from benchmarks.harness.treatments.kg_metrics import triple_f1

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


class KGRelationExtractionTreatment(Treatment):
    """Single KG relation extraction benchmark permutation.

    Runs NER-based entity pair extraction on RE benchmark datasets
    and computes triple-level F1 against gold-standard triples.
    """

    def __init__(self, model_slug: str, dataset: str):
        self._model_slug = model_slug
        self._dataset = dataset
        self._adapter = None

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
        return {
            "model_slug": self._model_slug,
            "dataset": self._dataset,
        }

    def setup(self, conn, db_path):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predicted_triples (
                id INTEGER PRIMARY KEY,
                text_id INTEGER,
                subject TEXT,
                predicate TEXT,
                object TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_timing (
                text_id INTEGER PRIMARY KEY,
                time_ms REAL
            )
        """)
        conn.commit()

        # Load adapter
        adapter_cls = NER_ADAPTERS.get(self._model_slug)
        if adapter_cls is None:
            log.warning("NER adapter for %s not available — using FTS5 fallback", self._model_slug)
            adapter_cls = FTS5Adapter

        self._adapter = adapter_cls()
        self._adapter.load()

        return {"model_slug": self._model_slug, "dataset": self._dataset}

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

        labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]
        total_triples = 0
        text_times = []
        all_predicted_triples: list[tuple[str, str, str]] = []
        all_gold_triples: list[tuple[str, str, str]] = []

        for text_entry in texts:
            text_id = text_entry["id"]
            text = text_entry["text"]

            t0 = time.perf_counter()
            mentions = self._adapter.extract(text, labels)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text_times.append(elapsed_ms)

            # Generate entity pair triples as crude relation proxy
            predicted = []
            for i, m1 in enumerate(mentions):
                for m2 in mentions[i + 1 :]:
                    triple = (m1.text, "related_to", m2.text)
                    predicted.append(triple)
                    conn.execute(
                        "INSERT INTO predicted_triples(text_id, subject, predicate, object) VALUES (?,?,?,?)",
                        (text_id, triple[0], triple[1], triple[2]),
                    )

            conn.execute(
                "INSERT OR IGNORE INTO text_timing(text_id, time_ms) VALUES (?,?)",
                (text_id, elapsed_ms),
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
        self._adapter = None
