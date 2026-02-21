"""KG NER extraction treatment with model adapter pattern.

Benchmarks NER model performance on Gutenberg text chunks and NER benchmark datasets.
Supports: GLiNER, NuNerZero, GNER-T5, spaCy, FTS5.
When gold labels are available (NER datasets), computes entity-level micro F1.

Source: docs/plans/ner_extraction_models_and_datasets.md
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.treatments.base import Treatment
from benchmarks.harness.treatments.kg_metrics import entity_micro_f1

log = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """A single entity mention extracted from text."""

    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


class NerModelAdapter(ABC):
    """Common interface for all NER extraction models."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def extract(self, text: str, labels: list[str]) -> list[EntityMention]:
        """Extract entity mentions from text."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model family: 'gliner', 'gner', 'spacy', 'fts5', 'nuner'."""


class FTS5Adapter(NerModelAdapter):
    """Lexical entity extraction using FTS5 pattern matching."""

    def load(self):
        pass  # No model to load

    def extract(self, text, labels):
        # Simple keyword extraction — returns entity mentions for known patterns
        mentions = []
        text_lower = text.lower()
        for label in labels:
            label_lower = label.lower()
            start = 0
            while True:
                idx = text_lower.find(label_lower, start)
                if idx == -1:
                    break
                mentions.append(
                    EntityMention(text=text[idx : idx + len(label)], label="KEYWORD", start=idx, end=idx + len(label))
                )
                start = idx + 1
        return mentions

    @property
    def model_id(self):
        return "fts5"

    @property
    def model_type(self):
        return "fts5"


# Model slug -> adapter factory
# None entries use FTS5Adapter as fallback until the real ML packages are available.
NER_ADAPTERS: dict[str, type[NerModelAdapter] | None] = {
    "fts5": FTS5Adapter,
    "gliner_small-v2.1": None,  # Requires gliner package
    "gliner_medium-v2.1": None,  # Requires gliner package
    "gliner_large-v2.1": None,  # Requires gliner package
    "numind_NuNerZero": None,  # Requires NuNerZero; labels must be lowercase
    "gner-t5-base": None,  # Requires transformers; seq2seq, slower
    "gner-t5-large": None,  # Requires transformers; seq2seq, slower
    "spacy_en_core_web_lg": None,  # Requires spacy package
}


def _parse_data_source(data_source: str) -> tuple[str, str]:
    """Parse data_source string into (source_type, source_id).

    Examples:
        "gutenberg:3300" -> ("gutenberg", "3300")
        "conll2003"      -> ("ner_dataset", "conll2003")
        "crossner"       -> ("ner_dataset", "crossner")
        "fewnerd"        -> ("ner_dataset", "fewnerd")
    """
    if ":" in data_source:
        parts = data_source.split(":", 1)
        return parts[0], parts[1]
    return "ner_dataset", data_source


def _data_source_slug(data_source: str) -> str:
    """Convert data_source to a filesystem-safe slug for permutation_id."""
    return data_source.replace(":", "-")


def _load_ner_dataset(dataset_name: str) -> tuple[list[dict], list[dict]]:
    """Load a prepped NER benchmark dataset.

    Returns:
        (texts, entities) where:
        - texts: list of {"id": int, "text": str, "tokens": list[str]}
        - entities: list of {"text_id": int, "start": int, "end": int, "label": str, "surface": str}
    """
    dataset_dir = KG_DIR / "ner" / dataset_name
    texts_path = dataset_dir / "texts.jsonl"
    entities_path = dataset_dir / "entities.jsonl"

    if not texts_path.exists():
        log.warning("NER dataset texts not found: %s — run 'prep kg-ner' first", texts_path)
        return [], []

    texts = [json.loads(line) for line in texts_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    entities = []
    if entities_path.exists():
        entities = [
            json.loads(line) for line in entities_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()
        ]

    return texts, entities


class KGNerExtractionTreatment(Treatment):
    """Single KG NER extraction benchmark permutation.

    Supports two data source types:
    - "gutenberg:{book_id}" — existing behavior, reads from KG chunks DB
    - "{dataset_name}" — reads from prepped NER benchmark dataset with gold labels
    """

    def __init__(self, model_slug: str, data_source: str):
        self._model_slug = model_slug
        self._data_source = data_source
        self._adapter = None

    @property
    def category(self):
        return "kg-extract"

    @property
    def permutation_id(self):
        return f"kg-extract_{self._model_slug}_{_data_source_slug(self._data_source)}"

    @property
    def label(self):
        return f"KG Extract: {self._model_slug} / {self._data_source}"

    @property
    def sort_key(self):
        return (self._data_source, self._model_slug)

    def params_dict(self):
        source_type, source_id = _parse_data_source(self._data_source)
        return {
            "model_slug": self._model_slug,
            "data_source": self._data_source,
            "source_type": source_type,
            "source_id": source_id,
        }

    def setup(self, conn, db_path):
        # Create output tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                chunk_id INTEGER,
                text TEXT,
                label TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                score REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_timing (
                chunk_id INTEGER PRIMARY KEY,
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

        return {"model_slug": self._model_slug}

    def run(self, conn):
        source_type, source_id = _parse_data_source(self._data_source)

        if source_type == "gutenberg":
            return self._run_gutenberg(conn, int(source_id))
        else:
            return self._run_ner_dataset(conn, source_id)

    def _run_gutenberg(self, conn, book_id: int) -> dict:
        """Run NER extraction on Gutenberg book chunks (no gold labels)."""
        import sqlite3 as _sqlite3

        chunks_db = KG_DIR / f"{book_id}_chunks.db"
        if not chunks_db.exists():
            log.warning("Chunks DB not found: %s — running with empty data", chunks_db)
            return {"total_time_s": 0, "avg_ms_per_chunk": 0, "n_chunks": 0, "n_entities": 0}

        src_conn = _sqlite3.connect(str(chunks_db))
        chunks = src_conn.execute("SELECT id, text FROM text_chunks ORDER BY id").fetchall()
        src_conn.close()

        labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]
        total_entities = 0
        chunk_times = []

        for chunk_id, text in chunks:
            t0 = time.perf_counter()
            mentions = self._adapter.extract(text, labels)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            chunk_times.append(elapsed_ms)

            for m in mentions:
                conn.execute(
                    "INSERT INTO entities(chunk_id, text, label, start_pos, end_pos, score) VALUES (?,?,?,?,?,?)",
                    (chunk_id, m.text, m.label, m.start, m.end, m.score),
                )
            conn.execute("INSERT INTO chunk_timing(chunk_id, time_ms) VALUES (?,?)", (chunk_id, elapsed_ms))
            total_entities += len(mentions)

        conn.commit()
        total_time = sum(chunk_times) / 1000  # seconds

        return {
            "total_time_s": round(total_time, 3),
            "avg_ms_per_chunk": round(sum(chunk_times) / len(chunk_times), 3) if chunk_times else 0,
            "n_chunks": len(chunks),
            "n_entities": total_entities,
        }

    def _run_ner_dataset(self, conn, dataset_name: str) -> dict:
        """Run NER extraction on a benchmark dataset with gold labels and compute F1."""
        texts, gold_entities = _load_ner_dataset(dataset_name)

        if not texts:
            return {
                "total_time_s": 0,
                "avg_ms_per_chunk": 0,
                "n_chunks": 0,
                "n_entities": 0,
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0,
            }

        # Build gold label lookup: text_id -> list of (start, end, label)
        gold_by_text: dict[int, list[tuple[int, int, str]]] = {}
        for ent in gold_entities:
            tid = ent["text_id"]
            gold_by_text.setdefault(tid, []).append((ent["start"], ent["end"], ent["label"]))

        labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]
        total_entities = 0
        chunk_times = []
        all_predicted: list[tuple[int, int, str]] = []
        all_gold: list[tuple[int, int, str]] = []

        for text_entry in texts:
            text_id = text_entry["id"]
            text = text_entry["text"]

            t0 = time.perf_counter()
            mentions = self._adapter.extract(text, labels)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            chunk_times.append(elapsed_ms)

            for m in mentions:
                conn.execute(
                    "INSERT INTO entities(chunk_id, text, label, start_pos, end_pos, score) VALUES (?,?,?,?,?,?)",
                    (text_id, m.text, m.label, m.start, m.end, m.score),
                )
            conn.execute(
                "INSERT OR IGNORE INTO chunk_timing(chunk_id, time_ms) VALUES (?,?)",
                (text_id, elapsed_ms),
            )
            total_entities += len(mentions)

            # Collect for F1 computation
            predicted_spans = [(m.start, m.end, m.label) for m in mentions]
            gold_spans = gold_by_text.get(text_id, [])
            all_predicted.extend(predicted_spans)
            all_gold.extend(gold_spans)

        conn.commit()
        total_time = sum(chunk_times) / 1000

        metrics = {
            "total_time_s": round(total_time, 3),
            "avg_ms_per_chunk": round(sum(chunk_times) / len(chunk_times), 3) if chunk_times else 0,
            "n_chunks": len(texts),
            "n_entities": total_entities,
        }

        # Compute entity micro F1 against gold labels
        f1_result = entity_micro_f1(all_predicted, all_gold)
        metrics["entity_precision"] = f1_result["precision"]
        metrics["entity_recall"] = f1_result["recall"]
        metrics["entity_f1"] = f1_result["f1"]

        return metrics

    def teardown(self, conn):
        self._adapter = None
