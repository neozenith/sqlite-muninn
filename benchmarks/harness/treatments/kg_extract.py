"""KG NER extraction treatment with model adapter pattern.

Benchmarks NER model performance on Gutenberg text chunks and NER benchmark datasets.
Supports: GLiNER, NuNerZero, GNER-T5, spaCy.
When gold labels are available (NER datasets), computes entity-level micro F1.

Source: docs/plans/ner_extraction_models_and_datasets.md
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from benchmarks.harness.common import GGUF_MODELS_DIR, KG_DIR
from benchmarks.harness.treatments.base import Treatment
from benchmarks.harness.treatments.kg_metrics import entity_micro_f1
from benchmarks.harness.treatments.kg_ner_adapters import (
    GLiNERAdapter,
    GNERAdapter,
    LlmNerAdapter,
    NuNerZeroAdapter,
    SpaCyAdapter,
)
from benchmarks.harness.treatments.kg_types import NerModelAdapter

log = logging.getLogger(__name__)


# Model slug -> adapter factory callable.
# Each entry is a callable that returns an NerModelAdapter instance.
# Factory lambdas parameterize model IDs for multi-variant adapter classes.
NER_ADAPTERS: dict[str, type[NerModelAdapter] | Callable[[], NerModelAdapter]] = {
    "gliner_small-v2.1": lambda: GLiNERAdapter("urchade/gliner_small-v2.1"),
    "gliner_medium-v2.1": lambda: GLiNERAdapter("urchade/gliner_medium-v2.1"),
    "gliner_large-v2.1": lambda: GLiNERAdapter("urchade/gliner_large-v2.1"),
    "numind_NuNerZero": NuNerZeroAdapter,
    "gner-t5-base": lambda: GNERAdapter("dyyyyyyyy/GNER-T5-base"),
    "gner-t5-large": lambda: GNERAdapter("dyyyyyyyy/GNER-T5-large"),
    "spacy_en_core_web_lg": SpaCyAdapter,
    "llm-qwen3-4b": lambda: LlmNerAdapter(str(GGUF_MODELS_DIR / "Qwen3-4B-Q4_K_M.gguf")),
    "llm-qwen3-8b": lambda: LlmNerAdapter(str(GGUF_MODELS_DIR / "Qwen3-8B-Q4_K_M.gguf")),
    "llm-phi4-mini": lambda: LlmNerAdapter(str(GGUF_MODELS_DIR / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")),
    "llm-gemma3-4b": lambda: LlmNerAdapter(str(GGUF_MODELS_DIR / "google_gemma-3-4b-it-Q4_K_M.gguf")),
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


def _load_ner_dataset(dataset_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        self._adapter: NerModelAdapter | None = None

    @property
    def requires_muninn(self) -> bool:
        return False

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
        adapter_factory = NER_ADAPTERS[self._model_slug]
        self._adapter = adapter_factory()
        self._adapter.load()

        return {"model_slug": self._model_slug}

    def run(self, conn):
        source_type, source_id = _parse_data_source(self._data_source)

        if source_type == "gutenberg":
            return self._run_gutenberg(conn, int(source_id))
        else:
            return self._run_ner_dataset(conn, source_id)

    def _run_gutenberg(self, conn, book_id: int) -> dict[str, Any]:
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
            assert self._adapter is not None
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

    def _run_ner_dataset(self, conn, dataset_name: str) -> dict[str, Any]:
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

        # Extract unique entity labels from gold annotations — each dataset uses its own label vocabulary
        labels = sorted({ent["label"] for ent in gold_entities}) if gold_entities else []
        if not labels:
            labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"]
        log.info("NER labels for %s: %s", dataset_name, labels)
        total_entities = 0
        chunk_times = []
        all_predicted: list[tuple[int, int, str]] = []
        all_gold: list[tuple[int, int, str]] = []

        for text_entry in texts:
            text_id = text_entry["id"]
            text = text_entry["text"]

            t0 = time.perf_counter()
            assert self._adapter is not None
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
