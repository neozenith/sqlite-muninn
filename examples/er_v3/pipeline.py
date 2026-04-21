"""ER pipeline using muninn_extract_er() C function.

Uses prebuilt embedded databases (same as er_v2/blocking.py prep) so
embedding cost is paid once. Each run only does blocking + scoring +
clustering inside the C function.
"""

import json
import logging
import shutil
import sqlite3
import time
from pathlib import Path

from .datasets import DATASETS, load_dataset
from .metrics import bcubed_f1, pairwise_f1

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")
MODELS_DIR = PROJECT_ROOT / "models"
PREP_DIR = Path(__file__).resolve().parent / "prep"

EMBED_MODEL_NAME = "NomicEmbed"
EMBED_MODEL_FILE = "nomic-embed-text-v1.5.Q8_0.gguf"
EMBED_PREFIX = "clustering: "


def _prep_db_path(dataset: str) -> Path:
    return PREP_DIR / f"{dataset}__{EMBED_MODEL_NAME}.db"


def _ensure_prep_db(dataset: str) -> Path:
    """Build the prebuilt embedded DB if it doesn't exist."""
    db_path = _prep_db_path(dataset)
    if db_path.exists():
        log.info("Prep DB exists: %s", db_path.name)
        return db_path

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DATASETS[dataset]
    entities_list, _ = load_dataset(cfg, limit=None)

    tmp_path = db_path.with_suffix(".tmp")
    conn = sqlite3.connect(str(tmp_path))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Register embedding model
    embed_path = str(MODELS_DIR / EMBED_MODEL_FILE)
    conn.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
        (EMBED_MODEL_NAME, embed_path),
    )

    # Create entities table
    conn.execute("CREATE TABLE entities(entity_id TEXT, name TEXT, source TEXT)")
    for e in entities_list:
        conn.execute("INSERT INTO entities VALUES(?, ?, ?)", (e.id, e.name, e.source))
    conn.commit()

    # Create HNSW index with prefix-prepended embeddings
    dim = conn.execute("SELECT muninn_model_dim(?)", (EMBED_MODEL_NAME,)).fetchone()[0]
    conn.execute(f"CREATE VIRTUAL TABLE entity_vecs USING hnsw_index(dimensions={dim}, metric=cosine)")
    conn.execute(
        "INSERT INTO entity_vecs(rowid, vector) SELECT rowid, muninn_embed(?, ? || name) FROM entities",
        (EMBED_MODEL_NAME, EMBED_PREFIX),
    )
    conn.commit()
    conn.close()

    shutil.move(str(tmp_path), str(db_path))
    log.info("Prepped %s: %d entities, dim=%d", db_path.name, len(entities_list), dim)
    return db_path


def run_er(
    dataset: str,
    *,
    k: int = 10,
    dist_threshold: float = 0.15,
    jw_weight: float = 0.3,
    borderline_delta: float = 0.0,
    chat_model: str | None = None,
    edge_betweenness_threshold: float | None = None,
) -> dict:
    """Run ER via muninn_extract_er() on a prebuilt embedded DB."""
    cfg = DATASETS[dataset]
    _, gold = load_dataset(cfg, limit=None)

    # Open prebuilt DB (preps on first use)
    db_path = _ensure_prep_db(dataset)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Call muninn_extract_er()
    t0 = time.perf_counter()
    result = conn.execute(
        "SELECT muninn_extract_er(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "entity_vecs",
            "name",
            k,
            dist_threshold,
            jw_weight,
            borderline_delta,
            chat_model,
            edge_betweenness_threshold,
            "same_source",  # DeepMatcher: skip same-source pairs
        ),
    ).fetchone()[0]
    elapsed = time.perf_counter() - t0

    clusters_json = json.loads(result)
    predicted = dict(clusters_json["clusters"])

    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)

    conn.close()

    return {
        "dataset": dataset,
        "n_entities": len(predicted),
        "bcubed_f1": bc["f1"],
        "bcubed_precision": bc["precision"],
        "bcubed_recall": bc["recall"],
        "pairwise_f1": pw["f1"],
        "pairwise_precision": pw["precision"],
        "pairwise_recall": pw["recall"],
        "elapsed_s": round(elapsed, 3),
        "params": {
            "k": k,
            "dist_threshold": dist_threshold,
            "jw_weight": jw_weight,
            "borderline_delta": borderline_delta,
            "edge_betweenness_threshold": edge_betweenness_threshold,
        },
    }
