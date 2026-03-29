"""HNSW embedding and KNN blocking — separated into prep and per-run stages.

Stage 1 (prep, once per dataset):
  embed_entities() — creates entities table + HNSW index in a persistent DB file.

Stage 2 (per-run):
  block() — opens the prebuilt DB, runs KNN search with a specific dist_threshold.

Stage 3 (per-run):
  leiden_cluster() — runs Leiden on match edges.
"""

import logging
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

from .datasets import DATASETS, load_dataset
from .models import EMBED_MODELS, EXTENSION_PATH, MODELS_DIR, ensure_model

log = logging.getLogger(__name__)

PREP_DIR = Path(__file__).resolve().parent / "prep"


def prep_db_path(dataset: str, embed_model: str) -> Path:
    """Path to the prebuilt embedded DB for a dataset."""
    return PREP_DIR / f"{dataset}__{embed_model}.db"


def prep_dataset(dataset: str, embed_model_name: str) -> Path:
    """Build the embedded DB for a dataset if it doesn't exist. Returns DB path."""
    db_path = prep_db_path(dataset, embed_model_name)
    if db_path.exists():
        log.info("Prep DB exists: %s", db_path.name)
        return db_path

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    embed_model = EMBED_MODELS[embed_model_name]
    prefix = embed_model.prefix

    cfg = DATASETS[dataset]
    entities, _gold = load_dataset(cfg, limit=None)

    # Build in a temp file, atomic move on success
    tmp_path = db_path.with_suffix(".tmp")
    conn = sqlite3.connect(str(tmp_path))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    ensure_model(embed_model)
    conn.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
        (embed_model.name, str(MODELS_DIR / embed_model.filename)),
    )

    # Create entities table
    conn.execute("CREATE TABLE entities(entity_id TEXT, name TEXT, source TEXT)")
    for e in entities:
        conn.execute("INSERT INTO entities VALUES(?, ?, ?)", (e.id, e.name, e.source))
    conn.commit()

    # Create HNSW index and embed
    dim = conn.execute("SELECT muninn_model_dim(?)", (embed_model.name,)).fetchone()[0]
    conn.execute(f"CREATE VIRTUAL TABLE entity_vecs USING hnsw_index(dimensions={dim}, metric=cosine)")

    if prefix:
        conn.execute(
            "INSERT INTO entity_vecs(rowid, vector) SELECT rowid, muninn_embed(?, ? || name) FROM entities",
            (embed_model.name, prefix),
        )
    else:
        conn.execute(
            "INSERT INTO entity_vecs(rowid, vector) SELECT rowid, muninn_embed(?, name) FROM entities",
            (embed_model.name,),
        )
    conn.commit()
    conn.close()

    shutil.move(str(tmp_path), str(db_path))
    log.info("Prepped %s: %d entities, dim=%d, model=%s", db_path.name, len(entities), dim, embed_model_name)
    return db_path


def open_prep_db(dataset: str, embed_model_name: str) -> sqlite3.Connection:
    """Open a prebuilt embedded DB (read-write for temp tables). Preps if needed."""
    db_path = prep_dataset(dataset, embed_model_name)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    return conn


def block(
    conn: sqlite3.Connection,
    k: int = 10,
    dist_threshold: float = 0.15,
) -> tuple[dict[int, str], dict[int, str], dict[tuple[int, int], float]]:
    """Run KNN blocking on a prebuilt embedded DB.

    Returns:
        id_map: rowid -> entity_id
        name_map: rowid -> entity name
        candidate_pairs: (min_rowid, max_rowid) -> cosine_distance
    """
    id_map: dict[int, str] = {}
    name_map: dict[int, str] = {}
    for row in conn.execute("SELECT rowid, entity_id, name FROM entities"):
        id_map[row[0]] = row[1]
        name_map[row[0]] = row[2]

    candidate_pairs: dict[tuple[int, int], float] = {}
    for rowid in id_map:
        vec = conn.execute("SELECT vector FROM entity_vecs WHERE rowid = ?", (rowid,)).fetchone()[0]
        neighbors = conn.execute(
            "SELECT rowid, distance FROM entity_vecs WHERE vector MATCH ? AND k = ?",
            (vec, k + 1),
        ).fetchall()
        for nid, dist in neighbors:
            if nid != rowid and dist <= dist_threshold:
                pair = (min(rowid, nid), max(rowid, nid))
                if pair not in candidate_pairs or dist < candidate_pairs[pair]:
                    candidate_pairs[pair] = dist

    log.info(
        "Blocking: %d pairs (k=%d, dist<=%.2f) from %d entities", len(candidate_pairs), k, dist_threshold, len(id_map)
    )
    return id_map, name_map, candidate_pairs


def leiden_cluster(
    conn: sqlite3.Connection,
    all_entity_ids: list[str],
    match_edges: list[tuple[str, str, float]],
) -> dict[str, int]:
    """Run Leiden clustering on match edges. Returns entity_id -> cluster_id."""
    conn.execute("DROP TABLE IF EXISTS _match_edges")
    conn.execute("CREATE TEMP TABLE _match_edges(src TEXT, dst TEXT, weight REAL)")
    for src, dst, w in match_edges:
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (src, dst, w))
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (dst, src, w))

    clusters: dict[str, int] = {}
    next_id = 0

    if match_edges:
        leiden_results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = '_match_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()
        comm_to_id: dict[int, int] = {}
        for node, comm_id in leiden_results:
            if comm_id not in comm_to_id:
                comm_to_id[comm_id] = next_id
                next_id += 1
            clusters[node] = comm_to_id[comm_id]

    for eid in all_entity_ids:
        if eid not in clusters:
            clusters[eid] = next_id
            next_id += 1

    n_multi = sum(1 for sz in _cluster_sizes(clusters).values() if sz > 1)
    log.info("Leiden: %d clusters (%d multi-member)", len(set(clusters.values())), n_multi)
    return clusters


def _cluster_sizes(clusters: dict[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for cid in clusters.values():
        sizes[cid] += 1
    return dict(sizes)
