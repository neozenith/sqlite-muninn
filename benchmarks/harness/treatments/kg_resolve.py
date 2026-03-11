"""KG entity resolution treatment.

Benchmarks entity resolution pipeline: HNSW blocking -> Jaro-Winkler matching -> Leiden clustering.
Two modes: KG coalescing on Gutenberg texts, ER benchmark datasets with ground truth.

Reference: benchmarks/demo_builder/phases/entity_resolution.py

Source: docs/plans/kg/03_entity_resolution.md
"""

import json
import logging
import sqlite3 as _sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.treatments.base import Treatment
from benchmarks.harness.treatments.kg_metrics import bcubed_f1

log = logging.getLogger(__name__)


# ── Jaro-Winkler similarity (pure Python) ────────────────────────


def _jaro_winkler(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s1_matched[i] = True
            s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


def _pack_vector(v: np.ndarray) -> bytes:
    """Pack a float32 numpy array into a BLOB for SQLite."""
    return bytes(v.astype(np.float32).tobytes())


# ── Core ER pipeline ─────────────────────────────────────────────


def _run_er_pipeline(
    conn: _sqlite3.Connection,
    entity_names: list[str],
    entity_vectors: np.ndarray,
    k_neighbors: int = 10,
    cosine_threshold: float = 0.4,
    match_threshold: float = 0.5,
) -> dict[str, str]:
    """Run the full HNSW blocking + Jaro-Winkler matching + Leiden clustering pipeline.

    Args:
        conn: SQLite connection with muninn extension loaded
        entity_names: List of unique entity names to resolve
        entity_vectors: Corresponding embedding vectors (N x dim)
        k_neighbors: KNN neighbors per entity for blocking
        cosine_threshold: Max cosine distance for candidate pairs
        match_threshold: Min combined score for match edges

    Returns:
        Dict mapping each entity name to its canonical (resolved) name.
    """
    dim = entity_vectors.shape[1]

    # ── Create HNSW index and insert embeddings ──────────────────
    conn.execute(
        f"CREATE VIRTUAL TABLE _er_vec USING hnsw_index(  dimensions={dim}, metric='cosine', m=16, ef_construction=200)"
    )

    for i, (_name, vec) in enumerate(zip(entity_names, entity_vectors, strict=True)):
        rowid = i + 1
        conn.execute(
            "INSERT INTO _er_vec (rowid, vector) VALUES (?, ?)",
            (rowid, _pack_vector(vec)),
        )

    # ── HNSW blocking: find candidate match pairs ────────────────
    candidate_pairs: list[tuple[str, str, float]] = []

    for i, name in enumerate(entity_names):
        rowid = i + 1
        vec_blob = _pack_vector(entity_vectors[i])

        neighbors = conn.execute(
            "SELECT rowid, distance FROM _er_vec WHERE vector MATCH ? AND k = ?",
            (vec_blob, k_neighbors + 1),
        ).fetchall()

        for neighbor_rowid, distance in neighbors:
            if neighbor_rowid == rowid:
                continue
            if distance > cosine_threshold:
                continue
            neighbor_name = entity_names[neighbor_rowid - 1]
            pair = tuple(sorted([name, neighbor_name]))
            candidate_pairs.append((pair[0], pair[1], distance))

    # Deduplicate candidate pairs
    seen_pairs: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str, float]] = []
    for n1, n2, dist in candidate_pairs:
        key = (n1, n2)
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append((n1, n2, dist))

    log.info("  HNSW blocking: %d candidate pairs from %d entities", len(unique_pairs), len(entity_names))

    # ── Matching cascade: score each candidate pair ──────────────
    match_edges: list[tuple[str, str, float]] = []

    for n1, n2, cosine_dist in unique_pairs:
        cosine_sim = 1.0 - cosine_dist

        if n1.lower() == n2.lower():
            match_edges.append((n1, n2, 0.9))
            continue

        jw = _jaro_winkler(n1.lower(), n2.lower())
        combined = 0.4 * jw + 0.6 * cosine_sim

        if combined > match_threshold:
            match_edges.append((n1, n2, combined))

    log.info("  Matching: %d pairs above threshold %.2f", len(match_edges), match_threshold)

    # ── Leiden clustering on match pairs ─────────────────────────
    conn.execute("CREATE TEMP TABLE _match_edges (src TEXT NOT NULL, dst TEXT NOT NULL, weight REAL DEFAULT 1.0)")
    conn.executemany(
        "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
        match_edges,
    )
    # Insert reverse edges (Leiden expects undirected)
    conn.executemany(
        "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
        [(n2, n1, w) for n1, n2, w in match_edges],
    )

    entity_to_canonical: dict[str, str] = {}

    if match_edges:
        leiden_results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = '_match_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()

        # Group by community
        communities: dict[int, list[str]] = {}
        for node, comm_id in leiden_results:
            communities.setdefault(comm_id, []).append(node)

        # For each community, pick canonical = first alphabetically (deterministic)
        for _comm_id, members in communities.items():
            canonical = sorted(members)[0]
            for member in members:
                entity_to_canonical[member] = canonical

        log.info("  Leiden: %d communities from %d matched entities", len(communities), len(leiden_results))

    # Entities not in any match pair are their own canonical form
    for name in entity_names:
        if name not in entity_to_canonical:
            entity_to_canonical[name] = name

    # Cleanup
    conn.execute("DROP TABLE IF EXISTS _match_edges")
    conn.execute("DROP TABLE IF EXISTS _er_vec")

    return entity_to_canonical


class LlmERAdapter:
    """LLM-based entity resolution via llama-cpp-python with structured JSON output.

    Uses a GGUF chat model to determine which entity mentions refer to the same
    real-world entity. Operates on candidate groups (e.g. from HNSW blocking)
    and returns merge groups.

    Not an ABC subclass — entity resolution does not have an adapter ABC.
    """

    def __init__(self, model_path: str, ctx_len: int = 4096):
        self._model_path = model_path
        self._ctx_len = ctx_len
        self._model: Llama | None = None

    def load(self):
        log.info("Loading LLM ER model: %s (ctx=%d)", self._model_path, self._ctx_len)
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self._ctx_len,
            n_gpu_layers=0,
            verbose=False,
            chat_format="chatml",
        )

    def should_merge(self, candidates: list[str]) -> list[list[str]]:
        """Given candidate entity mentions, group those referring to the same entity.

        Args:
            candidates: List of entity mention strings to consider for merging.

        Returns:
            List of merge groups, where each group is a list of mentions that
            should be resolved to the same canonical entity.
        """
        assert self._model is not None, "load() must be called before should_merge()"

        if len(candidates) < 2:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise entity resolution system. "
                    "Group entity mentions that refer to the same real-world entity. /no_think"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given these candidate entity mentions, group those that refer to the same "
                    "real-world entity. Return JSON: "
                    '{"groups": [["mention1", "mention2"], ...]}\n'
                    f"Candidates: {json.dumps(candidates)}"
                ),
            },
        ]

        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    }
                },
                "required": ["groups"],
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
            log.warning("LLM ER returned invalid JSON: %.200s", content)
            return []

        groups = parsed.get("groups", [])

        # Validate: only keep groups with 2+ members, and only members from candidates
        candidate_set = set(candidates)
        valid_groups = []
        for group in groups:
            if not isinstance(group, list):
                continue
            filtered = [m for m in group if isinstance(m, str) and m in candidate_set]
            if len(filtered) >= 2:
                valid_groups.append(filtered)

        return valid_groups

    @property
    def model_id(self) -> str:
        return Path(self._model_path).stem


class KGEntityResolutionTreatment(Treatment):
    """Single entity resolution benchmark permutation."""

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def requires_muninn(self) -> bool:
        return True  # Needs HNSW + Leiden TVFs

    @property
    def category(self):
        return "kg-resolve"

    @property
    def permutation_id(self):
        return f"kg-resolve_{self._dataset}"

    @property
    def label(self):
        return f"KG Resolve: {self._dataset}"

    @property
    def sort_key(self):
        return (self._dataset,)

    def params_dict(self):
        return {"dataset": self._dataset}

    def setup(self, conn, db_path):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT,
                cluster_id INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resolution_metrics (
                metric TEXT PRIMARY KEY,
                value REAL
            )
        """)
        conn.commit()

        return {"dataset": self._dataset}

    def run(self, conn):
        if self._dataset.isdigit():
            return self._run_kg_coalesce(conn)
        else:
            return self._run_er_dataset(conn)

    def teardown(self, conn):
        pass

    def _run_kg_coalesce(self, conn):
        """Coalesce entities from a KG extraction output using HNSW blocking + Jaro-Winkler + Leiden."""
        book_id = int(self._dataset)
        entities_db = KG_DIR / f"{book_id}_chunks.db"

        if not entities_db.exists():
            log.warning("Entities DB not found: %s", entities_db)
            return {"nodes_before": 0, "nodes_after": 0, "total_time_s": 0}

        # Load unique entity names from the chunks DB
        src_conn = _sqlite3.connect(str(entities_db))
        try:
            rows = src_conn.execute("SELECT DISTINCT name FROM entities ORDER BY name").fetchall()
        except _sqlite3.OperationalError:
            log.warning("No 'entities' table in %s — run NER extraction first", entities_db)
            src_conn.close()
            return {"nodes_before": 0, "nodes_after": 0, "total_time_s": 0}
        src_conn.close()

        entity_names = [r[0] for r in rows]
        nodes_before = len(entity_names)

        if nodes_before < 2:
            return {
                "nodes_before": nodes_before,
                "nodes_after": nodes_before,
                "singleton_ratio": 1.0,
                "blocking_time_s": 0.0,
                "matching_time_s": 0.0,
                "clustering_time_s": 0.0,
                "total_time_s": 0.0,
            }

        # Embed entity names
        t_embed_start = time.perf_counter()
        log.info("  Embedding %d entity names...", nodes_before)
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        entity_vectors = st_model.encode(entity_names, show_progress_bar=False, normalize_embeddings=True)
        entity_vectors = entity_vectors.astype(np.float32)
        t_embed = time.perf_counter() - t_embed_start

        # Run ER pipeline
        t_block_start = time.perf_counter()
        entity_to_canonical = _run_er_pipeline(conn, entity_names, entity_vectors)
        t_total = time.perf_counter() - t_block_start

        # Count canonical entities
        canonical_names = set(entity_to_canonical.values())
        nodes_after = len(canonical_names)
        singletons = sum(1 for name in entity_names if entity_to_canonical[name] == name)
        singleton_ratio = singletons / nodes_before if nodes_before > 0 else 0.0

        # Store entity->cluster mapping
        cluster_ids: dict[str, int] = {}
        for i, canonical in enumerate(sorted(canonical_names)):
            cluster_ids[canonical] = i
        for name in entity_names:
            cid = cluster_ids[entity_to_canonical[name]]
            conn.execute(
                "INSERT INTO entities(name, cluster_id) VALUES (?, ?)",
                (name, cid),
            )
        conn.commit()

        return {
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "singleton_ratio": round(singleton_ratio, 4),
            "embed_time_s": round(t_embed, 3),
            "pipeline_time_s": round(t_total, 3),
            "total_time_s": round(t_embed + t_total, 3),
        }

    def _run_er_dataset(self, conn):
        """Run entity resolution on a Febrl benchmark dataset with ground truth."""
        dataset_dir = KG_DIR / "er" / self._dataset

        if not dataset_dir.exists():
            log.warning("ER dataset not found: %s — run 'prep kg-er' first", dataset_dir)
            return {
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_f1": 0.0,
                "total_time_s": 0.0,
            }

        # Load Febrl data
        parquet_path = dataset_dir / f"{self._dataset}.parquet"
        if not parquet_path.exists():
            log.warning("Febrl parquet not found: %s", parquet_path)
            return {
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_f1": 0.0,
                "total_time_s": 0.0,
            }

        df = pd.read_parquet(parquet_path)

        # Extract record IDs and build ground truth clusters.
        # Febrl convention: IDs are "rec-N-org" (original) or "rec-N-dup-M" (duplicate).
        # Records with the same N belong to the same true cluster.
        record_ids = list(df.index)
        gold_clusters: dict[str, int] = {}
        for rec_id in record_ids:
            # Parse "rec-N-org" or "rec-N-dup-M" → cluster = N
            parts = str(rec_id).split("-")
            cluster_n = int(parts[1])
            gold_clusters[str(rec_id)] = cluster_n

        # Build entity names from the name columns (given_name + surname)
        entity_names = []
        for rec_id in record_ids:
            row = df.loc[rec_id]
            given = str(row.get("given_name", "")) if pd.notna(row.get("given_name")) else ""
            surname = str(row.get("surname", "")) if pd.notna(row.get("surname")) else ""
            name = f"{given} {surname}".strip()
            entity_names.append(name if name else str(rec_id))

        nodes_before = len(entity_names)

        if nodes_before < 2:
            return {
                "nodes_before": nodes_before,
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_f1": 0.0,
                "bcubed_f1": 0.0,
                "total_time_s": 0.0,
            }

        # Embed entity names
        t0 = time.perf_counter()
        log.info("  Embedding %d Febrl record names...", nodes_before)
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        entity_vectors = st_model.encode(entity_names, show_progress_bar=False, normalize_embeddings=True)
        entity_vectors = entity_vectors.astype(np.float32)

        # Run ER pipeline
        entity_to_canonical = _run_er_pipeline(conn, entity_names, entity_vectors)
        total_time = time.perf_counter() - t0

        # Build predicted clusters: map entity index to canonical cluster
        predicted_clusters: dict[str, int] = {}
        canonical_to_id: dict[str, int] = {}
        next_id = 0
        for i, name in enumerate(entity_names):
            canonical = entity_to_canonical[name]
            if canonical not in canonical_to_id:
                canonical_to_id[canonical] = next_id
                next_id += 1
            rec_id = str(record_ids[i])
            predicted_clusters[rec_id] = canonical_to_id[canonical]

        # Compute B-Cubed F1
        bcubed_result = bcubed_f1(predicted_clusters, gold_clusters)

        # Compute pairwise F1
        pairwise_result = _pairwise_f1(predicted_clusters, gold_clusters)

        # Store entities
        for i, name in enumerate(entity_names):
            cid = predicted_clusters[str(record_ids[i])]
            conn.execute("INSERT INTO entities(name, cluster_id) VALUES (?, ?)", (name, cid))
        conn.commit()

        return {
            "nodes_before": nodes_before,
            "nodes_after": len(canonical_to_id),
            "pairwise_precision": pairwise_result["precision"],
            "pairwise_recall": pairwise_result["recall"],
            "pairwise_f1": pairwise_result["f1"],
            "bcubed_precision": bcubed_result["precision"],
            "bcubed_recall": bcubed_result["recall"],
            "bcubed_f1": bcubed_result["f1"],
            "total_time_s": round(total_time, 3),
        }


def _pairwise_f1(predicted: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    """Compute pairwise precision, recall, F1 for clustering.

    Enumerates all pairs of elements that share a cluster in predicted/gold
    and computes set-based precision/recall/F1.
    """
    common = set(predicted.keys()) & set(gold.keys())
    if len(common) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Build cluster -> members
    pred_groups: dict[int, list[str]] = defaultdict(list)
    gold_groups: dict[int, list[str]] = defaultdict(list)
    for elem in common:
        pred_groups[predicted[elem]].append(elem)
        gold_groups[gold[elem]].append(elem)

    # Generate pair sets
    def _pairs(groups: dict[int, list[str]]) -> set[tuple[str, str]]:
        pair_set: set[tuple[str, str]] = set()
        for members in groups.values():
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    pair_set.add((min(a, b), max(a, b)))
        return pair_set

    pred_pairs = _pairs(pred_groups)
    gold_pairs = _pairs(gold_groups)

    if not pred_pairs and not gold_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
