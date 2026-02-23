"""Build the KG demo database for WASM and viz demos.

Produces a self-contained SQLite database with text chunks, FTS5, HNSW vector
indices, GLiNER entity extraction, GLiREL relation extraction, UMAP projections,
entity resolution via HNSW blocking + Jaro-Winkler + Leiden clustering, Node2Vec
structural embeddings, and provenance metadata.

Usage:
    python benchmarks/scripts/build_demo_db.py \
        --book-id 3300 \
        --output wasm/assets/3300.db \
        --embedding-model MiniLM \
        [--force]

Prerequisites:
    make all                              # builds the muninn extension
    make -C benchmarks prep-vectors       # precomputes MiniLM .npy embeddings
    uv pip install gliner glirel spacy sentence-transformers umap-learn numpy
    python -m spacy download en_core_web_sm
"""

import argparse
import datetime
import logging
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import spacy
import umap
from gliner import GLiNER
from glirel import GLiREL
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# ── Path constants ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"
VECTORS_DIR = BENCHMARKS_ROOT / "vectors"
KG_DIR = BENCHMARKS_ROOT / "kg"
TEXTS_DIR = BENCHMARKS_ROOT / "texts"
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Embedding model registry ─────────────────────────────────────

EMBEDDING_MODELS = {
    "MiniLM": {
        "st_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    "NomicEmbed": {
        "st_name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
    },
}

# ── NER labels for Wealth of Nations ─────────────────────────────

GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "economic concept",
    "commodity",
    "institution",
    "legal concept",
    "occupation",
]

# ── Relation labels for GLiREL ───────────────────────────────────

# GLiREL uses fixed_relation_types=True, so labels must be a flat list of strings.
GLIREL_LABELS = [
    "produces",
    "trades_with",
    "regulates",
    "employs",
    "located_in",
    "influences",
    "part_of",
    "opposes",
]

# ── Vector packing ────────────────────────────────────────────────


def pack_vector(v: np.ndarray | list[float]) -> bytes:
    """Pack a float list/array into a float32 BLOB for SQLite."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tobytes()
    return struct.pack(f"{len(v)}f", *v)


# ── Jaro-Winkler similarity (pure Python) ────────────────────────


def jaro_winkler(s1: str, s2: str) -> float:
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

    # Winkler prefix bonus (up to 4 chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


# ── GLiNER char-span to spaCy token-span converter ───────────────


def char_span_to_token_span(doc: spacy.tokens.Doc, char_start: int, char_end: int) -> tuple[int, int] | None:
    """Convert character offsets to spaCy token indices (inclusive start, exclusive end).

    Returns None if the span doesn't align with token boundaries.
    """
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span is None:
        return None
    return (span.start, span.end)


# ── Extension loading ─────────────────────────────────────────────


def load_muninn(conn: sqlite3.Connection) -> None:
    """Load the muninn extension into a SQLite connection."""
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Chunks + FTS + Chunk Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_1_chunks(conn: sqlite3.Connection, book_id: int, model_name: str) -> int:
    """Import chunks, build FTS5 index, load cached embeddings into HNSW."""
    t0 = time.monotonic()
    log.info("Phase 1: Chunks + FTS + chunk embeddings")

    # ── Import chunks from pre-built chunks DB ────────────────────
    chunks_db_path = KG_DIR / f"{book_id}_chunks.db"
    assert chunks_db_path.exists(), f"Chunks DB not found: {chunks_db_path}"

    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")

    src = sqlite3.connect(str(chunks_db_path))
    rows = src.execute("SELECT id, text FROM text_chunks ORDER BY id").fetchall()
    src.close()

    conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", rows)
    num_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    log.info("  Imported %d chunks from %s", num_chunks, chunks_db_path.name)

    # ── Build FTS5 index ──────────────────────────────────────────
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5("
        "  text, content=chunks, content_rowid=chunk_id"
        ")"
    )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    log.info("  Built FTS5 index (chunks_fts)")

    # ── Load cached chunk embeddings ──────────────────────────────
    npy_path = VECTORS_DIR / f"{model_name}_wealth_of_nations_docs.npy"
    assert npy_path.exists(), f"Cached embeddings not found: {npy_path}"

    vectors = np.load(str(npy_path))
    assert vectors.shape[0] == num_chunks, (
        f"Vector count mismatch: {vectors.shape[0]} vectors vs {num_chunks} chunks"
    )
    dim = vectors.shape[1]
    log.info("  Loaded %d vectors (dim=%d) from %s", vectors.shape[0], dim, npy_path.name)

    # ── Create HNSW index and insert vectors ──────────────────────
    conn.execute(
        f"CREATE VIRTUAL TABLE chunks_vec USING hnsw_index("
        f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
        f")"
    )

    for i in range(num_chunks):
        conn.execute(
            "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
            (i, pack_vector(vectors[i])),
        )

    log.info("  Inserted %d vectors into chunks_vec HNSW index", num_chunks)
    log.info("  Phase 1 complete (%.1fs)", time.monotonic() - t0)
    return num_chunks


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Entity Extraction (GLiNER zero-shot NER)
# ═══════════════════════════════════════════════════════════════════


def phase_2_ner(conn: sqlite3.Connection) -> int:
    """Extract entities from all chunks using GLiNER medium-v2.1."""
    t0 = time.monotonic()
    log.info("Phase 2: Entity extraction (GLiNER)")

    # ── Create entities table ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT,
            source TEXT NOT NULL,
            chunk_id INTEGER REFERENCES chunks(chunk_id),
            confidence REAL DEFAULT 1.0
        )
    """)
    conn.execute("CREATE INDEX idx_entities_name ON entities(name)")
    conn.execute("CREATE INDEX idx_entities_chunk ON entities(chunk_id)")

    # ── Load GLiNER model ─────────────────────────────────────────
    log.info("  Loading GLiNER medium-v2.1 model...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    log.info("  Model loaded")

    # ── Batch extract entities ────────────────────────────────────
    chunks = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    batch_size = 32
    total_entities = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [text for _, text in batch]
        chunk_ids = [cid for cid, _ in batch]

        results = model.batch_predict_entities(texts, GLINER_LABELS, threshold=0.3)

        insert_rows = []
        for chunk_id, entities in zip(chunk_ids, results):
            for ent in entities:
                insert_rows.append((
                    ent["text"],
                    ent["label"],
                    "gliner",
                    chunk_id,
                    ent["score"],
                ))

        conn.executemany(
            "INSERT INTO entities (name, entity_type, source, chunk_id, confidence)"
            " VALUES (?, ?, ?, ?, ?)",
            insert_rows,
        )
        total_entities += len(insert_rows)

        if (batch_start // batch_size) % 10 == 0:
            log.info(
                "  Processed %d/%d chunks (%d entities so far)",
                min(batch_start + batch_size, len(chunks)),
                len(chunks),
                total_entities,
            )

    log.info("  Extracted %d entity mentions across %d chunks", total_entities, len(chunks))
    log.info("  Phase 2 complete (%.1fs)", time.monotonic() - t0)
    return total_entities


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Relation Extraction (GLiREL zero-shot RE)
# ═══════════════════════════════════════════════════════════════════


def phase_3_re(conn: sqlite3.Connection) -> int:
    """Extract relations per chunk using GLiREL with GLiNER entity spans."""
    t0 = time.monotonic()
    log.info("Phase 3: Relation extraction (GLiREL)")

    # ── Create relations table ────────────────────────────────────
    conn.execute("""
        CREATE TABLE relations (
            relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            rel_type TEXT,
            weight REAL DEFAULT 1.0,
            chunk_id INTEGER,
            source TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_relations_src ON relations(src)")
    conn.execute("CREATE INDEX idx_relations_dst ON relations(dst)")

    # ── Load models ───────────────────────────────────────────────
    log.info("  Loading GLiREL large-v0 model...")
    # Workaround: glirel 1.2.1's _from_pretrained() declares proxies/resume_download
    # as required kwargs but huggingface_hub >=1.0 no longer passes them. Call directly.
    glirel_dir = snapshot_download("jackboyla/glirel-large-v0")
    re_model = GLiREL._from_pretrained(
        model_id=glirel_dir,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=True,
        token=None,
    )
    log.info("  Loading spaCy en_core_web_sm for tokenization...")
    nlp = spacy.load("en_core_web_sm")
    log.info("  Models loaded")

    # ── Process each chunk that has entities ──────────────────────
    chunks = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()

    # Pre-load entities grouped by chunk_id
    entity_rows = conn.execute(
        "SELECT chunk_id, name, entity_type, confidence FROM entities ORDER BY chunk_id"
    ).fetchall()
    entities_by_chunk: dict[int, list[tuple[str, str, float]]] = {}
    for chunk_id, name, etype, conf in entity_rows:
        entities_by_chunk.setdefault(chunk_id, []).append((name, etype, conf))

    total_relations = 0

    for chunk_id, text in chunks:
        chunk_entities = entities_by_chunk.get(chunk_id, [])
        if len(chunk_entities) < 2:
            continue  # Need at least 2 entities for relations

        # Tokenize with spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]

        # Convert entity mentions to token-level spans for GLiREL
        ner_spans = []
        for ent_name, ent_type, _conf in chunk_entities:
            # Find entity text in the chunk
            start_char = text.find(ent_name)
            if start_char == -1:
                # Try case-insensitive search
                lower_text = text.lower()
                start_char = lower_text.find(ent_name.lower())
            if start_char == -1:
                continue
            end_char = start_char + len(ent_name)

            token_span = char_span_to_token_span(doc, start_char, end_char)
            if token_span is None:
                continue

            ner_spans.append([token_span[0], token_span[1], ent_type, ent_name])

        if len(ner_spans) < 2:
            continue

        # Build position → entity name lookup for mapping GLiREL output back to NER entities.
        # GLiREL may extend span boundaries, so we match by overlap with NER spans.
        span_to_name: dict[tuple[int, int], str] = {}
        for span in ner_spans:
            span_to_name[(span[0], span[1])] = span[3]

        def _find_entity(pos: list[int]) -> str | None:
            """Map GLiREL head_pos/tail_pos to NER entity name via span overlap."""
            r_start, r_end = pos[0], pos[1]
            best_name = None
            best_overlap = 0
            for (s, e), name in span_to_name.items():
                overlap = max(0, min(e, r_end) - max(s, r_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = name
            return best_name

        # Extract relations
        relations = re_model.predict_relations(
            tokens, GLIREL_LABELS, threshold=0.5, ner=ner_spans, top_k=10
        )

        insert_rows = []
        for rel in relations:
            head = _find_entity(rel["head_pos"])
            tail = _find_entity(rel["tail_pos"])
            if head is None or tail is None or head == tail:
                continue
            insert_rows.append((
                head,
                tail,
                rel["label"],
                rel.get("score", 1.0),
                chunk_id,
                "glirel",
            ))

        if insert_rows:
            conn.executemany(
                "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
            total_relations += len(insert_rows)

        if chunk_id % 200 == 0:
            log.info("  Processed chunk %d (%d relations so far)", chunk_id, total_relations)

    log.info("  Extracted %d relations", total_relations)
    log.info("  Phase 3 complete (%.1fs)", time.monotonic() - t0)
    return total_relations


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Entity Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_4_entity_embeddings(conn: sqlite3.Connection, model_name: str) -> tuple[int, np.ndarray]:
    """Embed unique entity names and insert into HNSW index.

    Returns (num_entities, entity_vectors) for use by later phases.
    """
    t0 = time.monotonic()
    log.info("Phase 4: Entity embeddings")

    model_info = EMBEDDING_MODELS[model_name]
    dim = model_info["dim"]

    # ── Get unique entity names ───────────────────────────────────
    rows = conn.execute(
        "SELECT DISTINCT name FROM entities ORDER BY name"
    ).fetchall()
    entity_names = [r[0] for r in rows]
    log.info("  Found %d unique entity names", len(entity_names))

    # ── Embed with sentence-transformers ──────────────────────────
    log.info("  Loading SentenceTransformer %s...", model_info["st_name"])
    st_model = SentenceTransformer(model_info["st_name"])
    log.info("  Encoding %d entity names...", len(entity_names))
    entity_vectors = st_model.encode(entity_names, show_progress_bar=True, normalize_embeddings=True)
    entity_vectors = entity_vectors.astype(np.float32)
    log.info("  Encoded entity embeddings: shape=%s", entity_vectors.shape)

    # ── Create HNSW index ─────────────────────────────────────────
    conn.execute(
        f"CREATE VIRTUAL TABLE entities_vec USING hnsw_index("
        f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
        f")"
    )

    # ── Insert embeddings + build mapping table ───────────────────
    conn.execute(
        "CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)"
    )

    for i, (name, vec) in enumerate(zip(entity_names, entity_vectors)):
        rowid = i + 1
        conn.execute(
            "INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)",
            (rowid, pack_vector(vec)),
        )
        conn.execute(
            "INSERT INTO entity_vec_map (rowid, name) VALUES (?, ?)",
            (rowid, name),
        )

    log.info("  Inserted %d entity embeddings into entities_vec", len(entity_names))
    log.info("  Phase 4 complete (%.1fs)", time.monotonic() - t0)
    return len(entity_names), entity_vectors


# ═══════════════════════════════════════════════════════════════════
# Phase 5: UMAP Dimensionality Reduction
# ═══════════════════════════════════════════════════════════════════


def phase_5_umap(
    conn: sqlite3.Connection,
    num_chunks: int,
    chunk_vectors: np.ndarray,
    entity_vectors: np.ndarray,
) -> None:
    """Compute UMAP 2D + 3D projections for chunks and entities."""
    t0 = time.monotonic()
    log.info("Phase 5: UMAP dimensionality reduction")

    # ── UMAP 2D ───────────────────────────────────────────────────
    log.info("  Computing 2D UMAP on %d chunk vectors...", len(chunk_vectors))
    reducer_2d = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42
    )
    # Fit on chunks, transform entities into same space
    all_vectors = np.vstack([chunk_vectors, entity_vectors])
    proj_2d = reducer_2d.fit_transform(all_vectors)
    chunk_2d = proj_2d[: len(chunk_vectors)]
    entity_2d = proj_2d[len(chunk_vectors) :]

    # ── UMAP 3D ───────────────────────────────────────────────────
    log.info("  Computing 3D UMAP on %d vectors...", len(all_vectors))
    reducer_3d = umap.UMAP(
        n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42
    )
    proj_3d = reducer_3d.fit_transform(all_vectors)
    chunk_3d = proj_3d[: len(chunk_vectors)]
    entity_3d = proj_3d[len(chunk_vectors) :]

    # ── Store chunk projections ───────────────────────────────────
    conn.execute(
        "CREATE TABLE chunks_vec_umap ("
        "  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL"
        ")"
    )
    for i in range(num_chunks):
        conn.execute(
            "INSERT INTO chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            (i, float(chunk_2d[i, 0]), float(chunk_2d[i, 1]),
             float(chunk_3d[i, 0]), float(chunk_3d[i, 1]), float(chunk_3d[i, 2])),
        )

    # ── Store entity projections ──────────────────────────────────
    conn.execute(
        "CREATE TABLE entities_vec_umap ("
        "  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL"
        ")"
    )
    for i in range(len(entity_vectors)):
        rowid = i + 1  # Match entity_vec_map rowids (1-based)
        conn.execute(
            "INSERT INTO entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            (rowid, float(entity_2d[i, 0]), float(entity_2d[i, 1]),
             float(entity_3d[i, 0]), float(entity_3d[i, 1]), float(entity_3d[i, 2])),
        )

    log.info("  Stored UMAP projections: %d chunks + %d entities", num_chunks, len(entity_vectors))
    log.info("  Phase 5 complete (%.1fs)", time.monotonic() - t0)


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Entity Resolution (HNSW blocking + Jaro-Winkler + Leiden)
# ═══════════════════════════════════════════════════════════════════


def phase_6_entity_resolution(conn: sqlite3.Connection) -> tuple[int, int]:
    """Resolve entity synonyms using HNSW blocking, string similarity, and Leiden clustering.

    Returns (num_nodes, num_edges) in the coalesced graph.
    """
    t0 = time.monotonic()
    log.info("Phase 6: Entity resolution")

    # ── Get unique entities with mention counts ───────────────────
    entity_stats = conn.execute("""
        SELECT name, entity_type, count(*) as mention_count
        FROM entities
        GROUP BY name
        ORDER BY name
    """).fetchall()
    log.info("  %d unique entity names to resolve", len(entity_stats))

    entity_name_to_type = {name: etype for name, etype, _ in entity_stats}
    entity_name_to_count = {name: cnt for name, _, cnt in entity_stats}

    # ── HNSW blocking: find candidate match pairs ─────────────────
    # For each entity, KNN search to find close neighbors in embedding space
    entity_names_ordered = conn.execute(
        "SELECT name FROM entity_vec_map ORDER BY rowid"
    ).fetchall()
    entity_names_ordered = [r[0] for r in entity_names_ordered]

    # Build name → rowid mapping
    name_to_rowid = {}
    for row in conn.execute("SELECT rowid, name FROM entity_vec_map"):
        name_to_rowid[row[1]] = row[0]

    k_neighbors = 10
    candidate_pairs: list[tuple[str, str, float]] = []  # (name1, name2, cosine_dist)

    log.info("  HNSW blocking: finding %d nearest neighbors per entity...", k_neighbors)
    for name in entity_names_ordered:
        rowid = name_to_rowid[name]
        vec = conn.execute(
            "SELECT vector FROM entities_vec WHERE rowid = ?", (rowid,)
        ).fetchone()[0]

        # KNN search
        neighbors = conn.execute(
            "SELECT rowid, distance FROM entities_vec WHERE vector MATCH ? AND k = ?",
            (vec, k_neighbors + 1),  # +1 because self is included
        ).fetchall()

        for neighbor_rowid, distance in neighbors:
            if neighbor_rowid == rowid:
                continue
            if distance > 0.4:  # cosine distance threshold
                continue
            neighbor_name = entity_names_ordered[neighbor_rowid - 1]
            # Avoid duplicate pairs (only keep lexicographically ordered)
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

    log.info("  Found %d candidate pairs from HNSW blocking", len(unique_pairs))

    # ── Matching cascade: score each candidate pair ───────────────
    match_edges: list[tuple[str, str, float]] = []

    for n1, n2, cosine_dist in unique_pairs:
        cosine_sim = 1.0 - cosine_dist

        # Exact match
        if n1 == n2:
            match_edges.append((n1, n2, 1.0))
            continue

        # Case-insensitive exact
        if n1.lower() == n2.lower():
            match_edges.append((n1, n2, 0.9))
            continue

        # Jaro-Winkler on lowercased names
        jw = jaro_winkler(n1.lower(), n2.lower())

        # Combined score: 0.4 * Jaro-Winkler + 0.6 * cosine similarity
        combined = 0.4 * jw + 0.6 * cosine_sim

        if combined > 0.5:
            match_edges.append((n1, n2, combined))

    log.info("  %d match pairs above threshold 0.5", len(match_edges))

    # ── Leiden clustering on match pairs ──────────────────────────
    conn.execute(
        "CREATE TABLE _match_edges (src TEXT NOT NULL, dst TEXT NOT NULL, weight REAL DEFAULT 1.0)"
    )
    conn.executemany(
        "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
        match_edges,
    )

    # Also insert reverse edges (Leiden expects undirected)
    conn.executemany(
        "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
        [(n2, n1, w) for n1, n2, w in match_edges],
    )

    # Run Leiden if we have match edges
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

        # For each community, pick canonical = highest mention-count member
        for comm_id, members in communities.items():
            canonical = max(members, key=lambda n: entity_name_to_count.get(n, 0))
            for member in members:
                entity_to_canonical[member] = canonical

        log.info("  Leiden found %d communities from %d matched entities",
                 len(communities), len(leiden_results))

    # Entities not in any match pair are their own canonical form
    for name in entity_name_to_type:
        if name not in entity_to_canonical:
            entity_to_canonical[name] = name

    # ── Populate entity_clusters table ────────────────────────────
    conn.execute("CREATE TABLE entity_clusters (name TEXT PRIMARY KEY, canonical TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO entity_clusters (name, canonical) VALUES (?, ?)",
        entity_to_canonical.items(),
    )

    # ── Build clean graph: nodes + edges ──────────────────────────
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT,
            mention_count INTEGER DEFAULT 0
        )
    """)

    # Aggregate canonical entities
    canonical_stats: dict[str, dict] = {}
    for name, etype, count in entity_stats:
        canonical = entity_to_canonical[name]
        if canonical not in canonical_stats:
            canonical_stats[canonical] = {
                "entity_type": etype,
                "mention_count": 0,
            }
        canonical_stats[canonical]["mention_count"] += count

    # Insert nodes (sorted for deterministic node_ids)
    for canonical in sorted(canonical_stats):
        stats = canonical_stats[canonical]
        conn.execute(
            "INSERT INTO nodes (name, entity_type, mention_count) VALUES (?, ?, ?)",
            (canonical, stats["entity_type"], stats["mention_count"]),
        )

    num_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    log.info("  Built nodes table: %d canonical entities", num_nodes)

    # ── Coalesce relations into edges ─────────────────────────────
    conn.execute("""
        CREATE TABLE edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            rel_type TEXT,
            weight REAL DEFAULT 1.0,
            PRIMARY KEY (src, dst, rel_type)
        )
    """)

    # Aggregate relations using canonical names
    raw_relations = conn.execute(
        "SELECT src, dst, rel_type, weight FROM relations"
    ).fetchall()

    edge_agg: dict[tuple[str, str, str], float] = {}
    for src, dst, rel_type, weight in raw_relations:
        c_src = entity_to_canonical.get(src, src)
        c_dst = entity_to_canonical.get(dst, dst)
        # Skip self-loops
        if c_src == c_dst:
            continue
        key = (c_src, c_dst, rel_type)
        edge_agg[key] = edge_agg.get(key, 0.0) + weight

    conn.executemany(
        "INSERT OR IGNORE INTO edges (src, dst, rel_type, weight) VALUES (?, ?, ?, ?)",
        [(src, dst, rt, w) for (src, dst, rt), w in edge_agg.items()],
    )

    num_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    log.info("  Built edges table: %d coalesced edges", num_edges)

    # Clean up temporary table
    conn.execute("DROP TABLE _match_edges")

    log.info("  Phase 6 complete (%.1fs)", time.monotonic() - t0)
    return num_nodes, num_edges


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Node2Vec Structural Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_7_node2vec(conn: sqlite3.Connection) -> int:
    """Train Node2Vec on the coalesced graph and store structural embeddings."""
    t0 = time.monotonic()
    log.info("Phase 7: Node2Vec structural embeddings")

    n2v_dim = 64

    # ── Create output HNSW table ──────────────────────────────────
    conn.execute(
        f"CREATE VIRTUAL TABLE node2vec_emb USING hnsw_index("
        f"  dimensions={n2v_dim}, metric='cosine', m=16, ef_construction=200"
        f")"
    )

    # ── Build integer edge table for Node2Vec ─────────────────────
    # Node2Vec's graph_load reads edges via SQL. We create a dedicated
    # integer-keyed edge table where node IDs match the nodes table,
    # ensuring a deterministic mapping from HNSW rowid back to nodes.
    conn.execute("CREATE TABLE n2v_edges (src INTEGER NOT NULL, dst INTEGER NOT NULL)")

    # Insert edges mapped to node_ids, ordered by (src_id, dst_id)
    # so that the graph_load encounter order matches node_id order
    conn.execute("""
        INSERT INTO n2v_edges (src, dst)
        SELECT n1.node_id, n2.node_id
        FROM edges e
        JOIN nodes n1 ON n1.name = e.src
        JOIN nodes n2 ON n2.name = e.dst
        ORDER BY n1.node_id, n2.node_id
    """)

    n2v_edge_count = conn.execute("SELECT count(*) FROM n2v_edges").fetchone()[0]
    log.info("  Prepared %d integer edges for Node2Vec", n2v_edge_count)

    if n2v_edge_count == 0:
        log.info("  No edges — skipping Node2Vec training")
        conn.execute("DROP TABLE n2v_edges")
        log.info("  Phase 7 complete (%.1fs)", time.monotonic() - t0)
        return 0

    # ── Train Node2Vec ────────────────────────────────────────────
    # Parameters: p=0.5, q=0.5, walks=10, walk_length=40,
    #             window=5, neg_samples=5, learning_rate=0.025, epochs=5
    result = conn.execute(
        "SELECT node2vec_train("
        "  'n2v_edges', 'src', 'dst', 'node2vec_emb',"
        "  64, 0.5, 0.5, 10, 40, 5, 5, 0.025, 5"
        ")"
    ).fetchone()

    num_embedded = result[0]
    log.info("  Node2Vec embedded %d nodes (dim=%d)", num_embedded, n2v_dim)

    # Clean up
    conn.execute("DROP TABLE n2v_edges")

    log.info("  Phase 7 complete (%.1fs)", time.monotonic() - t0)
    return num_embedded


# ═══════════════════════════════════════════════════════════════════
# Phase 8: Metadata + Validation
# ═══════════════════════════════════════════════════════════════════


def phase_8_metadata(
    conn: sqlite3.Connection,
    book_id: int,
    model_name: str,
    num_chunks: int,
    num_entities: int,
    num_relations: int,
    num_nodes: int,
    num_edges: int,
    num_n2v: int,
) -> None:
    """Write metadata table, validate all tables, and VACUUM."""
    t0 = time.monotonic()
    log.info("Phase 8: Metadata + validation")

    # ── Write meta table ──────────────────────────────────────────
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

    meta_rows = [
        ("book_id", str(book_id)),
        ("text_file", f"gutenberg_{book_id}.txt"),
        ("embedding_model", model_name),
        ("embedding_dim", str(EMBEDDING_MODELS[model_name]["dim"])),
        ("ner_model", "urchade/gliner_medium-v2.1"),
        ("re_model", "jackboyla/glirel-large-v0"),
        ("strategies", "gliner+glirel"),
        ("num_chunks", str(num_chunks)),
        ("total_entities", str(num_entities)),
        ("total_relations", str(num_relations)),
        ("num_nodes", str(num_nodes)),
        ("num_edges", str(num_edges)),
        ("num_n2v_embeddings", str(num_n2v)),
        ("build_timestamp", datetime.datetime.now(datetime.UTC).isoformat()),
        ("builder", "build_demo_db.py"),
    ]
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

    # ── Validation pass ───────────────────────────────────────────
    log.info("  Validating tables...")
    expected_tables = [
        ("chunks", num_chunks),
        ("entities", None),  # Count varies
        ("relations", None),
        ("entity_clusters", None),
        ("entity_vec_map", None),
        ("nodes", num_nodes),
        ("edges", num_edges),
        ("chunks_vec_umap", num_chunks),
        ("entities_vec_umap", None),
        ("meta", len(meta_rows)),
    ]

    for table_name, expected_count in expected_tables:
        actual = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
        if expected_count is not None:
            assert actual == expected_count, (
                f"Table {table_name}: expected {expected_count} rows, got {actual}"
            )
        assert actual > 0, f"Table {table_name} is empty!"
        log.info("    %s: %d rows", table_name, actual)

    # Validate virtual tables (HNSW)
    for vt_name in ["chunks_vec", "entities_vec", "node2vec_emb"]:
        # HNSW tables store count in shadow table
        count = conn.execute(
            f'SELECT count(*) FROM "{vt_name}_nodes"'
        ).fetchone()[0]
        log.info("    %s: %d vectors", vt_name, count)

    # Validate FTS5
    fts_count = conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'labour'"
    ).fetchone()[0]
    log.info("    chunks_fts: FTS5 working (%d matches for 'labour')", fts_count)

    log.info("  Phase 8 complete (%.1fs)", time.monotonic() - t0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the KG demo database for WASM and viz demos."
    )
    parser.add_argument("--book-id", type=int, default=3300, help="Gutenberg book ID (default: 3300)")
    parser.add_argument(
        "--output", type=str, default="wasm/assets/3300.db",
        help="Output database path (default: wasm/assets/3300.db)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="MiniLM",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model name (default: MiniLM)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # ── Pre-flight checks ─────────────────────────────────────────
    assert Path(MUNINN_PATH).with_suffix(".dylib").exists() or \
           Path(MUNINN_PATH).with_suffix(".so").exists(), \
           f"Muninn extension not found at {MUNINN_PATH}. Run: make all"

    if output_path.exists():
        if args.force:
            log.info("Removing existing %s (--force)", output_path)
            output_path.unlink()
        else:
            log.error("Output file already exists: %s (use --force to overwrite)", output_path)
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Create database ───────────────────────────────────────────
    t_total = time.monotonic()
    log.info("Building demo database: %s", output_path)
    log.info("Book ID: %d, Embedding model: %s", args.book_id, args.embedding_model)

    conn = sqlite3.connect(str(output_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    load_muninn(conn)

    # ── Run all phases sequentially ───────────────────────────────
    num_chunks = phase_1_chunks(conn, args.book_id, args.embedding_model)
    conn.commit()

    num_entity_mentions = phase_2_ner(conn)
    conn.commit()

    num_relations = phase_3_re(conn)
    conn.commit()

    num_unique_entities, entity_vectors = phase_4_entity_embeddings(conn, args.embedding_model)
    conn.commit()

    # Load chunk vectors for UMAP (same vectors used in Phase 1)
    npy_path = VECTORS_DIR / f"{args.embedding_model}_wealth_of_nations_docs.npy"
    chunk_vectors = np.load(str(npy_path))

    phase_5_umap(conn, num_chunks, chunk_vectors, entity_vectors)
    conn.commit()

    num_nodes, num_edges = phase_6_entity_resolution(conn)
    conn.commit()

    num_n2v = phase_7_node2vec(conn)
    conn.commit()

    phase_8_metadata(
        conn, args.book_id, args.embedding_model,
        num_chunks, num_entity_mentions, num_relations,
        num_nodes, num_edges, num_n2v,
    )
    conn.commit()

    # ── VACUUM ────────────────────────────────────────────────────
    log.info("VACUUMing database...")
    conn.execute("PRAGMA journal_mode=DELETE")  # VACUUM requires DELETE mode
    conn.execute("VACUUM")
    conn.close()

    db_size = output_path.stat().st_size
    elapsed = time.monotonic() - t_total
    log.info("Done! %s (%.1f MB) built in %.1fs", output_path.name, db_size / 1e6, elapsed)


if __name__ == "__main__":
    main()
