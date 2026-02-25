"""Phase functions for the demo database build pipeline.

Each phase is a module-level function with uniform signature:
    (conn, ctx, ...) -> None

Where conn is the SQLite connection, ctx is the mutable PhaseContext,
and extra args are only what that specific phase needs. Each phase
writes its outputs to ctx fields for downstream phases to consume.

This module has top-level ML imports (numpy, spacy, umap, etc.) which
is safe because it is only imported inside _cmd_build() via deferred
import in cli.py.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from typing import TYPE_CHECKING

import numpy as np
import umap

from benchmarks.demo_builder.common import (
    char_span_to_token_span,
    jaro_winkler,
    load_chunk_vectors,
    pack_vector,
)
from benchmarks.demo_builder.constants import (
    EMBEDDING_MODELS,
    GLINER_LABELS,
    GLIREL_LABELS,
    KG_DIR,
)

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext
    from benchmarks.demo_builder.models import ModelPool

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Chunks + FTS + Chunk Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_1_chunks(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
    book_id: int,
    model_name: str,
    models: ModelPool,
) -> None:
    """Import chunks, build FTS5 index, load/compute embeddings, insert into HNSW.

    Writes to ctx: num_chunks, chunk_vectors.
    """
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
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(  text, content=chunks, content_rowid=chunk_id)")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    log.info("  Built FTS5 index (chunks_fts)")

    # ── Load or compute chunk embeddings ──────────────────────────
    chunk_texts = [text for _, text in rows]
    st_model = models.st_model(model_name)
    chunk_vectors = load_chunk_vectors(book_id, num_chunks, model_name, st_model, chunk_texts)
    dim = chunk_vectors.shape[1]

    # ── Create HNSW index and insert vectors ──────────────────────
    conn.execute(
        f"CREATE VIRTUAL TABLE chunks_vec USING hnsw_index("
        f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
        f")"
    )

    for i in range(num_chunks):
        conn.execute(
            "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
            (i, pack_vector(chunk_vectors[i])),
        )

    log.info("  Inserted %d vectors into chunks_vec HNSW index", num_chunks)
    log.info("  Phase 1 complete (%.1fs)", time.monotonic() - t0)

    ctx.num_chunks = num_chunks
    ctx.chunk_vectors = chunk_vectors


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Entity Extraction (GLiNER zero-shot NER)
# ═══════════════════════════════════════════════════════════════════


def phase_2_ner(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
    models: ModelPool,
) -> None:
    """Extract entities from all chunks using a pre-loaded GLiNER model.

    Writes to ctx: num_entity_mentions.
    """
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

    # ── Batch extract entities ────────────────────────────────────
    chunks = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    batch_size = 32
    total_entities = 0
    ner_model = models.ner_model

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [text for _, text in batch]
        chunk_ids = [cid for cid, _ in batch]

        results = ner_model.batch_predict_entities(texts, GLINER_LABELS, threshold=0.3)

        insert_rows = []
        for chunk_id, entities in zip(chunk_ids, results, strict=True):
            for ent in entities:
                insert_rows.append(
                    (
                        ent["text"],
                        ent["label"],
                        "gliner",
                        chunk_id,
                        ent["score"],
                    )
                )

        conn.executemany(
            "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
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

    ctx.num_entity_mentions = total_entities


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Relation Extraction (GLiREL zero-shot RE)
# ═══════════════════════════════════════════════════════════════════


def phase_3_re(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
    models: ModelPool,
) -> None:
    """Extract relations per chunk using pre-loaded GLiREL and spaCy models.

    Writes to ctx: num_relations.
    """
    t0 = time.monotonic()
    log.info("Phase 3: Relation extraction (GLiREL)")

    re_model = models.re_model
    nlp = models.nlp

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

        # Build position -> entity name lookup for mapping GLiREL output back to NER entities.
        # GLiREL may extend span boundaries, so we match by overlap with NER spans.
        span_to_name: dict[tuple[int, int], str] = {}
        for span in ner_spans:
            s_start: int = span[0]  # type: ignore[assignment]
            s_end: int = span[1]  # type: ignore[assignment]
            s_name: str = span[3]  # type: ignore[assignment]
            span_to_name[(s_start, s_end)] = s_name

        def _find_entity(pos: list[int], _lookup: dict[tuple[int, int], str] = span_to_name) -> str | None:
            """Map GLiREL head_pos/tail_pos to NER entity name via span overlap."""
            r_start, r_end = pos[0], pos[1]
            best_name = None
            best_overlap = 0
            for (s, e), name in _lookup.items():
                overlap = max(0, min(e, r_end) - max(s, r_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = name
            return best_name

        # Extract relations
        relations = re_model.predict_relations(tokens, GLIREL_LABELS, threshold=0.5, ner=ner_spans, top_k=10)

        insert_rows = []
        for rel in relations:
            head = _find_entity(rel["head_pos"])
            tail = _find_entity(rel["tail_pos"])
            if head is None or tail is None or head == tail:
                continue
            insert_rows.append(
                (
                    head,
                    tail,
                    rel["label"],
                    rel.get("score", 1.0),
                    chunk_id,
                    "glirel",
                )
            )

        if insert_rows:
            conn.executemany(
                "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
            total_relations += len(insert_rows)

        if chunk_id % 200 == 0:
            log.info("  Processed chunk %d (%d relations so far)", chunk_id, total_relations)

    log.info("  Extracted %d relations", total_relations)
    log.info("  Phase 3 complete (%.1fs)", time.monotonic() - t0)

    ctx.num_relations = total_relations


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Entity Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_4_entity_embeddings(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
    model_name: str,
    models: ModelPool,
) -> None:
    """Embed unique entity names and insert into HNSW index.

    Writes to ctx: num_unique_entities, entity_vectors.
    """
    t0 = time.monotonic()
    log.info("Phase 4: Entity embeddings")

    model_info = EMBEDDING_MODELS[model_name]
    dim = model_info["dim"]
    st_model = models.st_model(model_name)

    # ── Get unique entity names ───────────────────────────────────
    rows = conn.execute("SELECT DISTINCT name FROM entities ORDER BY name").fetchall()
    entity_names = [r[0] for r in rows]
    log.info("  Found %d unique entity names", len(entity_names))

    # ── Embed with sentence-transformers ──────────────────────────
    log.info("  Encoding %d entity names with %s...", len(entity_names), model_info["st_name"])
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
    conn.execute("CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)")

    for i, (name, vec) in enumerate(zip(entity_names, entity_vectors, strict=True)):
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

    ctx.num_unique_entities = len(entity_names)
    ctx.entity_vectors = entity_vectors


# ═══════════════════════════════════════════════════════════════════
# Phase 5: UMAP Dimensionality Reduction
# ═══════════════════════════════════════════════════════════════════


def phase_5_umap(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
) -> None:
    """Compute UMAP 2D + 3D projections for chunks and entities.

    Reads from ctx: chunk_vectors, entity_vectors, num_chunks.
    """
    t0 = time.monotonic()
    log.info("Phase 5: UMAP dimensionality reduction")

    assert ctx.chunk_vectors is not None, "Phase 5 requires chunk_vectors from Phase 1"
    assert ctx.entity_vectors is not None, "Phase 5 requires entity_vectors from Phase 4"
    chunk_vectors = ctx.chunk_vectors
    entity_vectors = ctx.entity_vectors
    num_chunks = ctx.num_chunks

    # ── UMAP 2D ───────────────────────────────────────────────────
    log.info("  Computing 2D UMAP on %d chunk vectors...", len(chunk_vectors))
    reducer_2d = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    # Fit on chunks, transform entities into same space
    all_vectors = np.vstack([chunk_vectors, entity_vectors])
    proj_2d = reducer_2d.fit_transform(all_vectors)
    chunk_2d = proj_2d[: len(chunk_vectors)]
    entity_2d = proj_2d[len(chunk_vectors) :]

    # ── UMAP 3D ───────────────────────────────────────────────────
    log.info("  Computing 3D UMAP on %d vectors...", len(all_vectors))
    reducer_3d = umap.UMAP(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    proj_3d = reducer_3d.fit_transform(all_vectors)
    chunk_3d = proj_3d[: len(chunk_vectors)]
    entity_3d = proj_3d[len(chunk_vectors) :]

    # ── Store chunk projections ───────────────────────────────────
    conn.execute(
        "CREATE TABLE chunks_vec_umap (  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
    )
    for i in range(num_chunks):
        conn.execute(
            "INSERT INTO chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            (
                i,
                float(chunk_2d[i, 0]),
                float(chunk_2d[i, 1]),
                float(chunk_3d[i, 0]),
                float(chunk_3d[i, 1]),
                float(chunk_3d[i, 2]),
            ),
        )

    # ── Store entity projections ──────────────────────────────────
    conn.execute(
        "CREATE TABLE entities_vec_umap (  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
    )
    for i in range(len(entity_vectors)):
        rowid = i + 1  # Match entity_vec_map rowids (1-based)
        conn.execute(
            "INSERT INTO entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            (
                rowid,
                float(entity_2d[i, 0]),
                float(entity_2d[i, 1]),
                float(entity_3d[i, 0]),
                float(entity_3d[i, 1]),
                float(entity_3d[i, 2]),
            ),
        )

    log.info("  Stored UMAP projections: %d chunks + %d entities", num_chunks, len(entity_vectors))
    log.info("  Phase 5 complete (%.1fs)", time.monotonic() - t0)


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Entity Resolution (HNSW blocking + Jaro-Winkler + Leiden)
# ═══════════════════════════════════════════════════════════════════


def phase_6_entity_resolution(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
) -> None:
    """Resolve entity synonyms using HNSW blocking, string similarity, and Leiden clustering.

    Writes to ctx: num_nodes, num_edges.
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
    entity_names_ordered = conn.execute("SELECT name FROM entity_vec_map ORDER BY rowid").fetchall()
    entity_names_ordered = [r[0] for r in entity_names_ordered]

    # Build name -> rowid mapping
    name_to_rowid = {}
    for row in conn.execute("SELECT rowid, name FROM entity_vec_map"):
        name_to_rowid[row[1]] = row[0]

    k_neighbors = 10
    candidate_pairs: list[tuple[str, str, float]] = []

    log.info("  HNSW blocking: finding %d nearest neighbors per entity...", k_neighbors)
    for name in entity_names_ordered:
        rowid = name_to_rowid[name]
        vec = conn.execute("SELECT vector FROM entities_vec WHERE rowid = ?", (rowid,)).fetchone()[0]

        # KNN search
        neighbors = conn.execute(
            "SELECT rowid, distance FROM entities_vec WHERE vector MATCH ? AND k = ?",
            (vec, k_neighbors + 1),
        ).fetchall()

        for neighbor_rowid, distance in neighbors:
            if neighbor_rowid == rowid:
                continue
            if distance > 0.4:
                continue
            neighbor_name = entity_names_ordered[neighbor_rowid - 1]
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
    conn.execute("CREATE TABLE _match_edges (src TEXT NOT NULL, dst TEXT NOT NULL, weight REAL DEFAULT 1.0)")
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
        for _comm_id, members in communities.items():
            canonical = max(members, key=lambda n: entity_name_to_count.get(n, 0))
            for member in members:
                entity_to_canonical[member] = canonical

        log.info("  Leiden found %d communities from %d matched entities", len(communities), len(leiden_results))

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
    canonical_stats: dict[str, dict[str, str | int]] = {}
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
    raw_relations = conn.execute("SELECT src, dst, rel_type, weight FROM relations").fetchall()

    edge_agg: dict[tuple[str, str, str], float] = {}
    for src, dst, rel_type, weight in raw_relations:
        c_src: str = entity_to_canonical.get(src, src)
        c_dst: str = entity_to_canonical.get(dst, dst)
        if c_src == c_dst:
            continue
        edge_key: tuple[str, str, str] = (c_src, c_dst, str(rel_type))
        edge_agg[edge_key] = edge_agg.get(edge_key, 0.0) + float(weight)

    conn.executemany(
        "INSERT OR IGNORE INTO edges (src, dst, rel_type, weight) VALUES (?, ?, ?, ?)",
        [(src, dst, rt, w) for (src, dst, rt), w in edge_agg.items()],
    )

    num_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    log.info("  Built edges table: %d coalesced edges", num_edges)

    # Clean up temporary table
    conn.execute("DROP TABLE _match_edges")

    log.info("  Phase 6 complete (%.1fs)", time.monotonic() - t0)

    ctx.num_nodes = num_nodes
    ctx.num_edges = num_edges


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Node2Vec Structural Embeddings
# ═══════════════════════════════════════════════════════════════════


def phase_7_node2vec(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
) -> None:
    """Train Node2Vec on the coalesced graph and store structural embeddings.

    Writes to ctx: num_n2v.
    """
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
    conn.execute("CREATE TABLE n2v_edges (src INTEGER NOT NULL, dst INTEGER NOT NULL)")

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
        log.info("  No edges -- skipping Node2Vec training")
        conn.execute("DROP TABLE n2v_edges")
        log.info("  Phase 7 complete (%.1fs)", time.monotonic() - t0)
        ctx.num_n2v = 0
        return

    # ── Train Node2Vec ────────────────────────────────────────────
    result = conn.execute(
        "SELECT node2vec_train(  'n2v_edges', 'src', 'dst', 'node2vec_emb',  64, 0.5, 0.5, 10, 40, 5, 5, 0.025, 5)"
    ).fetchone()

    num_embedded = result[0]
    log.info("  Node2Vec embedded %d nodes (dim=%d)", num_embedded, n2v_dim)

    # Clean up
    conn.execute("DROP TABLE n2v_edges")

    log.info("  Phase 7 complete (%.1fs)", time.monotonic() - t0)

    ctx.num_n2v = num_embedded


# ═══════════════════════════════════════════════════════════════════
# Phase 8: Metadata + Validation
# ═══════════════════════════════════════════════════════════════════


def phase_8_metadata(
    conn: sqlite3.Connection,
    ctx: PhaseContext,
    book_id: int,
    model_name: str,
) -> None:
    """Write metadata table, validate all tables.

    Reads from ctx: num_chunks, num_entity_mentions, num_relations, num_nodes, num_edges, num_n2v.
    """
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
        ("num_chunks", str(ctx.num_chunks)),
        ("total_entities", str(ctx.num_entity_mentions)),
        ("total_relations", str(ctx.num_relations)),
        ("num_nodes", str(ctx.num_nodes)),
        ("num_edges", str(ctx.num_edges)),
        ("num_n2v_embeddings", str(ctx.num_n2v)),
        ("build_timestamp", datetime.datetime.now(datetime.UTC).isoformat()),
        ("builder", "benchmarks.demo_builder"),
    ]
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

    # ── Validation pass ───────────────────────────────────────────
    log.info("  Validating tables...")
    expected_tables = [
        ("chunks", ctx.num_chunks),
        ("entities", None),
        ("relations", None),
        ("entity_clusters", None),
        ("entity_vec_map", None),
        ("nodes", ctx.num_nodes),
        ("edges", ctx.num_edges),
        ("chunks_vec_umap", ctx.num_chunks),
        ("entities_vec_umap", None),
        ("meta", len(meta_rows)),
    ]

    for table_name, expected_count in expected_tables:
        actual = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
        if expected_count is not None:
            assert actual == expected_count, f"Table {table_name}: expected {expected_count} rows, got {actual}"
        assert actual > 0, f"Table {table_name} is empty!"
        log.info("    %s: %d rows", table_name, actual)

    # Validate virtual tables (HNSW)
    for vt_name in ["chunks_vec", "entities_vec", "node2vec_emb"]:
        count = conn.execute(f'SELECT count(*) FROM "{vt_name}_nodes"').fetchone()[0]
        log.info("    %s: %d vectors", vt_name, count)

    # Validate FTS5
    fts_count = conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'labour'").fetchone()[0]
    log.info("    chunks_fts: FTS5 working (%d matches for 'labour')", fts_count)

    log.info("  Phase 8 complete (%.1fs)", time.monotonic() - t0)
