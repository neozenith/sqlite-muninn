"""
Knowledge Graph Extraction Pipeline — Entity/relation extraction from Gutenberg texts.

Three extraction strategies (independently guarded):
1. GLiNER zero-shot NER — Custom entity types. Requires `gliner`.
2. spaCy SVO triples — Subject-verb-object relation extraction. Requires `spacy`, `textacy`.
3. FTS5 concept discovery — Zero-dependency! Uses SQLite's built-in FTS5 + seed entity list.

Minimum viable run: With only `sentence-transformers` installed (already in benchmark deps),
the FTS5 strategy alone produces a useful KG with HNSW-indexed embeddings.

Output: SQLite database at benchmarks/kg/{book_id}.db
"""

import argparse
import collections
import logging
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from gliner import GLiNER

    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

try:
    import spacy
    from textacy.extract import subject_verb_object_triples

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MUNINN_PATH = str(PROJECT_ROOT / "muninn")
TEXTS_DIR = PROJECT_ROOT / "benchmarks" / "texts"
KG_DIR = PROJECT_ROOT / "benchmarks" / "kg"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# HNSW parameters for KG indexes
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

# Chunking parameters
CHUNK_WINDOW = 256
CHUNK_OVERLAP = 50

# Seed entities for the Wealth of Nations (FTS5 strategy).
# These are canonical economic concepts from Adam Smith's work that anchor
# the zero-dependency FTS5 extraction when no NER model is available.
ECONOMICS_SEED_ENTITIES = {
    # Core concepts
    "labour",
    "capital",
    "wages",
    "profit",
    "rent",
    "price",
    "market",
    "value",
    "money",
    "trade",
    "commerce",
    "industry",
    "agriculture",
    "manufacture",
    "stock",
    "revenue",
    "tax",
    "wealth",
    "nation",
    "commodity",
    "employment",
    "production",
    "consumption",
    "exchange",
    # Institutions and actors
    "merchant",
    "landlord",
    "farmer",
    "workman",
    "sovereign",
    "government",
    "parliament",
    "colony",
    "bank",
    "company",
    "corporation",
    # Economic mechanisms
    "supply",
    "demand",
    "competition",
    "monopoly",
    "bounty",
    "duty",
    "interest",
    "credit",
    "debt",
    "coin",
    "silver",
    "gold",
    "importation",
    "exportation",
    # Related concepts
    "property",
    "liberty",
    "poverty",
    "improvement",
    "invention",
    "division",
    "accumulation",
    "proportion",
    "quantity",
}


# ── Utilities ─────────────────────────────────────────────────────────


def pack_vector(v):
    """Pack a float list/array into a float32 BLOB for SQLite."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tobytes()
    return struct.pack(f"{len(v)}f", *v)


def chunk_fixed_tokens(text, window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP):
    """Split text into fixed-size token windows with overlap.

    Uses word-level tokenization as an approximation of sub-word tokens.
    Returns a list of text chunks.
    """
    words = text.split()
    if not words:
        return []

    stride = max(1, window - overlap)
    chunks = []
    for i in range(0, len(words), stride):
        chunk_words = words[i : i + window]
        if len(chunk_words) < window // 4:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


def normalize_entity_name(name):
    """Normalize an entity name for matching: lowercase, collapse whitespace, strip articles."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    # Strip leading articles
    for article in ("the ", "a ", "an "):
        if name.startswith(article):
            name = name[len(article) :]
    return name


# ── Schema ────────────────────────────────────────────────────────────


SCHEMA_SQL = """
-- Metadata
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- Text passages
CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL);

-- FTS5 index on chunks
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id);

-- Raw entities (pre-coalescing)
CREATE TABLE IF NOT EXISTS entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT,
    source TEXT NOT NULL,
    chunk_id INTEGER REFERENCES chunks(chunk_id),
    confidence REAL DEFAULT 1.0
);

-- Relations (graph edges)
CREATE TABLE IF NOT EXISTS relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel_type TEXT,
    weight REAL DEFAULT 1.0,
    chunk_id INTEGER,
    source TEXT NOT NULL
);

-- Indexes for efficient lookup
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst);
"""


def create_schema(conn):
    """Create the KG schema and HNSW virtual tables."""
    conn.executescript(SCHEMA_SQL)

    # HNSW for chunk embeddings (VSS entry point)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING hnsw_index(
            dimensions={EMBEDDING_DIM}, metric='cosine', m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}
        )
    """)

    # HNSW for entity name embeddings (used in coalescing for blocking)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING hnsw_index(
            dimensions={EMBEDDING_DIM}, metric='cosine', m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}
        )
    """)

    conn.commit()
    log.info("Schema created with HNSW tables (dim=%d)", EMBEDDING_DIM)


# ── Chunk insertion ───────────────────────────────────────────────────


def insert_chunks(conn, chunks):
    """Insert text chunks and rebuild FTS5 index."""
    conn.executemany(
        "INSERT INTO chunks (chunk_id, text) VALUES (?, ?)",
        enumerate(chunks, 1),
    )
    # Rebuild FTS5 content-sync index
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    conn.commit()
    log.info("Inserted %d chunks + rebuilt FTS5 index", len(chunks))


# ── Strategy 1: GLiNER zero-shot NER ─────────────────────────────────


GLINER_ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "economic concept",
    "commodity",
    "institution",
]


def extract_gliner_entities(conn, chunks):
    """Extract entities using GLiNER zero-shot NER.

    GLiNER allows custom entity types at inference time without fine-tuning.
    We use types relevant to economic texts.
    """
    if not HAS_GLINER:
        log.warning("GLiNER not available (pip install gliner)")
        return 0

    log.info("Loading GLiNER model...")
    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    entity_count = 0
    batch_size = 32

    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        batch_texts = list(batch_chunks)
        chunk_ids = list(range(batch_start + 1, batch_start + 1 + len(batch_texts)))

        predictions = model.batch_predict_entities(batch_texts, GLINER_ENTITY_TYPES, threshold=0.3)

        rows = []
        for chunk_id, preds in zip(chunk_ids, predictions, strict=False):
            for entity in preds:
                name = entity["text"].strip()
                if len(name) < 2:
                    continue
                rows.append((name, entity["label"], "gliner", chunk_id, entity["score"]))

        if rows:
            conn.executemany(
                "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            entity_count += len(rows)

        if (batch_start // batch_size) % 10 == 0:
            log.info(
                "  GLiNER: processed %d/%d chunks, %d entities so far",
                batch_start + len(batch_texts),
                len(chunks),
                entity_count,
            )

    conn.commit()
    log.info("GLiNER: extracted %d entities", entity_count)
    return entity_count


# ── Strategy 2: spaCy SVO triples ────────────────────────────────────


def extract_spacy_svo(conn, chunks):
    """Extract subject-verb-object triples using spaCy + textacy.

    Produces both entity records (subjects/objects) and relation records
    (the verb connecting them). Also extracts spaCy NER entities.
    """
    if not HAS_SPACY:
        log.warning("spaCy/textacy not available (pip install spacy textacy)")
        return 0, 0

    log.info("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                log.error("No spaCy English model found. Install one: python -m spacy download en_core_web_sm")
                return 0, 0

    entity_count = 0
    relation_count = 0

    for chunk_id, text in enumerate(chunks, 1):
        doc = nlp(text)

        # Extract NER entities
        ent_rows = []
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 2:
                continue
            ent_rows.append((name, ent.label_.lower(), "spacy_ner", chunk_id, 1.0))

        if ent_rows:
            conn.executemany(
                "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                ent_rows,
            )
            entity_count += len(ent_rows)

        # Extract SVO triples
        rel_rows = []
        for triple in subject_verb_object_triples(doc):
            subj = " ".join(t.text for t in triple.subject).strip()
            verb = " ".join(t.text for t in triple.verb).strip()
            obj = " ".join(t.text for t in triple.object).strip()
            if len(subj) < 2 or len(obj) < 2:
                continue
            rel_rows.append((subj, obj, verb, 1.0, chunk_id, "spacy_svo"))

        if rel_rows:
            conn.executemany(
                "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                rel_rows,
            )
            relation_count += len(rel_rows)

        if chunk_id % 100 == 0:
            log.info("  spaCy SVO: processed %d/%d chunks", chunk_id, len(chunks))

    conn.commit()
    log.info("spaCy SVO: %d entities, %d relations", entity_count, relation_count)
    return entity_count, relation_count


# ── Strategy 3: FTS5 concept discovery ────────────────────────────────


def extract_fts5_concepts(conn, chunks):
    """Zero-dependency entity extraction using FTS5 term frequencies + seed list.

    Uses the fts5vocab virtual table to find terms with high document frequency,
    then intersects with the seed entity list. Also discovers co-occurring seed
    terms within chunks to build relation edges.
    """
    log.info(
        "FTS5 concept discovery: scanning %d chunks against %d seed entities",
        len(chunks),
        len(ECONOMICS_SEED_ENTITIES),
    )

    # Create fts5vocab table for term statistics
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, row)")

    # Find seed terms that actually appear in the text
    found_terms = set()
    for term in ECONOMICS_SEED_ENTITIES:
        row = conn.execute("SELECT doc FROM chunks_vocab WHERE term = ?", (term,)).fetchone()
        if row and row[0] >= 2:  # Appears in at least 2 chunks
            found_terms.add(term)

    log.info("FTS5: %d/%d seed entities found in text", len(found_terms), len(ECONOMICS_SEED_ENTITIES))

    # For each found term, record which chunks contain it
    term_chunks = {}
    entity_count = 0

    for term in found_terms:
        # FTS5 MATCH query to find chunks containing this term
        # FTS5 content-sync tables expose rowid (= content_rowid = chunk_id), not column names
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank",
            (f'"{term}"',),
        ).fetchall()

        chunk_ids = [r[0] for r in rows]
        term_chunks[term] = set(chunk_ids)

        # Insert entity records
        for chunk_id in chunk_ids:
            conn.execute(
                "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                (term, "economic concept", "fts5", chunk_id, 1.0),
            )
            entity_count += 1

    # Build co-occurrence relations: terms appearing in the same chunk
    relation_count = 0
    terms_list = sorted(found_terms)
    cooccurrence = collections.Counter()

    for i, t1 in enumerate(terms_list):
        for t2 in terms_list[i + 1 :]:
            overlap = term_chunks[t1] & term_chunks[t2]
            if len(overlap) >= 2:  # Co-occur in at least 2 chunks
                cooccurrence[(t1, t2)] = len(overlap)

    for (t1, t2), count in cooccurrence.most_common():
        conn.execute(
            "INSERT INTO relations (src, dst, rel_type, weight, source) VALUES (?, ?, ?, ?, ?)",
            (t1, t2, "co_occurrence", float(count), "fts5"),
        )
        relation_count += 1

    conn.commit()
    log.info("FTS5: %d entity mentions, %d co-occurrence relations", entity_count, relation_count)
    return entity_count, relation_count


# ── Co-occurrence edges (cross-strategy) ──────────────────────────────


def build_cooccurrence_edges(conn):
    """Build co-occurrence edges: entities appearing in the same or adjacent chunks.

    Only creates edges for entity pairs with count >= 2. Works across all
    extraction strategies (the entities table is unified).
    """
    log.info("Building co-occurrence edges across all strategies...")

    # Get all (entity_name, chunk_id) pairs, deduplicated by name per chunk
    rows = conn.execute("""
        SELECT DISTINCT name, chunk_id FROM entities WHERE chunk_id IS NOT NULL
    """).fetchall()

    # Group by chunk_id
    chunk_entities = collections.defaultdict(set)
    for name, chunk_id in rows:
        normalized = normalize_entity_name(name)
        chunk_entities[chunk_id].add(normalized)

    # Count co-occurrences (same chunk or adjacent chunks)
    pair_count = collections.Counter()
    chunk_ids = sorted(chunk_entities.keys())

    for idx, cid in enumerate(chunk_ids):
        entities = chunk_entities[cid]
        # Same chunk
        entity_list = sorted(entities)
        for i, e1 in enumerate(entity_list):
            for e2 in entity_list[i + 1 :]:
                pair_count[(e1, e2)] += 1

        # Adjacent chunk
        if idx + 1 < len(chunk_ids) and chunk_ids[idx + 1] == cid + 1:
            next_entities = chunk_entities[cid + 1]
            for e1 in entities:
                for e2 in next_entities:
                    if e1 < e2:
                        pair_count[(e1, e2)] += 1
                    elif e2 < e1:
                        pair_count[(e2, e1)] += 1

    # Insert edges with count >= 2
    edge_count = 0
    for (e1, e2), count in pair_count.items():
        if count >= 2:
            conn.execute(
                "INSERT INTO relations (src, dst, rel_type, weight, source) VALUES (?, ?, ?, ?, ?)",
                (e1, e2, "co_occurrence_cross", float(count), "cooccurrence"),
            )
            edge_count += 1

    conn.commit()
    log.info("Cross-strategy co-occurrence: %d edges (from %d entity-chunk pairs)", edge_count, len(rows))
    return edge_count


# ── Embedding ─────────────────────────────────────────────────────────


def embed_chunks_and_entities(conn, chunks, model_name=DEFAULT_EMBEDDING_MODEL):
    """Embed chunks and entity names into their respective HNSW tables."""
    if not HAS_SENTENCE_TRANSFORMERS:
        log.warning("sentence-transformers not available — skipping embeddings")
        return

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Embed chunks
    log.info("Embedding %d chunks...", len(chunks))
    t0 = time.time()
    chunk_embeddings = model.encode(chunks, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    for chunk_id, emb in enumerate(chunk_embeddings, 1):
        conn.execute(
            "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
            (chunk_id, pack_vector(emb)),
        )
    conn.commit()
    log.info("Embedded %d chunks into chunks_vec (%.1fs)", len(chunks), time.time() - t0)

    # Embed unique entity names
    entity_names = [r[0] for r in conn.execute("SELECT DISTINCT name FROM entities").fetchall()]
    if not entity_names:
        log.warning("No entities to embed")
        return

    log.info("Embedding %d unique entity names...", len(entity_names))
    t0 = time.time()
    entity_embeddings = model.encode(entity_names, batch_size=256, normalize_embeddings=True, show_progress_bar=False)

    for idx, (_name, emb) in enumerate(zip(entity_names, entity_embeddings, strict=True), 1):
        conn.execute(
            "INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)",
            (idx, pack_vector(emb)),
        )
    conn.commit()

    # Store entity name -> rowid mapping in a helper table for coalescing
    conn.execute("CREATE TABLE IF NOT EXISTS entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO entity_vec_map (rowid, name) VALUES (?, ?)",
        [(idx, name) for idx, name in enumerate(entity_names, 1)],
    )
    conn.commit()
    log.info("Embedded %d entity names into entities_vec (%.1fs)", len(entity_names), time.time() - t0)


# ── Main pipeline ─────────────────────────────────────────────────────


def run_extraction(book_id, text_file=None, strategies=None, force=False, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Run the full extraction pipeline for a single book."""
    # Resolve text file
    if text_file:
        text_path = Path(text_file)
    else:
        text_path = TEXTS_DIR / f"gutenberg_{book_id}.txt"

    if not text_path.exists():
        log.error("Text file not found: %s", text_path)
        log.error("Run: python scripts/kg_gutenberg.py --download %d", book_id)
        sys.exit(1)

    # Output DB
    KG_DIR.mkdir(parents=True, exist_ok=True)
    db_path = KG_DIR / f"{book_id}.db"

    if db_path.exists() and not force:
        log.info("KG database already exists: %s (use --force to overwrite)", db_path)
        return db_path

    if db_path.exists() and force:
        db_path.unlink()
        log.info("Removed existing DB: %s", db_path)

    # Read and chunk text
    text = text_path.read_text(encoding="utf-8")
    log.info("Read %d chars from %s", len(text), text_path)

    chunks = chunk_fixed_tokens(text, window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP)
    log.info("Created %d chunks (window=%d, overlap=%d)", len(chunks), CHUNK_WINDOW, CHUNK_OVERLAP)

    # Create DB with extension loaded
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)
    log.info("Loaded muninn extension from %s", MUNINN_PATH)

    # Create schema and insert chunks
    create_schema(conn)
    insert_chunks(conn, chunks)

    # Determine which strategies to run
    if strategies is None:
        available = ["fts5"]
        if HAS_GLINER:
            available.append("gliner")
        if HAS_SPACY:
            available.append("spacy_svo")
        strategies = available
    else:
        strategies = [s.strip() for s in strategies.split(",")]

    log.info("Running strategies: %s", strategies)

    # Run strategies
    total_entities = 0
    total_relations = 0

    for strategy in strategies:
        if strategy == "gliner":
            n = extract_gliner_entities(conn, chunks)
            total_entities += n
        elif strategy == "spacy_svo":
            ne, nr = extract_spacy_svo(conn, chunks)
            total_entities += ne
            total_relations += nr
        elif strategy == "fts5":
            ne, nr = extract_fts5_concepts(conn, chunks)
            total_entities += ne
            total_relations += nr
        else:
            log.warning("Unknown strategy: %s", strategy)

    # Build cross-strategy co-occurrence edges
    nr = build_cooccurrence_edges(conn)
    total_relations += nr

    # Embed chunks and entities
    embed_chunks_and_entities(conn, chunks, embedding_model)

    # Write metadata
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('book_id', ?)", (str(book_id),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('text_file', ?)", (str(text_path),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('strategies', ?)", (",".join(strategies),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_entities', ?)", (str(total_entities),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_relations', ?)", (str(total_relations),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('num_chunks', ?)", (str(len(chunks)),))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)", (embedding_model,))
    conn.commit()

    # Summary
    final_entities = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
    final_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
    # HNSW virtual tables don't support count(*); use entity_vec_map as proxy
    entities_vec_count = conn.execute("SELECT count(*) FROM entity_vec_map").fetchone()[0]

    log.info("=" * 60)
    log.info("Extraction complete for book #%d", book_id)
    log.info("  Chunks:       %d (embedded)", len(chunks))
    log.info("  Entities:     %d", final_entities)
    log.info("  Relations:    %d", final_relations)
    log.info("  Entity vectors: %d", entities_vec_count)
    log.info("  Strategies:   %s", ", ".join(strategies))
    log.info("  Output:       %s", db_path)
    log.info("=" * 60)

    conn.close()
    return db_path


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="KG Entity/Relation Extraction Pipeline")
    parser.add_argument("--book-id", type=int, required=True, help="Gutenberg book ID")
    parser.add_argument("--text-file", type=str, help="Override text file path")
    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated: gliner,spacy_svo,fts5 (default: all available)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing DB")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    run_extraction(
        book_id=args.book_id,
        text_file=args.text_file,
        strategies=args.strategies,
        force=args.force,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
