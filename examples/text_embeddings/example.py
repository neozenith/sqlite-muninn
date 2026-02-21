"""
Text Embeddings — Semantic Search with Real Embedding Models

Demonstrates: sqlite-lembed (GGUF local models) and sqlite-rembed (OpenAI API)
paired with muninn's HNSW index for end-to-end text-in, semantic-search-out.

Sections run conditionally based on available dependencies:
  - lembed:  requires `pip install sqlite-lembed` + a GGUF model file
  - rembed:  requires `pip install sqlite-rembed` + OPENAI_API_KEY env var
"""

import logging
import os
import sqlite3
import urllib.request
from pathlib import Path

# Optional dependencies — top-level try/except per project convention
try:
    import sqlite_lembed

    HAS_LEMBED = True
except ImportError:
    HAS_LEMBED = False

try:
    import sqlite_rembed

    HAS_REMBED = True
except ImportError:
    HAS_REMBED = False

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "muninn")

# GGUF model config — set GGUF_MODEL_PATH to use a custom model file
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2.Q8_0.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf"
)

_env_model_path = os.environ.get("GGUF_MODEL_PATH", "")
GGUF_MODEL_PATH = Path(_env_model_path) if _env_model_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME
GGUF_IS_CUSTOM = bool(_env_model_path)

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536

# MiniLM GGUF model produces 384-dimensional embeddings
LEMBED_DIM = 384

# ── Sample documents ────────────────────────────────────────────────
DOCUMENTS = [
    (1, "The quick brown fox jumps over the lazy dog in the forest"),
    (2, "A neural network learns patterns from large datasets"),
    (3, "Fresh pasta with tomato sauce is a classic Italian dish"),
    (4, "The Mars rover collected soil samples from the crater"),
    (5, "SQLite is the most widely deployed database engine in the world"),
    (6, "Wolves and bears roam the dense woodland trails"),
    (7, "Gradient descent optimizes the loss function during training"),
    (8, "Stars and galaxies fill the observable universe"),
]

QUERIES = [
    "animals in the wild",
    "machine learning and artificial intelligence",
    "outer space exploration",
]


def ensure_gguf_model() -> bool:
    """Ensure the GGUF model file is available, downloading if needed.

    Returns True if the model is ready, False if it could not be obtained.
    """
    if GGUF_MODEL_PATH.exists():
        log.info("GGUF model found: %s", GGUF_MODEL_PATH)
        return True

    if GGUF_IS_CUSTOM:
        # User specified a custom path that doesn't exist — don't auto-download
        log.warning("GGUF_MODEL_PATH is set but file not found: %s", GGUF_MODEL_PATH)
        return False

    # Default path: create directory and download
    log.info("GGUF model not found at %s — downloading...", GGUF_MODEL_PATH)
    GGUF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download_with_progress(DEFAULT_MODEL_URL, GGUF_MODEL_PATH)
    except Exception:
        log.exception("Failed to download GGUF model from %s", DEFAULT_MODEL_URL)
        # Clean up partial download
        if GGUF_MODEL_PATH.exists():
            GGUF_MODEL_PATH.unlink()
        return False

    log.info("Downloaded model to %s (%.1f MB)", GGUF_MODEL_PATH, GGUF_MODEL_PATH.stat().st_size / 1e6)
    return True


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-example/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024  # 256 KB chunks

        with dest.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / 1e6
                    total_mb = total / 1e6
                    print(f"\r  Downloading {dest.name}: {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

        if total > 0:
            print()  # newline after progress


def print_search_results(db: sqlite3.Connection, index_table: str, query_blob: bytes, query_text: str) -> None:
    """Run KNN search and print results joined with document text."""
    print(f"\n  Query: \"{query_text}\"")
    results = db.execute(
        f"""
        SELECT v.rowid, v.distance, d.content
        FROM {index_table} v
        JOIN documents d ON d.id = v.rowid
        WHERE v.vector MATCH ? AND k = 3
        """,
        (query_blob,),
    ).fetchall()

    for rowid, distance, content in results:
        print(f"    #{rowid:<2d}  dist={distance:.4f}  {content}")


# ── Section: sqlite-lembed (local GGUF model) ──────────────────────
def run_lembed_example(db: sqlite3.Connection) -> None:
    """Embed documents using a local GGUF model via sqlite-lembed."""
    print("\n" + "=" * 60)
    print("Section: sqlite-lembed (local GGUF embedding)")
    print("=" * 60)

    sqlite_lembed.load(db)
    print("\n  Loaded sqlite-lembed extension.")

    # Register the GGUF model
    db.execute(
        "INSERT INTO temp.lembed_models(name, model) SELECT 'MiniLM', lembed_model_from_file(?)",
        (str(GGUF_MODEL_PATH),),
    )
    print(f"  Registered model: {GGUF_MODEL_PATH.name}")

    # Create HNSW index
    db.execute(
        f"""
        CREATE VIRTUAL TABLE lembed_vectors USING hnsw_index(
            dimensions={LEMBED_DIM}, metric='cosine'
        )
        """
    )
    print(f"  Created HNSW index: dim={LEMBED_DIM}, metric=cosine")

    # Embed and insert all documents
    db.execute(
        """
        INSERT INTO lembed_vectors(rowid, vector)
          SELECT id, lembed('MiniLM', content) FROM documents
        """
    )
    print(f"  Embedded and indexed {len(DOCUMENTS)} documents.")

    # Verify blob dimensions
    row = db.execute("SELECT vector FROM lembed_vectors WHERE rowid = 1").fetchone()
    assert row is not None, "Point lookup failed"
    assert len(row[0]) == LEMBED_DIM * 4, f"Expected {LEMBED_DIM * 4} bytes, got {len(row[0])}"

    # Semantic search
    print("\n  --- Semantic Search (lembed) ---")
    for query_text in QUERIES:
        query_blob = db.execute("SELECT lembed('MiniLM', ?)", (query_text,)).fetchone()[0]
        print_search_results(db, "lembed_vectors", query_blob, query_text)

    # Demonstrate auto-embed trigger
    print("\n  --- Auto-Embed Trigger ---")
    db.execute(
        """
        CREATE TEMP TRIGGER lembed_auto_embed AFTER INSERT ON documents
        BEGIN
          INSERT INTO lembed_vectors(rowid, vector)
            VALUES (NEW.id, lembed('MiniLM', NEW.content));
        END
        """
    )
    print("  Created TEMP trigger for auto-embedding.")

    db.execute("INSERT INTO documents(id, content) VALUES (100, 'Black holes warp spacetime near the event horizon')")
    print("  Inserted new document via trigger: 'Black holes warp spacetime near the event horizon'")

    # Search should now find the new document for space queries
    query_blob = db.execute("SELECT lembed('MiniLM', 'phenomena in space')", ()).fetchone()[0]
    results = db.execute(
        """
        SELECT v.rowid, v.distance, d.content
        FROM lembed_vectors v
        JOIN documents d ON d.id = v.rowid
        WHERE v.vector MATCH ? AND k = 3
        """,
        (query_blob,),
    ).fetchall()

    result_ids = {r[0] for r in results}
    print(f"\n  Search for \"phenomena in space\" top-3 IDs: {sorted(result_ids)}")
    assert 100 in result_ids, "Trigger-inserted document should appear in space-related search"
    print("  Trigger-inserted document found in results.")

    # Clean up trigger
    db.execute("DROP TRIGGER lembed_auto_embed")
    db.execute("DELETE FROM documents WHERE id = 100")
    db.execute("DELETE FROM lembed_vectors WHERE rowid = 100")

    print("\n  lembed section complete.")


# ── Section: sqlite-rembed (OpenAI API) ─────────────────────────────
def run_rembed_example(db: sqlite3.Connection) -> None:
    """Embed documents using OpenAI's API via sqlite-rembed."""
    print("\n" + "=" * 60)
    print("Section: sqlite-rembed (OpenAI API embedding)")
    print("=" * 60)

    sqlite_rembed.load(db)
    print("\n  Loaded sqlite-rembed extension.")

    # Register OpenAI client (reads OPENAI_API_KEY from environment)
    db.execute("INSERT INTO temp.rembed_clients(name, options) VALUES (?, 'openai')", (OPENAI_MODEL,))
    print(f"  Registered client: {OPENAI_MODEL}")

    # Create HNSW index for OpenAI dimensions
    db.execute(
        f"""
        CREATE VIRTUAL TABLE rembed_vectors USING hnsw_index(
            dimensions={OPENAI_DIM}, metric='cosine'
        )
        """
    )
    print(f"  Created HNSW index: dim={OPENAI_DIM}, metric=cosine")

    # Embed and insert all documents (one API call per row)
    print(f"  Embedding {len(DOCUMENTS)} documents via OpenAI API...")
    for doc_id, content in DOCUMENTS:
        embedding = db.execute("SELECT rembed(?, ?)", (OPENAI_MODEL, content)).fetchone()[0]
        db.execute("INSERT INTO rembed_vectors(rowid, vector) VALUES (?, ?)", (doc_id, embedding))
    print(f"  Indexed {len(DOCUMENTS)} documents.")

    # Verify blob dimensions
    row = db.execute("SELECT vector FROM rembed_vectors WHERE rowid = 1").fetchone()
    assert row is not None, "Point lookup failed"
    assert len(row[0]) == OPENAI_DIM * 4, f"Expected {OPENAI_DIM * 4} bytes, got {len(row[0])}"

    # Semantic search
    print("\n  --- Semantic Search (rembed / OpenAI) ---")
    for query_text in QUERIES:
        query_blob = db.execute("SELECT rembed(?, ?)", (OPENAI_MODEL, query_text)).fetchone()[0]
        print_search_results(db, "rembed_vectors", query_blob, query_text)

    print("\n  rembed section complete.")


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Text Embeddings Example ===")
    print(f"\n  Project root:  {PROJECT_ROOT}")
    print(f"  Extension:     {EXTENSION_PATH}")

    # ── Dependency checks ───────────────────────────────────────────
    can_lembed = True
    can_rembed = True

    if not HAS_LEMBED:
        log.warning("sqlite-lembed not installed. Install with: pip install sqlite-lembed")
        can_lembed = False
    elif not ensure_gguf_model():
        can_lembed = False

    if not HAS_REMBED:
        log.warning("sqlite-rembed not installed. Install with: pip install sqlite-rembed")
        can_rembed = False
    elif not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY is not set or empty. Skipping rembed examples.")
        can_rembed = False

    if not can_lembed and not can_rembed:
        log.warning(
            "Neither sqlite-lembed nor sqlite-rembed is available. "
            "Install at least one to run this example:\n"
            "  pip install sqlite-lembed   # for local GGUF models\n"
            "  pip install sqlite-rembed   # for OpenAI/Ollama API"
        )
        return

    # ── Setup database ──────────────────────────────────────────────
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    print("  Loaded muninn extension.\n")

    db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    db.executemany("INSERT INTO documents(id, content) VALUES (?, ?)", DOCUMENTS)
    print(f"  Created documents table with {len(DOCUMENTS)} rows.")

    # ── Run available sections ──────────────────────────────────────
    sections_run = 0

    if can_lembed:
        run_lembed_example(db)
        sections_run += 1

    if can_rembed:
        run_rembed_example(db)
        sections_run += 1

    # ── Summary ─────────────────────────────────────────────────────
    db.close()
    print("\n" + "=" * 60)
    print(f"Done. Ran {sections_run} section(s).")
    if not can_lembed:
        print("  Skipped: lembed (missing dependency or model)")
    if not can_rembed:
        print("  Skipped: rembed (missing dependency or OPENAI_API_KEY)")
    print("=" * 60)


if __name__ == "__main__":
    main()
