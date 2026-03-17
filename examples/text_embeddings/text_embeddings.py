"""
Text Embeddings — Semantic Search with muninn

Zero-dependency end-to-end example: load GGUF embedding models, embed
documents, build HNSW indices, and perform semantic similarity search —
all inside a single SQLite extension.

Demonstrates two models side-by-side:
  - all-MiniLM-L6-v2 (22M params, 384-dim, 23 MB) — fast baseline
  - nomic-embed-text-v1.5 (137M params, 768-dim, 146 MB) — higher quality

Only requires:
  - muninn extension (make all)
  - GGUF model files (auto-downloaded on first run)
"""

import logging
import sqlite3
import struct
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_IN_COLAB = "google.colab" in sys.modules

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    if _IN_COLAB:
        _REPO = Path("/content/sqlite-muninn")
        if not _REPO.exists():
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/neozenith/sqlite-muninn.git", str(_REPO)],
                check=True,
            )
        if not list((_REPO / "build").glob("muninn.*")):
            subprocess.run(["apt-get", "install", "-y", "libsqlite3-dev"], check=True)
            subprocess.run(["make", "all"], cwd=str(_REPO), check=True)
        PROJECT_ROOT = _REPO
    else:
        PROJECT_ROOT = Path.cwd().parent.parent  # local notebook CWD is examples/{name}/
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")
MODELS_DIR = PROJECT_ROOT / "models"


# ── Model Definitions ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Configuration for a GGUF embedding model."""

    name: str  # Registry name used in SQL
    filename: str  # GGUF file in models/
    url: str  # Download URL
    doc_prefix: str  # Prefix prepended to documents before embedding
    query_prefix: str  # Prefix prepended to queries before embedding


MODELS = [
    ModelConfig(
        name="MiniLM",
        filename="all-MiniLM-L6-v2.Q8_0.gguf",
        url="https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
        doc_prefix="",
        query_prefix="",
    ),
    ModelConfig(
        name="NomicEmbed",
        filename="nomic-embed-text-v1.5.Q8_0.gguf",
        url="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    # Qwen3-Embedding-8B disabled — 4.7 GB download, slow to load and embed
    # ModelConfig(
    #     name="Qwen3Embed8B",
    #     filename="Qwen3-Embedding-8B-Q4_K_M.gguf",
    #     url="https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q4_K_M.gguf",
    #     doc_prefix="",
    #     query_prefix="",
    # ),
]


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


def ensure_model(model: ModelConfig) -> bool:
    """Ensure a GGUF model file is available, downloading if needed."""
    path = MODELS_DIR / model.filename
    if path.exists():
        log.info("Model %s found: %s (%.1f MB)", model.name, path, path.stat().st_size / 1e6)
        return True

    log.info("Model %s not found — downloading %s...", model.name, model.filename)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _download_with_progress(model.url, path)
    except Exception:
        log.exception("Failed to download %s", model.filename)
        if path.exists():
            path.unlink()
        return False

    log.info("Downloaded %s (%.1f MB)", model.filename, path.stat().st_size / 1e6)
    return True


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-example/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024

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
            print()


def blob_to_floats(blob: bytes) -> list[float]:
    """Convert a float32 blob to a list of Python floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── Section 1: Model Loading & Inspection ────────────────────────────
def section_model_loading(db: sqlite3.Connection, models: list[ModelConfig]) -> dict[str, int]:
    """Load GGUF models and inspect their properties."""
    print("\n" + "=" * 60)
    print("Section 1: Model Loading & Inspection")
    print("=" * 60)

    dims: dict[str, int] = {}
    for model in models:
        path = MODELS_DIR / model.filename
        db.execute(
            "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
            (model.name, str(path)),
        )
        (dim,) = db.execute("SELECT muninn_model_dim(?)", (model.name,)).fetchone()
        dims[model.name] = dim
        print(f"\n  {model.name}: dim={dim}, file={model.filename}")
        if model.doc_prefix:
            print(f"    doc prefix: '{model.doc_prefix}'")
            print(f"    query prefix: '{model.query_prefix}'")

    # List all loaded models
    rows = db.execute("SELECT name, dim FROM muninn_models").fetchall()
    print(f"\n  All loaded models: {rows}")

    return dims


# ── Section 2: Embed Documents & Build HNSW Indices ─────────────────
def section_embed_and_index(db: sqlite3.Connection, models: list[ModelConfig], dims: dict[str, int]) -> None:
    """Embed all documents into per-model HNSW indices."""
    print("\n" + "=" * 60)
    print("Section 2: Embed Documents & Build HNSW Indices")
    print("=" * 60)

    for model in models:
        dim = dims[model.name]
        table = f"vectors_{model.name}"

        db.execute(f"CREATE VIRTUAL TABLE [{table}] USING hnsw_index(dimensions={dim}, metric=cosine)")
        print(f"\n  {model.name}: created HNSW index (dim={dim})")

        # Embed with doc_prefix prepended to each document
        if model.doc_prefix:
            db.execute(
                f"INSERT INTO [{table}](rowid, vector) SELECT id, muninn_embed(?, ? || content) FROM documents",
                (model.name, model.doc_prefix),
            )
        else:
            db.execute(
                f"INSERT INTO [{table}](rowid, vector) SELECT id, muninn_embed(?, content) FROM documents",
                (model.name,),
            )
        print(f"  {model.name}: embedded and indexed {len(DOCUMENTS)} documents")

        # Verify one vector
        row = db.execute(f"SELECT vector FROM [{table}] WHERE rowid = 1").fetchone()
        assert row is not None
        assert len(row[0]) == dim * 4
        print(f"  {model.name}: verified vector size = {len(row[0])} bytes ({dim} floats)")


# ── Section 3: Comparative Semantic Search ───────────────────────────
def section_semantic_search(db: sqlite3.Connection, models: list[ModelConfig]) -> None:
    """Perform KNN search with each model and compare rankings."""
    print("\n" + "=" * 60)
    print("Section 3: Comparative Semantic Search")
    print("=" * 60)

    for query_text in QUERIES:
        print(f'\n  Query: "{query_text}"')
        print(f"  {'─' * 54}")

        for model in models:
            table = f"vectors_{model.name}"
            prefixed_query = model.query_prefix + query_text

            query_blob = db.execute("SELECT muninn_embed(?, ?)", (model.name, prefixed_query)).fetchone()[0]

            results = db.execute(
                f"""
                SELECT v.rowid, v.distance, d.content
                FROM [{table}] v
                JOIN documents d ON d.id = v.rowid
                WHERE v.vector MATCH ? AND k = 3
                """,
                (query_blob,),
            ).fetchall()

            print(f"\n    {model.name} (dim={len(query_blob) // 4}):")
            for rowid, distance, content in results:
                print(f"      #{rowid:<2d}  dist={distance:.4f}  {content}")


# ── Section 4: Auto-Embed Trigger ───────────────────────────────────
def section_auto_embed_trigger(db: sqlite3.Connection, model: ModelConfig) -> None:
    """Demonstrate automatic embedding on INSERT via a SQLite trigger."""
    print("\n" + "=" * 60)
    print(f"Section 4: Auto-Embed Trigger (using {model.name})")
    print("=" * 60)

    table = f"vectors_{model.name}"
    prefix_expr = f"'{model.doc_prefix}' || " if model.doc_prefix else ""

    db.execute(
        f"""
        CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
        BEGIN
          INSERT INTO [{table}](rowid, vector)
            VALUES (NEW.id, muninn_embed('{model.name}', {prefix_expr}NEW.content));
        END
        """
    )
    print("\n  Created TEMP trigger for auto-embedding on INSERT.")

    db.execute("INSERT INTO documents(id, content) VALUES (100, 'Black holes warp spacetime near the event horizon')")
    print("  Inserted doc #100: 'Black holes warp spacetime near the event horizon'")

    prefixed_query = model.query_prefix + "phenomena in space"
    query_blob = db.execute("SELECT muninn_embed(?, ?)", (model.name, prefixed_query)).fetchone()[0]

    results = db.execute(
        f"""
        SELECT v.rowid, v.distance, d.content
        FROM [{table}] v
        JOIN documents d ON d.id = v.rowid
        WHERE v.vector MATCH ? AND k = 3
        """,
        (query_blob,),
    ).fetchall()

    result_ids = {r[0] for r in results}
    print('\n  Search for "phenomena in space" top-3:')
    for rowid, distance, content in results:
        print(f"    #{rowid:<3d}  dist={distance:.4f}  {content}")

    assert 100 in result_ids, "Trigger-inserted document should appear in space-related search"
    print("\n  Trigger-inserted document found in results.")

    # Clean up
    db.execute("DROP TRIGGER auto_embed")
    db.execute("DELETE FROM documents WHERE id = 100")
    db.execute(f"DELETE FROM [{table}] WHERE rowid = 100")


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== muninn Text Embeddings Example ===")
    print(f"\n  Project root:  {PROJECT_ROOT}")
    print(f"  Extension:     {EXTENSION_PATH}")

    # Ensure all models are available (download if needed)
    available_models = [m for m in MODELS if ensure_model(m)]
    if not available_models:
        log.error("No models available. Cannot proceed.")
        return

    print(f"\n  Models ready: {[m.name for m in available_models]}")

    # Connect and load muninn — that's all we need
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    print("  Loaded muninn extension (HNSW + GGUF embedding).\n")

    # Create documents table
    db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    db.executemany("INSERT INTO documents(id, content) VALUES (?, ?)", DOCUMENTS)
    print(f"  Created documents table with {len(DOCUMENTS)} rows.")

    # Run all sections
    dims = section_model_loading(db, available_models)
    section_embed_and_index(db, available_models, dims)
    section_semantic_search(db, available_models)
    section_auto_embed_trigger(db, available_models[0])

    # Done
    db.close()
    print("\n" + "=" * 60)
    print("Done. All sections completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
