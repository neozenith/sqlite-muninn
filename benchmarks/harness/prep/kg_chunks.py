"""Prep module: create KG chunk databases from Gutenberg texts.

Creates KG_DIR/{book_id}_chunks.db with a text_chunks table.
This is the common starting point for all KG pipeline treatments.
"""

import logging
import re
import sqlite3

from benchmarks.harness.common import KG_DIR, TEXTS_DIR
from benchmarks.harness.prep.common import fmt_size

log = logging.getLogger(__name__)


def _chunk_text(text, window=256, overlap=50):
    """Split text into fixed-size token windows with overlap."""
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


def create_chunks_db(book_id, window=256, overlap=50):
    """Create a chunks database from a Gutenberg text file.

    Args:
        book_id: Gutenberg book ID (text must already be downloaded).
        window: Chunk window size in tokens.
        overlap: Token overlap between chunks.

    Returns:
        Path to the created chunks database.
    """
    text_path = TEXTS_DIR / f"gutenberg_{book_id}.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Text not found: {text_path}. Run 'prep texts --book-id {book_id}' first.")

    KG_DIR.mkdir(parents=True, exist_ok=True)
    db_path = KG_DIR / f"{book_id}_chunks.db"

    text = text_path.read_text(encoding="utf-8")
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = _chunk_text(text, window=window, overlap=overlap)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            token_count INTEGER
        )
    """)
    conn.execute("DELETE FROM text_chunks")  # idempotent

    for i, chunk in enumerate(chunks):
        token_count = len(chunk.split())
        conn.execute("INSERT INTO text_chunks(id, text, token_count) VALUES (?, ?, ?)", (i, chunk, token_count))

    conn.commit()
    conn.close()

    log.info("  Book #%d: %d chunks -> %s", book_id, len(chunks), db_path)
    return db_path


def print_status():
    """Print status of KG chunk databases."""
    print("=== KG Chunk Database Status ===\n")
    print(f"  {'BOOK_ID':>7s}   {'FILE':<20s}   {'SIZE':>8s}   {'CHUNKS':>8s}   {'STATUS'}")
    print(f"  {'-' * 7}   {'-' * 20}   {'-' * 8}   {'-' * 8}   {'-' * 6}")

    default_ids = [3300]

    for book_id in default_ids:
        db_path = KG_DIR / f"{book_id}_chunks.db"
        if db_path.exists():
            size = db_path.stat().st_size
            conn = sqlite3.connect(str(db_path))
            try:
                count = conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
            except Exception:
                count = 0
            conn.close()
            print(f"  {book_id:>7d}   {db_path.name:<20s}   {fmt_size(size):>8s}   {count:>8,d}   READY")
        else:
            expected_name = f"{book_id}_chunks.db"
            print(f"  {book_id:>7d}   {expected_name:<20s}   {'':>8s}   {'':>8s}   MISSING")

    # Show any extra chunk DBs
    if KG_DIR.exists():
        for db_path in sorted(KG_DIR.glob("*_chunks.db")):
            stem_id = db_path.stem.split("_")[0]
            if stem_id.isdigit() and int(stem_id) not in default_ids:
                book_id = int(stem_id)
                size = db_path.stat().st_size
                conn = sqlite3.connect(str(db_path))
                try:
                    count = conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
                except Exception:
                    count = 0
                conn.close()
                print(f"  {book_id:>7d}   {db_path.name:<20s}   {fmt_size(size):>8s}   {count:>8,d}   READY (extra)")

    print()


def prep_kg_chunks(book_id=None, status_only=False, force=False):
    """Create chunk databases for KG pipeline benchmarks.

    Args:
        book_id: Specific book ID. If None, uses default (3300).
        status_only: If True, show status and return.
        force: If True, re-create databases even if they exist.
    """
    if status_only:
        print_status()
        return

    book_ids = [book_id] if book_id else [3300]

    log.info("Creating KG chunk databases...")
    for bid in book_ids:
        db_path = KG_DIR / f"{bid}_chunks.db"
        if db_path.exists() and not force:
            log.info("  Book #%d: chunk DB already exists at %s (use --force to re-create)", bid, db_path)
            continue
        if db_path.exists() and force:
            log.info("  Book #%d: --force, re-creating chunk DB", bid)
            db_path.unlink()
        create_chunks_db(bid)

    log.info("KG chunk prep complete. Databases in %s", KG_DIR)
