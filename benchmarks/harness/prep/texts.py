"""Prep module: fetch Gutenberg texts.

Ports from benchmarks/scripts/kg_gutenberg.py with enhanced discovery features:
- Gutendex catalog search with local caching
- Random book selection from a topic category
- Cached text listing with metadata
"""

import json
import logging
import random
import time
import urllib.request

from benchmarks.harness.common import TEXTS_DIR
from benchmarks.harness.prep.common import fmt_size

log = logging.getLogger(__name__)

# Default books for KG pipeline benchmarks
DEFAULT_BOOK_IDS = [3300]  # Wealth of Nations

# Gutendex API for catalog searches
GUTENDEX_BASE_URL = "https://gutendex.com/books"
CATALOG_CACHE_TTL = 86400  # 24 hours


# ── Gutendex catalog ─────────────────────────────────────────────


def _catalog_cache_path():
    """Return path to the Gutendex catalog cache file."""
    return TEXTS_DIR / "gutendex_catalog.json"


def search_gutendex(query, topic=None):
    """Search the Gutendex catalog API, returning a list of book dicts.

    Results are cached for 24h in TEXTS_DIR/gutendex_catalog.json
    keyed by (query, topic) to avoid repeated API calls.
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _catalog_cache_path()

    # Load existing catalog cache
    catalog = {}
    if cache_path.exists():
        try:
            catalog = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            catalog = {}

    cache_key = f"{query}|{topic or ''}"
    cached = catalog.get(cache_key)
    if cached and time.time() - cached.get("timestamp", 0) < CATALOG_CACHE_TTL:
        log.info("Using cached catalog results for %r (topic=%s)", query, topic)
        return cached["results"]

    # Build API URL
    params = f"search={urllib.request.quote(query)}"
    if topic:
        params += f"&topic={urllib.request.quote(topic)}"
    url = f"{GUTENDEX_BASE_URL}?{params}"

    log.info("Searching Gutendex: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = data.get("results", [])

    # Cache results
    catalog[cache_key] = {"timestamp": time.time(), "results": results}
    cache_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    log.info("Found %d results, cached to %s", len(results), cache_path)

    return results


def pick_random_book(topic="economics", exclude_ids=None):
    """Search catalog for a topic, filter for English + plain-text, pick random.

    Returns a Gutendex book dict, or None if no candidates found.
    """
    if exclude_ids is None:
        exclude_ids = set()

    results = search_gutendex(topic, topic=topic)

    candidates = []
    for book in results:
        book_id = book["id"]
        if book_id in exclude_ids:
            continue
        if "en" not in book.get("languages", []):
            continue
        formats = book.get("formats", {})
        has_text = any("text/plain" in fmt for fmt in formats)
        if not has_text:
            continue
        candidates.append(book)

    if not candidates:
        log.warning("No candidate books found for topic=%r (excluding %s)", topic, exclude_ids)
        return None

    chosen = random.choice(candidates)
    log.info("Randomly selected: [%d] %s", chosen["id"], chosen.get("title", "Unknown"))
    return chosen


def format_book_info(book):
    """Format a Gutendex book dict for display."""
    authors = ", ".join(a["name"] for a in book.get("authors", []))
    subjects = "; ".join(book.get("subjects", [])[:3])
    languages = ", ".join(book.get("languages", []))
    book_id = book["id"]
    title = book.get("title", "Unknown")
    return f"  [{book_id:>5d}] {title}\n         by {authors} | lang={languages}\n         {subjects}"


# ── Cached text management ───────────────────────────────────────


def list_cached_texts():
    """List all cached Gutenberg text files.

    Returns list of (book_id, path, size_bytes) tuples sorted by book_id.
    """
    if not TEXTS_DIR.exists():
        return []

    results = []
    for path in sorted(TEXTS_DIR.glob("gutenberg_*.txt")):
        stem = path.stem  # gutenberg_3300
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            book_id = int(parts[1])
            results.append((book_id, path, path.stat().st_size))
    return results


def get_cached_book_ids():
    """Return set of Gutenberg IDs for which we already have cached text."""
    return {book_id for book_id, _, _ in list_cached_texts()}


# ── Text download ────────────────────────────────────────────────


def download_gutenberg_text(book_id, force=False):
    """Download a single Gutenberg text, strip boilerplate, cache locally.

    Returns the path to the cached text file.
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEXTS_DIR / f"gutenberg_{book_id}.txt"

    if cache_path.exists() and not force:
        log.info("  Book #%d: cached at %s", book_id, cache_path)
        return cache_path

    if cache_path.exists() and force:
        log.info("  Book #%d: --force, re-downloading", book_id)
        cache_path.unlink()

    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    log.info("  Downloading book #%d from %s...", book_id, url)

    req = urllib.request.Request(url, headers={"User-Agent": "muninn-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_text = resp.read().decode("utf-8-sig")

    # Strip Gutenberg boilerplate
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    start_idx = 0
    for marker in start_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            start_idx = raw_text.index("\n", idx) + 1
            break

    end_idx = len(raw_text)
    for marker in end_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    clean_text = raw_text[start_idx:end_idx].strip()
    cache_path.write_text(clean_text, encoding="utf-8")
    log.info("  Book #%d: saved %d chars to %s", book_id, len(clean_text), cache_path)
    return cache_path


# ── Status display ───────────────────────────────────────────────


def print_status():
    """Print status of expected and cached text files."""
    print("=== Text Cache Status ===\n")
    print(f"  {'BOOK_ID':>7s}   {'FILE':<28s}   {'SIZE':>8s}   {'STATUS'}")
    print(f"  {'-' * 7}   {'-' * 28}   {'-' * 8}   {'-' * 6}")

    cached = {book_id: (path, size) for book_id, path, size in list_cached_texts()}

    # Show expected books
    for book_id in DEFAULT_BOOK_IDS:
        if book_id in cached:
            path, size = cached[book_id]
            print(f"  {book_id:>7d}   {path.name:<28s}   {fmt_size(size):>8s}   CACHED")
        else:
            expected_name = f"gutenberg_{book_id}.txt"
            print(f"  {book_id:>7d}   {expected_name:<28s}   {'':>8s}   MISSING")

    # Show any extra cached books
    extra_ids = sorted(set(cached.keys()) - set(DEFAULT_BOOK_IDS))
    for book_id in extra_ids:
        path, size = cached[book_id]
        print(f"  {book_id:>7d}   {path.name:<28s}   {fmt_size(size):>8s}   CACHED (extra)")

    print()


def print_cached_list():
    """Print detailed list of all cached texts."""
    cached = list_cached_texts()
    if not cached:
        print("No cached texts found.")
        print(f"Text directory: {TEXTS_DIR}")
        return

    print(f"=== Cached Gutenberg Texts ({len(cached)} files) ===\n")
    print(f"  {'BOOK_ID':>7s}   {'FILE':<28s}   {'SIZE':>8s}   {'WORDS':>8s}")
    print(f"  {'-' * 7}   {'-' * 28}   {'-' * 8}   {'-' * 8}")

    for book_id, path, size in cached:
        word_count = len(path.read_text(encoding="utf-8").split())
        print(f"  {book_id:>7d}   {path.name:<28s}   {fmt_size(size):>8s}   {word_count:>8,d}")

    print(f"\n  Directory: {TEXTS_DIR}")
    print()


# ── Main entry point ─────────────────────────────────────────────


def prep_texts(
    book_id=None, random_book=False, category="economics", list_cached=False, status_only=False, force=False
):
    """Download Gutenberg texts for KG pipeline benchmarks.

    Args:
        book_id: Specific book ID to download. If None, downloads DEFAULT_BOOK_IDS.
        random_book: If True, pick a random book from the Gutendex catalog.
        category: Gutenberg subject category for random selection (default: economics).
        list_cached: If True, list all cached texts and return.
        status_only: If True, show cache status and return.
        force: If True, re-download even if files exist.
    """
    if status_only:
        print_status()
        return

    if list_cached:
        print_cached_list()
        return

    if random_book:
        existing = get_cached_book_ids()
        book = pick_random_book(topic=category, exclude_ids=existing)
        if book is None:
            log.error("Could not find a random book for topic=%r", category)
            return
        download_gutenberg_text(book["id"], force=force)
        return

    if book_id:
        book_ids = [book_id]
    else:
        book_ids = list(DEFAULT_BOOK_IDS)

    log.info("Downloading %d Gutenberg text(s)...", len(book_ids))
    for bid in book_ids:
        download_gutenberg_text(bid, force=force)

    log.info("Text prep complete. Cached in %s", TEXTS_DIR)
