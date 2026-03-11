"""
LLM Extract — Structured NER & RE with muninn

Zero-dependency end-to-end example: load a GGUF chat model, extract named
entities, extract relations, and run summarisation — all via SQL functions
inside a single SQLite extension.

Demonstrates:
  1. Model loading via muninn_chat_model() + muninn_chat_models VT
  2. Plain chat completion via muninn_chat()
  3. NER via muninn_extract_entities() with calibrated confidence scores
  4. Relation extraction via muninn_extract_relations() chaining off NER
  5. Text summarisation via muninn_summarize()
  6. Combined NER+RE in one pass via muninn_extract_ner_re() — 2x throughput
  7. Two-pass SQL pipeline — NER->RE chained in SQL CTE (for comparison)
  8. Comparison of approaches 3+4 vs 6 vs 7 — timing and output diff
  9. Batch NER+RE via muninn_extract_ner_re_batch() — N docs in parallel
 10. Batch vs sequential timing comparison — speedup measurement

Default model: Qwen3-4B Q4_K_M (~2.5 GB, Apache 2.0, best sub-5B quality)

Only requires:
  - muninn extension (make all)
  - GGUF chat model file (auto-downloaded on first run)
"""

import argparse
import json
import logging
import random
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")
MODELS_DIR = PROJECT_ROOT / "models"


# ── Model Definitions ──────────────────────────────────────────────
@dataclass
class ChatModelConfig:
    """Configuration for a GGUF chat model."""

    name: str
    filename: str
    url: str
    size_gb: float


# Default: Qwen3-4B — best quality under 5B params, Apache 2.0 license
QWEN3_4B = ChatModelConfig(
    name="Qwen3-4B",
    filename="Qwen3-4B-Q4_K_M.gguf",
    url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
    size_gb=2.5,
)
# Alternatives (uncomment to use):
QWEN3_8B = ChatModelConfig(
    name="Qwen3-8B",
    filename="Qwen3-8B-Q4_K_M.gguf",
    url="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    size_gb=5.0,
)
GEMMA3_4B = ChatModelConfig(
    name="Gemma-3-4B",
    filename="google_gemma-3-4b-it-Q4_K_M.gguf",
    url="https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
    size_gb=2.5,
)
ENABLED_MODELS = [
    QWEN3_4B,
    QWEN3_8B,
    GEMMA3_4B,
]  # List of models to load into the registry (for testing muninn_chat_models VT)


# ── Sample documents ────────────────────────────────────────────────
DOCUMENTS = [
    (1, "Alice Smith founded ACME Corporation in New York City in 1987."),
    (2, "Bob Jones, CEO of TechStart, announced a partnership with ACME Corporation."),
    (3, "The European Central Bank raised interest rates to combat inflation in the eurozone."),
    (4, "Dr. Marie Curie discovered radium at the University of Paris in 1898."),
    (5, "Amazon acquired Whole Foods for $13.7 billion, reshaping the grocery industry."),
]

AG_NEWS_PATH = PROJECT_ROOT / "benchmarks" / "vectors" / "ag_news_queries.json"

performance_by_model: dict[str, dict[str, float]] = {}


def load_ag_news_docs(n: int, seed: int = 42) -> list[tuple[int, str]]:
    """Load N random documents from ag_news_queries.json."""
    data = json.loads(AG_NEWS_PATH.read_text(encoding="utf-8"))
    queries = data["queries"]
    rng = random.Random(seed)
    sample = rng.sample(queries, min(n, len(queries)))
    return [(i + 1, text) for i, text in enumerate(sample)]


def ensure_model(model: ChatModelConfig) -> bool:
    """Ensure a GGUF model file is available, downloading if needed."""
    path = MODELS_DIR / model.filename
    if path.exists():
        log.info("Model %s found: %s (%.1f MB)", model.name, path, path.stat().st_size / 1e6)
        return True

    log.info("Model %s not found — downloading %s (~%.1f GB)...", model.name, model.filename, model.size_gb)
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


def pp_json(raw: str) -> str:
    """Pretty-print a JSON string."""
    return json.dumps(json.loads(raw), indent=2)


def _fmt_elapsed(seconds: float, n_docs: int) -> str:
    """Format elapsed time with per-doc rate."""
    return f"{seconds:.2f}s ({seconds / n_docs:.2f}s/doc)" if n_docs > 0 else f"{seconds:.2f}s"


# ── Section 1: Model Loading ────────────────────────────────────────
def section_model_loading(db: sqlite3.Connection, model: ChatModelConfig) -> None:
    """Load a GGUF chat model into the muninn_chat_models registry."""
    print("\n" + "=" * 60)
    print("Section 1: Model Loading")
    print("=" * 60)

    path = MODELS_DIR / model.filename
    db.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
        (model.name, str(path)),
    )

    rows = db.execute("SELECT name, n_ctx FROM muninn_chat_models").fetchall()
    for name, n_ctx in rows:
        print(f"\n  Loaded: {name} (context window: {n_ctx} tokens)")


# ── Section 3: Named Entity Recognition ─────────────────────────────
def section_ner(db: sqlite3.Connection, model_name: str) -> tuple[dict[int, str], float]:
    """Extract entities from sample documents using GBNF grammar constraints."""
    print("\n" + "=" * 60)
    print("Section 3: Named Entity Recognition")
    print("=" * 60)

    labels = "person,organization,location,date"
    entities_by_doc: dict[int, str] = {}
    t0 = time.perf_counter()

    for doc_id, content in DOCUMENTS:
        results = db.execute(
            "SELECT muninn_extract_entities(?, ?, ?)",
            (model_name, content, labels),
        ).fetchone()
        result = results[0]
        entities_by_doc[doc_id] = result

        parsed = json.loads(result)
        n_ents = len(parsed.get("entities", []))
        print(f"\n  Doc #{doc_id}: {n_ents} entities found")
        for ent in parsed.get("entities", []):
            print(f"    - {ent['text']!r} ({ent['type']}, score={ent.get('score', 'N/A')})")

    elapsed = time.perf_counter() - t0
    print(f"\n  Elapsed: {_fmt_elapsed(elapsed, len(DOCUMENTS))}")
    return entities_by_doc, elapsed


# ── Section 4: Relation Extraction ──────────────────────────────────
def section_re(
    db: sqlite3.Connection, model_name: str, entities_by_doc: dict[int, str]
) -> tuple[dict[int, list], float]:
    """Chain NER output into relation extraction."""
    print("\n" + "=" * 60)
    print("Section 4: Relation Extraction")
    print("=" * 60)

    results_by_doc: dict[int, list] = {}
    t0 = time.perf_counter()

    for doc_id, content in DOCUMENTS:
        ents_json = entities_by_doc.get(doc_id, '{"entities":[]}')
        results = db.execute(
            "SELECT muninn_extract_relations(?, ?, ?)",
            (model_name, content, ents_json),
        ).fetchone()
        (result,) = results

        parsed = json.loads(result)
        rels = parsed.get("relations", [])
        results_by_doc[doc_id] = rels
        print(f"\n  Doc #{doc_id}: {len(rels)} relations found")
        for rel in rels:
            print(f"    - {rel['head']} --[{rel['rel']}]--> {rel['tail']} (score={rel.get('score', 'N/A')})")

    elapsed = time.perf_counter() - t0
    print(f"\n  Elapsed: {_fmt_elapsed(elapsed, len(DOCUMENTS))}")
    return results_by_doc, elapsed


# ── Section 6: Combined NER+RE in One Pass ────────────────────────
def section_combined_ner_re(db: sqlite3.Connection, model_name: str) -> tuple[dict[int, dict], float]:
    """Combined NER+RE via muninn_extract_ner_re — one LLM call per row."""
    print("\n" + "=" * 60)
    print("Section 6: Combined NER+RE (single LLM call per document)")
    print("=" * 60)

    entity_labels = "person,organization,location,date"
    relation_labels = "founded,acquired,partner_of,located_in,ceo_of"

    results_by_doc: dict[int, dict] = {}
    t0 = time.perf_counter()

    results = db.execute(
        """
        SELECT id, content,
               muninn_extract_ner_re(?, content, ?, ?) AS kg_json
        FROM documents
        """,
        (model_name, entity_labels, relation_labels),
    ).fetchall()

    elapsed = time.perf_counter() - t0

    for doc_id, content, kg_json in results:
        parsed = json.loads(kg_json)
        results_by_doc[doc_id] = parsed
        entities = parsed.get("entities", [])
        relations = parsed.get("relations", [])
        print(f"\n  Doc #{doc_id}: {len(entities)} entities, {len(relations)} relations")
        print(f"    Text: {content[:60]}...")
        for ent in entities:
            print(f"    E: {ent['text']!r} ({ent.get('type', '?')}, score={ent.get('score', 'N/A')})")
        for rel in relations:
            print(f"    R: {rel['head']} --[{rel['rel']}]--> {rel['tail']} (score={rel.get('score', 'N/A')})")

    print(f"\n  Elapsed: {_fmt_elapsed(elapsed, len(DOCUMENTS))}")
    return results_by_doc, elapsed


# ── Section 7: Bulk SQL Pipeline (two-pass NER→RE) ───────────────
def section_bulk_pipeline(db: sqlite3.Connection, model_name: str) -> tuple[dict[int, list], float]:
    """NER and RE chained in SQL CTE — two LLM calls per row (for comparison)."""
    print("\n" + "=" * 60)
    print("Section 7: Two-Pass Pipeline (NER -> RE in SQL CTE)")
    print("=" * 60)

    results_by_doc: dict[int, list] = {}
    t0 = time.perf_counter()

    results = db.execute(
        """
        WITH ner AS (
            SELECT id, content,
                   muninn_extract_entities(?, content, 'person,organization,location') AS entities_json
            FROM documents
        )
        SELECT id,
               muninn_extract_relations(?, content, entities_json) AS relations_json
        FROM ner
        """,
        (model_name, model_name),
    ).fetchall()

    elapsed = time.perf_counter() - t0

    for doc_id, relations_json in results:
        parsed = json.loads(relations_json)
        rels = parsed.get("relations", [])
        results_by_doc[doc_id] = rels
        print(f"\n  Doc #{doc_id}: {len(rels)} relations")
        for rel in rels:
            print(f"    {rel['head']} --[{rel['rel']}]--> {rel['tail']}")

    print(f"\n  Elapsed: {_fmt_elapsed(elapsed, len(DOCUMENTS))}")
    return results_by_doc, elapsed


# ── Section 9: Batch NER+RE (parallel sequences) ──────────────────
def section_batch_ner_re(db: sqlite3.Connection, model_name: str, batch_size: int = 4) -> tuple[dict[int, dict], float]:
    """Combined NER+RE via muninn_extract_ner_re_batch — N docs in parallel."""
    print("\n" + "=" * 60)
    print(f"Section 9: Batch NER+RE (batch_size={batch_size}, parallel sequences)")
    print("=" * 60)

    entity_labels = "person,organization,location,date"
    relation_labels = "founded,acquired,partner_of,located_in,ceo_of"

    texts = [content for _, content in DOCUMENTS]
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]
    texts_json = json.dumps(texts)

    t0 = time.perf_counter()
    row = db.execute(
        "SELECT muninn_extract_ner_re_batch(?, ?, ?, ?, ?)",
        (model_name, texts_json, entity_labels, relation_labels, batch_size),
    ).fetchone()
    elapsed = time.perf_counter() - t0

    results_arr = json.loads(row[0])
    results_by_doc: dict[int, dict] = {}

    for i, parsed in enumerate(results_arr):
        doc_id = doc_ids[i]
        results_by_doc[doc_id] = parsed
        entities = parsed.get("entities", [])
        relations = parsed.get("relations", [])
        print(f"\n  Doc #{doc_id}: {len(entities)} entities, {len(relations)} relations")
        for ent in entities:
            print(f"    E: {ent['text']!r} ({ent.get('type', '?')}, score={ent.get('score', 'N/A')})")
        for rel in relations:
            print(f"    R: {rel['head']} --[{rel['rel']}]--> {rel['tail']} (score={rel.get('score', 'N/A')})")

    print(f"\n  Elapsed: {_fmt_elapsed(elapsed, len(DOCUMENTS))}")
    return results_by_doc, elapsed


# ── Tabular Output ────────────────────────────────────────────────
METRIC_LABELS = [
    ("two_pass", "Two-pass (S3+S4)"),
    ("combined", "Combined NER+RE (S6)"),
    ("pipeline", "SQL CTE 2-pass (S7)"),
    ("batch", "Batch NER+RE (S9)"),
]


def _print_table(
    title: str,
    model_names: list[str],
    rows: list[tuple[str, list[str]]],
) -> None:
    """Print a table with model names as columns and metric labels as rows."""
    label_width = max(len(label) for _, label in METRIC_LABELS)
    col_width = max(12, *(len(m) for m in model_names))

    header = f"  {'Metric':<{label_width}}  " + "  ".join(f"{m:>{col_width}}" for m in model_names)
    sep = "  " + "-" * (label_width + 2 + (col_width + 2) * len(model_names))

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for label, values in rows:
        vals_str = "  ".join(f"{v:>{col_width}}" for v in values)
        print(f"  {label:<{label_width}}  {vals_str}")
    print(sep)


def print_timing_tables(n_docs: int) -> None:
    """Print timing table and speedup multiplier table."""
    model_names = list(performance_by_model.keys())

    # Determine which metrics are available (batch-only mode may have fewer)
    first = next(iter(performance_by_model.values()))
    available = [(k, label) for k, label in METRIC_LABELS if k == "two_pass" or k in first]

    # Build timing dict
    timings_by_metric: dict[str, list[float]] = {}
    for key, _ in available:
        vals = []
        for mn in model_names:
            t = performance_by_model[mn]
            if key == "two_pass":
                vals.append(t.get("ner", 0) + t.get("re", 0))
            else:
                vals.append(t.get(key, 0))
        timings_by_metric[key] = vals

    # Skip two_pass if ner/re not present
    if "ner" not in first:
        available = [(k, lab) for k, lab in available if k != "two_pass"]

    # Table 1: Absolute timings
    rows_abs = []
    for key, label in available:
        vals = timings_by_metric[key]
        formatted = [f"{v:.2f}s" for v in vals]
        rows_abs.append((label, formatted))
    # Per-doc row
    rows_abs.append(("", [""] * len(model_names)))  # spacer
    for key, label in available:
        vals = timings_by_metric[key]
        formatted = [f"{v / n_docs:.2f}s/doc" for v in vals]
        rows_abs.append((f"  {label} /doc", formatted))

    _print_table(f"Timing ({n_docs} documents)", model_names, rows_abs)

    # Table 2: Speedup multipliers (relative to slowest time per metric)
    rows_speed = []
    for key, label in available:
        vals = timings_by_metric[key]
        slowest = max(vals) if vals else 1.0
        formatted = [f"{slowest / v:.2f}x" if v > 0 else "N/A" for v in vals]
        rows_speed.append((label, formatted))

    _print_table("Speedup (vs slowest)", model_names, rows_speed)


# ── Main ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="muninn LLM Extract — NER & RE benchmark")
    parser.add_argument(
        "--ag-news-docs",
        type=int,
        default=0,
        metavar="N",
        help="Load N random docs from benchmarks/vectors/ag_news_queries.json (overrides built-in docs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ag_news sampling")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for batch NER+RE")
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Only run batch NER+RE (skip sequential sections — useful for large N)",
    )
    return parser.parse_args()


def main() -> None:
    global DOCUMENTS
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.ag_news_docs > 0:
        DOCUMENTS = load_ag_news_docs(args.ag_news_docs, seed=args.seed)
        log.info("Loaded %d ag_news documents (seed=%d)", len(DOCUMENTS), args.seed)

    for model in ENABLED_MODELS:
        if not ensure_model(model):
            log.error("Model not available. Cannot proceed.")
            return

    for model in ENABLED_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model.name} (~{model.size_gb} GB)  |  {len(DOCUMENTS)} documents")
        print(f"{'=' * 60}")

        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        db.load_extension(EXTENSION_PATH)

        db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
        db.executemany("INSERT INTO documents(id, content) VALUES (?, ?)", DOCUMENTS)

        section_model_loading(db, model)

        if args.batch_only:
            _, batch_elapsed = section_batch_ner_re(db, model.name, batch_size=args.batch_size)
            performance_by_model[model.name] = {"batch": batch_elapsed}
        else:
            entities_by_doc, ner_elapsed = section_ner(db, model.name)
            _, re_elapsed = section_re(db, model.name, entities_by_doc)
            _, combined_elapsed = section_combined_ner_re(db, model.name)
            _, pipeline_elapsed = section_bulk_pipeline(db, model.name)
            _, batch_elapsed = section_batch_ner_re(db, model.name, batch_size=args.batch_size)
            performance_by_model[model.name] = {
                "ner": ner_elapsed,
                "re": re_elapsed,
                "combined": combined_elapsed,
                "pipeline": pipeline_elapsed,
                "batch": batch_elapsed,
            }

    print_timing_tables(len(DOCUMENTS))


if __name__ == "__main__":
    main()
