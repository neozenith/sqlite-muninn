"""
LLM Extract — Comparing muninn GGUF models vs GLiNER2 for NER & RE

Consolidated benchmark comparing structured information extraction across
multiple GGUF chat models (via muninn SQL functions) and GLiNER2 (205M span
extraction model) on 5 curated documents.

Sections:
  1. GGUF model loading via muninn_chat_models virtual table
  2. Chat completion via muninn_chat() (GGUF only)
  3. Text summarisation via muninn_summarize() (GGUF only)
  4. NER comparison: muninn_extract_entities() per model vs GLiNER2
  5. RE comparison: muninn_extract_relations() per model vs GLiNER2
  6. Combined NER+RE: muninn_extract_ner_re() per model vs GLiNER2
  7. CTE pipeline: SQL CTE chaining NER→RE per model vs GLiNER2
  8. Summary timing + count comparison tables

Requirements:
  - muninn extension (make all)
  - GGUF chat models (auto-downloaded on first run)
  - gliner2 (uv add gliner2)
"""

import json
import logging
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from gliner2 import GLiNER2

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")
MODELS_DIR = PROJECT_ROOT / "models"


# ── Model Definitions ──────────────────────────────────────────────


@dataclass
class GgufModel:
    """GGUF chat model descriptor with auto-download URL."""

    name: str
    filename: str
    url: str
    size_gb: float


MODELS = [
    GgufModel(
        "Qwen3.5-4B",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        2.7,
    ),
    GgufModel(
        "Qwen3.5-9B",
        "Qwen3.5-9B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
        5.7,
    ),
    GgufModel(
        "Gemma-3-4B",
        "google_gemma-3-4b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
        2.5,
    ),
]


# ── Sample Documents ───────────────────────────────────────────────

DOCUMENTS = [
    (1, "Alice Smith founded ACME Corporation in New York City in 1987."),
    (2, "Bob Jones, CEO of TechStart, announced a partnership with ACME Corporation."),
    (3, "The European Central Bank raised interest rates to combat inflation in the eurozone."),
    (4, "Dr. Marie Curie discovered radium at the University of Paris in 1898."),
    (5, "Amazon acquired Whole Foods for $13.7 billion, reshaping the grocery industry."),
]

ENTITY_LABELS = ["person", "organization", "location", "date"]
ENTITY_LABELS_CSV = ",".join(ENTITY_LABELS)
RELATION_LABELS = ["founded", "acquired", "partner_of", "located_in", "ceo_of", "works_for", "discovered", "announced"]
RELATION_LABELS_CSV = ",".join(RELATION_LABELS)


# ── Accumulators ───────────────────────────────────────────────────

timings: dict[str, dict[str, float]] = {}
entity_counts: dict[str, dict[str, int]] = {}
relation_counts: dict[str, dict[str, int]] = {}


# ── Utilities ──────────────────────────────────────────────────────


def ensure_model(model: GgufModel) -> None:
    """Download GGUF model if not already present. Raises on failure."""
    path = MODELS_DIR / model.filename
    if path.exists():
        log.info("Model %s: %s (%.1f MB)", model.name, path, path.stat().st_size / 1e6)
        return

    log.info("Downloading %s (~%.1f GB)...", model.filename, model.size_gb)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_with_progress(model.url, path)
    log.info("Downloaded %s (%.1f MB)", model.filename, path.stat().st_size / 1e6)


def _download_with_progress(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-example/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)", end="", flush=True)
        if total > 0:
            print()


def _record_time(backend: str, metric: str, elapsed: float) -> None:
    timings.setdefault(backend, {})[metric] = elapsed


def _record_entities(backend: str, metric: str, count: int) -> None:
    entity_counts.setdefault(backend, {})[metric] = count


def _record_relations(backend: str, metric: str, count: int) -> None:
    relation_counts.setdefault(backend, {})[metric] = count


def _score_str(score: object) -> str:
    if isinstance(score, (int, float)):
        return f"{score:.2f}"
    return "—"


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _subheader(title: str) -> None:
    width = max(1, 60 - len(title))
    print(f"\n  ── {title} {'─' * width}")


# ── Section 1: Model Loading ──────────────────────────────────────


def load_gguf_models(db: sqlite3.Connection) -> list[str]:
    """Load all GGUF models into the muninn_chat_models registry."""
    _header("Section 1: GGUF Model Loading")

    names = []
    for model in MODELS:
        ensure_model(model)
        path = MODELS_DIR / model.filename
        t0 = time.perf_counter()
        db.execute(
            "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
            (model.name, str(path)),
        )
        elapsed = time.perf_counter() - t0
        _record_time(model.name, "model_load", elapsed)
        names.append(model.name)

    rows = db.execute("SELECT name, n_ctx FROM muninn_chat_models").fetchall()
    for name, n_ctx in rows:
        t = timings.get(name, {}).get("model_load", 0)
        print(f"  Loaded: {name} (context: {n_ctx} tokens, load: {t:.2f}s)")

    return names


def load_gliner2() -> GLiNER2:
    """Load the GLiNER2 model."""
    _subheader("GLiNER2 Loading")
    t0 = time.perf_counter()
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    elapsed = time.perf_counter() - t0
    _record_time("GLiNER2", "model_load", elapsed)
    print(f"  Loaded: GLiNER2 (205M params, load: {elapsed:.2f}s)")
    return model


# ── Section 2: Chat Completion (GGUF only) ────────────────────────


def section_chat(db: sqlite3.Connection, model_name: str) -> None:
    """Free-form chat completion via muninn_chat()."""
    _header(f"Section 2: Chat Completion — {model_name}")
    print("  (GGUF only — GLiNER2 is a span extraction model, not a generative LLM)")

    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Explain what a knowledge graph is in two sentences.",
    ]

    for prompt in prompts:
        t0 = time.perf_counter()
        result = db.execute("SELECT muninn_chat(?, ?)", (model_name, prompt)).fetchone()[0]
        elapsed = time.perf_counter() - t0

        clean = result.strip()
        print(f"\n  Prompt:   {prompt}")
        print(f"  Response: {clean[:300]}{'...' if len(clean) > 300 else ''}")
        print(f"  Time:     {elapsed:.2f}s")


# ── Section 3: Summarisation (GGUF only) ──────────────────────────


def section_summarize(db: sqlite3.Connection, model_name: str) -> None:
    """Text summarisation via muninn_summarize()."""
    _header(f"Section 3: Summarisation — {model_name}")
    print("  (GGUF only — GLiNER2 does not support summarisation)")

    all_text = " ".join(content for _, content in DOCUMENTS)

    t0 = time.perf_counter()
    summary = db.execute("SELECT muninn_summarize(?, ?, 64)", (model_name, all_text)).fetchone()[0]
    elapsed = time.perf_counter() - t0

    print(f"\n  Input ({len(all_text)} chars): {all_text[:100]}...")
    print(f"  Summary: {summary.strip()}")
    print(f"  Time:    {elapsed:.2f}s")

    print("\n  Per-document summaries:")
    for doc_id, content in DOCUMENTS:
        s = db.execute("SELECT muninn_summarize(?, ?, 32)", (model_name, content)).fetchone()[0]
        print(f"    Doc #{doc_id}: {s.strip()}")


# ── Section 4: NER Comparison ─────────────────────────────────────


def section_ner_comparison(
    db: sqlite3.Connection,
    gguf_models: list[str],
    gliner2: GLiNER2,
) -> tuple[dict[str, dict[int, str]], list[dict]]:
    """Compare NER across all backends.

    Returns (ner_json_by_model, gliner2_ner_results):
      - ner_json_by_model: raw NER JSON per GGUF model per doc (for feeding into RE)
      - gliner2_ner_results: cached GLiNER2 batch NER output (reused in later sections)
    """
    _header("Section 4: NER Comparison — muninn_extract_entities vs GLiNER2")

    ner_json_by_model: dict[str, dict[int, str]] = {}

    # ── GGUF models ──
    for model_name in gguf_models:
        _subheader(f"muninn NER — {model_name}")
        ner_by_doc: dict[int, str] = {}
        t0 = time.perf_counter()

        for doc_id, content in DOCUMENTS:
            raw = db.execute(
                "SELECT muninn_extract_entities(?, ?, ?)",
                (model_name, content, ENTITY_LABELS_CSV),
            ).fetchone()[0]
            ner_by_doc[doc_id] = raw
            parsed = json.loads(raw)
            ents = parsed.get("entities", [])
            print(f"    Doc #{doc_id}: {len(ents)} entities")
            for e in ents:
                print(f"      {e['text']!r} ({e['type']}, score={_score_str(e.get('score'))})")

        elapsed = time.perf_counter() - t0
        total = sum(len(json.loads(v).get("entities", [])) for v in ner_by_doc.values())
        _record_time(model_name, "ner", elapsed)
        _record_entities(model_name, "ner", total)
        print(f"    Total: {total} entities in {elapsed:.2f}s ({elapsed / len(DOCUMENTS):.2f}s/doc)")
        ner_json_by_model[model_name] = ner_by_doc

    # ── GLiNER2 ──
    _subheader("GLiNER2 NER")
    texts = [content for _, content in DOCUMENTS]
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]

    t0 = time.perf_counter()
    ner_results = gliner2.batch_extract_entities(texts, ENTITY_LABELS, batch_size=8, include_confidence=True)
    elapsed = time.perf_counter() - t0

    total = 0
    for i, result in enumerate(ner_results):
        doc_id = doc_ids[i]
        ents = []
        for label, items in result.get("entities", {}).items():
            for item in items:
                ents.append(item["text"])
                print(f"      {item['text']!r} ({label}, score={_score_str(item.get('confidence'))})")
        total += len(ents)
        print(f"    Doc #{doc_id}: {len(ents)} entities")

    _record_time("GLiNER2", "ner", elapsed)
    _record_entities("GLiNER2", "ner", total)
    print(f"    Total: {total} entities in {elapsed:.2f}s ({elapsed / len(DOCUMENTS):.2f}s/doc)")

    return ner_json_by_model, ner_results


# ── Section 5: RE Comparison ──────────────────────────────────────


def section_re_comparison(
    db: sqlite3.Connection,
    gguf_models: list[str],
    gliner2: GLiNER2,
    ner_json_by_model: dict[str, dict[int, str]],
) -> list[dict]:
    """Compare RE across all backends.

    muninn RE chains off prior NER output (entity-aware).
    GLiNER2 RE operates directly on text (no entity dependency).

    Returns gliner2_re_results: cached GLiNER2 batch RE output (reused in later sections).
    """
    _header("Section 5: RE Comparison — muninn_extract_relations vs GLiNER2")
    print("  Note: muninn RE requires prior NER entities as input;")
    print("        GLiNER2 RE operates directly on text (no entity dependency).")

    # ── GGUF models ──
    for model_name in gguf_models:
        _subheader(f"muninn RE — {model_name}")
        ner_by_doc = ner_json_by_model.get(model_name, {})
        t0 = time.perf_counter()
        total = 0

        for doc_id, content in DOCUMENTS:
            ents_json = ner_by_doc.get(doc_id, '{"entities":[]}')
            raw = db.execute(
                "SELECT muninn_extract_relations(?, ?, ?)",
                (model_name, content, ents_json),
            ).fetchone()[0]
            parsed = json.loads(raw)
            rels = parsed.get("relations", [])
            total += len(rels)
            print(f"    Doc #{doc_id}: {len(rels)} relations")
            for r in rels:
                print(f"      {r['head']} --[{r['rel']}]--> {r['tail']} (score={_score_str(r.get('score'))})")

        elapsed = time.perf_counter() - t0
        _record_time(model_name, "re", elapsed)
        _record_relations(model_name, "re", total)
        print(f"    Total: {total} relations in {elapsed:.2f}s ({elapsed / len(DOCUMENTS):.2f}s/doc)")

    # ── GLiNER2 ──
    _subheader("GLiNER2 RE")
    texts = [content for _, content in DOCUMENTS]
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]

    t0 = time.perf_counter()
    re_results = gliner2.batch_extract_relations(texts, RELATION_LABELS, batch_size=8)
    elapsed = time.perf_counter() - t0

    total = 0
    for i, result in enumerate(re_results):
        doc_id = doc_ids[i]
        rels = []
        for rel_type, items in result.get("relation_extraction", {}).items():
            for item in items:
                head, tail = item[0], item[1]
                rels.append(item)
                print(f"      {head} --[{rel_type}]--> {tail}")
        total += len(rels)
        print(f"    Doc #{doc_id}: {len(rels)} relations")

    _record_time("GLiNER2", "re", elapsed)
    _record_relations("GLiNER2", "re", total)
    print(f"    Total: {total} relations in {elapsed:.2f}s ({elapsed / len(DOCUMENTS):.2f}s/doc)")

    return re_results


# ── Section 6: Combined NER+RE Comparison ─────────────────────────


def section_combined_ner_re(
    db: sqlite3.Connection,
    gguf_models: list[str],
    gliner2_ner_results: list[dict],
    gliner2_re_results: list[dict],
) -> None:
    """Compare combined NER+RE in one pass.

    muninn: muninn_extract_ner_re() — single LLM call per document.
    GLiNER2: reuses cached batch results from Sections 4+5.
    """
    _header("Section 6: Combined NER+RE — muninn_extract_ner_re vs GLiNER2")

    # ── GGUF models: single LLM call per document ──
    for model_name in gguf_models:
        _subheader(f"muninn NER+RE — {model_name}")
        t0 = time.perf_counter()

        results = db.execute(
            """SELECT id, content,
                      muninn_extract_ner_re(?, content, ?, ?) AS kg_json
               FROM documents""",
            (model_name, ENTITY_LABELS_CSV, RELATION_LABELS_CSV),
        ).fetchall()
        elapsed = time.perf_counter() - t0

        total_e = 0
        total_r = 0
        for doc_id, content, kg_json in results:
            parsed = json.loads(kg_json)
            ents = parsed.get("entities", [])
            rels = parsed.get("relations", [])
            total_e += len(ents)
            total_r += len(rels)
            print(f"    Doc #{doc_id}: {len(ents)} entities, {len(rels)} relations")
            for e in ents:
                print(f"      E: {e['text']!r} ({e.get('type', '?')}, score={_score_str(e.get('score'))})")
            for r in rels:
                print(f"      R: {r['head']} --[{r['rel']}]--> {r['tail']} (score={_score_str(r.get('score'))})")

        _record_time(model_name, "combined", elapsed)
        _record_entities(model_name, "combined", total_e)
        _record_relations(model_name, "combined", total_r)
        print(f"    Total: {total_e} entities + {total_r} relations in {elapsed:.2f}s")

    # ── GLiNER2: reuse cached results from Sections 4+5 ──
    _subheader("GLiNER2 NER+RE (cached from Sections 4+5)")
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]
    elapsed = timings["GLiNER2"]["ner"] + timings["GLiNER2"]["re"]

    total_e = 0
    total_r = 0
    for i in range(len(DOCUMENTS)):
        doc_id = doc_ids[i]
        ents = []
        for label, items in gliner2_ner_results[i].get("entities", {}).items():
            for item in items:
                ents.append({"text": item["text"], "type": label, "score": item.get("confidence", 0)})
        rels = []
        for rel_type, items in gliner2_re_results[i].get("relation_extraction", {}).items():
            for item in items:
                rels.append({"head": item[0], "rel": rel_type, "tail": item[1]})
        total_e += len(ents)
        total_r += len(rels)
        print(f"    Doc #{doc_id}: {len(ents)} entities, {len(rels)} relations")
        for e in ents:
            print(f"      E: {e['text']!r} ({e['type']}, score={_score_str(e.get('score'))})")
        for r in rels:
            print(f"      R: {r['head']} --[{r['rel']}]--> {r['tail']}")

    _record_time("GLiNER2", "combined", elapsed)
    _record_entities("GLiNER2", "combined", total_e)
    _record_relations("GLiNER2", "combined", total_r)
    print(f"    Total: {total_e} entities + {total_r} relations in {elapsed:.2f}s (cached)")


# ── Section 7: CTE Pipeline vs GLiNER2 ───────────────────────────


def section_cte_pipeline(
    db: sqlite3.Connection,
    gguf_models: list[str],
    gliner2_ner_results: list[dict],
    gliner2_re_results: list[dict],
) -> None:
    """Compare SQL CTE two-pass pipeline (NER→RE chained) vs GLiNER2 combined.

    CTE: 2 LLM calls per document (NER then RE), chained in pure SQL.
    GLiNER2: reuses cached batch results from Sections 4+5.
    """
    _header("Section 7: CTE Pipeline (NER→RE in SQL) vs GLiNER2")
    print("  CTE: 2 LLM calls per doc, chained in pure SQL via WITH clause.")
    print("  GLiNER2: 2 batch calls total over all documents.")

    # ── GGUF models: SQL CTE pipeline ──
    for model_name in gguf_models:
        _subheader(f"muninn CTE — {model_name}")
        t0 = time.perf_counter()

        results = db.execute(
            """WITH ner AS (
                SELECT id, content,
                       muninn_extract_entities(?, content, ?) AS entities_json
                FROM documents
            )
            SELECT id, content, entities_json,
                   muninn_extract_relations(?, content, entities_json) AS relations_json
            FROM ner""",
            (model_name, ENTITY_LABELS_CSV, model_name),
        ).fetchall()
        elapsed = time.perf_counter() - t0

        total_e = 0
        total_r = 0
        for doc_id, content, ents_json, rels_json in results:
            ents = json.loads(ents_json).get("entities", [])
            rels = json.loads(rels_json).get("relations", [])
            total_e += len(ents)
            total_r += len(rels)
            print(f"    Doc #{doc_id}: {len(ents)} entities → {len(rels)} relations")
            for r in rels:
                print(f"      {r['head']} --[{r['rel']}]--> {r['tail']} (score={_score_str(r.get('score'))})")

        _record_time(model_name, "cte", elapsed)
        _record_entities(model_name, "cte", total_e)
        _record_relations(model_name, "cte", total_r)
        print(f"    Total: {total_e} entities + {total_r} relations in {elapsed:.2f}s")

    # ── GLiNER2: reuse cached results from Sections 4+5 ──
    _subheader("GLiNER2 NER+RE (cached from Sections 4+5)")
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]
    elapsed = timings["GLiNER2"]["ner"] + timings["GLiNER2"]["re"]

    total_e = 0
    total_r = 0
    for i in range(len(DOCUMENTS)):
        doc_id = doc_ids[i]
        n_ents = sum(len(items) for items in gliner2_ner_results[i].get("entities", {}).values())
        n_rels = sum(len(items) for items in gliner2_re_results[i].get("relation_extraction", {}).values())
        total_e += n_ents
        total_r += n_rels
        print(f"    Doc #{doc_id}: {n_ents} entities, {n_rels} relations")

    _record_time("GLiNER2", "cte", elapsed)
    _record_entities("GLiNER2", "cte", total_e)
    _record_relations("GLiNER2", "cte", total_r)
    print(f"    Total: {total_e} entities + {total_r} relations in {elapsed:.2f}s (cached)")


# ── Section 8: Summary Tables ─────────────────────────────────────

METRICS = [
    ("ner", "NER only"),
    ("re", "RE only"),
    ("combined", "Combined NER+RE"),
    ("cte", "CTE Pipeline"),
]


def _print_table(title: str, backends: list[str], rows: list[tuple[str, list[str]]]) -> None:
    col_w = max(14, *(len(b) for b in backends))
    label_w = max(len(label) for label, _ in rows) if rows else 20
    sep = "  " + "-" * (label_w + 2 + (col_w + 2) * len(backends))

    print(f"\n  {title}")
    print(sep)
    print(f"  {'':>{label_w}}  " + "  ".join(f"{b:>{col_w}}" for b in backends))
    print(sep)
    for label, vals in rows:
        print(f"  {label:>{label_w}}  " + "  ".join(f"{v:>{col_w}}" for v in vals))
    print(sep)


def print_summary_tables() -> None:
    """Print timing, per-doc, speedup, and extraction count tables."""
    _header("Section 8: Summary Comparison Tables")

    backends = list(timings.keys())
    n = len(DOCUMENTS)

    # Only show metrics that have data
    available = [(k, label) for k, label in METRICS if any(k in timings.get(b, {}) for b in backends)]

    # Table 1: Absolute timings
    rows = []
    for key, label in available:
        vals = [f"{timings.get(b, {}).get(key, 0):.2f}s" if key in timings.get(b, {}) else "—" for b in backends]
        rows.append((label, vals))
    _print_table(f"Absolute Timings ({n} documents)", backends, rows)

    # Table 2: Per-document timings
    rows = []
    for key, label in available:
        vals = []
        for b in backends:
            t = timings.get(b, {}).get(key)
            vals.append(f"{t / n:.3f}s/doc" if t is not None else "—")
        rows.append((label, vals))
    _print_table("Per-Document Timings", backends, rows)

    # Table 3: Speedup vs slowest per metric
    rows = []
    for key, label in available:
        vals_raw = [timings.get(b, {}).get(key) for b in backends]
        valid = [v for v in vals_raw if v is not None and v > 0]
        slowest = max(valid) if valid else 1.0
        vals = [f"{slowest / v:.1f}x" if v is not None and v > 0 else "—" for v in vals_raw]
        rows.append((label, vals))
    _print_table("Speedup (vs slowest per metric)", backends, rows)

    # Table 4: Entity counts
    rows = []
    for key, label in available:
        vals = [str(entity_counts.get(b, {}).get(key, "—")) for b in backends]
        rows.append((label, vals))
    _print_table("Entity Counts", backends, rows)

    # Table 5: Relation counts
    rows = []
    for key, label in available:
        vals = [str(relation_counts.get(b, {}).get(key, "—")) for b in backends]
        rows.append((label, vals))
    _print_table("Relation Counts", backends, rows)

    # Model load times
    rows = []
    load_vals = [f"{timings.get(b, {}).get('model_load', 0):.2f}s" for b in backends]
    rows.append(("Model load", load_vals))
    _print_table("Model Load Times", backends, rows)


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Ensure all GGUF models are available
    for model in MODELS:
        ensure_model(model)

    # Setup SQLite with muninn
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    db.executemany("INSERT INTO documents(id, content) VALUES (?, ?)", DOCUMENTS)

    # Section 1: Load all models
    gguf_names = load_gguf_models(db)
    gliner2 = load_gliner2()

    # Section 2-3: GGUF-only features (use first model)
    section_chat(db, gguf_names[0])
    section_summarize(db, gguf_names[0])

    # Section 4-7: Comparative benchmarks (GLiNER2 results cached from 4+5, reused in 6+7)
    ner_json, gliner2_ner = section_ner_comparison(db, gguf_names, gliner2)
    gliner2_re = section_re_comparison(db, gguf_names, gliner2, ner_json)
    section_combined_ner_re(db, gguf_names, gliner2_ner, gliner2_re)
    section_cte_pipeline(db, gguf_names, gliner2_ner, gliner2_re)

    # Section 8: Summary tables
    print_summary_tables()


if __name__ == "__main__":
    main()
