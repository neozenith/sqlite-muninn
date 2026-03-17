"""
LLM Summarize — Text summarisation via muninn_summarize()

Demonstrates document summarisation using a GGUF chat model.
muninn_summarize() handles <think> blocks internally — thinking models
reason first, then the response is extracted and returned clean.

Requirements:
  - muninn extension (make all)
  - GGUF chat model (auto-downloaded on first run)
"""

import logging
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


@dataclass
class GgufModel:
    name: str
    filename: str
    url: str
    size_gb: float


MODEL = GgufModel(
    "Qwen3.5-0.8B",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
    0.53,
)

# Multi-paragraph documents that benefit from summarization
DOCUMENTS = [
    (
        1,
        "Alice Smith founded ACME Corporation in New York City in 1987. Starting with just "
        "twelve employees in a small warehouse in Brooklyn, the company quickly grew to become "
        "a leader in manufacturing consumer goods across the eastern United States. By 1995, "
        "ACME had expanded to three factories, employed over 2,000 workers, and was generating "
        "annual revenues exceeding $500 million. Smith's innovative approach to supply chain "
        "management and her emphasis on quality control became case studies at Harvard Business School.",
    ),
    (
        2,
        "The European Central Bank raised interest rates by 25 basis points on Thursday, bringing "
        "the benchmark deposit rate to 4.0%, the highest level since the eurozone's creation in 1999. "
        "This marks the tenth consecutive increase since July 2022, when rates were still in negative "
        "territory at -0.5%. ECB President Christine Lagarde warned that inflation, currently at 5.3%, "
        "remains 'too high for too long' and signalled that further tightening could not be ruled out. "
        "Markets reacted negatively, with the Euro Stoxx 50 falling 1.2% and government bond yields "
        "rising across the bloc. Economists at Deutsche Bank and Goldman Sachs now expect rates to "
        "remain elevated through at least mid-2024.",
    ),
    (
        3,
        "Amazon acquired Whole Foods Market for $13.7 billion in June 2017, a deal that sent "
        "shockwaves through the grocery industry and accelerated the shift toward online food "
        "delivery. Within months of the acquisition, Amazon slashed prices on staple items by up "
        "to 40%, integrated Prime membership with in-store discounts, and began offering two-hour "
        "grocery delivery in major metropolitan areas through Amazon Fresh. Competitors responded "
        "aggressively: Walmart invested $3.3 billion in its online grocery platform, Kroger partnered "
        "with autonomous vehicle startup Nuro for last-mile delivery, and Instacart's valuation "
        "tripled to $7.6 billion as traditional grocers rushed to offer delivery services. By 2020, "
        "online grocery sales in the US had grown from 3% to 10% of total grocery spending.",
    ),
]


def ensure_model(model: GgufModel) -> None:
    path = MODELS_DIR / model.filename
    if path.exists():
        return
    log.info("Downloading %s (~%.1f GB)...", model.filename, model.size_gb)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(model.url, headers={"User-Agent": "muninn-example/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with path.open("wb") as f:
            while chunk := resp.read(256 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  {path.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)", end="", flush=True)
        if total > 0:
            print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    db.executemany("INSERT INTO documents(id, content) VALUES (?, ?)", DOCUMENTS)

    # Load model
    ensure_model(MODEL)
    path = MODELS_DIR / MODEL.filename
    t0 = time.perf_counter()
    db.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
        (MODEL.name, str(path)),
    )
    elapsed = time.perf_counter() - t0
    print(f"Loaded: {MODEL.name} (load: {elapsed:.2f}s)")

    # ── 1. Per-document summaries ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  1. Per-Document Summaries")
    print(f"{'=' * 60}")

    for doc_id, content in DOCUMENTS:
        t0 = time.perf_counter()
        summary = db.execute("SELECT muninn_summarize(?, ?)", (MODEL.name, content)).fetchone()[0]
        elapsed = time.perf_counter() - t0
        print(f"\n  Doc #{doc_id} ({elapsed:.2f}s, {len(content)} chars → {len(summary)} chars):")
        print(f"    Input:   {content}")
        print(f"    Summary: {summary or '[no summary produced]'}")

    # ── 2. Multi-document summary via SQL ─────────────────────────
    print(f"\n{'=' * 60}")
    print("  2. Multi-Document Summary (SQL aggregation)")
    print(f"{'=' * 60}")

    all_text = db.execute("SELECT group_concat(content, ' ') FROM documents").fetchone()[0]
    t0 = time.perf_counter()
    summary = db.execute("SELECT muninn_summarize(?, ?)", (MODEL.name, all_text)).fetchone()[0]
    elapsed = time.perf_counter() - t0
    print(f"\n  Input: {len(all_text)} chars from {len(DOCUMENTS)} documents:")
    print(f"  {all_text}")
    print(f"\n  Summary ({elapsed:.2f}s, {len(all_text)} → {len(summary)} chars):")
    print(f"  {summary or '[no summary produced]'}")

    # ── 3. SQL-native: summarise query results ────────────────────
    print(f"\n{'=' * 60}")
    print("  3. SQL-Native Summarisation (query → summarise)")
    print(f"{'=' * 60}")

    rows = db.execute(
        "SELECT id, content, muninn_summarize(?, content) AS summary FROM documents",
        (MODEL.name,),
    ).fetchall()
    for doc_id, content, summary in rows:
        print(f"\n  Doc #{doc_id}:")
        print(f"    Content: {content}")
        print(f"    Summary: {summary or '[no summary produced]'}")


if __name__ == "__main__":
    main()
