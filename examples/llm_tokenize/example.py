"""
LLM Tokenize — Inspect how GGUF models tokenize text

Demonstrates muninn_tokenize(), muninn_tokenize_text(), and muninn_token_count()
comparing an embed model (BERT/WordPiece) against a chat model (BPE).

With the unified model registry, all three tokenizer functions work with
any registered model — embed or chat.

Requirements:
  - muninn extension (make all)
  - GGUF models (auto-downloaded on first run)
"""

import json
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


CHAT_MODEL = GgufModel(
    "Qwen3.5-0.8B",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
    0.53,
)

EMBED_MODEL = GgufModel(
    "MiniLM",
    "all-MiniLM-L6-v2.Q8_0.gguf",
    "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
    0.023,
)

SAMPLES = [
    ("Short", "Alice Smith founded ACME Corporation in New York City in 1987."),
    ("Names", "Dr. Marie Curie discovered radium at the University of Paris in 1898."),
    (
        "Business",
        "Bob Jones, CEO of TechStart, announced a strategic partnership with ACME"
        " Corporation to develop AI-powered supply chain management tools.",
    ),
    (
        "Finance",
        "Amazon acquired Whole Foods for $13.7 billion in 2017, fundamentally reshaping"
        " the grocery industry and accelerating the shift toward online food delivery.",
    ),
    (
        "Multi-sentence",
        "The European Central Bank raised interest rates by 25 basis points on Thursday."
        " ECB President Christine Lagarde warned that inflation remains too high for too"
        " long. Markets reacted negatively, with the Euro Stoxx 50 falling 1.2%.",
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

    # ── Load models ──────────────────────────────────────────────
    ensure_model(EMBED_MODEL)
    path = MODELS_DIR / EMBED_MODEL.filename
    t0 = time.perf_counter()
    db.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
        (EMBED_MODEL.name, str(path)),
    )
    elapsed = time.perf_counter() - t0
    dim = db.execute("SELECT muninn_model_dim(?)", (EMBED_MODEL.name,)).fetchone()[0]
    print(f"Loaded: {EMBED_MODEL.name} (embed, dim={dim}, load: {elapsed:.2f}s)")

    ensure_model(CHAT_MODEL)
    path = MODELS_DIR / CHAT_MODEL.filename
    t0 = time.perf_counter()
    db.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
        (CHAT_MODEL.name, str(path)),
    )
    elapsed = time.perf_counter() - t0
    print(f"Loaded: {CHAT_MODEL.name} (chat, load: {elapsed:.2f}s)")

    models = [EMBED_MODEL.name, CHAT_MODEL.name]

    # ── 1. Side-by-side tokenization comparison ──────────────────
    print(f"\n{'=' * 70}")
    print("  1. Tokenizer Comparison: BERT WordPiece vs BPE")
    print(f"{'=' * 70}")

    for label, text in SAMPLES:
        print(f"\n  [{label}] {text}")
        for name in models:
            pieces = json.loads(db.execute("SELECT muninn_tokenize_text(?, ?)", (name, text)).fetchone()[0])
            count = db.execute("SELECT muninn_token_count(?, ?)", (name, text)).fetchone()[0]
            print(f"    {name} ({count} tokens): {pieces}")

    # ── 2. Detailed token mapping on a complex paragraph ────────
    #
    # Long medical/scientific compound words force heavy subword
    # decomposition. Shows how each tokenizer breaks apart words
    # like "neuropsychopharmacologist" and "electroencephalography".

    detail_text = (
        "The neuropsychopharmacologist at Massachusetts General Hospital published a groundbreaking "
        "paper on electroencephalography biomarkers for antidepressant responsiveness. Her "
        "counterintuitive findings about deoxyribonucleic acid methylation patterns were "
        "simultaneously praised by gastroenterologists and otorhinolaryngologists at the "
        "Czech Otolaryngological Society's conference on immunohistochemistry."
    )

    print(f"\n{'=' * 70}")
    print("  2. Token ID ↔ Piece Mapping (compound word decomposition)")
    print(f"{'=' * 70}")
    print(f"\n  {detail_text}")

    for name in models:
        ids = json.loads(db.execute("SELECT muninn_tokenize(?, ?)", (name, detail_text)).fetchone()[0])
        pieces = json.loads(db.execute("SELECT muninn_tokenize_text(?, ?)", (name, detail_text)).fetchone()[0])
        count = db.execute("SELECT muninn_token_count(?, ?)", (name, detail_text)).fetchone()[0]
        print(f"\n  {name} ({count} tokens): {pieces}")
        print(f"  {'Pos':>4}  {'ID':>8}  Piece")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 30}")
        for i, (tid, piece) in enumerate(zip(ids, pieces, strict=True)):
            print(f"  {i:>4}  {tid:>8}  {piece!r}")

    # ── 3. SQL: token-level query with json_each ─────────────────
    print(f"\n{'=' * 70}")
    print("  3. SQL: Per-Token Query via json_each()")
    print(f"{'=' * 70}")
    print("\n  Same paragraph, queried in pure SQL.")
    print(f"  Model: {CHAT_MODEL.name}")

    rows = db.execute(
        """SELECT ids.key AS pos, ids.value AS token_id, texts.value AS piece
           FROM json_each(muninn_tokenize(?, ?)) AS ids
           JOIN json_each(muninn_tokenize_text(?, ?)) AS texts ON texts.key = ids.key""",
        (CHAT_MODEL.name, detail_text, CHAT_MODEL.name, detail_text),
    ).fetchall()
    print(f"\n  {'Pos':>4}  {'ID':>8}  Piece")
    print(f"  {'─' * 4}  {'─' * 8}  {'─' * 30}")
    for pos, token_id, piece in rows:
        print(f"  {pos:>4}  {token_id:>8}  {piece!r}")

    # Count how many tokens are subword fragments (don't start with a space or special char)
    subword = [r for r in rows if r[2] and r[2][0] not in (' ', '.', ',', "'", '"')]
    print(f"\n  {len(subword)} of {len(rows)} tokens are subword fragments (no leading space).")

    # ── 4. Token efficiency comparison ────────────────────────────
    print(f"\n{'=' * 70}")
    print("  4. Token Efficiency: chars/token ratio by text type")
    print(f"{'=' * 70}")

    print(f"\n  {'Type':<15} {'Chars':>6}  ", end="")
    for name in models:
        print(f"  {name:>14} (c/t)", end="")
    print()
    print(f"  {'─' * 15} {'─' * 6}  ", end="")
    for _ in models:
        print(f"  {'─' * 20}", end="")
    print()

    for label, text in SAMPLES:
        n_chars = len(text)
        print(f"  {label:<15} {n_chars:>6}  ", end="")
        for name in models:
            count = db.execute("SELECT muninn_token_count(?, ?)", (name, text)).fetchone()[0]
            ratio = n_chars / count if count > 0 else 0
            print(f"  {count:>6} ({ratio:>4.1f})", end="       ")
        print()


if __name__ == "__main__":
    main()
