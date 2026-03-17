"""
LLM Chat — Free-form and structured chat completion via muninn_chat()

Demonstrates:
  1. Plain chat completion (with <think> block separation)
  2. Grammar-constrained structured JSON output via GBNF

muninn_chat(model, prompt [, grammar [, max_tokens]]) returns raw model
output. Qwen3 models emit <think>...</think> reasoning blocks before the
actual response — this example shows how to separate them.

Requirements:
  - muninn extension (make all)
  - GGUF chat models (auto-downloaded on first run)
"""

import json
import logging
import re
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


MODELS = [
    GgufModel(
        "Qwen3.5-0.8B",
        "Qwen3.5-0.8B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
        0.53,
    ),
    GgufModel(
        "Qwen3.5-2B",
        "Qwen3.5-2B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
        1.28,
    ),
]

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what a knowledge graph is in two sentences.",
]

# GBNF grammar for structured Q&A output:
# {"answer": "...", "confidence": 0.95}
GBNF_QA = r"""
root ::= "{" ws "\"answer\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
string ::= "\"" [^"\\]* "\""
number ::= [0-9] ("." [0-9]+)?
ws ::= [ \t\n]*
""".strip()

_THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def split_think(raw: str) -> tuple[str, str]:
    """Split <think>...</think> reasoning from the actual response.

    Returns (thinking, response). If no think block, thinking is empty.
    """
    m = _THINK_RE.search(raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", raw.strip()


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

    # Load models
    for model in MODELS:
        ensure_model(model)
        path = MODELS_DIR / model.filename
        t0 = time.perf_counter()
        db.execute(
            "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
            (model.name, str(path)),
        )
        elapsed = time.perf_counter() - t0
        n_ctx = db.execute("SELECT n_ctx FROM muninn_chat_models WHERE name = ?", (model.name,)).fetchone()[0]
        print(f"Loaded: {model.name} (context: {n_ctx}, load: {elapsed:.2f}s)")

    # ── 1. Plain chat — raw output with <think> separation ───────
    print(f"\n{'=' * 60}")
    print("  1. Plain Chat (with <think> block separation)")
    print(f"{'=' * 60}")

    for prompt in PROMPTS:
        print(f"\n  Prompt: {prompt}")
        print(f"  {'─' * 56}")

        for model in MODELS:
            t0 = time.perf_counter()
            raw = db.execute("SELECT muninn_chat(?, ?)", (model.name, prompt)).fetchone()[0]
            elapsed = time.perf_counter() - t0

            thinking, response = split_think(raw)
            print(f"\n  {model.name} ({elapsed:.2f}s):")
            if thinking:
                print(f"    [thinking] {thinking}")
            if response:
                print(f"    [response] {response}")
            else:
                print("    [response] (empty — model exhausted tokens on reasoning)")

    # ── 2. Grammar-constrained → structured JSON ─────────────────
    print(f"\n{'=' * 60}")
    print("  2. Grammar-Constrained Chat (GBNF → JSON)")
    print(f"{'=' * 60}")
    print("  GBNF forces output to match: {\"answer\": \"...\", \"confidence\": 0.95}")

    structured_prompts = [
        "What is the capital of France?",
        "What year did World War II end?",
        "Who wrote Romeo and Juliet?",
    ]

    for prompt in structured_prompts:
        print(f"\n  Prompt: {prompt}")

        for model in MODELS:
            t0 = time.perf_counter()
            raw = db.execute(
                "SELECT muninn_chat(?, ?, ?)",
                (model.name, prompt, GBNF_QA),
            ).fetchone()[0]
            elapsed = time.perf_counter() - t0

            # GBNF guarantees valid JSON — no think blocks possible
            parsed = json.loads(raw)
            print(f"    {model.name}: answer={parsed['answer']!r}  confidence={parsed['confidence']}  ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
