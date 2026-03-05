"""Prep module: download GGUF embedding models.

Ensures GGUF model files are available in the project's models/ directory.
These are used by muninn's native embed_gguf subsystem (muninn_embed_model /
muninn_embed SQL functions) for embedding benchmarks and examples.

Model definitions are authoritative here and mirrored in
examples/text_embeddings/example.py.
"""

import logging
import urllib.request
from pathlib import Path

from benchmarks.harness.common import GGUF_MODELS_DIR
from benchmarks.harness.prep.base import PrepTask
from benchmarks.harness.prep.common import fmt_size

log = logging.getLogger(__name__)


# ── Model catalog ────────────────────────────────────────────────

GGUF_MODELS: list[dict[str, str]] = [
    {
        "name": "MiniLM",
        "filename": "all-MiniLM-L6-v2.Q8_0.gguf",
        "url": "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
        "params": "22M",
        "dim": "384",
        "quant": "Q8_0",
        "doc_prefix": "",
        "query_prefix": "",
    },
    {
        "name": "NomicEmbed",
        "filename": "nomic-embed-text-v1.5.Q8_0.gguf",
        "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf",
        "params": "137M",
        "dim": "768",
        "quant": "Q8_0",
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    {
        "name": "BGE-Large",
        "filename": "bge-large-en-v1.5-q8_0.gguf",
        "url": "https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/resolve/main/bge-large-en-v1.5-q8_0.gguf",
        "params": "335M",
        "dim": "1024",
        "quant": "Q8_0",
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
]


# ── PrepTask ─────────────────────────────────────────────────────


class GGUFModelPrepTask(PrepTask):
    """PrepTask for downloading a single GGUF embedding model."""

    def __init__(self, model: dict[str, str]):
        self._model = model

    @property
    def task_id(self) -> str:
        return f"gguf:{self._model['name']}"

    @property
    def label(self) -> str:
        return f"{self._model['name']} ({self._model['params']}, dim={self._model['dim']}, {self._model['quant']})"

    def outputs(self) -> list[Path]:
        return [GGUF_MODELS_DIR / self._model["filename"]]

    def fetch(self, force: bool = False) -> None:
        path = GGUF_MODELS_DIR / self._model["filename"]

        if path.exists() and not force:
            log.info("  %s: cached at %s", self._model["name"], path)
            return

        if path.exists() and force:
            log.info("  %s: --force, re-downloading", self._model["name"])
            path.unlink()

        GGUF_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        url = self._model["url"]
        log.info("  Downloading %s from %s...", self._model["filename"], url)

        req = urllib.request.Request(url, headers={"User-Agent": "muninn-benchmark/1.0"})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 256 * 1024

            with path.open("wb") as f:
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
                        print(
                            f"\r  {self._model['filename']}: {mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )

            if total > 0:
                print()

        log.info("  %s: saved %.1f MB to %s", self._model["name"], path.stat().st_size / 1e6, path)


GGUF_PREP_TASKS: list[PrepTask] = [GGUFModelPrepTask(m) for m in GGUF_MODELS]


# ── Status display ───────────────────────────────────────────────


def print_status() -> None:
    """Print status of all GGUF model files."""
    print("=== GGUF Model Status ===\n")
    print(f"  {'MODEL':<16s}  {'FILE':<40s}  {'SIZE':>10s}  {'STATUS'}")
    print(f"  {'-' * 16}  {'-' * 40}  {'-' * 10}  {'-' * 7}")

    for model in GGUF_MODELS:
        path = GGUF_MODELS_DIR / model["filename"]
        if path.exists():
            size = fmt_size(path.stat().st_size)
            print(f"  {model['name']:<16s}  {model['filename']:<40s}  {size:>10s}  READY")
        else:
            print(f"  {model['name']:<16s}  {model['filename']:<40s}  {'':>10s}  MISSING")

    print(f"\n  Directory: {GGUF_MODELS_DIR}")
    print()


# ── Main entry point ─────────────────────────────────────────────


def prep_gguf(model_name: str | None = None, status_only: bool = False, force: bool = False) -> None:
    """Download GGUF embedding models to the project models/ directory.

    Args:
        model_name: Specific model name to download (e.g., 'MiniLM'). If None, downloads all.
        status_only: If True, show status and return.
        force: If True, re-download even if files exist.
    """
    if status_only:
        print_status()
        return

    if model_name:
        matched = [t for t in GGUF_PREP_TASKS if t._model["name"] == model_name]  # type: ignore[attr-defined]
        if not matched:
            valid = [m["name"] for m in GGUF_MODELS]
            log.error("Unknown GGUF model: %s. Available: %s", model_name, ", ".join(valid))
            return
        tasks = matched
    else:
        tasks = list(GGUF_PREP_TASKS)

    log.info("Downloading %d GGUF model(s)...", len(tasks))
    for task in tasks:
        task.run(force=force)

    log.info("GGUF prep complete. Models cached in %s", GGUF_MODELS_DIR)
