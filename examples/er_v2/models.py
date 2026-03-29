"""GGUF model configurations, downloading, and SQLite registration."""

import logging
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


@dataclass
class EmbedModelConfig:
    name: str
    filename: str
    url: str
    prefix: str = ""  # Task prefix prepended to text before embedding (e.g., "clustering: " for Nomic)


@dataclass
class ChatModelConfig:
    name: str
    filename: str
    url: str
    size_gb: float


# ── Embedding Models ──────────────────────────────────────────────

EMBED_MODELS: dict[str, EmbedModelConfig] = {
    "MiniLM": EmbedModelConfig(
        name="MiniLM",
        filename="all-MiniLM-L6-v2.Q8_0.gguf",
        url="https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
    ),
    "NomicEmbed": EmbedModelConfig(
        name="NomicEmbed",
        filename="nomic-embed-text-v1.5.Q8_0.gguf",
        url="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf",
        prefix="clustering: ",  # Symmetric clustering prefix per Nomic docs
    ),
}

DEFAULT_EMBED_MODEL = "MiniLM"

# ── Chat Models ───────────────────────────────────────────────────

CHAT_MODELS: dict[str, ChatModelConfig] = {
    "Qwen3.5-2B": ChatModelConfig(
        "Qwen3.5-2B",
        "Qwen3.5-2B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
        1.28,
    ),
    "Qwen3.5-4B": ChatModelConfig(
        "Qwen3.5-4B",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        2.7,
    ),
    "Gemma-3-1B": ChatModelConfig(
        "Gemma-3-1B",
        "google_gemma-3-1b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf",
        0.81,
    ),
    "Gemma-3-4B": ChatModelConfig(
        "Gemma-3-4B",
        "google_gemma-3-4b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
        2.5,
    ),
}


# ── Download / Registration ───────────────────────────────────────


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


def ensure_model(model: EmbedModelConfig | ChatModelConfig) -> Path:
    """Download GGUF model if not already present. Returns path."""
    path = MODELS_DIR / model.filename
    if path.exists():
        return path
    log.info("Downloading %s...", model.filename)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_with_progress(model.url, path)
    log.info("Downloaded %s (%.1f MB)", model.filename, path.stat().st_size / 1e6)
    return path


def create_db(embed_model_name: str = DEFAULT_EMBED_MODEL) -> sqlite3.Connection:
    """Create in-memory SQLite with muninn loaded and embedding model registered."""
    embed_model = EMBED_MODELS[embed_model_name]
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    ensure_model(embed_model)
    conn.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
        (embed_model.name, str(MODELS_DIR / embed_model.filename)),
    )
    log.info("Loaded embedding model: %s", embed_model.name)
    return conn


def register_chat_model(conn: sqlite3.Connection, model: ChatModelConfig) -> None:
    """Load and register a GGUF chat model into an existing connection."""
    ensure_model(model)
    conn.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
        (model.name, str(MODELS_DIR / model.filename)),
    )
    log.info("Loaded chat model: %s", model.name)


def cleanup_pipeline_tables(conn: sqlite3.Connection) -> None:
    """Drop pipeline tables so the connection can be reused."""
    for table in ["entities", "entity_vecs", "_match_edges"]:
        conn.execute(f"DROP TABLE IF EXISTS [{table}]")
