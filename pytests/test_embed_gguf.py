"""Tests for GGUF embedding functions via llama.cpp integration.

Uses a real GGUF model (all-MiniLM-L6-v2.Q8_0.gguf, ~36 MB) that is
auto-downloaded on first run. Tests cover:
  - Model lifecycle (load/unload via muninn_models VT)
  - Embedding generation and format validation
  - Semantic similarity ordering
  - Tokenization
  - Error handling
  - End-to-end HNSW integration
"""

import json
import math
import struct
import urllib.request
from pathlib import Path

import pytest

import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
GGUF_MODEL_PATH = MODELS_DIR / "all-MiniLM-L6-v2.Q8_0.gguf"
GGUF_MODEL_URL = "https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf"
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


@pytest.fixture(scope="session")
def gguf_model_path() -> Path:
    """Ensure the GGUF model is available, downloading if needed."""
    if not GGUF_MODEL_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GGUF model to {GGUF_MODEL_PATH}...")
        urllib.request.urlretrieve(GGUF_MODEL_URL, GGUF_MODEL_PATH)
        print(f"Downloaded {GGUF_MODEL_PATH.stat().st_size / 1e6:.1f} MB")
    return GGUF_MODEL_PATH


@pytest.fixture
def conn(gguf_model_path: Path) -> sqlite3.Connection:
    """Fresh in-memory connection with extension and model loaded."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    # Register the model
    db.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT 'MiniLM', muninn_embed_model(?)",
        (str(gguf_model_path),),
    )
    yield db
    # Unload model from process-global C registry before closing —
    # g_models[] persists across connections so stale entries cause
    # "already loaded" errors in subsequent fixtures.
    db.execute("DELETE FROM temp.muninn_models WHERE name = 'MiniLM'")
    db.close()


@pytest.fixture
def conn_no_model() -> sqlite3.Connection:
    """Fresh in-memory connection with extension loaded, no model."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    yield db
    db.close()


def blob_to_floats(blob: bytes) -> list[float]:
    """Convert a float32 blob to a list of Python floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Model Lifecycle Tests ─────────────────────────────────────


@pytest.mark.gguf
class TestModelLifecycle:
    def test_model_load_and_query(self, conn: sqlite3.Connection) -> None:
        """Model should appear in muninn_models after INSERT."""
        rows = conn.execute("SELECT name, dim FROM muninn_models").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "MiniLM"
        assert rows[0][1] == 384

    def test_model_dim(self, conn: sqlite3.Connection) -> None:
        """muninn_model_dim should return 384 for MiniLM."""
        (dim,) = conn.execute("SELECT muninn_model_dim('MiniLM')").fetchone()
        assert dim == 384

    def test_model_unload(self, conn: sqlite3.Connection) -> None:
        """DELETE from muninn_models should unload the model."""
        conn.execute("DELETE FROM temp.muninn_models WHERE name = 'MiniLM'")
        rows = conn.execute("SELECT name FROM muninn_models").fetchall()
        assert len(rows) == 0

    def test_model_load_bad_path(self, conn_no_model: sqlite3.Connection) -> None:
        """Loading a nonexistent GGUF file should raise an error."""
        with pytest.raises(sqlite3.OperationalError, match="failed to load"):
            conn_no_model.execute(
                "INSERT INTO temp.muninn_models(name, model) "
                "SELECT 'bad', muninn_embed_model('/nonexistent/model.gguf')"
            )

    def test_duplicate_model_name(self, conn: sqlite3.Connection, gguf_model_path: Path) -> None:
        """Loading a model with a duplicate name should raise an error."""
        with pytest.raises(sqlite3.OperationalError, match="already loaded"):
            conn.execute(
                "INSERT INTO temp.muninn_models(name, model) SELECT 'MiniLM', muninn_embed_model(?)",
                (str(gguf_model_path),),
            )


# ─── Embedding Tests ──────────────────────────────────────────


@pytest.mark.gguf
class TestEmbedding:
    def test_embedding_format(self, conn: sqlite3.Connection) -> None:
        """muninn_embed should return a float32 blob of correct size."""
        (blob,) = conn.execute("SELECT muninn_embed('MiniLM', 'hello world')").fetchone()
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # 384 dims * 4 bytes per float32

    def test_embedding_normalized(self, conn: sqlite3.Connection) -> None:
        """Embedding vectors should be L2-normalized (magnitude ≈ 1.0)."""
        (blob,) = conn.execute("SELECT muninn_embed('MiniLM', 'the quick brown fox')").fetchone()
        floats = blob_to_floats(blob)
        magnitude = math.sqrt(sum(x * x for x in floats))
        assert abs(magnitude - 1.0) < 0.01

    def test_semantic_similarity(self, conn: sqlite3.Connection) -> None:
        """Similar texts should have higher cosine similarity than dissimilar ones."""
        texts = [
            "The cat sat on the mat",
            "A kitten was sitting on a rug",
            "Quantum mechanics describes subatomic particles",
        ]
        embeddings = []
        for text in texts:
            (blob,) = conn.execute("SELECT muninn_embed('MiniLM', ?)", (text,)).fetchone()
            embeddings.append(blob_to_floats(blob))

        sim_similar = cosine_similarity(embeddings[0], embeddings[1])
        sim_dissimilar = cosine_similarity(embeddings[0], embeddings[2])

        # Cat/kitten should be more similar than cat/quantum
        assert sim_similar > sim_dissimilar

    def test_embed_unloaded_model(self, conn_no_model: sqlite3.Connection) -> None:
        """Embedding with an unloaded model should raise an error."""
        with pytest.raises(sqlite3.OperationalError, match="not loaded"):
            conn_no_model.execute("SELECT muninn_embed('nonexistent', 'hello')")


# ─── Tokenization Tests ──────────────────────────────────────


@pytest.mark.gguf
class TestTokenization:
    def test_tokenize_returns_json(self, conn: sqlite3.Connection) -> None:
        """muninn_tokenize should return a valid JSON array of integers."""
        (result,) = conn.execute("SELECT muninn_tokenize('MiniLM', 'hello world')").fetchone()
        tokens = json.loads(result)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_longer_text(self, conn: sqlite3.Connection) -> None:
        """Longer text should produce more tokens."""
        (short_result,) = conn.execute("SELECT muninn_tokenize('MiniLM', 'hi')").fetchone()
        (long_result,) = conn.execute(
            "SELECT muninn_tokenize('MiniLM', 'the quick brown fox jumps over the lazy dog')"
        ).fetchone()
        short_tokens = json.loads(short_result)
        long_tokens = json.loads(long_result)
        assert len(long_tokens) > len(short_tokens)

    def test_tokenize_unloaded_model(self, conn_no_model: sqlite3.Connection) -> None:
        """Tokenizing with an unloaded model should raise an error."""
        with pytest.raises(sqlite3.OperationalError, match="not loaded"):
            conn_no_model.execute("SELECT muninn_tokenize('nonexistent', 'hello')")


# ─── Tokenize Text Tests ─────────────────────────────────────


@pytest.mark.gguf
class TestTokenizeText:
    def test_tokenize_text_returns_json_strings(self, conn: sqlite3.Connection) -> None:
        """muninn_tokenize_text should return a JSON array of strings."""
        (result,) = conn.execute("SELECT muninn_tokenize_text('MiniLM', 'hello world')").fetchone()
        pieces = json.loads(result)
        assert isinstance(pieces, list)
        assert len(pieces) > 0
        assert all(isinstance(p, str) for p in pieces)

    def test_tokenize_text_matches_tokenize_length(self, conn: sqlite3.Connection) -> None:
        """Token text pieces should have same count as token IDs."""
        text = "the quick brown fox jumps over the lazy dog"
        (ids_json,) = conn.execute("SELECT muninn_tokenize('MiniLM', ?)", (text,)).fetchone()
        (text_json,) = conn.execute("SELECT muninn_tokenize_text('MiniLM', ?)", (text,)).fetchone()
        ids = json.loads(ids_json)
        pieces = json.loads(text_json)
        assert len(pieces) == len(ids)

    def test_tokenize_text_pieces_reconstruct(self, conn: sqlite3.Connection) -> None:
        """Joining token text pieces should approximately reconstruct the input."""
        text = "hello world"
        (result,) = conn.execute("SELECT muninn_tokenize_text('MiniLM', ?)", (text,)).fetchone()
        pieces = json.loads(result)
        # Filter out special tokens (BOS/EOS) which are typically [CLS], [SEP], etc.
        non_special = [p for p in pieces if not p.startswith("[")]
        reconstructed = "".join(non_special).strip()
        assert text in reconstructed or reconstructed in text or reconstructed.lower() == text.lower()

    def test_tokenize_text_unloaded_model(self, conn_no_model: sqlite3.Connection) -> None:
        """Tokenize text with an unloaded model should raise an error."""
        with pytest.raises(sqlite3.OperationalError, match="not loaded"):
            conn_no_model.execute("SELECT muninn_tokenize_text('nonexistent', 'hello')")

    def test_tokenize_text_json_each_compat(self, conn: sqlite3.Connection) -> None:
        """Result should work with json_each() for per-token SQL queries."""
        rows = conn.execute("SELECT value FROM json_each(muninn_tokenize_text('MiniLM', 'hello world'))").fetchall()
        assert len(rows) > 0
        assert all(isinstance(row[0], str) for row in rows)


# ─── End-to-End HNSW Integration ─────────────────────────────


@pytest.mark.gguf
class TestHNSWIntegration:
    def test_embed_into_hnsw_and_search(self, conn: sqlite3.Connection) -> None:
        """Embeddings should insert into hnsw_index and be searchable."""
        # Create HNSW index with matching dimensions
        conn.execute("CREATE VIRTUAL TABLE test_vectors USING hnsw_index(  dimensions=384, metric=cosine)")

        # Insert embeddings
        texts = [
            (1, "dogs are loyal pets"),
            (2, "cats are independent animals"),
            (3, "the stock market crashed today"),
            (4, "puppies love to play fetch"),
        ]

        for rowid, text in texts:
            conn.execute(
                "INSERT INTO test_vectors(rowid, vector) VALUES (?, muninn_embed('MiniLM', ?))",
                (rowid, text),
            )

        # Search for dog-related content
        results = conn.execute(
            "SELECT rowid, distance FROM test_vectors "
            "WHERE vector MATCH muninn_embed('MiniLM', 'canine companion') "
            "AND k = 4 "
            "ORDER BY distance",
        ).fetchall()

        assert len(results) == 4

        # Dog-related results (1, 4) should rank higher than stock market (3)
        rowids = [r[0] for r in results]
        dog_ranks = [rowids.index(r) for r in [1, 4]]
        stock_rank = rowids.index(3)
        assert all(dr < stock_rank for dr in dog_ranks)

    def test_blob_format_compatibility(self, conn: sqlite3.Connection) -> None:
        """muninn_embed output should be directly insertable into hnsw_index."""
        conn.execute("CREATE VIRTUAL TABLE compat_test USING hnsw_index(  dimensions=384, metric=cosine)")

        # This should work without any format conversion
        conn.execute("INSERT INTO compat_test(rowid, vector) VALUES (1, muninn_embed('MiniLM', 'test text'))")

        # Verify it's searchable
        results = conn.execute(
            "SELECT rowid FROM compat_test WHERE vector MATCH muninn_embed('MiniLM', 'test text') AND k = 1"
        ).fetchall()
        assert len(results) == 1
        assert results[0][0] == 1
