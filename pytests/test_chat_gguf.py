"""Integration tests for llama_chat.c SQL functions.

Tests registration and error handling of chat functions. Tests requiring
a real GGUF model are gated behind the MUNINN_CHAT_MODEL env var:

    MUNINN_CHAT_MODEL=models/Qwen3-4B-Q4_K_M.gguf pytest pytests/test_chat_gguf.py -v
"""

import json
import os

import pysqlite3 as sqlite3
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def conn(request):
    """Fresh in-memory SQLite connection with muninn loaded."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    conftest = request.config  # noqa: F841
    # Extension path is set by conftest.py's auto-build fixture
    from pathlib import Path

    ext_path = str(Path(__file__).resolve().parent.parent / "build" / "muninn")
    db.load_extension(ext_path)
    yield db
    db.close()


# ── Registration Tests ────────────────────────────────────────────────


def test_chat_functions_registered(conn):
    """All chat SQL functions should be registered and callable."""
    # muninn_chat — should fail with "model not loaded", not "no such function"
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_chat('nonexistent', 'hello')").fetchone()


def test_extract_entities_registered(conn):
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_extract_entities('nonexistent', 'text', 'person')").fetchone()


def test_extract_entities_unsupervised_registered(conn):
    """Unsupervised NER (2-arg form) should pass arg validation and reach model lookup."""
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_extract_entities('nonexistent', 'text')").fetchone()


def test_extract_relations_registered(conn):
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_extract_relations('nonexistent', 'text', '[]')").fetchone()


def test_extract_relations_unsupervised_registered(conn):
    """Unsupervised RE (2-arg form) should pass arg validation and reach model lookup."""
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_extract_relations('nonexistent', 'text')").fetchone()


def test_extract_ner_re_unsupervised_registered(conn):
    """Unsupervised NER+RE (2-arg form) should pass arg validation and reach model lookup."""
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_extract_ner_re('nonexistent', 'text')").fetchone()


def test_extract_ner_re_mixed_labels_error(conn):
    """NER+RE with only entity labels (no relation labels) should error."""
    with pytest.raises(sqlite3.OperationalError, match="requires both"):
        conn.execute("SELECT muninn_extract_ner_re('nonexistent', 'text', 'person')").fetchone()


def test_summarize_registered(conn):
    with pytest.raises(sqlite3.OperationalError, match="not loaded"):
        conn.execute("SELECT muninn_summarize('nonexistent', 'text')").fetchone()


# ── Chat Models VT Tests ─────────────────────────────────────────────


def test_chat_models_vtab_empty(conn):
    """The muninn_chat_models VT should exist and be empty initially."""
    rows = conn.execute("SELECT name, n_ctx FROM muninn_chat_models").fetchall()
    assert rows == []


def test_chat_model_bad_path(conn):
    """Loading a nonexistent GGUF file should fail gracefully."""
    with pytest.raises(sqlite3.OperationalError, match="failed to load"):
        conn.execute("SELECT muninn_chat_model('/nonexistent/bad.gguf')").fetchone()


def test_chat_model_with_ctx_arg(conn):
    """muninn_chat_model accepts an optional n_ctx argument."""
    with pytest.raises(sqlite3.OperationalError, match="failed to load"):
        conn.execute("SELECT muninn_chat_model('/nonexistent/bad.gguf', 4096)").fetchone()


# ── Model-Required Tests ─────────────────────────────────────────────
# These require a real GGUF chat model. Set env var to enable:
#   MUNINN_CHAT_MODEL=models/Qwen3-4B-Q4_K_M.gguf pytest pytests/test_chat_gguf.py -v


@pytest.fixture
def chat_model_path():
    path = os.environ.get("MUNINN_CHAT_MODEL")
    if path is None:
        pytest.skip("MUNINN_CHAT_MODEL env var not set")
    return path


@pytest.fixture
def conn_with_model(conn, chat_model_path):
    """Connection with a chat model loaded."""
    conn.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT 'test', muninn_chat_model(?)",
        (chat_model_path,),
    )
    return conn


def test_chat_plain(conn_with_model):
    """Basic chat completion should return non-empty text."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_chat('test', 'What is 2+2? Answer with just the number.')"
    ).fetchone()
    assert result is not None
    assert len(result) > 0
    assert "4" in result


def test_extract_entities_json(conn_with_model):
    """NER should return valid JSON with entities array."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_entities('test', 'Alice works at ACME in New York.', 'person,organization,location')"
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "entities" in parsed
    assert isinstance(parsed["entities"], list)
    assert len(parsed["entities"]) > 0

    # Check entity structure
    for ent in parsed["entities"]:
        assert "text" in ent
        assert "type" in ent


def test_extract_relations_json(conn_with_model):
    """RE should return valid JSON with relations array."""
    entities_json = '[{"text":"Alice","type":"person"},{"text":"ACME","type":"organization"}]'
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_relations('test', 'Alice founded ACME.', ?)",
        (entities_json,),
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "relations" in parsed
    assert isinstance(parsed["relations"], list)


def test_summarize(conn_with_model):
    """Summarisation should return non-empty text."""
    context = "Alice Smith is the founder of ACME Corp. ACME is located in New York."
    (result,) = conn_with_model.execute("SELECT muninn_summarize('test', ?, 64)", (context,)).fetchone()
    assert result is not None
    assert len(result) > 0


def test_bulk_ner_re_pipeline(conn_with_model):
    """NER -> RE chained in SQL should produce valid JSON output."""
    db = conn_with_model
    db.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
    db.execute("INSERT INTO docs VALUES (1, 'Alice founded ACME Corporation in New York.')")

    results = db.execute(
        """
        WITH ner AS (
            SELECT id, content,
                   muninn_extract_entities('test', content, 'person,organization,location') AS ents
            FROM docs
        )
        SELECT id,
               muninn_extract_relations('test', content, ents) AS rels
        FROM ner
        """
    ).fetchall()

    assert len(results) == 1
    doc_id, rels_json = results[0]
    assert doc_id == 1
    parsed = json.loads(rels_json)
    assert "relations" in parsed


# ── Unsupervised Mode Integration Tests ──────────────────────────


def test_extract_entities_unsupervised(conn_with_model):
    """Unsupervised NER should return valid JSON with entities."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_entities('test', 'Alice works at ACME in New York.')"
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "entities" in parsed
    assert isinstance(parsed["entities"], list)
    assert len(parsed["entities"]) > 0

    for ent in parsed["entities"]:
        assert "text" in ent
        assert "type" in ent


def test_extract_relations_unsupervised(conn_with_model):
    """Unsupervised RE should return valid JSON with relations."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_relations('test', 'Alice founded ACME Corporation in 1987.')"
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "relations" in parsed
    assert isinstance(parsed["relations"], list)


def test_extract_ner_re_unsupervised(conn_with_model):
    """Unsupervised NER+RE should return valid JSON with both entities and relations."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_ner_re('test', 'Alice founded ACME Corporation in New York City in 1987.')"
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "entities" in parsed
    assert "relations" in parsed
    assert isinstance(parsed["entities"], list)
    assert isinstance(parsed["relations"], list)
    assert len(parsed["entities"]) > 0


def test_extract_entities_unsupervised_with_skip_think(conn_with_model):
    """Unsupervised NER with inject_skip_think=1 should return valid JSON."""
    (result,) = conn_with_model.execute(
        "SELECT muninn_extract_entities('test', 'Alice works at ACME in New York.', 1)"
    ).fetchone()
    assert result is not None
    parsed = json.loads(result)
    assert "entities" in parsed
    assert isinstance(parsed["entities"], list)
