"""Tests for the constants module."""

from __future__ import annotations

from benchmarks.demo_builder.constants import (
    BOOK_ID_TO_DATASET,
    CHARS_PER_WORD_TOKEN,
    EMBEDDING_MODELS,
    GLINER_LABELS,
    GLIREL_LABELS,
    MUNINN_PATH,
    NER_MAX_TOKENS,
    NER_RE_CHUNK_CHARS_MAX,
    PHASE_NAMES,
    PROJECT_ROOT,
    RE_MAX_TOKENS,
)


def test_project_root_is_absolute() -> None:
    assert PROJECT_ROOT.is_absolute()


def test_project_root_contains_makefile() -> None:
    assert (PROJECT_ROOT / "Makefile").exists()


def test_muninn_path_is_under_build() -> None:
    assert "build/muninn" in MUNINN_PATH


def test_embedding_models_have_required_keys() -> None:
    for name, info in EMBEDDING_MODELS.items():
        assert "st_name" in info, f"{name} missing st_name"
        assert "dim" in info, f"{name} missing dim"
        assert isinstance(info["dim"], int)
        assert info["dim"] > 0


def test_phase_names_count() -> None:
    assert len(PHASE_NAMES) == 8


def test_gliner_labels_nonempty() -> None:
    assert len(GLINER_LABELS) > 0
    assert all(isinstance(label, str) for label in GLINER_LABELS)


def test_glirel_labels_nonempty() -> None:
    assert len(GLIREL_LABELS) > 0
    assert all(isinstance(label, str) for label in GLIREL_LABELS)


def test_book_id_to_dataset_values() -> None:
    for book_id, slug in BOOK_ID_TO_DATASET.items():
        assert isinstance(book_id, int)
        assert isinstance(slug, str)
        assert len(slug) > 0


def test_ner_re_chunk_chars_max_is_reasonable() -> None:
    """NER_RE_CHUNK_CHARS_MAX is a sensible value derived from model limits."""
    assert NER_RE_CHUNK_CHARS_MAX == int(min(NER_MAX_TOKENS, RE_MAX_TOKENS) * CHARS_PER_WORD_TOKEN)
    assert NER_RE_CHUNK_CHARS_MAX > 500, "Cap too small — would produce tiny chunks"
    assert NER_RE_CHUNK_CHARS_MAX < 3000, "Cap too large — NER/RE models would truncate"


def test_ner_re_cap_affects_large_models_only() -> None:
    """Models with chunk_chars <= NER_RE_CHUNK_CHARS_MAX are unaffected by the cap."""
    for name, info in EMBEDDING_MODELS.items():
        effective = min(info["chunk_chars"], NER_RE_CHUNK_CHARS_MAX)
        if info["chunk_chars"] <= NER_RE_CHUNK_CHARS_MAX:
            assert effective == info["chunk_chars"], f"{name} should not be capped"
        else:
            assert effective == NER_RE_CHUNK_CHARS_MAX, f"{name} should be capped"
