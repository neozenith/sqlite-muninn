"""Tests for the constants module."""

from __future__ import annotations

from benchmarks.demo_builder.constants import (
    BOOK_ID_TO_DATASET,
    EMBEDDING_MODELS,
    GLINER_LABELS,
    GLIREL_LABELS,
    MUNINN_PATH,
    PHASE_NAMES,
    PROJECT_ROOT,
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
