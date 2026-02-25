"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle."""

from __future__ import annotations

from benchmarks.demo_builder.phases.base import Phase
from benchmarks.demo_builder.phases.chunks import PhaseChunks
from benchmarks.demo_builder.phases.entity_embeddings import PhaseEntityEmbeddings
from benchmarks.demo_builder.phases.entity_resolution import PhaseEntityResolution
from benchmarks.demo_builder.phases.metadata import PhaseMetadata
from benchmarks.demo_builder.phases.ner import PhaseNER
from benchmarks.demo_builder.phases.node2vec import PhaseNode2Vec
from benchmarks.demo_builder.phases.re import PhaseRE
from benchmarks.demo_builder.phases.umap import PhaseUMAP

__all__ = [
    "Phase",
    "PhaseChunks",
    "PhaseEntityEmbeddings",
    "PhaseEntityResolution",
    "PhaseMetadata",
    "PhaseNER",
    "PhaseNode2Vec",
    "PhaseRE",
    "PhaseUMAP",
    "default_phases",
]


def default_phases(book_id: int, model_name: str) -> list[Phase]:
    """Create the default ordered list of build phases."""
    return [
        PhaseChunks(book_id, model_name),
        PhaseNER(),
        PhaseRE(),
        PhaseEntityEmbeddings(model_name),
        PhaseUMAP(),
        PhaseEntityResolution(),
        PhaseNode2Vec(),
        PhaseMetadata(book_id, model_name),
    ]
