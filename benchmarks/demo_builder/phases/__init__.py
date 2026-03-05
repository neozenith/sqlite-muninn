"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle."""

from __future__ import annotations

from benchmarks.demo_builder.phases.base import Phase
from benchmarks.demo_builder.phases.chunks import PhaseChunks
from benchmarks.demo_builder.phases.chunks_embeddings import PhaseChunksEmbeddings
from benchmarks.demo_builder.phases.entity_embeddings import PhaseEntityEmbeddings
from benchmarks.demo_builder.phases.entity_resolution import PhaseEntityResolution
from benchmarks.demo_builder.phases.metadata import PhaseMetadata
from benchmarks.demo_builder.phases.ner import PhaseNER
from benchmarks.demo_builder.phases.node2vec import PhaseNode2Vec
from benchmarks.demo_builder.phases.re import PhaseRE
from benchmarks.demo_builder.phases.umap import PhaseChunksUMAP, PhaseEntitiesUMAP

__all__ = [
    "Phase",
    "PhaseChunks",
    "PhaseChunksEmbeddings",
    "PhaseChunksUMAP",
    "PhaseEntitiesUMAP",
    "PhaseEntityEmbeddings",
    "PhaseEntityResolution",
    "PhaseMetadata",
    "PhaseNER",
    "PhaseNode2Vec",
    "PhaseRE",
    "default_phases",
]


def default_phases(book_id: int, model_name: str, legacy_models: bool = False) -> list[Phase]:
    """Create the default ordered list of build phases.

    legacy_models: if True, use the GLiNER + GLiREL + spaCy stack instead of
    the default GLiNER2 single-model backend for NER and RE phases.

    Order is a topological sort of the true data dependency DAG:

        chunks ──┬──→ chunks_embeddings ──→ chunks_umap ──────────────────┐
                 │                                                          │
                 └──→ ner ──┬──→ relations ────────────────────────┐        │
                            │                                       ↓        ↓
                            └──→ entity_embeddings ──┬──→ entities_umap ──→ metadata
                                                     │                      ↑
                                                     └──→ entity_resolution → node2vec ─┘

    For parallel execution use `manifest --makefile` to generate a Make-managed build.
    """
    ner_backend = "gliner" if legacy_models else "gliner2"
    re_backend = "glirel" if legacy_models else "gliner2"
    return [
        PhaseChunks(book_id, model_name),
        PhaseChunksEmbeddings(book_id, model_name),
        PhaseChunksUMAP(),
        PhaseNER(backend=ner_backend),
        PhaseRE(backend=re_backend),
        PhaseEntityEmbeddings(model_name),
        PhaseEntitiesUMAP(),
        PhaseEntityResolution(),
        PhaseNode2Vec(),
        PhaseMetadata(book_id, model_name),
    ]
