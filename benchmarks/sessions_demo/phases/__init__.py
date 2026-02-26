"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle."""

from __future__ import annotations

from benchmarks.sessions_demo.phases.base import Phase
from benchmarks.sessions_demo.phases.chunks import PhaseChunks
from benchmarks.sessions_demo.phases.embeddings import PhaseEmbeddings
from benchmarks.sessions_demo.phases.ingest import PhaseIngest

__all__ = [
    "Phase",
    "PhaseChunks",
    "PhaseEmbeddings",
    "PhaseIngest",
    "default_phases",
]


def default_phases() -> list[Phase]:
    """Create the default ordered list of build phases.

    Chunks must run before embeddings so that embedding targets
    are already sized within all model context windows.
    """
    return [
        PhaseIngest(),
        PhaseChunks(),
        PhaseEmbeddings(),
    ]
