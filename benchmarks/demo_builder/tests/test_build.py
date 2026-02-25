"""Tests for the build module — PhaseContext and DemoBuild staging lifecycle."""

from __future__ import annotations

import numpy as np

from benchmarks.demo_builder.build import PhaseContext


class TestPhaseContext:
    def test_defaults(self) -> None:
        ctx = PhaseContext()
        assert ctx.num_chunks == 0
        assert ctx.chunk_vectors is None
        assert ctx.num_entity_mentions == 0
        assert ctx.num_relations == 0
        assert ctx.num_unique_entities == 0
        assert ctx.entity_vectors is None
        assert ctx.num_nodes == 0
        assert ctx.num_edges == 0
        assert ctx.num_n2v == 0

    def test_mutable(self) -> None:
        ctx = PhaseContext()
        ctx.num_chunks = 100
        ctx.chunk_vectors = np.zeros((100, 384), dtype=np.float32)
        assert ctx.num_chunks == 100
        assert ctx.chunk_vectors.shape == (100, 384)
