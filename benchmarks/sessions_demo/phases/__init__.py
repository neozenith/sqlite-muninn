"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle.

The KG pipeline phases (NER, RE, entity_embeddings, entity_resolution, node2vec)
are imported directly from demo_builder. They work against sessions_demo's
PhaseContext via Python's structural (duck) typing — both contexts expose the
same attribute names (num_chunks, num_entity_mentions, entity_vectors, etc.)
and the phases only use TYPE_CHECKING imports for the context type hint.
"""

from __future__ import annotations

from benchmarks.demo_builder.phases.communities import PhaseCommunities
from benchmarks.demo_builder.phases.community_naming import PhaseCommunityNaming
from benchmarks.demo_builder.phases.entity_embeddings import PhaseEntityEmbeddings
from benchmarks.demo_builder.phases.entity_resolution import PhaseEntityResolution
from benchmarks.demo_builder.phases.ner import PhaseNER
from benchmarks.demo_builder.phases.node2vec import PhaseNode2Vec
from benchmarks.demo_builder.phases.re import PhaseRE
from benchmarks.sessions_demo.constants import (
    SESSION_GLINER2_NER_LABELS,
    SESSION_GLINER2_RE_LABELS,
    SESSION_GLIREL_LABELS,
    SESSION_NER_LABELS,
)
from benchmarks.sessions_demo.phases.base import Phase
from benchmarks.sessions_demo.phases.chunks import PhaseChunks
from benchmarks.sessions_demo.phases.chunks_vec import PhaseChunksVec
from benchmarks.sessions_demo.phases.ingest import PhaseIngest
from benchmarks.sessions_demo.phases.metadata import PhaseSessionMetadata
from benchmarks.sessions_demo.phases.umap import PhaseChunksUMAP, PhaseEntitiesUMAP

__all__ = [
    "Phase",
    "PhaseChunks",
    "PhaseChunksUMAP",
    "PhaseChunksVec",
    "PhaseCommunities",
    "PhaseCommunityNaming",
    "PhaseEntitiesUMAP",
    "PhaseEntityEmbeddings",
    "PhaseEntityResolution",
    "PhaseIngest",
    "PhaseNER",
    "PhaseNode2Vec",
    "PhaseRE",
    "PhaseSessionMetadata",
    "default_phases",
]


def default_phases(message_types: list[str] | None = None, legacy_models: bool = False) -> list[Phase]:
    """Create the default ordered list of build phases.

    message_types: event_type values to include when chunking. Passed directly
    to PhaseChunks. None defaults to DEFAULT_MESSAGE_TYPES (["user"]).

    legacy_models: if True, use the GLiNER + GLiREL + spaCy stack instead of
    the default GLiNER2 single-model backend for NER and RE phases.

    Phases are ordered to respect data dependencies — each phase sits
    immediately after its deepest dependency in the DAG:

      1  ingest            → events, event_edges, projects, sessions
      2  chunks            → event_message_chunks, chunks, chunks_fts
      3  chunks_vec        → chunks_vec (HNSW 768d)
      4  chunks_vec_umap   → chunks_vec_umap          (depends on: chunks_vec)
      5  ner               → entities, ner_chunks_log
      6  relations         → relations, re_chunks_log
      7  entity_embeddings → entities_vec (HNSW 768d), entity_vec_map
      8  entities_vec_umap → entities_vec_umap         (depends on: entity_embeddings)
      9  entity_resolution → entity_clusters, nodes, edges
     10  node2vec          → node2vec_emb (HNSW)
     11  communities       → leiden_communities        (depends on: relations)
     12  community_naming  → entity_cluster_labels, community_labels (depends on: communities)
     13  metadata          → meta
    """
    if legacy_models:
        ner_phase = PhaseNER(labels=SESSION_NER_LABELS, backend="gliner")
        re_phase = PhaseRE(labels=SESSION_GLIREL_LABELS, backend="glirel")
    else:
        ner_phase = PhaseNER(labels=SESSION_GLINER2_NER_LABELS, backend="gliner2")
        re_phase = PhaseRE(labels=SESSION_GLINER2_RE_LABELS, backend="gliner2")

    return [
        PhaseIngest(),  # 1
        PhaseChunks(message_types=message_types),  # 2
        PhaseChunksVec(),  # 3
        PhaseChunksUMAP(),  # 4  — chunks_vec_nodes only
        ner_phase,  # type: ignore[list-item]  # 5 — demo_builder Phase subclass
        re_phase,  # type: ignore[list-item]  # 6 — demo_builder Phase subclass
        PhaseEntityEmbeddings(model_name="NomicEmbed"),  # type: ignore[list-item]  # 7
        PhaseEntitiesUMAP(),  # 8  — entities_vec_nodes only
        PhaseEntityResolution(),  # type: ignore[list-item]  # 9
        PhaseNode2Vec(),  # type: ignore[list-item]  # 10
        PhaseCommunities(),  # type: ignore[list-item]  # 11
        PhaseCommunityNaming(),  # type: ignore[list-item]  # 12
        PhaseSessionMetadata(),  # 13
    ]
