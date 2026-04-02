"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle.

All phases are local to sessions_demo. No imports from demo_builder.
"""

from __future__ import annotations

from benchmarks.sessions_demo.constants import (
    SESSION_GLINER2_NER_LABELS,
    SESSION_GLINER2_RE_LABELS,
    SESSION_GLIREL_LABELS,
    SESSION_NER_LABELS,
)
from benchmarks.sessions_demo.phases.base import Phase
from benchmarks.sessions_demo.phases.chunks import PhaseChunks
from benchmarks.sessions_demo.phases.chunks_vec import PhaseChunksVec
from benchmarks.sessions_demo.phases.communities import PhaseCommunities
from benchmarks.sessions_demo.phases.community_naming import PhaseCommunityNaming
from benchmarks.sessions_demo.phases.entity_embeddings import PhaseEntityEmbeddings
from benchmarks.sessions_demo.phases.entity_resolution import PhaseEntityResolution
from benchmarks.sessions_demo.phases.ingest import PhaseIngest
from benchmarks.sessions_demo.phases.metadata import PhaseSessionMetadata
from benchmarks.sessions_demo.phases.ner import PhaseNER
from benchmarks.sessions_demo.phases.node2vec import PhaseNode2Vec
from benchmarks.sessions_demo.phases.re import PhaseRE
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
        PhaseChunksUMAP(),  # 4
        ner_phase,  # 5
        re_phase,  # 6
        PhaseEntityEmbeddings(),  # 7
        PhaseEntitiesUMAP(),  # 8
        PhaseEntityResolution(),  # 9
        PhaseNode2Vec(),  # 10
        PhaseCommunities(),  # 11
        PhaseCommunityNaming(),  # 12
        PhaseSessionMetadata(),  # 13
    ]
