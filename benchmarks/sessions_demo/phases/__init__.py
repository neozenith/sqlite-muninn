"""Build phases — each phase is a Phase subclass with setup/run/teardown lifecycle.

The KG pipeline phases (NER, RE, entity_embeddings, entity_resolution, node2vec)
are imported directly from demo_builder. They work against sessions_demo's
PhaseContext via Python's structural (duck) typing — both contexts expose the
same attribute names (num_chunks, num_entity_mentions, entity_vectors, etc.)
and the phases only use TYPE_CHECKING imports for the context type hint.
"""

from __future__ import annotations

from benchmarks.demo_builder.phases.entity_embeddings import PhaseEntityEmbeddings
from benchmarks.demo_builder.phases.entity_resolution import PhaseEntityResolution
from benchmarks.demo_builder.phases.ner import PhaseNER
from benchmarks.demo_builder.phases.node2vec import PhaseNode2Vec
from benchmarks.demo_builder.phases.re import PhaseRE
from benchmarks.sessions_demo.constants import SESSION_GLIREL_LABELS, SESSION_NER_LABELS
from benchmarks.sessions_demo.phases.base import Phase
from benchmarks.sessions_demo.phases.chunks import PhaseChunks
from benchmarks.sessions_demo.phases.embeddings import PhaseEmbeddings
from benchmarks.sessions_demo.phases.ingest import PhaseIngest
from benchmarks.sessions_demo.phases.metadata import PhaseSessionMetadata
from benchmarks.sessions_demo.phases.umap import PhaseChunksUMAP, PhaseEntitiesUMAP

__all__ = [
    "Phase",
    "PhaseChunks",
    "PhaseChunksUMAP",
    "PhaseEmbeddings",
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


def default_phases() -> list[Phase]:
    """Create the default ordered list of build phases.

    Phases are ordered to respect data dependencies — each phase sits
    immediately after its deepest dependency in the DAG:

      1  ingest            → events, event_edges, projects, sessions
      2  chunks            → event_message_chunks, chunks, chunks_fts
      3  embeddings        → chunks_vec (HNSW 768d)
      4  chunks_vec_umap   → chunks_vec_umap          (depends on: embeddings)
      5  ner               → entities, ner_chunks_log
      6  relations         → relations, re_chunks_log
      7  entity_embeddings → entities_vec (HNSW 768d), entity_vec_map
      8  entities_vec_umap → entities_vec_umap         (depends on: entity_embeddings)
      9  entity_resolution → entity_clusters, nodes, edges
     10  node2vec          → node2vec_emb (HNSW)
     11  metadata          → meta
    """
    return [
        PhaseIngest(),  # 1
        PhaseChunks(),  # 2
        PhaseEmbeddings(),  # 3
        PhaseChunksUMAP(),  # 4  — chunks_vec_nodes only
        PhaseNER(labels=SESSION_NER_LABELS),  # 5
        PhaseRE(labels=SESSION_GLIREL_LABELS),  # 6
        PhaseEntityEmbeddings(model_name="NomicEmbed"),  # 7
        PhaseEntitiesUMAP(),  # 8  — entities_vec_nodes only
        PhaseEntityResolution(),  # 9
        PhaseNode2Vec(),  # 10
        PhaseSessionMetadata(),  # 11
    ]
