"""Workload definitions: filter widths × query shapes over a sessions_demo.db-style DB.

A Workload is a (db_path, filter) pair. A Filter narrows chunks by project_id and
optionally a time window; the rest of the harness derives the allowed-canonical set
from that filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Filter:
    project_id: str | None = None
    days: int | None = None  # last-N days, None = no time bound

    @property
    def slug(self) -> str:
        parts = []
        parts.append("p-all" if self.project_id is None else f"p-{_short(self.project_id)}")
        parts.append("t-all" if self.days is None else f"t-{self.days}d")
        return "_".join(parts)


@dataclass(frozen=True)
class QuerySpec:
    metric: Literal["node_betweenness", "edge_betweenness", "degree"]
    top_k: int
    depth: int
    min_degree: int

    @property
    def slug(self) -> str:
        return f"{self.metric}_k{self.top_k}_d{self.depth}_m{self.min_degree}"


@dataclass(frozen=True)
class Workload:
    db_path: Path
    filter: Filter
    query: QuerySpec

    @property
    def slug(self) -> str:
        return f"{self.db_path.stem}__{self.filter.slug}__{self.query.slug}"


def _short(project_id: str) -> str:
    """Last path component of a project_id like '-Users-joshpeak-play-sqlite-vector-graph'."""
    return project_id.rstrip("-").rsplit("-", 1)[-1]


# Canonical query shapes — match the patterns from claude-code-sessions/.../kg/payload.py.
# The last shape (m=3) deliberately stresses min_degree pruning so we can measure
# whether a k-core decomposition actually pays off.
QUERY_SHAPES: list[QuerySpec] = [
    QuerySpec(metric="node_betweenness", top_k=3, depth=2, min_degree=2),
    QuerySpec(metric="edge_betweenness", top_k=5, depth=1, min_degree=1),
    QuerySpec(metric="degree", top_k=10, depth=3, min_degree=1),
    QuerySpec(metric="degree", top_k=20, depth=3, min_degree=3),
]


def filter_widths() -> list[Filter]:
    """Four filter widths spanning narrow → wide. `days` is relative to MAX(timestamp)
    in the DB (set in the strategy when building the WHERE clause), so the window is
    reproducible regardless of when the harness is run.
    """
    big_project = "-Users-joshpeak-play-sqlite-vector-graph"
    return [
        Filter(project_id=None, days=None),
        Filter(project_id=big_project, days=None),
        Filter(project_id=big_project, days=30),
        Filter(project_id=big_project, days=7),
    ]
