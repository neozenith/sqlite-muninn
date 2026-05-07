"""Strategy ABC: every strategy maps Workload -> Result with the same shape."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from benchmarks.kg_perf.workload import Workload


@dataclass
class Result:
    """One end-to-end query result: the subgraph we'd hand to the visualizer."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    seeds: list[str] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def node_set(self) -> frozenset[str]:
        return frozenset(self.nodes)

    @property
    def edge_set(self) -> frozenset[tuple[str, str]]:
        return frozenset(self.edges)

    @property
    def seed_set(self) -> frozenset[str]:
        return frozenset(self.seeds)


class Strategy(ABC):
    name: str = "abstract"

    def prepare(self, conn: sqlite3.Connection) -> None:
        """Optional: build any auxiliary tables, register prepared statements, etc."""

    @abstractmethod
    def run(self, conn: sqlite3.Connection, workload: Workload) -> Result: ...
