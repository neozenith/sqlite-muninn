"""Base Treatment ABC for all benchmark permutations.

A Treatment represents one specific benchmark configuration (one permutation).
The harness calls setup -> run -> teardown and collects metrics automatically.

Subclasses MUST implement:
    category        — e.g., 'vss', 'graph', 'adjacency', 'kg-extract'
    permutation_id  — unique human-readable slug used as folder name
    label           — human-readable description
    setup()         — create tables, load data
    run()           — execute benchmark, return metrics dict
    teardown()      — clean up

Subclasses MAY override:
    params_dict()   — return all parameters as flat dict for JSONL serialization
"""

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Treatment(ABC):
    """Base class for all benchmark treatments."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Benchmark category: 'vss', 'graph', 'centrality', 'community',
        'graph_vt', 'kg-extract', 'kg-resolve', 'kg-graphrag', 'node2vec'."""

    @property
    @abstractmethod
    def permutation_id(self) -> str:
        """Human-readable slug, unique across all permutations.

        Used as folder name under results/ and as the --id value for CLI.
        Example: 'vss_muninn-hnsw_MiniLM_ag-news_n5000'
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable description for manifest display.

        Example: 'VSS: muninn-hnsw / MiniLM / ag_news / N=5000'
        """

    @abstractmethod
    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        """Create tables, load data, prepare for benchmark.

        Returns a dict of setup_metrics (e.g., {'n_vectors_loaded': 5000}).
        """

    @abstractmethod
    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Execute the benchmark measurement.

        Returns a dict of treatment-specific metrics (e.g., {
            'insert_rate_vps': 12345.6,
            'search_latency_ms': 0.42,
            'recall_at_k': 0.95,
        }).
        """

    @abstractmethod
    def teardown(self, conn: sqlite3.Connection) -> None:
        """Clean up resources after benchmark."""

    def params_dict(self) -> dict[str, Any]:
        """Return all treatment parameters as a flat dict for JSONL serialization.

        Subclasses should override to include their specific parameters.
        The harness merges this with common metrics when writing JSONL.
        """
        return {}

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Return a tuple for sorting permutations by size (ascending).

        The first element should be the primary scaling dimension (the x-axis
        metric), followed by secondary dimensions for tie-breaking.
        Subclasses should override to reflect their scaling dimension.
        Default sorts by permutation_id.
        """
        return (self.permutation_id,)
