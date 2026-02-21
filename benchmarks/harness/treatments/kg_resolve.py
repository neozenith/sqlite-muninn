"""KG entity resolution treatment.

Benchmarks entity resolution pipeline: HNSW blocking → Jaro-Winkler matching → Leiden clustering.
Two modes: KG coalescing on Gutenberg texts, ER benchmark datasets with ground truth.

Source: docs/plans/entity_resolution_benchmarks.md
"""

import logging
import time

from benchmarks.harness.common import KG_DIR, load_muninn
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


class KGEntityResolutionTreatment(Treatment):
    """Single entity resolution benchmark permutation."""

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def category(self):
        return "kg-resolve"

    @property
    def permutation_id(self):
        return f"kg-resolve_{self._dataset}"

    @property
    def label(self):
        return f"KG Resolve: {self._dataset}"

    @property
    def sort_key(self):
        return (self._dataset,)

    def params_dict(self):
        return {"dataset": self._dataset}

    def setup(self, conn, db_path):
        load_muninn(conn)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT,
                cluster_id INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resolution_metrics (
                metric TEXT PRIMARY KEY,
                value REAL
            )
        """)
        conn.commit()

        return {"dataset": self._dataset}

    def run(self, conn):
        # For numeric dataset IDs (book_id), load from KG extraction output
        # For named datasets (febrl1, etc.), load from ER dataset directory
        if self._dataset.isdigit():
            return self._run_kg_coalesce(conn)
        else:
            return self._run_er_dataset(conn)

    def teardown(self, conn):
        pass

    def _run_kg_coalesce(self, conn):
        """Coalesce entities from a KG extraction output."""
        book_id = int(self._dataset)
        entities_db = KG_DIR / f"{book_id}_chunks.db"

        if not entities_db.exists():
            log.warning("Entities DB not found: %s", entities_db)
            return {"nodes_before": 0, "nodes_after": 0, "total_time_s": 0}

        t0 = time.perf_counter()
        # Placeholder: actual coalescing would load entities, run HNSW blocking,
        # Jaro-Winkler matching, and Leiden clustering
        total_time = time.perf_counter() - t0

        return {
            "nodes_before": 0,
            "nodes_after": 0,
            "singleton_ratio": 0.0,
            "blocking_time_s": 0.0,
            "matching_time_s": 0.0,
            "clustering_time_s": 0.0,
            "total_time_s": round(total_time, 3),
        }

    def _run_er_dataset(self, conn):
        """Run entity resolution on an ER benchmark dataset with ground truth."""
        dataset_dir = KG_DIR / "er" / self._dataset

        if not dataset_dir.exists():
            log.warning("ER dataset not found: %s — run 'prep kg-er' first", dataset_dir)
            return {
                "pairwise_precision": 0.0,
                "pairwise_recall": 0.0,
                "pairwise_f1": 0.0,
                "total_time_s": 0.0,
            }

        t0 = time.perf_counter()
        # Placeholder: actual ER pipeline would load dataset, run blocking + matching + clustering,
        # then compute Pairwise F1 and B-Cubed F1 against ground truth
        total_time = time.perf_counter() - t0

        return {
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "bcubed_f1": 0.0,
            "blocking_time_s": 0.0,
            "matching_time_s": 0.0,
            "clustering_time_s": 0.0,
            "total_time_s": round(total_time, 3),
        }
