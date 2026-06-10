"""Strategy registry — one entry per implementation we want to A/B."""

from __future__ import annotations

from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.strategies.baseline import BaselineStrategy
from benchmarks.kg_perf.strategies.chunk_canonical import ChunkCanonicalStrategy
from benchmarks.kg_perf.strategies.kcore import KCoreStrategy
from benchmarks.kg_perf.strategies.sql_subset import SqlSubsetStrategy
from benchmarks.kg_perf.strategies.topk_cache import TopKCacheStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    BaselineStrategy.name: BaselineStrategy,
    SqlSubsetStrategy.name: SqlSubsetStrategy,
    ChunkCanonicalStrategy.name: ChunkCanonicalStrategy,
    KCoreStrategy.name: KCoreStrategy,
    TopKCacheStrategy.name: TopKCacheStrategy,
}

__all__ = ["Result", "STRATEGIES", "Strategy"]
