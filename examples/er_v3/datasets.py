"""Dataset configurations, downloading, parsing, and ground truth construction.

Supports three DeepMatcher benchmark datasets:
  - Abt-Buy: E-commerce products (Abt.com vs Buy.com)
  - Amazon-Google: E-commerce products (Amazon vs Google Products)
  - DBLP-ACM: Academic papers (DBLP vs ACM Digital Library)
"""

import csv
import logging
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_BASE = PROJECT_ROOT / "tmp" / "er_benchmark"


# ── Data Structures ───────────────────────────────────────────────


@dataclass
class Entity:
    """A record from one side of a benchmark dataset."""

    id: str  # "a_0" or "b_115"
    name: str  # The entity text used for embedding and matching
    source: str  # "a" or "b"


@dataclass
class DatasetConfig:
    """Configuration for a DeepMatcher entity resolution dataset."""

    slug: str  # URL-safe identifier: "abt-buy"
    display_name: str  # Human-readable: "Abt-Buy"
    url_base: str  # HuggingFace resolve base
    name_column: str  # Column in tableA/B to use as entity name
    files: tuple[str, ...] = ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")

    @property
    def cache_dir(self) -> Path:
        return CACHE_BASE / self.slug


# ── Dataset Registry ──────────────────────────────────────────────

DATASETS: dict[str, DatasetConfig] = {
    "abt-buy": DatasetConfig(
        slug="abt-buy",
        display_name="Abt-Buy",
        url_base="https://huggingface.co/datasets/matchbench/Abt-Buy/resolve/main/",
        name_column="name",
    ),
    "amazon-google": DatasetConfig(
        slug="amazon-google",
        display_name="Amazon-Google",
        url_base="https://huggingface.co/datasets/matchbench/Amazon-Google/resolve/main/",
        name_column="title",
    ),
    "dblp-acm": DatasetConfig(
        slug="dblp-acm",
        display_name="DBLP-ACM",
        url_base="https://huggingface.co/datasets/matchbench/DBLP-ACM/resolve/main/",
        name_column="title",
    ),
}


# ── Union-Find for Ground Truth ───────────────────────────────────


class UnionFind:
    """Disjoint-set for building ground truth clusters from labeled pairs."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> dict[str, int]:
        """Return element -> cluster_id mapping."""
        cluster_map: dict[str, int] = {}
        root_to_id: dict[str, int] = {}
        next_id = 0
        for elem in self._parent:
            root = self.find(elem)
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            cluster_map[elem] = root_to_id[root]
        return cluster_map


# ── Download ──────────────────────────────────────────────────────


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "muninn-example/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)", end="", flush=True)
        if total > 0:
            print()


def download_dataset(cfg: DatasetConfig) -> Path:
    """Download dataset CSVs to cache directory. Returns cache path."""
    if all((cfg.cache_dir / f).exists() for f in cfg.files):
        log.info("Dataset %s cached at %s", cfg.slug, cfg.cache_dir)
        return cfg.cache_dir
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s dataset to %s", cfg.display_name, cfg.cache_dir)
    for filename in cfg.files:
        dest = cfg.cache_dir / filename
        if dest.exists():
            continue
        url = cfg.url_base + filename
        log.info("  %s", url)
        _download_file(url, dest)
    return cfg.cache_dir


# ── Parse + Load ──────────────────────────────────────────────────

# Placeholder values found in some datasets (e.g., DBLP-ACM uses "?" for missing titles)
_PLACEHOLDER_VALUES = {"", "?", "n/a", "none", "null", "unknown"}


def _is_valid_name(value: str | None) -> bool:
    """Filter out empty, null, and placeholder entity names."""
    return value is not None and value.strip().lower() not in _PLACEHOLDER_VALUES


def _parse_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(StringIO(text)))


def load_dataset(cfg: DatasetConfig, limit: int | None = None) -> tuple[list[Entity], dict[str, int]]:
    """Load a dataset. Returns (entities, ground_truth_clusters).

    Args:
        cfg: Dataset configuration.
        limit: Max number of entities to load. None = all.
              Entities are interleaved from both sources for balanced sampling.
    """
    cache = download_dataset(cfg)

    table_a = _parse_csv(cache / "tableA.csv")
    table_b = _parse_csv(cache / "tableB.csv")

    entities_a = [
        Entity(id=f"a_{r['id']}", name=r[cfg.name_column], source="a")
        for r in table_a
        if _is_valid_name(r.get(cfg.name_column))
    ]
    entities_b = [
        Entity(id=f"b_{r['id']}", name=r[cfg.name_column], source="b")
        for r in table_b
        if _is_valid_name(r.get(cfg.name_column))
    ]

    # Interleave sources for balanced --limit sampling
    entities: list[Entity] = []
    for i in range(max(len(entities_a), len(entities_b))):
        if i < len(entities_a):
            entities.append(entities_a[i])
        if i < len(entities_b):
            entities.append(entities_b[i])

    if limit is not None:
        entities = entities[:limit]

    # Build ground truth clusters from labeled pairs
    entity_ids = {e.id for e in entities}
    uf = UnionFind()
    for eid in entity_ids:
        uf.find(eid)

    for split in ["train", "valid", "test"]:
        rows = _parse_csv(cache / f"{split}.csv")
        for r in rows:
            if int(r["label"]) == 1:
                aid = f"a_{r['ltable_id']}"
                bid = f"b_{r['rtable_id']}"
                if aid in entity_ids and bid in entity_ids:
                    uf.union(aid, bid)

    gold = uf.clusters()

    # Stats
    cluster_sizes: dict[int, int] = defaultdict(int)
    for cid in gold.values():
        cluster_sizes[cid] += 1
    n_singletons = sum(1 for sz in cluster_sizes.values() if sz == 1)
    n_multi = sum(1 for sz in cluster_sizes.values() if sz > 1)

    log.info(
        "Loaded %s: %d entities (%d A, %d B), %d clusters (%d singletons, %d multi)",
        cfg.slug,
        len(entities),
        sum(1 for e in entities if e.source == "a"),
        sum(1 for e in entities if e.source == "b"),
        len(cluster_sizes),
        n_singletons,
        n_multi,
    )
    return entities, gold
