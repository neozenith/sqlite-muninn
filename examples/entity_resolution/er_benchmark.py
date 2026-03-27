"""
Entity Resolution Benchmark — String-Only vs LLM-Tiered on Abt-Buy

Evaluates muninn's entity resolution capabilities using the DeepMatcher
Abt-Buy dataset (product matching across Abt.com and Buy.com).

Two pipeline modes compared side-by-side:
  - string-only: HNSW blocking → Jaro-Winkler + cosine cascade → Leiden clustering
  - llm-tiered:  Same blocking + cascade, but borderline pairs routed to
                  muninn_chat() with GBNF grammar-constrained generation

Grammar debug subcommands (G6) test two GBNF formats at three tiers:
  - Format A (pairwise): {"match": true, "confidence": 0.95}
  - Format B (clustering): {"groups": [["NYC", "New York City"], ["London"]]}

Requirements:
  - muninn extension (make all)
  - GGUF embedding model (auto-downloaded, ~23 MB)
  - GGUF chat model for LLM tier (auto-downloaded, ~1.3 GB)

Usage:
  uv run examples/entity_resolution/er_benchmark.py string-only --limit 10
  uv run examples/entity_resolution/er_benchmark.py llm-tiered --limit 10
  uv run examples/entity_resolution/er_benchmark.py compare --limit 100
  uv run examples/entity_resolution/er_benchmark.py grammar-pairwise-raw --limit 10
"""

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

log = logging.getLogger(__name__)

_IN_COLAB = "google.colab" in sys.modules

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    if _IN_COLAB:
        import subprocess

        _REPO = Path("/content/sqlite-muninn")
        if not _REPO.exists():
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/neozenith/sqlite-muninn.git", str(_REPO)],
                check=True,
            )
        if not list((_REPO / "build").glob("muninn.*")):
            subprocess.run(["apt-get", "install", "-y", "libsqlite3-dev"], check=True)
            subprocess.run(["make", "all"], cwd=str(_REPO), check=True)
        PROJECT_ROOT = _REPO
    else:
        PROJECT_ROOT = Path.cwd().parent.parent

EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "tmp" / "er_benchmark" / "abt_buy"
RESULTS_DIR = PROJECT_ROOT / "examples" / "entity_resolution" / "results"


# ── Model Definitions ─────────────────────────────────────────────


@dataclass
class EmbedModelConfig:
    """GGUF embedding model descriptor."""

    name: str
    filename: str
    url: str


@dataclass
class ChatModelConfig:
    """GGUF chat model descriptor."""

    name: str
    filename: str
    url: str
    size_gb: float


EMBED_MODEL = EmbedModelConfig(
    name="MiniLM",
    filename="all-MiniLM-L6-v2.Q8_0.gguf",
    url="https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf",
)

CHAT_MODELS = [
    ChatModelConfig(
        "Qwen3.5-2B",
        "Qwen3.5-2B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
        1.28,
    ),
    ChatModelConfig(
        "Qwen3.5-4B",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        2.7,
    ),
    ChatModelConfig(
        "Gemma-3-4B",
        "google_gemma-3-4b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf",
        2.5,
    ),
]

DEFAULT_CHAT_MODEL = CHAT_MODELS[0]  # Qwen3.5-2B for individual subcommands


# ── Dataset ───────────────────────────────────────────────────────

DATASET_BASE_URL = "https://huggingface.co/datasets/matchbench/Abt-Buy/resolve/main/"
DATASET_FILES = ["tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv"]


@dataclass
class Entity:
    """A product entity from the Abt-Buy dataset."""

    id: str  # "a_0" or "b_115"
    name: str  # Product name
    source: str  # "a" or "b"
    description: str  # Product description (may be empty)


class UnionFind:
    """Disjoint-set data structure for building ground truth clusters."""

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


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress indicator."""
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
                    print(
                        f"\r  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)",
                        end="",
                        flush=True,
                    )
        if total > 0:
            print()


def ensure_model(model: EmbedModelConfig | ChatModelConfig) -> None:
    """Download GGUF model if not already present."""
    path = MODELS_DIR / model.filename
    if path.exists():
        log.info("Model %s: %s (%.1f MB)", model.name, path, path.stat().st_size / 1e6)
        return
    log.info("Downloading %s...", model.filename)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_with_progress(model.url, path)
    log.info("Downloaded %s (%.1f MB)", model.filename, path.stat().st_size / 1e6)


def download_dataset() -> Path:
    """Download Abt-Buy dataset CSVs to cache directory. Returns cache path."""
    if all((CACHE_DIR / f).exists() for f in DATASET_FILES):
        log.info("Dataset cached at %s", CACHE_DIR)
        return CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Abt-Buy dataset to %s", CACHE_DIR)
    for filename in DATASET_FILES:
        dest = CACHE_DIR / filename
        if dest.exists():
            continue
        url = DATASET_BASE_URL + filename
        log.info("  %s", url)
        _download_with_progress(url, dest)
    return CACHE_DIR


def _parse_csv(path: Path) -> list[dict[str, str]]:
    """Parse a CSV file into a list of dicts."""
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(StringIO(text))
    return list(reader)


def load_dataset(limit: int | None = None) -> tuple[list[Entity], dict[str, int]]:
    """Load Abt-Buy dataset. Returns (entities, ground_truth_clusters).

    Args:
        limit: Max number of entities to load. None = all.
              Entities are interleaved from both sources for balanced sampling.
    """
    cache = download_dataset()

    # Parse product tables
    table_a = _parse_csv(cache / "tableA.csv")
    table_b = _parse_csv(cache / "tableB.csv")

    entities_a = [
        Entity(id=f"a_{r['id']}", name=r["name"], source="a", description=r.get("description", "")) for r in table_a
    ]
    entities_b = [
        Entity(id=f"b_{r['id']}", name=r["name"], source="b", description=r.get("description", "")) for r in table_b
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
        uf.find(eid)  # Initialize all entities as singletons

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
        "Loaded %d entities (%d from A, %d from B), %d clusters (%d singletons, %d multi-member)",
        len(entities),
        sum(1 for e in entities if e.source == "a"),
        sum(1 for e in entities if e.source == "b"),
        len(cluster_sizes),
        n_singletons,
        n_multi,
    )
    return entities, gold


# ── GBNF Grammars (G6) ───────────────────────────────────────────

# Format A — Pairwise: one call per borderline pair, tiny output
GBNF_ER_PAIRWISE = (
    'root    ::= "{" ws "\\"match\\"" ws ":" ws boolean ws "," '
    'ws "\\"confidence\\"" ws ":" ws number ws "}" \n'
    'boolean ::= "true" | "false" \n'
    'number  ::= [0-9] ("." [0-9]+)? \n'
    'ws      ::= " "? \n'
)

# Format B — Clustering (strings): one call per HNSW neighborhood, names in output
# Used by grammar-debug subcommands to test name reproduction fidelity
GBNF_ER_CLUSTER = (
    'root        ::= "{" ws "\\"groups\\"" ws ":" ws "[" ws group-list ws "]" ws "}" \n'
    'group-list  ::= group (ws "," ws group)* \n'
    'group       ::= "[" ws string-list ws "]" \n'
    'string-list ::= string (ws "," ws string)* \n'
    'string      ::= "\\"" [^\\"\\\\]* "\\"" \n'
    "ws          ::= [ \\t\\n]* \n"
)

# Format B′ — Clustering (numbered): pipeline uses numbered indices for reliability
# LLM returns group indices instead of reproducing entity names verbatim
GBNF_ER_CLUSTER_NUM = (
    'root       ::= "{" ws "\\"groups\\"" ws ":" ws "[" ws group-list ws "]" ws "}" \n'
    'group-list ::= group (ws "," ws group)* \n'
    'group      ::= "[" ws int-list ws "]" \n'
    'int-list   ::= integer (ws "," ws integer)* \n'
    "integer    ::= [1-9] [0-9]* \n"
    "ws         ::= [ \\t\\n]* \n"
)

# ── Prompts ───────────────────────────────────────────────────────

ER_PAIRWISE_SYSTEM = (
    "You are an entity resolution expert. Given two product names, determine "
    "if they refer to the same real-world product. Respond with JSON containing "
    '"match" (boolean) and "confidence" (0-1 float).'
)

ER_PAIRWISE_PROMPT = (
    "Do these two product names refer to the same product?\n\n"
    "Product A: {name_a}\n"
    "Product B: {name_b}\n\n"
    "Respond in JSON."
)

ER_PAIRWISE_ONESHOT = (
    "Do these two product names refer to the same product?\n\n"
    "Example:\n"
    '  Product A: Sony VAIO VPC-EB15FM/BI 15.5" Notebook PC\n'
    "  Product B: Sony VAIO VPCEB15FM/BI Notebook\n"
    '  Answer: {{"match": true, "confidence": 0.92}}\n\n'
    "Now answer:\n"
    "Product A: {name_a}\n"
    "Product B: {name_b}\n\n"
    "Respond in JSON."
)

ER_CLUSTER_SYSTEM = (
    "You are an entity resolution expert. Given a list of product names, group "
    "the ones that refer to the same real-world product. Each group is a list of "
    "names. Products that don't match anything else appear as singleton groups."
)

ER_CLUSTER_NUM_SYSTEM = (
    "You are an entity resolution expert. Given a numbered list of product names, "
    "group the ones that refer to the same real-world product. Return groups as "
    "lists of numbers. Products with no match go in their own singleton group."
)

ER_CLUSTER_NUM_PROMPT = (
    "Group these products by whether they refer to the same product. "
    "Use their numbers, not names.\n\n"
    "{candidate_list}\n\n"
    'Respond with JSON: {{"groups": [[1, 3], [2], ...]}}.'
)

ER_CLUSTER_PROMPT = (
    "Group these product names by whether they refer to the same product:\n\n"
    "{candidate_list}\n\n"
    'Respond with JSON: {{"groups": [[...], ...]}}. '
    "Each group contains names that refer to the same product."
)

ER_CLUSTER_ONESHOT = (
    "Group these product names by whether they refer to the same product:\n\n"
    "Example:\n"
    '  1. Sony VAIO VPC-EB15FM/BI 15.5" Notebook PC\n'
    "  2. Sony VAIO VPCEB15FM/BI Notebook\n"
    '  3. Acer Aspire 5100-5033 15.4" Notebook\n'
    '  Answer: {{"groups": [["Sony VAIO VPC-EB15FM/BI 15.5\\" Notebook PC", '
    '"Sony VAIO VPCEB15FM/BI Notebook"], '
    '["Acer Aspire 5100-5033 15.4\\" Notebook"]]}}\n\n'
    "Now group these:\n"
    "{candidate_list}\n\n"
    'Respond with JSON: {{"groups": [[...], ...]}}.'
)


# ── Metrics ───────────────────────────────────────────────────────


def bcubed_f1(predicted: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    """B-Cubed F1 for clustering evaluation (primary ER metric).

    Each dict maps element_id -> cluster_id. Only elements in both dicts are evaluated.
    """
    common = set(predicted.keys()) & set(gold.keys())
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_by_cluster: dict[int, set[str]] = defaultdict(set)
    gold_by_cluster: dict[int, set[str]] = defaultdict(set)
    for elem in common:
        pred_by_cluster[predicted[elem]].add(elem)
        gold_by_cluster[gold[elem]].add(elem)

    total_p = 0.0
    total_r = 0.0
    for elem in common:
        pred_members = pred_by_cluster[predicted[elem]] & common
        gold_members = gold_by_cluster[gold[elem]] & common
        shared = pred_members & gold_members
        total_p += len(shared) / len(pred_members) if pred_members else 0.0
        total_r += len(shared) / len(gold_members) if gold_members else 0.0

    n = len(common)
    p = total_p / n
    r = total_r / n
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def pairwise_f1(predicted: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    """Pairwise F1 for clustering evaluation (secondary ER metric).

    Evaluates whether pairs of elements in the same predicted cluster
    are also in the same gold cluster.
    """
    common = sorted(set(predicted.keys()) & set(gold.keys()))
    if len(common) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_by_cluster: dict[int, set[str]] = defaultdict(set)
    gold_by_cluster: dict[int, set[str]] = defaultdict(set)
    for elem in common:
        pred_by_cluster[predicted[elem]].add(elem)
        gold_by_cluster[gold[elem]].add(elem)

    def _pairs(cluster_map: dict[int, set[str]]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for members in cluster_map.values():
            members_list = sorted(members)
            for i in range(len(members_list)):
                for j in range(i + 1, len(members_list)):
                    pairs.add((members_list[i], members_list[j]))
        return pairs

    pred_pairs = _pairs(pred_by_cluster)
    gold_pairs = _pairs(gold_by_cluster)

    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ── Jaro-Winkler ─────────────────────────────────────────────────


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler string similarity (0.0 to 1.0)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s1_matched[i] = True
            s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


# ── Database Setup ────────────────────────────────────────────────


def create_db(chat_model: ChatModelConfig | None = None) -> sqlite3.Connection:
    """Create in-memory SQLite with muninn loaded and models registered.

    Args:
        chat_model: If provided, also load this GGUF chat model.
    """
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Embedding model (always needed)
    ensure_model(EMBED_MODEL)
    embed_path = str(MODELS_DIR / EMBED_MODEL.filename)
    conn.execute(
        "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
        (EMBED_MODEL.name, embed_path),
    )
    log.info("Loaded embedding model: %s", EMBED_MODEL.name)

    if chat_model:
        register_chat_model(conn, chat_model)

    return conn


def register_chat_model(conn: sqlite3.Connection, model: ChatModelConfig) -> None:
    """Load and register a GGUF chat model into an existing connection."""
    ensure_model(model)
    chat_path = str(MODELS_DIR / model.filename)
    conn.execute(
        "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
        (model.name, chat_path),
    )
    log.info("Loaded chat model: %s", model.name)


def cleanup_pipeline_tables(conn: sqlite3.Connection) -> None:
    """Drop pipeline tables so the connection can be reused for another run."""
    for table in ["entities", "entity_vecs", "_match_edges"]:
        conn.execute(f"DROP TABLE IF EXISTS [{table}]")


# ── Pipeline: String-Only ─────────────────────────────────────────


def run_string_only(
    conn: sqlite3.Connection,
    entities: list[Entity],
    k: int = 10,
    dist_threshold: float = 0.4,
    match_threshold: float = 0.5,
) -> dict[str, int]:
    """Run string-only ER pipeline. Returns entity_id -> cluster_id.

    Stages:
      1. Embed entity names via muninn_embed() into HNSW index
      2. HNSW blocking: KNN search per entity (k neighbors, cosine distance filter)
      3. Matching cascade: exact -> case-insensitive -> JW + cosine combined
      4. Leiden clustering on match edges
    """
    id_map, name_map, candidate_pairs = _embed_and_block(conn, entities, k, dist_threshold)

    match_edges: list[tuple[str, str, float]] = []
    for (r1, r2), cosine_dist in candidate_pairs.items():
        n1 = name_map[r1]
        n2 = name_map[r2]
        cosine_sim = 1.0 - cosine_dist

        if n1 == n2:
            score = 1.0
        elif n1.lower() == n2.lower():
            score = 0.9
        else:
            jw = jaro_winkler(n1.lower(), n2.lower())
            score = 0.4 * jw + 0.6 * cosine_sim

        if score > match_threshold:
            match_edges.append((id_map[r1], id_map[r2], score))

    log.info("Matching: %d edges pass threshold %.2f", len(match_edges), match_threshold)
    return _leiden_cluster(conn, entities, match_edges)


def _leiden_cluster(
    conn: sqlite3.Connection,
    entities: list[Entity],
    match_edges: list[tuple[str, str, float]],
) -> dict[str, int]:
    """Run Leiden clustering on match edges. Returns entity_id -> cluster_id."""
    conn.execute("CREATE TEMP TABLE _match_edges(src TEXT, dst TEXT, weight REAL)")
    for src, dst, w in match_edges:
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (src, dst, w))
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (dst, src, w))

    clusters: dict[str, int] = {}
    next_id = 0

    if match_edges:
        leiden_results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = '_match_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()
        comm_to_id: dict[int, int] = {}
        for node, comm_id in leiden_results:
            if comm_id not in comm_to_id:
                comm_to_id[comm_id] = next_id
                next_id += 1
            clusters[node] = comm_to_id[comm_id]

    # Entities not in any match edge are singletons
    for e in entities:
        if e.id not in clusters:
            clusters[e.id] = next_id
            next_id += 1

    n_multi = sum(1 for sz in _cluster_sizes(clusters).values() if sz > 1)
    log.info("Leiden: %d clusters (%d multi-member)", len(set(clusters.values())), n_multi)
    return clusters


# ── Common: Embed + Block ─────────────────────────────────────────


def _embed_and_block(
    conn: sqlite3.Connection,
    entities: list[Entity],
    k: int = 10,
    dist_threshold: float = 0.4,
) -> tuple[dict[int, str], dict[int, str], dict[tuple[int, int], float]]:
    """Create entities table, embed into HNSW, run KNN blocking.

    Returns (id_map, name_map, candidate_pairs) where:
      id_map: rowid -> entity_id
      name_map: rowid -> entity name
      candidate_pairs: (min_rid, max_rid) -> cosine_distance
    """
    conn.execute("CREATE TABLE entities(entity_id TEXT, name TEXT, source TEXT)")
    for e in entities:
        conn.execute("INSERT INTO entities VALUES(?, ?, ?)", (e.id, e.name, e.source))

    dim = conn.execute("SELECT muninn_model_dim(?)", (EMBED_MODEL.name,)).fetchone()[0]
    conn.execute(f"CREATE VIRTUAL TABLE entity_vecs USING hnsw_index(dimensions={dim}, metric=cosine)")
    conn.execute(
        "INSERT INTO entity_vecs(rowid, vector) SELECT rowid, muninn_embed(?, name) FROM entities",
        (EMBED_MODEL.name,),
    )
    log.info("Embedded %d entities (dim=%d)", len(entities), dim)

    id_map: dict[int, str] = {}
    name_map: dict[int, str] = {}
    for row in conn.execute("SELECT rowid, entity_id, name FROM entities"):
        id_map[row[0]] = row[1]
        name_map[row[0]] = row[2]

    candidate_pairs: dict[tuple[int, int], float] = {}
    for rowid in id_map:
        vec = conn.execute("SELECT vector FROM entity_vecs WHERE rowid = ?", (rowid,)).fetchone()[0]
        neighbors = conn.execute(
            "SELECT rowid, distance FROM entity_vecs WHERE vector MATCH ? AND k = ?",
            (vec, k + 1),
        ).fetchall()
        for nid, dist in neighbors:
            if nid != rowid and dist <= dist_threshold:
                pair = (min(rowid, nid), max(rowid, nid))
                if pair not in candidate_pairs or dist < candidate_pairs[pair]:
                    candidate_pairs[pair] = dist

    log.info("HNSW blocking: %d candidate pairs (k=%d, dist<=%.2f)", len(candidate_pairs), k, dist_threshold)
    return id_map, name_map, candidate_pairs


# ── Pipeline: LLM-Pairwise ───────────────────────────────────────


def run_llm_pairwise(
    conn: sqlite3.Connection,
    entities: list[Entity],
    model_name: str,
    k: int = 10,
    dist_threshold: float = 0.4,
    llm_low: float = 0.3,
    llm_high: float = 0.7,
) -> dict[str, int]:
    """Run LLM-pairwise ER pipeline (Format A grammar). Returns entity_id -> cluster_id.

    Three matching tiers:
      score > llm_high  -> auto-accept (string similarity is confident)
      score < llm_low   -> auto-reject
      llm_low <= score <= llm_high -> one LLM call per pair via GBNF_ER_PAIRWISE
    """
    id_map, name_map, candidate_pairs = _embed_and_block(conn, entities, k, dist_threshold)

    match_edges: list[tuple[str, str, float]] = []
    llm_calls = 0
    llm_time = 0.0

    for (r1, r2), cosine_dist in candidate_pairs.items():
        n1 = name_map[r1]
        n2 = name_map[r2]
        cosine_sim = 1.0 - cosine_dist

        if n1 == n2:
            match_edges.append((id_map[r1], id_map[r2], 1.0))
            continue
        if n1.lower() == n2.lower():
            match_edges.append((id_map[r1], id_map[r2], 0.9))
            continue

        jw = jaro_winkler(n1.lower(), n2.lower())
        score = 0.4 * jw + 0.6 * cosine_sim

        if score > llm_high:
            match_edges.append((id_map[r1], id_map[r2], score))
        elif score >= llm_low:
            prompt = ER_PAIRWISE_PROMPT.format(name_a=n1, name_b=n2)
            t0 = time.perf_counter()
            result = conn.execute(
                "SELECT muninn_chat(?, ?, ?, ?, ?)",
                (model_name, prompt, GBNF_ER_PAIRWISE, 50, ER_PAIRWISE_SYSTEM),
            ).fetchone()[0]
            llm_time += time.perf_counter() - t0
            llm_calls += 1

            parsed = _parse_pairwise_result(result)
            if parsed and parsed.get("match"):
                confidence = parsed.get("confidence", 0.8)
                match_edges.append((id_map[r1], id_map[r2], confidence))

    log.info("LLM-pairwise: %d edges, %d LLM calls (%.2fs)", len(match_edges), llm_calls, llm_time)
    return _leiden_cluster(conn, entities, match_edges)


# ── Pipeline: LLM-Cluster ────────────────────────────────────────


def run_llm_cluster(
    conn: sqlite3.Connection,
    entities: list[Entity],
    model_name: str,
    k: int = 10,
    dist_threshold: float = 0.4,
    llm_low: float = 0.3,
    llm_high: float = 0.7,
) -> dict[str, int]:
    """Run LLM-cluster ER pipeline (Format B′ numbered grammar). Returns entity_id -> cluster_id.

    Instead of one LLM call per pair, sends entire neighborhoods for batch clustering.
    Uses GBNF_ER_CLUSTER_NUM with numbered indices for reliable output parsing.

    Stages:
      1. Embed + HNSW block (shared)
      2. Auto-accept confident pairs, collect borderline pairs
      3. Find connected components in borderline graph
      4. One LLM call per component → parse numbered groups → match edges
      5. Leiden clustering on all edges
    """
    id_map, name_map, candidate_pairs = _embed_and_block(conn, entities, k, dist_threshold)

    match_edges: list[tuple[str, str, float]] = []
    borderline_pairs: list[tuple[int, int]] = []

    for (r1, r2), cosine_dist in candidate_pairs.items():
        n1 = name_map[r1]
        n2 = name_map[r2]
        cosine_sim = 1.0 - cosine_dist

        if n1 == n2:
            match_edges.append((id_map[r1], id_map[r2], 1.0))
            continue
        if n1.lower() == n2.lower():
            match_edges.append((id_map[r1], id_map[r2], 0.9))
            continue

        jw = jaro_winkler(n1.lower(), n2.lower())
        score = 0.4 * jw + 0.6 * cosine_sim

        if score > llm_high:
            match_edges.append((id_map[r1], id_map[r2], score))
        elif score >= llm_low:
            borderline_pairs.append((r1, r2))

    # Find connected components in the borderline pair graph
    components = _connected_components(borderline_pairs)
    llm_calls = 0
    llm_time = 0.0

    for component in components:
        if len(component) < 2:
            continue
        # Build numbered candidate list
        comp_list = sorted(component)
        num_to_rid = {i + 1: rid for i, rid in enumerate(comp_list)}
        candidate_list = "\n".join(f"  {i + 1}. {name_map[rid]}" for i, rid in enumerate(comp_list))

        prompt = ER_CLUSTER_NUM_PROMPT.format(candidate_list=candidate_list)
        max_tokens = 30 * len(comp_list)

        t0 = time.perf_counter()
        result = conn.execute(
            "SELECT muninn_chat(?, ?, ?, ?, ?)",
            (model_name, prompt, GBNF_ER_CLUSTER_NUM, max_tokens, ER_CLUSTER_NUM_SYSTEM),
        ).fetchone()[0]
        llm_time += time.perf_counter() - t0
        llm_calls += 1

        # Parse groups and create edges
        groups = _parse_cluster_num_result(result, num_to_rid, id_map)
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    match_edges.append((group[i], group[j], 0.85))

    log.info(
        "LLM-cluster: %d edges, %d LLM calls (%.2fs), %d components",
        len(match_edges),
        llm_calls,
        llm_time,
        len(components),
    )
    return _leiden_cluster(conn, entities, match_edges)


def _connected_components(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Find connected components in an undirected pair graph via BFS."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    components: list[list[int]] = []
    for node in adj:
        if node in visited:
            continue
        component: list[int] = []
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            component.append(n)
            queue.extend(adj[n])
        components.append(component)
    return components


def _parse_cluster_num_result(
    result: str,
    num_to_rid: dict[int, int],
    id_map: dict[int, str],
) -> list[list[str]]:
    """Parse numbered cluster result into groups of entity_ids."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        log.warning("Failed to parse cluster result: %s", result[:100])
        return []

    groups: list[list[str]] = []
    if "groups" not in parsed or not isinstance(parsed["groups"], list):
        return []

    for group in parsed["groups"]:
        if not isinstance(group, list) or len(group) < 2:
            continue
        entity_ids: list[str] = []
        for num in group:
            num_int = int(num) if isinstance(num, (int, float, str)) else None
            if num_int and num_int in num_to_rid:
                entity_ids.append(id_map[num_to_rid[num_int]])
        if len(entity_ids) >= 2:
            groups.append(entity_ids)
    return groups


def _parse_pairwise_result(result: str) -> dict | None:
    """Parse pairwise LLM result, handling possible formatting issues."""
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        log.warning("Failed to parse LLM result: %s", result[:100])
        return None


# ── Reporting ─────────────────────────────────────────────────────


def _cluster_sizes(clusters: dict[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for cid in clusters.values():
        sizes[cid] += 1
    return dict(sizes)


def _f1_str(m: dict[str, float]) -> str:
    return f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"


def report_metrics(
    predicted: dict[str, int],
    gold: dict[str, int],
    elapsed: float,
    label: str,
) -> dict[str, dict[str, float] | float]:
    """Compute and print metrics for a pipeline run."""
    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)

    pred_sizes = _cluster_sizes(predicted)
    n_pred_clusters = len(pred_sizes)
    n_multi = sum(1 for sz in pred_sizes.values() if sz > 1)

    print(f"\n  ── {label} ─────────────────────────────────────")
    print(f"  Entities:     {len(predicted)}")
    print(f"  Clusters:     {n_pred_clusters} ({n_multi} multi-member)")
    print(f"  B-Cubed:      {_f1_str(bc)}")
    print(f"  Pairwise:     {_f1_str(pw)}")
    print(f"  Wall clock:   {elapsed:.2f}s")

    return {"bcubed": bc, "pairwise": pw, "elapsed": elapsed}


# ── Result Persistence ────────────────────────────────────────────


def _save_result(
    pipeline: str,
    model: str,
    limit: int | None,
    n_entities: int,
    bc: dict[str, float],
    pw: dict[str, float],
    elapsed: float,
    llm_calls: int = 0,
) -> Path:
    """Save a single benchmark result to RESULTS_DIR as JSON. Returns file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    slug = f"{pipeline}__{model}__{limit or 'full'}"
    filename = f"{ts}__{slug}.json"
    result = {
        "timestamp": ts,
        "pipeline": pipeline,
        "model": model,
        "limit": limit,
        "n_entities": n_entities,
        "bcubed_f1": bc["f1"],
        "bcubed_precision": bc["precision"],
        "bcubed_recall": bc["recall"],
        "pairwise_f1": pw["f1"],
        "pairwise_precision": pw["precision"],
        "pairwise_recall": pw["recall"],
        "elapsed_s": round(elapsed, 3),
        "llm_calls": llm_calls,
    }
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("Saved result: %s", path.name)
    return path


def _load_results() -> list[dict]:
    """Load all JSON result files from RESULTS_DIR, sorted by timestamp."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            log.warning("Skipping malformed result: %s", p.name)
    return results


# ── CLI Commands ──────────────────────────────────────────────────


def _resolve_model(args: argparse.Namespace) -> ChatModelConfig:
    """Resolve --model flag to a ChatModelConfig."""
    name = getattr(args, "model", DEFAULT_CHAT_MODEL.name)
    for m in CHAT_MODELS:
        if m.name == name:
            return m
    raise SystemExit(f"Unknown model: {name}. Available: {', '.join(m.name for m in CHAT_MODELS)}")


def cmd_string_only(args: argparse.Namespace) -> None:
    """Run string-only ER pipeline on Abt-Buy."""
    entities, gold = load_dataset(args.limit)
    conn = create_db()

    t0 = time.perf_counter()
    predicted = run_string_only(conn, entities)
    elapsed = time.perf_counter() - t0

    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)
    report_metrics(predicted, gold, elapsed, "String-Only")
    _save_result("string-only", "-", args.limit, len(entities), bc, pw, elapsed)
    conn.close()


def cmd_llm_tiered(args: argparse.Namespace) -> None:
    """Run LLM-pairwise ER pipeline on Abt-Buy with a single model."""
    model = _resolve_model(args)
    entities, gold = load_dataset(args.limit)
    conn = create_db(chat_model=model)

    t0 = time.perf_counter()
    predicted = run_llm_pairwise(conn, entities, model_name=model.name)
    elapsed = time.perf_counter() - t0

    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)
    report_metrics(predicted, gold, elapsed, f"LLM-Pairwise ({model.name})")
    _save_result("pairwise", model.name, args.limit, len(entities), bc, pw, elapsed)
    conn.close()


def cmd_compare(args: argparse.Namespace) -> None:
    """Full permutation matrix: {string-only, llm-pairwise, llm-cluster} x {models}.

    String-only runs once as baseline. Each LLM grammar format is tested across
    all 3 chat models. Results saved to results/ — run `analyse` to view the table.
    """
    entities, gold = load_dataset(args.limit)
    n_perms = 1 + 2 * len(CHAT_MODELS)
    print(f"\n  Running {n_perms} permutations: 1 baseline + {len(CHAT_MODELS)} models x 2 grammar formats")

    # Single connection throughout: muninn uses a global C model registry,
    # so creating multiple connections would trigger 'model already loaded'.
    conn = create_db()

    # 1. String-only baseline
    t0 = time.perf_counter()
    pred = run_string_only(conn, entities)
    elapsed = time.perf_counter() - t0
    bc = bcubed_f1(pred, gold)
    pw = pairwise_f1(pred, gold)
    report_metrics(pred, gold, elapsed, "String-Only")
    _save_result("string-only", "-", args.limit, len(entities), bc, pw, elapsed)

    # 2. LLM permutations: {pairwise, cluster} x {models}
    # Load one chat model at a time to conserve memory (~1-3 GB each)
    pipelines = [
        ("pairwise", run_llm_pairwise),
        ("cluster", run_llm_cluster),
    ]
    for model in CHAT_MODELS:
        register_chat_model(conn, model)
        for grammar_label, pipeline_fn in pipelines:
            cleanup_pipeline_tables(conn)
            label = f"{grammar_label}/{model.name}"
            log.info("Running: %s", label)
            t0 = time.perf_counter()
            pred = pipeline_fn(conn, entities, model_name=model.name)
            elapsed = time.perf_counter() - t0
            bc = bcubed_f1(pred, gold)
            pw = pairwise_f1(pred, gold)
            report_metrics(pred, gold, elapsed, label)
            _save_result(grammar_label, model.name, args.limit, len(entities), bc, pw, elapsed)
        # Unregister chat model to free memory before loading the next one
        conn.execute("DELETE FROM temp.muninn_chat_models WHERE name = ?", (model.name,))
        log.info("Unloaded chat model: %s", model.name)

    conn.close()
    log.info("All results saved to %s — run 'analyse' to view comparison table", RESULTS_DIR)


def cmd_analyse(args: argparse.Namespace) -> None:
    """Print comparison table from all accumulated result files."""
    results = _load_results()
    if not results:
        log.warning("No results found in %s — run some benchmarks first", RESULTS_DIR)
        return

    # Group by (limit, pipeline, model), keep latest per group
    latest: dict[tuple[str, str, int | None], dict] = {}
    for r in results:
        key = (r["pipeline"], r["model"], r.get("limit"))
        latest[key] = r  # Last-wins = newest timestamp (files sorted by name)

    rows = sorted(latest.values(), key=lambda r: (r.get("limit") or 99999, r["pipeline"], r["model"]))

    # Table header
    print(f"\n  {'=' * 105}")
    print(
        f"  {'Pipeline':<16} {'Model':<14} {'Limit':>6} {'N':>6} "
        f"{'B³ F1':>8} {'B³ P':>8} {'B³ R':>8} {'PW F1':>8} {'LLM#':>5} {'Time':>8}"
    )
    print(f"  {'-' * 105}")

    # Find baseline B³ F1 per limit for delta calculation
    baselines: dict[int | None, float] = {}
    for r in rows:
        if r["pipeline"] == "string-only":
            baselines[r.get("limit")] = r["bcubed_f1"]

    for r in rows:
        pipeline = r["pipeline"]
        model = r["model"]
        limit = r.get("limit")
        limit_str = str(limit) if limit else "full"
        baseline = baselines.get(limit, 0.0)
        delta = f"({r['bcubed_f1'] - baseline:+.3f})" if pipeline != "string-only" else ""
        print(
            f"  {pipeline:<16} {model:<14} {limit_str:>6} {r['n_entities']:>6} "
            f"{r['bcubed_f1']:>8.4f} {r['bcubed_precision']:>8.4f} {r['bcubed_recall']:>8.4f} "
            f"{r['pairwise_f1']:>8.4f} {r['llm_calls']:>5} {r['elapsed_s']:>7.1f}s {delta}"
        )

    print(f"  {'=' * 105}")
    print(f"\n  {len(rows)} results from {RESULTS_DIR}/")
    if getattr(args, "verbose", False):
        print(f"\n  Latest per group (total files: {len(results)}):")


def cmd_grammar_debug(args: argparse.Namespace) -> None:
    """Test GBNF grammar formats on a small set of entity pairs.

    Three tiers per format:
      raw     — no grammar, see raw LLM output
      grammar — with GBNF grammar, validate constrained output
      oneshot — grammar + one-shot example in prompt
    """
    model = _resolve_model(args)
    entities, _ = load_dataset(args.limit)
    conn = create_db(chat_model=model)

    # Use shared blocking to find interesting pairs
    _id_map, name_map, candidate_pairs = _embed_and_block(conn, entities)

    test_pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[int, int]] = set()
    for (r1, r2), dist in sorted(candidate_pairs.items(), key=lambda x: x[1]):
        if (r1, r2) not in seen and 0.05 < dist < 0.6:
            seen.add((r1, r2))
            test_pairs.append((name_map[r1], name_map[r2], dist))
            if len(test_pairs) >= 5:
                break

    if not test_pairs:
        log.warning("No borderline pairs found — try a larger --limit")
        conn.close()
        return

    grammar_format = args.grammar_format
    grammar_tier = args.grammar_tier

    print(f"\n{'=' * 70}")
    print(f"  Grammar Debug: format={grammar_format}, tier={grammar_tier}, model={model.name}")
    print(f"  Testing {len(test_pairs)} pairs")
    print(f"{'=' * 70}")

    success = 0
    total = len(test_pairs)

    if grammar_format == "pairwise":
        grammar = GBNF_ER_PAIRWISE if grammar_tier != "raw" else None
        for i, (n1, n2, dist) in enumerate(test_pairs):
            if grammar_tier == "oneshot":
                prompt = ER_PAIRWISE_ONESHOT.format(name_a=n1, name_b=n2)
            else:
                prompt = ER_PAIRWISE_PROMPT.format(name_a=n1, name_b=n2)

            print(f"\n  Pair {i + 1}: cosine_dist={dist:.3f}")
            print(f"    A: {n1[:80]}")
            print(f"    B: {n2[:80]}")

            if grammar:
                result = conn.execute(
                    "SELECT muninn_chat(?, ?, ?, ?, ?)",
                    (model.name, prompt, grammar, 50, ER_PAIRWISE_SYSTEM),
                ).fetchone()[0]
            else:
                result = conn.execute(
                    "SELECT muninn_chat(?, ?, NULL, ?, ?)",
                    (model.name, prompt, 100, ER_PAIRWISE_SYSTEM),
                ).fetchone()[0]

            print(f"    Output: {result[:200]}")
            parsed = _parse_pairwise_result(result)
            if parsed and "match" in parsed:
                print(f"    Parsed: match={parsed['match']}, confidence={parsed.get('confidence', '?')}")
                success += 1
            else:
                print("    Parsed: FAILED")

    elif grammar_format == "cluster":
        grammar = GBNF_ER_CLUSTER if grammar_tier != "raw" else None
        names: list[str] = []
        for n1, n2, _ in test_pairs:
            if n1 not in names:
                names.append(n1)
            if n2 not in names:
                names.append(n2)
        candidate_list = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(names))

        if grammar_tier == "oneshot":
            prompt = ER_CLUSTER_ONESHOT.format(candidate_list=candidate_list)
        else:
            prompt = ER_CLUSTER_PROMPT.format(candidate_list=candidate_list)

        print(f"\n  Candidates ({len(names)}):")
        for i, n in enumerate(names):
            print(f"    {i + 1}. {n[:80]}")

        max_tokens = 50 * len(names)
        if grammar:
            result = conn.execute(
                "SELECT muninn_chat(?, ?, ?, ?, ?)",
                (model.name, prompt, grammar, max_tokens, ER_CLUSTER_SYSTEM),
            ).fetchone()[0]
        else:
            result = conn.execute(
                "SELECT muninn_chat(?, ?, NULL, ?, ?)",
                (model.name, prompt, max_tokens, ER_CLUSTER_SYSTEM),
            ).fetchone()[0]

        print(f"\n  Output: {result[:500]}")
        try:
            parsed = json.loads(result)
            if "groups" in parsed and isinstance(parsed["groups"], list):
                print(f"  Parsed: {len(parsed['groups'])} groups")
                for gi, group in enumerate(parsed["groups"]):
                    print(f"    Group {gi + 1}: {group}")
                success = 1
            else:
                print("  Parsed: FAILED (missing 'groups' key)")
        except (json.JSONDecodeError, TypeError):
            print("  Parsed: FAILED (invalid JSON)")
        total = 1

    print(f"\n  Result: {success}/{total} successful parses")
    conn.close()


# ── CLI Parser ────────────────────────────────────────────────────


def _help(p: argparse.ArgumentParser):
    """Return a handler that prints help for parser p."""

    def _print_help(_: argparse.Namespace) -> None:
        p.print_help()

    return _print_help


MODEL_NAMES = [m.name for m in CHAT_MODELS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entity Resolution Benchmark — String-Only vs LLM-Tiered on Abt-Buy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=_help(parser))
    sub = parser.add_subparsers(dest="command", required=False)

    # string-only
    p = sub.add_parser("string-only", help="Run string-only ER pipeline (HNSW + JW + Leiden)")
    p.add_argument("--limit", type=int, default=None, help="Max entities to load (default: all)")
    p.set_defaults(func=cmd_string_only)

    # llm-tiered (pairwise grammar, single model)
    p = sub.add_parser("llm-tiered", help="Run LLM-pairwise ER pipeline (string + muninn_chat borderline)")
    p.add_argument("--limit", type=int, default=None, help="Max entities to load (default: all)")
    p.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_CHAT_MODEL.name, help="Chat model to use")
    p.set_defaults(func=cmd_llm_tiered)

    # compare (full permutation matrix: {string-only, pairwise, cluster} x {models})
    p = sub.add_parser("compare", help="Run all pipelines x models, save results to results/")
    p.add_argument("--limit", type=int, default=None, help="Max entities to load (default: all)")
    p.set_defaults(func=cmd_compare)

    # analyse (read accumulated results and print comparison table)
    p = sub.add_parser("analyse", help="Print comparison table from accumulated results")
    p.set_defaults(func=cmd_analyse)

    # Grammar debug subcommands (2 formats x 3 tiers = 6)
    for fmt in ["pairwise", "cluster"]:
        for tier in ["raw", "grammar", "oneshot"]:
            name = f"grammar-{fmt}-{tier}"
            help_text = f"Grammar debug: {fmt} format, {tier} tier"
            p = sub.add_parser(name, help=help_text)
            p.add_argument("--limit", type=int, default=20, help="Entities to load for pair discovery (default: 20)")
            p.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_CHAT_MODEL.name, help="Chat model to use")
            p.set_defaults(func=cmd_grammar_debug, grammar_format=fmt, grammar_tier=tier)

    return parser


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
