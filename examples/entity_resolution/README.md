# Entity Resolution Benchmark

Evaluates muninn's entity resolution pipeline on the **Abt-Buy** dataset
(DeepMatcher benchmark, ~2,173 product entities across Abt.com and Buy.com).

## Features

| Feature | Description |
|---------|-------------|
| **String-Only Pipeline** | HNSW blocking + Jaro-Winkler/cosine matching cascade + Leiden clustering |
| **LLM-Pairwise Pipeline** | Same blocking, borderline pairs routed 1:1 to `muninn_chat()` with GBNF grammar |
| **LLM-Cluster Pipeline** | Same blocking, borderline neighborhoods clustered by LLM using numbered-index grammar |
| **Permutation Matrix** | `compare` runs `{string-only, pairwise, cluster}` x `{Qwen3.5-2B, Qwen3.5-4B, Gemma-3-4B}` |
| **GBNF Grammar Debug** | 6 test configurations (2 formats x 3 tiers) for empirical grammar evaluation |
| **B-Cubed + Pairwise F1** | Standard ER metrics computed against ground truth clusters |
| **Logarithmic Scale-Up** | `--limit N` with tiers 10/100/1000/all for fast iteration |

## Setup

```bash
# Build the muninn extension
make all

# GGUF models are auto-downloaded on first run:
#   - all-MiniLM-L6-v2 (~23 MB) for embeddings
#   - Qwen3.5-2B (~1.3 GB) for LLM tier (only when needed)
```

## Usage

### Individual Modes (composable, each works in isolation)

Each run saves a JSON result to `results/`. Run any permutation independently, across sessions.

```bash
# String-only baseline — fast, no chat model needed
uv run examples/entity_resolution/er_benchmark.py string-only --limit 10
uv run examples/entity_resolution/er_benchmark.py string-only --limit 100
uv run examples/entity_resolution/er_benchmark.py string-only              # full dataset

# LLM-pairwise — one muninn_chat() call per borderline pair
uv run examples/entity_resolution/er_benchmark.py llm-tiered --limit 100
uv run examples/entity_resolution/er_benchmark.py llm-tiered --limit 100 --model Gemma-3-4B
```

### Comparison Mode (full permutation matrix)

Runs all 7 permutations and saves each to `results/`:

```bash
uv run examples/entity_resolution/er_benchmark.py compare --limit 100
```

### Analyse (read accumulated results)

Prints a comparison table from all JSON results accumulated to date:

```bash
uv run examples/entity_resolution/er_benchmark.py analyse
```

```
Pipeline         Model           Limit      N    B3 F1     B3 P     B3 R    PW F1  LLM#     Time
-------------------------------------------------------------------------------------------------
string-only      -                 100    100   0.7474   0.5967   1.0000   0.1905     0     0.7s
pairwise         Qwen3.5-2B       100    100   ...      ...      ...      ...    22     17.8s (+0.000)
cluster          Qwen3.5-2B       100    100   ...      ...      ...      ...     3      5.2s (+0.000)
...
```

### Grammar Debug (G6 — empirical GBNF evaluation)

Two grammar formats tested at three debug tiers:

```bash
# Format A — Pairwise: {"match": true, "confidence": 0.95}
uv run examples/entity_resolution/er_benchmark.py grammar-pairwise-raw --limit 20
uv run examples/entity_resolution/er_benchmark.py grammar-pairwise-grammar --limit 20
uv run examples/entity_resolution/er_benchmark.py grammar-pairwise-oneshot --limit 20

# Format B — Clustering: {"groups": [["NYC", "New York City"], ["London"]]}
uv run examples/entity_resolution/er_benchmark.py grammar-cluster-raw --limit 20
uv run examples/entity_resolution/er_benchmark.py grammar-cluster-grammar --limit 20
uv run examples/entity_resolution/er_benchmark.py grammar-cluster-oneshot --limit 20
```

## Scale-Up Tiers

| Tier | Entities | Purpose | Expected Runtime |
|------|----------|---------|-----------------|
| `--limit 10` | 10 | Smoke test — schema, I/O, metric plumbing | < 5s |
| `--limit 100` | 100 | Algorithm correctness — blocking, matching, clustering | ~30s |
| `--limit 1000` | 1,000 | Performance profiling — LLM call count, latency | ~minutes |
| (no limit) | ~2,173 | Full evaluation — publishable F1 for SOTA comparison | ~minutes |

## Pipeline Architecture

```
Abt-Buy CSVs (tableA + tableB)
    │
    ▼
Entity Embedding (muninn_embed → MiniLM 384d)
    │
    ▼
HNSW Blocking (k=10, cosine_dist ≤ 0.4)
    │
    ▼
┌────────────────────────────────────────────┐
│  Matching Cascade                          │
│  ┌──────────────────────┐                  │
│  │ Tier 1: Exact/icase  │──→ accept (1.0)  │
│  └──────────────────────┘                  │
│  ┌──────────────────────┐                  │
│  │ Tier 2: JW + cosine  │──→ score > 0.7?  │──→ accept
│  │  0.4*JW + 0.6*cos    │──→ score < 0.3?  │──→ reject
│  └──────────────────────┘                  │
│  ┌──────────────────────┐                  │
│  │ Tier 3: LLM (border) │──→ muninn_chat() │  (llm-tiered only)
│  │  GBNF grammar JSON   │     + GBNF       │
│  └──────────────────────┘                  │
└────────────────────────────────────────────┘
    │
    ▼
Leiden Clustering (graph_leiden TVF)
    │
    ▼
B-Cubed F1 + Pairwise F1 vs Ground Truth
```

## Dataset

The [Abt-Buy dataset](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)
from the DeepMatcher benchmark suite (University of Wisconsin). Products from
Abt.com and Buy.com with ground truth matching labels.

Auto-downloaded and cached in `tmp/er_benchmark/abt_buy/` on first run.

## References

- [MatchGPT (arXiv:2310.11244)](https://arxiv.org/abs/2310.11244) — LLM entity matching evaluation
- [GoldenMatch (2025)](https://github.com/benzsevern/goldenmatch) — LLM for borderline pairs only
- [LLM-CER (SIGMOD 2026)](https://arxiv.org/abs/2506.02509) — In-context clustering ER
- [GraLMatch (EDBT 2025)](https://arxiv.org/abs/2406.15015) — Graph cleanup via edge betweenness
- [DeepMatcher Datasets](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)
