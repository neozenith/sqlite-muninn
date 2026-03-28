# Entity Resolution Benchmark v2

Manifest-driven ER benchmark comparing string-only and LLM-cluster pipelines
across three DeepMatcher datasets and four chat models.

## Datasets

| Dataset | Domain | Entities | Source |
|---------|--------|----------|--------|
| Abt-Buy | E-commerce products | ~2,173 | Abt.com vs Buy.com |
| Amazon-Google | E-commerce products | ~4,589 | Amazon vs Google Products |
| DBLP-ACM | Academic papers | ~4,910 | DBLP vs ACM Digital Library |

## Pipelines

| Pipeline | Description | LLM? |
|----------|-------------|------|
| `string-only` | HNSW blocking + JW/cosine cascade + Leiden | No |
| `llm-cluster` | Same blocking, borderline components sent to muninn_chat() + Leiden | Yes |

## Models (for llm-cluster)

| Model | Params | Size |
|-------|--------|------|
| Qwen3.5-2B | 2B | 1.3 GB |
| Qwen3.5-4B | 4B | 2.7 GB |
| Gemma-3-1B | 1B | 0.8 GB |
| Gemma-3-4B | 4B | 2.5 GB |

## Usage

```bash
# See all permutations and their status
uv run -m examples.er_v2 manifest

# See what's missing, cheapest first
uv run -m examples.er_v2 manifest --missing --limit 10

# Generate runnable commands for missing permutations
uv run -m examples.er_v2 manifest --missing --commands

# Run a single permutation
uv run -m examples.er_v2 run --dataset abt-buy --pipeline string-only --limit 100
uv run -m examples.er_v2 run --dataset abt-buy --pipeline llm-cluster --model Qwen3.5-4B --limit 100

# Run full dataset (omit --limit)
uv run -m examples.er_v2 run --dataset dblp-acm --pipeline string-only

# View comparison table from all accumulated results
uv run -m examples.er_v2 analyse
```

## Workflow

1. `manifest --missing --commands` generates one command per missing permutation
2. Run them individually or pipe to `bash` / `parallel`
3. `analyse` reads whatever results have accumulated and prints the comparison table
4. Re-run any permutation with `--force` to overwrite

## File Organisation

```
er_v2/
├── __main__.py      # CLI: manifest, run, analyse
├── registry.py      # Permutation matrix definition
├── datasets.py      # Dataset configs, download, ground truth
├── models.py        # GGUF model configs, DB setup
├── blocking.py      # HNSW embed + KNN blocking (common pre-step)
├── string_only.py   # String-only matching (isolated implementation)
├── llm_cluster.py   # LLM cluster matching (isolated implementation)
├── metrics.py       # B-Cubed F1, Pairwise F1 (evaluation post-step)
├── jaro_winkler.py  # String similarity function
└── results/         # Output: {permutation_id}.json
```
