# Entity Resolution Benchmarks

Captured: 2026-02-16. Split from `kg_extraction_benchmark_spec.md`.

This document specifies the benchmark datasets, metrics, and evaluation methodology for entity resolution (coalescing) — the task of determining which extracted entities refer to the same real-world thing.

---

## Table of Contents

1. [Why Separate Benchmarks](#1-why-separate-benchmarks)
2. [Dedicated ER Datasets](#2-dedicated-er-datasets)
3. [ER Metrics](#3-er-metrics)
4. [What Our Pipeline Already Does](#4-what-our-pipeline-already-does)
5. [Dataset Download URLs](#5-dataset-download-urls)
6. [References](#references)

---

## 1. Why Separate Benchmarks

Entity extraction (NER) and entity resolution (coalescing) are **different tasks** evaluated separately:

- **NER:** "What spans in text are entities?" — measured by span-level F1
- **ER:** "Which extracted entities refer to the same real-world thing?" — measured by pairwise F1 and B-Cubed F1

The NER datasets (CoNLL-2003, etc.) have **no entity resolution annotations** and cannot be repurposed for coalescing benchmarks. Exception: OntoNotes 5.0 has within-document coreference annotations, but requires LDC access.

---

## 2. Dedicated ER Datasets

### Tier 1 — Tiny, free, zero setup (start here)

| Dataset | Entities | True Matches | Domain | Access |
|---------|----------|-------------|--------|--------|
| **Febrl 1** | 1,000 records | 500 pairs | Synthetic person names | `pip install recordlinkage` → `load_febrl1()` |
| **Febrl 4** | 10,000 records | 5,000 pairs | Synthetic person names | Same package → `load_febrl4a()`, `load_febrl4b()` |
| **BeerAdvo-RateBeer** | 450 pairs | 68 matches | Beer reviews | Direct download, <1MB |
| **Fodors-Zagats** | 946 pairs | 110 matches | Restaurants | Direct download, <1MB |

### Tier 2 — Small, free, directly analogous to KG coalescing

| Dataset | Entities | True Matches | Domain | Access | Why Relevant |
|---------|----------|-------------|--------|--------|-------------|
| **DBLP-ACM** | ~5K entities | 2,224 matches | Bibliographic (name variations) | Leipzig direct download | Name matching like KG entities |
| **Affiliations** | 2,260 entities | 330 clusters | Organization names | Leipzig direct download | **Closest to KG entity clustering** |
| **Abt-Buy** | 2,173 entities | 1,097 matches | E-commerce products | Leipzig direct download | Textual similarity matching |
| **MusicBrainz 20K** | 19K entities, 5 sources | 10K clusters | Music metadata | Leipzig direct download | Multi-source merging |

### Tier 3 — KG-specific

| Dataset | Size | Domain | Access | Why Relevant |
|---------|------|--------|--------|-------------|
| **MovieGraphBenchmark** | Varies | Movie KGs (IMDB/TMDB/TVDB) | `pip install moviegraphbenchmark` | **Actual KG entity resolution** across heterogeneous sources |
| **DBP15K** | ~15K aligned pairs | Cross-lingual DBpedia | GitHub | KG entity alignment with embeddings |
| **AIDA-CoNLL** | 1,393 docs | Entity linking to YAGO | Max Planck Institute | Entity linking (mention → KB entry) |

---

## 3. ER Metrics

| Metric | Description | Package |
|--------|-------------|---------|
| **Pairwise F1** | Of all entity pairs, which were correctly merged/not merged | `recordlinkage` |
| **B-Cubed F1** | Per-entity: what fraction of its cluster members are correct | Custom or `coval` |
| **Cluster purity** | Fraction of dominant class in each cluster | Custom |

---

## 4. What Our Pipeline Already Does

Current coalescing in `kg_coalesce.py`:

| Stage | Method | Corresponds To |
|-------|--------|---------------|
| 1. HNSW Blocking | KNN on entity embeddings, cosine < 0.4 | Standard ER blocking step |
| 2. Matching Cascade | Exact → substring → Jaro-Winkler → cosine | Standard ER matching step |
| 3. Leiden Clustering | `graph_leiden` on match edges | Standard ER clustering step |

**Raw vs coalesced data:** Both are preserved in the same DB:
- Raw: `entities` table + `relations` table
- Mapping: `entity_clusters` table (name → canonical)
- Clean: `nodes` table + `edges` table (canonical names, aggregated weights)

---

## 5. Dataset Download URLs

| Dataset | URL |
|---------|-----|
| Leipzig Benchmarks (DBLP-ACM, Abt-Buy, Affiliations, etc.) | https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution |
| DeepMatcher Benchmarks (BeerAdvo, Fodors, iTunes, etc.) | https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md |
| Febrl (built-in) | `pip install recordlinkage` |
| MovieGraphBenchmark | `pip install moviegraphbenchmark` |
| DBP15K | https://github.com/nju-websoft/OpenEA |
| WDC Products | https://webdatacommons.org/largescaleproductcorpus/wdc-products/ |

---

## References

### Entity Resolution
- Febrl (Christen & Pudjijono, 2008)
- DeepMatcher (Mudgal et al., SIGMOD 2018)
- Ditto (Li et al., VLDB 2020)
- MovieGraphBenchmark (ScaDS Leipzig, 2023)
