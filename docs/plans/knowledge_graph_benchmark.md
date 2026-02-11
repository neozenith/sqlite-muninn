# Knowledge Graph Benchmark Plan

A benchmark for **combined vector search + graph traversal** workflows — the GraphRAG pattern — exercising all three of vec_graph's subsystems (HNSW, graph TVFs, Node2Vec) together.

**Status:** Plan only. Not implemented. Requires primitive benchmarks (vector search + graph traversal) to be validated first.

---

## Table of Contents

1. [Vision](#vision)
2. [Research: State of the Art](#research-state-of-the-art)
   - [Vector-Seeded Graph Traversal](#vector-seeded-graph-traversal-the-graphrag-pattern)
   - [Graph Algorithms (extracted)](#graph-algorithms-community-detection-centrality-node2vec)
   - [Entity Coalescing and Synonym Merging](#entity-coalescing-and-synonym-merging)
   - [Temporal Knowledge Graphs](#temporal-knowledge-graphs)
3. [Entity Extraction Approaches](#entity-extraction-approaches)
   - [NER Models for Entity Extraction](#approach-1-ner-models-for-entity-extraction)
   - [Noun-Verb-Noun SVO Pattern](#approach-2-noun-verb-noun-svo-pattern-extraction)
   - [FTS5/BM25 for Concept Discovery](#approach-3-fts5bm25-for-concept-discovery)
   - [Hybrid Pipeline Recommendation](#hybrid-pipeline-recommendation)
4. [HuggingFace Model Recommendations](#huggingface-model-recommendations)
5. [Dataset: Economics Texts from Project Gutenberg](#dataset-economics-texts-from-project-gutenberg)
6. [Benchmark Workflow](#benchmark-workflow)
7. [Metrics](#metrics)
8. [New Graph Capabilities Required](#new-graph-capabilities-required)
9. [Key References and Prior Art](#key-references-and-prior-art)
10. [Prerequisites](#prerequisites)
11. [Implementation Sketch](#implementation-sketch)
12. [Open Questions](#open-questions)

---

## Vision

Current benchmarks test each subsystem in isolation:

- **Vector search benchmarks** compare HNSW vs brute-force at varying N and dimension
- **Graph traversal benchmarks** compare TVFs vs CTEs vs GraphQLite on synthetic topologies

The knowledge graph benchmark tests the *composition* — the workflow where a vector similarity search provides an entry point into a graph, and graph traversal expands the context. This is the core pattern behind GraphRAG, where:

1. **VSS entry point**: Query vector -> HNSW search -> find nearest graph node
2. **Graph expansion**: From that node -> BFS/DFS to explore k-hop neighbors
3. **Context assembly**: Collect text from traversed nodes as retrieval context
4. **Graph analytics**: Betweenness centrality identifies bridge concepts; PageRank finds authoritative nodes
5. **Hierarchical retrieval** (advanced): Louvain/Leiden communities -> supernode embeddings -> multi-level search

vec_graph is uniquely positioned here — it's the only SQLite extension combining HNSW + graph TVFs + Node2Vec in a single shared library. The benchmark would demonstrate whether this integration provides measurable advantages over stitching separate tools together.

### What We Already Have

The vec_graph extension currently implements:

| Capability | TVF/Function | Status |
|-----------|-------------|--------|
| BFS traversal | `graph_bfs` | Done |
| DFS traversal | `graph_dfs` | Done |
| Shortest path (Dijkstra) | `graph_shortest_path` | Done |
| Connected components | `graph_components` (Union-Find) | Done |
| PageRank | `graph_pagerank` (power method, configurable damping/iterations) | Done |
| Node2Vec embeddings | `node2vec_train()` (biased walks + SGNS) | Done |
| HNSW vector index | `hnsw_index` virtual table | Done |

What's **missing** for the full KG benchmark:
- Betweenness centrality (needed for identifying bridge concepts)
- Community detection (Louvain/Leiden — currently only connected components)
- Degree centrality / closeness centrality (nice-to-have)

---

## Research: State of the Art

### Vector-Seeded Graph Traversal (The GraphRAG Pattern)

The core insight behind GraphRAG is that **vector similarity search alone misses relational context**. A query about "How does the division of labour affect wages?" might find passages about wages but miss causally-linked concepts like "productivity" or "market price" that are connected via graph edges but not semantically similar to the query.

The pattern works in stages:

```
Query "How does division of labour affect wages?"
  │
  ├─ [1] VSS Entry Point ──────── HNSW search → nearest passage nodes
  │                                (finds: "wages", "labour", "price")
  │
  ├─ [2] Graph Expansion ──────── BFS 2-hop from entry points
  │                                (discovers: "productivity", "market price",
  │                                 "rent", "profit" via CAUSES/COMPOSED_OF edges)
  │
  ├─ [3] Centrality Ranking ───── Betweenness centrality scores bridge nodes
  │                                (ranks "market price" highest — it bridges
  │                                 labour/wages cluster to rent/profit cluster)
  │
  └─ [4] Context Assembly ─────── Collect passage text from traversed+ranked nodes
                                   (richer context than VSS alone)
```

**Key research systems:**

- **Microsoft GraphRAG (2024)**: Extracts KG from documents via LLM, builds community hierarchy (Leiden algorithm), generates community summaries, retrieval via summary search → drill into communities. The gold standard for hierarchical GraphRAG.
- **HybridRAG (2024)**: Combines vector DB + graph DB with joint scoring (vector similarity + graph distance). Exactly what vec_graph enables in a single SQLite extension.
- **NaviX (VLDB 2025)**: Native dual indexing in the DB kernel — vector index + graph index with pruning strategies that leverage both simultaneously. The research frontier.
- **Deep GraphRAG (2025)**: Multi-hop reasoning via graph-guided evidence chains starting from VSS results.

**Why this matters for vec_graph**: Unlike systems that stitch together separate vector and graph databases, vec_graph can do VSS → graph traversal → Node2Vec in a single SQLite connection. This eliminates serialization overhead and enables joined queries that reference both indexes.

---

### Graph Algorithms (Community Detection, Centrality, Node2Vec)

> **Full research and implementation plans extracted to [Graph Algorithms Plan](graph_algorithms.md).**
>
> That document covers:
> - **Community detection**: Louvain algorithm (modularity optimization, O(n log n)), Leiden refinement (guaranteed connectivity), implementation sketches (~300 lines C)
> - **Centrality measures**: Betweenness (Brandes' O(VE) algorithm), closeness, degree centrality — TVF interfaces and implementation plans
> - **Node2Vec deep dive**: How our p,q biased walks work, comparison with DeepWalk/LINE/GraphSAGE/GAT, why Node2Vec is the right choice for zero-dependency C
> - **Knowledge Graph Embeddings (KGE)**: TransE, RotatE, ComplEx — relation-aware alternatives and why we defer them
>
> **KG benchmark dependencies on these algorithms:**
> - **Phase C** (Graph Analytics) requires `graph_betweenness` and `graph_louvain`
> - **Phase D** (Hierarchical Retrieval) requires `graph_louvain` or `graph_leiden`
> - **Phase E** (Node2Vec Sweep) uses the existing `node2vec_train()`
> - Centrality-guided retrieval (betweenness to prioritize BFS-expanded nodes) vs uniform retrieval is a key benchmark comparison

---

### Entity Coalescing and Synonym Merging

A critical challenge in KG construction: "division of labour", "division of labor", and "the labour is divided" all refer to the same concept. Without merging, the graph becomes fragmented.

#### Coalescing Pipeline

```
Raw Entities  →  Blocking  →  Matching  →  Merging  →  Canonical Graph
(many dupes)    (group by    (pairwise    (resolve     (clean nodes)
                 similarity)  comparison)  clusters)
```

**Stage 1: Blocking** — Group candidate duplicates to avoid O(n²) all-pairs comparison.

- **Embedding-based blocking**: Embed all entity surface forms with a sentence-transformer, then use HNSW to find nearest neighbors. Entities within cosine distance < 0.3 are candidates. **This directly uses our HNSW index!**
- **Token-based blocking (lighter)**: Group entities sharing >50% of tokens after lowercasing and stemming.

**Stage 2: Matching** — Decide which candidates are truly the same entity.

Cascade approach (cheap → expensive):
1. **Exact string match** after normalization (lowercase, strip articles)
2. **Fuzzy string match** (edit distance < 3, or Jaro-Winkler > 0.85)
3. **Word vector similarity** (cosine > 0.8 in the embedding space)
4. **WordNet synsets** (share a synset — e.g., "wages" and "pay")

**Stage 3: Merging** — Resolve connected components of matched pairs.

- Use **graph clustering** (Louvain on the match graph) to prevent over-merging. Pure transitive closure is dangerous: if A≈B and B≈C but A≉C, transitive closure merges all three. Louvain cuts weak edges automatically.
- Select canonical form: most frequent surface form, or the form from the manual seed list.

**For our benchmark**: The HNSW-based blocking step is a compelling use case — it demonstrates vec_graph eating its own dog food. Entity surface forms get embedded, inserted into an HNSW index, then nearest-neighbor search finds candidates for merging. This is a vector-search-to-graph-construction pipeline that exercises the full vec_graph stack.

### Temporal Knowledge Graphs

#### The Problem: Knowledge Changes Over Time

A standard knowledge graph is a snapshot — it tells you what is currently true. But in real-world applications, facts evolve:

- An agent session at 2pm establishes "the auth module uses JWT"
- A session at 4pm discovers "the auth module was refactored to use OAuth"
- A schema change in the data warehouse renames `user_id` to `account_id` on March 1st
- A cloud infrastructure audit shows a VPC peering was added last Tuesday

Without temporal awareness, later facts silently overwrite earlier ones and history is lost. With it, you can ask both "what is true now?" and "what was true at time X?"

#### Bi-Temporal Data Model

The state of the art (used by Zep/Graphiti, arXiv:2501.13956) tracks **two** time dimensions per edge:

| Timestamp | Meaning | Example |
|-----------|---------|---------|
| **Valid time** | When the fact was true in the real world | "Auth module used JWT from Jan to March 2025" |
| **Transaction time** | When the system recorded this fact | "We learned this from session log #47 on Feb 3" |

This enables four classes of queries:

1. **Current state**: What is true right now? (standard graph query)
2. **Point-in-time**: What was true on date X? (valid time filter)
3. **As-of**: What did the system know at time T? (transaction time filter)
4. **Audit trail**: When did fact X enter/leave the graph? (bi-temporal join)

#### Schema Pattern for vec_graph

Temporal awareness does not require changes to the C extension. It's a schema pattern on the edge table, combined with application-level query construction:

```sql
-- Temporal edge table: adds valid_from/valid_to + recorded_at
CREATE TABLE kg_edges_temporal (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    valid_from TEXT NOT NULL,         -- ISO 8601 timestamp
    valid_to TEXT DEFAULT '9999-12-31T23:59:59Z',  -- NULL = still valid
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    source_session TEXT               -- which agent session recorded this
);
CREATE INDEX idx_kg_temporal_src ON kg_edges_temporal(src, valid_to);
CREATE INDEX idx_kg_temporal_dst ON kg_edges_temporal(dst, valid_to);

-- View: current edges only (for standard graph traversal)
CREATE VIEW kg_edges_current AS
SELECT src, dst, relation, weight
FROM kg_edges_temporal
WHERE valid_to >= strftime('%Y-%m-%dT%H:%M:%SZ', 'now');

-- View: edges valid at a specific point in time
-- (parameterized via application code, not possible as a static view)
-- SELECT src, dst, relation, weight
-- FROM kg_edges_temporal
-- WHERE valid_from <= ?1 AND valid_to >= ?1;
```

The existing graph TVFs work unchanged — they accept any edge table, so pointing them at `kg_edges_current` gives current-state traversal, while a point-in-time temp table gives historical traversal:

```sql
-- Current-state traversal (standard)
SELECT node, depth FROM graph_bfs
WHERE edge_table = 'kg_edges_current' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = ? AND max_depth = 2;

-- Point-in-time traversal (via temp table)
CREATE TEMP TABLE kg_edges_at_t AS
SELECT src, dst, relation, weight FROM kg_edges_temporal
WHERE valid_from <= '2025-01-15T00:00:00Z' AND valid_to >= '2025-01-15T00:00:00Z';

SELECT node, depth FROM graph_bfs
WHERE edge_table = 'kg_edges_at_t' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = ? AND max_depth = 2;
```

#### Why This Matters for Agent Memory

When indexing agent session logs, the same fact can appear multiple times with different values across sessions. The bi-temporal model provides:

- **Contradiction resolution**: Session #50 says "uses 3.11", session #55 says "uses 3.12" → not a conflict, but a state change over time
- **Decision reconstruction**: What did the agent know when it made decision X? Query as-of the decision's transaction time
- **Lineage tracking**: For database schemas and data warehouse query logs, understanding *when* a schema changed or *when* a query pattern shifted is the core value proposition
- **Memory decay**: Old, unconfirmed facts can be flagged or downweighted based on valid_to proximity, enabling a natural "forgetting" mechanism

#### Prior Art

- **Zep/Graphiti** (arXiv:2501.13956): Bi-temporal knowledge graph engine for AI agent memory. Uses Neo4j. The temporal model described above is adapted from their approach.
- **SQl:2011 temporal tables**: The SQL standard includes `PERIOD FOR` and `SYSTEM_TIME` versioning. SQLite does not implement these, but the schema pattern above achieves the same semantics.
- **Datomic**: Immutable database where every fact is timestamped. The transaction-time dimension is built into the storage engine. Influential on the bi-temporal approach.
- **Event sourcing**: The broader pattern of storing state changes rather than current state. Temporal edges are event-sourced facts.

---

## Entity Extraction Approaches

For reproducibility, we want deterministic extraction (not LLM-based). Three complementary approaches, each exercising different parts of the stack:

### Approach 1: NER Models for Entity Extraction

Use Named Entity Recognition to find entities in passage chunks.

**Recommended model: GLiNER (zero-shot NER)**

GLiNER (Generalist and Lightweight NER, NAACL 2024) allows defining **custom entity types at inference time** without fine-tuning. This is ideal for domain-specific text:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")  # 350 MB, CPU-feasible

# Define domain-specific entity types for economic text
labels = [
    "person", "place", "nation", "commodity",
    "institution", "economic_concept", "monetary_unit", "law"
]

entities = model.predict_entities(
    "The division of labour in the pin factory at Birmingham increases productivity tenfold.",
    labels
)
# → [{"text": "division of labour", "label": "economic_concept", "score": 0.91},
#    {"text": "pin factory", "label": "institution", "score": 0.78},
#    {"text": "Birmingham", "label": "place", "score": 0.95}]
```

**Why GLiNER over traditional NER**: Standard NER models (dslim/bert-base-NER, spaCy) only detect 4 entity types (PER/ORG/LOC/MISC). They'll miss "division of labour" as an economic concept entirely. GLiNER's zero-shot approach lets us define domain-appropriate types.

**Alternative NER models for comparison:**

| Model | Params | Size | Entity Types | CPU Speed | Notes |
|-------|--------|------|-------------|-----------|-------|
| `urchade/gliner_small-v2.1` | 44M backbone | 350 MB | Custom (zero-shot) | Fast | **Recommended**. DeBERTa-v3-small. |
| `urchade/gliner_medium-v2.1` | 86M backbone | 600 MB | Custom (zero-shot) | Medium | Better accuracy. |
| `dslim/distilbert-NER` | 66M | 250 MB | 4 (PER/ORG/LOC/MISC) | Fast | Lighter but limited types. |
| spaCy `en_core_web_lg` | — | 560 MB | 18 (OntoNotes) | Fast | Includes POS tagger and dep parser for SVO approach. |
| `flair/ner-english-ontonotes-fast` | ~50M | 300 MB | 18 (OntoNotes) | Medium | Catches MONEY, DATE, NORP — relevant for economic text. |

**Challenge for 18th-century text**: Pre-trained NER struggles with archaic spellings, non-standard place names, and long complex sentences. A **manual seed list** of ~50 key entities with aliases remains essential for high recall on Smith's terminology.

### Approach 2: Noun-Verb-Noun (SVO) Pattern Extraction

The grammatical structure of language naturally encodes knowledge graph triples:

- **Nouns** (NOUN, PROPN) → candidate **nodes** (things / entities)
- **Verbs** (VERB) → candidate **edges** (how things relate)
- **Subject-Verb-Object** triples → directed graph edges

```python
import spacy
import textacy.extract

nlp = spacy.load("en_core_web_lg")  # or en_core_web_md for lighter

doc = nlp("The sovereign regulates commerce through Parliament.")
triples = list(textacy.extract.subject_verb_object_triples(doc))
# → [SVOTriple(subject=[sovereign], verb=[regulates], object=[commerce])]
# Graph edge: sovereign --REGULATES--> commerce
```

**How textacy's SVO extraction works internally:**
- **Subjects**: Tokens with `nsubj` or `nsubjpass` dependency whose head is a VERB
- **Verbs**: All VERB-tagged tokens, expanded to include auxiliaries and negations
- **Objects**: Tokens with `dobj`, `pobj` (when head has `agent` dep), or `xcomp` dependency
- **Expansion**: Nouns are expanded to include conjuncts and compounds (`expand_noun()`), verbs include aux/neg (`expand_verb()`)

**Advantages of SVO extraction:**
- Uses only spaCy (12-560 MB depending on model) — no GPU needed
- Captures the actual grammatical structure of the text
- Works on any domain without training data
- Naturally produces (entity, relation, entity) triples for the graph

**Disadvantages:**
- Crude: misses implicit relations, struggles with passive voice and multi-clause sentences
- Verb lemmas as edge labels produce noisy/inconsistent relation types
- Smith's 18th-century prose uses very long sentences with semicolons — consider splitting on semicolons before processing

**Alternative: Stanford OpenIE** (`pip install stanford_openie`) extracts open-domain relation triples via natural logic. More robust than SVO but requires Java runtime. A GPU-accelerated Python port (`triplet-extract`) is also available.

### Approach 3: FTS5/BM25 for Concept Discovery

A "zero-model" approach that uses SQLite's built-in FTS5 to identify important terms as candidate graph nodes.

**How BM25 works**: Okapi BM25 scores a term's importance in a document relative to the corpus:

```
BM25(t,d) = IDF(t) × [f(t,d) × (k₁ + 1)] / [f(t,d) + k₁ × (1 - b + b × |d|/avgdl)]
```

- **IDF(t)**: Inverse document frequency — rare terms score higher
- **f(t,d)**: Term frequency in document d
- **k₁, b**: Tuning parameters (typically k₁=1.2, b=0.75)
- **|d|/avgdl**: Document length normalization

**Key insight**: Terms with high BM25 scores across multiple documents are "important words" — good candidates for concept nodes in the knowledge graph. A word like "labour" that appears frequently but not in every passage has high BM25 importance. A word like "the" that appears everywhere has near-zero importance (IDF ≈ 0).

**SQLite FTS5 implementation:**

```sql
-- 1. Create FTS5 index over passage chunks
CREATE VIRTUAL TABLE won_fts USING fts5(passage_text, tokenize='porter');
INSERT INTO won_fts(rowid, passage_text) SELECT rowid, text FROM passages;

-- 2. Create vocabulary table to inspect term statistics
CREATE VIRTUAL TABLE won_vocab USING fts5vocab(won_fts, row);
-- Schema: (term TEXT, doc INTEGER, cnt INTEGER)
--   term = indexed token
--   doc  = number of documents containing term
--   cnt  = total occurrences across all documents

-- 3. Find important terms (high document frequency but not universal)
SELECT term, doc, cnt,
       CAST(doc AS REAL) / (SELECT COUNT(*) FROM passages) AS doc_ratio
FROM won_vocab
WHERE doc > 5                    -- appears in at least 5 passages
  AND doc < (SELECT COUNT(*) * 0.5 FROM passages)  -- not in >50% of passages
ORDER BY doc DESC
LIMIT 200;
-- This gives us the ~200 most important but not ubiquitous terms

-- 4. Use BM25 to rank passages for a specific term
SELECT rowid, bm25(won_fts) AS score
FROM won_fts
WHERE won_fts MATCH 'labour'
ORDER BY bm25(won_fts)  -- lower = better match (FTS5 convention)
LIMIT 10;

-- 5. Co-occurrence analysis: terms that appear together become edges
-- (Application-level: for each pair of important terms,
--  count passages where both appear → edge weight)
```

**This approach produces:**
- **Nodes**: Important terms from the vocabulary table (filtered to nouns via POS tagging or a stopword list)
- **Edges**: Co-occurrence within passages (two terms in the same passage → edge with weight = co-occurrence count)
- **Entry points**: BM25 search replaces vector search as the graph entry point

**FTS5 + HNSW hybrid retrieval:**

```sql
-- Hybrid scoring: combine BM25 keyword relevance with vector similarity
-- Step 1: BM25 entry points
SELECT rowid, bm25(won_fts) AS bm25_score FROM won_fts WHERE won_fts MATCH ?;

-- Step 2: VSS entry points
SELECT rowid, distance AS vss_score FROM kg_vectors WHERE knn_search(vector, ?, 10);

-- Step 3: Reciprocal Rank Fusion (application-level)
-- RRF(d) = Σ 1/(k + rank_in_list)  for each retrieval list
-- Then graph expansion from top-ranked nodes
```

**Value for benchmarking**: This creates a natural baseline — "Can BM25 + graph expansion match or exceed VSS + graph expansion?" If FTS5 performs competitively, it validates that the graph structure adds value regardless of the entry-point method. If HNSW significantly outperforms, it demonstrates the value of semantic search for graph-seeded retrieval.

### Hybrid Pipeline Recommendation

The recommended pipeline combines all three approaches:

```
Wealth of Nations text (2,500 passages)
  │
  ├─[FTS5] Build vocab index → identify ~200 important terms (concept candidates)
  │
  ├─[GLiNER] Zero-shot NER → extract typed entities (person, place, institution, concept)
  │
  ├─[spaCy SVO] Dependency parse → extract (subject, verb, object) relation triples
  │
  ├─[Merge] Union of entities from all three sources
  │         GLiNER entities → typed nodes
  │         FTS5 terms → concept nodes (if not already found by GLiNER)
  │         SVO subjects/objects → additional nodes
  │
  ├─[Coalesce] HNSW-based entity resolution
  │            Embed all entity names → insert into HNSW → find near-duplicates
  │            "division of labour" ≈ "labour is divided" → merge
  │
  ├─[Edges] SVO verbs → typed edges (verb lemma = relation)
  │         Co-occurrence in passage → MENTIONED_WITH edges
  │         Chapter hierarchy → PART_OF edges
  │
  └─[Output] SQLite database with nodes table + edges table + HNSW index
```

---

## HuggingFace Model Recommendations

### Tier 1: Minimal Pipeline (< 500 MB total, CPU-only)

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Entity extraction | `urchade/gliner_small-v2.1` | 350 MB | Zero-shot NER with custom entity types |
| Relation extraction | spaCy `en_core_web_md` (dep parse + SVO) | 40 MB | No additional model needed |
| Node embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 80 MB | Already in benchmark suite. 384d. |
| Concept extraction | `keybert` with MiniLM-L6-v2 | 0 MB | Shares embedding model |

### Tier 2: Better Quality (< 2.5 GB, still CPU-feasible)

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Entity extraction | `urchade/gliner_medium-v2.1` | 600 MB | Better accuracy |
| Relation extraction | `Babelscape/rebel-large` | 1.6 GB | End-to-end triplets, 220+ Wikidata relation types |
| Node embeddings | `BAAI/bge-base-en-v1.5` | 440 MB | Instruction-tunable |
| Topic discovery | BERTopic (lightweight install) | 0 MB | Reuses embedding model |

### Tier 3: Fastest Possible (< 100 MB)

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Entity extraction | spaCy `en_core_web_sm` | 12 MB | Only 4 entity types |
| Relation extraction | spaCy dep parse (same model) | 0 MB | SVO from dependency tree |
| Node embeddings | `minishlab/potion-base-8M` | 30 MB | 500x faster than MiniLM, ~80% quality |
| Concept extraction | TF-IDF (sklearn) | 0 MB | No neural model |

### Special: Relation Extraction via REBEL

REBEL (`Babelscape/rebel-large`) generates (subject, relation, object) triples directly from text as a seq2seq task (BART-large backbone, 400M params):

```python
from transformers import pipeline
extractor = pipeline('translation_xx_to_yy', model='Babelscape/rebel-large')
result = extractor("The wealth of nations depends on the division of labour.")
# Parses output → (division of labour, subclass of, economic concept), etc.
```

REBEL extracts Wikidata-typed relations (220+ types including `country_of_origin`, `instance_of`, `part_of`, `has_use`, `located_in`). Heavier than SVO extraction but produces more consistent, typed relations.

---

## Dataset: Economics Texts from Project Gutenberg

The benchmark pipeline is **generalizable** — it builds a knowledge graph from any economics text, not just a single hard-coded book. The primary dataset is Adam Smith's *The Wealth of Nations* (Gutenberg #3300), which is already available in the benchmark suite. Additionally, the benchmark can pull a **random book from Project Gutenberg's economics category** and run the identical concept mapping pipeline, validating that the approach generalizes beyond one text.

### Primary Dataset: Wealth of Nations (Gutenberg #3300)

Already available as ~2,500 passage chunks. Serves as the **reference dataset** — all metric baselines and ground-truth annotations are built against this text.

### Random Economics Book: Project Gutenberg Integration

The benchmark can fetch a random economics text from Project Gutenberg's catalog (~400+ English-language books) and perform the same entity extraction, relation mapping, and graph construction pipeline.

**Source catalog**: The [Gutendex API](https://gutendex.com/) provides JSON access to Project Gutenberg metadata. The `topic=economics` parameter matches books tagged with the "Economics" bookshelf or LCSH subject.

**Example economics texts in the catalog:**

| Gutenberg ID | Title | Author | Era |
|---|---|---|---|
| 3300 | An Inquiry into the Nature and Causes of the Wealth of Nations | Adam Smith | 1776 |
| 33310 | On The Principles of Political Economy, and Taxation | David Ricardo | 1817 |
| 30107 | Principles of Political Economy (Abridged) | John Stuart Mill | 1848 |
| 61 | The Communist Manifesto | Karl Marx & Friedrich Engels | 1848 |
| 833 | The Theory of the Leisure Class | Thorstein Veblen | 1899 |
| 15776 | The Economic Consequences of the Peace | John Maynard Keynes | 1919 |
| 55308 | Progress and Poverty, Volumes I and II | Henry George | 1879 |
| 46423 | A Contribution to the Critique of Political Economy | Karl Marx | 1859 |
| 40077 | The Principles of Economics | Frank A. Fetter | 1905 |
| 24518 | Memoirs of Extraordinary Popular Delusions and the Madness of Crowds | Charles Mackay | 1841 |

**Book selection and download pipeline:**

```python
import random
import re
import time
import requests

GUTENDEX_BASE = "https://gutendex.com/books"
ECONOMICS_KEYWORDS = {
    "economics", "political economy", "economic", "capitalism",
    "finance", "wealth", "commerce", "trade", "marxian",
}
DELAY = 2  # seconds between requests (Gutenberg robot policy)


def fetch_economics_catalog() -> list[dict]:
    """Fetch all English economics books from Gutendex, filtering out 'Home Economics' etc."""
    books = []
    url = GUTENDEX_BASE
    params = {"topic": "economics", "languages": "en"}

    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        for book in data["results"]:
            # Require a plain-text download URL
            text_url = next(
                (u for mime, u in book["formats"].items()
                 if "text/plain" in mime and "utf-8" in mime),
                None,
            )
            if not text_url:
                continue

            # Filter to genuine economics (exclude "Home Economics", cookbooks, etc.)
            all_tags = [s.lower() for s in book.get("subjects", []) + book.get("bookshelves", [])]
            if not any(kw in tag for tag in all_tags for kw in ECONOMICS_KEYWORDS):
                continue

            books.append({
                "id": book["id"],
                "title": book["title"],
                "authors": [a["name"] for a in book["authors"]],
                "text_url": text_url,
                "download_count": book.get("download_count", 0),
            })

        url = data.get("next")
        params = {}  # next URL includes params
        if url:
            time.sleep(DELAY)

    return books


def download_and_clean(text_url: str) -> str:
    """Download plain text and strip Gutenberg header/footer boilerplate."""
    time.sleep(DELAY)
    raw = requests.get(text_url).text

    start = re.search(r"\*\*\*\s*START OF (?:THE |THIS )PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       raw, re.IGNORECASE)
    end = re.search(r"\*\*\*\s*END OF (?:THE |THIS )PROJECT GUTENBERG EBOOK.*?\*\*\*",
                     raw, re.IGNORECASE)

    body = raw[start.end():end.start()] if start and end else raw
    # Strip "Produced by ..." preamble
    body = re.sub(r"^(?:Produced|Transcribed|E-text prepared) by .+?\n\n",
                  "", body.strip(), count=1, flags=re.DOTALL | re.IGNORECASE)
    return body.strip()


def select_random_economics_book(exclude_ids: set[int] | None = None) -> tuple[dict, str]:
    """Select a random economics book and return (metadata, clean_text)."""
    catalog = fetch_economics_catalog()
    if exclude_ids:
        catalog = [b for b in catalog if b["id"] not in exclude_ids]
    book = random.choice(catalog)
    text = download_and_clean(book["text_url"])
    return book, text
```

**Usage in the benchmark:**

```python
# Default: use Wealth of Nations (cached locally)
book_id = 3300

# Or: pull a random economics book for generalization testing
book, text = select_random_economics_book(exclude_ids={3300})
passages = chunk_text(text, chunk_size=500)  # Same chunking as WoN
# → Same entity extraction → same graph construction → same benchmark queries
```

**Caching strategy**: Downloaded texts are cached in `benchmarks/texts/{gutenberg_id}.txt` to avoid re-downloading. The Gutendex catalog response is cached for 24 hours (see project caching rules). The benchmark records the Gutenberg ID in its JSONL output for reproducibility.

### Why Random Book Selection Matters

Running the same pipeline on an **unseen text** validates that:

1. **The extraction pipeline generalizes** — GLiNER entity types and SVO patterns work on Ricardo, Mill, Keynes, not just Smith
2. **Graph structure varies meaningfully** — different authors produce different concept topologies (Marx's KG will have different clusters than Smith's)
3. **Benchmark metrics are robust** — retrieval quality shouldn't depend on one specific text's structure
4. **The HNSW + graph composition pattern works on arbitrary KGs** — not just the one we tuned for

This also enables a **cross-book comparison**: build KGs from two economics texts, then use Node2Vec embeddings to find structurally similar concepts across books (e.g., Smith's "division of labour" ↔ Ricardo's "comparative advantage").

### Entity Types

These entity types are **domain-generic for economics texts** — they apply to any book in the catalog:

| Type | WoN Examples | Generic Examples | Expected Count | Extraction Source |
|------|-------------|-----------------|---------------|-------------------|
| **Concept** | division of labour, market price, natural price | comparative advantage, surplus value, marginal utility | ~100-300 | GLiNER + FTS5 |
| **Actor** | labourer, merchant, sovereign | capitalist, proletarian, consumer | ~30-80 | GLiNER + SVO subjects |
| **Institution** | East India Company, Bank of England | Federal Reserve, guilds, unions | ~10-50 | GLiNER |
| **Place** | England, Holland, Bengal | Europe, colonies, factories | ~20-60 | GLiNER |
| **Work** | (chapter/book references) | (chapter/section references) | ~20-80 | Pattern matching |
| **Commodity** | corn, silver, gold, wool | iron, cotton, machinery | ~15-50 | GLiNER + FTS5 |

### Relation Types

| Relation | Description | Example | Extraction Source |
|----------|-------------|---------|-------------------|
| `CAUSES` | Causal relationship | division_of_labour CAUSES productivity_increase | SVO (verb: "causes", "increases", "produces") |
| `COMPOSED_OF` | Part-whole | national_wealth COMPOSED_OF land, labour, stock | SVO (verb: "comprises", "consists of") |
| `REGULATES` | Governance/control | sovereign REGULATES commerce | SVO (verb: "regulates", "controls") |
| `ARGUES_IN` | Textual reference | smith ARGUES_IN book_1_ch_1 | Pattern matching |
| `LOCATED_IN` | Geographic | east_india_company LOCATED_IN bengal | GLiNER co-occurrence |
| `TRADES_WITH` | Commerce | england TRADES_WITH holland | SVO (verb: "trades", "exports") |
| `MENTIONED_WITH` | Co-occurrence | labour MENTIONED_WITH wages | FTS5 co-occurrence |

### Graph Structure (Per Book)

- **Nodes**: ~200-500 entities + ~500-3,000 passage chunks (varies by book length)
- **Edges**: ~2,000-15,000 relations (entity-entity + entity-passage links)
- **Chapter hierarchy**: Book → Chapter → Passage as a tree backbone
- **Properties**: Each entity node has a text description + embedding; each passage node has original text + embedding
- **Node types**: `entity` (extracted concepts/people/places) and `passage` (text chunks)

Graph size varies significantly by text — *The Communist Manifesto* (~30 pages) produces a much smaller graph than *Wealth of Nations* (~400 pages). The benchmark records `n_entities`, `n_passages`, and `n_edges` in its JSONL output so results can be normalized by graph size.

---

## Benchmark Workflow

### Phase A: Build the Knowledge Graph in vec_graph

```sql
-- 1. Create HNSW index for node embeddings
CREATE VIRTUAL TABLE kg_vectors USING hnsw_index(dim=384, metric=cosine, M=16, ef_construction=200);

-- 2. Insert passage embeddings (from pre-computed .npy cache)
INSERT INTO kg_vectors(rowid, vector) VALUES (?, ?);

-- 3. Create edge table for graph structure
CREATE TABLE kg_edges (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0
);
CREATE INDEX idx_kg_src ON kg_edges(src);
CREATE INDEX idx_kg_dst ON kg_edges(dst);

-- 4. Insert entity-entity and entity-passage edges
INSERT INTO kg_edges(src, dst, relation) VALUES (?, ?, ?);

-- 5. Generate Node2Vec embeddings (bridges graph + vector subsystems)
SELECT node2vec_train(
    'kg_edges', 'src', 'dst',      -- edge table
    'kg_vectors',                    -- target HNSW table
    384,                             -- dimension
    10, 80,                          -- walks_per_node, walk_length
    1.0, 1.0,                        -- p, q (BFS/DFS bias)
    5, 5, 0.025, 5                   -- window, negative, lr, epochs
);

-- 6. Create FTS5 index for BM25 baseline
CREATE VIRTUAL TABLE kg_fts USING fts5(text, tokenize='porter');
INSERT INTO kg_fts(rowid, text) SELECT id, description FROM nodes;
```

### Phase B: GraphRAG-style Retrieval Queries

Each query simulates a retrieval-augmented generation (RAG) lookup:

```sql
-- Step 1: Vector similarity search for entry point
SELECT rowid, distance
FROM kg_vectors
WHERE knn_search(vector, ?, 5);  -- top-5 nearest to query embedding

-- Step 2: Graph expansion from each entry point
SELECT node, depth, parent
FROM graph_bfs
WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = ?             -- from Step 1 result
  AND max_depth = 2              -- 2-hop neighborhood
  AND direction = 'both';

-- Step 3: Centrality-guided ranking (NEW)
-- Compute betweenness centrality on the subgraph of traversed nodes
-- Prioritize bridge nodes in context assembly

-- Step 4: Collect context (join traversed nodes with passage text)
-- Application-level: assemble retrieved text for LLM context
-- Weight by: centrality_score × (1 / (1 + depth))
```

### Phase C: Graph Analytics Queries (NEW)

Benchmark the graph analytics primitives on the KG:

```sql
-- Betweenness centrality: find bridge concepts
SELECT node, centrality
FROM graph_betweenness
WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
ORDER BY centrality DESC
LIMIT 20;
-- Expected: "market price", "labour", "capital" as top bridge concepts

-- PageRank: find authoritative nodes (already implemented)
SELECT node, rank
FROM graph_pagerank
WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
ORDER BY rank DESC
LIMIT 20;

-- Community detection (requires new implementation)
SELECT node, community_id
FROM graph_louvain
WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst';
-- Expected communities: {labour, wages, productivity}, {rent, land, landlord},
--                       {commerce, trade, merchant}, etc.
```

### Phase D: Hierarchical Retrieval (Advanced)

```sql
-- 1. Detect communities (Louvain/Leiden)
SELECT node, community_id
FROM graph_louvain
WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst';

-- 2. Compute supernode embeddings (mean of member embeddings)
-- Application-level: for each community, average the member vectors

-- 3. Insert supernode embeddings into a separate HNSW index
CREATE VIRTUAL TABLE kg_supernodes USING hnsw_index(dim=384, metric=cosine);
INSERT INTO kg_supernodes(rowid, vector) VALUES (?, ?);

-- 4. Multi-level retrieval: search supernodes first, then drill into community
SELECT rowid, distance FROM kg_supernodes WHERE knn_search(vector, ?, 3);
-- Then expand within the selected community
```

### Phase E: Node2Vec Hyperparameter Sweep

```python
# Sweep p,q values to find optimal graph embeddings for retrieval
for p in [0.25, 0.5, 1.0, 2.0, 4.0]:
    for q in [0.25, 0.5, 1.0, 2.0, 4.0]:
        # Train Node2Vec with these parameters
        # Evaluate retrieval quality (context recall/precision)
        # Record (p, q, recall, precision, training_time)
```

**Hypothesis**: Low p (BFS-like walks, stay local) captures economic concept clusters best. The KG has natural clusters (labour/wages, rent/land, commerce/trade) and BFS-like walks keep embeddings within these clusters.

### Phase F: Temporal Knowledge Graph Queries

Test the bi-temporal schema pattern for agent memory use cases. This phase operates on a temporally-annotated version of the KG where edges carry `valid_from`, `valid_to`, and `recorded_at` timestamps.

```sql
-- 1. Create temporal edge table
CREATE TABLE kg_edges_temporal (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    valid_from TEXT NOT NULL,
    valid_to TEXT DEFAULT '9999-12-31T23:59:59Z',
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    source_session TEXT
);

-- 2. Current-state view for standard traversal
CREATE VIEW kg_edges_current AS
SELECT src, dst, relation, weight
FROM kg_edges_temporal
WHERE valid_to >= strftime('%Y-%m-%dT%H:%M:%SZ', 'now');

-- 3. Point-in-time snapshot traversal
CREATE TEMP TABLE kg_edges_snapshot AS
SELECT src, dst, relation, weight FROM kg_edges_temporal
WHERE valid_from <= ? AND valid_to >= ?;

SELECT node, depth FROM graph_bfs
WHERE edge_table = 'kg_edges_snapshot' AND src_col = 'src' AND dst_col = 'dst'
  AND start_node = ? AND max_depth = 2;

-- 4. Temporal diff: what edges changed between two points in time?
SELECT src, dst, relation,
       CASE WHEN valid_from > ?1 THEN 'added' ELSE 'removed' END AS change_type
FROM kg_edges_temporal
WHERE (valid_from > ?1 AND valid_from <= ?2)       -- newly added
   OR (valid_to > ?1 AND valid_to <= ?2);           -- newly expired
```

**Benchmark questions:**

- How does point-in-time filtering affect traversal performance vs the current-state view?
- What is the storage overhead of temporal edges vs non-temporal (extra columns + index)?
- Can Node2Vec embeddings trained on the current graph still retrieve relevant nodes from a historical snapshot, or do embeddings need retraining per time period?
- How does graph density change over time as edges accumulate (temporal graphs grow monotonically)?

---

## Metrics

### Retrieval Quality

| Metric | Description |
|--------|-------------|
| **Context Recall** | Fraction of ground-truth relevant passages found in retrieved context |
| **Context Precision** | Fraction of retrieved passages that are actually relevant |
| **Hop Efficiency** | Mean graph hops needed to reach a relevant passage from VSS entry point |
| **Coverage** | Fraction of unique entities touched by expansion |
| **Bridge Discovery** | Fraction of ground-truth bridge concepts found via betweenness centrality |

### Graph Analytics Performance

| Metric | Description |
|--------|-------------|
| **Betweenness time** | Time to compute betweenness centrality for all nodes |
| **PageRank time** | Time for PageRank convergence (already benchmarked) |
| **Louvain time** | Time for community detection |
| **Community quality** | Modularity score of detected communities |
| **Centrality correlation** | Spearman correlation between centrality scores and ground-truth importance |

### End-to-End Performance

| Metric | Description |
|--------|-------------|
| **End-to-end latency** | Total time for VSS + graph expansion + context assembly |
| **VSS time** | HNSW search component |
| **Expansion time** | Graph traversal component |
| **Node2Vec training time** | One-time cost for embedding generation |
| **Memory** | Peak RSS during combined operation |

### Temporal Performance

| Metric | Description |
|--------|-------------|
| **Snapshot creation time** | Time to materialize a point-in-time temp table from temporal edges |
| **Temporal vs current traversal** | Latency ratio: traversal on snapshot table vs current-state view |
| **Storage overhead** | Size of temporal edge table vs non-temporal (extra columns + indices) |
| **Temporal edge accumulation** | Edge count growth rate as sessions add new temporal facts |
| **Embedding staleness** | Retrieval quality of Node2Vec embeddings trained at time T when querying at T+delta |

### Comparison Baselines

| Baseline | Description |
|----------|-------------|
| **VSS-only** | Pure HNSW search, no graph expansion (standard RAG) |
| **Graph-only** | Random start node + BFS, no vector guidance |
| **BM25 + Graph** | FTS5 keyword search entry point + graph expansion |
| **BM25-only** | Pure FTS5 search, no graph (traditional IR baseline) |
| **Centrality-guided** | BFS expansion + betweenness centrality ranking vs uniform ranking |
| **Separate tools** | sqlite-vec for VSS + GraphQLite for traversal (no integration) |

---

## New Graph Capabilities Required

> **Full implementation plans, algorithm details, and TVF interface designs extracted to [Graph Algorithms Plan](graph_algorithms.md).**
>
> Summary of what the KG benchmark needs from that plan:
>
> | Algorithm | TVF | KG Benchmark Phase | Priority |
> |-----------|-----|-------------------|----------|
> | Betweenness centrality | `graph_betweenness` | Phase C (Graph Analytics) | P1 |
> | Louvain community detection | `graph_louvain` | Phase C + D (Hierarchical Retrieval) | P2 |
> | Leiden refinement | `graph_leiden` | Phase D (upgrade from Louvain) | P3 |
> | Degree/closeness centrality | `graph_degree`, `graph_closeness` | Nice-to-have | P4 |
>
> **Key insight**: Betweenness centrality identifies "bridge" concepts between KG clusters (e.g., "market price" connecting the labour/wages cluster to the rent/profit cluster). Louvain/Leiden community detection enables hierarchical retrieval — search community supernodes first, then drill into the matching community.

---

## Key References and Prior Art

### Microsoft GraphRAG (2024)
- Extract KG from documents using LLM
- Build community hierarchy (Leiden algorithm)
- Generate community summaries at each level
- Retrieval: search summaries, drill into communities
- **Relevance**: The gold standard for GraphRAG; our benchmark tests the primitives this pattern needs

### Deep GraphRAG (2025)
- Hierarchical retrieval with multi-hop reasoning
- Graph-guided evidence chain construction
- **Relevance**: Demonstrates the value of multi-hop graph traversal starting from VSS results

### HybridRAG (2024)
- Combines vector DB (for passage retrieval) with graph DB (for entity relationships)
- Joint scoring: vector similarity + graph distance
- **Relevance**: Exactly the pattern vec_graph enables in a single SQLite extension

### NaviX (VLDB 2025)
- Native dual indexing: vector index + graph index in DB kernel
- Pruning strategies that leverage both indexes simultaneously
- **Relevance**: The research frontier; vec_graph's architecture could enable similar optimizations

### Multi-Scale Node Embeddings (2024)
- Supernode embeddings as aggregates of community member embeddings
- Enables hierarchical search: coarse (community) -> fine (node)
- **Relevance**: Node2Vec + community detection + HNSW = this pattern in vec_graph

### Grover & Leskovec (2016) — Node2Vec
- Biased random walks with p,q parameters
- Captures homophily (BFS-like) vs structural equivalence (DFS-like)
- Skip-gram with negative sampling for embedding learning
- **Relevance**: Already implemented in vec_graph

### Blondel et al. (2008) — Louvain
- Fast modularity optimization for community detection
- O(n log n) complexity
- **Relevance**: Needed for hierarchical retrieval (Phase D)

### Traag, Waltman & van Eck (2019) — Leiden
- Fixes Louvain's connectivity guarantees via refinement phase
- Used by Microsoft GraphRAG
- **Relevance**: Upgrade path from Louvain

### Brandes (2001) — Betweenness Centrality
- O(VE) algorithm for computing betweenness of all nodes
- **Relevance**: Needed for centrality-guided retrieval (Phase C)

### GLiNER (NAACL 2024) — Zero-Shot NER
- Generalist and Lightweight Named Entity Recognition
- Custom entity types without fine-tuning
- DeBERTa backbone, Apache 2.0 license
- **Relevance**: Entity extraction from Wealth of Nations text

### REBEL (ACL 2021) — Relation Extraction
- End-to-end relation extraction as seq2seq (BART-large)
- 220+ Wikidata relation types
- **Relevance**: Alternative to SVO extraction for relation typing

### Vesper-Memory (2025)
- AI agent memory with semantic search + knowledge graphs + multi-hop reasoning
- Uses **Personalized PageRank** for retrieval ranking — similar to our centrality-guided approach
- Docker-based Python service architecture
- **Relevance**: Validates PageRank-based ranking in the retrieval pipeline. Our benchmark tests the same pattern (centrality-guided retrieval vs uniform) but natively inside SQLite rather than as a separate service. Vesper-Memory is a key comparison target for the "embedded vs service" performance question.

### Zep/Graphiti (2025) — Temporal Agent Memory
- Temporally-aware knowledge graph engine for AI agent memory (arXiv:2501.13956)
- Bi-temporal data model: valid time (when fact was true) + transaction time (when system recorded it)
- Hybrid retrieval: semantic embeddings + BM25 + graph traversal
- Requires Neo4j + Python server
- **Relevance**: The temporal KG schema pattern in Phase F is adapted from Graphiti's approach. Our benchmark tests whether the same bi-temporal semantics work efficiently as a pure SQLite schema pattern without a dedicated temporal graph engine.

---

## Prerequisites

Before implementing this benchmark:

1. **Primitive benchmarks validated** (Phases 1-6 of the current plan)
   - vec_graph HNSW search performance characterized across dimensions and N
   - vec_graph graph TVF performance characterized across topologies
   - Node2Vec training time and embedding quality validated

2. **New graph capabilities implemented**
   - Betweenness centrality TVF (Priority 1)
   - Louvain community detection TVF (Priority 2)

3. **Entity extraction pipeline**
   - GLiNER zero-shot NER for entity extraction
   - spaCy dependency parse for SVO relation extraction
   - FTS5 vocabulary analysis for concept discovery
   - Entity coalescing via HNSW-based near-duplicate detection
   - Cached KG as SQLite database alongside the benchmark

4. **Ground truth queries**
   - 50-100 retrieval questions about Wealth of Nations content
   - Human-annotated relevant passages for each question
   - Relevance judgments at passage and entity level
   - Ground-truth bridge concepts (for betweenness centrality validation)

5. **Python dependencies** (benchmark group in pyproject.toml)
   - `gliner` — zero-shot NER
   - `spacy` + `en_core_web_lg` — dependency parsing and SVO
   - `textacy` — SVO triple extraction
   - `sentence-transformers` — node embeddings (already present)
   - `requests` — Gutendex API access for random book selection (already present)
   - `keybert` — keyword extraction (optional)

---

## Implementation Sketch

### File Structure

```
benchmarks/
  scripts/
    benchmark_kg.py            # KG benchmark runner (works on any economics text)
    benchmark_kg_analyze.py    # KG benchmark analysis + charts
    kg_extract.py              # Entity/relation extraction pipeline
    kg_coalesce.py             # Entity resolution + synonym merging
    kg_gutenberg.py            # Project Gutenberg catalog + download + caching
  kg/                          # Cached knowledge graphs (one SQLite DB per book)
    kg_3300.db                 # Wealth of Nations KG (reference)
    kg_33310.db                # Ricardo's Principles (example random pick)
  texts/                       # Cached plain text downloads
    3300.txt                   # Wealth of Nations (stripped boilerplate)
    33310.txt                  # Ricardo's Principles
  results/kg_*.jsonl           # KG benchmark results (includes gutenberg_id)
  vectors/kg_*.npy             # Pre-computed entity embeddings per book
```

### JSONL Schema

```json
{
    "timestamp": "...",
    "benchmark_type": "knowledge_graph",
    "gutenberg_id": 3300,
    "book_title": "An Inquiry into the Nature and Causes of the Wealth of Nations",
    "book_author": "Smith, Adam",
    "workflow": "vss_then_expand",
    "vss_engine": "vec_graph",
    "graph_engine": "vec_graph",
    "model": "all-MiniLM-L6-v2",
    "dim": 384,
    "n_entities": 400,
    "n_passages": 2500,
    "n_edges": 7500,
    "expansion_depth": 2,
    "vss_k": 5,
    "vss_time_ms": 0.15,
    "expansion_time_ms": 1.2,
    "centrality_time_ms": 0.5,
    "total_time_ms": 1.85,
    "context_recall": 0.78,
    "context_precision": 0.62,
    "hop_efficiency": 1.4,
    "n_queries": 50,
    "node2vec_train_s": 12.5,
    "node2vec_p": 1.0,
    "node2vec_q": 1.0,
    "betweenness_time_ms": 45.0,
    "louvain_time_ms": 12.0,
    "louvain_modularity": 0.42,
    "n_communities": 15
}
```

### Makefile Targets (in `benchmarks/Makefile`)

```makefile
kg-extract:                                    ## Extract KG from Wealth of Nations (reference)
	../.venv/bin/python scripts/kg_extract.py --book-id 3300

kg-extract-random:                             ## Extract KG from a random Gutenberg economics book
	../.venv/bin/python scripts/kg_extract.py --random-economics --exclude 3300

kg-extract-book:                               ## Extract KG from a specific book (BOOK_ID=...)
	../.venv/bin/python scripts/kg_extract.py --book-id $(BOOK_ID)

kg-coalesce:                                   ## Entity resolution + dedup (all cached KGs)
	../.venv/bin/python scripts/kg_coalesce.py

kg: vec_graph                                  ## Run KG benchmark on all cached KGs
	../.venv/bin/python scripts/benchmark_kg.py

kg-analyze:                                    ## Analyze KG results → charts
	../.venv/bin/python scripts/benchmark_kg_analyze.py
```

---

## Open Questions

1. **Entity extraction quality**: How reliable is GLiNER zero-shot NER on 18th-century economic text? May need custom patterns for archaic terminology. The manual seed list of ~50 key entities remains essential.

2. **Ground truth annotation**: Who annotates the 50-100 retrieval questions? Could use LLM-generated questions with human validation. Need both passage-level and entity-level relevance judgments.

3. **Node2Vec hyperparameters**: What (p, q) values best capture the WoN's conceptual structure? The benchmark should sweep these. Hypothesis: low p (BFS-like) captures economic clusters.

4. **Louvain vs Leiden**: Start with Louvain (simpler), upgrade to Leiden for guaranteed connectivity? Microsoft GraphRAG uses Leiden.

5. **Betweenness centrality scalability**: Brandes' algorithm is O(VE). For ~3K nodes / ~10K edges, this is fine. But if the graph grows to 100K nodes, we'd need approximate betweenness (random sampling).

6. **FTS5 + HNSW fusion**: What's the best way to combine BM25 scores and vector distances? Reciprocal Rank Fusion is simple but there may be better approaches.

7. **REBEL vs SVO extraction**: REBEL produces cleaner, typed relations but is 1.6 GB and slower. SVO extraction is fast but noisy. Do typed relations actually improve retrieval quality enough to justify the model weight?

8. **Comparison fairness**: How to fairly compare "vec_graph integrated" vs "separate tools"? The separate-tools baseline needs equivalent functionality without the single-extension advantage.

9. **Entity coalescing threshold**: What cosine similarity threshold should trigger entity merging? Too low → over-merging ("wages" ≈ "prices"), too high → fragmentation. Need to experiment.

10. **Graph density impact**: How does KG edge density affect retrieval? A sparse graph (only high-confidence edges) vs a dense graph (including co-occurrence edges) likely have very different retrieval characteristics.

11. **Random book text quality**: Project Gutenberg economics texts span 1776-1920s. Older texts have more archaic language. How well does the NER/SVO pipeline handle texts from different eras? The `topic=economics` filter also catches some "Home Economics" (cookbooks) — the keyword filter in `kg_gutenberg.py` handles this, but edge cases may slip through.

12. **Cross-book concept alignment**: When building KGs from two different economics books, can Node2Vec embeddings identify structurally equivalent concepts across graphs? (e.g., Smith's "division of labour" ↔ Ricardo's "comparative advantage"). This requires a shared embedding space or alignment step.

13. **Book length normalization**: *The Communist Manifesto* (~30 pages) vs *Wealth of Nations* (~400 pages) produce vastly different graph sizes. Should benchmark metrics be normalized by graph size, or should we set a minimum text length threshold for the random book selection?

14. **Temporal edge simulation**: The Wealth of Nations doesn't naturally have temporal data — it's a static text. How should we simulate temporal edges for the Phase F benchmark? Options: (a) assign chapter-order timestamps to edges (earlier chapters = earlier valid_from), (b) simulate agent sessions that progressively build the KG, (c) use a real agent session log corpus instead. Approach (b) is most realistic for the agent memory use case.

15. **Gutendex API reliability**: The Gutendex API is a third-party service (not run by Project Gutenberg). If it becomes unavailable, the fallback is the [offline CSV catalog](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz) (~14 MB). Should both code paths exist from the start?
