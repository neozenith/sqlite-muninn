# Knowledge Graph Benchmark Plan

A benchmark for **combined vector search + graph traversal** workflows — the GraphRAG pattern — exercising all three of vec_graph's subsystems (HNSW, graph TVFs, Node2Vec) together.

**Status:** Plan only. Not implemented. Requires primitive benchmarks (vector search + graph traversal) to be validated first.

---

## Table of Contents

1. [Vision](#vision)
2. [Research: State of the Art](#research-state-of-the-art)
   - [Vector-Seeded Graph Traversal](#vector-seeded-graph-traversal-the-graphrag-pattern)
   - [Community Detection: Louvain and Leiden](#community-detection-louvain-and-leiden)
   - [Graph Centrality Measures](#graph-centrality-measures)
   - [Node2Vec Deep Dive and Alternatives](#node2vec-deep-dive-and-alternatives)
   - [Knowledge Graph Embeddings (KGE)](#knowledge-graph-embeddings-kge)
   - [Entity Coalescing and Synonym Merging](#entity-coalescing-and-synonym-merging)
3. [Entity Extraction Approaches](#entity-extraction-approaches)
   - [NER Models for Entity Extraction](#approach-1-ner-models-for-entity-extraction)
   - [Noun-Verb-Noun SVO Pattern](#approach-2-noun-verb-noun-svo-pattern-extraction)
   - [FTS5/BM25 for Concept Discovery](#approach-3-fts5bm25-for-concept-discovery)
   - [Hybrid Pipeline Recommendation](#hybrid-pipeline-recommendation)
4. [HuggingFace Model Recommendations](#huggingface-model-recommendations)
5. [Dataset: Wealth of Nations](#dataset-the-wealth-of-nations-as-a-knowledge-graph)
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

### Community Detection: Louvain and Leiden

#### What is Louvain?

The **Louvain algorithm** (Blondel et al., 2008) detects communities in large networks by optimizing **modularity** — a measure of how densely connected nodes within a community are compared to random connections.

**Algorithm (two phases, iterated):**

1. **Local moving**: Each node is moved to the neighboring community that maximizes modularity gain. Repeat until no improvement.
2. **Aggregation**: Collapse each community into a single supernode. Build a new graph of supernodes. Return to phase 1.

**Modularity** Q measures the fraction of edges within communities minus the expected fraction if edges were random:

```
Q = (1/2m) Σ [A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)
```

where `A_ij` is the adjacency matrix, `k_i` is the degree of node i, `m` is total edges, and δ is 1 when nodes i,j are in the same community.

**Complexity**: O(n log n) — fast enough for million-node graphs.

**Problem**: Louvain can produce **badly connected or even disconnected communities**. In experiments, up to 25% of communities are badly connected and up to 16% are disconnected (Traag et al., 2019). This happens because the greedy local moving phase can trap nodes in communities they shouldn't belong to.

#### What is Leiden?

The **Leiden algorithm** (Traag, Waltman & van Eck, 2019) fixes Louvain's connectivity guarantees by adding a **refinement phase**:

1. **Local moving** (same as Louvain)
2. **Refinement**: Communities may be split to guarantee all communities are **well-connected**. Only nodes well-connected within their community are retained.
3. **Aggregation** (based on the *refined* partition, not the local-moving partition)

**Key improvements:**
- **Guaranteed connectivity**: All identified communities are both well-separated and well-connected
- **Faster convergence**: Actually runs faster than Louvain in practice
- **Better partitions**: Higher modularity scores at convergence
- **Subset optimality**: When converged, all vertices are optimally assigned

**For vec_graph**: Microsoft GraphRAG uses Leiden for community hierarchy. Adding Leiden to `graph_components` (or as a new `graph_communities` TVF) would enable the hierarchical retrieval pattern in Phase C of the benchmark. Connected components (what we have now) only finds disconnected subgraphs — Louvain/Leiden finds *densely-connected clusters within a single connected component*.

**Implementation consideration**: The simplest C implementation of Louvain is ~300 lines. Leiden adds ~100 more for the refinement phase. Both fit naturally as a new algorithm option in `graph_components` or as a separate `graph_louvain` / `graph_leiden` TVF.

---

### Graph Centrality Measures

#### Betweenness Centrality

**What it is**: Betweenness centrality measures how often a node lies on the shortest path between other node pairs. Nodes with high betweenness are "bridge" concepts connecting otherwise separate clusters.

```
BC(v) = Σ_{s≠v≠t} [σ_st(v) / σ_st]
```

where σ_st is the total number of shortest paths from s to t, and σ_st(v) is the number of those paths passing through v.

**Why it matters for knowledge graphs**: In a Wealth of Nations KG, "market price" would have high betweenness because it bridges the labour/wages cluster to the rent/profit cluster. Identifying these bridge concepts helps:

1. **Retrieval**: Prioritize bridge nodes in context assembly (they connect disparate topics)
2. **Graph summarization**: Bridge nodes are natural candidates for community summaries
3. **Query routing**: If a query touches two clusters, the bridge concept is likely relevant

**Algorithm**: Brandes' algorithm computes betweenness for all nodes in O(VE) time (unweighted) or O(VE + V² log V) (weighted). For our ~3K node / ~10K edge KG, this runs in milliseconds.

**Implementation for vec_graph**: A new `graph_betweenness` TVF returning `(node TEXT, centrality REAL)`. Uses Brandes' algorithm internally. Could also expose `graph_closeness` and `graph_degree` in the same pass.

#### Other Centrality Measures

| Measure | Formula | Meaning | Use in KG |
|---------|---------|---------|-----------|
| **Degree centrality** | `deg(v) / (N-1)` | How many direct connections | Hub entities (most-referenced concepts) |
| **Closeness centrality** | `(N-1) / Σ d(v,u)` | Average distance to all other nodes | Central concepts accessible from everywhere |
| **Eigenvector centrality** | `x_v ∝ Σ A_{vu} × x_u` | Connected to other well-connected nodes | Authoritative concepts (like PageRank but undirected) |
| **PageRank** | Power iteration with damping | Random walk stationary distribution | **Already implemented** in vec_graph |

**Benchmark plan**: Compare centrality-guided retrieval (use betweenness to prioritize nodes after BFS expansion) vs uniform retrieval (all BFS-discovered nodes weighted equally). The hypothesis is that centrality-guided retrieval achieves better precision with fewer nodes.

---

### Node2Vec Deep Dive and Alternatives

#### How Node2Vec Works (Already in vec_graph)

Node2Vec (Grover & Leskovec, 2016) learns vector embeddings for graph nodes using a two-step process:

1. **Biased random walks**: Simulate walks on the graph where the bias is controlled by two parameters:
   - **p** (return parameter): Controls likelihood of returning to the previous node. Low p → stay local (BFS-like). High p → explore (DFS-like).
   - **q** (in-out parameter): Controls preference for inward vs outward nodes. Low q → explore outward (DFS-like). High q → stay close (BFS-like).

2. **Skip-gram with Negative Sampling (SGNS)**: Treat random walk sequences as "sentences" and node IDs as "words". Train a Word2Vec-style model to predict context nodes from target nodes.

**What the parameters capture:**
- **p=1, q=1**: Equivalent to DeepWalk (uniform random walks). Captures a balance of structural and community information.
- **Low p, high q**: BFS-like walks. Captures **homophily** — nodes in the same community get similar embeddings.
- **High p, low q**: DFS-like walks. Captures **structural equivalence** — nodes with similar structural roles (e.g., all "hub" nodes) get similar embeddings, even if they're in different communities.

**Our implementation** (`src/node2vec.c`):
- Full biased random walk with second-order Markov chain
- Skip-gram with negative sampling using frequency^0.75 distribution
- Sigmoid lookup table (1000 entries) for speed
- L2-normalized output embeddings (ready for cosine similarity in HNSW)
- Linear learning rate decay
- Supports undirected graphs, ~10K nodes max (linear node lookup)

#### Alternatives to Node2Vec

| Method | Type | Advantages | Disadvantages | Best For |
|--------|------|-----------|---------------|----------|
| **DeepWalk** (2014) | Shallow / random walks | Simplest implementation, uniform walks | No bias control (p,q); less expressive | Simple graphs where you don't need homophily/structural tuning |
| **LINE** (2015) | Shallow / edge sampling | Explicitly optimizes 1st+2nd order proximity; scales to millions | Doesn't capture higher-order structure | Very large graphs; when memory is limited |
| **GraphSAGE** (2017) | Deep / GNN | **Inductive** — generalizes to unseen nodes; uses node features | Requires node feature vectors as input; heavier to train | Dynamic graphs with new nodes; when node features exist |
| **GAT** (2018) | Deep / GNN | Attention mechanism highlights important neighbors; best quality | Heavy compute (GPU needed); not easily portable to C | When quality matters more than portability |
| **GCN** (2017) | Deep / GNN | Simple, strong baseline for node classification | Fixed aggregation (mean), not inductive | Static graphs with labels for supervised learning |
| **TransE** (2013) | KGE / translational | Specifically designed for (head, relation, tail) triples | Requires typed relations; can't model symmetric relations | Knowledge graphs with typed edges (not our case) |
| **RotatE** (2019) | KGE / rotational | Handles symmetry, antisymmetry, inversion, composition | More complex; still requires typed relations | Complex relation patterns in KGs |

**Why Node2Vec is the right choice for vec_graph:**

1. **Zero-dependency**: Runs in pure C with no ML framework. GraphSAGE/GAT require PyTorch/TensorFlow.
2. **No node features needed**: Node2Vec works from graph topology alone. GraphSAGE needs feature vectors.
3. **Tunable**: The p,q parameters let us control what the embeddings capture.
4. **Compatible with HNSW**: Output embeddings go directly into our HNSW index for similarity search.
5. **Fast for small graphs**: For ~3K nodes, Node2Vec trains in seconds. GNNs would be overkill.

**When to consider alternatives**: If vec_graph ever needs to support graphs >100K nodes or inductive settings (embedding new nodes without retraining), GraphSAGE would be worth investigating. For now, Node2Vec is ideal.

**Benchmark opportunity**: Sweep p,q values on the Wealth of Nations KG and measure how different walk biases affect retrieval quality. Hypothesis: low p (BFS-like) captures the economic concept clusters best, while high p (DFS-like) captures structural roles (all "institution" nodes get similar embeddings regardless of which cluster they're in).

---

### Knowledge Graph Embeddings (KGE)

KGE methods differ from Node2Vec in that they explicitly model **typed relations** between entities:

- **TransE**: Models relations as translations: `head + relation ≈ tail` in vector space. "England" + "TRADES_WITH" ≈ "Holland". Simple but can't model symmetric relations (if A trades with B, B trades with A — but TransE would need two different vectors).
- **RotatE**: Models relations as rotations in complex vector space. Handles symmetry, antisymmetry, inversion, and composition patterns.
- **ComplEx**: Uses complex-valued embeddings with Hermitian dot product. Good for symmetric and antisymmetric relations.

**Relevance to our benchmark**: The Wealth of Nations KG has typed relations (CAUSES, COMPOSED_OF, etc.). KGE methods could produce better embeddings for relation-aware retrieval. However, they require a training pipeline that's harder to implement in pure C. The pragmatic approach: use Node2Vec for structure-only embeddings stored in HNSW, and handle relation types at the query level (e.g., filter BFS expansion by relation type).

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

## Dataset: The Wealth of Nations as a Knowledge Graph

Adam Smith's *The Wealth of Nations* (Gutenberg #3300) is already available as a text dataset in the benchmark suite (~2,500 passage chunks). For the KG benchmark, these passages become the source for entity and relation extraction.

### Entity Types

| Type | Examples | Expected Count | Extraction Source |
|------|----------|---------------|-------------------|
| **Concept** | division of labour, market price, natural price, rent, profit, wages | ~200 | GLiNER + FTS5 |
| **Actor** | labourer, merchant, sovereign, landlord | ~50 | GLiNER + SVO subjects |
| **Institution** | East India Company, Bank of England, Parliament | ~30 | GLiNER |
| **Place** | England, Holland, Bengal, China, North America | ~40 | GLiNER |
| **Work** | (chapter/book references within the text) | ~50 | Pattern matching |
| **Commodity** | corn, silver, gold, wool, linen | ~30 | GLiNER + FTS5 |

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

### Graph Structure

- **Nodes**: ~400 entities + ~2,500 passage chunks = ~2,900 nodes
- **Edges**: ~5,000-10,000 relations (entity-entity + entity-passage links)
- **Chapter hierarchy**: Book -> Chapter -> Passage as a tree backbone
- **Properties**: Each entity node has a text description + embedding; each passage node has original text + embedding
- **Node types**: `entity` (extracted concepts/people/places) and `passage` (text chunks)

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

### Priority 1: Betweenness Centrality

**TVF**: `graph_betweenness(edge_table, src_col, dst_col, weight_col, normalized)`

**Output**: `(node TEXT, centrality REAL)`

**Algorithm**: Brandes' algorithm — O(VE) unweighted, O(VE + V² log V) weighted.

**Implementation sketch** (~200 lines of C):
1. For each node s: run BFS/Dijkstra from s, tracking shortest path counts and predecessors
2. Backward accumulation: propagate dependency scores from leaves to root
3. Normalize: divide by (N-1)(N-2) for undirected or (N-1)(N-2)/2 for directed

**Why it's valuable**: Identifies "bridge" concepts between KG clusters. In the Wealth of Nations, high-betweenness nodes like "market price" connect the labour/wages cluster to the rent/profit cluster — exactly the concepts needed for cross-topic retrieval.

### Priority 2: Louvain Community Detection

**TVF**: `graph_louvain(edge_table, src_col, dst_col, weight_col, resolution)`

**Output**: `(node TEXT, community_id INTEGER, modularity REAL)`

**Algorithm**: Louvain — O(n log n) average case.

**Implementation sketch** (~300 lines of C):
1. Initialize: each node in its own community
2. Local moving: for each node, compute modularity gain for moving to each neighbor's community. Move to best.
3. Repeat until no improvement.
4. Aggregation: collapse communities into supernodes, build new graph. Return to step 2.

**Upgrade path**: Leiden adds a refinement phase (~100 more lines) guaranteeing connected communities. Start with Louvain, upgrade to Leiden later.

### Nice-to-Have: Degree/Closeness Centrality

These are simpler and could be exposed alongside betweenness:

- **Degree centrality**: Just count edges per node. Trivial.
- **Closeness centrality**: Run BFS from each node, compute average distance. O(V(V+E)).

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
   - `keybert` — keyword extraction (optional)

---

## Implementation Sketch

### File Structure

```
python/
  benchmark_kg.py              # KG benchmark runner
  benchmark_kg_analyze.py      # KG benchmark analysis + charts
  kg_extract.py                # Entity/relation extraction pipeline
  kg_coalesce.py               # Entity resolution + synonym merging
benchmarks/
  kg/                          # Cached knowledge graph (SQLite DB + metadata)
  results/kg_*.jsonl           # KG benchmark results
  vectors/kg_*.npy             # Pre-computed entity embeddings
```

### JSONL Schema

```json
{
    "timestamp": "...",
    "benchmark_type": "knowledge_graph",
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

### Makefile Targets

```makefile
benchmark-kg-extract:                          ## Extract KG from Wealth of Nations
	.venv/bin/python python/kg_extract.py

benchmark-kg-coalesce:                         ## Entity resolution + dedup
	.venv/bin/python python/kg_coalesce.py

benchmark-kg: vec_graph$(EXT)                  ## Run KG benchmark
	.venv/bin/python python/benchmark_kg.py

benchmark-kg-analyze:                          ## Analyze KG results → charts
	.venv/bin/python python/benchmark_kg_analyze.py
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
