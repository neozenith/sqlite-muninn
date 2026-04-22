---
name: muninn-graphrag
description: >
  Builds end-to-end GraphRAG retrieval over a text corpus entirely in SQLite:
  chunk → embed → extract entities and relations → build knowledge graph →
  detect communities → label clusters → retrieve via vector seed + graph
  expansion + centrality ranking. Composes muninn_embed, hnsw_index,
  muninn_extract_ner_re_batch, graph_leiden, muninn_label_groups, and
  muninn_extract_er. Use when the user mentions "GraphRAG", "Graph RAG",
  "knowledge graph RAG", "retrieval-augmented generation", "RAG pipeline in
  SQLite", "KG retrieval", "entity resolution", "muninn_extract_er",
  "community labeling", "KG indexing", or wants to build Microsoft GraphRAG-
  style retrieval on a corpus.
license: MIT
---

# muninn-graphrag — Composite retrieval pipeline in SQL

GraphRAG combines vector retrieval (for recall) with knowledge-graph structure (for reasoning/explanation) and community summaries (for coverage). muninn implements every stage in SQL — no Python model server, no extra extensions.

This skill is the orchestration layer. It assumes you already know the building blocks:

- [muninn-embed-text](../muninn-embed-text/SKILL.md) — `muninn_embed`
- [muninn-vector-search](../muninn-vector-search/SKILL.md) — `hnsw_index`
- [muninn-chat-extract](../muninn-chat-extract/SKILL.md) — `muninn_extract_ner_re_batch`
- [muninn-graph-algorithms](../muninn-graph-algorithms/SKILL.md) — `graph_leiden`, `graph_node_betweenness`

## Pipeline shape

```
Build phase:
  corpus → chunk → embed → HNSW index
                     ↓
                   NER+RE → knowledge-graph edges → Leiden → labels

Retrieve phase:
  query → embed → KNN seed → BFS expand → rank by centrality → top-k
```

## Prerequisites

```sql
.load ./muninn

INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');

INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');
```

For long corpora, Qwen3.5-4B Q4_K_M on Metal is a good speed/quality trade-off.

## Build phase

### 1 — Source corpus

```sql
CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT);

INSERT INTO documents(content) VALUES
  ('Tesla acquired Maxwell Technologies in 2019 for $218 million.'),
  ('SpaceX, founded by Elon Musk in 2002, launched Falcon 9 in 2010.'),
  ('Apple released the iPhone in 2007; Steve Jobs unveiled it at Macworld.'),
  ('Tim Cook succeeded Steve Jobs as Apple CEO in 2011.'),
  ('Elon Musk became CEO of Tesla in 2008.'),
  ('Maxwell Technologies was based in San Diego.'),
  ('Steve Jobs co-founded Apple with Steve Wozniak in 1976.'),
  ('SpaceX is headquartered in Hawthorne, California.');
```

### 2 — Chunk (if needed) and embed

```sql
-- Short docs; skip chunking. Check length first:
SELECT id, muninn_token_count('MiniLM', content) AS tokens FROM documents;

CREATE VIRTUAL TABLE doc_vec USING hnsw_index(dimensions=384, metric='cosine');
INSERT INTO doc_vec(rowid, vector)
  SELECT id, muninn_embed('MiniLM', content) FROM documents;
```

For longer documents, chunk by paragraph in your host language or via a recursive CTE that splits on sentence boundaries.

### 3 — Extract entities + relations

```sql
CREATE TABLE kg_triples(
  doc_id INTEGER, head TEXT, rel TEXT, tail TEXT, score REAL
);

-- Batch NER+RE is ~3-5× faster than per-row calls
WITH batches AS (
  SELECT json_group_array(content) AS texts,
         json_group_array(id) AS ids
  FROM documents
)
INSERT INTO kg_triples(doc_id, head, rel, tail, score)
SELECT
  json_extract(ids, '$[' || b.key || ']'),
  json_extract(r.value, '$.head'),
  json_extract(r.value, '$.rel'),
  json_extract(r.value, '$.tail'),
  json_extract(r.value, '$.score')
FROM batches,
     json_each(muninn_extract_ner_re_batch(
       'Qwen3.5-4B',
       (SELECT texts FROM batches),
       'person,organization,date,location,product',
       'founded,acquired,launched,released,leads,based_in',
       4
     )) b,
     json_each(json_extract(b.value, '$.relations')) r;
```

### 4 — Build KG edge table

```sql
CREATE TABLE kg_edges(src TEXT, dst TEXT, rel TEXT, weight REAL);

INSERT INTO kg_edges(src, dst, rel, weight)
  SELECT head, tail, rel, AVG(score)
  FROM kg_triples
  GROUP BY head, tail, rel;
```

### 5 — Leiden communities

```sql
CREATE TABLE kg_communities AS
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND resolution = 1.0;
```

### 6 — Label each community

```sql
CREATE TABLE kg_labels AS
SELECT group_id AS community_id, label, member_count FROM muninn_label_groups
  WHERE model = 'Qwen3.5-4B'
    AND membership_table = 'kg_communities'
    AND group_col = 'community_id'
    AND member_col = 'node'
    AND min_group_size = 2
    AND max_members_in_prompt = 10;
```

Now you have: a vector index over chunks, a KG edge table, community membership per entity, and a human-readable label per community.

## Retrieve phase

### A — VSS seed

```sql
WITH query AS (
  SELECT muninn_embed('MiniLM', 'Who leads Apple today?') AS q
)
SELECT rowid, distance FROM doc_vec, query
  WHERE vector MATCH query.q AND k = 3;
```

### B — Graph expansion

For each seed document, pull entities mentioned, then BFS 1 hop in the KG to gather related entities.

```sql
WITH query AS (
  SELECT muninn_embed('MiniLM', 'Who leads Apple today?') AS q
),
seeds AS (
  SELECT v.rowid AS doc_id
  FROM doc_vec v, query
  WHERE v.vector MATCH query.q AND k = 3
),
seed_entities AS (
  SELECT DISTINCT head AS entity FROM kg_triples WHERE doc_id IN (SELECT doc_id FROM seeds)
  UNION
  SELECT DISTINCT tail FROM kg_triples WHERE doc_id IN (SELECT doc_id FROM seeds)
)
SELECT DISTINCT node FROM graph_bfs
  WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = (SELECT entity FROM seed_entities LIMIT 1)
    AND max_depth = 2 AND direction = 'both';
```

### C — Rank by centrality

```sql
WITH bc AS (
  SELECT node, centrality FROM graph_node_betweenness
    WHERE edge_table = 'kg_edges' AND src_col = 'src' AND dst_col = 'dst'
      AND direction = 'both' AND normalized = 1
)
SELECT node, centrality FROM bc ORDER BY centrality DESC LIMIT 10;
```

### D — Assemble context

Join the ranked entities with their document text and community labels to form a prompt payload for a downstream LLM:

```sql
SELECT
  e.node            AS entity,
  l.label           AS community,
  group_concat(d.content, ' || ') AS evidence
FROM kg_communities c
JOIN kg_labels l USING (community_id)
JOIN kg_edges e ON (e.src = c.node OR e.dst = c.node)
JOIN kg_triples t ON (t.head = c.node OR t.tail = c.node)
JOIN documents d ON d.id = t.doc_id
WHERE c.node IN (SELECT node FROM /* your ranked seed set */ LIMIT 5)
GROUP BY e.node, l.label;
```

## Entity resolution shortcut

`muninn_extract_er` runs the full deduplication cascade (KNN blocking → Jaro-Winkler × cosine scoring → LLM refinement → optional edge-betweenness bridge removal → Leiden) in a single call. Use it after NER when the same entity appears under variants.

```sql
-- Build an entity-name vector index first
CREATE VIRTUAL TABLE entity_vec USING hnsw_index(dimensions=384, metric='cosine');
INSERT INTO entity_vec(rowid, vector)
  SELECT rowid, muninn_embed('MiniLM', head) FROM kg_triples;

SELECT muninn_extract_er(
  'entity_vec',   -- HNSW with entity embeddings
  'head',         -- name column on kg_triples
  10,             -- KNN neighbors
  0.3,            -- max cosine distance for candidate pairs
  0.5,            -- Jaro-Winkler weight (cosine gets 1-0.5 = 0.5)
  0.1,            -- LLM refinement band (0 disables LLM)
  'Qwen3.5-4B',   -- chat model, required when band > 0
  NULL,           -- edge_betweenness_threshold (NULL = skip)
  'same_source'   -- type guard
);
```

Returns `{"clusters": {"entity_name": cluster_id, ...}}`. Cluster ids are 0-indexed.

## Pipeline performance targets

| Stage | Throughput (M1 Metal, MiniLM + Qwen3.5-4B Q4_K_M) |
|-------|----------------------------------------------------|
| Embed | ~5k docs/sec |
| NER+RE (single) | ~1-3 docs/sec |
| NER+RE (batch 4) | ~4-10 docs/sec |
| Leiden on 10k-node KG | seconds |
| Betweenness on 10k-node KG | 10s-minutes (O(VE)) |

WASM: expect 5-10× slower across the board; not recommended for full pipelines over >1k docs.

## Common pitfalls

- **Qwen3.5 `<think>` leaks into relations** — set `skip_think = 1` in `muninn_extract_ner_re_batch` calls for throughput.
- **Community labels look generic** — raise `min_group_size` to 5+ and provide a `system_prompt` scoped to your domain.
- **KG too dense to Leiden cleanly** — increase `resolution` in `graph_leiden`; run edge-betweenness pruning before clustering for very dense graphs.
- **Re-running is slow** — none of these tables are temporary. Persist the DB and incrementally append new docs; only re-run Leiden/betweenness when the graph structure materially changes.

## See also

- [graphrag-cookbook.md](../../docs/graphrag-cookbook.md) — extended reference with Mermaid pipeline diagram
- [entity-resolution.md](../../docs/entity-resolution.md) — `muninn_extract_er` cascade in depth
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) — the original paper this implementation tracks
