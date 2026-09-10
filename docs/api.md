# API Reference

Every SQL symbol that muninn registers with SQLite. Each entry follows the same template: one-line purpose → signature → minimal example → parameter table → return type → full recipe → see-also.

All symbols are registered by `sqlite3_muninn_init` (i.e. triggered by `.load ./muninn` in the CLI or `sqlite3_load_extension` in host languages). Every supported platform ships the same SQL surface — see [index.md#status](index.md#status) for the platform matrix.

## Contents

- [Vector search](#vector-search)
    - [`hnsw_index`](#hnsw_index) — HNSW KNN virtual table
    - [`muninn_embed_model`](#muninn_embed_model), [`muninn_embed`](#muninn_embed), [`muninn_model_dim`](#muninn_model_dim), [`muninn_models`](#muninn_models)
- [Graph traversal TVFs](#graph-traversal-tvfs)
    - [`graph_bfs`, `graph_dfs`](#graph_bfs-graph_dfs) · [`graph_shortest_path`](#graph_shortest_path) · [`graph_components`](#graph_components) · [`graph_pagerank`](#graph_pagerank)
- [Centrality](#centrality)
    - [`graph_degree`](#graph_degree) · [`graph_node_betweenness`](#graph_node_betweenness) · [`graph_edge_betweenness`](#graph_edge_betweenness) · [`graph_closeness`](#graph_closeness)
- [Community detection](#community-detection)
    - [`graph_leiden`](#graph_leiden) · [`graph_conductance`](#graph_conductance)
- [Adjacency cache](#adjacency-cache)
    - [`graph_adjacency`](#graph_adjacency)
- [Graph selector](#graph-selector)
    - [`graph_select`](#graph_select)
- [Graph embeddings](#graph-embeddings)
    - [`node2vec_train`](#node2vec_train)
- [LLM chat and extraction](#llm-chat-and-extraction)
    - [`muninn_chat_model`](#muninn_chat_model), [`muninn_chat`](#muninn_chat), [`muninn_chat_models`](#muninn_chat_models)
    - [`muninn_extract_entities`](#muninn_extract_entities), [`muninn_extract_relations`](#muninn_extract_relations), [`muninn_extract_ner_re`](#muninn_extract_ner_re)
    - [`muninn_extract_entities_batch`](#muninn_extract_entities_batch), [`muninn_extract_ner_re_batch`](#muninn_extract_ner_re_batch)
    - [`muninn_summarize`](#muninn_summarize)
- [Entity resolution](#entity-resolution)
    - [`muninn_extract_er`](#muninn_extract_er)
- [Group labeling](#group-labeling)
    - [`muninn_label_groups`](#muninn_label_groups)
- [Tokenizers](#tokenizers)
    - [`muninn_tokenize`](#muninn_tokenize), [`muninn_tokenize_text`](#muninn_tokenize_text), [`muninn_token_count`](#muninn_token_count)

Shared conventions that apply to everything below: [graph TVF constraint syntax](#graph-tvf-constraint-syntax), [vector blob format](#vector-blob-format).

---

## Vector search

### `hnsw_index`

Hierarchical Navigable Small World KNN index as a SQLite virtual table, based on Malkov & Yashunin (2018) ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

**Signature**

```sql
CREATE VIRTUAL TABLE name USING hnsw_index(
    dimensions=N,           -- required, vector dimensionality
    metric='cosine',        -- optional, 'l2' | 'cosine' | 'inner_product'
    m=16,                   -- optional, max neighbors per layer, ≥ 2
    ef_construction=200     -- optional, insert beam width, ≥ 1
);
```

**Columns**

| Column | Type | Hidden | Role |
|--------|------|--------|------|
| `rowid` | INTEGER | implicit | Application-chosen ID |
| `vector` | BLOB | no | `float32[dim]` blob — input for INSERT, MATCH constraint for search |
| `distance` | REAL | no | Computed distance (SELECT output only) |
| `k` | INTEGER | yes | Top-k for KNN search |
| `ef_search` | INTEGER | yes | Search beam width (default: `k * 2`) |

**Example — KNN search**

```sql
SELECT rowid, distance FROM vec WHERE vector MATCH ?query AND k = 10;
```

```text
rowid  distance
-----  --------
42     0.0321
17     0.0877
...
```

**Distance metrics**

| Metric | Formula | Range | Notes |
|--------|---------|-------|-------|
| `l2` | Squared Euclidean (`sum((a-b)²)`, no sqrt) | `[0, ∞)` | Monotonic with true L2 — cheaper and comparison-safe |
| `cosine` | `1 - cos(a, b)` | `[0, 2]` | Vectors should be L2-normalized for meaningful results |
| `inner_product` | `-dot(a, b)` | `(-∞, ∞)` | Negated so smaller = more similar |

Distance is computed via ARM NEON (Apple Silicon) or x86 SSE with a scalar fallback — see `src/vec_math.c`.

**Operations**

```sql
INSERT INTO vec(rowid, vector) VALUES (42, ?blob);      -- auto or explicit rowid
DELETE FROM vec WHERE rowid = 42;                        -- soft-delete + neighbor reconnect
DROP TABLE vec;                                           -- removes all shadow tables
-- UPDATE is not supported.
```

**Shadow tables** (auto-managed): `{name}_config`, `{name}_nodes`, `{name}_edges`.

**Full recipe**

```sql
.load ./muninn
CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=4, metric='l2');
INSERT INTO vec(rowid, vector) VALUES (1, X'0000803F0000003F000080BE0000803F');
SELECT rowid, distance FROM vec
  WHERE vector MATCH X'0000803F0000003F000080BE0000803F' AND k = 1;
```

**See also**: [Vector blob format](#vector-blob-format), [Text Embeddings guide](text-embeddings.md).

---

### `muninn_embed_model`

Load a GGUF embedding model into memory and return an opaque handle.

**Signature**

```sql
muninn_embed_model(
    path TEXT,            -- filesystem path to the .gguf file
    n_ctx INTEGER = ?     -- optional context length override (default: model metadata)
) -> POINTER
```

The returned pointer is only useful as the `model` column of an `INSERT` into `temp.muninn_models` — it is not a blob and cannot be stored.

**Example**

```sql
INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/all-MiniLM-L6-v2.Q8_0.gguf');
```

**Parameters**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `path` | TEXT | yes | — | Absolute or relative path to a GGUF embedding model |
| `n_ctx` | INTEGER | no | from metadata, capped at 8192 | Max tokens per embed call |

**Pooling**: Determined by the model's GGUF metadata (BERT → MEAN, Qwen3 → LAST). Never hardcode in your code.

**See also**: [`muninn_models`](#muninn_models), [`muninn_embed`](#muninn_embed), [Text Embeddings](text-embeddings.md#loading-a-gguf-model).

---

### `muninn_embed`

Generate an L2-normalized float32 embedding blob from text.

**Signature**

```sql
muninn_embed(
    model_name TEXT,      -- name registered in temp.muninn_models
    text TEXT             -- input text (single passage)
) -> BLOB
```

**Example**

```sql
SELECT length(muninn_embed('MiniLM', 'hello')) AS bytes;
```

```text
bytes
-----
1536       -- 384 floats × 4 bytes
```

**Returns**: `BLOB`. Raw little-endian IEEE 754 float32 array, already L2-normalized. Byte length is `4 * dim`. Matches the vector format expected by [`hnsw_index`](#hnsw_index).

**Performance**: On macOS, embeddings run on the Metal GPU by default; override with `MUNINN_GPU_LAYERS=0` for CPU. Single-text throughput is ~5k embeddings/sec for MiniLM on M1. No batched variant — submit a single text per call, but many calls can run concurrently from separate connections.

**See also**: [`muninn_embed_model`](#muninn_embed_model), [`hnsw_index`](#hnsw_index).

---

### `muninn_model_dim`

Return the embedding dimensionality of a registered model.

**Signature**

```sql
muninn_model_dim(model_name TEXT) -> INTEGER
```

**Example**

```sql
SELECT muninn_model_dim('MiniLM');
```

```text
384
```

Use this to size the `dimensions` parameter when creating an `hnsw_index` from an unknown model.

---

### `muninn_models`

Eponymous virtual table that tracks loaded embedding models for the connection. Session-scoped — use `temp.muninn_models`.

**Columns**

| Column | Type | Hidden | Role |
|--------|------|--------|------|
| `name` | TEXT | no | User-chosen alias |
| `model` | POINTER | yes (INSERT-only) | Value returned by `muninn_embed_model` |
| `dim` | INTEGER | no | Embedding dimension |

**Operations**

```sql
-- Register
INSERT INTO temp.muninn_models(name, model)
  SELECT 'MiniLM', muninn_embed_model('models/MiniLM.gguf');

-- List
SELECT name, dim FROM temp.muninn_models;

-- Unload
DELETE FROM temp.muninn_models WHERE name = 'MiniLM';
```

---

## Graph traversal TVFs

### Graph TVF constraint syntax

Every graph TVF accepts arguments as `WHERE`-clause constraints on hidden columns, not positional function arguments. This is how SQLite table-valued functions work when xBestIndex uses hidden columns as the parameter channel.

```sql
-- Right:
SELECT ... FROM graph_bfs
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'alice' AND max_depth = 3;

-- Wrong (won't parse):
SELECT ... FROM graph_bfs('edges', 'src', 'dst', 'alice', 3);
```

Shared constraints accepted by every graph TVF:

| Constraint | Type | Required | Description |
|-----------|------|----------|-------------|
| `edge_table` | TEXT | yes | Name of the edge table |
| `src_col` | TEXT | yes | Source column name |
| `dst_col` | TEXT | yes | Destination column name |
| `weight_col` | TEXT | no | Optional weight column (REAL) |
| `direction` | TEXT | no | `'forward'` (default), `'reverse'`, `'both'` |
| `timestamp_col`, `time_start`, `time_end` | TEXT | no | Temporal edge filtering (ISO 8601 strings) |

Table and column names are validated through `id_validate.c` before interpolation — injection-safe.

---

### `graph_bfs` / `graph_dfs`

Breadth- or depth-first traversal from a start node.

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Visited node ID |
| `depth` | INTEGER | Hop distance from start |
| `parent` | TEXT | Parent in traversal tree (NULL for start) |

**Extra constraints**

| Constraint | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_node` | TEXT | yes | — | Start node ID |
| `max_depth` | INTEGER | no | 100 | Maximum hop count |

**Example**

```sql
SELECT node, depth FROM graph_bfs
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'alice' AND max_depth = 2 AND direction = 'both';
```

```text
node    depth
------  -----
alice   0
bob     1
carol   1
dave    2
```

---

### `graph_shortest_path`

Unweighted BFS or weighted Dijkstra shortest path.

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node on the path |
| `distance` | REAL | Cumulative distance from start |
| `path_order` | INTEGER | Position in path (0-indexed) |

**Extra constraints**: `start_node` (required), `end_node` (required). If `weight_col` is provided, uses Dijkstra; otherwise BFS.

```sql
SELECT node, distance, path_order FROM graph_shortest_path
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND start_node = 'a' AND end_node = 'z' AND weight_col = 'weight';
```

---

### `graph_components`

Connected components via union-find with path compression. Treats edges as undirected regardless of `direction`.

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node ID |
| `component_id` | INTEGER | 0-based component index |
| `component_size` | INTEGER | Node count in that component |

```sql
SELECT node, component_id, component_size FROM graph_components
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst';
```

---

### `graph_pagerank`

Iterative PageRank via power method (Page, Brin, Motwani & Winograd, 1999).

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node ID |
| `rank` | REAL | PageRank score (sums to ~1.0 across all nodes) |

**Extra constraints**

| Constraint | Type | Default | Description |
|-----------|------|---------|-------------|
| `damping` | REAL | 0.85 | Random-restart probability |
| `iterations` | INTEGER | 20 | Power iteration steps |

```sql
SELECT node, rank FROM graph_pagerank
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND damping = 0.85 AND iterations = 30
  ORDER BY rank DESC LIMIT 10;
```

---

## Centrality

All four centrality TVFs share the [common constraint syntax](#graph-tvf-constraint-syntax) plus `normalized` (default: `0` for degree/betweenness, `1` for closeness) and temporal filtering.

### `graph_degree`

Per-node degree (in, out, total). Cheapest centrality — run this first.

**Output**: `node TEXT, in_degree REAL, out_degree REAL, degree REAL, centrality REAL`.

```sql
SELECT node, in_degree, out_degree FROM graph_degree
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
  ORDER BY out_degree DESC LIMIT 5;
```

---

### `graph_node_betweenness`

Node betweenness centrality via Brandes (2001), O(VE). Identifies bridge nodes that sit on many shortest paths.

**Output**: `node TEXT, centrality REAL`.

**Extra constraint**

| Constraint | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_approx_threshold` | INTEGER | 50000 | If `|V| >` threshold, use ceil(√N) source-sampling approximation |

```sql
SELECT node, centrality FROM graph_node_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND normalized = 1
  ORDER BY centrality DESC LIMIT 10;
```

**Performance**: O(VE) exact; set a lower `auto_approx_threshold` to accept approximation on graphs you know are large.

---

### `graph_edge_betweenness`

Edge betweenness centrality via Brandes, returns one row per edge in the graph.

**Output**: `src TEXT, dst TEXT, centrality REAL`.

Same constraints as `graph_node_betweenness`.

```sql
SELECT src, dst, centrality FROM graph_edge_betweenness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
  ORDER BY centrality DESC LIMIT 10;
```

---

### `graph_closeness`

Closeness centrality with Wasserman-Faust normalization (default on) for disconnected graphs.

**Output**: `node TEXT, centrality REAL`.

```sql
SELECT node, centrality FROM graph_closeness
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both';
```

See [Centrality and Community](centrality-community.md) for when-to-use guidance and combining with Leiden.

---

## Community detection

### `graph_leiden`

Community detection via the Leiden algorithm (Traag, Waltman & van Eck, 2019) — guarantees well-connected communities, unlike Louvain.

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node ID |
| `community_id` | INTEGER | Contiguous 0-based community index |
| `modularity` | REAL | Global modularity (repeated on every row) |

**Extra constraint**

| Constraint | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | REAL | 1.0 | Higher → more, smaller communities. Lower → fewer, larger |

```sql
SELECT node, community_id FROM graph_leiden
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both' AND resolution = 1.0;
```

See [Centrality and Community](centrality-community.md#leiden-community-detection) for the resolution sweep recipe and Microsoft-GraphRAG-style supernode pattern.

---

### `graph_conductance`

Score any node-to-group membership over a graph: per-group `internal`, `cut`, `vol`, and conductance `phi` (Kannan, Vempala & Vetta, 2004). `graph_leiden`'s `modularity` is global and only defined for the partition Leiden found; conductance is per-group, bounded in `[0, 1]`, and has the same meaning regardless of how many groups exist or where they came from.

**Definition**

For a group `S` in graph `V`:

```text
internal(S) = summed weight of edges with both endpoints in S
cut(S)      = summed weight of edges with exactly one endpoint in S
vol(S)      = 2 * internal(S) + cut(S)
phi(S)      = cut(S) / min(vol(S), vol(V) - vol(S))
```

`phi = 0` is a closed group; `phi = 1` is a group where every edge leaves. Unweighted graphs give every edge weight `1.0`, so `internal` and `cut` are plain counts.

**Example**

```sql
SELECT group_id, size, internal, cut, vol, round(phi, 3) AS phi
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
    AND membership_table = 'assignment'
    AND group_col = 'team' AND member_col = 'node'
  ORDER BY phi;
```

```text
group_id  size  internal  cut  vol   phi
--------  ----  --------  ---  ----  -----
left      4     5.0       1.0  11.0  0.143
right     3     3.0       1.0  7.0   0.143
```

**Output**

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | TEXT | Group identifier, verbatim from `group_col` (INTEGER values are returned as their text form) |
| `size` | INTEGER | Number of graph nodes in the group |
| `internal` | REAL | Summed weight of edges with both endpoints inside |
| `cut` | REAL | Summed weight of edges with exactly one endpoint inside |
| `vol` | REAL | `2 * internal + cut` |
| `phi` | REAL | `cut / min(vol, total_vol - vol)` |

**Extra constraints** (in addition to the [shared constraints](#graph-tvf-constraint-syntax))

| Constraint | Type | Required | Description |
|-----------|------|----------|-------------|
| `membership_table` | TEXT | yes | Table mapping nodes to groups |
| `group_col` | TEXT | yes | Column holding the group identifier |
| `member_col` | TEXT | yes | Column holding the node ID |

The membership triple is the same shape [`muninn_label_groups`](#muninn_label_groups) takes, so one `(node, community_id)` table feeds both. `internal`, `cut`, and `vol` are REAL rather than INTEGER so that weighted and unweighted inputs share one schema, matching [`graph_degree`](#graph_degree).

**Returns**: one row per distinct `group_col` value that has at least one member in the loaded graph, in first-seen order. `phi` is `0.0` (never NULL or NaN) when the denominator is zero. Semantics:

- A graph node with no membership row is *ungrouped*. Its edges still count toward `cut` for whichever endpoint is grouped, and toward the total volume.
- A membership row whose node has no edges in the (possibly temporally filtered) graph is ignored.
- A node listed more than once keeps its first group.
- Conductance is defined on undirected graphs; `direction = 'both'` is the meaningful setting. `forward` and `reverse` are accepted for consistency and score the same edge rows.
- A group with no internal edges scores `phi = 1.0`; a group whose every neighbour is inside it scores `phi = 0.0`. Both are arithmetically correct and both are noise on tiny groups, which is why `size`, `internal`, and `vol` are returned alongside `phi`. Filter on `vol` rather than trusting `phi` alone.

**Full recipe**

```sql
.load ./muninn

CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0);
INSERT INTO edges VALUES
  ('alice', 'bob',   1.0), ('alice', 'carol', 1.0), ('bob',   'carol', 1.0),
  ('bob',   'dave',  1.0), ('carol', 'dave',  1.0),
  ('dave',  'eve',   1.0),   -- bridge
  ('eve',   'frank', 1.0), ('eve',   'grace', 1.0), ('frank', 'grace', 1.0);

-- Score the partition Leiden found
CREATE TEMP TABLE discovered AS
  SELECT node, community_id FROM graph_leiden
    WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
      AND direction = 'both' AND resolution = 1.0;

SELECT group_id, size, internal, cut, vol, round(phi, 3) AS phi
  FROM graph_conductance
  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
    AND membership_table = 'discovered'
    AND group_col = 'community_id' AND member_col = 'node'
    AND vol >= 5
  ORDER BY phi;
```

```text
group_id  size  internal  cut  vol   phi
--------  ----  --------  ---  ----  -----
0         4     5.0       1.0  11.0  0.143
1         3     3.0       1.0  7.0   0.143
```

Performance: one pass over the loaded adjacency, O(E) time and O(k) extra memory for k groups. Reading the membership table is a single `SELECT`.

**See also**: [`graph_leiden`](#graph_leiden), [`muninn_label_groups`](#muninn_label_groups), [Conductance guide](centrality-community.md#conductance-scoring-a-partition) for comparing declared against discovered groupings.

Since: v0.5.0

---

## Adjacency cache

### `graph_adjacency`

Persistent CSR (Compressed Sparse Row) adjacency index over an existing edge table. Triggers on the source edge table log INSERT/UPDATE/DELETE deltas; the CSR rebuilds lazily — incrementally for small deltas (< ~5%), fully for large ones (> ~30%).

**Signature**

```sql
CREATE VIRTUAL TABLE g USING graph_adjacency(
    edge_table='edges',       -- required
    src_col='src',            -- required
    dst_col='dst',            -- required
    weight_col='weight'       -- optional, enables weighted_* columns
);
```

**Columns**

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | Node ID |
| `node_idx` | INTEGER | 0-based internal index (stable across rebuilds) |
| `in_degree`, `out_degree` | INTEGER | Unweighted edge counts |
| `weighted_in_degree`, `weighted_out_degree` | REAL | Sum of edge weights (0 if no weight_col) |

**Administrative commands**

`graph_adjacency` follows the FTS5 command-column convention — the hidden column named after the table accepts command strings.

```sql
INSERT INTO g(g) VALUES ('rebuild');              -- force full rebuild
INSERT INTO g(g) VALUES ('incremental_rebuild');  -- merge deltas into CSR
```

**Shadow tables**: `{name}_config`, `{name}_nodes`, `{name}_degree`, `{name}_csr_fwd`, `{name}_csr_rev`, `{name}_delta`.

**When to use it**: if you query centrality/community/traversal TVFs repeatedly on the same edge table, the CSR cache eliminates redundant table scans. For one-off queries on small graphs, the plain TVFs are equally fast and simpler.

---

## Graph selector

### `graph_select`

dbt-inspired node selector DSL — ancestors, descendants, closures, and set operations on arbitrary edge tables.

**Signature** (positional arguments, unusually for a graph TVF)

```sql
graph_select(edge_table, src_col, dst_col, selector) -> (node, depth, direction)
```

**Example**

```sql
-- Everything within 2 hops of 'C' in both directions
SELECT node, depth, direction FROM graph_select('edges', 'src', 'dst', '2+C+2');
```

```text
node  depth  direction
----  -----  ---------
C     0      self
B     1      ancestor
A     2      ancestor
D     1      descendant
E     2      descendant
```

**Selector grammar**

| Syntax | Meaning |
|--------|---------|
| `node` | Just the node |
| `+node` | Node + all ancestors |
| `node+` | Node + all descendants |
| `N+node` / `node+N` | Depth-limited ancestors / descendants |
| `N+node+M` | Both, depth-limited |
| `+node+` | Both, unlimited |
| `@node` | Transitive build closure (descendants + their ancestors) |
| `A B` | Union (space-separated) |
| `A,B` | Intersection (comma-separated) |
| `not A` | Complement of `A` |

See [Graph Select](graph-select.md) for detailed grammar, precedence rules, and lineage recipes.

---

## Graph embeddings

### `node2vec_train`

Train Node2Vec embeddings (Grover & Leskovec, 2016) and write them into an existing [`hnsw_index`](#hnsw_index).

**Signature** (positional; all 12 args required)

```sql
node2vec_train(
    edge_table TEXT,
    src_col TEXT,
    dst_col TEXT,
    output_table TEXT,     -- must already exist as hnsw_index
    dimensions INTEGER,    -- must match the HNSW table
    p REAL, q REAL,        -- walk bias (p=q=1 → DeepWalk)
    num_walks INTEGER,
    walk_length INTEGER,
    window INTEGER,
    neg_samples INTEGER,
    learning_rate REAL,
    epochs INTEGER
) -> INTEGER               -- count of nodes embedded
```

**Example**

```sql
CREATE VIRTUAL TABLE user_emb USING hnsw_index(dimensions=64, metric='cosine');

SELECT node2vec_train(
  'edges', 'src', 'dst', 'user_emb',
  64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 5
);
```

Edges are treated as undirected. See [Node2Vec](node2vec.md) for p/q tuning and dimension sizing.

---

## LLM chat and extraction

All `muninn_*` LLM functions require a GGUF chat model registered in `temp.muninn_chat_models` (except the tokenizer functions, which accept any loaded model). See [Chat and Extraction](chat-and-extraction.md) for model recommendations.

### `muninn_chat_model`

Load a GGUF chat model and return an opaque handle.

```sql
muninn_chat_model(path TEXT, n_ctx INTEGER = ?) -> POINTER
```

Default `n_ctx` = `max(8192, train_ctx / 8)`, capped at the training context. Use the optional override only when you have memory headroom to use more of the model's native context window.

```sql
INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');
```

---

### `muninn_chat`

Free-form generation, optionally constrained by a GBNF grammar.

**Signature**

```sql
muninn_chat(
    model_name TEXT,
    prompt TEXT,
    grammar TEXT = NULL,       -- optional GBNF
    max_tokens INTEGER = n_ctx,
    system_prompt TEXT = NULL,
    skip_think INTEGER = 0     -- 1 → inject closed <think></think> (Qwen3.5 etc.)
) -> TEXT
```

**Example**

```sql
SELECT muninn_chat(
  'Qwen3.5-4B',
  'Reply with JSON: {"greeting": "<word>"}',
  'root ::= "{\"greeting\":\"hello\"}"',  -- tiny GBNF grammar
  32
);
```

---

### `muninn_chat_models`

Eponymous virtual table for chat model lifecycle — same pattern as `muninn_models`.

**Columns**: `name TEXT`, `model POINTER` (INSERT-only, hidden), `n_ctx INTEGER`.

```sql
DELETE FROM temp.muninn_chat_models WHERE name = 'Qwen3.5-4B';
```

---

### `muninn_extract_entities`

Named entity recognition with grammar-constrained JSON output.

**Signature**

```sql
muninn_extract_entities(
    model_name TEXT,
    text TEXT,
    labels TEXT = NULL,        -- comma-separated list, e.g. 'person,org,date'
    skip_think INTEGER = 0
) -> JSON
```

**Supervised** — labels provided:

```sql
SELECT muninn_extract_entities('Qwen3.5-4B',
  'Elon Musk founded Tesla in 2003.',
  'person,organization,date');
```

```text
{"entities":[
  {"text":"Elon Musk","type":"person","score":0.98},
  {"text":"Tesla","type":"organization","score":0.97},
  {"text":"2003","type":"date","score":0.95}
]}
```

**Unsupervised** (open extraction) — omit `labels`:

```sql
SELECT muninn_extract_entities('Qwen3.5-4B', 'Elon Musk founded Tesla in 2003.');
```

**Returns**: TEXT with SQLite subtype `'J'` — `json_each(result, '$.entities')` works directly without a `json(...)` wrap.

---

### `muninn_extract_relations`

Relation extraction between entities.

```sql
muninn_extract_relations(
    model_name TEXT,
    text TEXT,
    entities_json TEXT = NULL,   -- supervised: array of {"text": ..., "type": ...}
    skip_think INTEGER = 0
) -> JSON   -- {"relations": [{"head": ..., "rel": ..., "tail": ..., "score": ...}]}
```

---

### `muninn_extract_ner_re`

Combined NER + RE in a single generation — useful when the two tasks share context.

```sql
muninn_extract_ner_re(
    model_name TEXT,
    text TEXT,
    ent_labels TEXT = NULL,
    rel_labels TEXT = NULL,
    skip_think INTEGER = 0
) -> JSON   -- {"entities": [...], "relations": [...]}
```

---

### `muninn_extract_entities_batch`

Multi-sequence batch NER via `llama_batch`. Each prompt gets its own `seq_id` in the KV cache.

```sql
muninn_extract_entities_batch(
    model_name TEXT,
    texts_json TEXT,           -- JSON array of strings: ["text1", "text2", ...]
    labels TEXT = NULL,
    batch_size INTEGER = 4     -- max 8
) -> JSON                       -- array of per-text NER results
```

---

### `muninn_extract_ner_re_batch`

Combined NER + RE, batched.

```sql
muninn_extract_ner_re_batch(
    model_name TEXT,
    texts_json TEXT,
    ent_labels TEXT = NULL,
    rel_labels TEXT = NULL,
    batch_size INTEGER = 4
) -> JSON
```

---

### `muninn_summarize`

Abstractive summarization. Qwen3.5 `<think>` blocks are stripped from the output automatically.

```sql
muninn_summarize(
    model_name TEXT,
    text TEXT,
    max_tokens INTEGER = n_ctx
) -> TEXT
```

See [Chat and Extraction](chat-and-extraction.md) for prompt tuning, grammar authoring, and batch strategy.

---

## Entity resolution

### `muninn_extract_er`

End-to-end entity resolution pipeline: KNN blocking → pairwise scoring (Jaro-Winkler × cosine) → optional LLM refinement in the borderline band → optional edge-betweenness bridge removal → Leiden clustering.

**Signature**

```sql
muninn_extract_er(
    hnsw_table TEXT,                -- HNSW vtable containing entity embeddings
    name_col TEXT,                  -- entity name column on the *source* table
    k INTEGER,                      -- KNN neighbors per entity
    dist_threshold REAL,            -- max cosine distance for candidate pairs
    jw_weight REAL,                 -- 0..1, JW contribution weight (1-jw_weight = cosine)
    borderline_delta REAL,          -- LLM refinement window (0 disables LLM)
    chat_model TEXT,                -- required if borderline_delta > 0, else NULL
    edge_betweenness_threshold REAL, -- NULL to skip bridge removal
    type_guard TEXT                 -- 'same_source' | 'diff_type' | NULL
) -> JSON  -- {"clusters":{"entity_id": cluster_id, ...}}
```

The implicit match threshold is `1 - dist_threshold + borderline_delta`. See [Entity Resolution](entity-resolution.md) for the full cascade walk-through.

---

## Group labeling

### `muninn_label_groups`

TVF that reads a membership table (e.g. Leiden output), batches members by group, calls a chat model for a concise label per group.

**Signature** (constraint-style)

```sql
SELECT group_id, label, member_count FROM muninn_label_groups
  WHERE model = 'Qwen3.5-4B'
    AND membership_table = 'cluster_members'
    AND group_col = 'cluster_id'
    AND member_col = 'member_name'
    AND min_group_size = 3
    AND max_members_in_prompt = 10
    AND system_prompt = 'Output ONLY a concise label (3-8 words).';
```

| Constraint | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | TEXT | yes | — | Chat model name |
| `membership_table` | TEXT | yes | — | Source table |
| `group_col` | TEXT | yes | — | Column containing group IDs |
| `member_col` | TEXT | yes | — | Column containing member names |
| `min_group_size` | INTEGER | no | 1 | Skip groups smaller than this |
| `max_members_in_prompt` | INTEGER | no | 20 | Truncate member list in prompt |
| `system_prompt` | TEXT | no | generic | Custom system message |

**Output**: `group_id TEXT, label TEXT, member_count INTEGER`.

---

## Tokenizers

All three functions accept the name of any model registered in `temp.muninn_models` *or* `temp.muninn_chat_models` — they use whichever has a matching name.

### `muninn_tokenize`

Return JSON array of token IDs. Result has SQLite subtype `'J'`.

```sql
SELECT muninn_tokenize('MiniLM', 'Hello, world!');
```

```text
[7592, 1010, 2088, 999]
```

### `muninn_tokenize_text`

Return JSON array of token text pieces. Useful for inspecting how a tokenizer segments input.

```sql
SELECT muninn_tokenize_text('MiniLM', 'Hello, world!');
```

```text
["hello", ",", "world", "!"]
```

### `muninn_token_count`

Return the token count as an integer — cheaper than materializing the token array.

```sql
SELECT muninn_token_count('MiniLM', 'Hello, world!');
```

```text
4
```

---

## Shared conventions

### Vector blob format

Every function that accepts or returns a vector uses the same encoding: **raw little-endian IEEE 754 `float32` array, no header, `4 × dimensions` bytes**. This applies to `muninn_embed` output, `hnsw_index` insert/match values, and the `vector` column in HNSW shadow tables. See [Text Embeddings → Vector format reference](text-embeddings.md#vector-format-reference) for per-language construction snippets.

### NULL behavior

Most functions return NULL when given NULL input (standard SQL coercion). Exceptions:

- `muninn_embed`, `muninn_chat`, and the extraction family raise `sqlite3_result_error` if the named model is not registered.
- `hnsw_index` INSERT raises an error if the vector blob length does not equal `4 * dimensions` — no silent truncation.

### Environment variables

| Var | Default | Effect |
|-----|---------|--------|
| `MUNINN_GPU_LAYERS` | 99 on macOS, 0 on Linux | Layers offloaded to GPU (Metal on macOS) |
| `MUNINN_LOG_LEVEL` | silent | `verbose` / `warn` / `error` — llama.cpp log verbosity |
