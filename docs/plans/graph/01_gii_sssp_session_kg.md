# Phase 1-2: GII Core + SSSP Shadow Tables

**Date:** 2026-02-24 (consolidated 2026-02-25)
**Status:** Not started
**Depends on:** None (foundational phases)
**Blocks:** Phases 3, 4, 5, 6

---

## 1. Overview

This document consolidates the **GII (Graph Incremental Index)** virtual table
design (namespace scoping, blocked CSR, delta cascade) with the **SSSP shadow
tables** (shared shortest-path module, cached all-pairs distances, SSSP delta
queue). These form a single foundational unit because:

1. **SSSP depends on GII namespace and delta cascade.** The `_sssp` and
   `_sssp_delta` shadow tables are keyed by `namespace_id` and populated by the
   CSR rebuild's downstream emission logic. Designing them separately risks
   impedance mismatch.

2. **The success criteria requires both.** A complete session-log Knowledge
   Graph demo builder (Section 2) exercises scoped namespaces, incremental CSR
   rebuild, and centrality analysis (betweenness/closeness) powered by SSSP
   caching. Neither phase alone satisfies the acceptance test.

3. **The delta cascade is a single event pipeline.** Changes flow:
   `_delta` → CSR rebuild → `_sssp_delta` → TVF cache read.
   Splitting this across documents obscures the end-to-end sequence.

### Deliverables

**Phase A — GII Core:**

1. `namespace_cols` parameter on `CREATE VIRTUAL TABLE`
2. `{name}_namespace` shadow table (namespace registry)
3. All existing shadow tables keyed by `namespace_id`
4. Triggers that capture scope column values and route deltas per namespace
5. Per-namespace full rebuild and incremental rebuild
6. Namespace filter on xBestIndex/xFilter (hidden column)
7. Updated `graph_data_load_from_gii()` accepting an optional namespace key
8. Backward compatibility: omitting `namespace_cols` is identical to current behavior
9. `features` parameter parsing and downstream delta table creation

**Phase B — SSSP Shadow Tables:**

1. `graph_sssp.c` / `graph_sssp.h` — shared SSSP module extracted from `graph_centrality.c`
2. `{name}_sssp` shadow table — all-pairs distances/sigma as packed BLOBs
3. `{name}_sssp_delta` shadow table — stale source-node tracking
4. Generation counter protocol for staleness detection
5. Integration with betweenness TVF (predecessor reconstruction from cached dist[])
6. Integration with closeness TVF (partial recomputation from delta queue)

**Phase C — Session-Log Demo Builder:**

1. New demo_builder variant: `SessionLogKG` — ingests Claude Code `*.jsonl` logs
2. Produces a KG database following the existing demo_builder table convention
3. GII virtual table with `namespace_cols='project_id,session_id'`
4. SSSP caching for tool/file centrality analysis
5. Viz tool compatibility via auto-discovery (zero changes to viz)

---

## 2. Success Criteria: Session-Log Knowledge Graph

> **Escalators, Not Stairs:** The GII/SSSP implementation is not complete when it
> passes unit tests. It is complete when it powers a real-world knowledge graph
> that a human can explore in the viz tool. The session-log KG is that acceptance
> test.

### 2.1 What It Is

A new demo_builder DB variant that ingests Claude Code session logs from
`~/.claude/projects/` and produces a KG database compatible with the viz tool's
auto-discovery conventions. This database is the primary artifact consumed by the
`.claude/skills/introspect/` skill for graph-based session analysis.

### 2.2 Source Data: Claude Code JSONL Logs

The introspect skill already parses JSONL session logs at:

```
~/.claude/projects/{project-path-kebab-cased}/{session_uuid}.jsonl
~/.claude/projects/{project-path-kebab-cased}/{session_uuid}/*.jsonl  # subagents
```

Each JSONL event contains:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `user`, `assistant`, `tool_use`, `tool_result`, `thinking`, `summary`, `progress` |
| `timestamp` | ISO 8601 | Event time |
| `uuid` | string | Unique event ID |
| `parentUuid` | string | Parent event ID (event tree) |
| `sessionId` | string | Session UUID |
| `agentId` | string | Subagent ID (if sidechain) |
| `isSidechain` | bool | True if subagent |
| `message.role` | string | `user` or `assistant` |
| `message.content` | string | Message text |
| `message.model` | string | Model ID (e.g., `claude-opus-4-5-20251101`) |
| `message.usage` | object | Token counts (input, output, cache_read, cache_creation) |
| `name` | string | Tool name (for `tool_use` events) |
| `input` | object | Tool input (for `tool_use` events) |

### 2.3 Extraction Pipeline (8 Phases)

The session-log demo builder follows the same 8-phase pipeline as the existing
Gutenberg-based demo builder, but with domain-specific extraction:

| Phase | Gutenberg Variant | Session-Log Variant |
|-------|-------------------|---------------------|
| 1. Chunks + FTS + Embeddings | Text paragraphs from book | User/assistant messages as chunks |
| 2. NER (Entity Extraction) | GLiNER zero-shot NER | Rule-based: tool names, file paths, model IDs, session IDs |
| 3. RE (Relation Extraction) | GLiREL zero-shot RE | Structural: parent-child events, tool invocations, file operations |
| 4. Entity Embeddings | sentence-transformers on entity names | sentence-transformers on entity names |
| 5. UMAP | 2D + 3D projections | 2D + 3D projections |
| 6. Entity Resolution | HNSW blocking + Jaro-Winkler + Leiden | HNSW blocking + Jaro-Winkler + Leiden |
| 7. Node2Vec | Structural embeddings | Structural embeddings |
| 8. Metadata + Validation | Book metadata | Project/session metadata |

### 2.4 Entity Types

| Entity Type | Source | Example |
|-------------|--------|---------|
| `tool` | `tool_use` events → `name` field | `Bash`, `Read`, `Edit`, `Write`, `Grep`, `Glob` |
| `file` | Tool inputs → `file_path`, `path` fields | `src/gii.c`, `docs/plans/graph/01.md` |
| `model` | `message.model` field | `claude-opus-4-5-20251101` |
| `session` | `sessionId` field | `535770ff-8b4f-4187` (truncated for display) |
| `agent` | `agentId` field (subagents) | `a4a6e3b` |
| `concept` | NER on message text (optional, can use GLiNER) | `HNSW`, `delta cascade`, `namespace` |
| `error` | Tool results with error patterns | `FileNotFoundError`, `SyntaxError` |

### 2.5 Relation Types

| Relation | Source → Target | Extraction Rule |
|----------|----------------|-----------------|
| `uses_tool` | session/agent → tool | `tool_use` event in session |
| `reads_file` | session/agent → file | `Read` tool with `file_path` input |
| `writes_file` | session/agent → file | `Write`/`Edit` tool with `file_path` input |
| `spawns_agent` | session → agent | `isSidechain=true` with matching `sessionId` |
| `responds_to` | event → event | `parentUuid` linkage |
| `runs_command` | agent → tool | `Bash` tool invocations |
| `searches_for` | agent → file | `Grep`/`Glob` tool invocations |
| `mentions` | chunk → concept | NER on message text |
| `uses_model` | session → model | `message.model` field |
| `errors_on` | tool → error | Tool results containing error messages |

### 2.6 Schema Mapping to Demo Builder Convention

The session-log variant produces the same 10 user tables + 3 HNSW VTs + 1 FTS5
table as the Gutenberg variant. The viz tool auto-discovers these by naming
convention — zero changes required.

```sql
-- Phase 1: Chunks + FTS + Embeddings
CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id);
CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(dimensions=384, metric='cosine', m=16, ef_construction=200);

-- Phase 2: Entity Extraction (rule-based, not GLiNER)
CREATE TABLE entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT,             -- tool, file, model, session, agent, concept, error
    source TEXT NOT NULL,         -- 'rule_based' or 'gliner'
    chunk_id INTEGER REFERENCES chunks(chunk_id),
    confidence REAL DEFAULT 1.0
);

-- Phase 3: Relation Extraction (structural, not GLiREL)
CREATE TABLE relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel_type TEXT,                -- uses_tool, reads_file, writes_file, spawns_agent, etc.
    weight REAL DEFAULT 1.0,
    chunk_id INTEGER,
    source TEXT NOT NULL          -- 'structural' or 'glirel'
);

-- Phase 4: Entity Embeddings
CREATE VIRTUAL TABLE entities_vec USING hnsw_index(dimensions=384, metric='cosine');
CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL);

-- Phase 5: UMAP
CREATE TABLE chunks_vec_umap (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL);
CREATE TABLE entities_vec_umap (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL);

-- Phase 6: Entity Resolution → nodes + edges
CREATE TABLE entity_clusters (name TEXT PRIMARY KEY, canonical TEXT NOT NULL);
CREATE TABLE nodes (node_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
                    entity_type TEXT, mention_count INTEGER DEFAULT 0);
CREATE TABLE edges (src TEXT NOT NULL, dst TEXT NOT NULL, rel_type TEXT,
                    weight REAL DEFAULT 1.0, PRIMARY KEY (src, dst, rel_type));

-- Phase 7: Node2Vec
CREATE VIRTUAL TABLE node2vec_emb USING hnsw_index(dimensions=64, metric='cosine');

-- Phase 8: Metadata
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
```

### 2.7 GII Integration

After the demo_builder produces the database, the introspect skill (or a
post-build step) creates a GII virtual table on the `edges` table:

```sql
-- Scoped by project and session for multi-tenant analysis
CREATE VIRTUAL TABLE kg_graph USING gii(
    edge_table='edges',
    src_col='src',
    dst_col='dst',
    weight_col='weight',
    namespace_cols='project_id,session_id',
    features='sssp'
);
```

**Note:** The `edges` table as produced by the demo builder does not have
`project_id` / `session_id` columns. To support GII scoping, the session-log
variant must add these columns to the `edges` table:

```sql
CREATE TABLE edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel_type TEXT,
    weight REAL DEFAULT 1.0,
    project_id TEXT,             -- NEW: scope column for GII namespace
    session_id TEXT,             -- NEW: scope column for GII namespace
    PRIMARY KEY (src, dst, rel_type, project_id, session_id)
);
```

This allows centrality and community queries scoped to a single session:

```sql
-- Betweenness centrality for a specific session's tool/file graph
SELECT node, centrality
FROM graph_betweenness
WHERE gii_table = 'kg_graph'
  AND namespace = 'project-x' || X'00' || 'session-abc'
ORDER BY centrality DESC;
```

### 2.8 SSSP Integration

With `features='sssp'` enabled, the GII caches all-pairs shortest paths for
each namespace. For a typical session with ~50-200 nodes (tools + files +
concepts), the SSSP cache is trivial (~2 KB per source) and enables instant
betweenness/closeness queries:

| Session Size | Nodes | SSSP Cache | Cache Time |
|-------------|-------|------------|------------|
| Small (100 events) | ~30 | 7 KB | <1ms |
| Medium (500 events) | ~100 | 80 KB | ~10ms |
| Large (2000 events) | ~300 | 720 KB | ~50ms |

These sizes are well within the `max_sssp_nodes=10000` default threshold.

### 2.9 Viz Compatibility

The viz tool auto-discovers tables by naming convention. Because the session-log
variant follows the same schema as the Gutenberg variant, ALL viz features work
without modification:

| Viz Feature | Supported | Table Used |
|-------------|-----------|------------|
| Pipeline Explorer (7 stages) | Yes | chunks, chunks_vec_nodes, entities, relations, entity_clusters, nodes/edges, node2vec_emb_nodes |
| GraphRAG query | Yes | chunks_fts, entities_vec, entity_vec_map, relations, entity_clusters |
| KG Search (FTS + VSS + BFS) | Yes | chunks_fts, chunks_vec, entities_vec, graph_bfs |
| Graph Explorer (Cytoscape.js) | Yes | nodes, edges |
| VSS Explorer (Deck.GL) | Yes | chunks_vec_umap, entities_vec_umap |

### 2.10 Acceptance Tests

| # | Test | Pass Criteria |
|---|------|---------------|
| 1 | Session-log demo builder produces valid DB | All 10 user tables + 3 HNSW VTs + 1 FTS5 populated |
| 2 | GII VT created with namespace scoping | `CREATE VIRTUAL TABLE` succeeds, `_namespace` table populated |
| 3 | Per-session CSR isolation | Inserting edges in session A does NOT trigger rebuild in session B |
| 4 | SSSP cache populated on first centrality query | `_sssp` shadow table has V rows after first betweenness query |
| 5 | Incremental update works | Add 10 events to a session → only that session's CSR blocks rebuild |
| 6 | Viz tool loads DB without changes | `make -C viz dev` → open DB → all 7 pipeline stages display |
| 7 | GraphRAG returns meaningful results | Query "which tools were most used" → returns ranked tool nodes |
| 8 | Betweenness identifies bridge files | Files that connect multiple tool clusters ranked high |

---

## 3. GII Design

### 3.1 SQL Syntax

```sql
-- Scoped GII: partition by project_id and session_id
CREATE VIRTUAL TABLE g USING gii(
    edge_table='edges',
    src_col='src',
    dst_col='dst',
    weight_col='weight',
    namespace_cols='project_id,session_id',
    features='sssp,components,communities'
);

-- Unscoped (backward compatible): equivalent to current behavior
CREATE VIRTUAL TABLE g USING gii(
    edge_table='edges',
    src_col='src',
    dst_col='dst'
);
```

The `namespace_cols` parameter is a comma-separated list of column names from the edge
table. Each column must pass `id_validate()`. The order matters: it defines the
composite key ordering for the namespace hash.

The `features` parameter is a comma-separated list of downstream layers to enable.
Valid values: `sssp`, `components`, `communities`. When a feature is enabled, the GII
creates the corresponding downstream delta tables and emits change notifications to
them during CSR rebuild.

### 3.2 Namespace Key Computation

A composite scope combination (e.g., `project_id=42, session_id='abc'`) is mapped to
a single `namespace_id INTEGER` via:

1. **Text representation:** Concatenate the TEXT values of each scope column with a
   NUL byte separator: `"42\0abc"`. SQLite coerces all values to TEXT for this purpose.
2. **Hash:** Compute DJB2 hash of the concatenated byte string (reusing the existing
   `graph_str_hash()` from `graph_common.h`, extended to handle embedded NUL bytes).
3. **Registry lookup:** Look up the hash in `{name}_namespace`. If not found, INSERT
   a new row with the next available `namespace_id` (auto-increment).
4. **Collision handling:** The registry stores the full composite key text alongside the
   hash. On lookup, compare the stored key to resolve collisions (linear probe in the
   registry table, which is a B-tree and handles this naturally via the `scope_key` UNIQUE
   constraint).

The `namespace_id` is a dense integer starting at 0. When `namespace_cols` is omitted,
there is a single implicit namespace with `namespace_id = 0` and `scope_key = ''`.

### 3.3 Shadow Table Schema Changes

All shadow tables gain a `namespace_id` column as part of their primary key. The
namespace registry is a new shadow table.

### 3.4 Data Flow

```
Edge INSERT into source table
    |
    v
Trigger fires: captures NEW.src, NEW.dst, NEW.weight, scope col values
    |
    v
INSERT INTO {name}_delta(namespace_id, src, dst, weight, op, scope_key)
    |                     ^
    |      namespace_id resolved at query time (not trigger time)
    |      trigger stores raw scope_key; namespace_id resolved during rebuild
    v
gii_ensure_fresh() called on next query
    |
    +-- delta count > 0 -> rebuild (full or incremental)
    |   |
    |   +-- Group deltas by scope_key -> resolve to namespace_id
    |   |
    |   +-- Per-namespace: load CSR blocks, apply deltas, store
    |   |
    |   +-- Emit downstream deltas (_sssp_delta, _comp_delta, _comm_delta)
    |   |
    |   +-- Update per-namespace generation in _config
    |
    +-- delta count = 0 -> serve from shadow tables
        |
        +-- Filter by namespace_id in _nodes JOIN _degree query
```

**Key design decision:** Triggers store the raw `scope_key` (concatenated scope column
values) in the delta table rather than resolving `namespace_id` at trigger time. This
avoids the trigger needing to call a UDF or query the namespace registry. Resolution
happens during rebuild, which is the only time `namespace_id` matters.

### 3.5 Temporal Deferral Note

True temporal graph support — interval-based edge validity, overlap queries,
multi-validity edges, and open-ended intervals — requires a fundamentally different
CSR design where edges carry temporal metadata and the index structure supports
time-range queries natively.

This is deferred to a future **TGII** (Temporal Graph Incremental Index) construct.

For now, temporal filtering is handled at query time by the existing TVFs via SQL
WHERE clauses on raw tables. `graph_data_load()` already supports `timestamp_col`,
`time_start`, and `time_end` parameters, which apply temporal predicates when loading
edges directly from the source table.

The GII CSR does NOT store timestamps. Temporal queries bypass the CSR and hit the
raw edge tables through `graph_data_load()`. This means temporal queries do not
benefit from the CSR acceleration, but they remain fully functional.

The GII architecture is designed to be composable: a future TGII would wrap or extend
GII (e.g., maintaining multiple CSR snapshots keyed by time intervals), not replace it.
The namespace partitioning, delta cascade, and downstream emission patterns established
here apply equally to a temporal extension.

---

## 4. Delta Cascade Architecture

The delta cascade is the centerpiece of the GII's incremental maintenance strategy.
It defines how changes propagate from raw edge mutations through the CSR layer and
into downstream analytical layers (SSSP, components, communities). Each layer
maintains its own delta queue, and changes flow lazily from lower layers to higher
layers.

### 4.1 Per-Layer Delta Queues

The GII maintains a cascade of delta queues. Each layer has its own delta table:

```
_delta       ->  edge-level changes (this phase)
_sssp_delta  ->  stale SSSP source-node indices (this phase)
_comp_delta  ->  nodes with potentially changed components (Phase 3)
_comm_delta  ->  nodes in changed neighborhoods for Leiden (Phase 4)
```

The `_delta` and `_sssp_delta` tables are implemented in this phase. The `_comp_delta`
and `_comm_delta` tables are created when their corresponding feature is enabled via
the `features` parameter, but they are populated by the CSR rebuild logic defined here.

**Eager emission, lazy consumption:** When the CSR is rebuilt, the GII eagerly writes
affected node information into all downstream delta tables. But each downstream layer
only reads its delta queue when a query touches it. If no query touches SSSP, the
`_sssp_delta` table accumulates entries indefinitely without triggering any work.

**One-way flow:** Each delta table is a unidirectional queue from its producer
(the CSR rebuild) to its consumer (the downstream TVF). The producer writes; the
consumer reads and clears.

### 4.2 Threshold-Based Rebuild Strategy

Each layer uses a threshold-based strategy to decide between selective, delta-flush,
and full rebuild:

```
                              ┌─────────────────────────────────┐
                              │     CSR Delta Processing        │
                              └────────────────┬────────────────┘
                                               │
                         delta_ratio = |delta| / total_edges
                                               │
                    ┌──────────────────────────┬┼──────────────────────────┐
                    │                          ││                          │
          ratio < θ_selective          θ_selective <= ratio          ratio >= θ_full
          (default 5%)                 < θ_full (default 30%)       (default 30%)
                    │                          │                           │
          ┌─────────▼─────────┐    ┌──────────▼──────────┐    ┌─────────▼─────────┐
          │ Selective Block   │    │   Delta Flush        │    │  Full Rebuild      │
          │ Rebuild           │    │                      │    │                    │
          │ Only affected     │    │ Flush entire delta   │    │ Discard CSR        │
          │ CSR blocks        │    │ queue, rebuild       │    │ Rebuild from       │
          │ (O(block_size))   │    │ affected namespace   │    │ scratch via SQL    │
          └─────────┬─────────┘    └──────────┬──────────┘    └─────────┬─────────┘
                    │                          │                         │
                    │ gen_adj unchanged        │ gen_adj unchanged       │ gen_adj++
                    │ emit to _sssp_delta      │ emit to _sssp_delta    │ generation mismatch
                    │                          │                         │ signals full SSSP
                    v                          v                         v
          downstream delta emit     downstream delta emit      no downstream delta
```

The thresholds are configurable per GII instance via `_config`:

```sql
INSERT OR REPLACE INTO "{name}_config" (key, value) VALUES
    ('theta_selective', '0.05'),   -- below 5%: selective block rebuild
    ('theta_full', '0.30');        -- above 30%: full rebuild
```

### 4.3 Downstream Delta Emission

During CSR rebuild, the GII identifies which nodes were affected and writes their
indices into downstream delta tables. The emission depends on the rebuild strategy:

| Rebuild Strategy | _sssp_delta | _comp_delta | _comm_delta |
|-----------------|-------------|-------------|-------------|
| Selective block | All nodes in rebuilt blocks | Same | Same |
| Delta flush | All src/dst nodes from applied deltas | Same | Same |
| Full rebuild | Generation bump (no delta written) | Generation bump | Generation bump |

For selective and delta-flush strategies, emission is conservative: some emitted nodes
may not actually be affected. The downstream consumer handles this efficiently (e.g.,
closeness can selectively recompute only stale sources).

### 4.4 Event Sequence Diagram

```
 User INSERTs                 _delta          CSR Layer         _sssp_delta       SSSP Layer
 ─────────────               ────────        ──────────        ─────────────     ────────────
      │                                          │                                    │
      │── INSERT edge ───────►│                  │                                    │
      │── INSERT edge ───────►│                  │                                    │
      │                       │                  │                                    │
      │                       │                  │                                    │
      │── QUERY (triggers rebuild) ─────────────►│                                    │
      │                       │◄── read deltas ──│                                    │
      │                       │                  │── apply to CSR blocks               │
      │                       │── clear() ──────►│                                    │
      │                       │                  │── identify affected nodes           │
      │                       │                  │── emit ──────────────────►│          │
      │                       │                  │                          │          │
      │                       │                  │◄── return CSR data ──────│          │
      │◄─ query results ──────│                  │                          │          │
      │                                          │                          │          │
      │                                          │                          │          │
      │── QUERY centrality ──────────────────────────────────────────────────────────►│
      │                                          │                          │◄── read ─│
      │                                          │                          │   stale  │
      │                                          │                          │   count  │
      │                                          │                          │          │
      │                                          │                     ┌────┼──────────┤
      │                                          │                     │ rebuild stale │
      │                                          │                     │ sources only  │
      │                                          │                     └────┼──────────┤
      │                                          │                          │── clear()│
      │◄─ centrality results ────────────────────────────────────────────────────────┤│
```

### 4.5 Event Pipeline: Complete Scenario Walkthroughs

#### Scenario 1: Small Change (1 edge insert, ratio < 5%)

```
1. User: INSERT INTO edges VALUES('Read', 'src/gii.c', 'reads_file', 1.0, 'proj-x', 'session-a');
2. Trigger:  INSERT INTO kg_graph_delta (scope_key='proj-x\0session-a', src='Read', dst='src/gii.c', weight=1.0, op=1)
3. Later:    SELECT * FROM graph_betweenness WHERE gii_table='kg_graph' AND ...
4. gii_ensure_fresh():
   a. delta count = 1, total_edges = 200, ratio = 0.005 (0.5%) < θ_selective (5%)
   b. Strategy: SELECTIVE BLOCK REBUILD
   c. Resolve scope_key 'proj-x\0session-a' -> namespace_id=3
   d. Load CSR block containing src='Read' (block_id = hash(Read) / BLOCK_SIZE)
   e. Apply delta: add edge Read->src/gii.c to forward CSR, reverse to rev CSR
   f. Store updated block
   g. generation_adj[ns=3] UNCHANGED
   h. Emit: INSERT INTO kg_graph_sssp_delta (namespace_id=3, source_idx=idx_of(Read))
   i. Emit: INSERT INTO kg_graph_sssp_delta (namespace_id=3, source_idx=idx_of(src/gii.c))
   j. Clear: DELETE FROM kg_graph_delta WHERE scope_key='proj-x\0session-a'
5. Betweenness: checks _sssp_delta: 2 stale sources.
   Since betweenness needs all-pairs: FULL SSSP rebuild for namespace 3.
   Write all V rows to _sssp. Clear _sssp_delta.
6. Return centrality scores.
```

#### Scenario 2: Medium Change (500 edges, 5% <= ratio < 30%)

```
1. User bulk-inserts 500 events from a new session import.
2. Trigger fires 500 times -> 500 rows in _delta with scope_key='proj-x\0session-b'
3. Query triggers gii_ensure_fresh():
   a. delta count = 500, total_edges = 2000, ratio = 25% -> DELTA FLUSH strategy
   b. Resolve scope_key -> namespace_id=7
   c. Read all 500 deltas for namespace 7
   d. Apply all deltas to the in-memory CSR for namespace 7
   e. Write back entire CSR for namespace 7
   f. generation_adj[ns=7] UNCHANGED
   g. Collect unique src/dst indices from the 500 deltas -> ~400 affected nodes
   h. Emit 400 rows into _sssp_delta for namespace 7
   i. Also emit 400 rows into _comp_delta if components feature enabled
   j. Clear _delta for scope_key='proj-x\0session-b'
4. Closeness query on namespace 7:
   sssp_delta ratio = 400/2000 = 20% > θ_sssp_selective (10%) but < θ_sssp_full (50%)
   -> FULL SSSP rebuild for namespace 7 (cheaper at this ratio)
```

#### Scenario 3: Large Change (3000 edges, ratio >= 30%)

```
1. Bulk import of 3000 events.
2. 3000 delta rows.
3. gii_ensure_fresh():
   a. ratio = 3000/4000 = 75% >= θ_full (30%) -> FULL REBUILD
   b. Discard all CSR data for this namespace
   c. Re-query edge table with SQL WHERE scope columns match
   d. Build CSR from scratch
   e. generation_adj[ns=X] INCREMENTED (e.g., 5 -> 6)
   f. NO downstream delta emission (generation bump is sufficient)
   g. Clear _delta for this namespace
4. SSSP query: gen_sssp (5) != gen_adj (6) -> generation mismatch -> FULL SSSP REBUILD
```

#### Scenario 4: Multi-Namespace Isolation

```
1. Insert 1 edge into session-a (namespace 3), 1 edge into session-b (namespace 7).
2. _delta has 2 rows with different scope_keys.
3. gii_ensure_fresh():
   a. Group deltas by scope_key: {ns=3: 1 delta, ns=7: 1 delta}
   b. For ns=3: ratio = 1/200 = 0.5% -> SELECTIVE BLOCK REBUILD of ns=3 only
   c. For ns=7: ratio = 1/2000 = 0.05% -> SELECTIVE BLOCK REBUILD of ns=7 only
   d. Emit to _sssp_delta: 2 rows for ns=3, 2 rows for ns=7
   e. CSR of ns=3 is NOT affected by ns=7's rebuild and vice versa
4. Betweenness query scoped to ns=3: only processes _sssp_delta for ns=3
   ns=7's _sssp_delta accumulates but is not consumed until someone queries ns=7
```

---

## 5. GiiVtab Struct Changes

**File:** `src/gii.c`, lines 33-42

Current struct:

```c
typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;
    int64_t generation;
} GiiVtab;
```

New struct:

```c
typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name;      /* virtual table name (for shadow table prefixes) */
    char *edge_table;
    char *src_col;
    char *dst_col;
    char *weight_col;     /* NULL if unweighted */
    int64_t generation;   /* global generation counter */

    /* Namespace support */
    char **namespace_cols;     /* array of scope column names, NULL if unscoped */
    int namespace_col_count;   /* 0 if unscoped */

    /* Feature flags (bitmask) */
    int features;              /* FEATURE_SSSP | FEATURE_COMPONENTS | FEATURE_COMMUNITIES */
} GiiVtab;

/* Feature bitmask values */
#define FEATURE_SSSP         0x01
#define FEATURE_COMPONENTS   0x02
#define FEATURE_COMMUNITIES  0x04
```

The `gii_xDisconnect()` function must free `namespace_cols`:

```c
static int gii_xDisconnect(sqlite3_vtab *pVTab) {
    GiiVtab *vtab = (GiiVtab *)pVTab;
    sqlite3_free(vtab->vtab_name);
    sqlite3_free(vtab->edge_table);
    sqlite3_free(vtab->src_col);
    sqlite3_free(vtab->dst_col);
    sqlite3_free(vtab->weight_col);
    for (int i = 0; i < vtab->namespace_col_count; i++)
        sqlite3_free(vtab->namespace_cols[i]);
    sqlite3_free(vtab->namespace_cols);
    sqlite3_free(vtab);
    return SQLITE_OK;
}
```

---

## 6. Shadow Table Schemas (Complete)

All `CREATE TABLE` statements for the new schema. These replace the current statements
in `gii_create_shadow_tables()`.

### 6.1 Core Shadow Tables (Phase A)

```sql
-- Config: global KV store (unchanged schema, new key conventions)
CREATE TABLE IF NOT EXISTS "{name}_config" (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Namespace registry (NEW)
CREATE TABLE IF NOT EXISTS "{name}_namespace" (
    namespace_id INTEGER PRIMARY KEY,
    scope_key    TEXT UNIQUE NOT NULL,
    scope_hash   INTEGER NOT NULL
);

-- Node registry: scoped string ID <-> integer index
CREATE TABLE IF NOT EXISTS "{name}_nodes" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    id           TEXT NOT NULL,
    PRIMARY KEY (namespace_id, idx)
);
CREATE INDEX IF NOT EXISTS "{name}_nodes_id"
    ON "{name}_nodes"(namespace_id, id);

-- Degree sequence: scoped
CREATE TABLE IF NOT EXISTS "{name}_degree" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    idx          INTEGER NOT NULL,
    in_deg       INTEGER,
    out_deg      INTEGER,
    w_in_deg     REAL,
    w_out_deg    REAL,
    PRIMARY KEY (namespace_id, idx)
);

-- Forward CSR: scoped blocked storage
CREATE TABLE IF NOT EXISTS "{name}_csr_fwd" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);

-- Reverse CSR: scoped blocked storage
CREATE TABLE IF NOT EXISTS "{name}_csr_rev" (
    namespace_id INTEGER NOT NULL DEFAULT 0,
    block_id     INTEGER NOT NULL,
    offsets      BLOB,
    targets      BLOB,
    weights      BLOB,
    PRIMARY KEY (namespace_id, block_id)
);

-- Delta log: stores raw scope_key for later resolution
CREATE TABLE IF NOT EXISTS "{name}_delta" (
    rowid     INTEGER PRIMARY KEY,
    scope_key TEXT NOT NULL DEFAULT '',
    src       TEXT,
    dst       TEXT,
    weight    REAL,
    op        INTEGER
);
CREATE INDEX IF NOT EXISTS "{name}_delta_scope"
    ON "{name}_delta"(scope_key);
```

### 6.2 SSSP Shadow Tables (Phase B)

Created when `features` includes `'sssp'`:

```sql
-- All-pairs shortest-path cache
CREATE TABLE IF NOT EXISTS "{name}_sssp" (
    namespace_id  INTEGER NOT NULL,
    source_idx    INTEGER NOT NULL,
    distances     BLOB NOT NULL,        -- double[V], little-endian
    sigma         BLOB,                 -- double[V], little-endian (NULL for closeness-only)
    PRIMARY KEY (namespace_id, source_idx)
);

-- Stale source-node tracking
CREATE TABLE IF NOT EXISTS "{name}_sssp_delta" (
    namespace_id  INTEGER NOT NULL,
    source_idx    INTEGER NOT NULL,
    PRIMARY KEY (namespace_id, source_idx)
);
```

**Column semantics for `_sssp`:**

| Column | Type | Description |
|--------|------|-------------|
| `namespace_id` | INTEGER | Scope partition. 0 for non-scoped VTs |
| `source_idx` | INTEGER | Node index of the SSSP source (0..V-1) |
| `distances` | BLOB | `double[V]` packed little-endian. `dist[i]` = shortest distance. -1.0 = unreachable |
| `sigma` | BLOB | `double[V]` packed little-endian. `sigma[i]` = shortest-path count. NULL if only closeness computed |

**What is NOT cached:**

- **Predecessor lists (`pred[]`).** O(VE) worst case in total. Betweenness
  reconstructs `pred[]` on-the-fly from cached `dist[]` (see Section 9).
- **Stack order.** Reconstructed from `dist[]` by sorting nodes by non-decreasing
  distance. Costs O(V log V) per source.

**BLOB Encoding:** All values are stored as raw `double` (IEEE 754 binary64) in
platform-native byte order. No byte-swapping needed.

```c
/* Write dist[] to BLOB */
sqlite3_bind_blob(stmt, col, ws->dist,
                  ws->node_count * (int)sizeof(double), SQLITE_TRANSIENT);

/* Read dist[] from BLOB */
const double *cached_dist = (const double *)sqlite3_column_blob(stmt, col);
int n_doubles = sqlite3_column_bytes(stmt, col) / (int)sizeof(double);
```

---

## 7. Trigger SQL

### 7.1 Unscoped (namespace_cols omitted)

Identical to current behavior, with the addition of `scope_key = ''`:

```sql
-- AFTER INSERT
CREATE TRIGGER IF NOT EXISTS "{name}_ai" AFTER INSERT ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', NEW."{src_col}", NEW."{dst_col}", NEW."{weight_expr}", 1);
END;

-- AFTER DELETE
CREATE TRIGGER IF NOT EXISTS "{name}_ad" AFTER DELETE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', OLD."{src_col}", OLD."{dst_col}", OLD."{weight_expr}", 2);
END;

-- AFTER UPDATE
CREATE TRIGGER IF NOT EXISTS "{name}_au" AFTER UPDATE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', OLD."{src_col}", OLD."{dst_col}", OLD."{weight_expr}", 2);
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES ('', NEW."{src_col}", NEW."{dst_col}", NEW."{weight_expr}", 1);
END;
```

### 7.2 Scoped (namespace_cols = 'project_id,session_id')

The scope_key expression concatenates scope column values with NUL separators:

```sql
-- AFTER INSERT
CREATE TRIGGER IF NOT EXISTS "{name}_ai" AFTER INSERT ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(NEW."project_id" AS TEXT) || X'00' || CAST(NEW."session_id" AS TEXT),
        NEW."{src_col}",
        NEW."{dst_col}",
        NEW."{weight_expr}",
        1
    );
END;

-- AFTER DELETE
CREATE TRIGGER IF NOT EXISTS "{name}_ad" AFTER DELETE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(OLD."project_id" AS TEXT) || X'00' || CAST(OLD."session_id" AS TEXT),
        OLD."{src_col}",
        OLD."{dst_col}",
        OLD."{weight_expr}",
        2
    );
END;

-- AFTER UPDATE
CREATE TRIGGER IF NOT EXISTS "{name}_au" AFTER UPDATE ON "{edge_table}" BEGIN
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(OLD."project_id" AS TEXT) || X'00' || CAST(OLD."session_id" AS TEXT),
        OLD."{src_col}",
        OLD."{dst_col}",
        OLD."{weight_expr}",
        2
    );
    INSERT INTO "{name}_delta"(scope_key, src, dst, weight, op)
    VALUES (
        CAST(NEW."project_id" AS TEXT) || X'00' || CAST(NEW."session_id" AS TEXT),
        NEW."{src_col}",
        NEW."{dst_col}",
        NEW."{weight_expr}",
        1
    );
END;
```

### 7.3 Trigger Generation Code

The trigger SQL is generated dynamically in `gii_create_triggers()`:

```c
/* Build scope_key expression based on namespace_cols */
char *scope_expr;
if (vtab->namespace_col_count == 0) {
    scope_expr = sqlite3_mprintf("''");
} else {
    /* Concatenate: CAST(NEW."col1" AS TEXT) || X'00' || CAST(NEW."col2" AS TEXT) ... */
    scope_expr = sqlite3_mprintf("CAST(%s.\"%w\" AS TEXT)", row_prefix, vtab->namespace_cols[0]);
    for (int i = 1; i < vtab->namespace_col_count; i++) {
        char *prev = scope_expr;
        scope_expr = sqlite3_mprintf("%s || X'00' || CAST(%s.\"%w\" AS TEXT)",
                                     prev, row_prefix, vtab->namespace_cols[i]);
        sqlite3_free(prev);
    }
}
```

---

## 8. Namespace Key Computation

### 8.1 Hash Function

Extend `graph_str_hash()` to handle embedded NUL bytes:

```c
/* DJB2 hash with explicit length (handles embedded NUL bytes) */
static uint64_t scope_key_hash(const char *key, int key_len) {
    uint64_t hash = 5381;
    for (int i = 0; i < key_len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)key[i];
    }
    return hash;
}
```

### 8.2 Auto-Registration

```c
/*
 * Look up or create a namespace_id for the given scope_key.
 * Returns the namespace_id (>= 0) or -1 on error.
 */
static int64_t namespace_resolve(sqlite3 *db, const char *vtab_name,
                                  const char *scope_key, int key_len) {
    /* First, try to find an existing entry */
    char *sql = sqlite3_mprintf(
        "SELECT namespace_id FROM \"%w_namespace\" WHERE scope_key = ?1",
        vtab_name);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return -1;

    sqlite3_bind_text(stmt, 1, scope_key, key_len, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        int64_t ns_id = sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return ns_id;
    }
    sqlite3_finalize(stmt);

    /* Not found -- insert new entry */
    uint64_t hash = scope_key_hash(scope_key, key_len);
    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_namespace\"(scope_key, scope_hash) VALUES (?1, ?2)"
        " RETURNING namespace_id", vtab_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return -1;

    sqlite3_bind_text(stmt, 1, scope_key, key_len, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 2, (int64_t)hash);
    rc = sqlite3_step(stmt);
    int64_t ns_id = (rc == SQLITE_ROW) ? sqlite3_column_int64(stmt, 0) : -1;
    sqlite3_finalize(stmt);
    return ns_id;
}
```

### 8.3 Scope Key Decomposition

For queries that filter by individual scope columns (not the composite key):

```c
/*
 * Find all namespace_ids matching partial scope column values.
 * E.g., "all namespaces where project_id = 'proj-x'" regardless of session_id.
 */
static int namespace_find_matching(
    sqlite3 *db, const char *vtab_name,
    const char **col_names, const char **col_values, int col_count,
    int64_t **out_ids, int *out_count
);
```

---

## 9. SSSP Module: graph_sssp.c / graph_sssp.h

### 9.1 Types to Extract from graph_centrality.c

The following types and functions are currently `static` in `graph_centrality.c`
(lines 96-335). They must be moved to the new module:

```c
/* From graph_centrality.c lines 103-165 */
typedef struct { int node; double dist; } DPQEntry;
typedef struct { DPQEntry *entries; int size; int capacity; } DoublePQ;

/* From graph_centrality.c lines 177-205 */
typedef struct { int *items; int count; int capacity; } IntList;

/* From graph_centrality.c line 168 */
static int double_eq(double a, double b);
```

### 9.2 graph_sssp.h — Public API

```c
/*
 * graph_sssp.h -- Single-source shortest paths for graph algorithms
 *
 * Provides BFS (unweighted) and Dijkstra (weighted) SSSP with full
 * Brandes working-set output: dist[], sigma[], pred[], and BFS order
 * stack. Used by betweenness, closeness, and SSSP shadow table cache.
 */
#ifndef GRAPH_SSSP_H
#define GRAPH_SSSP_H

#include "graph_load.h"

/* -- Double-precision priority queue (Dijkstra) ----------- */

typedef struct {
    int node;
    double dist;
} SsspPQEntry;

typedef struct {
    SsspPQEntry *entries;
    int size;
    int capacity;
} SsspPQ;

void sssp_pq_init(SsspPQ *pq, int capacity);
void sssp_pq_destroy(SsspPQ *pq);
void sssp_pq_push(SsspPQ *pq, int node, double dist);
SsspPQEntry sssp_pq_pop(SsspPQ *pq);

/* -- Predecessor list (dynamic int array) ----------------- */

typedef struct {
    int *items;
    int count;
    int capacity;
} SsspPredList;

void sssp_pred_init(SsspPredList *l);
void sssp_pred_push(SsspPredList *l, int val);
void sssp_pred_clear(SsspPredList *l);
void sssp_pred_destroy(SsspPredList *l);

/* -- Epsilon comparison for tie detection ----------------- */

int sssp_double_eq(double a, double b);

/* -- SSSP working set ------------------------------------- */

/*
 * Caller-allocated working arrays for SSSP. Reusable across
 * multiple source invocations (betweenness/closeness loop).
 *
 * All arrays are of size node_count. Caller is responsible for
 * allocation and deallocation.
 */
typedef struct {
    int node_count;
    double *dist;           /* [V] shortest distance from source */
    double *sigma;          /* [V] shortest-path count (NULL ok for closeness-only) */
    SsspPredList *pred;     /* [V] predecessor lists (NULL ok for closeness-only) */
    int *stack;             /* [V] BFS/Dijkstra order (NULL ok for closeness-only) */
    int stack_size;         /* number of entries in stack[] after SSSP */
} SsspWorkingSet;

/*
 * Allocate a working set for V nodes.
 * If needs_brandes is true, allocates sigma[], pred[], and stack[].
 * If needs_brandes is false, only allocates dist[].
 * Returns 0 on success, -1 on allocation failure.
 */
int sssp_working_set_init(SsspWorkingSet *ws, int node_count, int needs_brandes);

/* Free all memory owned by the working set. */
void sssp_working_set_destroy(SsspWorkingSet *ws);

/* -- Core SSSP functions ---------------------------------- */

/*
 * BFS-based SSSP for unweighted graphs.
 * Sets ws->dist[v] = shortest distance from source.
 * If ws->sigma is non-NULL, sets shortest-path counts.
 * If ws->pred is non-NULL, fills predecessor lists.
 * If ws->stack is non-NULL, fills BFS order stack.
 * direction: "forward", "reverse", or "both" (NULL = "forward").
 */
void sssp_bfs(const GraphData *g, int source,
              SsspWorkingSet *ws, const char *direction);

/*
 * Dijkstra-based SSSP for weighted graphs.
 * Same interface and semantics as sssp_bfs.
 */
void sssp_dijkstra(const GraphData *g, int source,
                   SsspWorkingSet *ws, const char *direction);

/*
 * Run the appropriate SSSP variant based on g->has_weights.
 * Convenience wrapper that dispatches to sssp_bfs or sssp_dijkstra.
 */
void sssp_run(const GraphData *g, int source,
              SsspWorkingSet *ws, const char *direction);

#endif /* GRAPH_SSSP_H */
```

### 9.3 Key Design Decisions

1. **Renamed types.** `DoublePQ` becomes `SsspPQ`, `IntList` becomes
   `SsspPredList`. The `sssp_` prefix prevents collisions with the existing
   `PriorityQueue` in `priority_queue.h` (used by HNSW).

2. **SsspWorkingSet with optional fields.** Closeness only needs `dist[]`.
   By making `sigma`, `pred`, and `stack` optional (NULL-checked in the
   SSSP functions), closeness avoids allocating O(V) arrays it never reads.

3. **Ownership semantics.** The caller owns the `SsspWorkingSet` and can
   reuse it across the V iterations of the all-pairs loop.

---

## 10. Generation Counter Protocol

### 10.1 Generation Storage

The GII's `_config` shadow table stores generation counters:

```sql
-- Written by CSR rebuild:
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_adj', '42');

-- Written by SSSP rebuild:
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_sssp', '42');
```

The `generation_adj` counter increments on every full CSR rebuild. The
`generation_sssp` counter records which `generation_adj` the SSSP cache was
computed against.

### 10.2 Interaction Between Generation Counter and Delta Queue

The generation counter is the **primary** staleness mechanism. The `_sssp_delta`
queue is an **additional** optimization for partial recomputation:

- **Full CSR rebuild** (ratio >= 30%): `generation_adj` bumped. `_sssp_delta`
  queue ignored — generation mismatch signals full SSSP rebuild.
- **Selective/delta-flush** (ratio < 30%): `generation_adj` NOT bumped. Affected
  nodes written to `_sssp_delta`. Generation still matches, but delta queue
  signals partial staleness.

### 10.3 Staleness Check

```
On SSSP cache read:
    G_adj  = SELECT value FROM {name}_config WHERE key = 'generation_adj'
    G_sssp = SELECT value FROM {name}_config WHERE key = 'generation_sssp'

    if G_sssp != G_adj:
        cache is STALE (generation mismatch) -> full SSSP rebuild
    else:
        check _sssp_delta for this namespace
        if _sssp_delta is empty:
            cache is FRESH -> read from _sssp
        else:
            cache is PARTIALLY STALE -> see Section 10.4
```

### 10.4 SSSP Threshold-Based Rebuild Strategy

When betweenness or closeness is invoked and SSSP is enabled:

1. **Generation match?** If `gen_sssp != gen_adj` → full SSSP rebuild (all V sources).
2. **`_sssp_delta` count?**
   - `sssp_delta_ratio = |_sssp_delta for this namespace| / V`
   - ratio < θ_sssp_selective (default 0.10): Recompute ONLY the stale sources.
   - ratio >= θ_sssp_full (default 0.50): Full all-pairs SSSP rebuild.

### 10.5 Correctness Caveats: Betweenness vs Closeness

**Important caveat:** A single edge change can theoretically affect shortest paths
from ANY source. The `_sssp_delta` approach is an optimization that trades theoretical
correctness for practical performance.

**For betweenness: always recompute all sources when any delta exists.**
Betweenness centrality accumulates path-count contributions from ALL sources.
Partial recomputation is INCORRECT for betweenness — the delta queue tells us
SOMETHING changed, but betweenness needs all-pairs. If `_sssp_delta` is empty AND
generation matches → cache hit. Otherwise → full all-pairs SSSP rebuild.

**For closeness: can use partial recomputation.** Closeness of node v only depends
on v's own SSSP. Recomputing only stale sources and reading fresh sources from cache
is correct.

| TVF | Delta empty + gen match | Delta non-empty + gen match | Gen mismatch |
|-----|------------------------|----------------------------|--------------|
| Betweenness | Cache hit (0 work) | Full SSSP rebuild (all V) | Full SSSP rebuild (all V) |
| Closeness | Cache hit (0 work) | Selective rebuild (stale only) | Full SSSP rebuild (all V) |

### 10.6 Namespace Scoping

When namespace support is present, each namespace has independent generation tracking:

```sql
INSERT OR REPLACE INTO "{name}_config" (key, value)
    VALUES ('generation_sssp:' || namespace_id, generation_adj_value);
```

---

## 11. SSSP Integration with Betweenness TVF

### 11.1 Current Flow (bet_filter)

```
1. Parse argv -> GraphLoadConfig
2. Load graph -> GraphData (from GII or raw SQL)
3. Allocate: CB[V], dist[V], sigma[V], delta[V], stack[V], pred[V]
4. For each source s in {0..V-1} (or sampled subset):
   a. sssp_bfs/dijkstra(g, s, dist, sigma, pred, stack, &stack_size, direction)
   b. Backward accumulation: walk stack[] in reverse, accumulate delta via pred[]
   c. CB[v] += delta[v] for all v != s
5. Normalize CB[v] based on direction
6. Output (node_name, CB[v]) tuples
```

### 11.2 Modified Flow (with SSSP cache)

```
1. Parse argv -> GraphLoadConfig
2. Check for GII: is_gii(config, &vtab_name, &namespace_id)
3. If GII detected AND features & FEATURE_SSSP:
   a. sssp_cache_check(db, vtab_name, namespace_id, &gen_adj, &gen_sssp, &delta_count)
   b. If cache HIT (gen match + delta empty):
      i.  For each source s in {0..V-1}:
          - Read dist[s], sigma[s] from _sssp
          - Reconstruct pred[] from dist[] (Section 11.3)
          - Reconstruct stack[] from dist[] (Section 11.4)
          - Backward accumulation (same as current)
      ii. Total SSSP work: ZERO (all from cache)
   c. If STALE (gen mismatch or any delta):
      i.  Full all-pairs recompute (betweenness cannot use partial)
      ii. For each source s: run sssp_run(), cache result to _sssp
      iii. Clear _sssp_delta, update gen_sssp
      iv. Then proceed with backward accumulation from fresh cache
4. If NOT GII: fall through to current uncached path
```

### 11.3 Predecessor Reconstruction

Predecessors can be reconstructed from `dist[]` without any loss of correctness.
For each settled node `w` in the SSSP from source `s`, a node `v` is a predecessor
of `w` if and only if:

```
dist[v] + weight(v, w) == dist[w]   (for weighted graphs)
dist[v] + 1 == dist[w]              (for unweighted graphs)
```

```c
/*
 * Reconstruct pred[w] from cached dist[] for a given source.
 * For each node w (in BFS order), scan its in-neighbors and check
 * the shortest-path condition.
 */
static void reconstruct_predecessors(
    const GraphData *g, const double *dist,
    SsspPredList *pred, int V, const char *direction
) {
    /* Clear all predecessor lists */
    for (int w = 0; w < V; w++)
        sssp_pred_clear(&pred[w]);

    /* For each node w, check its in-neighbors */
    for (int w = 0; w < V; w++) {
        if (dist[w] < 0.0) continue;  /* unreachable */

        /* Get in-neighbors of w (reverse adjacency) */
        int n_in; const int32_t *in_nbrs; const double *in_wts;
        graph_neighbors(g, w, "reverse", &n_in, &in_nbrs, &in_wts);

        for (int j = 0; j < n_in; j++) {
            int v = in_nbrs[j];
            if (dist[v] < 0.0) continue;

            double edge_w = in_wts ? in_wts[j] : 1.0;
            if (sssp_double_eq(dist[v] + edge_w, dist[w])) {
                /* Deduplicate: check last added */
                if (pred[w].count == 0 || pred[w].items[pred[w].count - 1] != v)
                    sssp_pred_push(&pred[w], v);
            }
        }
    }
}
```

**Complexity:** O(E) per source (same as forward SSSP). With V sources, total
predecessor reconstruction cost is O(VE), same as computing SSSP from scratch.
The saving is that we avoid the Dijkstra priority queue operations.

### 11.4 Stack Reconstruction

The BFS/Dijkstra settlement stack is the reverse-topological order of the SSSP
DAG. It can be reconstructed from `dist[]` by sorting:

```c
/*
 * Reconstruct the BFS stack from cached dist[].
 * Returns nodes sorted by non-decreasing distance (BFS order).
 */
static int reconstruct_stack(const double *dist, int V,
                              int *stack_out) {
    int stack_size = 0;
    for (int i = 0; i < V; i++) {
        if (dist[i] >= 0.0)
            stack_out[stack_size++] = i;
    }

    /* Sort by non-decreasing dist (stable for equal distances) */
    /* Using insertion sort for simplicity; V <= 10000 */
    for (int i = 1; i < stack_size; i++) {
        int key = stack_out[i];
        double key_dist = dist[key];
        int j = i - 1;
        while (j >= 0 && dist[stack_out[j]] > key_dist) {
            stack_out[j + 1] = stack_out[j];
            j--;
        }
        stack_out[j + 1] = key;
    }
    return stack_size;
}
```

---

## 12. SSSP Integration with Closeness TVF

### 12.1 Current Flow (clo_filter)

```
1. Parse argv -> GraphLoadConfig
2. Load graph -> GraphData
3. Allocate: dist[V], sigma[V], pred[V], stack[V]  (sigma/pred/stack unused!)
4. For each source s in {0..V-1}:
   a. sssp_bfs/dijkstra(g, s, dist, sigma, pred, stack, &stack_size, direction)
   b. closeness[s] = compute from dist[] only
5. Output (node_name, closeness[s]) tuples
```

**Waste:** Closeness allocates and fills `sigma[V]`, `pred[V]`, and `stack[V]`
for every source, despite only needing `dist[]`. This is 32V bytes wasted per
iteration (8V sigma + 4V stack + ~20V pred pointers).

### 12.2 Modified Flow (with SSSP cache)

```
1. Parse argv -> GraphLoadConfig
2. Check for GII with SSSP feature
3. If GII with SSSP cache:
   a. sssp_cache_check(...)
   b. If FRESH: read dist[] from _sssp for each source, compute closeness
   c. If PARTIALLY STALE: read fresh rows, recompute stale rows, update _sssp
   d. If STALE: full recompute, write all to _sssp
4. If NOT GII: use sssp_run() with needs_brandes=0 (allocates only dist[])
   This alone saves 32V bytes per source even without caching.
```

---

## 13. SSSP Size Limits and Feasibility

### Storage Requirements

| V | dist matrix | sigma matrix | Total (both) | Total (dist only) | Feasible? |
|---:|------------:|-------------:|-------------:|------------------:|-----------|
| 100 | 80 KB | 80 KB | 160 KB | 80 KB | Trivial |
| 500 | 2 MB | 2 MB | 4 MB | 2 MB | Trivial |
| 1,000 | 8 MB | 8 MB | 16 MB | 8 MB | Yes |
| 2,000 | 32 MB | 32 MB | 64 MB | 32 MB | Yes |
| 5,000 | 200 MB | 200 MB | 400 MB | 200 MB | Marginal |
| 10,000 | 800 MB | 800 MB | 1.6 GB | 800 MB | Limit |
| 20,000 | 3.2 GB | 3.2 GB | 6.4 GB | 3.2 GB | Infeasible |

**Formula:** `V * V * sizeof(double) = V^2 * 8 bytes` per matrix.

### Configurable Threshold

```sql
CREATE VIRTUAL TABLE g USING gii(
    edge_table='edges', src_col='src', dst_col='dst',
    features='sssp',
    max_sssp_nodes=5000   -- override default 10000
);
```

**When V > max_sssp_nodes:**
- Shadow tables are created (schema fixed at VT creation)
- `_sssp` remains empty
- Shared SSSP module used for live computation (uncached mode)
- `_config` stores `sssp_skipped_reason = 'node_count_exceeds_threshold'`

**When V <= max_sssp_nodes:**
- SSSP cached on first betweenness or closeness query
- Subsequent queries read from cache if generation is fresh
- `INSERT INTO g(g) VALUES('rebuild_sssp')` forces explicit recomputation

### Disk vs Memory

The shadow table is on-disk (SQLite B-tree). SSSP results are read row-by-row
during TVF execution, not loaded into memory all at once. Peak memory during
betweenness with cache is ~52V bytes per source (same as current implementation).

---

## 14. xBestIndex/xFilter Changes

### 14.1 Current Pattern

The GII VT currently uses a single `xBestIndex` that returns all rows:

```c
/* Current: no namespace filtering */
idxNum = 0;  /* no constraints */
```

### 14.2 New Pattern

Add a hidden column `namespace` (type TEXT) that users can filter on:

```sql
-- Filter by namespace (scoped query)
SELECT * FROM g WHERE namespace = 'proj-x' || X'00' || 'session-a';

-- No namespace filter (returns all namespaces, backward compatible)
SELECT * FROM g;
```

The `xBestIndex` advertises the `namespace` constraint. When present, `xFilter`
reads only the matching namespace's shadow tables. When absent, it iterates all
namespaces.

### 14.3 Cost Estimation

```c
/* With namespace constraint: estimated output = rows for one namespace */
pIdxInfo->estimatedCost = 100.0;   /* single namespace */

/* Without namespace constraint: all namespaces */
pIdxInfo->estimatedCost = 10000.0; /* union of all namespaces */
```

---

## 15. Backward Compatibility

The following guarantees ensure existing users are unaffected:

| Feature | Current Behavior | New Behavior | Compatible? |
|---------|-----------------|--------------|-------------|
| `namespace_cols` omitted | Single implicit graph | Same (namespace_id=0 everywhere) | Yes |
| `features` omitted | No SSSP/component/community shadow tables | Same | Yes |
| Trigger SQL | `_delta(src, dst, weight, op)` | `_delta(scope_key, src, dst, weight, op)` with scope_key='' | Yes |
| `graph_data_load_from_gii()` | No namespace param | Optional namespace param (NULL = all) | Yes |
| `is_gii()` | Returns vtab_name | Returns vtab_name + namespace_id | Yes (extra output ignored by callers) |
| Shadow table names | `{name}_*` | Same | Yes |
| xBestIndex | No hidden columns | Hidden namespace column (ignored if not constrained) | Yes |

---

## 16. Implementation Steps

### Phase A: GII Core (Steps A1-A9)

**Step A1: Parse `namespace_cols` and `features` in xCreate/xConnect**

Parse the comma-separated `namespace_cols` parameter and validate each column name
with `id_validate()`. Parse `features` into a bitmask. Store both in `GiiVtab`.

**File:** `src/gii.c`

**Step A2: Create `{name}_namespace` Shadow Table**

Add the namespace registry table in `gii_create_shadow_tables()`.

**File:** `src/gii.c`

**Step A3: Add `namespace_id` to Shadow Tables**

Modify all existing shadow table CREATE statements to include `namespace_id` in their
primary keys. Default to 0 for backward compatibility.

**File:** `src/gii.c`

**Step A4: Modify Triggers to Capture Scope Column Values**

Update `gii_create_triggers()` to generate scope_key expressions from namespace_cols.

**File:** `src/gii.c`

**Step A5: Modify Full Rebuild to Iterate Per-Namespace**

Update `gii_rebuild()` to group deltas by scope_key, resolve namespace_ids, and
rebuild CSR independently per namespace.

**File:** `src/gii.c`

**Step A6: Modify Incremental Rebuild to Handle Per-Namespace Blocks**

Update selective block rebuild and delta flush to operate within a single namespace.

**File:** `src/gii.c`

**Step A7: Add Namespace Filter to xBestIndex/xFilter**

Add the hidden `namespace` column and implement constraint handling.

**File:** `src/gii.c`

**Step A8: Update `graph_data_load_from_gii()`**

Add optional `namespace_id` parameter. When provided, filter shadow table reads
to the specified namespace.

**File:** `src/graph_load.c`

**Step A9: Update `is_gii()` and All TVFs That Call It**

Extend `is_gii()` to optionally return the detected `namespace_id`. Update
betweenness, closeness, Leiden, and degree TVFs.

**Files:** `src/graph_load.c`, `src/graph_centrality.c`, `src/graph_community.c`

### Phase B: SSSP Module + Shadow Tables (Steps B1-B10)

**Step B1: Create graph_sssp.h**

Create `src/graph_sssp.h` with the public API defined in Section 9.

**Step B2: Create graph_sssp.c**

Move SSSP types and functions from `graph_centrality.c` to the new module. Rename
types with `sssp_` prefix. Implement `SsspWorkingSet` with optional fields.

**File:** `src/graph_sssp.c`

**Step B3: Update graph_centrality.c**

Remove moved static functions. Add `#include "graph_sssp.h"`. Update `bet_filter` and
`clo_filter` to use the new API.

**File:** `src/graph_centrality.c`

**Step B4: Update build configuration**

Add `graph_sssp.c` to the source file list and `graph_sssp.h` to headers.

**File:** `scripts/generate_build.py`

**Step B5: Add _sssp and _sssp_delta shadow table creation**

Conditionally create SSSP shadow tables when `features` includes `'sssp'`.

**File:** `src/gii.c`

**Step B6: Add SSSP delta emission from CSR rebuild**

Implement `sssp_delta_emit()` called after CSR rebuild completes.

**File:** `src/gii.c`

**Step B7: Add SSSP cache population**

Implement `sssp_cache_populate()` for writing all-pairs SSSP to `_sssp`.

**File:** `src/gii.c`

**Step B8: Add SSSP cache read in bet_filter and clo_filter**

Implement `sssp_cache_check()`, `sssp_cache_read_row()`, `sssp_delta_read()`.

**Files:** `src/graph_centrality.c`, `src/gii.c`

**Step B9: Add generation check logic**

Implement the two-tier staleness protocol (generation counter + delta queue).

**File:** `src/gii.c`

**Step B10: Add namespace_id scoping to SSSP**

Key all SSSP operations by `namespace_id`.

**Files:** `src/gii.c`, `src/graph_centrality.c`

### Phase C: Session-Log Demo Builder (Steps C1-C5)

**Step C1: Create SessionLogKG phase classes**

New Phase subclasses in `benchmarks/demo_builder/phases/`:

- `PhaseSessionChunks` — Parse JSONL events, create chunks from user/assistant messages
- `PhaseSessionNER` — Rule-based entity extraction (tools, files, models, sessions)
- `PhaseSessionRE` — Structural relation extraction (parent-child, tool usage, file ops)

These replace the GLiNER/GLiREL phases for the session-log variant (no ML models needed).

**Files:** `benchmarks/demo_builder/phases/session_chunks.py`,
`benchmarks/demo_builder/phases/session_ner.py`,
`benchmarks/demo_builder/phases/session_re.py`

**Step C2: Create SessionLogBuild orchestrator**

A `DemoBuild` subclass (or variant) that uses session-log phases instead of Gutenberg
phases. Discovers JSONL files from `~/.claude/projects/`.

**File:** `benchmarks/demo_builder/session_build.py`

**Step C3: Reuse shared phases**

Entity embeddings, UMAP, entity resolution, Node2Vec, and metadata phases are
reused verbatim from the existing demo_builder. The only difference is the source
data pipeline (steps C1-C2).

**Step C4: Add GII post-build step**

After the demo_builder produces the database, create the GII virtual table with
namespace scoping on the edges table.

**Step C5: Integration test**

End-to-end test: ingest a sample session log → produce DB → load in viz → verify
all 7 pipeline stages display correctly.

**File:** `benchmarks/demo_builder/tests/test_session_build.py`

---

## 17. Verification Steps

### GII Tests (Phase A)

**Test A1: Scoped VT Creation and Basic Operation**

```python
def test_scoped_gii_creation():
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL, project_id TEXT, session_id TEXT)")
    conn.execute("""CREATE VIRTUAL TABLE g USING gii(
        edge_table='edges', src_col='src', dst_col='dst', weight_col='weight',
        namespace_cols='project_id,session_id', features='sssp')""")

    # Insert edges into two different scopes
    conn.execute("INSERT INTO edges VALUES ('A','B',1.0,'p1','s1')")
    conn.execute("INSERT INTO edges VALUES ('C','D',1.0,'p1','s2')")

    # Verify namespace registry
    ns = conn.execute("SELECT * FROM g_namespace ORDER BY namespace_id").fetchall()
    assert len(ns) == 2
    assert ns[0][1] == 'p1\x00s1'  # scope_key
    assert ns[1][1] == 'p1\x00s2'
```

**Test A2: Independent CSR Blocks Per Namespace**

```python
def test_namespace_isolation():
    # Insert 100 edges into scope s1, 1 edge into scope s2
    # Rebuild should only affect s1's CSR blocks
    # Verify s2's _csr_fwd BLOBs are unchanged
```

**Test A3: Scoped Incremental Rebuild**

```python
def test_scoped_incremental():
    # Build initial CSR for scope s1 (1000 edges)
    # Insert 10 more edges into scope s1
    # Verify selective block rebuild (ratio < 5%)
    # Verify _sssp_delta populated for affected nodes
```

**Test A4: TVF with Namespace Filter**

```python
def test_tvf_namespace_filter():
    # graph_betweenness should accept gii_table + namespace
    # Results should only reflect edges in that namespace
```

**Test A5: Backward Compatibility (No namespace_cols)**

```python
def test_backward_compat():
    # Create GII without namespace_cols
    # Verify single namespace_id=0
    # Verify all existing TVF queries work unchanged
```

**Test A6: Cross-Namespace Protection**

```python
def test_cross_namespace_protection():
    # Insert into scope A, delete from scope B
    # Verify scope A's CSR is unaffected by scope B's changes
```

**Test A7: Scope Key Collision Resistance**

```python
def test_scope_key_collision():
    # Create scopes with DJB2-collision-prone keys
    # Verify distinct namespace_ids assigned
```

**Test A8: Threshold-Based Rebuild Strategy Selection**

```python
def test_threshold_rebuild():
    # Insert edges to trigger each threshold band
    # Verify selective (< 5%), delta-flush (5-30%), full (>= 30%)
```

### SSSP Tests (Phase B)

**Test B1: SSSP Module Extraction**

```c
/* C unit test: verify sssp_bfs and sssp_dijkstra produce same results as
   the original static functions in graph_centrality.c */
void test_sssp_bfs_matches_original(void);
void test_sssp_dijkstra_matches_original(void);
void test_sssp_working_set_closeness_only(void);
```

**Test B2: SSSP Cache Population**

```python
def test_sssp_cache_populated():
    # Create GII with features='sssp'
    # Query graph_betweenness -> triggers SSSP cache
    # Verify _sssp table has V rows for the namespace
```

**Test B3: SSSP Cache Hit**

```python
def test_sssp_cache_hit():
    # Query betweenness twice without edge changes
    # Second query should be significantly faster (cache hit)
```

**Test B4: SSSP Delta Queue**

```python
def test_sssp_delta_queue():
    # Insert 5 edges (< 5% ratio)
    # Verify _sssp_delta has entries for affected nodes
    # Query closeness -> selective rebuild
    # Verify _sssp_delta cleared after query
```

**Test B5: Generation Mismatch Forces Full Rebuild**

```python
def test_generation_mismatch():
    # Trigger full CSR rebuild (>= 30% edges changed)
    # Verify generation_adj incremented
    # Query betweenness -> full SSSP rebuild (not selective)
```

**Test B6: Betweenness Always Full, Closeness Selective**

```python
def test_betweenness_vs_closeness_strategy():
    # Insert 5 edges -> _sssp_delta has entries
    # Query betweenness -> full all-pairs (betweenness can't use partial)
    # Insert 5 more edges -> _sssp_delta has entries again
    # Query closeness -> selective (only stale sources recomputed)
```

### Session-Log Demo Builder Tests (Phase C)

**Test C1: JSONL Parsing**

```python
def test_session_chunks_parsing():
    # Create a sample JSONL file with known events
    # Run PhaseSessionChunks
    # Verify chunks table has expected rows
```

**Test C2: Rule-Based Entity Extraction**

```python
def test_session_ner():
    # Create chunks with tool_use events
    # Run PhaseSessionNER
    # Verify entities table has tool, file, model entities
```

**Test C3: Structural Relation Extraction**

```python
def test_session_re():
    # Create chunks with parent-child events
    # Run PhaseSessionRE
    # Verify relations table has uses_tool, reads_file, etc.
```

**Test C4: End-to-End DB Production**

```python
def test_session_log_e2e():
    # Ingest a sample session log (included in test fixtures)
    # Verify all 10 user tables populated
    # Verify 3 HNSW VTs and 1 FTS5 table functional
```

**Test C5: GII Virtual Table on Session DB**

```python
def test_gii_on_session_db():
    # Create GII with namespace_cols on session-log DB
    # Verify per-session namespace isolation
    # Query betweenness -> SSSP cache populated
    # Verify bridge files identified by centrality
```

**Test C6: Viz Compatibility**

```python
def test_viz_auto_discovery():
    # Load session-log DB via viz KG service
    # get_pipeline_summary() returns 7 stages all available
    # run_graphrag_query("which tools") returns meaningful results
```

---

## 18. References

### SQLite Virtual Tables

- [Virtual Table Methods](https://www.sqlite.org/vtab.html)
- [Shadow Tables](https://www.sqlite.org/vtab.html#shadow_tables)
- [xBestIndex](https://www.sqlite.org/vtab.html#xbestindex)
- [FTS5](https://www.sqlite.org/fts5.html) — nearest analogy for VT with shadow tables

### Graph Storage Approaches

- CSR (Compressed Sparse Row) — standard for static graph analytics
- Blocked CSR — enables incremental updates without full rebuild
- Delta merge — borrowed from LSM-tree literature

### Namespace/Projection Prior Art

- DuckDB PGQ: `MATCH` clauses scope to `PROPERTY GRAPH` definitions (compile-time, not runtime)
- Neo4j GDS: `gds.graph.project()` creates named in-memory projections
- Apache AGE: Schema-level graph names (`SET graph_path = 'my_graph'`)
- None of these support row-level namespace scoping via scope columns on the edge table

### SSSP Algorithms

- Brandes (2001) — "A Faster Algorithm for Betweenness Centrality" — O(VE) unweighted
- Dijkstra (1959) — O(E + V log V) with Fibonacci heap, O(E log V) with binary heap
- BFS — O(V + E) for unweighted SSSP

### Session-Log KG Prior Art

- Claude Code session logs are append-only JSONL files (never deleted by compaction)
- The introspect skill at `.claude/skills/introspect/` provides the parsing infrastructure
- The existing demo_builder at `benchmarks/demo_builder/` provides the KG pipeline framework

---

**Prev:** [Gap Analysis](./00_gap_analysis.md) | **Next:** [Phase 3 — Components Shadow Table](./03_components_shadow_tables.md)
