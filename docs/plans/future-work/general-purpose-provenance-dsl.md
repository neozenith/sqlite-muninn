# Future Work — General-purpose provenance DSL

**Status:** Deferred (Option C of `docs/plans/adv-centrality-filtering.md` G1 schema-flexibility ADR)
**Predecessor plan:** `docs/plans/adv-centrality-filtering.md` ships Option B — parameterized xCreate args with a fixed source-table cascade shape (chunked-text → entities → resolved-entities)

## Context

The provenance subsystem proposed in `adv-centrality-filtering.md` (gap G1) ships with a parameterized but **shape-fixed** contract: four named source tables, denormalized `(project_id, timestamp)`-style filter columns, and a cluster-mapping step. Consumers that fit this shape (claude-code-sessions, chat logs, RAG over documents) can use the subsystem by varying source-table and filter-column names at `USING provenance(...)` time.

Consumers whose join graph **does not** fit this shape — e.g., a 6-table cascade with two parallel ER paths, or a corpus with no entity-resolution step — currently have no path forward inside the subsystem. They would either:
- Build a separate VT subsystem (forking ~400 LoC of trigger machinery), or
- Maintain their own provenance table by hand without the GII generation-counter cascade

This document captures the deferred design that would close that gap.

## What this would look like

A JSON-described join-graph DSL passed at xCreate:

```sql
CREATE VIRTUAL TABLE prov USING provenance(
  source_tables='[
    {"name": "events", "keys": ["id"], "propagate_cols": ["project_id", "timestamp"]},
    {"name": "event_message_chunks", "fk": {"events": "event_id"}, "keys": ["chunk_id"]},
    {"name": "entities", "fk": {"event_message_chunks": "chunk_id"}, "keys": ["chunk_id", "name"]},
    {"name": "entity_clusters", "join_on": {"entities.name": "name"}, "denormalize": "canonical"}
  ]',
  output_table='_gii_provenance',
  output_cols='canonical, project_id, timestamp'
);
```

The DSL describes:
- An ordered list of source tables and their primary keys
- Foreign-key relationships between them
- Which columns to propagate (denormalize) into the output table
- Which terminal table provides the canonical/resolved value

Implementation requires:
- A recursive descent parser for the JSON join-graph
- Per-source-table trigger generation that walks the graph from the inserted row to the output rows
- Identifier validation for an unbounded set of names
- Test fixtures covering 2-table, 3-table, 4-table, and N-table cascades with various FK shapes

## Why deferred

- **No second consumer asking for it yet.** Option B's contract covers the immediate three workload classes (sessions, chat, RAG). Premature generalization risks shipping a DSL nobody uses.
- **DSL parser is a new sub-project.** Estimated 3-5× the implementation effort of Option B for unclear payoff at current scope.
- **Identifier validation surface explodes.** Recursive validation of arbitrary join graphs is significantly harder to keep injection-safe than a bounded set of `≤8` named tables.
- **Maintenance forever.** A DSL is a public API; once shipped, it cannot be removed without breaking consumers.

## Trigger conditions to revisit

Land Option C (or a subset of it) when **any** of these become true:

1. A specific consumer has a corpus that does not fit Option B's cascade shape and is willing to commit to the work in writing
2. Three or more consumers ask for non-4-table cascades within a 6-month window
3. The autoresearch loop (G4) discovers that the cascade shape itself is a productive optimization axis (i.e., proposing strategies that vary the source-table count and benefit measurably)
4. A separate use case (e.g., audit-log lineage tracking) wants the same trigger machinery without the ER cluster-mapping step

## Predecessors and related work

- `docs/plans/adv-centrality-filtering.md` — Option B implementation, shipping
- `docs/plans/graph/01_gii_sssp_session_kg.md` — GII delta-cascade philosophy that any DSL implementation must inherit
- Wikipedia: Patrick Valduriez, "Join indices", *ACM TODS* 12(2):218-246, 1987 — the algebraic foundation
- DuckPGQ (https://github.com/cwida/duckpgq-extension) — example of a SQL/PGQ-style join-graph DSL inside an analytical engine; reference for parser shape if this lands
