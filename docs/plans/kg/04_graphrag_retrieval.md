# Phase 4: GraphRAG Retrieval Quality

## Goal

Implement retrieval quality benchmarks comparing VSS-only vs VSS+graph expansion.

## Implementation

1. Load KG from demo DB (chunks, chunks_vec HNSW, nodes, edges, FTS)
2. Generate test queries — sample N chunks as pseudo-queries (self-retrieval baseline)
3. VSS entry — embed query, search `chunks_vec` for top-K nearest
4. BM25 entry — search `chunks_fts` for top-K
5. Graph expansion — for each seed chunk, find linked entities, then other chunks containing those entities (BFS depth 1 or 2)
6. Measure passage recall against gold chunks (same entity cluster)

## Expansion Strategies

| Config | Entry | Expansion |
|--------|-------|-----------|
| `vss_none` | HNSW cosine | None |
| `vss_bfs1` | HNSW cosine | BFS depth 1 |
| `vss_bfs2` | HNSW cosine | BFS depth 2 |
| `bm25_none` | FTS5 | None |
| `bm25_bfs1` | FTS5 | BFS depth 1 |
| `bm25_bfs2` | FTS5 | BFS depth 2 |

Requires: `requires_muninn = True` (HNSW + graph_bfs)

## Files

- **MODIFY**: `benchmarks/harness/treatments/kg_graphrag.py`
