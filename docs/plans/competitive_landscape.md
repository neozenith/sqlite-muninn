# Competitive Landscape Analysis

An analysis of whether the vec_graph / sqlite-muninn project occupies a unique niche, and who the closest competitors are.

**Status:** Research completed 2026-02-11. Should be refreshed periodically.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [SQLite Vector Extensions](#sqlite-vector-extensions)
3. [SQLite Graph Extensions](#sqlite-graph-extensions)
4. [GraphRAG Implementations](#graphrag-implementations)
5. [AI Agent Memory Systems](#ai-agent-memory-systems)
6. [Non-SQLite Graph+Vector Databases](#non-sqlite-graphvector-databases)
7. [Hybrid Graph+Vector+SQLite Projects](#hybrid-graphvectorsqlite-projects)
8. [Comparison Matrix](#comparison-matrix)
9. [Unique Positioning](#unique-positioning)
10. [Strategic Risks](#strategic-risks)

---

## Executive Summary

**No other project combines HNSW vector search, graph traversal TVFs, and Node2Vec embedding generation in a single zero-dependency C11 SQLite extension.** The landscape is fragmented across three axes:

- **Vector-only SQLite extensions:** sqlite-vec, sqlite-vector, vectorlite
- **Graph-only SQLite extensions:** GraphQLite, sqlite-graph, simple-graph
- **Full vector+graph solutions:** Neo4j (Java server), Kuzu (archived), Zep/Mem0 (Python+Docker)

The closest embedded competitor (Kuzu) was abandoned in October 2025, leaving the niche vacant. The AI agent memory ecosystem (Mem0, Zep/Graphiti, Cognee) has validated that vector+graph is the right architecture for agent memory, but all require multi-service infrastructure.

---

## SQLite Vector Extensions

### sqlite-vec (Alex Garcia)

- **GitHub:** [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) | ~6.6k stars
- **What:** Pure C, zero-dependency vector search. KNN with L2, cosine, Hamming. Float32, int8, binary. SIMD-accelerated.
- **Limitations:** Brute-force scan only (no HNSW index, though it's on the roadmap). No graph capabilities.
- **Relevance:** The gold standard for SQLite vector extensions. Mozilla-backed, well-funded (Fly.io, Turso, SQLite Cloud). Our vector-side benchmark baseline.

### sqlite-vector (SQLiteAI)

- **GitHub:** [sqliteai/sqlite-vector](https://github.com/sqliteai/sqlite-vector)
- **What:** Cross-platform vector search with quantization (Float32/16, BFloat16, Int8, UInt8, 1Bit). Scan-based. 30MB default memory.
- **Limitations:** No HNSW, no graph. Different philosophy — emphasizes quantized brute-force.
- **Relevance:** Part of the SQLiteAI suite (sqlite-ai, sqlite-rag). The extension we benchmark against in `benchmarks/`.

### vectorlite

- **GitHub:** [1yefuwang1/vectorlite](https://github.com/1yefuwang1/vectorlite)
- **What:** SQLite extension wrapping hnswlib for HNSW-based ANN search. Google Highway for SIMD. Claims 8x-100x faster than sqlite-vec.
- **Limitations:** C++ with external dependencies (hnswlib, Highway). Not zero-dependency.
- **Relevance:** Closest to our HNSW implementation, but has dependencies and no graph features.

### sqlite-vss (Deprecated)

- **GitHub:** [asg017/sqlite-vss](https://github.com/asg017/sqlite-vss) | ~1.6k stars
- **What:** SQLite extension wrapping Facebook Faiss. Predecessor to sqlite-vec.
- **Status:** Deprecated.

---

## SQLite Graph Extensions

### GraphQLite (colliery-io)

- **GitHub:** [colliery-io/graphqlite](https://github.com/colliery-io/graphqlite)
- **What:** Cypher query language + 15 graph algorithms (PageRank, Dijkstra, BFS/DFS, Louvain community detection, connected components).
- **Tech:** Rust with Python bindings.
- **Limitations:** No vector search, no HNSW, no Node2Vec. Rust dependency. Alpha-stage.
- **Relevance:** The closest graph-side competitor. Implements many of the same graph algorithms. Recently posted to Hacker News.

### sqlite-graph (AgentFlare)

- **GitHub:** [agentflare-ai/sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) | v0.1.0-alpha
- **What:** C99 zero-dependency extension with Cypher query parser (lexer, parser, planner, Volcano-model executor).
- **Limitations:** Very early alpha. No vector search, no graph algorithms like PageRank.
- **Relevance:** Similar zero-dependency C philosophy. Focused on Cypher rather than TVF-based traversal.

### simple-graph

- **GitHub:** [dpapathanasiou/simple-graph](https://github.com/dpapathanasiou/simple-graph) | ~1.4k stars
- **What:** Graph database in SQLite using JSON documents as nodes. Traversal via recursive CTEs in pure SQL.
- **Limitations:** Pure SQL approach — no compiled extension, no algorithms beyond basic path traversal.
- **Relevance:** Proves the market for graph-in-SQLite. We outperform significantly on algorithmic capabilities.

---

## GraphRAG Implementations

### Microsoft GraphRAG

- **GitHub:** [microsoft/graphrag](https://github.com/microsoft/graphrag)
- **What:** The original GraphRAG paper implementation. Community hierarchy via Leiden algorithm, global/local search modes.
- **Tech:** Python, requires Neo4j for graph + Qdrant/similar for vectors + LLM API.
- **Relevance:** The gold standard for GraphRAG methodology. Our knowledge graph benchmark (see `docs/plans/knowledge_graph_benchmark.md`) tests the primitives this pattern needs.

### LightRAG / nano-graphrag

- **What:** Lightweight Python GraphRAG alternatives. Use NetworkX for graph, pluggable vector stores.
- **Relevance:** No SQLite integration. Application-layer orchestration.

### Stephen Collins' SQLite GraphRAG Tutorial

- **What:** Blog tutorial using plain SQLite tables + OpenAI for basic GraphRAG.
- **Relevance:** Demonstrates the exact use case we're built for, implemented entirely in Python application code without native extensions. Validates the market need.

---

## AI Agent Memory Systems

### Mem0

- **GitHub:** [mem0ai/mem0](https://github.com/mem0ai/mem0) | ~46.8k stars
- **What:** "Universal memory layer for AI Agents." Hybrid vector + graph storage. Five pillars: LLM fact extraction, vector storage (Qdrant), graph storage (Neo4j), history logging (SQLite), memory management.
- **Key insight:** Uses SQLite only for audit logging, not for vector or graph search. Requires Qdrant + Neo4j.
- **Relevance:** Validates vector+graph as the right memory architecture. We could replace both their vector and graph backends in a lightweight deployment.

### Zep / Graphiti

- **GitHub:** [getzep/graphiti](https://github.com/getzep/graphiti) | ~22k stars
- **What:** Temporally-aware knowledge graph for agent memory. Hybrid retrieval: semantic embeddings + BM25 + graph traversal. Bi-temporal data model.
- **Tech:** Python, requires Neo4j (primary). Research paper: arXiv:2501.13956.
- **Relevance:** **Architecturally closest to what we enable at the application layer.** Combines vector + graph for memory retrieval. But requires Neo4j + Python server. We could be the embedded alternative.

### Cognee

- **GitHub:** [topoteretes/cognee](https://github.com/topoteretes/cognee) | ~8-11k stars
- **What:** "Memory for AI Agents in 6 lines of code." Poly-store architecture: multiple graph DBs (Neo4j, Kuzu, FalkorDB, NetworkX) and vector DBs (Qdrant, Weaviate, etc.).
- **Relevance:** Uses SQLite only for relational metadata. Multi-service dependency.

### Letta (formerly MemGPT)

- **GitHub:** [letta-ai/letta](https://github.com/letta-ai/letta) | ~15.2k stars
- **What:** Stateful agents with self-editing memory. Four tiers: core, message, archival (vector DB), recall.
- **Relevance:** No native graph traversal. No SQLite-native vector search. Requires server infrastructure. Community has requested knowledge graph integration but it's not shipped.

### CrewAI / LangChain Memory

- **What:** CrewAI uses ChromaDB for vectors + SQLite for task result storage. LangChain has SqliteSaver for state, community sqlite-vec integration is basic.
- **Relevance:** SQLite used for simple persistence, not as the vector/graph engine.

---

## Non-SQLite Graph+Vector Databases

### Neo4j

- **What:** Leading graph database. Native HNSW vector indexes. Cypher queries. Graph Data Science library with PageRank, community detection, Node2Vec.
- **Relevance:** **Neo4j is the full-featured server-based version of what we aim to do in SQLite.** Our positioning: "Neo4j-class capabilities in a single SQLite extension file."

### Kuzu (Archived October 2025)

- **GitHub:** [kuzudb/kuzu](https://github.com/kuzudb/kuzu) | ~3.6k stars (archived)
- **What:** Embedded property graph database with Cypher, HNSW vector search, full-text search. C++ core.
- **Status:** **Abandoned.** Forks (Bighorn, Ladybug) attempting continuation with uncertain futures.
- **Relevance:** Was the closest embedded competitor. Its abandonment leaves the embedded vector+graph niche vacant. We fill this gap.

### DuckDB + DuckPGQ

- **What:** DuckDB community extension adds SQL/PGQ graph syntax (SQL:2023 standard).
- **Relevance:** OLAP-focused, not SQLite. Different use case.

### Weaviate / Milvus

- **What:** Vector databases. Weaviate has cross-references (graph-like). Milvus is vector-only.
- **Relevance:** Server-based, not embeddable, no true graph algorithms.

---

## Hybrid Graph+Vector+SQLite Projects

### LiteGraph

- **GitHub:** [litegraphdb/litegraph](https://github.com/jchristn/LiteGraph)
- **What:** Property graph database with relational, vector (cosine similarity), and MCP support. **Built on SQLite.** Web UI, multi-tenancy.
- **Tech:** C# / .NET
- **Limitations:** Not a native C extension (requires .NET runtime). Vector search is basic cosine similarity, not HNSW. No Node2Vec. No graph algorithms like PageRank.
- **Relevance:** Most architecturally similar project found. Uses SQLite, combines graph + vector, has MCP support. But .NET dependency makes it unsuitable for embedded/edge use cases.

### Vesper-Memory

- **GitHub:** [fitz2882/vesper-memory](https://github.com/fitz2882/vesper-memory)
- **What:** AI agent memory with semantic search + knowledge graphs + multi-hop reasoning via Personalized PageRank.
- **Tech:** Python, Docker-based.
- **Relevance:** Similar concept (PageRank for retrieval ranking) but is a Python service, not a database extension.

---

## Comparison Matrix

| Project | HNSW | Graph Algos | Node2Vec | SQLite Ext | Zero-Dep C | Single File |
|---------|:----:|:-----------:|:--------:|:----------:|:----------:|:-----------:|
| **sqlite-muninn (this)** | Yes | BFS/DFS/SP/PR/CC | Yes | Yes | Yes | Yes |
| sqlite-vec | No | No | No | Yes | Yes | Yes |
| sqlite-vector | No | No | No | Yes | Yes | Yes |
| vectorlite | Yes | No | No | Yes | No (C++) | No |
| GraphQLite | No | Yes (Cypher) | No | Yes | No (Rust) | No |
| sqlite-graph | No | Cypher parser | No | Yes | Yes (C99) | Yes |
| simple-graph | No | CTEs only | No | No (SQL) | N/A | N/A |
| LiteGraph | Cosine | Basic | No | No (.NET) | No | No |
| Neo4j | Yes | Yes + GDS | Yes | No (Java) | No | No |
| Kuzu (archived) | Yes | Yes | No | No | No (C++) | No |

---

## Unique Positioning

### What Makes This Project Novel

The specific combination that no other project replicates:

1. **HNSW vector index** — not brute-force scan (unlike sqlite-vec, sqlite-vector)
2. **Graph traversal TVFs** — BFS, DFS, shortest path, connected components, PageRank (unlike all vector-only extensions)
3. **Node2Vec embedding generation** — bridges graph topology to vector space (requires both subsystems; impossible by combining two separate extensions)
4. **Zero external dependencies** — pure C11 (unlike vectorlite/C++, GraphQLite/Rust, LiteGraph/.NET)
5. **Single loadable SQLite extension** — one `.so`/`.dylib` file, `.load ./muninn`

### The Value Proposition

> **Neo4j-class capabilities (HNSW + graph algorithms + Node2Vec) inside SQLite with zero dependencies and a single file load.**

This makes it uniquely suited for:

- **Edge/mobile GraphRAG** — no server infrastructure needed
- **AI agent local-first memory** — semantic + relational retrieval in one embedded store
- **Rapid GraphRAG prototyping** — without Neo4j/Qdrant/Docker setup
- **Embedded applications** — where adding Java/Python/.NET runtimes is not an option

### The "Embedded Zep" Pitch

The agent memory ecosystem (Mem0 at 46.8k stars, Zep at 22k) has validated that **vector + graph is the correct architecture for AI memory**. But they all require:

- Neo4j or FalkorDB (graph server)
- Qdrant or Weaviate (vector server)
- Python orchestration layer
- Docker for deployment

sqlite-muninn provides the same architectural pattern — vector entry point → graph expansion → centrality ranking — in a single file that works anywhere SQLite does. It's the embedded, serverless, zero-infrastructure alternative.

---

## Strategic Risks

### Risk 1: Convergence of Existing Extensions

**Threat:** sqlite-vec adds HNSW (on their roadmap) + GraphQLite matures → users combine two extensions.

**Mitigation:** Node2Vec inherently requires both subsystems working in concert. The unified design enables features (like using graph structure to improve vector search, or vector similarity to seed graph traversal) that are architecturally impossible when bolting two separate extensions together. The GraphRAG workflow (VSS → graph expansion → centrality ranking → context assembly) benefits from a single-extension implementation with shared memory.

### Risk 2: Agent Memory Frameworks Add SQLite Backends

**Threat:** Mem0, Zep, or Cognee add a "SQLite mode" using sqlite-vec or similar.

**Mitigation:** Application-layer SQLite integration will always be slower than native C extension integration. Also, these frameworks are Python-only; sqlite-muninn works from any language that can load SQLite extensions (Python, Node.js, Go, Rust, C, C++, Ruby, etc.).

### Risk 3: Neo4j Lite / Embedded Neo4j

**Threat:** Neo4j releases an embedded/serverless mode competing in the same niche.

**Mitigation:** Neo4j is Java-based with a large runtime. SQLite's ubiquity (billions of deployed instances) and zero-configuration nature are structural advantages that a Java-based solution cannot match.

### Risk 4: New Entrants in the Kuzu Vacuum

**Threat:** Kuzu's abandonment creates opportunity for new embedded graph+vector databases.

**Mitigation:** Building on SQLite (the most deployed database engine in history) rather than a custom storage engine reduces adoption friction dramatically. SQLite compatibility is a moat.
