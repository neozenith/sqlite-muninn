# Project Rename: sqlite-vector-graph → sqlite-muninn

Rename the project from `sqlite-vector-graph` / `vec_graph` to `sqlite-muninn` / `muninn`, establishing a memorable brand identity with a cohesive naming architecture across all ecosystems.

**Status:** Plan only. Not yet implemented.

---

## Table of Contents

1. [Decision](#decision)
2. [Why Muninn](#why-muninn)
3. [Name Availability](#name-availability)
4. [Naming Architecture](#naming-architecture)
5. [SEO Strategy](#seo-strategy)
6. [Brand Identity](#brand-identity)
7. [Rename Checklist](#rename-checklist)
8. [Risks and Mitigations](#risks-and-mitigations)

---

## Decision

**Old name:** `sqlite-vector-graph` (repo), `vec_graph` (C extension, Python import)

**New name:** `sqlite-muninn` (repo/packages), `muninn` (C extension, brand)

**Rationale:** The old name describes the implementation mechanism (vector + graph). The new name describes the *purpose* — a memory system for knowledge retrieval. As the project grows beyond raw HNSW/graph primitives into GraphRAG, knowledge indexing, community detection, and agent memory, the old name becomes increasingly misleading.

---

## Why Muninn

**Muninn** (Old Norse: *Muninn*, "memory") is one of Odin's two ravens in Norse mythology. Every day, Huginn ("thought") and Muninn ("memory") fly across all the realms gathering information, then return to perch on Odin's shoulders and whisper everything they've learned.

From the Poetic Edda (Grimnismal, stanza 20):

> *Huginn and Muninn fly each day over the wide world.*
> *I fear for Huginn that he may not return,*
> *yet I worry more for Muninn.*

Odin fears losing Memory more than Thought.

### The Metaphor

| Norse Myth | This Library |
|-----------|-------------|
| Muninn flies out across the realms | Indexer crawls codebases, docs, session logs, infrastructure |
| Muninn observes and encodes what it sees | Vector embeddings capture semantic meaning |
| Muninn traces connections between realms | Graph edges encode relationships |
| Muninn returns knowledge to Odin | Graph traversal retrieves connected context |
| Without Muninn, Odin loses his power | Without memory, an AI agent is stateless |

### Shortlist Considered

The final four contenders before selecting Muninn:

| Name | Strength | Why Not Chosen |
|------|----------|---------------|
| **grimoire** | Strongest "deep dark lore" imagery | No layered wordplay; slightly generic |
| **tome** | Short, evocative, scholarly weight | Too generic; no story behind it |
| **lore** | Most self-describing ("it manages lore") | Likely taken on PyPI; no distinctive brand |
| **engram** | Best pun (engram/n-gram) | Doesn't capture the retrieval/agent angle as strongly |
| **muninn** | Norse mythology, raven logo, "memory" meaning, brand story | **Selected** |

---

## Name Availability

Checked 2026-02-11:

| Ecosystem | `muninn` | `sqlite-muninn` |
|-----------|----------|-----------------|
| **PyPI** | **Taken** — `stcorp/muninn` v7.2.1, active data catalogue (uses SQLite internally) | **Available** |
| **NPM** | Taken — `wopehq/muninn`, HTML parser, 35 downloads/week | **Available** |
| **GitHub** | 276 repos, none dominant. `colliery-io/muninn` (Rust, AI agent space, 11 stars) is closest | **Zero results** |

The `sqlite-` prefix avoids all collisions and follows the established SQLite extension naming convention (`sqlite-vec`, `sqlite-vss`, `sqlite-http`).

---

## Naming Architecture

| Layer | Old | New | Notes |
|-------|-----|-----|-------|
| **GitHub repo** | `sqlite-vector-graph` | `sqlite-muninn` | Follows `sqlite-{name}` convention |
| **PyPI package** | (unpublished) | `sqlite-muninn` | `pip install sqlite-muninn` |
| **NPM main package** | (unpublished) | `sqlite-muninn` | `npm install sqlite-muninn` |
| **NPM platform packages** | (unpublished) | `@sqlite-muninn/{platform}` | e.g. `@sqlite-muninn/darwin-arm64` |
| **Python import** | `vec_graph` | `sqlite_muninn` | `import sqlite_muninn` |
| **C shared library** | `vec_graph.so/.dylib/.dll` | `muninn.so/.dylib/.dll` | `.load ./muninn` in SQLite |
| **C entry point** | `sqlite3_vecgraph_init` | `sqlite3_muninn_init` | SQLite auto-discovers from filename |
| **C header** | `vec_graph.h` | `muninn.h` | Public API header |
| **Internal prefix** | `vecgraph_` / `vec_graph_` | `muninn_` | For C symbols, shadow tables, etc. |

### SQLite Loading

```sql
-- Old
.load ./vec_graph

-- New
.load ./muninn
```

SQLite's `load_extension()` auto-discovers the entry point `sqlite3_muninn_init` from the filename `muninn`.

### Python API

```python
# Old
import sqlite3
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
conn.load_extension("./vec_graph")

# New
import sqlite3
import sqlite_muninn

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_muninn.load(conn)
```

---

## SEO Strategy

The short package name is for humans. SEO keywords go in metadata:

| Channel | Where Keywords Live |
|---------|-------------------|
| **GitHub** | Repo description + topics (tags) |
| **PyPI** | `keywords` field in pyproject.toml + long description |
| **NPM** | `keywords` array in package.json + README |
| **Google** | README H1 + description + content |

### GitHub Repo Description

> Muninn — HNSW vector search, graph traversal & knowledge graphs for SQLite

### GitHub Topics

```
sqlite, vector-search, knowledge-graph, hnsw, graph-traversal,
node2vec, graphrag, sqlite-extension, embeddings, rag
```

### README H1

```markdown
# Muninn

**HNSW vector search + graph traversal + knowledge graphs for SQLite**

A zero-dependency C11 SQLite extension combining vector similarity search,
graph traversal TVFs, and Node2Vec embedding generation in a single loadable library.
```

### PyPI Keywords

```
sqlite, vector, search, hnsw, knowledge-graph, graph, traversal,
node2vec, embeddings, graphrag, rag, sqlite-extension
```

---

## Brand Identity

### Logo Concept

A raven silhouette — options for incorporating the tech identity:

- Graph nodes/edges subtly patterned into the wing feathers
- A raven carrying a glowing node in its talons
- A raven perched on a graph structure (like Odin's shoulder)
- Minimalist raven head profile with a single vector/node as the eye

### Color Palette Suggestions

- Primary: deep charcoal/black (raven)
- Accent: amber/gold (Odin's wisdom, the glowing knowledge)
- Background: dark navy or off-white

### Tagline Options

- "Memory for your data" (direct)
- "The raven remembers" (mythological)
- "Vector search + graph traversal for SQLite" (technical)

---

## Rename Checklist

### Phase 1: C Extension (Core)

- [ ] Rename `src/vec_graph.c` → `src/muninn.c`
- [ ] Rename `src/vec_graph.h` → `src/muninn.h`
- [ ] Update entry point: `sqlite3_vecgraph_init` → `sqlite3_muninn_init`
- [ ] Update all internal `#include "vec_graph.h"` references
- [ ] Update `Makefile`: output target `vec_graph$(EXT)` → `muninn$(EXT)`
- [ ] Update `Makefile`: all references to `vec_graph` in build rules
- [ ] Update C test files that reference `vec_graph`
- [ ] Verify `make all && make test` passes

### Phase 2: Python Integration

- [ ] Update `pytests/conftest.py`: extension loading path
- [ ] Update any Python imports/references to `vec_graph`
- [ ] Update benchmark scripts in `benchmarks/scripts/`
- [ ] Verify `make test-python` passes

### Phase 3: Documentation

- [ ] Update `README.md` — new name, description, loading instructions
- [ ] Update `CLAUDE.md` — all references to `vec_graph`
- [ ] Update `docs/plans/distribution_and_ci.md` — package names, wheel tags, npm structure
- [ ] Update `docs/plans/knowledge_graph_benchmark.md` — references to `vec_graph`
- [ ] Update any other docs under `docs/`

### Phase 4: Project Infrastructure

- [ ] Update `pyproject.toml` — project name and metadata
- [ ] Update `.github/workflows/` — any references to `vec_graph`
- [ ] Update `benchmarks/Makefile` — extension path references
- [ ] Rename GitHub repository: `sqlite-vector-graph` → `sqlite-muninn`
- [ ] Update GitHub repo description and topics
- [ ] Set up redirect from old repo name (GitHub does this automatically)

### Phase 5: Memory / Agent Context

- [ ] Update `.claude/` memory files referencing `vec_graph`
- [ ] Update `MEMORY.md` with new project name and conventions

### Phase 6: Publishing Prep (Future — Defer Until Distribution Plan)

- [ ] Register `sqlite-muninn` on PyPI (can reserve with empty package)
- [ ] Register `sqlite-muninn` on NPM
- [ ] Register `@sqlite-muninn` org on NPM for platform packages
- [ ] Create logo assets
- [ ] Update `docs/plans/distribution_and_ci.md` with finalized names

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| `colliery-io/muninn` (Rust, AI agent space) causes brand confusion | Low | Different ecosystem (Rust vs C/SQLite); `sqlite-` prefix disambiguates |
| `stcorp/muninn` on PyPI causes import confusion | None | We use `sqlite-muninn` / `sqlite_muninn`, not bare `muninn` |
| Someone registers `sqlite-muninn` on PyPI/NPM before us | Low | Register early (Phase 6) or reserve with placeholder packages |
| Existing links/references to `sqlite-vector-graph` break | Certain | GitHub auto-redirects renamed repos; update docs and bookmarks |
| C symbol rename (`vecgraph_` → `muninn_`) misses internal references | Medium | Use `grep -r vecgraph src/` and `grep -r vec_graph` to find all references |

---

## Naming Alternatives (Archived)

If `muninn` proves problematic, the runner-up names were:

1. **grimoire** — strongest "deep dark lore" imagery
2. **engram** — best NLP pun (engram/n-gram)
3. **lore** — most self-describing
4. **arcana** — deep secrets known to initiates
5. **codex** — ancient manuscript + "code indexed"

Full brainstorming notes from the naming session are not included here — they explored ~36 candidates across Norse mythology, forbidden texts, neuroscience, and pop culture references.
