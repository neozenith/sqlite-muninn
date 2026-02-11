# Project Rename: sqlite-vector-graph → sqlite-muninn

Rename the project from `sqlite-vector-graph` / `vec_graph` to `sqlite-muninn` / `muninn`, establishing a memorable brand identity with a cohesive naming architecture across all ecosystems.

**Status:** Mechanical rename completed. Brand identity incubating in [`brand_identity.md`](brand_identity.md).

---

## Table of Contents

1. [Decision](#decision)
2. [Name Availability](#name-availability)
3. [Naming Architecture](#naming-architecture)
4. [Rename Checklist](#rename-checklist)
5. [Risks and Mitigations](#risks-and-mitigations)

---

## Decision

**Old name:** `sqlite-vector-graph` (repo), `vec_graph` (C extension, Python import)

**New name:** `sqlite-muninn` (repo/packages), `muninn` (C extension, brand)

**Rationale:** The old name describes the implementation mechanism (vector + graph). The new name describes the *purpose* — a memory system for knowledge retrieval. As the project grows beyond raw HNSW/graph primitives into GraphRAG, knowledge indexing, community detection, and agent memory, the old name becomes increasingly misleading.

For background on the name choice, mythology, and shortlist, see [`brand_identity.md`](brand_identity.md).

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

---

## Rename Checklist

### Phase 1: C Extension (Core) — DONE

- [x] Rename `src/vec_graph.c` → `src/muninn.c`
- [x] Rename `src/vec_graph.h` → `src/muninn.h`
- [x] Update entry point: `sqlite3_vecgraph_init` → `sqlite3_muninn_init`
- [x] Update all internal `#include "vec_graph.h"` references
- [x] Update `Makefile`: output target `vec_graph$(EXT)` → `muninn$(EXT)`
- [x] Update `Makefile`: all references to `vec_graph` in build rules
- [x] Update C test files that reference `vec_graph`
- [x] Verify `make all && make test` passes

### Phase 2: Python Integration — DONE

- [x] Update `pytests/conftest.py`: extension loading path
- [x] Update any Python imports/references to `vec_graph`
- [x] Update benchmark scripts in `benchmarks/scripts/`
- [x] Verify `make test-python` passes

### Phase 3: Documentation — DONE

- [x] Update `README.md` — new name, description, loading instructions
- [x] Update `CLAUDE.md` — all references to `vec_graph`
- [x] Update `docs/plans/distribution_and_ci.md` — package names, wheel tags, npm structure
- [x] Update `docs/plans/knowledge_graph_benchmark.md` — references to `vec_graph`
- [x] Update any other docs under `docs/`

### Phase 4: Project Infrastructure — DONE (partial)

- [x] Update `pyproject.toml` — project name and metadata
- [ ] Update `.github/workflows/` — any references to `vec_graph` (none exist yet)
- [x] Update `benchmarks/Makefile` — extension path references
- [ ] Rename GitHub repository: `sqlite-vector-graph` → `sqlite-muninn` (manual GitHub operation, post-merge)
- [ ] Update GitHub repo description and topics (manual, post-merge)
- [x] Set up redirect from old repo name (GitHub does this automatically)

### Phase 5: Memory / Agent Context — DONE

- [x] Update `.claude/` memory files referencing `vec_graph`
- [x] Update `MEMORY.md` with new project name and conventions

### Phase 6: Publishing Prep (Future — Defer Until Distribution Plan)

- [ ] Register `sqlite-muninn` on PyPI (can reserve with empty package)
- [ ] Register `sqlite-muninn` on NPM
- [ ] Register `@sqlite-muninn` org on NPM for platform packages
- [ ] Create logo assets (see [`brand_identity.md`](brand_identity.md))
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
