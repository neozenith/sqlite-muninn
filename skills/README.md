# Agent Skills for sqlite-muninn

This directory ships [Claude Code Skills](https://code.claude.com/docs/en/skills) alongside the library. When Claude Code is run inside a project that has installed muninn (or a plugin marketplace pointing at this repo), these skills become discoverable automatically — Claude reads their `description` fields and loads the matching SKILL.md when the user's task maps to one of the workflows below.

## Install

The canonical install path uses [skills.sh](https://skills.sh) — a package manager for agent skills:

```bash
# Interactive (opt-in each skill)
npx skills add neozenith/sqlite-muninn

# Project-scoped, non-interactive
npx skills add neozenith/sqlite-muninn --agent claude-code codex --yes

# User-scoped (~/.claude/skills), non-interactive
npx skills add neozenith/sqlite-muninn --agent claude-code codex --global --yes
```

Alternatively, add this repository as a Claude Code plugin marketplace and install the published plugin:

```
/plugin marketplace add neozenith/sqlite-muninn
/plugin install muninn
```

Either path drops the same SKILL.md directories into your Claude skills root. No further registration is required — Claude's metadata-only first pass reads every `description` field, and loads the full `SKILL.md` content only when a user's task matches the trigger phrases.

## What ships

Nine task-granular skills, aligned 1:1 with the nine top-level `docs/` pages.

| Skill | Primary trigger phrases | Backing doc |
|-------|-------------------------|-------------|
| [`muninn-setup`](./muninn-setup/SKILL.md) | "install muninn", "load the extension", ".load ./muninn", "pip install sqlite-muninn", "npm install sqlite-muninn" | [getting-started.md](../docs/getting-started.md) |
| [`muninn-vector-search`](./muninn-vector-search/SKILL.md) | "vector search", "HNSW", "KNN", "nearest neighbor", "similarity search in SQLite" | [api.md#hnsw_index](../docs/api.md) |
| [`muninn-embed-text`](./muninn-embed-text/SKILL.md) | "text embedding", "muninn_embed", "GGUF embedding", "semantic search", "MiniLM" | [text-embeddings.md](../docs/text-embeddings.md) |
| [`muninn-chat-extract`](./muninn-chat-extract/SKILL.md) | "muninn_chat", "NER", "relation extraction", "LLM in SQL", "GBNF grammar", "summarize", "Qwen3.5" | [chat-and-extraction.md](../docs/chat-and-extraction.md) |
| [`muninn-graph-algorithms`](./muninn-graph-algorithms/SKILL.md) | "BFS", "PageRank", "betweenness", "Leiden", "community detection", "graph algorithm in SQLite" | [centrality-community.md](../docs/centrality-community.md) |
| [`muninn-graph-select`](./muninn-graph-select/SKILL.md) | "dbt selector", "lineage query", "impact analysis", "ancestors", "descendants", "graph_select" | [graph-select.md](../docs/graph-select.md) |
| [`muninn-node2vec`](./muninn-node2vec/SKILL.md) | "node2vec", "graph embedding", "DeepWalk", "structural embedding", "similar nodes" | [node2vec.md](../docs/node2vec.md) |
| [`muninn-graphrag`](./muninn-graphrag/SKILL.md) | "GraphRAG", "KG retrieval", "knowledge graph", "muninn_extract_er", "entity resolution" | [graphrag-cookbook.md](../docs/graphrag-cookbook.md) |
| [`muninn-troubleshoot`](./muninn-troubleshoot/SKILL.md) | "unable to open shared library", "SQLITE_OMIT_LOAD_EXTENSION", "CMake hangs", "model not registered" | [getting-started.md § Common pitfalls](../docs/getting-started.md) |

Every skill covers all five runtimes where relevant: SQLite CLI, C, Python, Node.js, and WASM.

## How the skills are organized

Each skill is a single directory with a `SKILL.md` entrypoint at its root:

```
skills/
├── CLAUDE.md                    # curation guide (read-this-first when editing)
├── README.md                    # this file
├── muninn-setup/
│   └── SKILL.md
├── muninn-vector-search/
│   └── SKILL.md
├── muninn-embed-text/
│   └── SKILL.md
├── muninn-chat-extract/
│   └── SKILL.md
├── muninn-graph-algorithms/
│   └── SKILL.md
├── muninn-graph-select/
│   └── SKILL.md
├── muninn-node2vec/
│   └── SKILL.md
├── muninn-graphrag/
│   └── SKILL.md
└── muninn-troubleshoot/
    └── SKILL.md
```

Deeper reference material (when warranted) lives in `references/*.md` subfolders — Claude loads them only when explicitly linked from the SKILL.md during a live conversation.

## Authoring

If you are *editing* skills, read [`CLAUDE.md`](./CLAUDE.md) first. It contains the curation rules we maintain — naming conventions, frontmatter schema, trigger-phrase guidance, anti-patterns, and a summary of prior-art research on how other OSS libraries (Supabase, Vercel, DuckDB, PlanetScale, Anthropic) distribute skills.

## License

MIT, same as the library.
