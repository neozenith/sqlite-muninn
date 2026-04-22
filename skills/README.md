# Agent Skills for sqlite-muninn

Ships nine task-granular [agent skills](https://agentskills.io) alongside the library so your AI coding assistant (Claude Code or OpenAI Codex CLI) knows the exact SQL calling conventions, model registration steps, and cross-runtime patterns for muninn — without re-reading the docs each session.

The same `skills/<name>/SKILL.md` files are consumed by both ecosystems.

## Install — pick one path

All three paths are **shell-only** — no TUI / slash commands required.

### A. skills.sh (one command, both ecosystems)

```bash
# Interactive (prompts which skills to enable, which agent(s))
npx skills add neozenith/sqlite-muninn

# Non-interactive: all skills, both agents, user-global install
npx skills add neozenith/sqlite-muninn --agent claude-code codex --global --yes

# Non-interactive: project-local (writes into the current repo's .claude/ and .agents/)
npx skills add neozenith/sqlite-muninn --agent claude-code codex --yes
```

[skills.sh](https://skills.sh) copies `SKILL.md` directories directly — it does not use the plugin systems. Resulting install paths:

| Agent | Scope | Install path |
|-------|-------|--------------|
| Claude Code | user-global | `~/.claude/skills/<name>/` |
| Claude Code | project-local | `.claude/skills/<name>/` |
| Codex | user-global | `~/.codex/skills/<name>/` (also `~/.agents/skills/<name>/`) |
| Codex | project-local | `.agents/skills/<name>/` |

### B. Claude Code plugin (CLI)

```bash
# Register this repo as a marketplace
claude plugin marketplace add neozenith/sqlite-muninn

# Install the plugin (default scope: user)
claude plugin install muninn@sqlite-muninn
```

Scope flag controls where the registration is persisted:

```bash
# Personal, every project       → ~/.claude/settings.json
claude plugin install muninn@sqlite-muninn                            # (default: --scope user)

# Team shared, commit into repo → <repo>/.claude/settings.json
claude plugin marketplace add neozenith/sqlite-muninn --scope project
claude plugin install muninn@sqlite-muninn --scope project

# Personal + repo-scoped        → <repo>/.claude/settings.local.json  (usually .gitignored)
claude plugin marketplace add neozenith/sqlite-muninn --scope local
claude plugin install muninn@sqlite-muninn --scope local
```

Other useful subcommands:

```bash
claude plugin list                              # show installed plugins
claude plugin disable muninn@sqlite-muninn      # keep installed, stop loading
claude plugin enable  muninn@sqlite-muninn      # re-enable
claude plugin update  muninn@sqlite-muninn      # pull latest from the source repo
claude plugin uninstall muninn@sqlite-muninn    # remove
claude plugin marketplace update sqlite-muninn  # refresh marketplace metadata
claude plugin validate .claude-plugin/marketplace.json   # validate manifest against Claude's schema
```

The marketplace manifest lives at [`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json) — it declares a plugin called `muninn` that enumerates all nine skill directories (the pattern used by [`anthropics/skills`](https://github.com/anthropics/skills)).

### C. Codex CLI plugin (CLI + TOML)

Codex exposes marketplace registration as a CLI subcommand, but enabling a specific plugin is a `~/.codex/config.toml` edit (matching Codex's existing persistence model for plugin state):

```bash
# Register the marketplace via CLI
codex marketplace add neozenith/sqlite-muninn

# Enable the plugin by appending to ~/.codex/config.toml
cat >> ~/.codex/config.toml <<'TOML'

[plugins."muninn@sqlite-muninn"]
enabled = true
TOML
```

To pin a specific git ref (tag / branch / commit):

```bash
codex marketplace add neozenith/sqlite-muninn --ref v0.4.0
```

For monorepos where the plugin lives in a subdirectory:

```bash
codex marketplace add org/big-repo --sparse path/to/plugin --sparse .claude-plugin
```

The Codex plugin manifest is at [`.codex-plugin/plugin.json`](../.codex-plugin/plugin.json). Codex also reads `.claude-plugin/marketplace.json` as a fallback, so the two manifests cooperate.

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

Every skill covers all five runtimes where applicable: SQLite CLI, C, Python, Node.js, and WASM.

## Repo layout (plugin-ecosystem files)

```
sqlite-muninn/
├── AGENTS.md                      # symlink → CLAUDE.md (Codex contributor entrypoint)
├── CLAUDE.md                      # Claude Code contributor entrypoint
├── .claude-plugin/
│   └── marketplace.json           # Claude Code marketplace + plugin definition
├── .codex-plugin/
│   └── plugin.json                # Codex plugin manifest
└── skills/
    ├── CLAUDE.md                  # curation guide (read before editing)
    ├── README.md                  # this file
    ├── muninn-setup/SKILL.md
    ├── muninn-vector-search/SKILL.md
    ├── muninn-embed-text/SKILL.md
    ├── muninn-chat-extract/SKILL.md
    ├── muninn-graph-algorithms/SKILL.md
    ├── muninn-graph-select/SKILL.md
    ├── muninn-node2vec/SKILL.md
    ├── muninn-graphrag/SKILL.md
    └── muninn-troubleshoot/SKILL.md
```

The top-level `skills/` tree is the single source of truth. Both the Claude Code marketplace manifest and the Codex plugin manifest reference the same `SKILL.md` files — no duplication, no drift.

## Authoring

If you are *editing* skills or the plugin manifests, read [`CLAUDE.md`](./CLAUDE.md) first. It contains the curation rules — naming conventions, frontmatter schema, trigger-phrase guidance, the manifest-parity rule, anti-patterns, and a summary of prior-art research on how other OSS libraries distribute skills.

## Why two manifests in one repo

Claude Code and Codex both implement [Agent Skills](https://agentskills.io), but their plugin systems diverged slightly:

- **Claude Code** looks for `.claude-plugin/marketplace.json` and offers `claude plugin` shell subcommands for install/enable/disable/validate/update.
- **Codex** looks for `.codex-plugin/plugin.json` *and* reads `.claude-plugin/marketplace.json` as a fallback. Shell exposes only `codex marketplace add`; enable state lives in `~/.codex/config.toml`.

Ship both. Maintenance cost is two small JSON files, and the reference repos ([`zeabur/agent-skills`](https://github.com/zeabur/agent-skills) is the canonical example) converge on this dual-manifest pattern.

## License

MIT, same as the library.
