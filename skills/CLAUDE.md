# skills/ — Skill Curation Guide

This file is loaded whenever an assistant edits files under `skills/`. It defines how we author, name, and maintain the `/muninn-*` agent skills that ship *with* the library so downstream Claude Code users get them automatically when they install muninn.

**Scope:** applies to every file under `skills/` (this `CLAUDE.md`, the top-level `README.md`, each `skills/<name>/SKILL.md`, and any `skills/<name>/references/*.md` or `skills/<name>/scripts/*`).

---

## Why we ship skills

Library users who run Claude Code already have the authoritative reference in `docs/`. Skills are not a second copy of the docs — they are **procedural knowledge for an agent**: how to recover from a build failure, how to phrase a `muninn_embed` call inside a trigger, which model to register when the user says "semantic search." If a skill only restates what `docs/*.md` already explains, delete the skill; if it teaches Claude to *do* something correctly without re-reading the docs, keep it.

Distribution channels — all CLI, all verified against `claude plugin --help` and `codex marketplace --help`:

- `npx skills add neozenith/sqlite-muninn …` (see [skills.sh](https://skills.sh/)) — copies SKILL.md directories into `~/.claude/skills/` or `~/.codex/skills/`, bypasses the plugin layer.
- `claude plugin marketplace add neozenith/sqlite-muninn` + `claude plugin install muninn@sqlite-muninn` — reads `.claude-plugin/marketplace.json`.
- `codex marketplace add neozenith/sqlite-muninn` + `[plugins."muninn@sqlite-muninn"] enabled = true` in `~/.codex/config.toml` — reads `.codex-plugin/plugin.json`.

All three read the same top-level `skills/<name>/SKILL.md` files. Do not duplicate the skill tree into `.claude-plugin/`, `.codex-plugin/`, or anywhere else.

---

## Absolute rules

1. **One skill = one workflow.** Prefer `muninn-vector-search` (what the user does) over `muninn-hnsw-api` (a C file). If you cannot describe the skill in a single gerund phrase — "searching vectors," "extracting entities," "debugging a build" — split or merge until you can.

2. **Frontmatter is mandatory and bounded.**

    ```yaml
    ---
    name: muninn-vector-search        # ≤64 chars, lowercase-kebab, no "claude"/"anthropic"
    description: >                    # ≤1024 chars, third person, INCLUDES TRIGGER PHRASES
      Builds, queries, and maintains HNSW vector indexes in SQLite via the
      hnsw_index virtual table. Covers vector blob encoding across Python,
      Node.js, and C. Use when the user mentions "vector search", "HNSW",
      "nearest neighbor", "embedding search", "KNN", "similarity search
      in SQLite", or creates a `hnsw_index` virtual table.
    license: MIT
    ---
    ```

    The `description` field is the only thing Claude reads before deciding whether to load the skill — front-load the *trigger phrases* a user would actually type.

3. **Filename is always `SKILL.md`** (uppercase). Layout:

    ```
    skills/
    └── muninn-vector-search/
        ├── SKILL.md            # required entrypoint
        ├── references/         # progressive-disclosure deep-dives
        │   └── *.md
        └── scripts/            # runnable helpers (optional)
            └── *.py / *.sh
    ```

    Per Anthropic's spec, `SKILL.md` must be under 500 lines. Push long recipes into `references/*.md` and link from the entrypoint.

4. **Every code block is labeled and runnable.** Inherits the rule from `docs/CLAUDE.md`: language-tagged fence (```` ```sql `, ` ```python `, ` ```bash ` ````), and every SQL block shows expected output. Skills diverging from docs actively mis-train the agent — run every block through `sqlite3` before committing.

5. **Use the registered SQL names.** Same as `docs/CLAUDE.md` rule #11: `graph_node_betweenness`, not `graph_betweenness`. `muninn_extract_entities`, not `muninn_ner`. When the API reference changes a name, the skill changes too — `grep -r <old_name> skills/` on every rename.

6. **Five paradigm lanes in every skill where relevant.** When a skill covers API surface, show the invocation in:

    - **SQLite CLI** — `.load ./muninn`, pure SQL
    - **C** — `sqlite3_auto_extension(sqlite3_muninn_init);` or `sqlite3_load_extension`
    - **Python** — `import sqlite_muninn; sqlite_muninn.load(db)`
    - **Node.js** — `import { load } from "sqlite-muninn"; load(db);`
    - **WASM** — `sqlite3.oo1.OpfsDb` + loaded `sqlite3_muninn_init`

    If a paradigm genuinely doesn't apply (e.g. `muninn-troubleshoot` in Node has different content than Python), say so explicitly rather than inventing pseudo-examples.

7. **Constraint-form graph TVFs only.** Every `graph_*` example (except `graph_select`) uses `WHERE edge_table = '...' AND src_col = '...' AND dst_col = '...'`. The positional form (`graph_bfs('edges', 'src', 'dst', ...)`) does not parse and teaching it is a documentation lie.

8. **Every `muninn_embed*` / `muninn_chat*` / `muninn_extract_*` example shows model registration first** — the `INSERT INTO temp.muninn_models(name, model) SELECT 'alias', muninn_embed_model('path.gguf');` line is not optional; the functions throw without it.

9. **Trigger phrases are in the `description`, not just the body.** Claude's skill-selection pass reads only the frontmatter. If the user would say "knowledge graph extraction" but that phrase only appears in the body, the skill never loads.

10. **No emoji, no superlatives.** Drop "powerful," "blazing-fast," "simply," "just," "easy." Skills compete for Claude's attention on *accuracy*, not marketing.

---

## Anti-patterns (do not ship these)

1. **The monolithic everything-skill.** The archived `skills/muninn/SKILL.md` (see `tmp/archived-skills/`) was one 350-line skill covering vector search, graph, embeddings, and chat. It always loaded, competed with every other skill in the session, and taught outdated positional graph-TVF syntax. Split by workflow.

2. **Descriptions that don't trigger.** `description: "Helps with muninn."` — Claude cannot route to that. Every description must contain at least three concrete phrases a user would type (API name, capability name, or domain term).

3. **Silent drift from the registered SQL surface.** The archived skill referenced `graph_betweenness` (does not exist — it is `graph_node_betweenness`). Run `grep -E 'muninn_|graph_|hnsw_' skills/**/*.md` against `src/muninn.c` registration calls before every merge.

4. **Promoting a deprecated companion extension.** The archived skill led with `sqlite-lembed` / `sqlite-rembed` over muninn's own `muninn_embed`. We ship our own GGUF pipeline with Metal GPU — document it first, alternatives second.

5. **Cross-language examples that don't actually work.** If you write a Node.js snippet, it must import `sqlite-muninn` exactly as the npm package exports it. Do not invent API shapes.

6. **Fallback chains / "skip with warning" advice.** Inherits from `~/.claude/CLAUDE.md`: skills teach requirements, not graceful degradation. If the user asks for GGUF embedding, the skill installs a GGUF model — it does not fall back to a regex.

7. **Deep reference nesting.** `SKILL.md` → `references/advanced.md` → `references/details.md` fails progressive-disclosure because Claude partial-reads. Keep references one level deep.

8. **Orphaned scripts.** Every file under `scripts/` must be linked from the `SKILL.md`. Unreferenced files are dead weight that inflates the plugin tarball.

9. **"Voodoo constants."** A snippet with `ef_search = 47  # magic` teaches the wrong lesson. Every numeric constant in bundled code gets a one-line comment explaining *why*.

10. **Confusing install with load.** `pip install sqlite-muninn` is install. `sqlite_muninn.load(db)` is load. A skill that tells users to "install with `sqlite_muninn.load(db)`" is a recipe for silent failure.

---

## The skill set we ship

Nine task-granular skills, aligned with the nine top-level `docs/` pages. Each is a separate directory.

| Skill | Trigger workflow | Primary doc |
|-------|------------------|-------------|
| `muninn-setup` | Install + load + smoke-test across SQLite CLI / C / Python / Node.js / WASM | [getting-started.md](../docs/getting-started.md) |
| `muninn-vector-search` | Build and query HNSW indexes, including blob encoding per language | [api.md#hnsw_index](../docs/api.md) |
| `muninn-embed-text` | GGUF text embedding — register model, embed, compose with HNSW | [text-embeddings.md](../docs/text-embeddings.md) |
| `muninn-chat-extract` | `muninn_chat`, NER, RE, summarize, GBNF grammars, batch variants | [chat-and-extraction.md](../docs/chat-and-extraction.md) |
| `muninn-graph-algorithms` | BFS/DFS/shortest path/PageRank/components + centrality + Leiden | [centrality-community.md](../docs/centrality-community.md) |
| `muninn-graph-select` | dbt-style lineage DSL (`+C`, `@C`, `1+C+1`, set ops) | [graph-select.md](../docs/graph-select.md) |
| `muninn-node2vec` | Structural graph embeddings written into HNSW | [node2vec.md](../docs/node2vec.md) |
| `muninn-graphrag` | Composite retrieval pipeline (chunk → embed → NER → KG → Leiden → label → retrieve) | [graphrag-cookbook.md](../docs/graphrag-cookbook.md) |
| `muninn-troubleshoot` | Build failures, platform caveats, extension-loading errors across all 5 runtimes | [getting-started.md § Common pitfalls](../docs/getting-started.md) |

Do not add a tenth skill without first asking: can this be a `references/*.md` inside an existing skill? If yes, it does not need its own slot.

---

## Plugin-ecosystem files (dual Claude Code + Codex support)

The `skills/` tree is the single source of truth. Two thin manifest files at the repo root expose the same bundle to both plugin ecosystems:

| File | Purpose | Authoritative schema |
|------|---------|-----------------------|
| [`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json) | Claude Code marketplace + plugin definition (one manifest, `strict: false` so per-plugin `plugin.json` is not needed) | [code.claude.com/docs/en/plugin-marketplaces](https://code.claude.com/docs/en/plugin-marketplaces) |
| [`.codex-plugin/plugin.json`](../.codex-plugin/plugin.json) | Codex plugin manifest (includes the `interface` block with `displayName`, `defaultPrompt`, etc.) | [developers.openai.com/codex/plugins/build](https://developers.openai.com/codex/plugins/build) |
| [`AGENTS.md`](../AGENTS.md) | Symlink → `CLAUDE.md`. Codex reads `AGENTS.md` for per-project instructions; Claude Code reads `CLAUDE.md`. One source, two entrypoints. | [Codex AGENTS.md docs](https://developers.openai.com/codex/guides/agents-md) |

Codex also accepts `.claude-plugin/marketplace.json` as a fallback marketplace source — the two manifests do not compete.

### Rules for maintaining parity

1. **Every skill added / renamed / removed requires both manifests updated.**
    - Add or remove the path in `.claude-plugin/marketplace.json` → `plugins[0].skills[]` (explicit path, one per skill).
    - `.codex-plugin/plugin.json` uses `"skills": "./skills/"` auto-discovery — no edit needed for add/remove, but re-run the validator to confirm.
2. **Run `claude plugin validate .claude-plugin/marketplace.json` on every manifest change.** Non-negotiable. It runs Claude's Zod schema against the file and fails fast on unknown fields, missing required fields, or bad nesting. If this command reports `✔ Validation passed`, the manifest will load. If not, it won't.
3. **Never add a `$schema` key to the Claude manifest.** Claude's schema uses `additionalProperties: false` — unrecognized keys fail validation. This is counterintuitive if you are used to VS Code-friendly JSON where `$schema` is a helpful hint; here it is a hard error.
4. **Never promote `./skills/` wildcard to the Claude marketplace manifest.** `anthropics/skills` uses an explicit path array and so must we — the wildcard form is undocumented for Claude Code and may silently register zero skills.
5. **`strict: false` is load-bearing.** It tells the marketplace "this plugin entry is the complete definition" — don't flip to `strict: true` unless you also add a `.claude-plugin/plugin.json` per-plugin manifest with the full schema.
6. **Version fields must match — never hand-edit them.** The canonical version lives in the repo-root `VERSION` file. Run `make version-stamp` (or `uv run scripts/generate_build.py version`) to propagate it into `.claude-plugin/marketplace.json` (both `metadata.version` and `plugins[0].version`), `.codex-plugin/plugin.json` (`version`), and `npm/package.json`. The stamp targets are declared in `VERSION_STAMP_TARGETS` inside `scripts/generate_build.py` — add new files there, not by hand-editing elsewhere. Check-only mode: `uv run scripts/generate_build.py version --check` (exits 1 if any target is stale — useful in CI).
7. **Do not edit `AGENTS.md` directly.** It is a symlink to `CLAUDE.md`. Edit `CLAUDE.md` and the change propagates. If the symlink is ever materialized into a file (Windows contributor, some CI flow), re-symlink with `ln -sf CLAUDE.md AGENTS.md`.

### Shell CLI commands we support (authoritative — not slash commands)

All three install paths are CLI-only. Claude Code's TUI `/plugin …` slash commands exist but we do not document them — the shell equivalents work identically and compose with shell scripts / CI.

| Path | Shell commands |
|------|----------------|
| skills.sh | `npx skills add neozenith/sqlite-muninn --agent claude-code codex --yes` |
| Claude Code plugin | `claude plugin marketplace add neozenith/sqlite-muninn`<br>`claude plugin install muninn@sqlite-muninn` (`--scope user\|project\|local`) |
| Codex plugin | `codex marketplace add neozenith/sqlite-muninn`<br>append `[plugins."muninn@sqlite-muninn"] enabled = true` to `~/.codex/config.toml` |

For the full discovered `claude plugin` subtree — `list`, `enable`, `disable`, `uninstall`, `update`, `validate` — see `claude plugin --help`. When these outputs change (CLI update), re-sync `skills/README.md`.

---

## Prior art — what we learned from other OSS libraries

Research summary from April 2026 (see `tmp/skills-research/` for the raw reports, or re-run the general-purpose agent prompts preserved there).

### Positive examples worth copying

- **[duckdb/duckdb-skills](https://github.com/duckdb/duckdb-skills)** — six task-oriented skills (`attach-db`, `query`, `read-file`, `duckdb-docs`, `read-memories`, `install-duckdb`), one verb per skill. Closest conceptual fit to muninn because DuckDB is also an embedded SQL engine with extensions. The verb-per-skill granularity is what we adopted.
- **[anthropics/skills](https://github.com/anthropics/skills)** — the authoritative frontmatter contract plus `skill-creator` meta-skill. We mirror the `SKILL.md`-in-directory layout from here.
- **[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)** — seven discipline-sized skills, each with runnable `scripts/` + `references/`. Installable via `npx skills add`. Good precedent for bundling helper scripts alongside prose.
- **[supabase/agent-skills](https://github.com/supabase/agent-skills)** — hybrid granularity: one broad `supabase` skill + a narrow `supabase-postgres-best-practices` skill. Shows that it's OK to mix product-wide and surgical skills in the same repo.
- **[planetscale/database-skills](https://github.com/planetscale/database-skills)** — one skill per database product (`mysql`, `postgres`, `vitess`, `neki`). Useful reference for per-surface granularity — but note that Planet Scale's skills trend too broad; muninn's per-workflow decomposition is tighter.
- **[zeabur/agent-skills](https://github.com/zeabur/agent-skills)** — the canonical **dual Claude Code + Codex** reference. Ships both `.claude-plugin/` and `.codex-plugin/` pointing at a single top-level `skills/` directory — the layout we adopted.

### Negative examples to avoid

- **[alirezarezvani/claude-skills](https://github.com/alirezarezvani/claude-skills)** (232+ skills) and **[sickn33/antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills)** (1,400+ skills) — aggregator repos with mass-produced, vague descriptions. At this scale, Claude's description-based routing degrades because every skill competes with every other. Cautionary tale for us: stay at single-digit skill count until we have measurable routing problems to solve.
- **The archived `skills/muninn/SKILL.md`** in this repo — canonical monolith: one skill, always loaded, covered too much surface area, taught the wrong TVF calling convention, and referenced renamed C files (`embed_gguf.c` → `llama_embed.c`). Preserved in `tmp/archived-skills/` as a cautionary reference when we are tempted to re-merge.

### Patterns Anthropic's docs explicitly endorse

- Progressive disclosure via `references/*.md` (load only when triggered).
- `description` doubles as router — third person, concrete trigger phrases.
- Bundled runnable scripts under `scripts/` (execute without polluting the context window with their source).
- Keep `SKILL.md` under 500 lines; use `<details>` blocks for legacy behavior.
- Pick one default tool; mention escape hatches, don't offer N equal alternatives.

Sources:
- [Claude Code skills docs](https://code.claude.com/docs/en/skills)
- [Agent Skills overview (API)](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Anthropic engineering blog: Equipping agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Agent Skills open standard](https://agentskills.io)
- [Atomic-skills analysis (ralphable.com)](https://ralphable.com/blog/claude-code-hallucination-problem-atomic-skills-reliable-output) — documents 12–18% silent-logic-error rate in multi-step skill tasks; atomic pass/fail criteria drop this >60%. We cite `pytests/` as the pass/fail harness in our skills.

---

## Authoring checklist before committing a skill change

- [ ] Frontmatter: `name` ≤ 64 chars lowercase-kebab, `description` ≤ 1024 chars in third person, includes ≥3 trigger phrases
- [ ] `SKILL.md` under 500 lines; longer content under `references/`
- [ ] Every SQL block shows expected output
- [ ] Every graph TVF example uses `WHERE edge_table = ...` constraint syntax (except `graph_select`)
- [ ] Every `muninn_embed*` / `muninn_chat*` / `muninn_extract_*` example shows model registration first
- [ ] SQL names grep cleanly against `src/muninn.c` registration calls
- [ ] All five language paradigms covered (or explicit "N/A for WASM because…")
- [ ] No emoji; no "simply," "just," "easy," "blazing"
- [ ] No fallback / try-except-skip chains; requirements are mandatory
- [ ] Referenced `references/*.md` and `scripts/*` files actually exist and are linked
- [ ] `.claude-plugin/marketplace.json` `plugins[0].skills[]` enumerates every skill directory (added/removed/renamed entries updated)
- [ ] `.codex-plugin/plugin.json` version matches `.claude-plugin/marketplace.json` version
- [ ] `AGENTS.md` symlink still resolves to `CLAUDE.md` (`readlink AGENTS.md` prints `CLAUDE.md`)
- [ ] `claude plugin validate .claude-plugin/marketplace.json` returns `✔ Validation passed`
- [ ] No `$schema` key in `.claude-plugin/marketplace.json` (the Claude validator rejects it)
- [ ] `skills/README.md` install commands still match the shell subcommand tree from `claude plugin --help` and `codex marketplace --help` (CLI surfaces change between releases)

---

## When to archive vs delete

If a skill turns out to be misaligned (too broad, teaches wrong patterns, duplicates docs), move it to `tmp/archived-skills/` rather than `git rm`. Rationale: preserves the negative example for future curation decisions; easy to reinstate if the workflow turns out to matter. The archive directory is gitignored — it is a local scratchpad, not a second source of truth.
