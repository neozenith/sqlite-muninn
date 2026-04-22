# skills/ — Skill Curation Guide

This file is loaded whenever an assistant edits files under `skills/`. It defines how we author, name, and maintain the `/muninn-*` agent skills that ship *with* the library so downstream Claude Code users get them automatically when they install muninn.

**Scope:** applies to every file under `skills/` (this `CLAUDE.md`, the top-level `README.md`, each `skills/<name>/SKILL.md`, and any `skills/<name>/references/*.md` or `skills/<name>/scripts/*`).

---

## Why we ship skills

Library users who run Claude Code already have the authoritative reference in `docs/`. Skills are not a second copy of the docs — they are **procedural knowledge for an agent**: how to recover from a build failure, how to phrase a `muninn_embed` call inside a trigger, which model to register when the user says "semantic search." If a skill only restates what `docs/*.md` already explains, delete the skill; if it teaches Claude to *do* something correctly without re-reading the docs, keep it.

Distribution channel: `npx skills add neozenith/sqlite-muninn` (see [skills.sh](https://skills.sh/)) and/or Claude Code's `/plugin marketplace add` command. Both read the same `skills/` directory layout — do not fork into a `.claude-plugin/` tree.

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

## Prior art — what we learned from other OSS libraries

Research summary from April 2026 (see `tmp/skills-research/` for the raw reports, or re-run the general-purpose agent prompts preserved there).

### Positive examples worth copying

- **[duckdb/duckdb-skills](https://github.com/duckdb/duckdb-skills)** — six task-oriented skills (`attach-db`, `query`, `read-file`, `duckdb-docs`, `read-memories`, `install-duckdb`), one verb per skill. Closest conceptual fit to muninn because DuckDB is also an embedded SQL engine with extensions. The verb-per-skill granularity is what we adopted.
- **[anthropics/skills](https://github.com/anthropics/skills)** — the authoritative frontmatter contract plus `skill-creator` meta-skill. We mirror the `SKILL.md`-in-directory layout from here.
- **[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)** — seven discipline-sized skills, each with runnable `scripts/` + `references/`. Installable via `npx skills add`. Good precedent for bundling helper scripts alongside prose.
- **[supabase/agent-skills](https://github.com/supabase/agent-skills)** — hybrid granularity: one broad `supabase` skill + a narrow `supabase-postgres-best-practices` skill. Shows that it's OK to mix product-wide and surgical skills in the same repo.
- **[planetscale/database-skills](https://github.com/planetscale/database-skills)** — one skill per database product (`mysql`, `postgres`, `vitess`, `neki`). Useful reference for per-surface granularity — but note that Planet Scale's skills trend too broad; muninn's per-workflow decomposition is tighter.

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

---

## When to archive vs delete

If a skill turns out to be misaligned (too broad, teaches wrong patterns, duplicates docs), move it to `tmp/archived-skills/` rather than `git rm`. Rationale: preserves the negative example for future curation decisions; easy to reinstate if the workflow turns out to matter. The archive directory is gitignored — it is a local scratchpad, not a second source of truth.
