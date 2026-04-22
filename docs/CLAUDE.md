# docs/ — Documentation Conventions

This file is loaded every time an assistant edits files under `docs/*.md`. It distills what separates well-documented OSS projects (sqlite-vec, pgvector, SQLite core, Redis commands, FastAPI, Tailwind, Chroma, Qdrant) from forgettable ones, and encodes those lessons as rules specific to muninn.

**Scope:** applies to `docs/*.md` base-layer pages only. Does NOT apply to `docs/plans/`, `docs/logo/`, `docs/misc/`, `docs/diagrams/`, or `docs/benchmarks/` — those have their own conventions.

---

## Absolute Rules (non-negotiable)

1. **Value before prose.** The first screen of every top-level page must contain something a reader can copy and run. No architecture essay, no "muninn is a C11 extension…" preamble. A one-sentence tagline → a runnable 10–15 line SQL snippet → expected output. Everything else follows.

2. **One-sentence tagline.** Format: `SQLite extension for <verb 1>, <verb 2>, and <verb 3> — all in SQL.` No adjectives like "powerful," "blazing-fast," "state-of-the-art." Verbs, not boasts.

3. **Every code block is labeled.** Fence every block with a language (` ```sql `, ` ```bash `, ` ```python `, ` ```c `) OR a filename header comment (`-- sqlite3 shell`, `# requires Python 3.11+`). Never naked ` ``` `.

4. **Every SQL example shows expected output.** As the pipe-delimited table SQLite prints (`.mode column` or `.mode box`). If the output is long, truncate with `...` and a comment. No example is complete without the result.

5. **Per-function page template is identical.** For every SQL function and virtual table in `api.md` (or a dedicated page), use this order, no substitutions:
   1. One-line purpose (imperative: "Extract named entities from text.")
   2. **Signature** block: bold name, italic params, list all overloads together
   3. Minimal 3-line example — just the call and expected result
   4. **Parameters** table: name, type, required?, default, description
   5. **Returns**: type and NULL behavior
   6. **Full recipe**: CREATE + INSERT + SELECT end-to-end
   7. **See also** cross-refs to related functions
   8. `Since: vX.Y` footer if version-scoped

6. **Task-based information architecture.** Organize `docs/` by what a user wants to *do* (`text-embeddings.md`, `centrality-community.md`, `entity-resolution.md`, `graphrag-cookbook.md`), not by internal module (`llama_chat.md`, `graph_tvf.md`). The 12 subsystems are implementation detail; users want verbs. Architecture details live only in `architecture.md`.

7. **Pin the honesty callout up top.** Pre-release status, platform support matrix (macOS-full / Linux-CPU / Windows-experimental / WASM), current llama.cpp submodule tag, GPU-layer requirements (`MUNINN_GPU_LAYERS=99` on macOS). First 200 words of `index.md`, not an appendix.

8. **Install is a matrix, not an essay.** Rows = platform (macOS / Linux / Windows / WASM), columns = method (prebuilt / pip / npm / build-from-source). One copy-paste command per cell. Pattern mirrors pgvector and sqlite-vec.

9. **Recipes must be runnable — not hypothetical.** If you write `INSERT INTO X SELECT muninn_embed(...)`, the adjacent setup (load model, create table, register) must be visible in the same code block or the one immediately above. No "assume the model is loaded" hand-waving. A reader should be able to copy the page top-to-bottom into `sqlite3` and have it work.

10. **No emoji, no marketing superlatives, no minimizers.** Drop "simply," "just," "easy," "blazingly fast." If GGUF model registration is awkward, say so and show the exact workaround. Honesty > polish.

---

## muninn-specific rules

11. **Every SQL-visible name uses the exact registered name.** `graph_node_betweenness` NOT `graph_betweenness`. `muninn_extract_entities` NOT `muninn_ner`. `hnsw_index` NOT `hnsw_vtab`. When in doubt, grep `src/muninn.c` and the registration functions — the name in `sqlite3_create_function_v2` / `xCreate` is authoritative.

12. **Every `muninn_*` function example shows model registration first.** GGUF functions are useless without a loaded model. Pattern: `INSERT INTO temp.muninn_models(name, model) SELECT 'MiniLM', muninn_embed_model('models/...gguf');` appears in every example that uses `muninn_embed`. Same for `temp.muninn_chat_models` + `muninn_chat_model`.

13. **Shadow-table mentions belong on the page that creates them.** The HNSW page documents `_config/_nodes/_edges`. The `graph_adjacency` page documents `_config/_nodes/_degree/_csr_fwd/_csr_rev/_delta`. Do not create a separate "Internals" page and scatter shadow-table docs across it.

14. **Vector blob format is documented *once* — in `text-embeddings.md` — with explicit `struct.pack`/`Float32Array`/C snippets, and cross-linked from every other page that mentions vectors.** Do not re-explain the float32 little-endian layout on each page.

15. **Graph TVFs use HIDDEN-column constraint syntax, not positional args.** Every graph TVF example reads `WHERE edge_table = '...' AND src_col = '...' AND dst_col = '...'`, never `graph_bfs('edges', 'src', 'dst', ...)`. This is the real calling convention — violating it in examples teaches users wrong patterns.

16. **GGUF/LLM extraction functions (`muninn_extract_*`) must show the JSON response shape** — the exact `{"entities":[...]}` or `{"clusters":{...}}` structure — inline with the call, like a Stripe docs three-column example. Mention that results carry JSON subtype `'J'` so `json_each()` works directly.

17. **Per-function performance notes when relevant.** A one-liner like "Betweenness is O(VE); enable `auto_approx_threshold` for |V| > 50,000" or "HNSW search is O(log N); `ef_search` trades recall for speed." Matches Redis's time-complexity convention. Skip when not load-bearing.

18. **Cross-reference, don't duplicate.** If `graphrag-cookbook.md` needs the Leiden parameter table, link to `centrality-community.md#leiden-community-detection` rather than re-printing the table. Link rot is easier to fix than keeping five copies in sync.

---

## Anti-patterns to avoid (seen in muninn's old docs)

1. **"Seven subsystems" language.** muninn currently registers 12 registration functions exposing 30+ SQL symbols. Count-based marketing grows stale; always prefer the verb-based summary ("vector search, graph algorithms, GGUF inference").
2. **Architecture essay before install.** The current `architecture.md` opens with internal module layering before the reader has loaded the extension. Wrong order for every audience except contributors.
3. **Inconsistent TVF examples.** Some current examples call `graph_degree('edges', 'src', 'dst', ...)` (positional) and others use `WHERE edge_table = ...` (constraint). Only the constraint form is registered; the positional form is a documentation lie.
4. **Promoting external extensions over our own.** The current `text-embeddings.md` leads with `sqlite-lembed` even though muninn's own `muninn_embed` (GGUF + Metal GPU) has been shipping. Always document our own surface first; alternatives second.
5. **Missing NER/RE/ER/label_groups entirely.** ~30% of muninn's SQL surface area is undocumented in the current docs. Documentation must cover every registered function — if it's callable, it's documented.
6. **Stale subsystem counts and outdated filenames in prose.** "`embed_gguf.c`" was renamed to `llama_embed.c`. Prose that names internal files rots on every refactor. Prefer SQL-visible names (which are stable registration contracts) over C filenames (which are internal).
7. **Diagrams referenced but absent.** `architecture.md` currently uses `![Extension Overview](diagrams/architecture-overview.png)` — check the file exists at `docs/diagrams/` before referencing it. A broken image is worse than no image.
8. **"Further Reading" as a dumping ground.** Don't end every page with the same five links. Cross-references should be specific: link to an anchor, explain *why* the reader should follow it in one clause ("for the `resolution` parameter tuning, see [Leiden](centrality-community.md#leiden-community-detection)").

---

## Template for a new SQL function page section

Every function block should look exactly like this — copy the skeleton and fill it in:

```markdown
### `function_name`

One-line imperative purpose.

**Signature**

```sql
function_name(
    required_arg TEXT,         -- what it is
    optional_arg INTEGER = 10  -- what it controls, default shown
) -> RETURN_TYPE
```

**Example**

```sql
SELECT function_name('hello', 5) AS result;
```

```text
result
------
<expected output>
```

**Parameters**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `required_arg` | TEXT | yes | — | … |
| `optional_arg` | INTEGER | no | 10 | … |

**Returns**: TYPE. NULL when … .

**Full recipe**

```sql
-- Minimal end-to-end usage
.load ./muninn
-- setup
-- call
-- expected output
```

**See also**: [`related_function`](#related_function), [Guide page](../guide.md#anchor).
```

---

## Authoring checklist before committing a doc change

- [ ] Page opens with a one-sentence tagline and a runnable example (for top-level pages)
- [ ] Every SQL block shows expected output
- [ ] Every code fence has a language or filename label
- [ ] Every SQL function name matches `src/muninn.c` registration — greps cleanly
- [ ] Every GGUF example includes model registration
- [ ] Graph TVFs use `WHERE edge_table = ...` constraint syntax
- [ ] No stale subsystem counts or old filenames in prose
- [ ] Broken diagram links checked against `docs/diagrams/`
- [ ] Cross-references are anchored and specific
- [ ] No emoji; no "simply," "just," "easy," "blazing"
