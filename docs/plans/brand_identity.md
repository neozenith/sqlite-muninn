# Brand Identity: Muninn

Incubation document for logo, messaging, and visual identity. Extracted from `rename_to_muninn.md` after the mechanical rename was completed.

**Status:** Incubating — not yet implemented.

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

### README H1 (Future)

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

## Logo Concept

A raven silhouette — options for incorporating the tech identity:

- Graph nodes/edges subtly patterned into the wing feathers
- A raven carrying a glowing node in its talons
- A raven perched on a graph structure (like Odin's shoulder)
- Minimalist raven head profile with a single vector/node as the eye

---

## Color Palette Suggestions

- Primary: deep charcoal/black (raven)
- Accent: amber/gold (Odin's wisdom, the glowing knowledge)
- Background: dark navy or off-white

---

## Tagline Options

- "Memory for your data" (direct)
- "The raven remembers" (mythological)
- "Vector search + graph traversal for SQLite" (technical)
