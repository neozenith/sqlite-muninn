---
name: muninn-graph-select
description: >
  Writes dbt-inspired lineage queries using the graph_select TVF — ancestors,
  descendants, depth-limited traversal, transitive closures, and set operations
  on any SQLite edge table. Selector DSL supports +node (ancestors), node+
  (descendants), N+node+M (depth limits), @node (closure), space (union),
  comma (intersection), and "not" (complement). Use when the user mentions
  "graph_select", "dbt selector", "lineage query", "dependency graph",
  "upstream", "downstream", "ancestors", "descendants", "transitive closure",
  "impact analysis", "build closure", "dead code detection", or wants dbt-style
  node selection syntax on any SQLite edge table.
license: MIT
---

# muninn-graph-select — dbt-style lineage queries

`graph_select` is the one graph TVF in muninn that uses **positional arguments** instead of WHERE-constraint syntax. The reason: the selector is a single string DSL, not a column-by-column filter.

Useful for: dbt-style dependency analysis, build-system impact ("what rebuilds if X changes"), data-pipeline lineage, dead-code detection.

## Signature

```sql
graph_select(
    edge_table TEXT,    -- source edge table
    src_col    TEXT,    -- source (parent) column
    dst_col    TEXT,    -- destination (child) column
    selector   TEXT     -- DSL expression
) -> (node TEXT, depth INTEGER, direction TEXT)
```

Output `direction` is one of `'self'`, `'ancestor'`, `'descendant'`.

## Example graph

```
A   Y
 \ / \
  B   E
  |   |
  C   F
 / \
D   E

edges: A→B, Y→E, B→C, C→D, C→E, E→F
```

```sql
.load ./muninn

CREATE TABLE deps (src TEXT, dst TEXT);
INSERT INTO deps VALUES
  ('A','B'), ('Y','E'),
  ('B','C'),
  ('C','D'), ('C','E'),
  ('E','F');
```

## Selector grammar cheat sheet

| Syntax | Meaning | On the example graph |
|--------|---------|----------------------|
| `node` | Just the node | `C` → {C} |
| `+node` | Node + all ancestors | `+C` → {A, B, C} |
| `node+` | Node + all descendants | `C+` → {C, D, E, F} |
| `N+node` | Depth-limited ancestors (N hops up) | `1+C` → {B, C} |
| `node+N` | Depth-limited descendants | `C+1` → {C, D, E} |
| `N+node+M` | Both, depth-limited | `1+C+1` → {B, C, D, E} |
| `+node+` | Both, unlimited | `+C+` → {A, B, C, D, E, F} |
| `@node` | Transitive build closure (descendants + their ancestors) | `@C` → {A, B, C, D, E, F, Y} |
| `A B` | Union (space-separated) | `D B` → {D, B} |
| `A,B` | Intersection | `+D,+E` → {C} |
| `not A` | Complement | `not C+` → nodes not in {C, D, E, F} |

Precedence, high → low: atoms (`@`, depth prefixes/suffixes) > `not` > `,` > space.

## Common recipes

### Ancestors: "what does X depend on?"

```sql
SELECT node, depth FROM graph_select('deps', 'src', 'dst', '+C');
```

```text
node  depth
----  -----
C     0
B     1
A     2
```

### Descendants: "what depends on X?"

```sql
SELECT node, depth FROM graph_select('deps', 'src', 'dst', 'C+');
```

```text
node  depth
----  -----
C     0
D     1
E     1
F     2
```

### Build closure: "what must rebuild if I change X?"

`@X` = descendants of X, **plus** all ancestors of each descendant. This is exactly the set of nodes that could depend, transitively, on anything downstream of X.

```sql
SELECT node FROM graph_select('deps', 'src', 'dst', '@C');
-- {A, B, C, D, E, F, Y}
```

### Common ancestors (intersection)

```sql
SELECT node FROM graph_select('deps', 'src', 'dst', '+D,+E');
-- {A, B, C}   (nodes upstream of BOTH D and E)
```

### Depth-limited blast radius

```sql
-- 1 hop in each direction from C
SELECT node, depth, direction FROM graph_select('deps', 'src', 'dst', '1+C+1');
-- {B (ancestor, 1), C (self, 0), D (descendant, 1), E (descendant, 1)}
```

### Dead-code detection: "not reachable from any entrypoint"

```sql
-- Everything not descendant of root 'A'
SELECT node FROM graph_select('deps', 'src', 'dst', 'not A+');
```

### Union across multiple sources

```sql
-- Everything reachable from A or Y
SELECT DISTINCT node FROM graph_select('deps', 'src', 'dst', 'A+ Y+');
```

## Runtime variants

### Python

```python
import sqlite3
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)

# Note: graph_select args are positional, passed via ? placeholders
rows = db.execute(
  "SELECT node, depth FROM graph_select(?, ?, ?, ?)",
  ('deps', 'src', 'dst', '+C')
).fetchall()
```

### Node.js

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);

const rows = db.prepare(
  "SELECT node, depth, direction FROM graph_select(?, ?, ?, ?)"
).all("deps", "src", "dst", "@C");
```

## Common pitfalls

- **Do not use WHERE-constraint syntax here** — unlike `graph_bfs` / `graph_pagerank`, this TVF uses positional args.
- **Space in the selector is union, not concatenation** — `A B` is two nodes joined by union. Quoting the string is mandatory: `graph_select('deps', 'src', 'dst', 'A B')`.
- **`@X` includes Y that is upstream of a Y-descendant sibling** — the closure walks *both* directions from every descendant, so unrelated ancestor-branches appear. If you want strict dominance, post-filter.
- **Cycles** are handled — each node appears once with its minimum observed depth.
- **No weights** — `graph_select` is unweighted. For shortest weighted paths, see [muninn-graph-algorithms](../muninn-graph-algorithms/SKILL.md).

## When to use `graph_select` vs. `graph_bfs`

| Goal | Best TVF |
|------|----------|
| "All descendants of X up to depth 3" | `graph_bfs WHERE start_node = 'X' AND max_depth = 3` |
| "Ancestors AND descendants, set-combined with other queries" | `graph_select('@X')` and `graph_select('+A,+B')` |
| "Common ancestors of D and E" | `graph_select('+D,+E')` |
| "Traversal tree with parent pointers" | `graph_bfs` (provides `parent` column) |

`graph_select` is declarative set algebra. `graph_bfs`/`graph_dfs` are imperative traversals that surface parent pointers and explicit depth ordering.

## See also

- [muninn-graph-algorithms](../muninn-graph-algorithms/SKILL.md) — BFS/DFS/centrality/community
- [graph-select.md](../../docs/graph-select.md) — full grammar reference
- [dbt node selection](https://docs.getdbt.com/reference/node-selection/syntax) — the syntax this DSL mirrors
