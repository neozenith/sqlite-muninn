# Graph Select

`graph_select` is a dbt-inspired node selector DSL for querying graph lineage — ancestors, descendants, depth-limited traversal, transitive closures, and set operations — in a compact text grammar. Built for dependency analysis, build-system lineage, data-pipeline impact, and dead-code detection.

If you know dbt's [node selection syntax](https://docs.getdbt.com/reference/node-selection/syntax), you already know most of this.

## Signature

Unusually for a muninn graph TVF, `graph_select` uses **positional** arguments (not `WHERE`-constraint syntax):

```sql
graph_select(
    edge_table TEXT,      -- source edge table
    src_col TEXT,         -- source (parent) column
    dst_col TEXT,         -- destination (child) column
    selector TEXT         -- selector DSL expression
) -> (node TEXT, depth INTEGER, direction TEXT)
```

Output columns:

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | A selected node ID |
| `depth` | INTEGER | Hop distance from the anchor node (0 for self) |
| `direction` | TEXT | `'self'`, `'ancestor'`, or `'descendant'` |

## Example graph

Every example below uses this dependency graph:

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

## Grammar

```
expression  := term ( SPACE term )*              -- union
term        := "not" atom                         -- complement
             | atom ( "," atom )*                 -- intersection
atom        := [ "@" ] depth_spec                 -- @ = transitive build closure
depth_spec  := [ INT "+" ] identifier [ "+" [ INT ] ]
```

Precedence, highest → lowest:

1. `@` (closure) and depth prefixes/suffixes (part of an atom)
2. `not` (unary complement)
3. `,` (binary intersection)
4. ` ` whitespace (n-ary union)

So `+A B+` is `(+A) ∪ (B+)`, and `+A,+B not C` is `((+A) ∩ (+B)) ∪ (not C)`.

## Operator table

| Syntax | Meaning | On the example graph |
|--------|---------|----------------------|
| `node` | Just the node | `C` → {C} |
| `+node` | Node + all ancestors | `+C` → {A, B, C} |
| `node+` | Node + all descendants | `C+` → {C, D, E, F} |
| `N+node` | Depth-limited ancestors (N hops up, inclusive) | `1+C` → {B, C} |
| `node+N` | Depth-limited descendants | `C+1` → {C, D, E} |
| `N+node+M` | Both directions, depth-limited | `1+C+1` → {B, C, D, E} |
| `+node+` | Unlimited both directions | `+C+` → {A, B, C, D, E, F} |
| `@node` | Transitive build closure (descendants + all their ancestors) | `@C` → {A, B, C, D, E, F, Y} |
| `A B` | Union | `D B` → {D, B} |
| `A,B` | Intersection | `+D,+E` → {C} (common ancestors) |
| `not A` | Complement (everything not in A) | `not C+` → all nodes except {C, D, E, F} |

## Recipes

### Ancestors — "what does X depend on?"

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

### Descendants — "what depends on X?"

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

### Transitive build closure — "what must rebuild if X changes?"

```sql
SELECT node FROM graph_select('deps', 'src', 'dst', '@C');
```

Returns descendants of C **plus all their ancestors** — the full dependency set needed for a clean rebuild. Equivalent to dbt's `@` selector.

```text
node
----
A
B
C
D
E
F
Y        -- ancestor of E, which is a descendant of C
```

### Depth-limited impact radius

```sql
-- Everything within 1 hop of C in both directions
SELECT node, depth, direction FROM graph_select('deps', 'src', 'dst', '1+C+1');
```

```text
node  depth  direction
----  -----  ---------
C     0      self
B     1      ancestor
D     1      descendant
E     1      descendant
```

### Common ancestors (intersection)

```sql
-- What is common upstream of D and E?
SELECT node FROM graph_select('deps', 'src', 'dst', '+D,+E');
```

```text
node
----
A
B
C
```

### Union

```sql
-- D's subgraph OR B's subgraph
SELECT DISTINCT node FROM graph_select('deps', 'src', 'dst', 'D+ B+');
```

### Complement — "everything unrelated to X"

```sql
-- Nodes that are not in C's subgraph (useful for dead-code analysis)
SELECT node FROM graph_select('deps', 'src', 'dst', 'not C+');
```

```text
node
----
A
B
Y
```

## Use-case patterns

### dbt-style data-lineage queries

When your `deps` table models a DAG of data models, `graph_select` answers the standard dbt questions:

| Question | Selector |
|----------|----------|
| "Run this model and everything downstream of it" | `model_name+` |
| "Test upstream dependencies of this model" | `+model_name` |
| "What's the full closure around this set of models?" | `@model_a @model_b` |
| "What depends on both A and B?" | `A+,B+` (intersection) |
| "Everything **except** the staging layer" | `not staging+` |

### Build-system impact analysis

Given an edge table representing source-file `#include` or `import` relations, `graph_select` tells you exactly which tests to re-run when a file changes: `changed_file+`.

### Knowledge-graph sub-graph extraction

For retrieval-augmented generation, you often want "the subgraph within K hops of a seed node." `N+seed+M` gives you that directly.

### Dead-code detection

Union all entry points, complement the result: `not (entry_point_1+ entry_point_2+ ...)`. Nodes in the complement are unreachable from any entry point.

## Performance

`graph_select` loads the graph into memory on each call (like all scan-on-query graph TVFs). For repeated selectors against the same graph, create a [`graph_adjacency`](api.md#graph_adjacency) virtual table first — future versions may teach `graph_select` to read from it directly.

## See also

- [API Reference — `graph_select`](api.md#graph_select)
- [dbt node selection syntax](https://docs.getdbt.com/reference/node-selection/syntax) — the DSL this selector is modeled after
- [API Reference — graph traversal TVFs](api.md#graph-traversal-tvfs) — `graph_bfs` / `graph_dfs` for unselected traversal
