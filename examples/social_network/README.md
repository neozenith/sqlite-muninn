# Social Network — Friends of Friends

Explore social connections using BFS and DFS graph traversal.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| BFS traversal | `SELECT node, depth, parent FROM graph_bfs WHERE ... AND direction = 'both'` |
| Depth-limited search | `AND max_depth = 2` |
| DFS traversal | `SELECT node, depth, parent FROM graph_dfs WHERE ...` |
| Undirected edges | `direction = 'both'` follows edges in both directions |

## Data

8 people in two clusters connected by a bridge:

```
  Alice ─── Bob          Eve ─── Frank
    │     ╱    │          │     ╱    │
  Carol ─ Dave ──────── Eve   Grace ─ Heidi
```

Edges are stored once (e.g., `Alice→Bob`) but `direction='both'` traverses both ways.

## Run

```bash
make all
python examples/social_network/example.py
```

## Expected Output

- **Depth 1**: Alice's direct friends (Bob, Carol)
- **Depth 2**: Friends-of-friends (Dave)
- **Depth 4**: Entire network (all 8 people)
- **DFS vs BFS**: Same nodes reached, different visit order
