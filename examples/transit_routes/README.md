# Transit Routes — Shortest Paths

Compare fewest-stops (BFS) vs fastest-time (Dijkstra) routing.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Unweighted shortest path | `AND weight_col IS NULL` (BFS, fewest hops) |
| Weighted shortest path | `AND weight_col = 'travel_time'` (Dijkstra, minimum total weight) |
| Path reconstruction | Output includes `node`, `distance`, `path_order` |

## Data

A transit network where the direct route is slow but the indirect route is fast:

```
  Central ──15min──→ North ──10min──→ Airport
    │                  ↑
    │ 5min          6min │
    ↓                  │
  South ──12min──→  East
    │
    │ 7min
    ↓
   West ──────7min──────→ Airport
```

## Run

```bash
make all
python examples/transit_routes/example.py
```

## Expected Output

| Metric | Fewest Stops | Fastest Time |
|--------|-------------|--------------|
| Path | Central → North → Airport | Central → South → West → Airport |
| Hops | 2 | 3 |
| Time | 25 min | 19 min |
