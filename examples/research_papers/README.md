# Research Papers — Citation Analysis

Discover research communities and influential papers using graph algorithms.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Connected components | `SELECT node, component_id, component_size FROM graph_components` |
| PageRank | `SELECT node, rank FROM graph_pagerank WHERE ... AND damping = 0.85` |

## Data

12 papers forming 3 disconnected citation clusters:

- **ML cluster** (5 papers) — ML-Survey is the hub (cited by 4 papers)
- **DB cluster** (4 papers) — DB-Indexing is the hub (cited by 3 papers)
- **NLP cluster** (3 papers) — small group, no dominant hub

## Run

```bash
make all
python examples/research_papers/example.py
```

## Expected Output

- `graph_components` identifies exactly 3 clusters with sizes 5, 4, 3
- `graph_pagerank` ranks ML-Survey highest (most incoming citations)
- PageRank values sum to approximately 1.0

## How PageRank Works Here

PageRank models citations as "votes of importance." A paper cited by many others gets a higher rank. The `damping` factor (0.85) means there's a 15% chance of randomly jumping to any paper, preventing rank from concentrating too heavily in one place.
