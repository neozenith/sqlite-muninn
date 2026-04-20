# sqlite-muninn

<div align="center">
    <img src="https://joshpeak.net/sqlite-muninn/assets/muninn_logo_transparent.png" alt="Muninn Raven Logo" width=480px/>
    <p><i>Odin's mythic <a href="https://en.wikipedia.org/wiki/Huginn_and_Muninn">raven of Memory</a>.</i></p>
</div>

```text
Huginn and Muninn fly each day over the wide world.
I fear for Huginn that he may not return,
yet I worry more for Muninn.

- Poetic Edda (Grimnismal, stanza 20)
```

_Odin fears losing Memory more than Thought._

This project aims to build **agentic memory** and **knowledge graph** primitives for sqlite as a native C extension. It is an advanced collection of knowledge graph primitives like Vector Similarity Search, HNSW Indexes, Graph database, Community Detection, Node2Vec capabilities and loading GGUF models via llama.cpp integration.

**[Documentation](https://neozenith.github.io/sqlite-muninn/)** | **[GitHub](https://github.com/neozenith/sqlite-muninn)**

| Package Index | Published Version | Downloads | 
|---|---|---|
| PyPI | [![PyPI](https://img.shields.io/pypi/v/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) | [![PyPI Downloads](https://img.shields.io/pypi/dm/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) |
| npm | [![npm](https://img.shields.io/npm/v/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) | [![npm Downloads](https://img.shields.io/npm/dm/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) |

## Features

- **HNSW Vector Index** &mdash; O(log N) approximate nearest neighbor search with incremental insert/delete
- **Graph Traversal** &mdash; BFS, DFS, shortest path, connected components, PageRank on any edge table, dbt syntax graph node selection.
- **`llama.cpp` native models**; Load and use GGUF LLM models natively in sqlite.
- **Centrality Measures** &mdash; Degree, betweenness (Brandes), and closeness centrality with weighted/temporal support
- **Community Detection** &mdash; Leiden algorithm for discovering graph communities with modularity scoring
- **Node2Vec** &mdash; Learn structural node embeddings from graph topology, store in HNSW for similarity search
- **Zero dependencies** &mdash; compiles to a single `.dylib`/`.so`/`.dll`
- **SIMD accelerated** &mdash; ARM NEON and x86 SSE distance functions

## Build

Requires SQLite development headers and a C11 compiler.

```bash
# macOS (Homebrew SQLite recommended)
brew install sqlite
make all

# Linux
sudo apt-get install libsqlite3-dev
make all

# Run tests
make test        # C unit tests
make test-python # Python integration tests
make test-all    # Both
```

## Quick Start

```sql
.load ./muninn

-- Create an HNSW vector index
CREATE VIRTUAL TABLE my_vectors USING hnsw_index(
    dimensions=384, metric='cosine', m=16, ef_construction=200
);

-- Insert vectors
INSERT INTO my_vectors (rowid, vector) VALUES (1, ?);  -- 384-dim float32 blob

-- KNN search
SELECT rowid, distance FROM my_vectors
WHERE vector MATCH ?query AND k = 10 AND ef_search = 64;

-- Graph traversal on any edge table
SELECT node, depth FROM graph_bfs
WHERE edge_table = 'friendships' AND src_col = 'user_a'
  AND dst_col = 'user_b' AND start_node = 'alice' AND max_depth = 3
  AND direction = 'both';

-- Connected components
SELECT node, component_id, component_size FROM graph_components
WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b';

-- PageRank
SELECT node, rank FROM graph_pagerank
WHERE edge_table = 'citations' AND src_col = 'citing' AND dst_col = 'cited'
  AND damping = 0.85 AND iterations = 20;

-- Betweenness centrality (find bridge nodes)
SELECT node, centrality FROM graph_betweenness
WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b'
  AND direction = 'both'
ORDER BY centrality DESC LIMIT 10;

-- Community detection (Leiden algorithm)
SELECT node, community_id, modularity FROM graph_leiden
WHERE edge_table = 'friendships' AND src_col = 'user_a' AND dst_col = 'user_b';

-- Learn structural embeddings from graph topology
SELECT node2vec_train(
    'friendships', 'user_a', 'user_b', 'my_vectors',
    64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 5
);
```

## Examples

Self-contained examples in the [`examples/`](examples/) directory:

| Example | Demonstrates |
|---------|-------------|
| [Semantic Search](examples/semantic_search/) | HNSW index, KNN queries, point lookup, delete |
| [Movie Recommendations](examples/movie_recommendations/) | Vector similarity for content-based recommendations |
| [Social Network](examples/social_network/) | Graph TVFs on a social graph (BFS, components, PageRank) |
| [Research Papers](examples/research_papers/) | Citation graph analysis with Node2Vec embeddings |
| [Transit Routes](examples/transit_routes/) | Shortest path and graph traversal on route networks |

```bash
make all
python examples/semantic_search/semantic_search.py
```

## Benchmarks

The project includes a comprehensive benchmark suite comparing muninn against other SQLite extensions across real-world workloads.

**Vector search** benchmarks compare against [sqlite-vector](https://github.com/nicepkg/sqlite-vector), [sqlite-vec](https://github.com/asg017/sqlite-vec), and [vectorlite](https://github.com/nicepkg/vectorlite) using 3 embedding models (MiniLM, MPNet, BGE-Large) and 2 text datasets (AG News, Wealth of Nations) at scales up to 250K vectors.

**Graph traversal** benchmarks compare muninn TVFs against recursive CTEs and [GraphQLite](https://github.com/nicepkg/graphqlite) on synthetic graphs (Erdos-Renyi, Barabasi-Albert) at scales up to 100K nodes.

Results include interactive Plotly charts for insert throughput, search latency, recall, database size, and tipping-point analysis. See the [full benchmark results](https://neozenith.github.io/sqlite-muninn/benchmarks/) on the documentation site.

```bash
make -C benchmarks help       # List all benchmark targets
make -C benchmarks analyze    # Generate charts and reports from existing results
```

## Project Structure

```
src/                  C11 source (extension entry point, HNSW, graph TVFs, Node2Vec)
test/                 C unit tests (custom minimal framework)
pytests/              Python integration tests (pytest)
examples/             Self-contained usage examples
benchmarks/
  scripts/            Benchmark runners and analysis scripts
  charts/             Plotly JSON chart specs (committed for docs site)
  results/            JSONL benchmark data (generated, not committed)
docs/                 MkDocs documentation source
```

## Documentation

Full documentation is published at **[neozenith.github.io/sqlite-muninn](https://neozenith.github.io/sqlite-muninn/)** via MkDocs Material with interactive Plotly charts.

```bash
make docs-serve    # Local dev server with live reload
make docs-build    # Build static site
```

## Research References

| Feature | Paper |
|---------|-------|
| HNSW | Malkov & Yashunin, TPAMI 2020 |
| MN-RU insert repair | arXiv:2407.07871, 2024 |
| Patience early termination | SISAP 2025 |
| Betweenness centrality | Brandes, J. Math. Sociol. 2001 |
| Leiden community detection | Traag, Waltman & van Eck, Sci. Rep. 2019 |
| Node2Vec | Grover & Leskovec, KDD 2016 |
| SGNS | Mikolov et al., 2013 |

## License

MIT. See [LICENSE](LICENSE).

