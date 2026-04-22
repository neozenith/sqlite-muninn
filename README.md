# sqlite-muninn

<div align="center">
    <img src="https://joshpeak.net/sqlite-muninn/assets/muninn_logo_transparent.png" alt="Muninn Raven Logo" width=480px/>
    <p><i>Odin's mythic <a href="https://en.wikipedia.org/wiki/Huginn_and_Muninn">raven of Memory</a>.</i></p>
</div>

This project aims to build **agentic memory** and **knowledge graph** primitives for sqlite as a native C extension and also made available for Python, Node.JS and WASM. 

It is **an advanced collection of knowledge graph primitives** like Vector Similarity Search, HNSW Indexes, Graph database, Community Detection, Node2Vec capabilities and loading GGUF models via llama.cpp integration.

```text
Huginn and Muninn fly each day over the wide world.
I fear for Huginn that he may not return,
yet I worry more for Muninn.

- Poetic Edda (Grimnismal, stanza 20)
```

_Odin fears losing Memory more than Thought._

Huginn and Muninn are the two ravens of Odin and their names translate to _Thoughts_ and _Memory_.


**[Documentation](https://neozenith.github.io/sqlite-muninn/)** | **[GitHub](https://github.com/neozenith/sqlite-muninn)**

| Package Index | Published Version | Downloads | 
|---|---|---|
| PyPI | [![PyPI](https://img.shields.io/pypi/v/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) | [![PyPI Downloads](https://img.shields.io/pypi/dm/sqlite-muninn.svg)](https://pypi.org/project/sqlite-muninn/) |
| npm | [![npm](https://img.shields.io/npm/v/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) | [![npm Downloads](https://img.shields.io/npm/dm/sqlite-muninn.svg)](https://www.npmjs.com/package/sqlite-muninn) |

## Features

- **HNSW Vector Index** - O(log N) approximate nearest neighbor search with incremental insert/delete
- **Graph Traversal** - BFS, DFS, shortest path, connected components, PageRank on any edge table, dbt syntax graph node selection.
- **`llama.cpp` native models**; Load and use GGUF LLM models natively in sqlite.
- **Centrality Measures** - Degree, betweenness (Brandes), and closeness centrality with weighted/temporal support
- **Community Detection** - Leiden algorithm for discovering graph communities with modularity scoring
- **Node2Vec** - Learn structural node embeddings from graph topology, store in HNSW for similarity search
- **Zero dependencies** - compiles to a single `.dylib`/`.so`/`.dll`
- **SIMD accelerated** - ARM NEON and x86 SSE distance functions


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

