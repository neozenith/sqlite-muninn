# muninn Examples

Interactive examples demonstrating the muninn SQLite extension. Each example
runs as a standalone Python script or as a Google Colab notebook.

## Vector Search

| Example | Description | Colab |
|---------|-------------|-------|
| [semantic_search](semantic_search/) | HNSW index with hand-crafted vectors — CREATE, INSERT, KNN, DELETE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/semantic_search/semantic_search.ipynb) |
| [text_embeddings](text_embeddings/) | GGUF embedding models → HNSW index → semantic search (MiniLM vs Nomic) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/text_embeddings/text_embeddings.ipynb) |

## Graph Traversal

| Example | Description | Colab |
|---------|-------------|-------|
| [social_network](social_network/) | BFS and DFS traversal on a clustered social graph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/social_network/social_network.ipynb) |
| [transit_routes](transit_routes/) | Shortest path — BFS (fewest hops) vs Dijkstra (minimum weight) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/transit_routes/transit_routes.ipynb) |
| [research_papers](research_papers/) | PageRank and connected components on a citation graph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/research_papers/research_papers.ipynb) |
| [movie_recommendations](movie_recommendations/) | Node2Vec embeddings from co-viewing patterns → HNSW similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/movie_recommendations/movie_recommendations.ipynb) |

## LLM (GGUF Chat Models)

| Example | Description | Colab |
|---------|-------------|-------|
| [llm_chat](llm_chat/) | Free-form and grammar-constrained chat completion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_chat/llm_chat.ipynb) |
| [llm_summarize](llm_summarize/) | Per-document and multi-document summarisation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_summarize/llm_summarize.ipynb) |
| [llm_extract](llm_extract/) | NER + RE extraction — muninn GGUF vs GLiNER2 benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_extract/llm_extract.ipynb) |
| [llm_tokenize](llm_tokenize/) | Tokenizer inspection — BERT WordPiece vs BPE side-by-side | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_tokenize/llm_tokenize.ipynb) |

## Running Locally

```bash
# Build the extension
make all

# Run any example as a standalone script
python examples/semantic_search/semantic_search.py
```

Most examples have zero Python dependencies beyond the muninn extension.
The LLM examples auto-download GGUF models on first run.
