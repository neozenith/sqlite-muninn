# LLM Tokenize — Inspect How GGUF Models Tokenize Text

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_tokenize/llm_tokenize.ipynb)

Compares tokenisation across an embed model (BERT/WordPiece) and a chat model
(BPE) using `muninn_tokenize()`, `muninn_tokenize_text()`, and
`muninn_token_count()`. All three functions work with any registered model type
via the unified model registry.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Token count | `SELECT muninn_token_count('model', 'text')` |
| Token IDs | `SELECT muninn_tokenize('model', 'text')` → JSON array |
| Token text pieces | `SELECT muninn_tokenize_text('model', 'text')` → JSON array |
| SQL-native token query | `SELECT * FROM json_each(muninn_tokenize('model', 'text'))` |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
python examples/llm_tokenize/llm_tokenize.py
```

Models auto-download on first run.
