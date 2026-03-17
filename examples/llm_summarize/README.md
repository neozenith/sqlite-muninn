# LLM Summarize — Text Summarisation via muninn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_summarize/llm_summarize.ipynb)

Demonstrates document summarisation using `muninn_summarize()` with a GGUF
chat model. Handles `<think>` blocks internally — thinking models reason first,
then the response is extracted and returned clean.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Per-document summary | `SELECT muninn_summarize('model', content) FROM docs` |
| Multi-document summary | `SELECT muninn_summarize('model', group_concat(content))` |
| SQL-native summarisation | Query results piped directly into summarisation |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
python examples/llm_summarize/llm_summarize.py
```

Model auto-downloads on first run (~0.5 GB).
