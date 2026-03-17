# LLM Chat — Free-form and Structured Chat Completion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neozenith/sqlite-muninn/blob/main/examples/llm_chat/example.ipynb)

Demonstrates `muninn_chat()` for plain chat completion and grammar-constrained
structured JSON output via GBNF.

## What This Demonstrates

| Feature | SQL |
|---------|-----|
| Plain chat completion | `SELECT muninn_chat('model', 'prompt')` |
| Grammar-constrained output | `SELECT muninn_chat('model', 'prompt', 'grammar')` |
| Think block separation | Qwen3 `<think>...</think>` reasoning extracted from response |

## Prerequisites

Build the muninn extension — that's it. No `pip install` needed.

```bash
make all
```

## Run

```bash
python examples/llm_chat/example.py
```

Models auto-download on first run (~0.5–1.5 GB).
