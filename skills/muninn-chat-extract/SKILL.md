---
name: muninn-chat-extract
description: >
  Runs GGUF chat models and structured extraction (NER, relation extraction,
  combined NER+RE, summarization) inside SQLite via muninn_chat(),
  muninn_extract_entities(), muninn_extract_relations(), muninn_extract_ner_re(),
  muninn_summarize(), and their _batch variants. Supports supervised
  (labels provided) and unsupervised (open-extraction) modes, and GBNF
  grammar-constrained JSON output. Use when the user mentions "muninn_chat",
  "named entity recognition", "NER", "relation extraction", "RE", "LLM in SQL",
  "GGUF chat model", "Qwen3.5", "Qwen 3.5", "GBNF grammar", "grammar-constrained",
  "summarization in SQLite", "structured extraction", or wants to run an
  instruction-tuned model inside SQLite.
license: MIT
---

# muninn-chat-extract — LLM chat and structured extraction in SQL

`muninn_chat` and the `muninn_extract_*` family run llama.cpp with GBNF grammar sampling to produce either free-form text or guaranteed well-formed JSON. Metal GPU on macOS; CPU on Linux/Windows/WASM.

## Step 1 — Register a chat model

```bash
# Qwen3.5-4B-Instruct at Q4_K_M is a good general-purpose default (~2.6 GB)
mkdir -p models
# Download from the Qwen Hugging Face org page, e.g.:
#   Qwen/Qwen3.5-4B-Instruct-GGUF/Qwen3.5-4B-Instruct.Q4_K_M.gguf
```

```sql
.load ./muninn

INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');
```

`muninn_chat_model(path, n_ctx?)` returns an opaque handle — only useful as an INSERT value into `temp.muninn_chat_models`. Default `n_ctx = max(8192, train_ctx / 8)`, capped at the model's training context.

## Free-form chat

```sql
SELECT muninn_chat(
  'Qwen3.5-4B',
  'Explain SQLite virtual tables in one sentence.',
  NULL,        -- grammar: NULL = unconstrained
  64,          -- max_tokens
  'You are a terse technical writer.',  -- system_prompt
  0            -- skip_think: 0 = allow <think> reasoning blocks (Qwen3.5)
);
```

### GBNF grammar constraints

```sql
-- Guaranteed JSON object with a single "greeting" string field
SELECT muninn_chat(
  'Qwen3.5-4B',
  'Greet the user.',
  'root ::= "{\"greeting\":\"" [a-zA-Z ]+ "\"}"',
  64
);
```

```text
{"greeting":"Hello there"}
```

Grammar sampler rejects invalid tokens at each step — outputs parse or the call raises.

## Supervised NER — `muninn_extract_entities`

```sql
SELECT muninn_extract_entities(
  'Qwen3.5-4B',
  'Elon Musk founded Tesla in 2003 in Palo Alto.',
  'person,organization,date,location'
);
```

```text
{"entities":[
  {"text":"Elon Musk","type":"person","score":0.98},
  {"text":"Tesla","type":"organization","score":0.97},
  {"text":"2003","type":"date","score":0.95},
  {"text":"Palo Alto","type":"location","score":0.94}
]}
```

The return has SQLite subtype `'J'` — `json_each()` works directly without `json(...)` wrapping:

```sql
SELECT value ->> 'text', value ->> 'type'
FROM json_each(
  muninn_extract_entities('Qwen3.5-4B', 'Elon Musk founded Tesla.', 'person,org'),
  '$.entities'
);
```

## Unsupervised NER (open-label)

Omit the `labels` argument to let the model choose its own type taxonomy.

```sql
SELECT muninn_extract_entities('Qwen3.5-4B', 'Elon Musk founded Tesla in 2003.');
```

Useful for exploratory ingestion where you don't know the domain entity types upfront.

## Relation extraction

```sql
SELECT muninn_extract_relations(
  'Qwen3.5-4B',
  'Elon Musk founded Tesla in 2003.',
  '[{"text":"Elon Musk","type":"person"},{"text":"Tesla","type":"organization"}]'
);
```

```text
{"relations":[
  {"head":"Elon Musk","rel":"founded","tail":"Tesla","score":0.96}
]}
```

Supervised (entities provided) gives best precision. Omit `entities_json` for joint entity + relation discovery.

## Combined NER + RE (single pass)

```sql
SELECT muninn_extract_ner_re(
  'Qwen3.5-4B',
  'SpaceX, founded by Elon Musk in 2002, launched Falcon 9 in 2010.',
  'person,organization,date,product',
  'founded,launched'
);
```

```text
{
  "entities":[{"text":"SpaceX","type":"organization"}, ...],
  "relations":[{"head":"Elon Musk","rel":"founded","tail":"SpaceX"}, ...]
}
```

Single generation shares context across NER and RE — cheaper than two separate calls for the same input.

## Batch extraction (multi-sequence via `llama_batch`)

```sql
SELECT muninn_extract_entities_batch(
  'Qwen3.5-4B',
  json('["Apple released iPhone in 2007.","Steve Jobs unveiled it at Macworld."]'),
  'person,organization,product,date',
  4    -- batch_size (max 8)
);
```

Returns a JSON array of per-text NER results. Each prompt gets a unique `seq_id` in the KV cache, processed concurrently. Use when you have 4-8 short inputs and want to amortize GPU dispatch overhead.

## Abstractive summarization

```sql
SELECT muninn_summarize(
  'Qwen3.5-4B',
  'SQLite is a C-language library that implements a small, fast, self-contained...',
  128   -- max_tokens
);
```

Qwen3.5 `<think>` reasoning blocks are stripped from the output automatically.

## Runtime variants

### Python

```python
import sqlite3, json
import sqlite_muninn

db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_muninn.load(db)

db.execute("""
  INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf')
""")

raw = db.execute("""
  SELECT muninn_extract_entities(
    'Qwen3.5-4B',
    'Tim Cook leads Apple in Cupertino.',
    'person,organization,location'
  )
""").fetchone()[0]

print(json.loads(raw))
```

### Node.js

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);

db.exec(`
  INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B', muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');
`);

const result = db.prepare(`
  SELECT muninn_extract_ner_re(?, ?, ?, ?) AS r
`).get("Qwen3.5-4B", "SpaceX launched Falcon 9 in 2010.", "org,product,date", "launched");

console.log(JSON.parse(result.r));
```

## Common pitfalls

- **Error: "model not registered"** — register with `INSERT INTO temp.muninn_chat_models(...)` before calling `muninn_chat` / `muninn_extract_*`. Models are session-scoped.
- **Long wall-clock on first call** — llama.cpp warms up the KV cache on first prompt. Subsequent calls are much faster.
- **Empty `entities` array on long text** — text exceeds `n_ctx`. Either chunk before calling, or increase `n_ctx` when loading the model (but watch GPU memory).
- **`<think>` tags in output** — set `skip_think = 1` to inject closed `<think></think>` tokens and bypass Qwen3.5 reasoning for faster extraction.
- **WASM CPU too slow for real-time extraction** — WASM is fine for demos; for production-quality LLM extraction, use native Python/Node.js/CLI.

## Related functions

| Function | Purpose |
|----------|---------|
| `muninn_chat_model(path, n_ctx?)` | Load GGUF chat model |
| `muninn_chat(model, prompt, grammar?, max_tokens?, system?, skip_think?)` | Free-form generation |
| `muninn_extract_entities(model, text, labels?, skip_think?)` | NER with JSON subtype `'J'` |
| `muninn_extract_relations(model, text, entities_json?, skip_think?)` | RE |
| `muninn_extract_ner_re(model, text, ent_labels?, rel_labels?, skip_think?)` | Joint NER+RE |
| `muninn_extract_entities_batch(model, texts_json, labels?, batch_size?)` | Multi-sequence NER |
| `muninn_extract_ner_re_batch(model, texts_json, ent_labels?, rel_labels?, batch_size?)` | Multi-sequence NER+RE |
| `muninn_summarize(model, text, max_tokens?)` | Abstractive summary |
| `muninn_extract_er(...)` | Entity resolution pipeline (see muninn-graphrag) |

## See also

- [muninn-graphrag](../muninn-graphrag/SKILL.md) — compose extraction + embeddings + graph into full retrieval
- [muninn-embed-text](../muninn-embed-text/SKILL.md) — embedding counterpart
- [chat-and-extraction.md](../../docs/chat-and-extraction.md) — full reference with grammar authoring
