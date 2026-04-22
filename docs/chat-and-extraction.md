# Chat and Extraction

Run a GGUF LLM — chat completion, summarization, NER, relation extraction — from inside SQLite. The extraction functions use grammar-constrained decoding, so JSON output is guaranteed well-formed.

## Load a chat model

Like embedding models, chat models live in a session-scoped registry:

```sql
.load ./muninn

INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B',
         muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');

SELECT name, n_ctx FROM temp.muninn_chat_models;
```

```text
name        n_ctx
----------  -----
Qwen3.5-4B  8192
```

Pass an optional second argument to override context length (capped at the model's training context):

```sql
INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B',
         muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf', 16384);
```

## Recommended GGUF chat models

| Model | Quant | File | Notes |
|-------|-------|------|-------|
| Qwen2.5-3B-Instruct | Q4_K_M | 1.9 GB | Fast, capable, no reasoning mode |
| Qwen3.5-4B-Instruct | Q4_K_M | 2.4 GB | Has `<think>` reasoning mode; use `skip_think=1` to bypass |
| Llama-3.2-3B-Instruct | Q4_K_M | 2.0 GB | Meta's small model, strong general instruction following |
| Phi-3.5-mini-instruct | Q4_K_M | 2.4 GB | Microsoft, strong on reasoning tasks |
| Mistral-7B-Instruct-v0.3 | Q4_K_M | 4.1 GB | Well-known baseline |

Larger models give better extraction accuracy but slower per-token generation. For production NER/RE, the sweet spot is 3–7B parameters at Q4_K_M quantization on Metal.

## `muninn_chat` — free-form generation

```sql
SELECT muninn_chat(
  'Qwen3.5-4B',
  'Explain HNSW indexes in one sentence.',
  NULL,          -- no grammar
  64,            -- max_tokens
  NULL,          -- default system prompt
  1              -- skip_think = 1 (strip Qwen3.5 <think> blocks)
);
```

```text
HNSW is a graph-based approximate nearest-neighbor index that organizes vectors
into a hierarchy of navigable small-world graphs for O(log N) search.
```

Full signature:

```sql
muninn_chat(
    model_name TEXT,
    prompt TEXT,
    grammar TEXT       = NULL,    -- GBNF expression, constrains token sampling
    max_tokens INTEGER = n_ctx,
    system_prompt TEXT = NULL,
    skip_think INTEGER = 0        -- 1 → inject closed <think></think>, bypassing reasoning
) -> TEXT
```

### GBNF grammar example

```sql
-- Force output to one of three sentiment labels
SELECT muninn_chat(
  'Qwen3.5-4B',
  'Sentiment of: "The food was incredible, but the service was terrible."',
  'root ::= "positive" | "negative" | "neutral"',
  8
);
```

```text
negative
```

GBNF grammars guarantee the output shape regardless of the model — the sampler rejects any token that would violate the grammar. For more on GBNF, see [llama.cpp docs](https://github.com/ggerganov/llama.cpp/tree/master/grammars).

---

## `muninn_extract_entities` — named entity recognition

### Supervised mode (labels provided)

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

### Unsupervised mode (open extraction)

Omit the labels argument to let the model propose its own types:

```sql
SELECT muninn_extract_entities(
  'Qwen3.5-4B',
  'Elon Musk founded Tesla in 2003 in Palo Alto.'
);
```

```text
{"entities":[
  {"text":"Elon Musk","type":"person","score":0.97},
  {"text":"Tesla","type":"company","score":0.96},
  {"text":"2003","type":"year","score":0.93},
  ...
]}
```

Unsupervised types are the model's own taxonomy — useful for exploratory analysis, less useful when types must match a downstream schema.

### Working with the JSON result

The returned TEXT has SQLite subtype `'J'`, so `json_each()` works directly:

```sql
WITH ner AS (
  SELECT muninn_extract_entities('Qwen3.5-4B',
           'Elon Musk founded Tesla in 2003.',
           'person,organization,date') AS result
)
SELECT value ->> 'text' AS entity,
       value ->> 'type' AS type
  FROM ner, json_each(ner.result, '$.entities');
```

```text
entity     type
---------  ------------
Elon Musk  person
Tesla      organization
2003       date
```

---

## `muninn_extract_relations` — relation extraction

### Supervised — entities already identified

```sql
SELECT muninn_extract_relations(
  'Qwen3.5-4B',
  'Tesla acquired Maxwell Technologies in 2019 for $218 million.',
  json('[{"text":"Tesla","type":"organization"},
         {"text":"Maxwell Technologies","type":"organization"}]')
);
```

```text
{"relations":[
  {"head":"Tesla","rel":"acquired","tail":"Maxwell Technologies","score":0.98}
]}
```

### Unsupervised — discover both entities and relations

```sql
SELECT muninn_extract_relations(
  'Qwen3.5-4B',
  'Tesla acquired Maxwell Technologies in 2019 for $218 million.'
);
```

---

## `muninn_extract_ner_re` — combined NER + RE

One generation covers both tasks when the entity spans inform the relation extraction:

```sql
SELECT muninn_extract_ner_re(
  'Qwen3.5-4B',
  'Tesla acquired Maxwell Technologies in 2019.',
  'person,organization',           -- entity labels
  'acquired,founded,employed_by'   -- relation labels
);
```

```text
{
  "entities":[
    {"text":"Tesla","type":"organization","score":0.97},
    {"text":"Maxwell Technologies","type":"organization","score":0.95}
  ],
  "relations":[
    {"head":"Tesla","rel":"acquired","tail":"Maxwell Technologies","score":0.98}
  ]
}
```

---

## Batch variants

Single-text extraction costs one full prompt evaluation per row. For bulk processing, the batch variants submit multiple prompts as multi-sequence `llama_batch` calls, sharing KV cache setup:

```sql
SELECT muninn_extract_entities_batch(
  'Qwen3.5-4B',
  json('["Tesla was founded by Elon Musk.",
        "SpaceX launched Falcon 9 in 2010.",
        "Apple released iPhone in 2007."]'),
  'person,organization,date',
  4                                  -- batch_size, max 8
);
```

```text
[
  {"entities":[{"text":"Tesla","type":"organization","score":0.97},
               {"text":"Elon Musk","type":"person","score":0.98}]},
  {"entities":[{"text":"SpaceX","type":"organization","score":0.96},
               {"text":"Falcon 9","type":"organization","score":0.94},
               {"text":"2010","type":"date","score":0.95}]},
  {"entities":[{"text":"Apple","type":"organization","score":0.97},
               {"text":"iPhone","type":"organization","score":0.92},
               {"text":"2007","type":"date","score":0.96}]}
]
```

Batch variants return a JSON **array** of per-text results in input order.

| Function | Output schema |
|----------|--------------|
| `muninn_extract_entities_batch` | `[{"entities":[...]}, ...]` |
| `muninn_extract_ner_re_batch` | `[{"entities":[...], "relations":[...]}, ...]` |

Batch size caps at 8 — larger batches degrade per-item quality as KV-cache contention increases.

---

## `muninn_summarize`

```sql
SELECT muninn_summarize(
  'Qwen3.5-4B',
  'Long article text goes here...',
  256                -- max_tokens for the summary
);
```

Qwen3.5 `<think>` reasoning blocks are stripped from the output automatically — you get only the final summary text.

---

## End-to-end: building a knowledge graph from text

```sql
.load ./muninn

INSERT INTO temp.muninn_chat_models(name, model)
  SELECT 'Qwen3.5-4B',
         muninn_chat_model('models/Qwen3.5-4B-Instruct.Q4_K_M.gguf');

-- Source documents
CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT);
INSERT INTO docs(content) VALUES
  ('Tesla acquired Maxwell Technologies in 2019 for $218 million.'),
  ('SpaceX launched Falcon 9 in 2010, founded by Elon Musk.'),
  ('Apple released iPhone in 2007; Steve Jobs unveiled it on stage.');

-- Extract NER + RE per document (batched)
CREATE TEMP TABLE extractions AS
SELECT id,
       value AS extraction
  FROM docs,
       json_each(
         muninn_extract_ner_re_batch(
           'Qwen3.5-4B',
           (SELECT json_group_array(content) FROM docs),
           'person,organization,date',
           'acquired,founded,released,employed_by',
           4
         )
       );

-- Materialize entities and relations as proper tables
CREATE TABLE entities AS
SELECT DISTINCT
       e.value ->> 'text' AS name,
       e.value ->> 'type' AS type
  FROM extractions, json_each(extractions.extraction, '$.entities') e;

CREATE TABLE relations AS
SELECT r.value ->> 'head' AS src,
       r.value ->> 'rel'  AS relation,
       r.value ->> 'tail' AS dst
  FROM extractions, json_each(extractions.extraction, '$.relations') r;

-- Now run any graph TVF over the extracted relations
SELECT node, centrality FROM graph_node_betweenness
  WHERE edge_table = 'relations' AND src_col = 'src' AND dst_col = 'dst'
    AND direction = 'both'
  ORDER BY centrality DESC LIMIT 5;
```

This is the recipe used by the [GraphRAG Cookbook](graphrag-cookbook.md) — from raw text to a queryable knowledge graph in one SQL script.

---

## Performance notes

| Operation | Approx throughput on M1 Pro, Qwen3.5-4B Q4_K_M, Metal |
|-----------|------------------------------------------------------|
| `muninn_chat` | ~60 tokens/sec |
| `muninn_extract_entities` (single) | ~1 text/sec for short passages |
| `muninn_extract_entities_batch` (size 4) | ~2.5× speedup over singles |
| `muninn_extract_ner_re_batch` (size 4) | ~2× speedup over two separate calls |

These numbers scale roughly linearly with model size and inversely with quantization precision.

## See also

- [API Reference — LLM chat and extraction](api.md#llm-chat-and-extraction)
- [Entity Resolution](entity-resolution.md) — combines extraction + clustering + LLM borderline refinement
- [GraphRAG Cookbook](graphrag-cookbook.md) — extraction + graph retrieval end-to-end
