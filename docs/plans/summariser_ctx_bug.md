# Chat-model context auto-trim silently cripples summarisation

> Handoff note (2026-06-02). Found while building a hierarchical summariser on top
> of `muninn_chat` (consumer repo: `claude-code-sessions`). Not a crash — a *silent*
> behaviour that cost several multi-hour benchmark runs before we traced it.

## TL;DR

`muninn_chat_model(path)` (no explicit `n_ctx`) **auto-trims** the loaded context to
`max(8192, n_ctx_train / 8)`. For summarisation — where prompts are *large by nature*
(whole sessions, multi-document merges) — this is the wrong default, and it is
**inverted**: the larger a model's native training context (i.e. the more capable it
is at long context), the *smaller* the fraction it is given. It is also **silent** at
the default log level, so downstream only sees `prompt (N tokens) exceeds context (M)`
with no hint that `M` was self-imposed at 1/8 of the model's real capability.

**Ask:** for summarisation-class workloads, don't auto-trim by default — or at minimum
log loudly when trimming. Details + recommendations below.

## How it manifested

Consumer workload: summarise ~580 real sessions, then roll them up across a scope
hierarchy with a re-grounding merge that injects source excerpts. Measured prompt
sizes (LLM tokens):

- session-extraction prompts: median ~580, p95 ~21k, tail to ~250k;
- re-ground merge prompts: median ~10k, p95 ~38k, **max ~74k**.

Models were registered with `INSERT INTO temp.muninn_chat_models(name, model) SELECT
?, muninn_chat_model(?)` — i.e. **no `n_ctx`**. Result:

| Model | `n_ctx_train` | Auto-trimmed `n_ctx` | Outcome |
|-------|--------------:|---------------------:|---------|
| Llama-3.1-8B | 131072 (128K) | **16384** (= 128K/8) | merges (28–74k) overflow → cells abort, 0 rollups |
| Qwen3.5-2B   | 262144 (256K) | **32768** (= 256K/8) | merges (37–39k) overflow → reground unusable |

We initially read these as *model/strategy* failures ("reground doesn't scale", "Llama
fails even on 128k") and nearly recorded that as a benchmark verdict. It was neither —
the context was trimmed to 1/8 and never approached the model's real window.

## Root cause

`src/llama_chat.c`, `load_chat_model()` (≈ lines 66–75):

```c
/* Dynamic context: max(DEFAULT_N_CTX, train_ctx / 8).
 * Gives 256K models 32K context, 128K models 16K, while never going below 8K.
 * Must also accommodate batch_size × (prompt + output) tokens for batch inference. */
int n_ctx_train = (int)llama_model_n_ctx_train(model);
if (n_ctx <= 0) {
    int dynamic_ctx = n_ctx_train > 0 ? n_ctx_train / 8 : 0;
    n_ctx = dynamic_ctx > DEFAULT_N_CTX ? dynamic_ctx : DEFAULT_N_CTX;   /* DEFAULT_N_CTX = 8192 */
}
if (n_ctx_train > 0 && n_ctx > n_ctx_train)
    n_ctx = n_ctx_train;
...
cparams.n_ctx   = (uint32_t)n_ctx;
cparams.n_batch = (uint32_t)n_ctx;   /* context doubles as batch size */
```

Two things make this bite summarisation specifically:

1. **The /8 trim is tuned for interactive/batch chat, not summarisation.** Chat turns are
   small; the comment even notes the default must "accommodate `batch_size × (prompt +
   output)` for batch inference", and `n_batch = n_ctx`, so a big context costs KV memory
   *twice* (context + batch). The conservative /8 is a memory-safety choice for the batch
   path. Summarisation is single-sequence with deliberately *large* prompts — the opposite
   profile — so the safe-for-batch default is exactly wrong here.
2. **It is inverted vs capability.** `train/8` means the model with the biggest native
   window gets the smallest *fraction*: 128K→16K, 256K→32K. You reach for a long-context
   model precisely to fit big prompts, and the loader quietly takes that away.

And it is **silent**: nothing is logged at default level when the context is reduced, so
the only symptom is a later `exceeds context (16384)` that looks like a hard model limit.

## Recommendations (sqlite-muninn)

1. **Don't auto-trim chat-model context by default.** Default to `n_ctx_train`
   (llama.cpp's own behaviour), or gate the /8 trim behind an opt-in
   (`MUNINN_CHAT_CTX_TRIM=1` or a build flag). Summarisation users want the real window.
2. **If a memory-safety cap stays, never trim silently.** Log at default level, e.g.
   `muninn_chat_model: context auto-reduced 131072 -> 16384 (train/8 default); pass
   muninn_chat_model(path, n_ctx) to override`. A one-line warning would have saved hours.
3. **Decouple `n_batch` from `n_ctx`.** `cparams.n_batch = n_ctx` is what makes large
   contexts doubly memory-hungry and is the likely reason the /8 exists. Bounding
   `n_batch` independently (e.g. `min(n_ctx, 2048)` for single-seq decode) would let
   `n_ctx` default high without the batch-memory blowup.
4. **Disambiguate the `n_ctx` column vs the function arg.** The vtable exposes an `n_ctx`
   column, which *looks* like it sets context — but it is post-load metadata; only
   `muninn_chat_model(path, n_ctx)` (2nd arg) sizes the context. We lost a run inserting
   into the column and assuming it applied. Either make the column authoritative (apply at
   load) or reject/warn when an inserted `n_ctx` differs from the loaded value.
5. **Keep the `n_ctx` column as the verification path** — it does correctly reflect
   `out->n_ctx`, so `SELECT n_ctx FROM temp.muninn_chat_models` is a good post-load check.
   Worth documenting as the way to confirm the loaded window.

## Consumer-side fix (already applied in claude-code-sessions, for reference)

- Register with the explicit 2nd arg:
  `INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?, ?)`
  with params `(name, path, n_ctx)`.
- **Size `n_ctx` from measured prompt-token distributions, not the model max.** A token
  census over the real corpus put p99 merge ≈ 54k and max ≈ 74k, so 64k (65536) holds
  ~99–100% of both extraction prompts and merges; 128k added <1% at ~2× KV RAM. We set
  the default to 65536.
- **Verify before long runs:** load the model, then `SELECT n_ctx FROM
  temp.muninn_chat_models` and assert it equals the request.

## A note on the opposite over-correction ("just use 256K")

The fix is **not** "max out the context". Going to 256K because a model supports it would
waste large amounts of KV-cache RAM (compounded by `n_batch = n_ctx`) for ~zero benefit
beyond ~64–96K on this workload. The lesson is to size context to the *measured* prompt
distribution — frugally — and to make the default neither silently tiny nor needlessly huge.
