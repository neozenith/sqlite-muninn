/*
 * llama_chat.c — GGUF chat/completion inference via llama.cpp
 *
 * Autoregressive text generation with optional GBNF grammar constraints.
 * Mirrors the llama_embed.c model registry pattern but uses a separate
 * registry for chat models (different context configuration: no pooling,
 * with sampler chain, larger default context window).
 *
 * KV cache is cleared via llama_memory_clear() between SQL calls so
 * each muninn_chat() invocation starts with a fresh context.
 */

#include "llama_chat.h"
#include "llama_common.h"
#include "llama_constants.h"

SQLITE_EXTENSION_INIT3

#include <llama.h>
#include <yyjson.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ──────────────────────────────────────────────────────────────────
 * Configuration
 * ────────────────────────────────────────────────────────────── */

#define DEFAULT_N_CTX 8192
#define MAX_BATCH_SEQS 8
#define DEFAULT_BATCH_SIZE 4

/* ──────────────────────────────────────────────────────────────────
 * Model Loading
 * ────────────────────────────────────────────────────────────── */

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    int n_ctx;
} LoadedChatModel;

static const char *CHAT_MODEL_PTR_TYPE = "muninn_chat_model";

static int load_chat_model(const char *path, int n_ctx, LoadedChatModel *out, char *errbuf, int errbuf_sz) {
    muninn_ensure_backend();

    struct llama_model_params mparams = llama_model_default_params();
#ifndef MUNINN_DEFAULT_GPU_LAYERS
#define MUNINN_DEFAULT_GPU_LAYERS 0
#endif
    int ngl = MUNINN_DEFAULT_GPU_LAYERS;
    const char *ngl_env = getenv("MUNINN_GPU_LAYERS");
    if (ngl_env)
        ngl = atoi(ngl_env);
    mparams.n_gpu_layers = ngl;
    mparams.use_mmap = 1;

    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (!model) {
        snprintf(errbuf, errbuf_sz, "muninn_chat_model: failed to load '%s'", path);
        return -1;
    }

    /* Dynamic context: max(DEFAULT_N_CTX, train_ctx / 8).
     * Gives 256K models 32K context, 128K models 16K, while never going below 8K.
     * Must also accommodate batch_size × (prompt + output) tokens for batch inference. */
    int n_ctx_train = (int)llama_model_n_ctx_train(model);
    if (n_ctx <= 0) {
        int dynamic_ctx = n_ctx_train > 0 ? n_ctx_train / 8 : 0;
        n_ctx = dynamic_ctx > DEFAULT_N_CTX ? dynamic_ctx : DEFAULT_N_CTX;
    }
    if (n_ctx_train > 0 && n_ctx > n_ctx_train)
        n_ctx = n_ctx_train;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)n_ctx;
    cparams.n_batch = (uint32_t)n_ctx;
    cparams.n_ubatch = 64;     /* Smaller ubatch = better GPU occupancy for small decode batches */
    cparams.embeddings = 0;    /* Chat mode: no pooled embeddings */
    cparams.kv_unified = true; /* Single KV buffer across sequences = better Metal coalescing */

    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.n_seq_max = MAX_BATCH_SEQS; /* Support batched multi-sequence generation */

    /* AUTO lets llama.cpp choose the best attention implementation per-device. */

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        snprintf(errbuf, errbuf_sz, "muninn_chat_model: failed to create context for '%s'", path);
        return -1;
    }

    out->model = model;
    out->ctx = ctx;
    out->n_ctx = n_ctx;
    return 0;
}

/* ──────────────────────────────────────────────────────────────────
 * Core Generation Function
 *
 * Clears KV cache, tokenizes the formatted prompt, decodes prompt
 * batch, then runs autoregressive sampling until EOG or max_tokens.
 * ────────────────────────────────────────────────────────────── */

static int chat_generate(MuninnModelEntry *me, const char *prompt, const char *grammar_gbnf, int max_tokens,
                         char **out_text, int *out_len, char *errbuf, int errbuf_sz) {
    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);

    /* Clear KV cache for a fresh start */
    llama_memory_clear(llama_get_memory(me->ctx), 1);

    /* Tokenize prompt (two-pass: measure then fill) */
    int n_prompt = llama_tokenize(vocab, prompt, (int)strlen(prompt), NULL, 0, 1, 1);
    if (n_prompt < 0)
        n_prompt = -n_prompt;
    if (n_prompt == 0) {
        snprintf(errbuf, errbuf_sz, "muninn_chat: empty prompt after tokenization");
        return -1;
    }
    if (n_prompt > me->n_ctx - 4) {
        snprintf(errbuf, errbuf_sz, "muninn_chat: prompt (%d tokens) exceeds context (%d)", n_prompt, me->n_ctx);
        return -1;
    }

    llama_token *tokens = (llama_token *)malloc((size_t)n_prompt * sizeof(llama_token));
    if (!tokens) {
        snprintf(errbuf, errbuf_sz, "muninn_chat: OOM allocating prompt tokens");
        return -1;
    }
    int actual = llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens, n_prompt, 1, 1);
    if (actual < 0) {
        free(tokens);
        snprintf(errbuf, errbuf_sz, "muninn_chat: tokenization failed");
        return -1;
    }

    /* Create sampler chain: [grammar] + greedy (temp=0 deterministic) */
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (grammar_gbnf && grammar_gbnf[0]) {
        llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar_gbnf, "root"));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    /* Decode prompt batch */
    struct llama_batch batch = llama_batch_get_one(tokens, actual);
    int rc = llama_decode(me->ctx, batch);
    free(tokens);
    if (rc != 0) {
        llama_sampler_free(smpl);
        snprintf(errbuf, errbuf_sz, "muninn_chat: prompt decode failed (rc=%d)", rc);
        return -1;
    }

    /* Allocate output buffer (generous: avg ~4 bytes/token, with headroom) */
    int buf_cap = max_tokens * 16 + 256;
    char *buf = (char *)malloc((size_t)buf_cap);
    if (!buf) {
        llama_sampler_free(smpl);
        snprintf(errbuf, errbuf_sz, "muninn_chat: OOM allocating output buffer");
        return -1;
    }
    int buf_len = 0;

    /* Autoregressive generation loop */
    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(smpl, me->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token))
            break;

        /* Convert token to text piece */
        char piece[256];
        int piece_len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, 1);
        if (piece_len <= 0)
            continue;

        /* Grow buffer if needed */
        if (buf_len + piece_len + 1 > buf_cap) {
            buf_cap = buf_cap * 2 + piece_len;
            char *new_buf = (char *)realloc(buf, (size_t)buf_cap);
            if (!new_buf) {
                free(buf);
                llama_sampler_free(smpl);
                snprintf(errbuf, errbuf_sz, "muninn_chat: OOM growing output buffer");
                return -1;
            }
            buf = new_buf;
        }
        memcpy(buf + buf_len, piece, (size_t)piece_len);
        buf_len += piece_len;

        /* Decode single token for next iteration */
        batch = llama_batch_get_one(&new_token, 1);
        rc = llama_decode(me->ctx, batch);
        if (rc != 0)
            break;
    }

    buf[buf_len] = '\0';
    llama_sampler_free(smpl);

    *out_text = buf;
    *out_len = buf_len;
    return 0;
}

/* ──────────────────────────────────────────────────────────────────
 * Batched Generation
 *
 * Process N independent prompts in parallel through one KV cache.
 * Each prompt is assigned a unique sequence ID. Prompt tokens are
 * decoded in a single llama_decode() call, then autoregressive
 * generation proceeds with one token per active sequence per step.
 *
 * This gives ~2-3x speedup at N=4 because the GPU/CPU computes
 * all N next-tokens simultaneously in one forward pass.
 * ────────────────────────────────────────────────────────────── */

typedef struct {
    char *text; /* Generated text (heap-allocated, caller frees) */
    int len;    /* Text length in bytes */
    int done;   /* 1 = finished (EOG, error, or skipped) */
} BatchSlot;

static int chat_generate_batch(MuninnModelEntry *me, const char **prompts, int n_prompts, const char *grammar_gbnf,
                               int max_tokens, BatchSlot *slots, char *errbuf, int errbuf_sz) {
    if (n_prompts <= 0 || n_prompts > MAX_BATCH_SEQS) {
        snprintf(errbuf, errbuf_sz, "batch size %d out of range [1,%d]", n_prompts, MAX_BATCH_SEQS);
        return -1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);
    llama_memory_t mem = llama_get_memory(me->ctx);
    llama_memory_clear(mem, 1);

    /* ── Tokenize all prompts ── */
    llama_token *tok_buf[MAX_BATCH_SEQS] = {0};
    int tok_len[MAX_BATCH_SEQS] = {0};
    int total_prompt_tokens = 0;

    for (int s = 0; s < n_prompts; s++) {
        slots[s].text = NULL;
        slots[s].len = 0;
        slots[s].done = 0;

        int n = llama_tokenize(vocab, prompts[s], (int)strlen(prompts[s]), NULL, 0, 1, 1);
        if (n < 0)
            n = -n;
        if (n == 0 || n > me->n_ctx - 4) {
            slots[s].done = 1;
            continue;
        }
        tok_buf[s] = (llama_token *)malloc((size_t)n * sizeof(llama_token));
        if (!tok_buf[s]) {
            slots[s].done = 1;
            continue;
        }
        int actual = llama_tokenize(vocab, prompts[s], (int)strlen(prompts[s]), tok_buf[s], n, 1, 1);
        if (actual < 0) {
            free(tok_buf[s]);
            tok_buf[s] = NULL;
            slots[s].done = 1;
            continue;
        }
        tok_len[s] = actual;
        total_prompt_tokens += actual;
    }

    int n_active = 0;
    for (int s = 0; s < n_prompts; s++)
        if (!slots[s].done)
            n_active++;

    if (n_active == 0) {
        snprintf(errbuf, errbuf_sz, "all %d prompts failed tokenization", n_prompts);
        return -1;
    }

    if (total_prompt_tokens + n_active > me->n_ctx) {
        snprintf(errbuf, errbuf_sz, "batch prompts (%d tokens) exceed context (%d)", total_prompt_tokens, me->n_ctx);
        for (int s = 0; s < n_prompts; s++)
            free(tok_buf[s]);
        return -1;
    }

    /* ── Build prompt batch (all sequences, logits only on last token each) ── */
    int batch_cap = total_prompt_tokens;
    if (batch_cap < n_prompts)
        batch_cap = n_prompts;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    for (int s = 0; s < n_prompts; s++) {
        if (!tok_buf[s])
            continue;
        for (int t = 0; t < tok_len[s]; t++) {
            int idx = batch.n_tokens;
            batch.token[idx] = tok_buf[s][t];
            batch.pos[idx] = t;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = s;
            batch.logits[idx] = (t == tok_len[s] - 1) ? 1 : 0;
            batch.n_tokens++;
        }
    }
    int rc = llama_decode(me->ctx, batch);
    if (rc != 0) {
        snprintf(errbuf, errbuf_sz, "prompt batch decode failed (rc=%d)", rc);
        llama_batch_free(batch);
        for (int s = 0; s < n_prompts; s++)
            free(tok_buf[s]);
        return -1;
    }

    /* ── Per-sequence generation state ── */
    struct llama_sampler *smpls[MAX_BATCH_SEQS] = {0};
    int i_batch_arr[MAX_BATCH_SEQS];
    int n_past[MAX_BATCH_SEQS];
    int buf_caps[MAX_BATCH_SEQS];

    int pos = 0;
    for (int s = 0; s < n_prompts; s++) {
        if (!tok_buf[s]) {
            i_batch_arr[s] = -1;
            continue;
        }
        i_batch_arr[s] = pos + tok_len[s] - 1;
        n_past[s] = tok_len[s];

        smpls[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (grammar_gbnf && grammar_gbnf[0])
            llama_sampler_chain_add(smpls[s], llama_sampler_init_grammar(vocab, grammar_gbnf, "root"));
        llama_sampler_chain_add(smpls[s], llama_sampler_init_greedy());

        buf_caps[s] = max_tokens * 16 + 256;
        slots[s].text = (char *)malloc((size_t)buf_caps[s]);
        if (!slots[s].text) {
            slots[s].done = 1;
            n_active--;
        }
        pos += tok_len[s];
    }

    for (int s = 0; s < n_prompts; s++) {
        free(tok_buf[s]);
        tok_buf[s] = NULL;
    }

    /* ── Autoregressive generation loop ── */
    for (int step = 0; step < max_tokens && n_active > 0; step++) {
        batch.n_tokens = 0;

        for (int s = 0; s < n_prompts; s++) {
            if (slots[s].done)
                continue;

            llama_token new_token = llama_sampler_sample(smpls[s], me->ctx, i_batch_arr[s]);

            if (llama_vocab_is_eog(vocab, new_token)) {
                slots[s].done = 1;
                n_active--;
                llama_memory_seq_rm(mem, s, -1, -1);
                continue;
            }

            char piece[256];
            int piece_len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, 1);
            if (piece_len > 0) {
                if (slots[s].len + piece_len + 1 > buf_caps[s]) {
                    buf_caps[s] = buf_caps[s] * 2 + piece_len;
                    char *nb = (char *)realloc(slots[s].text, (size_t)buf_caps[s]);
                    if (!nb) {
                        slots[s].done = 1;
                        n_active--;
                        continue;
                    }
                    slots[s].text = nb;
                }
                memcpy(slots[s].text + slots[s].len, piece, (size_t)piece_len);
                slots[s].len += piece_len;
            }

            int idx = batch.n_tokens;
            batch.token[idx] = new_token;
            batch.pos[idx] = n_past[s];
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = s;
            batch.logits[idx] = 1;
            i_batch_arr[s] = batch.n_tokens;
            batch.n_tokens++;
            n_past[s]++;
        }

        if (batch.n_tokens == 0)
            break;

        rc = llama_decode(me->ctx, batch);
        if (rc != 0)
            break;
    }

    /* ── Finalize ── */
    for (int s = 0; s < n_prompts; s++) {
        if (slots[s].text)
            slots[s].text[slots[s].len] = '\0';
        if (smpls[s])
            llama_sampler_free(smpls[s]);
    }
    llama_batch_free(batch);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────
 * Chat Template Formatting
 *
 * Uses the model's built-in chat template via llama_chat_apply_template().
 * For convenience functions, builds system+user message pairs.
 * ────────────────────────────────────────────────────────────── */

static char *format_chat_messages(MuninnModelEntry *me, const char *system_msg, const char *user_msg,
                                  int inject_skip_think, int *out_len) {
    struct llama_chat_message msgs[2];
    int n_msg = 0;

    if (system_msg && system_msg[0]) {
        msgs[n_msg].role = "system";
        msgs[n_msg].content = system_msg;
        n_msg++;
    }
    msgs[n_msg].role = "user";
    msgs[n_msg].content = user_msg;
    n_msg++;

    const char *tmpl = llama_model_chat_template(me->model, NULL);

    /* First call: measure required buffer size */
    int32_t needed = llama_chat_apply_template(tmpl, msgs, (size_t)n_msg, 1, NULL, 0);
    if (needed <= 0)
        return NULL;

    char *buf = (char *)malloc((size_t)(needed + 64)); /* +64 for think tag */
    if (!buf)
        return NULL;

    int32_t written = llama_chat_apply_template(tmpl, msgs, (size_t)n_msg, 1, buf, needed + 1);
    if (written < 0) {
        free(buf);
        return NULL;
    }
    buf[written] = '\0';

    /* Qwen3.5 thinking models: llama_chat_apply_template() uses a hardcoded
     * ChatML handler that doesn't inject the <think> prefix the Jinja2 template
     * would. When inject_skip_think=1, we inject an empty closed think block
     * to disable reasoning. Default (0) = no injection, bare ChatML. */
    if (inject_skip_think == 1 && tmpl && strstr(tmpl, "enable_thinking")) {
        static const char tag[] = "<think>\n\n</think>\n\n";
        int tag_len = (int)(sizeof(tag) - 1);
        memcpy(buf + written, tag, (size_t)tag_len);
        written += tag_len;
        buf[written] = '\0';
    }

    if (out_len)
        *out_len = written;
    return buf;
}

/* ──────────────────────────────────────────────────────────────────
 * Qwen3 <think> block stripping
 *
 * Qwen3/3.5 models may emit <think>...</think> blocks before output
 * even with /no_think in the user prompt. Strip everything up to
 * and including the closing </think> tag.
 *
 * Also handles truncated think blocks: when max_tokens is exhausted
 * mid-reasoning, the output starts with <think> but never closes.
 * In that case, the entire output is reasoning with no response —
 * return pointer to end of string (empty result).
 * ────────────────────────────────────────────────────────────── */

const char *strip_think_block(const char *text) {
    const char *end_tag = strstr(text, "</think>");
    if (end_tag) {
        const char *after = end_tag + 8; /* strlen("</think>") == 8 */
        /* Skip whitespace after the tag */
        while (*after == ' ' || *after == '\n' || *after == '\r' || *after == '\t')
            after++;
        return after;
    }
    /* Truncated think block: <think> opened but never closed.
     * The model exhausted tokens on reasoning. Return empty.
     * Skip leading whitespace — some models emit \n before <think>. */
    const char *t = text;
    while (*t == ' ' || *t == '\n' || *t == '\r' || *t == '\t')
        t++;
    if (strncmp(t, "<think>", 7) == 0)
        return text + strlen(text);
    return text;
}

/* ──────────────────────────────────────────────────────────────────
 * Grammar-constrained JSON result
 *
 * With GBNF grammar active, each generated token is guaranteed valid.
 * The only failure mode is max_tokens truncation (model didn't finish
 * before the token budget ran out). In that case we return fallback_json.
 *
 * wrap_key: if non-NULL and the model produced a bare JSON array,
 * wrap it as {"<wrap_key>": [...]} to normalize the output format.
 * Small models often emit bare arrays instead of the wrapper object.
 * ────────────────────────────────────────────────────────────── */

static void result_json_output(sqlite3_context *ctx, char *output, int output_len, const char *fallback_json,
                               const char *wrap_key) {
    /* Strip <think>...</think> prefix if present (GBNF allows it optionally) */
    const char *json_start = strip_think_block(output);
    int json_len = output_len - (int)(json_start - output);

    yyjson_doc *doc = yyjson_read(json_start, (size_t)json_len, 0);
    if (doc) {
        yyjson_val *root = yyjson_doc_get_root(doc);

        /* If model produced bare array and we have a wrap key, normalize to object */
        if (wrap_key && yyjson_is_arr(root)) {
            yyjson_mut_doc *mut = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *obj = yyjson_mut_obj(mut);
            yyjson_mut_doc_set_root(mut, obj);
            yyjson_mut_val *arr_copy = yyjson_val_mut_copy(mut, root);
            yyjson_mut_obj_add_val(mut, obj, wrap_key, arr_copy);
            yyjson_doc_free(doc);

            size_t write_len = 0;
            char *minified = yyjson_mut_write(mut, YYJSON_WRITE_NOFLAG, &write_len);
            yyjson_mut_doc_free(mut);
            if (minified) {
                sqlite3_result_text(ctx, minified, (int)write_len, free);
                sqlite3_result_subtype(ctx, (unsigned int)'J');
                free(output);
                return;
            }
        } else {
            size_t write_len = 0;
            char *minified = yyjson_write(doc, YYJSON_WRITE_NOFLAG, &write_len);
            yyjson_doc_free(doc);
            if (minified) {
                sqlite3_result_text(ctx, minified, (int)write_len, free);
                sqlite3_result_subtype(ctx, (unsigned int)'J');
                free(output);
                return;
            }
        }
    }
    /* Truncated by max_tokens — return fallback */
    sqlite3_result_text(ctx, fallback_json, -1, SQLITE_STATIC);
    sqlite3_result_subtype(ctx, (unsigned int)'J');
    free(output);
}

/* ──────────────────────────────────────────────────────────────────
 * Extract argument disambiguation
 *
 * Shared by all extract functions. Parses optional args starting at
 * argv[start_idx]. TEXT = labels (supervised), INTEGER = inject_skip_think
 * (unsupervised). Returns labels=NULL for unsupervised mode.
 * ────────────────────────────────────────────────────────────── */

typedef struct {
    const char *labels;    /* NULL if unsupervised */
    int inject_skip_think; /* 0 = off (default), 1 = inject closed think block */
} ExtractArgs;

static ExtractArgs parse_extract_args(int argc, sqlite3_value **argv, int start_idx) {
    ExtractArgs args = {NULL, 0};
    if (argc > start_idx && sqlite3_value_type(argv[start_idx]) == SQLITE_TEXT) {
        args.labels = (const char *)sqlite3_value_text(argv[start_idx]);
        if (argc > start_idx + 1 && sqlite3_value_type(argv[start_idx + 1]) == SQLITE_INTEGER)
            args.inject_skip_think = sqlite3_value_int(argv[start_idx + 1]) ? 1 : 0;
    } else if (argc > start_idx && sqlite3_value_type(argv[start_idx]) == SQLITE_INTEGER) {
        args.inject_skip_think = sqlite3_value_int(argv[start_idx]) ? 1 : 0;
    }
    return args;
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_chat_model(path TEXT [, n_ctx INTEGER]) -> POINTER
 * ────────────────────────────────────────────────────────────── */

static void fn_chat_model(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_chat_model: path must be TEXT", -1);
        return;
    }
    const char *path = (const char *)sqlite3_value_text(argv[0]);
    int n_ctx = 0; /* 0 = use dynamic default: max(DEFAULT_N_CTX, train_ctx / 8) */
    if (argc >= 2 && sqlite3_value_type(argv[1]) == SQLITE_INTEGER) {
        n_ctx = sqlite3_value_int(argv[1]);
    }

    char errbuf[256];
    LoadedChatModel lm;
    if (load_chat_model(path, n_ctx, &lm, errbuf, sizeof(errbuf)) != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    LoadedChatModel *heap = (LoadedChatModel *)sqlite3_malloc(sizeof(LoadedChatModel));
    if (!heap) {
        llama_free(lm.ctx);
        llama_model_free(lm.model);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *heap = lm;
    sqlite3_result_pointer(ctx, heap, CHAT_MODEL_PTR_TYPE, NULL);
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_chat(model, prompt [, grammar [, max_tokens [, system_prompt]]]) -> TEXT
 * ────────────────────────────────────────────────────────────── */

static void fn_chat(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(ctx, "muninn_chat: requires (model, prompt)", -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *prompt = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !prompt) {
        sqlite3_result_error(ctx, "muninn_chat: model and prompt must be TEXT", -1);
        return;
    }

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_chat: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const char *grammar = NULL;
    if (argc >= 3 && sqlite3_value_type(argv[2]) == SQLITE_TEXT) {
        grammar = (const char *)sqlite3_value_text(argv[2]);
    }
    int max_tokens = me->n_ctx;
    if (argc >= 4 && sqlite3_value_type(argv[3]) == SQLITE_INTEGER) {
        max_tokens = sqlite3_value_int(argv[3]);
    }
    const char *system_msg = NULL;
    if (argc >= 5 && sqlite3_value_type(argv[4]) == SQLITE_TEXT) {
        system_msg = (const char *)sqlite3_value_text(argv[4]);
    }

    /* Format with optional system message using model chat template */
    int fmt_len = 0;
    char *formatted = format_chat_messages(me, system_msg, prompt, 0, &fmt_len);
    if (!formatted) {
        sqlite3_result_error(ctx, "muninn_chat: failed to format chat template", -1);
        return;
    }

    char errbuf[256];
    char *output = NULL;
    int output_len = 0;
    int rc = chat_generate(me, formatted, grammar, max_tokens, &output, &output_len, errbuf, sizeof(errbuf));
    free(formatted);

    if (rc != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    sqlite3_result_text(ctx, output, output_len, free);
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_extract_entities(model, text [, labels [, inject_skip_think]]) -> TEXT
 *
 * Supervised (labels provided): extract entities of the specified types.
 * Unsupervised (labels omitted): open extraction of all notable entities.
 * inject_skip_think: 0=off (default), 1=inject closed think block to
 * suppress reasoning on Qwen3.5 models.
 *
 * Returns JSON: {"entities":[{"text":"...","type":"...","score":0.0-1.0},...]}
 * ────────────────────────────────────────────────────────────── */

static void fn_extract_entities(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(ctx, "muninn_extract_entities: requires (model, text [, labels [, inject_skip_think]])",
                             -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !text) {
        sqlite3_result_error(ctx, "muninn_extract_entities: model and text must be TEXT", -1);
        return;
    }

    ExtractArgs args = parse_extract_args(argc, argv, 2);

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_extract_entities: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const char *sys_prompt = args.labels ? SYS_NER_SUP : SYS_NER_UNSUP;

    int user_len = (int)strlen(text) + (args.labels ? (int)strlen(args.labels) : 0) + 128;
    char *user_prompt = (char *)malloc((size_t)user_len);
    if (!user_prompt) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    if (args.labels)
        snprintf(user_prompt, user_len, "Extract entities of types: %s\nText: %s", args.labels, text);
    else
        snprintf(user_prompt, user_len, "Text: %s", text);

    char *formatted = format_chat_messages(me, sys_prompt, user_prompt, args.inject_skip_think, NULL);
    free(user_prompt);
    if (!formatted) {
        sqlite3_result_error(ctx, "muninn_extract_entities: template formatting failed", -1);
        return;
    }

    char errbuf[256];
    char *output = NULL;
    int output_len = 0;
    int rc = chat_generate(me, formatted, GBNF_NER, me->n_ctx, &output, &output_len, errbuf, sizeof(errbuf));
    free(formatted);
    if (rc != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    result_json_output(ctx, output, output_len, "{\"entities\":[]}", "entities");
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_extract_relations(model, text [, entities_json [, inject_skip_think]]) -> TEXT
 *
 * Supervised (entities_json provided): extract relations between given entities.
 * Unsupervised (entities_json omitted): discover entities and relations from text.
 *
 * Returns JSON: {"relations":[{"head":"...","rel":"...","tail":"...","score":0.0-1.0},...]}
 * ────────────────────────────────────────────────────────────── */

static void fn_extract_relations(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(
            ctx, "muninn_extract_relations: requires (model, text [, entities_json [, inject_skip_think]])", -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !text) {
        sqlite3_result_error(ctx, "muninn_extract_relations: model and text must be TEXT", -1);
        return;
    }

    ExtractArgs args = parse_extract_args(argc, argv, 2);

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_extract_relations: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const char *sys_prompt = args.labels ? SYS_RE_SUP : SYS_RE_UNSUP;

    int user_len = (int)strlen(text) + (args.labels ? (int)strlen(args.labels) : 0) + 128;
    char *user_prompt = (char *)malloc((size_t)user_len);
    if (!user_prompt) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    if (args.labels)
        snprintf(user_prompt, user_len, "Entities: %s\nText: %s", args.labels, text);
    else
        snprintf(user_prompt, user_len, "Text: %s", text);

    char *formatted = format_chat_messages(me, sys_prompt, user_prompt, args.inject_skip_think, NULL);
    free(user_prompt);
    if (!formatted) {
        sqlite3_result_error(ctx, "muninn_extract_relations: template formatting failed", -1);
        return;
    }

    char errbuf[256];
    char *output = NULL;
    int output_len = 0;
    int rc = chat_generate(me, formatted, GBNF_RE, me->n_ctx, &output, &output_len, errbuf, sizeof(errbuf));
    free(formatted);
    if (rc != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    result_json_output(ctx, output, output_len, "{\"relations\":[]}", "relations");
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_extract_ner_re(model, text [, entity_labels,
 *                                     relation_labels [, inject_skip_think]]) -> TEXT
 *
 * Supervised (both label sets provided): extract entities + relations of given types.
 * Unsupervised (labels omitted): open extraction of all entities + relations.
 * No mixed mode — call separate functions if you want supervised NER + open RE.
 *
 * Combined NER + RE in a single LLM call for 2x throughput.
 * Returns JSON: {"entities":[{"text":"...","type":"...","score":0.0-1.0},...],
 *                "relations":[{"head":"...","rel":"...","tail":"...","score":0.0-1.0},...]}
 * ────────────────────────────────────────────────────────────── */

static void fn_extract_ner_re(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(
            ctx, "muninn_extract_ner_re: requires (model, text [, ent_labels, rel_labels [, inject_skip_think]])", -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !text) {
        sqlite3_result_error(ctx, "muninn_extract_ner_re: model and text must be TEXT", -1);
        return;
    }

    /* Disambiguate supervised vs unsupervised:
     * - argc >= 4 with TEXT at [2] and TEXT at [3]: supervised (ent_labels, rel_labels)
     * - argc >= 3 with TEXT at [2] only: error — need both label sets or neither
     * - INTEGER at [2]: unsupervised with inject_skip_think
     * - argc == 2: unsupervised, no flags */
    const char *entity_labels = NULL;
    const char *relation_labels = NULL;
    int inject_skip_think = 0;

    if (argc > 2 && sqlite3_value_type(argv[2]) == SQLITE_TEXT) {
        entity_labels = (const char *)sqlite3_value_text(argv[2]);
        if (argc > 3 && sqlite3_value_type(argv[3]) == SQLITE_TEXT) {
            relation_labels = (const char *)sqlite3_value_text(argv[3]);
            if (argc > 4 && sqlite3_value_type(argv[4]) == SQLITE_INTEGER)
                inject_skip_think = sqlite3_value_int(argv[4]) ? 1 : 0;
        } else {
            sqlite3_result_error(ctx, "muninn_extract_ner_re: supervised mode requires both ent_labels and rel_labels",
                                 -1);
            return;
        }
    } else if (argc > 2 && sqlite3_value_type(argv[2]) == SQLITE_INTEGER) {
        inject_skip_think = sqlite3_value_int(argv[2]) ? 1 : 0;
    }

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_extract_ner_re: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const char *sys_prompt = entity_labels ? SYS_NER_RE_SUP : SYS_NER_RE_UNSUP;

    int user_len = (int)strlen(text) + (entity_labels ? (int)strlen(entity_labels) : 0) +
                   (relation_labels ? (int)strlen(relation_labels) : 0) + 192;
    char *user_prompt = (char *)malloc((size_t)user_len);
    if (!user_prompt) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    if (entity_labels)
        snprintf(user_prompt, user_len, "Extract entities of types: %s\nExtract relations of types: %s\nText: %s",
                 entity_labels, relation_labels, text);
    else
        snprintf(user_prompt, user_len, "Text: %s", text);

    char *formatted = format_chat_messages(me, sys_prompt, user_prompt, inject_skip_think, NULL);
    free(user_prompt);
    if (!formatted) {
        sqlite3_result_error(ctx, "muninn_extract_ner_re: template formatting failed", -1);
        return;
    }

    char errbuf[256];
    char *output = NULL;
    int output_len = 0;
    int rc = chat_generate(me, formatted, GBNF_NER_RE, me->n_ctx, &output, &output_len, errbuf, sizeof(errbuf));
    free(formatted);
    if (rc != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    result_json_output(ctx, output, output_len, "{\"entities\":[],\"relations\":[]}", NULL);
}

/* ──────────────────────────────────────────────────────────────────
 * Batch SQL Helper: format prompts, call chat_generate_batch with
 * GBNF grammar, build output JSON array from grammar-guaranteed results.
 * ────────────────────────────────────────────────────────────── */

static void batch_extract_core(sqlite3_context *ctx, MuninnModelEntry *me, yyjson_doc *in_doc, yyjson_val *in_arr,
                               const char *sys_prompt, const char *grammar_gbnf, const char *fallback_json,
                               const char *wrap_key, int max_tokens, int batch_size,
                               void (*build_user_prompt)(char *buf, int buf_sz, const char *text, const void *extra),
                               const void *extra) {
    int n_texts = (int)yyjson_arr_size(in_arr);

    yyjson_mut_doc *out_doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *out_arr = yyjson_mut_arr(out_doc);
    yyjson_mut_doc_set_root(out_doc, out_arr);

    for (int chunk_start = 0; chunk_start < n_texts; chunk_start += batch_size) {
        int chunk_n = n_texts - chunk_start;
        if (chunk_n > batch_size)
            chunk_n = batch_size;

        const char *prompts[MAX_BATCH_SEQS] = {0};
        char *user_bufs[MAX_BATCH_SEQS] = {0};
        char *fmt_bufs[MAX_BATCH_SEQS] = {0};

        for (int i = 0; i < chunk_n; i++) {
            yyjson_val *text_val = yyjson_arr_get(in_arr, (size_t)(chunk_start + i));
            const char *text = yyjson_get_str(text_val);
            if (!text)
                text = "";

            int user_len = (int)strlen(text) + 512;
            user_bufs[i] = (char *)malloc((size_t)user_len);
            if (!user_bufs[i])
                continue;
            build_user_prompt(user_bufs[i], user_len, text, extra);

            fmt_bufs[i] = format_chat_messages(me, sys_prompt, user_bufs[i], 0, NULL);
            prompts[i] = fmt_bufs[i] ? fmt_bufs[i] : "";
        }

        BatchSlot slots[MAX_BATCH_SEQS];
        memset(slots, 0, sizeof(slots));
        char errbuf[256];
        chat_generate_batch(me, prompts, chunk_n, grammar_gbnf, max_tokens, slots, errbuf, sizeof(errbuf));

        for (int i = 0; i < chunk_n; i++) {
            int added = 0;
            if (slots[i].text && slots[i].len > 0) {
                /* Grammar guarantees valid JSON — parse directly */
                yyjson_doc *res_doc = yyjson_read(slots[i].text, (size_t)slots[i].len, 0);
                if (res_doc) {
                    yyjson_val *res_root = yyjson_doc_get_root(res_doc);
                    /* Wrap bare array in object if needed */
                    if (wrap_key && yyjson_is_arr(res_root)) {
                        yyjson_mut_val *wrapper = yyjson_mut_obj(out_doc);
                        yyjson_mut_val *arr_copy = yyjson_val_mut_copy(out_doc, res_root);
                        yyjson_mut_obj_add_val(out_doc, wrapper, wrap_key, arr_copy);
                        yyjson_mut_arr_append(out_arr, wrapper);
                    } else {
                        yyjson_mut_val *rv = yyjson_val_mut_copy(out_doc, res_root);
                        yyjson_mut_arr_append(out_arr, rv);
                    }
                    yyjson_doc_free(res_doc);
                    added = 1;
                }
            }
            if (!added) {
                yyjson_doc *fb_doc = yyjson_read(fallback_json, strlen(fallback_json), 0);
                yyjson_mut_val *fb = yyjson_val_mut_copy(out_doc, yyjson_doc_get_root(fb_doc));
                yyjson_mut_arr_append(out_arr, fb);
                yyjson_doc_free(fb_doc);
            }
            free(slots[i].text);
            free(user_bufs[i]);
            free(fmt_bufs[i]);
        }
    }

    yyjson_doc_free(in_doc);

    size_t out_len = 0;
    char *out_json = yyjson_mut_write(out_doc, YYJSON_WRITE_NOFLAG, &out_len);
    yyjson_mut_doc_free(out_doc);
    if (out_json) {
        sqlite3_result_text(ctx, out_json, (int)out_len, free);
        sqlite3_result_subtype(ctx, (unsigned int)'J');
    } else {
        sqlite3_result_error(ctx, "batch JSON serialization failed", -1);
    }
}

/* ── Prompt builder callbacks ── */

typedef struct {
    const char *labels; /* NULL for unsupervised */
} NerPromptExtra;

static void build_ner_prompt(char *buf, int buf_sz, const char *text, const void *extra) {
    const NerPromptExtra *e = (const NerPromptExtra *)extra;
    if (e->labels)
        snprintf(buf, (size_t)buf_sz, "Extract entities of types: %s\nText: %s", e->labels, text);
    else
        snprintf(buf, (size_t)buf_sz, "Text: %s", text);
}

typedef struct {
    const char *entity_labels;   /* NULL for unsupervised */
    const char *relation_labels; /* NULL for unsupervised */
} NerRePromptExtra;

static void build_ner_re_prompt(char *buf, int buf_sz, const char *text, const void *extra) {
    const NerRePromptExtra *e = (const NerRePromptExtra *)extra;
    if (e->entity_labels)
        snprintf(buf, (size_t)buf_sz, "Extract entities of types: %s\nExtract relations of types: %s\nText: %s",
                 e->entity_labels, e->relation_labels, text);
    else
        snprintf(buf, (size_t)buf_sz, "Text: %s", text);
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_extract_entities_batch(model, texts_json [, labels [, batch_size]])
 *
 * Supervised (labels TEXT): extract entities of the specified types.
 * Unsupervised (labels omitted or INTEGER): open extraction of all entities.
 *
 * Input:  JSON array of texts ["text1", "text2", ...]
 * Output: JSON array of results [{"entities":[...]}, {"entities":[...]}, ...]
 * ────────────────────────────────────────────────────────────── */

static void fn_extract_entities_batch(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(
            ctx, "muninn_extract_entities_batch: requires (model, texts_json [, labels [, batch_size]])", -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *texts_json = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !texts_json) {
        sqlite3_result_error(ctx, "muninn_extract_entities_batch: model and texts_json must be TEXT", -1);
        return;
    }

    /* Parse optional labels + batch_size using type disambiguation */
    const char *labels = NULL;
    int batch_size = DEFAULT_BATCH_SIZE;
    int next_idx = 2;

    if (argc > next_idx && sqlite3_value_type(argv[next_idx]) == SQLITE_TEXT) {
        labels = (const char *)sqlite3_value_text(argv[next_idx]);
        next_idx++;
    }
    if (argc > next_idx && sqlite3_value_type(argv[next_idx]) == SQLITE_INTEGER) {
        batch_size = sqlite3_value_int(argv[next_idx]);
        if (batch_size < 1)
            batch_size = 1;
        if (batch_size > MAX_BATCH_SEQS)
            batch_size = MAX_BATCH_SEQS;
    }

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_extract_entities_batch: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    yyjson_doc *in_doc = yyjson_read(texts_json, strlen(texts_json), 0);
    if (!in_doc) {
        sqlite3_result_error(ctx, "muninn_extract_entities_batch: invalid JSON array", -1);
        return;
    }
    yyjson_val *in_arr = yyjson_doc_get_root(in_doc);
    if (!yyjson_is_arr(in_arr)) {
        yyjson_doc_free(in_doc);
        sqlite3_result_error(ctx, "muninn_extract_entities_batch: input must be JSON array", -1);
        return;
    }
    if (yyjson_arr_size(in_arr) == 0) {
        yyjson_doc_free(in_doc);
        sqlite3_result_text(ctx, "[]", 2, SQLITE_STATIC);
        sqlite3_result_subtype(ctx, (unsigned int)'J');
        return;
    }

    const char *sys_prompt = labels ? SYS_NER_SUP : SYS_NER_UNSUP;

    NerPromptExtra extra = {.labels = labels};
    batch_extract_core(ctx, me, in_doc, in_arr, sys_prompt, GBNF_NER, "{\"entities\":[]}", "entities", me->n_ctx,
                       batch_size, build_ner_prompt, &extra);
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_extract_ner_re_batch(model, texts_json
 *                                           [, entity_labels, relation_labels
 *                                            [, batch_size]])
 *
 * Supervised (both label sets TEXT): extract entities + relations of given types.
 * Unsupervised (labels omitted or INTEGER): open extraction.
 *
 * Combined NER + RE batch: 2x throughput vs sequential, N× parallelism.
 * Input:  JSON array of texts
 * Output: JSON array of {entities:[...], relations:[...]}
 * ────────────────────────────────────────────────────────────── */

static void fn_extract_ner_re_batch(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2) {
        sqlite3_result_error(
            ctx, "muninn_extract_ner_re_batch: requires (model, texts_json [, ent_labels, rel_labels [, batch_size]])",
            -1);
        return;
    }
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *texts_json = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !texts_json) {
        sqlite3_result_error(ctx, "muninn_extract_ner_re_batch: model and texts_json must be TEXT", -1);
        return;
    }

    /* Parse optional labels + batch_size using type disambiguation */
    const char *entity_labels = NULL;
    const char *relation_labels = NULL;
    int batch_size = DEFAULT_BATCH_SIZE;
    int next_idx = 2;

    if (argc > next_idx && sqlite3_value_type(argv[next_idx]) == SQLITE_TEXT) {
        entity_labels = (const char *)sqlite3_value_text(argv[next_idx]);
        next_idx++;
        if (argc > next_idx && sqlite3_value_type(argv[next_idx]) == SQLITE_TEXT) {
            relation_labels = (const char *)sqlite3_value_text(argv[next_idx]);
            next_idx++;
        } else {
            sqlite3_result_error(
                ctx, "muninn_extract_ner_re_batch: supervised mode requires both ent_labels and rel_labels", -1);
            return;
        }
    }
    if (argc > next_idx && sqlite3_value_type(argv[next_idx]) == SQLITE_INTEGER) {
        batch_size = sqlite3_value_int(argv[next_idx]);
        if (batch_size < 1)
            batch_size = 1;
        if (batch_size > MAX_BATCH_SEQS)
            batch_size = MAX_BATCH_SEQS;
    }

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_extract_ner_re_batch: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    yyjson_doc *in_doc = yyjson_read(texts_json, strlen(texts_json), 0);
    if (!in_doc) {
        sqlite3_result_error(ctx, "muninn_extract_ner_re_batch: invalid JSON array", -1);
        return;
    }
    yyjson_val *in_arr = yyjson_doc_get_root(in_doc);
    if (!yyjson_is_arr(in_arr)) {
        yyjson_doc_free(in_doc);
        sqlite3_result_error(ctx, "muninn_extract_ner_re_batch: input must be JSON array", -1);
        return;
    }
    if (yyjson_arr_size(in_arr) == 0) {
        yyjson_doc_free(in_doc);
        sqlite3_result_text(ctx, "[]", 2, SQLITE_STATIC);
        sqlite3_result_subtype(ctx, (unsigned int)'J');
        return;
    }

    const char *sys_prompt = entity_labels ? SYS_NER_RE_SUP : SYS_NER_RE_UNSUP;

    NerRePromptExtra extra = {.entity_labels = entity_labels, .relation_labels = relation_labels};
    batch_extract_core(ctx, me, in_doc, in_arr, sys_prompt, GBNF_NER_RE, "{\"entities\":[],\"relations\":[]}", NULL,
                       me->n_ctx, batch_size, build_ner_re_prompt, &extra);
}

/* ──────────────────────────────────────────────────────────────────
 * SQL Function: muninn_summarize(model TEXT, text TEXT [, max_tokens INT]) -> TEXT
 * ────────────────────────────────────────────────────────────── */

static void fn_summarize(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);
    if (!name || !text) {
        sqlite3_result_error(ctx, "muninn_summarize: model and text must be TEXT", -1);
        return;
    }

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_CHAT);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_summarize: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    /* Default: let the model generate until EOG (end-of-generation) token.
     * n_ctx is the hard ceiling. Thinking models use part of the budget for
     * internal reasoning; strip_think_block() removes it from the result. */
    int max_tokens = me->n_ctx;
    if (argc >= 3 && sqlite3_value_type(argv[2]) == SQLITE_INTEGER) {
        max_tokens = sqlite3_value_int(argv[2]);
    }

    int user_len = (int)strlen(text) + 256;
    char *user_prompt = (char *)malloc((size_t)user_len);
    if (!user_prompt) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    snprintf(user_prompt, user_len, "Produce a one-sentence summary of this text:\n\n%s", text);

    char *formatted = format_chat_messages(me, NULL, user_prompt, 0, NULL);
    free(user_prompt);
    if (!formatted) {
        sqlite3_result_error(ctx, "muninn_summarize: template formatting failed", -1);
        return;
    }

    char errbuf[256];
    char *output = NULL;
    int output_len = 0;
    int rc = chat_generate(me, formatted, NULL, max_tokens, &output, &output_len, errbuf, sizeof(errbuf));
    free(formatted);
    if (rc != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    const char *clean = strip_think_block(output);
    if (clean != output) {
        int clean_len = (int)strlen(clean);
        sqlite3_result_text(ctx, clean, clean_len, SQLITE_TRANSIENT);
        free(output);
    } else {
        sqlite3_result_text(ctx, output, output_len, free);
    }
}

/* ──────────────────────────────────────────────────────────────────
 * Eponymous Virtual Table: muninn_chat_models
 *
 * Same pattern as muninn_models in llama_embed.c but for the separate
 * chat model registry.
 *
 *   CREATE TABLE x(name TEXT NOT NULL, model HIDDEN, n_ctx INTEGER)
 *
 * INSERT/DELETE manage the chat model lifecycle.
 * ────────────────────────────────────────────────────────────── */

typedef struct {
    sqlite3_vtab base;
} ChatModelsVtab;
typedef struct {
    sqlite3_vtab_cursor base;
    int current;
} ChatModelsCursor;

static int chat_models_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab,
                               char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(name TEXT NOT NULL, model HIDDEN, n_ctx INTEGER)");
    if (rc != SQLITE_OK)
        return rc;

    ChatModelsVtab *vtab = sqlite3_malloc(sizeof(ChatModelsVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(*vtab));
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int chat_models_disconnect(sqlite3_vtab *pVtab) {
    sqlite3_free(pVtab);
    return SQLITE_OK;
}

static int chat_models_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVtab;
    ChatModelsCursor *cur = sqlite3_malloc(sizeof(ChatModelsCursor));
    if (!cur)
        return SQLITE_NOMEM;
    memset(cur, 0, sizeof(*cur));
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int chat_models_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

static void chat_models_advance(ChatModelsCursor *cur) {
    int cap = muninn_registry_capacity();
    while (cur->current < cap) {
        MuninnModelEntry *e = muninn_registry_at(cur->current);
        if (e && e->in_use && e->type == MUNINN_MODEL_CHAT)
            break;
        cur->current++;
    }
}

static int chat_models_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc,
                              sqlite3_value **argv) {
    (void)idxNum;
    (void)idxStr;
    (void)argc;
    (void)argv;
    ChatModelsCursor *cur = (ChatModelsCursor *)pCursor;
    cur->current = 0;
    chat_models_advance(cur);
    return SQLITE_OK;
}

static int chat_models_next(sqlite3_vtab_cursor *pCursor) {
    ChatModelsCursor *cur = (ChatModelsCursor *)pCursor;
    cur->current++;
    chat_models_advance(cur);
    return SQLITE_OK;
}

static int chat_models_eof(sqlite3_vtab_cursor *pCursor) {
    return ((ChatModelsCursor *)pCursor)->current >= muninn_registry_capacity();
}

static int chat_models_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    ChatModelsCursor *cur = (ChatModelsCursor *)pCursor;
    MuninnModelEntry *me = muninn_registry_at(cur->current);
    switch (col) {
    case 0:
        sqlite3_result_text(ctx, me->name, -1, SQLITE_TRANSIENT);
        break;
    case 1:
        sqlite3_result_null(ctx);
        break; /* model: write-only hidden column */
    case 2:
        sqlite3_result_int(ctx, me->n_ctx);
        break;
    default:
        sqlite3_result_null(ctx);
        break;
    }
    return SQLITE_OK;
}

static int chat_models_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((ChatModelsCursor *)pCursor)->current;
    return SQLITE_OK;
}

static int chat_models_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pInfo) {
    (void)pVtab;
    pInfo->estimatedCost = 10.0;
    pInfo->estimatedRows = muninn_registry_capacity();
    return SQLITE_OK;
}

static int chat_models_update(sqlite3_vtab *pVtab, int argc, sqlite3_value **argv, sqlite3_int64 *pRowid) {
    (void)pRowid;

    if (argc == 1) {
        /* DELETE: argv[0] = rowid */
        int idx = (int)sqlite3_value_int64(argv[0]);
        MuninnModelEntry *me = muninn_registry_at(idx);
        if (me && me->in_use && me->type == MUNINN_MODEL_CHAT) {
            muninn_registry_remove(me->name);
        }
        return SQLITE_OK;
    }

    if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        /* INSERT: argv[2]=name, argv[3]=model (POINTER), argv[4]=n_ctx */
        const char *name = (const char *)sqlite3_value_text(argv[2]);
        if (!name || !name[0]) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_chat_models: name must be non-empty TEXT");
            return SQLITE_ERROR;
        }

        LoadedChatModel *lm = (LoadedChatModel *)sqlite3_value_pointer(argv[3], CHAT_MODEL_PTR_TYPE);
        if (!lm) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_chat_models: model column must be muninn_chat_model(path)");
            return SQLITE_ERROR;
        }

        int rc = muninn_registry_add(name, lm->model, lm->ctx, 0, lm->n_ctx, MUNINN_MODEL_CHAT);
        if (rc == -1) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_chat_models: model '%s' already loaded", name);
            return SQLITE_ERROR;
        }
        if (rc == -2) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_chat_models: registry full (max %d)", muninn_registry_capacity());
            return SQLITE_ERROR;
        }

        /* Transfer ownership — clear pointer so destructor doesn't double-free */
        lm->model = NULL;
        lm->ctx = NULL;
        return SQLITE_OK;
    }

    return SQLITE_ERROR;
}

static sqlite3_module chat_models_module = {
    .iVersion = 0,
    .xCreate = NULL, /* eponymous-only */
    .xConnect = chat_models_connect,
    .xBestIndex = chat_models_best_index,
    .xDisconnect = chat_models_disconnect,
    .xDestroy = NULL,
    .xOpen = chat_models_open,
    .xClose = chat_models_close,
    .xFilter = chat_models_filter,
    .xNext = chat_models_next,
    .xEof = chat_models_eof,
    .xColumn = chat_models_column,
    .xRowid = chat_models_rowid,
    .xUpdate = chat_models_update,
};

/* ──────────────────────────────────────────────────────────────────
 * Function Registration
 * ────────────────────────────────────────────────────────────── */

int chat_register_functions(sqlite3 *db) {
    int rc;

    /* muninn_chat_model(path [, n_ctx]) -> POINTER */
    rc = sqlite3_create_function(db, "muninn_chat_model", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_chat_model, NULL,
                                 NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_chat(model, prompt [, grammar [, max_tokens]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_chat", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_chat, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_extract_entities(model, text [, labels [, inject_skip_think]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_extract_entities", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL,
                                 fn_extract_entities, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_extract_relations(model, text [, entities_json [, inject_skip_think]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_extract_relations", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL,
                                 fn_extract_relations, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_extract_ner_re(model, text [, ent_labels, rel_labels [, inject_skip_think]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_extract_ner_re", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL,
                                 fn_extract_ner_re, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_extract_entities_batch(model, texts_json [, labels [, batch_size]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_extract_entities_batch", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL,
                                 fn_extract_entities_batch, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_extract_ner_re_batch(model, texts_json [, ent_labels, rel_labels [, batch_size]]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_extract_ner_re_batch", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL,
                                 fn_extract_ner_re_batch, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_summarize(model, text [, max_tokens]) -> TEXT */
    rc = sqlite3_create_function(db, "muninn_summarize", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_summarize, NULL,
                                 NULL);
    if (rc != SQLITE_OK)
        return rc;

    /* muninn_chat_models eponymous virtual table */
    rc = sqlite3_create_module(db, "muninn_chat_models", &chat_models_module, NULL);
    if (rc != SQLITE_OK)
        return rc;

    return SQLITE_OK;
}
