/*
 * llama_embed.c — GGUF model embedding via llama.cpp
 *
 * Wraps llama.cpp's extern "C" API to provide text embedding
 * and model management from SQL.
 *
 * Architecture:
 *   - Uses shared model registry from llama_common.c
 *   - muninn_models eponymous virtual table for embed model lifecycle
 *   - Scalar functions for embedding and metadata
 *
 * The llama.cpp API is pure C (extern "C"), so this file compiles
 * as C11 despite linking against C++ internals at link time.
 */
#include "llama_embed.h"
#include "llama_common.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

SQLITE_EXTENSION_INIT3

#include "llama.h"

/* ═══════════════════════════════════════════════════════════════════
 * Model Loading
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    int n_embd;
    int n_ctx;
} LoadedModel;

/*
 * Load a GGUF model for embedding inference.
 * n_ctx_override: if > 0, use this context size; otherwise auto-detect from model metadata.
 * Returns 0 on success, -1 on failure (writes error to errbuf).
 */
static int load_gguf_model(const char *path, int n_ctx_override, LoadedModel *out, char *errbuf, int errbuf_sz) {
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
        snprintf(errbuf, (size_t)errbuf_sz, "failed to load GGUF model: %s", path);
        return -1;
    }

    int n_embd = llama_model_n_embd(model);
    if (n_embd <= 0) {
        snprintf(errbuf, (size_t)errbuf_sz, "model has invalid embedding dimension: %d", n_embd);
        llama_model_free(model);
        return -1;
    }

    /* Determine context size: caller override > model metadata > fallback 512.
     * Capped at 8192 for memory sanity — embedding queries are typically short. */
    int n_ctx;
    if (n_ctx_override > 0) {
        n_ctx = n_ctx_override;
    } else {
        int n_ctx_train = (int)llama_model_n_ctx_train(model);
        n_ctx = (n_ctx_train > 0) ? n_ctx_train : 512;
        if (n_ctx > 8192)
            n_ctx = 8192;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)n_ctx;
    cparams.n_batch = (uint32_t)n_ctx;
    /* Encoder models require n_ubatch >= n_tokens because llama_encode()
     * processes the full sequence at once (no micro-batching possible). */
    cparams.n_ubatch = (uint32_t)n_ctx;
    cparams.embeddings = 1;
    /* Let each model's GGUF metadata control pooling type:
     *   BERT models (MiniLM, Nomic) → MEAN pooling
     *   Decoder models (Qwen3-Embedding) → LAST token pooling */
    cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        snprintf(errbuf, (size_t)errbuf_sz, "failed to create llama context for: %s", path);
        llama_model_free(model);
        return -1;
    }

    out->model = model;
    out->ctx = ctx;
    out->n_embd = n_embd;
    out->n_ctx = n_ctx;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Embedding Generation
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Generate an L2-normalized embedding for the given text.
 * Writes n_embd floats to out[]. Returns n_embd on success, -1 on error.
 */
static int embed_text(MuninnModelEntry *me, const char *text, float *out, int max_dim, char *errbuf, int errbuf_sz) {
    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);
    int text_len = (int)strlen(text);

    /* Tokenize — first call to determine token count */
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, 1, 1);
    if (n_tokens < 0)
        n_tokens = -n_tokens; /* returns negative count when buffer too small */

    if (n_tokens == 0) {
        snprintf(errbuf, (size_t)errbuf_sz, "tokenization produced 0 tokens");
        return -1;
    }

    if (n_tokens > me->n_ctx) {
        snprintf(errbuf, (size_t)errbuf_sz, "text too long: %d tokens exceeds context of %d", n_tokens, me->n_ctx);
        return -1;
    }

    llama_token *tokens = (llama_token *)malloc((size_t)n_tokens * sizeof(llama_token));
    if (!tokens) {
        snprintf(errbuf, (size_t)errbuf_sz, "out of memory allocating tokens");
        return -1;
    }

    int actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, 1, 1);
    if (actual < 0) {
        snprintf(errbuf, (size_t)errbuf_sz, "tokenization failed");
        free(tokens);
        return -1;
    }

    /* Create batch and encode */
    struct llama_batch batch = llama_batch_get_one(tokens, actual);

    int rc = llama_encode(me->ctx, batch);
    if (rc != 0) {
        snprintf(errbuf, (size_t)errbuf_sz, "llama_encode failed with code %d", rc);
        free(tokens);
        return -1;
    }

    /* Retrieve pooled embeddings (sequence 0) */
    float *emb = llama_get_embeddings_seq(me->ctx, 0);
    if (!emb) {
        /* Fallback: try getting embeddings for the last token */
        emb = llama_get_embeddings_ith(me->ctx, actual - 1);
    }
    if (!emb) {
        snprintf(errbuf, (size_t)errbuf_sz, "failed to retrieve embeddings");
        free(tokens);
        return -1;
    }

    int dim = me->n_embd;
    if (dim > max_dim)
        dim = max_dim;

    /* L2 normalize */
    float norm = 0.0f;
    for (int i = 0; i < dim; i++)
        norm += emb[i] * emb[i];

    if (norm > 0.0f) {
        norm = sqrtf(norm);
        for (int i = 0; i < dim; i++)
            out[i] = emb[i] / norm;
    } else {
        memcpy(out, emb, (size_t)dim * sizeof(float));
    }

    free(tokens);
    return dim;
}

/* ═══════════════════════════════════════════════════════════════════
 * SQL Scalar Functions
 * ═══════════════════════════════════════════════════════════════════ */

/* Pointer type tag for muninn_embed_model() */
static const char *EMBED_MODEL_PTR_TYPE = "muninn_embed_model";

/*
 * muninn_embed_model(path TEXT [, n_ctx INTEGER]) -> POINTER
 *
 * Load a GGUF model file and return an opaque pointer handle.
 * Optional n_ctx overrides the context window size (default: model metadata, capped at 8192).
 * Used with: INSERT INTO temp.muninn_models(name, model)
 *              SELECT 'name', muninn_embed_model('path.gguf');
 *              SELECT 'name', muninn_embed_model('path.gguf', 2048);
 */
static void fn_embed_model(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_embed_model: path must be TEXT", -1);
        return;
    }

    const char *path = (const char *)sqlite3_value_text(argv[0]);
    int n_ctx_override = 0; /* 0 = auto-detect from model metadata */
    if (argc >= 2 && sqlite3_value_type(argv[1]) == SQLITE_INTEGER) {
        n_ctx_override = sqlite3_value_int(argv[1]);
    }

    char errbuf[256];
    LoadedModel lm;

    if (load_gguf_model(path, n_ctx_override, &lm, errbuf, (int)sizeof(errbuf)) != 0) {
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    /* Pack into a heap-allocated struct that the VT will own */
    LoadedModel *heap = (LoadedModel *)sqlite3_malloc(sizeof(LoadedModel));
    if (!heap) {
        llama_free(lm.ctx);
        llama_model_free(lm.model);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *heap = lm;

    sqlite3_result_pointer(ctx, heap, EMBED_MODEL_PTR_TYPE, NULL);
}

/*
 * muninn_embed(model_name TEXT, text TEXT) -> BLOB
 *
 * Generate a float32 embedding blob for the given text.
 */
static void fn_embed(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT || sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_embed: expected (model_name TEXT, text TEXT)", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);

    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_EMBED);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_embed: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    float *buf = (float *)sqlite3_malloc(me->n_embd * (int)sizeof(float));
    if (!buf) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    char errbuf[256];
    int dim = embed_text(me, text, buf, me->n_embd, errbuf, (int)sizeof(errbuf));
    if (dim < 0) {
        sqlite3_free(buf);
        sqlite3_result_error(ctx, errbuf, -1);
        return;
    }

    sqlite3_result_blob(ctx, buf, dim * (int)sizeof(float), sqlite3_free);
}

/*
 * muninn_model_dim(model_name TEXT) -> INTEGER
 *
 * Return the embedding dimension of a loaded model.
 */
static void fn_model_dim(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_model_dim: model_name must be TEXT", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    MuninnModelEntry *me = muninn_registry_find_type(name, MUNINN_MODEL_EMBED);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_model_dim: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    sqlite3_result_int(ctx, me->n_embd);
}

/* ═══════════════════════════════════════════════════════════════════
 * muninn_models Eponymous Virtual Table
 *
 * Provides model lifecycle management:
 *   INSERT INTO temp.muninn_models(name, model)
 *     SELECT 'MiniLM', muninn_embed_model('path.gguf');
 *   DELETE FROM temp.muninn_models WHERE name = 'MiniLM';
 *   SELECT name, dim FROM temp.muninn_models;
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
} ModelsVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    int current; /* index into unified registry */
} ModelsCursor;

static int models_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab,
                          char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;

    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(name TEXT NOT NULL, model HIDDEN, dim INTEGER)");
    if (rc != SQLITE_OK)
        return rc;

    ModelsVtab *vtab = (ModelsVtab *)sqlite3_malloc(sizeof(ModelsVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(ModelsVtab));
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int models_disconnect(sqlite3_vtab *pVtab) {
    sqlite3_free(pVtab);
    return SQLITE_OK;
}

static int models_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pInfo) {
    (void)pVtab;

    /* Check for name = ? constraint */
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (pInfo->aConstraint[i].usable && pInfo->aConstraint[i].iColumn == 0 &&
            pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
            pInfo->aConstraintUsage[i].argvIndex = 1;
            pInfo->idxNum = 1;
            pInfo->estimatedCost = 1.0;
            return SQLITE_OK;
        }
    }

    pInfo->idxNum = 0;
    pInfo->estimatedCost = 100.0;
    return SQLITE_OK;
}

static int models_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVtab;
    ModelsCursor *cur = (ModelsCursor *)sqlite3_malloc(sizeof(ModelsCursor));
    if (!cur)
        return SQLITE_NOMEM;
    memset(cur, 0, sizeof(ModelsCursor));
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int models_close(sqlite3_vtab_cursor *pCursor) {
    sqlite3_free(pCursor);
    return SQLITE_OK;
}

/* Advance to the next in-use embed model slot */
static void models_advance(ModelsCursor *cur) {
    int cap = muninn_registry_capacity();
    while (cur->current < cap) {
        MuninnModelEntry *e = muninn_registry_at(cur->current);
        if (e && e->in_use && e->type == MUNINN_MODEL_EMBED)
            break;
        cur->current++;
    }
}

static int models_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    ModelsCursor *cur = (ModelsCursor *)pCursor;

    int cap = muninn_registry_capacity();
    if (idxNum == 1 && argc >= 1) {
        /* name = ? lookup */
        const char *name = (const char *)sqlite3_value_text(argv[0]);
        cur->current = cap; /* EOF by default */
        if (name) {
            for (int i = 0; i < cap; i++) {
                MuninnModelEntry *e = muninn_registry_at(i);
                if (e && e->in_use && e->type == MUNINN_MODEL_EMBED &&
                    strcmp(e->name, name) == 0) {
                    cur->current = i;
                    break;
                }
            }
        }
    } else {
        /* Full scan */
        cur->current = 0;
        models_advance(cur);
    }

    return SQLITE_OK;
}

static int models_next(sqlite3_vtab_cursor *pCursor) {
    ModelsCursor *cur = (ModelsCursor *)pCursor;
    cur->current++;
    models_advance(cur);
    return SQLITE_OK;
}

static int models_eof(sqlite3_vtab_cursor *pCursor) {
    return ((ModelsCursor *)pCursor)->current >= muninn_registry_capacity();
}

static int models_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    ModelsCursor *cur = (ModelsCursor *)pCursor;
    MuninnModelEntry *me = muninn_registry_at(cur->current);

    switch (col) {
    case 0: /* name */
        sqlite3_result_text(ctx, me->name, -1, SQLITE_TRANSIENT);
        break;
    case 1: /* model (hidden) */
        sqlite3_result_null(ctx);
        break;
    case 2: /* dim */
        sqlite3_result_int(ctx, me->n_embd);
        break;
    }
    return SQLITE_OK;
}

static int models_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((ModelsCursor *)pCursor)->current;
    return SQLITE_OK;
}

/*
 * xUpdate handles INSERT and DELETE on muninn_models.
 *
 * INSERT: argv[0]=NULL, argv[1]=NULL, argv[2]=name, argv[3]=model_ptr, argv[4]=dim(ignored)
 * DELETE: argv[0]=rowid
 */
static int models_update(sqlite3_vtab *pVtab, int argc, sqlite3_value **argv, sqlite3_int64 *pRowid) {
    (void)pVtab;

    int cap = muninn_registry_capacity();

    /* DELETE */
    if (argc == 1) {
        int idx = (int)sqlite3_value_int64(argv[0]);
        MuninnModelEntry *e = muninn_registry_at(idx);
        if (e && e->in_use && e->type == MUNINN_MODEL_EMBED) {
            muninn_registry_remove(e->name);
        }
        return SQLITE_OK;
    }

    /* INSERT (argc >= 3, argv[0]=NULL means INSERT) */
    if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        if (sqlite3_value_type(argv[2]) != SQLITE_TEXT) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_models: name must be TEXT");
            return SQLITE_ERROR;
        }

        const char *name = (const char *)sqlite3_value_text(argv[2]);

        /* Get the model pointer from muninn_embed_model() */
        LoadedModel *lm = (LoadedModel *)sqlite3_value_pointer(argv[3], EMBED_MODEL_PTR_TYPE);
        if (!lm) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_models: model column must be muninn_embed_model(path)");
            return SQLITE_ERROR;
        }

        int rc = muninn_registry_add(name, lm->model, lm->ctx, lm->n_embd, lm->n_ctx, MUNINN_MODEL_EMBED);
        if (rc == -1) {
            pVtab->zErrMsg = sqlite3_mprintf("muninn_models: model '%s' already loaded", name);
            return SQLITE_ERROR;
        }
        if (rc == -2) {
            llama_free(lm->ctx);
            llama_model_free(lm->model);
            pVtab->zErrMsg = sqlite3_mprintf("muninn_models: registry full (max %d models)", cap);
            return SQLITE_ERROR;
        }

        /* Ownership transferred to registry; zero out the LoadedModel
         * so the pointer destructor doesn't double-free */
        lm->model = NULL;
        lm->ctx = NULL;

        /* Return the rowid of the newly inserted model */
        for (int i = 0; i < cap; i++) {
            MuninnModelEntry *e = muninn_registry_at(i);
            if (e && e->in_use && strcmp(e->name, name) == 0) {
                *pRowid = i;
                break;
            }
        }

        return SQLITE_OK;
    }

    /* UPDATE — not supported */
    pVtab->zErrMsg = sqlite3_mprintf("muninn_models: UPDATE not supported");
    return SQLITE_ERROR;
}

static sqlite3_module models_module = {
    .iVersion = 0,
    .xCreate = NULL, /* eponymous-only: no xCreate means temp VT */
    .xConnect = models_connect,
    .xBestIndex = models_best_index,
    .xDisconnect = models_disconnect,
    .xDestroy = NULL,
    .xOpen = models_open,
    .xClose = models_close,
    .xFilter = models_filter,
    .xNext = models_next,
    .xEof = models_eof,
    .xColumn = models_column,
    .xRowid = models_rowid,
    .xUpdate = models_update,
    .xBegin = NULL,
    .xSync = NULL,
    .xCommit = NULL,
    .xRollback = NULL,
    .xFindFunction = NULL,
    .xRename = NULL,
    .xSavepoint = NULL,
    .xRelease = NULL,
    .xRollbackTo = NULL,
    .xShadowName = NULL,
};

/* ═══════════════════════════════════════════════════════════════════
 * Registration
 * ═══════════════════════════════════════════════════════════════════ */

int embed_register_functions(sqlite3 *db) {
    int rc;

    rc = sqlite3_create_function(db, "muninn_embed_model", -1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_embed_model,
                                 NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    rc = sqlite3_create_function(db, "muninn_embed", 2, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_embed, NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    rc = sqlite3_create_function(db, "muninn_model_dim", 1, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_model_dim, NULL,
                                 NULL);
    if (rc != SQLITE_OK)
        return rc;

    rc = sqlite3_create_module(db, "muninn_models", &models_module, NULL);
    if (rc != SQLITE_OK)
        return rc;

    return SQLITE_OK;
}
