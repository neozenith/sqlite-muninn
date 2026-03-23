/*
 * llama_common.c — Shared model registry, backend init, and tokenizer functions
 *
 * This file provides the unified infrastructure used by both llama_embed.c
 * and llama_chat.c:
 *   - Single llama.cpp backend initialization (idempotent)
 *   - Unified model registry (embed + chat models in one array)
 *   - Tokenizer SQL functions that work with any registered model
 */
#include "llama_common.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

SQLITE_EXTENSION_INIT3

#include "llama.h"
#include "yyjson.h"

/* ═══════════════════════════════════════════════════════════════════
 * Backend Initialization
 * ═══════════════════════════════════════════════════════════════════ */

static int g_backend_initialized = 0;
static enum ggml_log_level g_log_level = GGML_LOG_LEVEL_NONE;

/*
 * Log callback that filters by g_log_level.
 * Default level is NONE (silent). Set MUNINN_LOG_LEVEL=verbose|warn|error
 * to see llama.cpp output on stderr.
 */
static void muninn_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level >= g_log_level && g_log_level != GGML_LOG_LEVEL_NONE) {
        fputs(text, stderr);
    }
}

void muninn_ensure_backend(void) {
    if (!g_backend_initialized) {
        const char *env = getenv("MUNINN_LOG_LEVEL");
        if (env && strcmp(env, "verbose") == 0) {
            g_log_level = GGML_LOG_LEVEL_DEBUG;
        } else if (env && strcmp(env, "warn") == 0) {
            g_log_level = GGML_LOG_LEVEL_WARN;
        } else if (env && strcmp(env, "error") == 0) {
            g_log_level = GGML_LOG_LEVEL_ERROR;
        }
        llama_log_set(muninn_log_callback, NULL);
        llama_backend_init();
        g_backend_initialized = 1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Model Registry
 * ═══════════════════════════════════════════════════════════════════ */

static MuninnModelEntry g_models[MUNINN_MAX_MODELS];

MuninnModelEntry *muninn_registry_find(const char *name) {
    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (g_models[i].in_use && strcmp(g_models[i].name, name) == 0)
            return &g_models[i];
    }
    return NULL;
}

MuninnModelEntry *muninn_registry_find_type(const char *name, MuninnModelType type) {
    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (g_models[i].in_use && g_models[i].type == type && strcmp(g_models[i].name, name) == 0)
            return &g_models[i];
    }
    return NULL;
}

int muninn_registry_add(const char *name, struct llama_model *model, struct llama_context *ctx, int n_embd, int n_ctx,
                        MuninnModelType type) {
    if (muninn_registry_find(name) != NULL)
        return -1; /* duplicate name */

    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (!g_models[i].in_use) {
            g_models[i].model = model;
            g_models[i].ctx = ctx;
            g_models[i].n_embd = n_embd;
            g_models[i].n_ctx = n_ctx;
            g_models[i].type = type;
            snprintf(g_models[i].name, MUNINN_MAX_MODEL_NAME, "%s", name);
            g_models[i].in_use = 1;
            return 0;
        }
    }
    return -2; /* registry full */
}

void muninn_registry_remove(const char *name) {
    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (g_models[i].in_use && strcmp(g_models[i].name, name) == 0) {
            if (g_models[i].ctx)
                llama_free(g_models[i].ctx);
            if (g_models[i].model)
                llama_model_free(g_models[i].model);
            memset(&g_models[i], 0, sizeof(MuninnModelEntry));
            return;
        }
    }
}

int muninn_registry_capacity(void) {
    return MUNINN_MAX_MODELS;
}

MuninnModelEntry *muninn_registry_at(int index) {
    if (index < 0 || index >= MUNINN_MAX_MODELS)
        return NULL;
    return &g_models[index];
}

/* ═══════════════════════════════════════════════════════════════════
 * SQL Function: muninn_tokenize(model_name TEXT, text TEXT) -> TEXT (JSON)
 *
 * Returns token IDs as a JSON array. Works with any registered model
 * (embed or chat). Result has JSON subtype ('J').
 * ═══════════════════════════════════════════════════════════════════ */

static void fn_tokenize(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT || sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_tokenize: expected (model_name TEXT, text TEXT)", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);

    MuninnModelEntry *me = muninn_registry_find(name);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_tokenize: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);
    int text_len = (int)strlen(text);

    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, 1, 1);
    if (n_tokens < 0)
        n_tokens = -n_tokens;

    llama_token *tokens = (llama_token *)malloc((size_t)n_tokens * sizeof(llama_token));
    if (!tokens) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    int actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, 1, 1);
    if (actual < 0) {
        free(tokens);
        sqlite3_result_error(ctx, "muninn_tokenize: tokenization failed", -1);
        return;
    }

    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, arr);
    for (int i = 0; i < actual; i++) {
        yyjson_mut_arr_add_int(doc, arr, tokens[i]);
    }
    free(tokens);

    size_t json_len = 0;
    char *json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, &json_len);
    yyjson_mut_doc_free(doc);
    if (!json) {
        sqlite3_result_error(ctx, "muninn_tokenize: JSON serialization failed", -1);
        return;
    }
    sqlite3_result_text(ctx, json, (int)json_len, free);
    sqlite3_result_subtype(ctx, (unsigned int)'J');
}

/* ═══════════════════════════════════════════════════════════════════
 * SQL Function: muninn_tokenize_text(model_name TEXT, text TEXT) -> TEXT (JSON)
 *
 * Returns human-readable token pieces as a JSON array of strings.
 * Each token ID is converted back to its vocab piece via
 * llama_token_to_piece. Works with any registered model.
 * Result has JSON subtype ('J').
 * ═══════════════════════════════════════════════════════════════════ */

static void fn_tokenize_text(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT || sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_tokenize_text: expected (model_name TEXT, text TEXT)", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);

    MuninnModelEntry *me = muninn_registry_find(name);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_tokenize_text: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);
    int text_len = (int)strlen(text);

    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, 1, 1);
    if (n_tokens < 0)
        n_tokens = -n_tokens;

    llama_token *tokens = (llama_token *)malloc((size_t)n_tokens * sizeof(llama_token));
    if (!tokens) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    int actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, 1, 1);
    if (actual < 0) {
        free(tokens);
        sqlite3_result_error(ctx, "muninn_tokenize_text: tokenization failed", -1);
        return;
    }

    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, arr);

    char piece_buf[256];
    for (int i = 0; i < actual; i++) {
        int32_t piece_len = llama_token_to_piece(vocab, tokens[i], piece_buf, (int32_t)sizeof(piece_buf), 0, 1);
        if (piece_len < 0)
            piece_len = 0;
        yyjson_mut_arr_add_strncpy(doc, arr, piece_buf, (size_t)piece_len);
    }
    free(tokens);

    size_t json_len = 0;
    char *json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, &json_len);
    yyjson_mut_doc_free(doc);
    if (!json) {
        sqlite3_result_error(ctx, "muninn_tokenize_text: JSON serialization failed", -1);
        return;
    }
    sqlite3_result_text(ctx, json, (int)json_len, free);
    sqlite3_result_subtype(ctx, (unsigned int)'J');
}

/* ═══════════════════════════════════════════════════════════════════
 * SQL Function: muninn_token_count(model_name TEXT, text TEXT) -> INTEGER
 *
 * Lightweight token count — no buffer allocation, no JSON.
 * Works with any registered model.
 * ═══════════════════════════════════════════════════════════════════ */

static void fn_token_count(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;

    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT || sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
        sqlite3_result_error(ctx, "muninn_token_count: expected (model_name TEXT, text TEXT)", -1);
        return;
    }

    const char *name = (const char *)sqlite3_value_text(argv[0]);
    const char *text = (const char *)sqlite3_value_text(argv[1]);

    MuninnModelEntry *me = muninn_registry_find(name);
    if (!me) {
        char err[128];
        snprintf(err, sizeof(err), "muninn_token_count: model '%s' not loaded", name);
        sqlite3_result_error(ctx, err, -1);
        return;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(me->model);
    int n_tokens = llama_tokenize(vocab, text, (int)strlen(text), NULL, 0, 1, 1);
    if (n_tokens < 0)
        n_tokens = -n_tokens;

    sqlite3_result_int(ctx, n_tokens);
}

/* ═══════════════════════════════════════════════════════════════════
 * Registration
 * ═══════════════════════════════════════════════════════════════════ */

int common_register_functions(sqlite3 *db) {
    int rc;

    rc = sqlite3_create_function(db, "muninn_tokenize", 2, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_tokenize, NULL,
                                 NULL);
    if (rc != SQLITE_OK)
        return rc;

    rc = sqlite3_create_function(db, "muninn_tokenize_text", 2, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_tokenize_text,
                                 NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    rc = sqlite3_create_function(db, "muninn_token_count", 2, SQLITE_UTF8 | SQLITE_INNOCUOUS, NULL, fn_token_count,
                                 NULL, NULL);
    if (rc != SQLITE_OK)
        return rc;

    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Test Helpers (compiled only with -DMUNINN_TESTING)
 * ═══════════════════════════════════════════════════════════════════ */

#ifdef MUNINN_TESTING

void muninn_test_reset_backend(void) {
    g_backend_initialized = 0;
    g_log_level = GGML_LOG_LEVEL_NONE;
}

int muninn_test_inject_dummy(const char *name, MuninnModelType type) {
    return muninn_registry_add(name, NULL, NULL, 0, 0, type);
}

void muninn_test_remove_dummy(const char *name) {
    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (g_models[i].in_use && strcmp(g_models[i].name, name) == 0 && !g_models[i].model) {
            memset(&g_models[i], 0, sizeof(MuninnModelEntry));
            return;
        }
    }
}

void muninn_test_clear_all_dummies(void) {
    for (int i = 0; i < MUNINN_MAX_MODELS; i++) {
        if (g_models[i].in_use && !g_models[i].model) {
            memset(&g_models[i], 0, sizeof(MuninnModelEntry));
        }
    }
}

#endif /* MUNINN_TESTING */
