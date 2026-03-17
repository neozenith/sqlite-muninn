/*
 * llama_common.h — Shared model registry and tokenizer functions
 *
 * Provides the unified model registry used by both llama_embed.c (embedding
 * models) and llama_chat.c (chat/completion models). Tokenizer functions
 * work with any registered model regardless of type.
 *
 * SQL functions registered by common_register_functions():
 *   muninn_tokenize(model, text)      — Return JSON token ID array
 *   muninn_tokenize_text(model, text) — Return JSON token text pieces array
 *   muninn_token_count(model, text)   — Return token count as INTEGER
 */
#ifndef LLAMA_COMMON_H
#define LLAMA_COMMON_H

#include "sqlite3ext.h"

/* Forward declarations — avoids pulling in llama.h from every consumer */
struct llama_model;
struct llama_context;

/* ── Model Types ──────────────────────────────────────────────── */

typedef enum {
    MUNINN_MODEL_EMBED = 1,
    MUNINN_MODEL_CHAT = 2,
} MuninnModelType;

/* ── Model Registry Entry ─────────────────────────────────────── */

#define MUNINN_MAX_MODELS 16
#define MUNINN_MAX_MODEL_NAME 64

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    int n_embd;   /* embedding dimension (0 for chat models) */
    int n_ctx;    /* context window size */
    MuninnModelType type;
    char name[MUNINN_MAX_MODEL_NAME];
    int in_use;
} MuninnModelEntry;

/* ── Backend Initialization ───────────────────────────────────── */

/* Idempotent — safe to call from both embed and chat subsystems.
 * Reads MUNINN_LOG_LEVEL env var on first call. */
void muninn_ensure_backend(void);

/* ── Registry Functions ───────────────────────────────────────── */

/* Find any model by name (embed or chat) */
MuninnModelEntry *muninn_registry_find(const char *name);

/* Find a model by name, restricted to a specific type */
MuninnModelEntry *muninn_registry_find_type(const char *name, MuninnModelType type);

/* Add a model to the registry. Returns 0 on success, -1 duplicate, -2 full. */
int muninn_registry_add(const char *name, struct llama_model *model,
                        struct llama_context *ctx, int n_embd, int n_ctx,
                        MuninnModelType type);

/* Remove a model by name (frees model + context). */
void muninn_registry_remove(const char *name);

/* Iteration helpers for virtual tables */
int muninn_registry_capacity(void);
MuninnModelEntry *muninn_registry_at(int index);

/* ── SQL Registration ─────────────────────────────────────────── */

/* Register tokenizer SQL functions (muninn_tokenize, muninn_tokenize_text,
 * muninn_token_count). Called once from sqlite3_muninn_init(). */
int common_register_functions(sqlite3 *db);

/* ── Test Helpers ─────────────────────────────────────────────── */

#ifdef MUNINN_TESTING
void muninn_test_reset_backend(void);
int muninn_test_inject_dummy(const char *name, MuninnModelType type);
void muninn_test_remove_dummy(const char *name);
void muninn_test_clear_all_dummies(void);
#endif

#endif /* LLAMA_COMMON_H */
