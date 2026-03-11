/*
 * llama_embed.h — GGUF model embedding via llama.cpp
 *
 * Provides text embedding, tokenization, and model management
 * using GGUF models loaded through llama.cpp's inference engine.
 *
 * SQL functions registered:
 *   muninn_embed_model(path)        — Load GGUF model, return opaque handle
 *   muninn_embed(model, text)       — Generate float32 embedding blob
 *   muninn_model_dim(model)         — Query embedding dimension
 *   muninn_tokenize(model, text)    — Return JSON token array
 *   muninn_token_count(model, text) — Return token count as INTEGER
 *
 * Model registration via eponymous virtual table:
 *   INSERT INTO temp.muninn_models(name, model) SELECT ...
 */
#ifndef LLAMA_EMBED_H
#define LLAMA_EMBED_H

#include "sqlite3ext.h"

/*
 * Register all GGUF embedding functions and the muninn_models
 * virtual table with SQLite. Returns SQLITE_OK on success.
 */
int embed_register_functions(sqlite3 *db);

#endif /* LLAMA_EMBED_H */
