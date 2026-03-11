/*
 * llama_chat.h — GGUF chat/completion inference via llama.cpp
 *
 * Provides autoregressive text generation with optional GBNF grammar
 * constraints using GGUF models loaded through llama.cpp's inference engine.
 *
 * SQL functions registered:
 *   muninn_chat_model(path [, n_ctx])               — Load GGUF chat model, return opaque handle
 *   muninn_chat(model, prompt [, grammar [, max_tokens]]) — Generate text completion
 *   muninn_extract_entities(model, text, labels_csv)      — NER with confidence scores [0.0-1.0]
 *   muninn_extract_relations(model, text, entities_json)  — RE with confidence scores [0.0-1.0]
 *   muninn_extract_ner_re(model, text, ent_labels, rel_labels) — Combined NER+RE in one pass
 *   muninn_summarize(model, text [, max_tokens])          — Summarise text
 *
 * Model registration via eponymous virtual table:
 *   INSERT INTO temp.muninn_chat_models(name, model) SELECT ...
 */
#ifndef LLAMA_CHAT_H
#define LLAMA_CHAT_H

#include "sqlite3ext.h"

/*
 * Register all GGUF chat functions and the muninn_chat_models
 * virtual table with SQLite. Returns SQLITE_OK on success.
 */
int chat_register_functions(sqlite3 *db);

/* Exposed for direct testing — pure string operations, no side effects */
const char *strip_think_block(const char *text);
const char *find_json_object(const char *text, int *out_len);

#ifdef LLAMA_CHAT_TESTING
void chat_test_reset_backend(void);
int chat_test_inject_dummy(const char *name);
void chat_test_remove_dummy(const char *name);
void chat_test_clear_all_dummies(void);
int chat_test_max_models(void);
#endif

#endif /* LLAMA_CHAT_H */
