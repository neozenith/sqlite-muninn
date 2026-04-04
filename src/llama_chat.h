/*
 * llama_chat.h — GGUF chat/completion inference via llama.cpp
 *
 * Provides autoregressive text generation with optional GBNF grammar
 * constraints using GGUF models loaded through llama.cpp's inference engine.
 *
 * SQL functions registered:
 *   muninn_chat_model(path [, n_ctx])                              — Load GGUF chat model
 *   muninn_chat(model, prompt [, grammar [, max_tokens]])          — Generate text completion
 *   muninn_extract_entities(model, text [, labels [, skip_think]]) — NER (supervised/unsupervised)
 *   muninn_extract_relations(model, text [, entities [, skip_think]]) — RE (supervised/unsupervised)
 *   muninn_extract_ner_re(model, text [, ent_labels, rel_labels [, skip_think]]) — NER+RE
 *   muninn_extract_entities_batch(model, texts_json [, labels [, batch_size]])   — Batch NER
 *   muninn_extract_ner_re_batch(model, texts_json [, ent, rel [, batch_size]])   — Batch NER+RE
 *   muninn_summarize(model, text [, max_tokens])                   — Summarise text
 *
 * Omitting labels enables unsupervised (open) extraction — the LLM discovers
 * entity types and relation types from the text itself.
 *
 * Model registration via eponymous virtual table:
 *   INSERT INTO temp.muninn_chat_models(name, model) SELECT ...
 */
#ifndef LLAMA_CHAT_H
#define LLAMA_CHAT_H

#include "sqlite3ext.h"
#include "llama_common.h"

/*
 * Register all GGUF chat functions and the muninn_chat_models
 * virtual table with SQLite. Returns SQLITE_OK on success.
 */
int chat_register_functions(sqlite3 *db);

/* Exposed for direct testing — pure string operations, no side effects */
const char *strip_think_block(const char *text);

/* Exposed for llama_label_groups.c and er.c — chat generation internals */
char *format_chat_messages(MuninnModelEntry *me, const char *system_msg, const char *user_msg, int inject_skip_think,
                           int *out_len);
int chat_generate(MuninnModelEntry *me, const char *prompt, const char *grammar_gbnf, int max_tokens, char **out_text,
                  int *out_len, char errbuf[], int errbuf_sz);

#endif /* LLAMA_CHAT_H */
