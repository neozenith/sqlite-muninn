/*
 * test_llama_chat.c — Unit + integration tests for the GGUF chat subsystem
 *
 * Always-run tests: registration, error handling, VT schema, string helpers,
 *                   NULL argument handling, logging, registry limits.
 * Model-gated tests: full integration when models/Qwen3-*.gguf exists.
 * Each model is loaded ONCE into the process-global registry and
 * reused across all integration tests to avoid OOM.
 *
 * The test runner is invoked from the project root, so model paths
 * are relative: "models/Qwen3-4B-Q4_K_M.gguf" etc.
 */
#include "test_common.h"
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int chat_register_functions(sqlite3 *db);

/* Exposed helpers from llama_chat.c */
extern const char *strip_think_block(const char *text);

/* Unified test helpers from llama_common.c (compiled with -DMUNINN_TESTING) */
typedef enum {
    MUNINN_MODEL_EMBED = 1,
    MUNINN_MODEL_CHAT = 2,
} MuninnModelType;
extern void muninn_test_reset_backend(void);
extern int muninn_test_inject_dummy(const char *name, MuninnModelType type);
extern void muninn_test_remove_dummy(const char *name);
extern void muninn_test_clear_all_dummies(void);
extern int muninn_registry_capacity(void);

/* ── Model discovery ─────────────────────────────────────────────
 * Check for GGUF models in models/ at suite start. Integration
 * tests run for every model that exists on disk. */

typedef struct {
    const char *path;
    const char *name;
    int available;
} TestModel;

static TestModel g_test_models[] = {
    {"models/Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B", 0},
    {"models/Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B", 0},
};
#define N_TEST_MODELS (int)(sizeof(g_test_models) / sizeof(g_test_models[0]))

static int g_any_model = 0;

static void discover_models(void) {
    for (int i = 0; i < N_TEST_MODELS; i++) {
        FILE *f = fopen(g_test_models[i].path, "rb");
        if (f) {
            fclose(f);
            g_test_models[i].available = 1;
            g_any_model = 1;
            printf("  (model found: %s)\n", g_test_models[i].path);
        }
    }
    if (!g_any_model) {
        printf("  (no GGUF models in models/ — integration tests skipped)\n");
    }
}

/* ── Helper: open DB with functions registered ────────────────── */

static sqlite3 *open_test_db(void) {
    sqlite3 *db = NULL;
    sqlite3_open(":memory:", &db);
    chat_register_functions(db);
    return db;
}

/* ── Helper: load model into global registry (via VT INSERT) ──── */

static int load_model_into_registry(sqlite3 *db, const char *name, const char *path) {
    char sql[512];
    snprintf(sql, sizeof(sql),
             "INSERT INTO temp.muninn_chat_models(name, model) "
             "SELECT '%s', muninn_chat_model('%s')",
             name, path);

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (errmsg) {
        fprintf(stderr, "    load error: %s\n", errmsg);
        sqlite3_free(errmsg);
    }
    return rc;
}

/* ═══════════════════════════════════════════════════════════════
 * UNIT TESTS: strip_think_block
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_strip_think_block_with_tag) {
    /* Text with a <think>...</think> block should return pointer past the tag */
    const char *input = "<think>some reasoning</think>actual content";
    const char *result = strip_think_block(input);
    ASSERT(result != input);
    ASSERT(strcmp(result, "actual content") == 0);
}

TEST(test_strip_think_block_with_whitespace) {
    /* Whitespace after closing tag should be skipped */
    const char *input = "<think>reasoning</think>  \n\tcontent";
    const char *result = strip_think_block(input);
    ASSERT(strcmp(result, "content") == 0);
}

TEST(test_strip_think_block_no_tag) {
    /* No think block → returns original pointer */
    const char *input = "just plain text without any think tags";
    const char *result = strip_think_block(input);
    ASSERT(result == input);
}

TEST(test_strip_think_block_empty) {
    const char *input = "";
    const char *result = strip_think_block(input);
    ASSERT(result == input);
}

TEST(test_strip_think_block_truncated) {
    /* Open <think> without </think> → truncated reasoning, return empty */
    const char *input = "<think>open without close due to max_tokens";
    const char *result = strip_think_block(input);
    ASSERT(result != input);
    ASSERT(strlen(result) == 0);
}

TEST(test_strip_think_block_not_at_start) {
    /* <think> not at start → not a think block prefix, return original */
    const char *input = "prefix <think>reasoning";
    const char *result = strip_think_block(input);
    ASSERT(result == input);
}

/* ═══════════════════════════════════════════════════════════════
 * ALWAYS-RUN TESTS: Registration and Error Handling
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_chat_register_functions) {
    sqlite3 *db = open_test_db();

    /* muninn_chat_model registered (wrong arg type → error, not "no such function") */
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat_model(42)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* muninn_chat registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_chat('x', 'y')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* muninn_extract_entities registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('x', 'y', 'z')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* muninn_extract_relations registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_relations('x', 'y', 'z')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* muninn_summarize registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_summarize('x', 'y')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_models_vtab_exists) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT name, n_ctx FROM muninn_chat_models", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_DONE, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_unloaded_model_error) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat('nonexistent', 'hello')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_model_bad_path) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat_model('/nonexistent/path.gguf')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "failed to load") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_model_with_ctx_arg) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat_model('/nonexistent/path.gguf', 4096)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "failed to load") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_extract_entities_unloaded) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(
        db, "SELECT muninn_extract_entities('nope', 'Alice works at ACME', 'person,organization')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_extract_relations_unloaded) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc =
        sqlite3_prepare_v2(db, "SELECT muninn_extract_relations('nope', 'Alice works at ACME', '[]')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_summarize_unloaded) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_summarize('nope', 'some text')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_models_insert_bad_pointer) {
    sqlite3 *db = open_test_db();

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, "INSERT INTO temp.muninn_chat_models(name, model) VALUES ('test', 'not_a_pointer')", NULL,
                          NULL, &errmsg);
    ASSERT(rc != SQLITE_OK);
    if (errmsg)
        sqlite3_free(errmsg);

    sqlite3_close(db);
}

TEST(test_chat_models_insert_empty_name) {
    sqlite3 *db = open_test_db();

    char *errmsg = NULL;
    int rc = sqlite3_exec(db,
                          "INSERT INTO temp.muninn_chat_models(name, model) "
                          "SELECT '', muninn_chat_model('/nonexistent.gguf')",
                          NULL, NULL, &errmsg);
    ASSERT(rc != SQLITE_OK);
    if (errmsg)
        sqlite3_free(errmsg);

    sqlite3_close(db);
}

/* ═══════════════════════════════════════════════════════════════
 * NULL ARGUMENT TESTS
 *
 * SQL functions should handle NULL args gracefully with errors.
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_chat_single_arg) {
    /* muninn_chat with only 1 arg → "requires (model, prompt)" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat('model')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_null_model) {
    /* muninn_chat(NULL, 'prompt') → NULL name error */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat(NULL, 'prompt')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_chat_null_prompt) {
    /* muninn_chat('model', NULL) → NULL prompt error */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_chat('model', NULL)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_entities_null_args) {
    sqlite3 *db = open_test_db();

    /* NULL model name */
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities(NULL, 'text', 'person')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    /* NULL text */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('model', NULL, 'person')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* NULL labels */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('model', 'text', NULL)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_relations_null_args) {
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_relations(NULL, 'text', '[]')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_null_args) {
    sqlite3 *db = open_test_db();

    /* NULL model */
    sqlite3_stmt *stmt = NULL;
    int rc =
        sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re(NULL, 'text', 'person', 'produces')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    /* NULL text */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('model', NULL, 'person', 'produces')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* NULL entity labels */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('model', 'text', NULL, 'produces')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* NULL relation labels */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('model', 'text', 'person', NULL)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Unloaded model */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('nonexistent', 'text', 'person', 'produces')", -1, &stmt,
                            NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_summarize_null_args) {
    sqlite3 *db = open_test_db();

    /* NULL model */
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_summarize(NULL, 'text')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "must be TEXT") != NULL);
    sqlite3_finalize(stmt);

    /* NULL text */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_summarize('model', NULL)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

/* ═══════════════════════════════════════════════════════════════
 * UNSUPERVISED MODE TESTS
 *
 * Verify that extract functions accept 2-arg form (model, text)
 * for unsupervised mode. Without a loaded model these hit the
 * "not loaded" error — confirming they passed arg validation.
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_entities_unsupervised_2arg) {
    /* muninn_extract_entities('model', 'text') → unsupervised → "not loaded" (not "requires") */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('nope', 'Alice works at ACME')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_entities_unsupervised_with_skip_think) {
    /* muninn_extract_entities('model', 'text', 1) → unsupervised + skip_think → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc =
        sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('nope', 'Alice works at ACME', 1)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_relations_unsupervised_2arg) {
    /* muninn_extract_relations('model', 'text') → unsupervised → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_relations('nope', 'Alice founded ACME')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_relations_unsupervised_with_skip_think) {
    /* muninn_extract_relations('model', 'text', 1) → unsupervised + skip_think → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc =
        sqlite3_prepare_v2(db, "SELECT muninn_extract_relations('nope', 'Alice founded ACME', 1)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_unsupervised_2arg) {
    /* muninn_extract_ner_re('model', 'text') → unsupervised → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('nope', 'Alice founded ACME')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_unsupervised_with_skip_think) {
    /* muninn_extract_ner_re('model', 'text', 1) → unsupervised + skip_think → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('nope', 'Alice founded ACME', 1)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_mixed_labels_error) {
    /* muninn_extract_ner_re('model', 'text', 'person') → only entity labels → error */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re('nope', 'text', 'person')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "requires both") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_entities_batch_unsupervised) {
    /* muninn_extract_entities_batch('model', texts_json) → unsupervised → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities_batch('nope', '[\"Alice works at ACME\"]')", -1,
                                &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_batch_unsupervised) {
    /* muninn_extract_ner_re_batch('model', texts_json) → unsupervised → "not loaded" */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re_batch('nope', '[\"Alice founded ACME\"]')", -1, &stmt,
                                NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "not loaded") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

TEST(test_ner_re_batch_mixed_labels_error) {
    /* muninn_extract_ner_re_batch('model', texts, 'person') → only entity labels → error */
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    int rc =
        sqlite3_prepare_v2(db, "SELECT muninn_extract_ner_re_batch('nope', '[\"text\"]', 'person')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "requires both") != NULL);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

/* ═══════════════════════════════════════════════════════════════
 * LOGGING TESTS
 *
 * Test the MUNINN_LOG_LEVEL env var parsing. Uses reset helper
 * to re-trigger ensure_backend_chat() with different values.
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_log_level_verbose) {
    muninn_test_reset_backend();
    setenv("MUNINN_LOG_LEVEL", "verbose", 1);

    sqlite3 *db = open_test_db();
    /* Trigger ensure_backend_chat by calling a function that requires it */
    sqlite3_stmt *stmt = NULL;
    sqlite3_prepare_v2(db, "SELECT muninn_chat_model('/nonexistent.gguf')", -1, &stmt, NULL);
    sqlite3_step(stmt); /* will fail with "failed to load" but backend is now init'd */
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    unsetenv("MUNINN_LOG_LEVEL");
}

TEST(test_log_level_warn) {
    muninn_test_reset_backend();
    setenv("MUNINN_LOG_LEVEL", "warn", 1);

    sqlite3 *db = open_test_db();
    sqlite3_stmt *stmt = NULL;
    sqlite3_prepare_v2(db, "SELECT muninn_chat_model('/nonexistent.gguf')", -1, &stmt, NULL);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    unsetenv("MUNINN_LOG_LEVEL");
}

TEST(test_log_level_error) {
    muninn_test_reset_backend();
    setenv("MUNINN_LOG_LEVEL", "error", 1);

    sqlite3 *db = open_test_db();
    sqlite3_stmt *stmt = NULL;
    sqlite3_prepare_v2(db, "SELECT muninn_chat_model('/nonexistent.gguf')", -1, &stmt, NULL);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    /* Clean up: reset to default (no logging) */
    muninn_test_reset_backend();
    unsetenv("MUNINN_LOG_LEVEL");
}

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY TESTS (using test helpers)
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_registry_duplicate) {
    /* Injecting a dummy with the same name should fail */
    int rc = muninn_test_inject_dummy("__dup_test__", MUNINN_MODEL_CHAT);
    ASSERT_EQ_INT(0, rc);

    rc = muninn_test_inject_dummy("__dup_test__", MUNINN_MODEL_CHAT);
    ASSERT_EQ_INT(-1, rc);

    muninn_test_remove_dummy("__dup_test__");
}

TEST(test_registry_full) {
    /* Fill registry to max capacity, then verify overflow returns -2 */
    int max = muninn_registry_capacity();
    char name[64];

    for (int i = 0; i < max; i++) {
        snprintf(name, sizeof(name), "__full_test_%d__", i);
        muninn_test_inject_dummy(name, MUNINN_MODEL_CHAT);
    }

    int rc = muninn_test_inject_dummy("__overflow__", MUNINN_MODEL_CHAT);
    ASSERT_EQ_INT(-2, rc);

    muninn_test_clear_all_dummies();
}

/* ═══════════════════════════════════════════════════════════════
 * VT EDGE CASE TESTS
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_vtab_select_hidden_column) {
    /* Selecting the HIDDEN model column should return NULL */
    sqlite3 *db = open_test_db();

    /* Add a dummy to have a row to select */
    muninn_test_inject_dummy("__hidden_test__", MUNINN_MODEL_CHAT);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, "SELECT name, model, n_ctx FROM muninn_chat_models", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    int found = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *row_name = (const char *)sqlite3_column_text(stmt, 0);
        if (row_name && strcmp(row_name, "__hidden_test__") == 0) {
            found = 1;
            /* model column (HIDDEN) should be NULL */
            ASSERT_EQ_INT(SQLITE_NULL, sqlite3_column_type(stmt, 1));
        }
    }
    ASSERT(found);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    muninn_test_remove_dummy("__hidden_test__");
}

TEST(test_vtab_update_unsupported) {
    /* UPDATE on the VT should return an error */
    sqlite3 *db = open_test_db();

    /* Need a row to UPDATE */
    muninn_test_inject_dummy("__update_test__", MUNINN_MODEL_CHAT);

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, "UPDATE muninn_chat_models SET name = 'new' WHERE name = '__update_test__'", NULL, NULL,
                          &errmsg);
    /* Should fail — we don't support UPDATE */
    ASSERT(rc != SQLITE_OK);
    if (errmsg)
        sqlite3_free(errmsg);

    sqlite3_close(db);
    muninn_test_remove_dummy("__update_test__");
}

/* ═══════════════════════════════════════════════════════════════
 * MODEL-GATED INTEGRATION TESTS
 *
 * Each model is loaded ONCE into the process-global chat registry
 * before all integration tests. Individual tests open fresh DBs
 * and use the model by name (the registry is process-global, so
 * any DB connection can use any loaded model).
 * ═══════════════════════════════════════════════════════════════ */

/* ─── Integration: VT shows loaded model ───────────────────────── */

static void integ_vtab_select(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    sqlite3_stmt *stmt = NULL;
    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT name, n_ctx FROM muninn_chat_models WHERE name = '%s'", name);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Scan all rows and find our model (eponymous VT may not push WHERE down) */
    int found = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *row_name = (const char *)sqlite3_column_text(stmt, 0);
        if (row_name && strcmp(row_name, name) == 0) {
            found = 1;
            int n_ctx = sqlite3_column_int(stmt, 1);
            ASSERT(n_ctx > 0);
        }
    }
    ASSERT(found);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: duplicate INSERT rejected ───────────────────── */

static void integ_vtab_duplicate(const char *name, const char *path) {
    sqlite3 *db = open_test_db();

    /* Model is already in the global registry — INSERT again should fail */
    int rc = load_model_into_registry(db, name, path);
    ASSERT(rc != SQLITE_OK);

    sqlite3_close(db);
}

/* ─── Integration: plain chat ──────────────────────────────────── */

static void integ_chat_plain(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT muninn_chat('%s', 'What is 2+2? Answer with just the number.')", name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(strlen(result) > 0);
    ASSERT(strstr(result, "4") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: chat with grammar ───────────────────────────── */

static void integ_chat_grammar(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[512];
    snprintf(sql, sizeof(sql),
             "SELECT muninn_chat('%s', 'Is the sky blue? Answer yes or no.', "
             "'root ::= \"yes\" | \"no\"')",
             name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(strcmp(result, "yes") == 0 || strcmp(result, "no") == 0);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: chat with max_tokens ────────────────────────── */

static void integ_chat_max_tokens(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT muninn_chat('%s', 'Tell me a long story about dragons.', NULL, 5)", name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(strlen(result) < 200);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: NER extraction ──────────────────────────────── */

static void integ_extract_entities(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[512];
    snprintf(sql, sizeof(sql),
             "SELECT muninn_extract_entities('%s', "
             "'Alice Smith founded ACME Corporation in New York City.', "
             "'person,organization,location')",
             name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(result[0] == '{');
    ASSERT(strstr(result, "\"entities\"") != NULL);
    /* Should find at least one entity */
    ASSERT(strstr(result, "\"text\"") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: RE extraction ───────────────────────────────── */

static void integ_extract_relations(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[512];
    snprintf(sql, sizeof(sql),
             "SELECT muninn_extract_relations('%s', "
             "'Alice founded ACME Corporation.', "
             "'[{\"text\":\"Alice\",\"type\":\"person\"},"
             "{\"text\":\"ACME Corporation\",\"type\":\"organization\"}]')",
             name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(result[0] == '{');
    ASSERT(strstr(result, "\"relations\"") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: summarization ───────────────────────────────── */

static void integ_summarize(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[512];
    snprintf(sql, sizeof(sql),
             "SELECT muninn_summarize('%s', "
             "'Alice Smith is the CEO of ACME Corporation. She founded the company in 1987.', 64)",
             name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(strlen(result) > 0);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: summarize with default max_tokens ───────────── */

static void integ_summarize_default(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT muninn_summarize('%s', 'The quick brown fox jumps over the lazy dog.')", name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(strlen(result) > 0);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: Combined NER+RE in one pass ─────────────────── */

static void integ_extract_ner_re(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char sql[512];
    snprintf(sql, sizeof(sql),
             "SELECT muninn_extract_ner_re('%s', "
             "'Alice Smith founded ACME Corporation in New York City in 1987.', "
             "'person,organization,location', 'founded,located_in')",
             name);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *result = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(result != NULL);
    ASSERT(result[0] == '{');
    /* Must contain both entities and relations arrays */
    ASSERT(strstr(result, "\"entities\"") != NULL);
    ASSERT(strstr(result, "\"relations\"") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: NER→RE pipeline in SQL CTE ──────────────────── */

static void integ_ner_re_pipeline(const char *name, const char *path) {
    (void)path;
    sqlite3 *db = open_test_db();

    char *errmsg = NULL;
    int rc = sqlite3_exec(db,
                          "CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT);"
                          "INSERT INTO docs VALUES (1, 'Alice founded ACME Corporation in New York City.');",
                          NULL, NULL, &errmsg);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    if (errmsg)
        sqlite3_free(errmsg);

    char sql[1024];
    snprintf(sql, sizeof(sql),
             "WITH ner AS ("
             "  SELECT id, content, "
             "    muninn_extract_entities('%s', content, 'person,organization,location') AS ents "
             "  FROM docs"
             ") "
             "SELECT id, muninn_extract_relations('%s', content, ents) AS rels FROM ner",
             name, name);

    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    ASSERT_EQ_INT(1, sqlite3_column_int(stmt, 0));
    const char *rels = (const char *)sqlite3_column_text(stmt, 1);
    ASSERT(rels != NULL);
    ASSERT(rels[0] == '{');
    ASSERT(strstr(rels, "\"relations\"") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Integration: VT DELETE + chat_registry_remove ────────────── */

static void integ_vtab_delete(const char *name, const char *path) {
    /*
     * Load the same model under a temporary name, then DELETE it.
     * This exercises the VT DELETE path and chat_registry_remove().
     */
    sqlite3 *db = open_test_db();
    char tmp_name[128];
    snprintf(tmp_name, sizeof(tmp_name), "%s_del_test", name);

    /* Load model under temporary name */
    int rc = load_model_into_registry(db, tmp_name, path);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Verify it appears in the VT */
    sqlite3_stmt *stmt = NULL;
    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT rowid, name FROM muninn_chat_models WHERE name = '%s'", tmp_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    int found = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *row_name = (const char *)sqlite3_column_text(stmt, 1);
        if (row_name && strcmp(row_name, tmp_name) == 0) {
            found = 1;
        }
    }
    ASSERT(found);
    sqlite3_finalize(stmt);

    /* DELETE the model */
    snprintf(sql, sizeof(sql), "DELETE FROM muninn_chat_models WHERE name = '%s'", tmp_name);
    char *errmsg = NULL;
    rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    if (errmsg)
        sqlite3_free(errmsg);

    /* Verify it's gone */
    snprintf(sql, sizeof(sql), "SELECT name FROM muninn_chat_models WHERE name = '%s'", tmp_name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    found = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *row_name = (const char *)sqlite3_column_text(stmt, 0);
        if (row_name && strcmp(row_name, tmp_name) == 0)
            found = 1;
    }
    ASSERT(!found);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

/* ─── Integration: VT INSERT with empty name (model-gated) ──── */

static void integ_vtab_insert_empty_name(const char *name, const char *path) {
    (void)name;
    sqlite3 *db = open_test_db();

    /* INSERT with empty name should fail at VT level */
    char sql[512];
    snprintf(sql, sizeof(sql),
             "INSERT INTO temp.muninn_chat_models(name, model) "
             "SELECT '', muninn_chat_model('%s')",
             path);

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    ASSERT(rc != SQLITE_OK);
    if (errmsg) {
        ASSERT(strstr(errmsg, "non-empty") != NULL);
        sqlite3_free(errmsg);
    }

    sqlite3_close(db);
}

/* ─── Integration: VT INSERT when registry is full ─────────────── */

static void integ_vtab_registry_full(const char *name, const char *path) {
    (void)name;
    sqlite3 *db = open_test_db();

    /* Fill remaining slots with dummies */
    int max = muninn_registry_capacity();
    for (int i = 0; i < max; i++) {
        char dname[64];
        snprintf(dname, sizeof(dname), "__full_%d__", i);
        muninn_test_inject_dummy(dname, MUNINN_MODEL_CHAT);
    }

    /* INSERT should fail with "registry full" */
    char sql[512];
    snprintf(sql, sizeof(sql),
             "INSERT INTO temp.muninn_chat_models(name, model) "
             "SELECT '__overflow__', muninn_chat_model('%s')",
             path);

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    ASSERT(rc != SQLITE_OK);
    if (errmsg) {
        ASSERT(strstr(errmsg, "full") != NULL || strstr(errmsg, "already") != NULL);
        sqlite3_free(errmsg);
    }

    /* Clean up dummies (keep real models) */
    muninn_test_clear_all_dummies();

    sqlite3_close(db);
}

/* ─── Integration: context overflow with tiny n_ctx ────────────── */

static void integ_context_overflow(const char *name, const char *path) {
    /*
     * Load model with a tiny context window (128 tokens), then send
     * a prompt that far exceeds it. This exercises the "prompt exceeds
     * context" error path in chat_generate and its propagation to
     * fn_chat, fn_extract_entities, fn_extract_relations, fn_summarize.
     */
    (void)name;
    sqlite3 *db = open_test_db();

    /* Load model with n_ctx=128 */
    char sql[512];
    snprintf(sql, sizeof(sql),
             "INSERT INTO temp.muninn_chat_models(name, model) "
             "SELECT '__tiny_ctx__', muninn_chat_model('%s', 128)",
             path);

    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        /* Model might not support context < 128, skip */
        if (errmsg)
            sqlite3_free(errmsg);
        sqlite3_close(db);
        return;
    }

    /* Build a long prompt (~10KB) that will exceed 128 tokens */
    char long_text[10240];
    memset(long_text, 0, sizeof(long_text));
    for (int i = 0; i < 50; i++) {
        strcat(long_text, "The quick brown fox jumps over the lazy dog in the park. "
                          "Alice and Bob went to the store to buy some groceries today. ");
    }

    sqlite3_stmt *stmt = NULL;

    /* Test 1: muninn_chat with overflow */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_chat('__tiny_ctx__', ?)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    sqlite3_bind_text(stmt, 1, long_text, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    ASSERT(strstr(sqlite3_errmsg(db), "exceeds") != NULL || strstr(sqlite3_errmsg(db), "failed") != NULL);
    sqlite3_finalize(stmt);

    /* Test 2: muninn_extract_entities with overflow */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_entities('__tiny_ctx__', ?, 'person')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    sqlite3_bind_text(stmt, 1, long_text, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Test 3: muninn_extract_relations with overflow */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_extract_relations('__tiny_ctx__', ?, '[]')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    sqlite3_bind_text(stmt, 1, long_text, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Test 4: muninn_summarize with overflow */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_summarize('__tiny_ctx__', ?)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    sqlite3_bind_text(stmt, 1, long_text, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Clean up: DELETE the tiny-ctx model */
    rc = sqlite3_exec(db, "DELETE FROM muninn_chat_models WHERE name = '__tiny_ctx__'", NULL, NULL, &errmsg);
    if (errmsg)
        sqlite3_free(errmsg);

    sqlite3_close(db);
}

/* ── Wrapper macro: run integration test for a specific model ──── */

#define INTEG(fn, idx)                                                                                                 \
    do {                                                                                                               \
        if (g_test_models[idx].available) {                                                                            \
            char _label[128];                                                                                          \
            snprintf(_label, sizeof(_label), "%s[%s]", #fn, g_test_models[idx].name);                                  \
            test_begin(_label);                                                                                        \
            fn(g_test_models[idx].name, g_test_models[idx].path);                                                      \
            if (!test_failed_flag())                                                                                   \
                test_pass();                                                                                           \
        }                                                                                                              \
    } while (0)

/* ═══════════════════════════════════════════════════════════════
 * Suite entry point
 * ═══════════════════════════════════════════════════════════════ */

void test_llama_chat(void) {
    /* ── Unit tests (no model, no SQLite) ──────────────────── */
    RUN_TEST(test_strip_think_block_with_tag);
    RUN_TEST(test_strip_think_block_with_whitespace);
    RUN_TEST(test_strip_think_block_no_tag);
    RUN_TEST(test_strip_think_block_empty);
    RUN_TEST(test_strip_think_block_truncated);
    RUN_TEST(test_strip_think_block_not_at_start);

    /* ── Logging env var tests ─────────────────────────────── */
    RUN_TEST(test_log_level_verbose);
    RUN_TEST(test_log_level_warn);
    RUN_TEST(test_log_level_error);

    /* ── Registry tests ────────────────────────────────────── */
    RUN_TEST(test_registry_duplicate);
    RUN_TEST(test_registry_full);

    /* ── Registration + error tests (no model) ─────────────── */
    RUN_TEST(test_chat_register_functions);
    RUN_TEST(test_chat_models_vtab_exists);
    RUN_TEST(test_chat_unloaded_model_error);
    RUN_TEST(test_chat_model_bad_path);
    RUN_TEST(test_chat_model_with_ctx_arg);
    RUN_TEST(test_extract_entities_unloaded);
    RUN_TEST(test_extract_relations_unloaded);
    RUN_TEST(test_summarize_unloaded);
    RUN_TEST(test_chat_models_insert_bad_pointer);
    RUN_TEST(test_chat_models_insert_empty_name);

    /* ── NULL argument tests ───────────────────────────────── */
    RUN_TEST(test_chat_single_arg);
    RUN_TEST(test_chat_null_model);
    RUN_TEST(test_chat_null_prompt);
    RUN_TEST(test_entities_null_args);
    RUN_TEST(test_relations_null_args);
    RUN_TEST(test_ner_re_null_args);
    RUN_TEST(test_summarize_null_args);

    /* ── Unsupervised mode tests ──────────────────────────── */
    RUN_TEST(test_entities_unsupervised_2arg);
    RUN_TEST(test_entities_unsupervised_with_skip_think);
    RUN_TEST(test_relations_unsupervised_2arg);
    RUN_TEST(test_relations_unsupervised_with_skip_think);
    RUN_TEST(test_ner_re_unsupervised_2arg);
    RUN_TEST(test_ner_re_unsupervised_with_skip_think);
    RUN_TEST(test_ner_re_mixed_labels_error);
    RUN_TEST(test_entities_batch_unsupervised);
    RUN_TEST(test_ner_re_batch_unsupervised);
    RUN_TEST(test_ner_re_batch_mixed_labels_error);

    /* ── VT edge cases ─────────────────────────────────────── */
    RUN_TEST(test_vtab_select_hidden_column);
    RUN_TEST(test_vtab_update_unsupported);

    /* ── Model-gated integration tests ─────────────────────── */
    discover_models();
    if (!g_any_model)
        return;

    /*
     * Load each available model ONCE into the process-global registry.
     * The model stays loaded across all subsequent tests — no need to
     * reload per-test. This keeps memory usage at ~2.5 GB per model
     * instead of N × 2.5 GB.
     */
    for (int i = 0; i < N_TEST_MODELS; i++) {
        if (!g_test_models[i].available)
            continue;

        printf("  (loading %s...)\n", g_test_models[i].name);
        sqlite3 *db = open_test_db();
        int rc = load_model_into_registry(db, g_test_models[i].name, g_test_models[i].path);
        sqlite3_close(db);

        if (rc != SQLITE_OK) {
            printf("  (SKIP %s: load failed)\n", g_test_models[i].name);
            g_test_models[i].available = 0;
        }
    }

    /* Run integration tests for each loaded model */
    for (int i = 0; i < N_TEST_MODELS; i++) {
        if (!g_test_models[i].available)
            continue;

        printf("  [%s integration]\n", g_test_models[i].name);

        INTEG(integ_vtab_select, i);
        INTEG(integ_vtab_duplicate, i);
        INTEG(integ_chat_plain, i);
        INTEG(integ_chat_grammar, i);
        INTEG(integ_chat_max_tokens, i);
        INTEG(integ_extract_entities, i);
        INTEG(integ_extract_relations, i);
        INTEG(integ_extract_ner_re, i);
        INTEG(integ_summarize, i);
        INTEG(integ_summarize_default, i);
        INTEG(integ_ner_re_pipeline, i);
    }

    /*
     * Tests below load additional model instances.
     * Run only for the first available model to avoid excessive memory.
     */
    for (int i = 0; i < N_TEST_MODELS; i++) {
        if (!g_test_models[i].available)
            continue;

        printf("  [%s extra integration]\n", g_test_models[i].name);

        INTEG(integ_vtab_delete, i);
        INTEG(integ_vtab_insert_empty_name, i);
        INTEG(integ_vtab_registry_full, i);
        INTEG(integ_context_overflow, i);

        break; /* Only run once — these load extra models */
    }
}
