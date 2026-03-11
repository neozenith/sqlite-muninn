/*
 * test_llama_embed.c — Unit tests for the GGUF embedding model registry
 *
 * Tests the model registry (add/find/remove) and embed_register_functions()
 * without requiring a real GGUF model file. Integration tests with real
 * models are in pytests/test_embed_gguf.py.
 */
#include "test_common.h"
#include <sqlite3.h>
#include <string.h>

/* ─── Model Registry Internals (white-box testing) ──────────── */

/* We test the registry through SQL since the registry API is static.
 * These tests verify that embed_register_functions() registers all
 * expected SQL functions and the muninn_models virtual table. */

extern int embed_register_functions(sqlite3 *db);

/* ─── Test: function registration ──────────────────────────────── */

TEST(test_embed_register_functions) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT(rc == SQLITE_OK);

    /* Must set up the extension API pointer for SQLITE_EXTENSION_INIT3 modules */
    rc = embed_register_functions(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Verify muninn_embed_model is registered (calling with wrong type should
     * give an error, not "no such function") */
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db, "SELECT muninn_embed_model(42)", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    /* Should get SQLITE_ERROR (wrong argument type), not "no such function" */
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Verify muninn_embed is registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_embed('x', 'y')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    /* Should fail with "model not loaded", not "no such function" */
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Verify muninn_model_dim is registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_model_dim('x')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    /* Verify muninn_tokenize is registered */
    rc = sqlite3_prepare_v2(db, "SELECT muninn_tokenize('x', 'y')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

/* ─── Test: muninn_models VT schema ───────────────────────────── */

TEST(test_models_vtab_exists) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT(rc == SQLITE_OK);

    rc = embed_register_functions(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Query the muninn_models eponymous VT — should return empty, not error */
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db, "SELECT name, dim FROM muninn_models", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* No models loaded, so first step should be SQLITE_DONE */
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_DONE, rc);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
}

/* ─── Test: error on unloaded model ────────────────────────────── */

TEST(test_embed_unloaded_model_error) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT(rc == SQLITE_OK);

    rc = embed_register_functions(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Trying to embed with a non-existent model should give a clear error */
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db, "SELECT muninn_embed('nonexistent', 'hello')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);

    const char *err = sqlite3_errmsg(db);
    ASSERT(err != NULL);
    ASSERT(strstr(err, "not loaded") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Test: embed_model with nonexistent file ─────────────────── */

TEST(test_embed_model_bad_path) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT(rc == SQLITE_OK);

    rc = embed_register_functions(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Loading a nonexistent GGUF file should fail gracefully */
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db, "SELECT muninn_embed_model('/nonexistent/path.gguf')", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ERROR, rc);

    const char *err = sqlite3_errmsg(db);
    ASSERT(err != NULL);
    ASSERT(strstr(err, "failed to load") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* ─── Suite entry point ───────────────────────────────────────── */

void test_llama_embed(void) {
    RUN_TEST(test_embed_register_functions);
    RUN_TEST(test_models_vtab_exists);
    RUN_TEST(test_embed_unloaded_model_error);
    RUN_TEST(test_embed_model_bad_path);
}
