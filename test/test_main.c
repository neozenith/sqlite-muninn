/*
 * test_main.c — Minimal C test runner
 *
 * Each test module provides a run function that returns pass/fail counts.
 * We aggregate and report results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>

/*
 * Extension source files use SQLITE_EXTENSION_INIT3 which declares
 *   extern const sqlite3_api_routines *sqlite3_api;
 * Normally this pointer is set by sqlite3_load_extension(). For the test
 * runner (which links libsqlite3 directly), we capture it via
 * sqlite3_auto_extension() at startup so that embed_register_functions()
 * and other extension code can call sqlite3_create_function() etc.
 */
const sqlite3_api_routines *sqlite3_api = NULL;

static int capture_sqlite3_api(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
    (void)db;
    (void)pzErrMsg;
    sqlite3_api = pApi;
    return SQLITE_OK;
}

/* Test result tracking */
static int total_passed = 0;
static int total_failed = 0;
static int cur_failed = 0; /* tracks if current test has failed */
static const char *current_test = NULL;

void test_begin(const char *name) {
    current_test = name;
    cur_failed = 0;
}

void test_pass(void) {
    printf("  PASS: %s\n", current_test);
    total_passed++;
}

void test_fail(const char *file, int line, const char *expr) {
    printf("  FAIL: %s (%s:%d: %s)\n", current_test, file, line, expr);
    total_failed++;
    cur_failed = 1;
}

int test_failed_flag(void) {
    return cur_failed;
}

/* Assertion macros (defined in test_main.h but inlined here for simplicity) */

/* External test suites */
extern void test_vec_math(void);
extern void test_priority_queue(void);
extern void test_hnsw_algo(void);
extern void test_id_validate(void);
extern void test_graph_load(void);
extern void test_graph_csr(void);
extern void test_graph_selector(void);
extern void test_llama_embed(void);
extern void test_llama_chat(void);
extern void test_string_sim(void);

int main(void) {
    /* Capture the real sqlite3_api pointer so extension code works */
    sqlite3_auto_extension((void (*)(void))capture_sqlite3_api);
    {
        sqlite3 *tmp;
        sqlite3_open(":memory:", &tmp);
        sqlite3_close(tmp);
    }
    sqlite3_reset_auto_extension();

    printf("=== sqlite-muninn test suite ===\n\n");

    printf("[vec_math]\n");
    test_vec_math();

    printf("\n[priority_queue]\n");
    test_priority_queue();

    printf("\n[hnsw_algo]\n");
    test_hnsw_algo();

    printf("\n[id_validate]\n");
    test_id_validate();

    printf("\n[graph_load]\n");
    test_graph_load();

    printf("\n[graph_csr]\n");
    test_graph_csr();

    printf("\n[graph_selector]\n");
    test_graph_selector();

    printf("\n[llama_embed]\n");
    test_llama_embed();

    printf("\n[llama_chat]\n");
    test_llama_chat();

    printf("\n[string_sim]\n");
    test_string_sim();

    printf("\n=== Results: %d passed, %d failed ===\n", total_passed, total_failed);

    return total_failed > 0 ? 1 : 0;
}
