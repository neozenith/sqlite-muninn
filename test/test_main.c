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
 * graph_load.c uses SQLITE_EXTENSION_INIT3 which declares sqlite3_api as extern.
 * The test runner links libsqlite3 directly and never calls graph_data_load(),
 * so a NULL pointer is safe here — it just satisfies the linker.
 */
const sqlite3_api_routines *sqlite3_api = NULL;

/* Test result tracking */
static int total_passed = 0;
static int total_failed = 0;
static int cur_failed = 0;  /* tracks if current test has failed */
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

int main(void) {
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

    printf("\n=== Results: %d passed, %d failed ===\n",
           total_passed, total_failed);

    return total_failed > 0 ? 1 : 0;
}
