/*
 * test_gii_sssp_shadow.c — GII SSSP shadow-table feature-flag tests (G4).
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g4` → ./build/test_runner --filter=test_g4_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <string.h>

extern int adjacency_register_module(sqlite3 *db);

/* Returns 1 if a regular table named `name` exists in the schema, else 0. */
static int has_table(sqlite3 *db, const char *name) {
    sqlite3_stmt *stmt = NULL;
    int n = 0;
    if (sqlite3_prepare_v2(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", -1, &stmt, NULL) ==
        SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            n = 1;
        }
        sqlite3_finalize(stmt);
    }
    return n;
}

/* Returns the CREATE TABLE SQL of `name`, or NULL if the table doesn't exist.
 * Caller must sqlite3_free() the returned string. */
static char *table_sql(sqlite3 *db, const char *name) {
    sqlite3_stmt *stmt = NULL;
    char *out = NULL;
    if (sqlite3_prepare_v2(db, "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", -1, &stmt, NULL) ==
        SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *s = (const char *)sqlite3_column_text(stmt, 0);
            if (s) {
                out = sqlite3_mprintf("%s", s);
            }
        }
        sqlite3_finalize(stmt);
    }
    return out;
}

/* T4.1 — features='sssp' is opt-in. Without it, _sssp/_sssp_delta
 * shadow tables must NOT be created. With it, both must exist with
 * the schema documented in docs/plans/adv-centrality-filtering.md
 * G4 ("Schema (verbatim — this is the contract; G5 reads from it)"). */
TEST(test_g4_schema_creates_with_feature_flag) {
    int rc;

    /* (a) Without features='sssp': default GII has no SSSP tables. */
    sqlite3 *db1 = NULL;
    rc = sqlite3_open(":memory:", &db1);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db1);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db1, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db1, "CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table=edges, src_col=src, dst_col=dst)",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    ASSERT_EQ_INT(0, has_table(db1, "g_sssp"));
    ASSERT_EQ_INT(0, has_table(db1, "g_sssp_delta"));
    sqlite3_close(db1);

    /* (b) With features='sssp': both shadow tables exist with the
     * documented schema. */
    sqlite3 *db2 = NULL;
    rc = sqlite3_open(":memory:", &db2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db2, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db2,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* _sssp: namespace_id, source_idx, distances, sigma + PRIMARY KEY */
    char *sql = table_sql(db2, "g_sssp");
    ASSERT(sql != NULL);
    ASSERT(strstr(sql, "namespace_id") != NULL);
    ASSERT(strstr(sql, "source_idx") != NULL);
    ASSERT(strstr(sql, "distances") != NULL);
    ASSERT(strstr(sql, "sigma") != NULL);
    ASSERT(strstr(sql, "PRIMARY KEY") != NULL);
    sqlite3_free(sql);

    /* _sssp_delta: namespace_id, source_idx, PRIMARY KEY */
    sql = table_sql(db2, "g_sssp_delta");
    ASSERT(sql != NULL);
    ASSERT(strstr(sql, "namespace_id") != NULL);
    ASSERT(strstr(sql, "source_idx") != NULL);
    ASSERT(strstr(sql, "PRIMARY KEY") != NULL);
    sqlite3_free(sql);

    sqlite3_close(db2);
}

void test_gii_sssp_shadow(void) {
    RUN_TEST(test_g4_schema_creates_with_feature_flag);
}
