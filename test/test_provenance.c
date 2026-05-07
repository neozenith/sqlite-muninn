/*
 * test_provenance.c — Unit tests for the gii_provenance virtual table module.
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g1` → ./build/test_runner --filter=test_g1_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <string.h>

extern int provenance_register_module(sqlite3 *db);

/* T1.1 — provenance_register_module + xCreate produces the shadow table
 * with schema (namespace_id, chunk_id, canonical, project_id, timestamp,
 * PRIMARY KEY (namespace_id, chunk_id, canonical)) per
 * docs/plans/adv-centrality-filtering.md "Provenance Shadow Table". */
TEST(test_g1_schema_creates_with_xcreate) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    char *errmsg = NULL;
    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);

    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db,
                            "SELECT sql FROM sqlite_master "
                            "WHERE type = 'table' AND name = '_gii_provenance'",
                            -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *sql = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(sql != NULL);
    ASSERT(strstr(sql, "namespace_id") != NULL);
    ASSERT(strstr(sql, "chunk_id") != NULL);
    ASSERT(strstr(sql, "canonical") != NULL);
    ASSERT(strstr(sql, "project_id") != NULL);
    ASSERT(strstr(sql, "timestamp") != NULL);
    ASSERT(strstr(sql, "PRIMARY KEY") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

void test_provenance(void) {
    RUN_TEST(test_g1_schema_creates_with_xcreate);
}
