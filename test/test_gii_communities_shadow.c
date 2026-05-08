/*
 * test_gii_communities_shadow.c — GII communities shadow-table tests (G6).
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g6` → ./build/test_runner --filter=test_g6_
 */
#include "test_common.h"
#include <math.h>
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

extern int adjacency_register_module(sqlite3 *db);

/* Returns 1 if a regular table named `name` exists, else 0. */
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

/* Returns the CREATE TABLE SQL of `name`, or NULL if absent. Caller frees. */
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

/* Returns 1 if a config row exists for the given (vt, key), else 0. */
static int has_config_key(sqlite3 *db, const char *vt_name, const char *key) {
    char *sql = sqlite3_mprintf("SELECT 1 FROM \"%w_config\" WHERE key = ?", vt_name);
    if (!sql) {
        return 0;
    }
    sqlite3_stmt *stmt = NULL;
    int n = 0;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            n = 1;
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_free(sql);
    return n;
}

/* Read a TEXT-typed config value as a fresh sqlite3-allocated string,
 * or NULL if absent. */
static char *config_get_text(sqlite3 *db, const char *vt_name, const char *key) {
    char *sql = sqlite3_mprintf("SELECT value FROM \"%w_config\" WHERE key = ?", vt_name);
    if (!sql) {
        return NULL;
    }
    sqlite3_stmt *stmt = NULL;
    char *out = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *t = (const char *)sqlite3_column_text(stmt, 0);
            if (t) {
                out = sqlite3_mprintf("%s", t);
            }
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_free(sql);
    return out;
}

/* T6.1 — features='communities' creates the _communities shadow table
 * and seeds the four documented _config keys. Without the flag, neither
 * the table nor the keys appear (the feature is opt-in).
 *
 * Schema contract per docs/plans/adv-centrality-filtering.md G6
 * ("Schema (verbatim — this is the contract; G7 reads from it)"):
 *   _communities(namespace_id INTEGER DEFAULT 0,
 *                node_idx INTEGER NOT NULL,
 *                community_id INTEGER NOT NULL,
 *                PRIMARY KEY (namespace_id, node_idx))
 *
 * Config keys (per the table at line 1288 of the plan):
 *   communities_generation  int64    G_adj at which the partition was computed
 *   communities_resolution  double   gamma parameter
 *   communities_modularity  double   final Q (informational + warm-start QA)
 *   num_communities         int      K (number of distinct communities)
 *
 * The state-machine reads these to decide CACHE_HIT / WARM_START /
 * COLD_START — communities_generation < 0 signals 'never computed', so
 * the seeded sentinel must be -1. */
TEST(test_g6_schema_and_config_keys) {
    int rc;

    /* (a) Default VT: no _communities table, no comm config keys. */
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

    ASSERT_EQ_INT(0, has_table(db1, "g_communities"));
    ASSERT_EQ_INT(0, has_config_key(db1, "g", "communities_generation"));
    ASSERT_EQ_INT(0, has_config_key(db1, "g", "communities_resolution"));
    ASSERT_EQ_INT(0, has_config_key(db1, "g", "communities_modularity"));
    ASSERT_EQ_INT(0, has_config_key(db1, "g", "num_communities"));
    sqlite3_close(db1);

    /* (b) features='communities': table exists with documented schema,
     * all four config keys seeded. */
    sqlite3 *db2 = NULL;
    rc = sqlite3_open(":memory:", &db2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db2, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db2,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Schema check on _communities. */
    char *sql = table_sql(db2, "g_communities");
    ASSERT(sql != NULL);
    ASSERT(strstr(sql, "namespace_id") != NULL);
    ASSERT(strstr(sql, "node_idx") != NULL);
    ASSERT(strstr(sql, "community_id") != NULL);
    ASSERT(strstr(sql, "PRIMARY KEY") != NULL);
    sqlite3_free(sql);

    /* Config keys present and at sentinel/default values. */
    char *gen = config_get_text(db2, "g", "communities_generation");
    char *res = config_get_text(db2, "g", "communities_resolution");
    char *mod = config_get_text(db2, "g", "communities_modularity");
    char *num = config_get_text(db2, "g", "num_communities");
    ASSERT(gen != NULL);
    ASSERT(res != NULL);
    ASSERT(mod != NULL);
    ASSERT(num != NULL);

    /* communities_generation must be -1 so check_communities_cache
     * recognizes 'never computed' → COLD_START. */
    ASSERT_EQ_INT(-1, atoi(gen));
    /* communities_resolution must be a sentinel (-1.0) so resolution
     * mismatch on first read pushes COLD_START. */
    ASSERT(atof(res) < 0.0);
    /* Initial modularity = 0, num_communities = 0 (informational). */
    ASSERT(fabs(atof(mod)) < 1e-9);
    ASSERT_EQ_INT(0, atoi(num));

    sqlite3_free(gen);
    sqlite3_free(res);
    sqlite3_free(mod);
    sqlite3_free(num);

    /* (c) Composition with sssp: features='sssp,communities' creates
     * BOTH shadow surfaces. Forward-compatibility check on the
     * features_contains parser. */
    sqlite3 *db3 = NULL;
    rc = sqlite3_open(":memory:", &db3);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db3);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db3, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db3,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp,communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(1, has_table(db3, "g_sssp"));
    ASSERT_EQ_INT(1, has_table(db3, "g_communities"));
    sqlite3_close(db3);
}

void test_gii_communities_shadow(void) {
    RUN_TEST(test_g6_schema_and_config_keys);
}
