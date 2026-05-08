/*
 * test_gii_communities_consume.c — G7 community-filter wiring tests.
 *
 * Tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g7` → ./build/test_runner --filter=test_g7_
 */
#include "test_common.h"
#include <math.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int adjacency_register_module(sqlite3 *db);
extern int community_register_tvfs(sqlite3 *db);
extern int centrality_register_tvfs(sqlite3 *db);
extern sqlite3_int64 config_get_int64_public(sqlite3 *db, const char *name, const char *key, sqlite3_int64 def);

/* Count rows produced by sql. Returns -1 on prepare/step failure. */
static int count_rows(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    int n = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        n = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return n;
}

/* Helper: count distinct community_id values returned by graph_leiden
 * for a given (edge_table, resolution). */
static int leiden_distinct_communities_seen(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db,
                                "SELECT COUNT(DISTINCT community_id) FROM graph_leiden "
                                "WHERE edge_table = 'g' AND src_col = 'src' AND dst_col = 'dst' "
                                "  AND resolution = 1.0 AND direction = 'both'",
                                -1, &stmt, NULL);
    if (rc != SQLITE_OK)
        return -1;
    int n = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        n = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return n;
}

/* Helper: how many graph_leiden rows have community_id = 999 (the
 * poison value). Lets us prove HIT dispatch read from the corrupted
 * cache instead of recomputing. */
static int leiden_rows_with_id(sqlite3 *db, int target_id) {
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db,
                                "SELECT COUNT(*) FROM graph_leiden "
                                "WHERE edge_table = 'g' AND src_col = 'src' AND dst_col = 'dst' "
                                "  AND resolution = 1.0 AND direction = 'both' "
                                "  AND community_id = ?",
                                -1, &stmt, NULL);
    if (rc != SQLITE_OK)
        return -1;
    sqlite3_bind_int(stmt, 1, target_id);
    int n = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        n = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return n;
}

/* T7.1 — graph_leiden's lei_filter dispatches on CommCacheState.
 * Cache HIT must short-circuit the run_leiden compute path and load
 * the partition from <vt>_communities directly.
 *
 * Test methodology (poison-cache pattern, mirrors T5.3):
 *   1. First graph_leiden query: cache COLD → run_leiden + write back.
 *   2. Verify cache populated (rows in _communities, generation set).
 *   3. Poison: UPDATE _communities SET community_id = 999, then INSERT
 *      OR REPLACE communities_generation = G_adj so check_communities_cache
 *      sees HIT.
 *   4. Second graph_leiden query: must return community_id = 999 for
 *      every node. If lei_filter doesn't consult the cache, it runs
 *      Leiden again and returns the real partition (NOT 999). */
TEST(test_g7_leiden_cache_hit) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = community_register_tvfs(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Two 4-cliques + bridge — clear community structure, modularity
     * non-trivial. Same fixture as T6.4. */
    rc = sqlite3_exec(db,
                      "CREATE TABLE edges(src TEXT, dst TEXT);"
                      "INSERT INTO edges(src, dst) VALUES "
                      "  ('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d'),"
                      "  ('e','f'),('e','g'),('e','h'),('f','g'),('f','h'),('g','h'),"
                      "  ('d','e');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* (1) First call → COLD_START path: must run_leiden + write back. */
    int distinct_first = leiden_distinct_communities_seen(db);
    ASSERT(distinct_first >= 2); /* Two 4-cliques → at least 2 communities. */

    /* (2) Cache populated by the cold-start write-back. */
    ASSERT_EQ_INT(8, count_rows(db, "SELECT COUNT(*) FROM g_communities WHERE namespace_id = 0"));
    sqlite3_int64 G_adj = config_get_int64_public(db, "g", "generation", 0);
    sqlite3_int64 G_comm = config_get_int64_public(db, "g", "communities_generation", -1);
    ASSERT(G_comm == G_adj); /* Write-back set communities_generation to G_adj. */

    /* (3) Poison the cache and ensure HIT state. The community_id=999
     * sentinel is far outside any plausible Leiden output (which uses
     * small contiguous integer IDs). */
    rc = sqlite3_exec(db, "UPDATE g_communities SET community_id = 999", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(8, count_rows(db, "SELECT COUNT(*) FROM g_communities WHERE community_id = 999"));

    /* (4) Second call → HIT path: must return community_id = 999 for
     * every row (8 rows total). If the cache is bypassed and Leiden
     * runs again, this count would be 0. */
    int rows_999 = leiden_rows_with_id(db, 999);
    ASSERT_EQ_INT(8, rows_999);

    sqlite3_close(db);
}

/* T7.2 — every centrality TVF declares community_filter and
 * community_resolution as hidden columns.
 *
 * Hidden columns aren't visible in SELECT * or PRAGMA table_info,
 * but referencing them in a WHERE clause triggers SQLite's
 * column-name validation against sqlite3_declare_vtab. If the
 * column isn't in the schema string, sqlite3_prepare_v2 fails.
 * That's the cheapest "column declared on VT" probe available. */
TEST(test_g7_hidden_cols_declared) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = centrality_register_tvfs(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* All four centrality TVFs must accept community_filter and
     * community_resolution in WHERE. The values bound aren't
     * exercised by T7.2 — only that prepare succeeds (the columns
     * exist in the declared schema). */
    const char *queries[] = {
        "SELECT * FROM graph_node_betweenness "
        "WHERE edge_table = 'e' AND src_col = 's' AND dst_col = 'd' "
        "  AND community_filter = 0 AND community_resolution = 1.0",
        "SELECT * FROM graph_edge_betweenness "
        "WHERE edge_table = 'e' AND src_col = 's' AND dst_col = 'd' "
        "  AND community_filter = 0 AND community_resolution = 1.0",
        "SELECT * FROM graph_closeness "
        "WHERE edge_table = 'e' AND src_col = 's' AND dst_col = 'd' "
        "  AND community_filter = 0 AND community_resolution = 1.0",
        "SELECT * FROM graph_degree "
        "WHERE edge_table = 'e' AND src_col = 's' AND dst_col = 'd' "
        "  AND community_filter = 0 AND community_resolution = 1.0",
    };

    for (size_t i = 0; i < sizeof(queries) / sizeof(queries[0]); i++) {
        sqlite3_stmt *stmt = NULL;
        int prc = sqlite3_prepare_v2(db, queries[i], -1, &stmt, NULL);
        ASSERT_EQ_INT(SQLITE_OK, prc);
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
}

void test_gii_communities_consume(void) {
    RUN_TEST(test_g7_leiden_cache_hit);
    RUN_TEST(test_g7_hidden_cols_declared);
}
