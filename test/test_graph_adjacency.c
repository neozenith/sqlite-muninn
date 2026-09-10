/*
 * test_graph_adjacency.c — Unit tests for the graph_adjacency virtual table
 *
 * Drives the virtual table through SQL on an in-memory database. The real
 * sqlite3_api routines table is captured by test_main.c, so extension code
 * compiled with SQLITE_EXTENSION_INIT3 runs inside the test binary.
 */
#include "test_common.h"
#include "graph_load.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declared here rather than via graph_adjacency.h: that header pulls in
 * sqlite3ext.h, whose macros redirect every sqlite3_* call through the
 * extension API table, which is not what test code should use. */
extern int adjacency_register_module(sqlite3 *db);
extern int is_graph_adjacency(sqlite3 *db, const char *name);
extern int graph_data_load_from_adjacency(sqlite3 *db, const char *vtab_name, GraphData *g, char **pzErrMsg);

/* ─── Helpers ──────────────────────────────────────────────── */

static sqlite3 *open_db(void) {
    sqlite3 *db = NULL;
    sqlite3_open(":memory:", &db);
    adjacency_register_module(db);
    return db;
}

static int run_sql(sqlite3 *db, const char *sql) {
    return sqlite3_exec(db, sql, NULL, NULL, NULL);
}

static sqlite3_int64 scalar_int(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    sqlite3_int64 v = -1;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW)
        v = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return v;
}

static double scalar_double(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    double v = -1.0;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW)
        v = sqlite3_column_double(stmt, 0);
    sqlite3_finalize(stmt);
    return v;
}

/* Returns strdup'd text (caller frees) or NULL */
static char *scalar_text(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    char *v = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
        const char *t = (const char *)sqlite3_column_text(stmt, 0);
        if (t)
            v = strdup(t);
    }
    sqlite3_finalize(stmt);
    return v;
}

/* Triangle A->B (1.0), B->C (2.0), C->A (3.0) with weighted adjacency VT "g" */
static sqlite3 *open_triangle_db(void) {
    sqlite3 *db = open_db();
    run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0);"
                "INSERT INTO edges VALUES ('A','B',1.0), ('B','C',2.0), ('C','A',3.0);"
                "CREATE VIRTUAL TABLE g USING graph_adjacency("
                "  edge_table='edges', src_col='src', dst_col='dst', weight_col='weight');");
    return db;
}

/* ─── Creation ─────────────────────────────────────────────── */

TEST(test_adj_create_basic) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 3);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g WHERE node IN ('A','B','C')"), 3);
    sqlite3_close(db);
}

TEST(test_adj_create_unweighted) {
    sqlite3 *db = open_db();
    ASSERT_EQ_INT(run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT);"
                              "INSERT INTO edges VALUES ('X','Y');"
                              "CREATE VIRTUAL TABLE g USING graph_adjacency("
                              "  edge_table='edges', src_col='src', dst_col='dst');"),
                  SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 2);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_out_degree FROM g WHERE node='X'"), 1.0, 1e-9);
    sqlite3_close(db);
}

TEST(test_adj_create_empty_edge_table) {
    sqlite3 *db = open_db();
    ASSERT_EQ_INT(run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT);"
                              "CREATE VIRTUAL TABLE g USING graph_adjacency("
                              "  edge_table='edges', src_col='src', dst_col='dst');"),
                  SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 0);
    sqlite3_close(db);
}

TEST(test_adj_create_missing_args_fails) {
    sqlite3 *db = open_db();
    run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT)");
    ASSERT(run_sql(db, "CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges')") != SQLITE_OK);
    ASSERT(run_sql(db, "CREATE VIRTUAL TABLE g2 USING graph_adjacency("
                       "edge_table='no_such', src_col='src', dst_col='dst')") != SQLITE_OK);
    ASSERT(run_sql(db, "CREATE VIRTUAL TABLE g3 USING graph_adjacency("
                       "edge_table='edges; DROP', src_col='src', dst_col='dst')") != SQLITE_OK);
    sqlite3_close(db);
}

TEST(test_adj_shadow_tables_and_triggers) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
                                      "('g_config','g_nodes','g_degree','g_csr_fwd','g_csr_rev','g_delta')"),
                  6);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN "
                                      "('g_ai','g_ad','g_au')"),
                  3);
    sqlite3_close(db);
}

TEST(test_adj_config_metadata) {
    sqlite3 *db = open_triangle_db();
    char *et = scalar_text(db, "SELECT value FROM g_config WHERE key='edge_table'");
    ASSERT(et && strcmp(et, "edges") == 0);
    free(et);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT value FROM g_config WHERE key='node_count'"), 3);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT value FROM g_config WHERE key='edge_count'"), 3);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT value FROM g_config WHERE key='block_size'"), 4096);
    ASSERT((int)scalar_int(db, "SELECT value FROM g_config WHERE key='generation'") >= 1);
    sqlite3_close(db);
}

/* ─── Reads ────────────────────────────────────────────────── */

TEST(test_adj_degrees) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT SUM(in_degree) FROM g"), 3);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT SUM(out_degree) FROM g"), 3);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT in_degree FROM g WHERE node='A'"), 1);
    /* weighted: A out 1.0 (A->B), A in 3.0 (C->A) */
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_out_degree FROM g WHERE node='A'"), 1.0, 1e-9);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_in_degree FROM g WHERE node='A'"), 3.0, 1e-9);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_out_degree FROM g WHERE node='B'"), 2.0, 1e-9);
    sqlite3_close(db);
}

TEST(test_adj_node_idx_sequential) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT MIN(node_idx) FROM g"), 0);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT MAX(node_idx) FROM g"), 2);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(DISTINCT node_idx) FROM g"), 3);
    sqlite3_close(db);
}

TEST(test_adj_point_lookup) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g WHERE node='A'"), 1);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g WHERE node='Z'"), 0);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g WHERE node_idx=1"), 1);
    sqlite3_close(db);
}

TEST(test_adj_csr_blobs_exist) {
    sqlite3 *db = open_triangle_db();
    ASSERT((int)scalar_int(db, "SELECT length(offsets) FROM g_csr_fwd WHERE block_id=0") > 0);
    ASSERT((int)scalar_int(db, "SELECT length(targets) FROM g_csr_rev WHERE block_id=0") > 0);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g_csr_fwd"), 1);
    sqlite3_close(db);
}

/* ─── Triggers and rebuilds ────────────────────────────────── */

TEST(test_adj_insert_triggers_rebuild) {
    sqlite3 *db = open_triangle_db();
    run_sql(db, "INSERT INTO edges VALUES ('D','A',1.0)");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 4);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT in_degree FROM g WHERE node='A'"), 2);
    sqlite3_close(db);
}

TEST(test_adj_delete_triggers_rebuild) {
    sqlite3 *db = open_triangle_db();
    run_sql(db, "DELETE FROM edges WHERE src='C' AND dst='A'");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT out_degree FROM g WHERE node='C'"), 0);
    sqlite3_close(db);
}

TEST(test_adj_update_triggers_rebuild) {
    sqlite3 *db = open_triangle_db();
    run_sql(db, "UPDATE edges SET weight = 10.0 WHERE src='A' AND dst='B'");
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_out_degree FROM g WHERE node='A'"), 10.0, 1e-9);
    sqlite3_close(db);
}

TEST(test_adj_delta_pending_then_consumed) {
    sqlite3 *db = open_triangle_db();
    run_sql(db, "INSERT INTO edges VALUES ('D','E',1.0)");
    /* Trigger queued the delta; reading the VT consumes it */
    ASSERT((int)scalar_int(db, "SELECT COUNT(*) FROM g_delta") >= 1);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 5);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g_delta"), 0);
    sqlite3_close(db);
}

TEST(test_adj_rebuild_commands) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT(run_sql(db, "INSERT INTO g(g) VALUES ('rebuild')"), SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 3);

    run_sql(db, "INSERT INTO edges VALUES ('D','A',1.0)");
    ASSERT_EQ_INT(run_sql(db, "INSERT INTO g(g) VALUES ('incremental_rebuild')"), SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g WHERE node='D'"), 1);

    ASSERT(run_sql(db, "INSERT INTO g(g) VALUES ('bad_command')") != SQLITE_OK);
    ASSERT(run_sql(db, "INSERT INTO g(node) VALUES ('Z')") != SQLITE_OK);
    ASSERT(run_sql(db, "DELETE FROM g WHERE node='A'") != SQLITE_OK);
    sqlite3_close(db);
}

TEST(test_adj_small_delta_uses_incremental) {
    sqlite3 *db = open_triangle_db();
    char sql[128];
    for (int i = 0; i < 200; i++) {
        snprintf(sql, sizeof(sql), "INSERT INTO edges VALUES ('N%d','N%d',1.0)", i, i + 1);
        run_sql(db, sql);
    }
    run_sql(db, "INSERT INTO g(g) VALUES ('rebuild')");
    sqlite3_int64 gen_before = scalar_int(db, "SELECT value FROM g_config WHERE key='generation'");
    /* One insert is far below the 10% threshold: incremental path */
    run_sql(db, "INSERT INTO edges VALUES ('N0','N50',1.0)");
    ASSERT((int)scalar_int(db, "SELECT COUNT(*) FROM g") > 200);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT out_degree FROM g WHERE node='N0'"), 2);
    ASSERT(scalar_int(db, "SELECT value FROM g_config WHERE key='generation'") > gen_before);
    sqlite3_close(db);
}

TEST(test_adj_incremental_delete_and_new_nodes) {
    sqlite3 *db = open_triangle_db();
    char sql[128];
    for (int i = 0; i < 200; i++) {
        snprintf(sql, sizeof(sql), "INSERT INTO edges VALUES ('N%d','N%d',1.0)", i, i + 1);
        run_sql(db, sql);
    }
    run_sql(db, "INSERT INTO g(g) VALUES ('rebuild')");
    /* Small delta mixing a delete, a weight update, and a brand-new node */
    run_sql(db, "DELETE FROM edges WHERE src='N10';"
                "UPDATE edges SET weight=5.0 WHERE src='N20';"
                "INSERT INTO edges VALUES ('N199','NEW',1.0);");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT out_degree FROM g WHERE node='N10'"), 0);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT weighted_out_degree FROM g WHERE node='N20'"), 5.0, 1e-9);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT in_degree FROM g WHERE node='NEW'"), 1);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT out_degree FROM g WHERE node='N199'"), 2);
    sqlite3_close(db);
}

TEST(test_adj_large_delta_uses_full_rebuild) {
    sqlite3 *db = open_triangle_db();
    run_sql(db, "INSERT INTO g(g) VALUES ('rebuild')");
    char sql[128];
    for (int i = 0; i < 40; i++) {
        snprintf(sql, sizeof(sql), "INSERT INTO edges VALUES ('M%d','M%d',1.0)", i, i + 1);
        run_sql(db, sql);
    }
    /* 40 deltas > max(10, 3/10): full rebuild path */
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 3 + 41);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT value FROM g_config WHERE key='edge_count'"), 43);
    sqlite3_close(db);
}

TEST(test_adj_multi_block_csr) {
    /* > 4096 nodes forces more than one CSR block per direction */
    sqlite3 *db = open_db();
    run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT)");
    run_sql(db, "BEGIN");
    sqlite3_stmt *ins = NULL;
    sqlite3_prepare_v2(db, "INSERT INTO edges VALUES (?1, ?2)", -1, &ins, NULL);
    char a[16], b[16];
    for (int i = 0; i < 5000; i++) {
        snprintf(a, sizeof(a), "n%d", i);
        snprintf(b, sizeof(b), "n%d", i + 1);
        sqlite3_bind_text(ins, 1, a, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(ins, 2, b, -1, SQLITE_TRANSIENT);
        sqlite3_step(ins);
        sqlite3_reset(ins);
    }
    sqlite3_finalize(ins);
    run_sql(db, "COMMIT");
    ASSERT_EQ_INT(run_sql(db, "CREATE VIRTUAL TABLE g USING graph_adjacency("
                              "edge_table='edges', src_col='src', dst_col='dst')"),
                  SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 5001);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g_csr_fwd"), 2);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g_csr_rev"), 2);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT in_degree FROM g WHERE node='n4500'"), 1);

    /* Incremental rebuild touching only the second block */
    run_sql(db, "INSERT INTO edges VALUES ('n4500','n4600')");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT out_degree FROM g WHERE node='n4500'"), 2);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT in_degree FROM g WHERE node='n4600'"), 2);

    /* Loading the whole graph merges the blocks back together */
    GraphData gd;
    graph_data_init(&gd);
    char *err = NULL;
    ASSERT_EQ_INT(graph_data_load_from_adjacency(db, "g", &gd, &err), SQLITE_OK);
    ASSERT_EQ_INT(gd.node_count, 5001);
    ASSERT_EQ_INT(gd.edge_count, 5001);
    int i4500 = graph_data_find(&gd, "n4500");
    ASSERT(i4500 >= 0);
    ASSERT_EQ_INT(gd.out[i4500].count, 2);
    graph_data_destroy(&gd);
    sqlite3_close(db);
}

/* ─── Programmatic API ─────────────────────────────────────── */

TEST(test_adj_is_graph_adjacency) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT(is_graph_adjacency(db, "g"), 1);
    ASSERT_EQ_INT(is_graph_adjacency(db, "edges"), 0);
    ASSERT_EQ_INT(is_graph_adjacency(db, "nope"), 0);
    sqlite3_close(db);
}

TEST(test_adj_load_graph_data_fresh) {
    sqlite3 *db = open_triangle_db();
    GraphData g;
    graph_data_init(&g);
    char *err = NULL;
    ASSERT_EQ_INT(graph_data_load_from_adjacency(db, "g", &g, &err), SQLITE_OK);
    ASSERT_EQ_INT(g.node_count, 3);
    ASSERT_EQ_INT(g.edge_count, 3);
    int a = graph_data_find(&g, "A");
    ASSERT(a >= 0);
    ASSERT_EQ_INT(g.out[a].count, 1);
    ASSERT_EQ_INT(g.in[a].count, 1);
    ASSERT_EQ_FLOAT(g.in[a].edges[0].weight, 3.0, 1e-9);
    graph_data_destroy(&g);
    sqlite3_close(db);
}

TEST(test_adj_load_graph_data_stale_falls_back) {
    sqlite3 *db = open_triangle_db();
    /* Queue a delta without reading the VT: cache is stale */
    run_sql(db, "INSERT INTO edges VALUES ('D','A',1.0)");
    ASSERT((int)scalar_int(db, "SELECT COUNT(*) FROM g_delta") >= 1);
    GraphData g;
    graph_data_init(&g);
    char *err = NULL;
    ASSERT_EQ_INT(graph_data_load_from_adjacency(db, "g", &g, &err), SQLITE_OK);
    ASSERT_EQ_INT(g.node_count, 4);
    ASSERT_EQ_INT(g.edge_count, 4);
    graph_data_destroy(&g);
    sqlite3_close(db);
}

TEST(test_adj_load_graph_data_missing_vt) {
    sqlite3 *db = open_db();
    GraphData g;
    graph_data_init(&g);
    char *err = NULL;
    ASSERT(graph_data_load_from_adjacency(db, "missing", &g, &err) != SQLITE_OK);
    sqlite3_free(err);
    graph_data_destroy(&g);
    sqlite3_close(db);
}

/* ─── Lifecycle ────────────────────────────────────────────── */

TEST(test_adj_reconnect_persists) {
    /* xConnect path: a second connection to a file-backed DB sees the cache */
    /* build/ always exists when test_runner runs; tmp/ does not on CI runners */
    const char *path = "build/test_graph_adjacency_reconnect.db";
    sqlite3 *db = NULL;
    sqlite3_open(path, &db);
    adjacency_register_module(db);
    run_sql(db, "DROP TABLE IF EXISTS g; DROP TABLE IF EXISTS edges;"
                "CREATE TABLE edges (src TEXT, dst TEXT);"
                "INSERT INTO edges VALUES ('A','B'), ('B','C');"
                "CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst');");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM g"), 3);
    sqlite3_close(db);

    sqlite3 *db2 = NULL;
    sqlite3_open(path, &db2);
    adjacency_register_module(db2);
    ASSERT_EQ_INT((int)scalar_int(db2, "SELECT COUNT(*) FROM g"), 3);
    ASSERT_EQ_INT((int)scalar_int(db2, "SELECT out_degree FROM g WHERE node='B'"), 1);
    run_sql(db2, "DROP TABLE g; DROP TABLE edges;");
    sqlite3_close(db2);
}

TEST(test_adj_drop_cleans_shadow_tables) {
    sqlite3 *db = open_triangle_db();
    ASSERT_EQ_INT(run_sql(db, "DROP TABLE g"), SQLITE_OK);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'g_%'"), 0);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"), 0);
    sqlite3_close(db);
}

/* ─── Entry point ──────────────────────────────────────────── */

void test_graph_adjacency(void) {
    RUN_TEST(test_adj_create_basic);
    RUN_TEST(test_adj_create_unweighted);
    RUN_TEST(test_adj_create_empty_edge_table);
    RUN_TEST(test_adj_create_missing_args_fails);
    RUN_TEST(test_adj_shadow_tables_and_triggers);
    RUN_TEST(test_adj_config_metadata);
    RUN_TEST(test_adj_degrees);
    RUN_TEST(test_adj_node_idx_sequential);
    RUN_TEST(test_adj_point_lookup);
    RUN_TEST(test_adj_csr_blobs_exist);
    RUN_TEST(test_adj_insert_triggers_rebuild);
    RUN_TEST(test_adj_delete_triggers_rebuild);
    RUN_TEST(test_adj_update_triggers_rebuild);
    RUN_TEST(test_adj_delta_pending_then_consumed);
    RUN_TEST(test_adj_rebuild_commands);
    RUN_TEST(test_adj_small_delta_uses_incremental);
    RUN_TEST(test_adj_incremental_delete_and_new_nodes);
    RUN_TEST(test_adj_large_delta_uses_full_rebuild);
    RUN_TEST(test_adj_multi_block_csr);
    RUN_TEST(test_adj_is_graph_adjacency);
    RUN_TEST(test_adj_load_graph_data_fresh);
    RUN_TEST(test_adj_load_graph_data_stale_falls_back);
    RUN_TEST(test_adj_load_graph_data_missing_vt);
    RUN_TEST(test_adj_reconnect_persists);
    RUN_TEST(test_adj_drop_cleans_shadow_tables);
}
