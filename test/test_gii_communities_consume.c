/*
 * test_gii_communities_consume.c — G7 community-filter wiring tests.
 *
 * Tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g7` → ./build/test_runner --filter=test_g7_
 */
#include "test_common.h"
#include "graph_load.h"
#include <math.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int *build_community_mask(const GraphData *g, const int *partition, int target_community_id);
extern int induce_subgraph(const GraphData *g, const int *mask, GraphData *out_g, int **out_to_orig);
extern int brandes_compute(const GraphData *g, const char *direction, int auto_approx, int normalized, double *CB,
                           double *EB);
extern unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric,
                                   int top_k, int depth, int min_degree, sqlite3_int64 g_adj, sqlite3_int64 g_prov,
                                   int community_filter, double community_resolution);

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

/* T7.3 — build_community_mask + induce_subgraph parity.
 *
 * Path graph a-b-c-d-e-f, partitioned into communities by index half:
 *   {0,1,2} → community 1, {3,4,5} → community 2.
 *
 * Inducing community 1 drops the c-d cross-edge and yields a 3-node
 * sub-path a-b-c. Manually loading the same a-b-c path produces an
 * equivalent graph; brandes_compute on both must produce identical
 * CB[] within float tolerance.
 *
 * "Filter parity" here means parity with a hand-rolled subgraph load,
 * NOT parity with full-graph-then-post-filter (the latter is the
 * semantically-wrong baseline the plan is replacing — see
 * docs/plans/adv-centrality-filtering.md "induced-subgraph Brandes
 * is semantically correct vs full-graph-then-post-filter"). */
TEST(test_g7_filter_parity) {
    /* Use a fresh in-memory DB — graph_data_load needs a SQL backing.
     * No GII / centrality TVF needed for T7.3; this exercises the
     * helpers in isolation. */
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE TABLE edges_full(src TEXT, dst TEXT);"
                      "INSERT INTO edges_full(src, dst) VALUES "
                      "  ('a','b'),('b','c'),('c','d'),('d','e'),('e','f');"
                      "CREATE TABLE edges_a(src TEXT, dst TEXT);"
                      "INSERT INTO edges_a(src, dst) VALUES "
                      "  ('a','b'),('b','c');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Load the full path graph. */
    GraphData g_full;
    graph_data_init(&g_full);
    GraphLoadConfig cfg_full = {.edge_table = "edges_full",
                                .src_col = "src",
                                .dst_col = "dst",
                                .weight_col = NULL,
                                .direction = "both",
                                .timestamp_col = NULL,
                                .time_start = NULL,
                                .time_end = NULL};
    char *errmsg = NULL;
    rc = graph_data_load(db, &cfg_full, &g_full, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(6, g_full.node_count);

    /* Plant partition: first three nodes → community 1, rest → 2.
     * Indices match scan order: a=0, b=1, c=2, d=3, e=4, f=5. */
    const int partition[6] = {1, 1, 1, 2, 2, 2};

    /* (1) Mask = build_community_mask(g, partition, 1). */
    int *mask = build_community_mask(&g_full, partition, 1);
    ASSERT(mask != NULL);
    ASSERT_EQ_INT(1, mask[0]);
    ASSERT_EQ_INT(1, mask[1]);
    ASSERT_EQ_INT(1, mask[2]);
    ASSERT_EQ_INT(0, mask[3]);
    ASSERT_EQ_INT(0, mask[4]);
    ASSERT_EQ_INT(0, mask[5]);

    /* (2) Induce subgraph. */
    GraphData g_induced;
    int *to_orig = NULL;
    rc = induce_subgraph(&g_full, mask, &g_induced, &to_orig);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(3, g_induced.node_count);
    ASSERT(to_orig != NULL);

    /* (3) Manually load the equivalent 3-node path. */
    GraphData g_manual;
    graph_data_init(&g_manual);
    GraphLoadConfig cfg_a = {.edge_table = "edges_a",
                             .src_col = "src",
                             .dst_col = "dst",
                             .weight_col = NULL,
                             .direction = "both",
                             .timestamp_col = NULL,
                             .time_start = NULL,
                             .time_end = NULL};
    rc = graph_data_load(db, &cfg_a, &g_manual, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(3, g_manual.node_count);

    /* (4) Brandes on both — must match per node. For a 3-node path a-b-c
     * with direction='both', betweenness is:
     *   a (idx 0): 0
     *   b (idx 1): 1.0  (the unique a↔c shortest path passes through b)
     *   c (idx 2): 0
     * Both g_induced and g_manual should produce these exact values. */
    double *CB_induced = (double *)calloc(3, sizeof(double));
    double *CB_manual = (double *)calloc(3, sizeof(double));
    rc = brandes_compute(&g_induced, "both", 0, 0, CB_induced, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = brandes_compute(&g_manual, "both", 0, 0, CB_manual, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    for (int i = 0; i < 3; i++) {
        ASSERT(fabs(CB_induced[i] - CB_manual[i]) < 1e-9);
    }

    /* Cleanup. */
    free(mask);
    free(to_orig);
    free(CB_induced);
    free(CB_manual);
    graph_data_destroy(&g_induced);
    graph_data_destroy(&g_manual);
    graph_data_destroy(&g_full);
    sqlite3_close(db);
}

/* T7.4 — G2's signature function MUST hash community_filter and
 * community_resolution. Without them, two top-K queries that differ
 * only on community filter would collide on cache key and read each
 * other's results — silent wrong answers.
 *
 * The hash primitive is DJB2 today; G2 T2.1 will swap to xxh3. T7.4
 * tests "does community_X change the output," not collision strength
 * (T2.4's concern). All other params held constant; only the
 * community fields vary. */
TEST(test_g7_g2_signature_includes_community) {
    const char *prov = "events_msg_chunks_p";
    const char *pred = "{\"project\":\"a\",\"days\":7}";
    const char *metric = "node_betweenness";

    /* Baseline: a specific community/resolution pair. */
    unsigned int s_baseline = topk_signature(prov, pred, metric, /*top_k=*/100, /*depth=*/2, /*min_degree=*/3,
                                             /*g_adj=*/42, /*g_prov=*/17, /*community_filter=*/5,
                                             /*community_resolution=*/1.0);

    /* Same params, different community_filter → must differ. */
    unsigned int s_other_community = topk_signature(prov, pred, metric, 100, 2, 3, 42, 17,
                                                    /*community_filter=*/6, /*community_resolution=*/1.0);
    ASSERT(s_baseline != s_other_community);

    /* Same params, different community_resolution → must differ. */
    unsigned int s_other_resolution = topk_signature(prov, pred, metric, 100, 2, 3, 42, 17,
                                                     /*community_filter=*/5, /*community_resolution=*/0.5);
    ASSERT(s_baseline != s_other_resolution);

    /* Identical inputs (same community, same resolution) → identical
     * signatures. Stable for cache lookup. */
    unsigned int s_repeat = topk_signature(prov, pred, metric, 100, 2, 3, 42, 17, 5, 1.0);
    ASSERT_EQ_INT((int)s_baseline, (int)s_repeat);

    /* Sanity: non-community params still influence the hash (regression
     * guard against a stub that ignores everything). */
    unsigned int s_other_topk = topk_signature(prov, pred, metric, /*top_k=*/200, 2, 3, 42, 17, 5, 1.0);
    ASSERT(s_baseline != s_other_topk);
}

void test_gii_communities_consume(void) {
    RUN_TEST(test_g7_leiden_cache_hit);
    RUN_TEST(test_g7_hidden_cols_declared);
    RUN_TEST(test_g7_filter_parity);
    RUN_TEST(test_g7_g2_signature_includes_community);
}
