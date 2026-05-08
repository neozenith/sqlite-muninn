/*
 * test_gii_sssp_shadow.c — GII SSSP shadow-table feature-flag tests (G4).
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g4` → ./build/test_runner --filter=test_g4_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

extern int adjacency_register_module(sqlite3 *db);
extern int sssp_shadow_put(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx, const double *dist,
                           const double *sigma, int n);
extern int sssp_shadow_get(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx, double **out_dist,
                           double **out_sigma, int *out_n);
extern int sssp_shadow_clear_delta(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx);

/* Threshold dispatch — see graph_adjacency.h for the full enum and
 * function declaration. */
typedef enum {
    REBUILD_SELECTIVE = 0,
    REBUILD_DELTA_FLUSH = 1,
    REBUILD_FULL = 2
} SsspRebuildStrategy;

extern SsspRebuildStrategy sssp_classify_rebuild(int delta_count, int total_edges, double theta_selective,
                                                 double theta_full);

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

/* T4.2 — sssp_shadow_put / sssp_shadow_get round-trip preserves the
 * exact byte layout of dist[] and sigma[]. Native byte order is the
 * contract per docs/plans/adv-centrality-filtering.md G4 ("BLOB
 * encoding contract") — no portable serialization, just sqlite3
 * blob bind/column with sizeof(double) * V bytes each.
 *
 * Also exercises sssp_shadow_clear_delta as a sanity check that the
 * delta-queue clear path runs without errors before T4.4 lands the
 * cascade emit. */
TEST(test_g4_blob_round_trip) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Five-node SSSP source: dist sentinel -1.0 marks unreachable;
     * sigma[i] is the path count to node i. Values picked to span
     * sign + zero + integers + non-trivial fractions to catch any
     * sloppy byte handling. */
    const double dist[5] = {0.0, 1.5, 3.25, -1.0, 7.125};
    const double sigma[5] = {1.0, 1.0, 2.0, 0.0, 4.0};

    rc = sssp_shadow_put(db, "g", /*namespace_id=*/0, /*source_idx=*/3, dist, sigma, 5);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    double *out_dist = NULL;
    double *out_sigma = NULL;
    int out_n = 0;
    rc = sssp_shadow_get(db, "g", 0, 3, &out_dist, &out_sigma, &out_n);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(5, out_n);
    ASSERT(out_dist != NULL);
    ASSERT(out_sigma != NULL);

    /* Byte-exact compare on both BLOBs. */
    ASSERT(memcmp(dist, out_dist, sizeof(double) * 5) == 0);
    ASSERT(memcmp(sigma, out_sigma, sizeof(double) * 5) == 0);

    free(out_dist);
    free(out_sigma);

    /* clear_delta on a non-existent (namespace, source) pair must be
     * a no-op success — exercises the SQL path, sets up T4.4's
     * cascade tests. */
    rc = sssp_shadow_clear_delta(db, "g", 0, 3);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    sqlite3_close(db);
}

/* Helper: read a TEXT config value as a string. Returns NULL if the
 * row is missing or the query fails. Caller sqlite3_free()s. */
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

/* T4.3 — threshold dispatch. The CSR-rebuild strategy is chosen from
 * the change ratio |delta| / total_edges and two configurable
 * thresholds (theta_selective, theta_full):
 *
 *   ratio < theta_selective       → REBUILD_SELECTIVE   (block-level)
 *   theta_selective ≤ ratio < theta_full → REBUILD_DELTA_FLUSH (namespace)
 *   ratio ≥ theta_full            → REBUILD_FULL        (whole graph + gen++)
 *
 * Verifies (a) the defaults are persisted in _config when features='sssp',
 * (b) the pure classifier function dispatches each ratio band to the
 * correct enum value with both default and custom thresholds, and (c)
 * the empty-graph edge case (delta=0, total=0) defaults to FULL. */
TEST(test_g4_threshold_dispatch) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* (a) Defaults must be persisted on xCreate when SSSP is enabled. */
    char *theta_sel = config_get_text(db, "g", "theta_selective");
    char *theta_full = config_get_text(db, "g", "theta_full");
    ASSERT(theta_sel != NULL && strcmp(theta_sel, "0.05") == 0);
    ASSERT(theta_full != NULL && strcmp(theta_full, "0.30") == 0);
    sqlite3_free(theta_sel);
    sqlite3_free(theta_full);

    /* (b) Classifier band coverage with the documented defaults. */
    ASSERT_EQ_INT(REBUILD_SELECTIVE, (int)sssp_classify_rebuild(2, 100, 0.05, 0.30));   /* 2% < 5% */
    ASSERT_EQ_INT(REBUILD_DELTA_FLUSH, (int)sssp_classify_rebuild(15, 100, 0.05, 0.30)); /* 15% in [5%,30%) */
    ASSERT_EQ_INT(REBUILD_FULL, (int)sssp_classify_rebuild(40, 100, 0.05, 0.30));        /* 40% ≥ 30% */

    /* Boundary: ratio exactly == theta_selective is in the delta_flush
     * band (closed-open semantic). */
    ASSERT_EQ_INT(REBUILD_DELTA_FLUSH, (int)sssp_classify_rebuild(5, 100, 0.05, 0.30));
    /* Boundary: ratio exactly == theta_full lands in REBUILD_FULL. */
    ASSERT_EQ_INT(REBUILD_FULL, (int)sssp_classify_rebuild(30, 100, 0.05, 0.30));

    /* (b') Custom thresholds change the bands accordingly. */
    ASSERT_EQ_INT(REBUILD_SELECTIVE, (int)sssp_classify_rebuild(7, 100, 0.10, 0.50));
    ASSERT_EQ_INT(REBUILD_DELTA_FLUSH, (int)sssp_classify_rebuild(20, 100, 0.10, 0.50));
    ASSERT_EQ_INT(REBUILD_FULL, (int)sssp_classify_rebuild(60, 100, 0.10, 0.50));

    /* (c) Empty-graph edge case: ratio undefined → safest under-
     * invalidator is FULL (anything firing on an empty graph is a
     * fresh-start). */
    ASSERT_EQ_INT(REBUILD_FULL, (int)sssp_classify_rebuild(0, 0, 0.05, 0.30));

    sqlite3_close(db);
}

void test_gii_sssp_shadow(void) {
    RUN_TEST(test_g4_schema_creates_with_feature_flag);
    RUN_TEST(test_g4_blob_round_trip);
    RUN_TEST(test_g4_threshold_dispatch);
}
