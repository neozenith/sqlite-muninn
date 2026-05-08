/*
 * test_gii_sssp_shadow.c — GII SSSP shadow-table feature-flag tests (G4).
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g4` → ./build/test_runner --filter=test_g4_
 */
#include "test_common.h"
#include "graph_load.h"
#include <math.h>
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

/* Layout-compatible mirror of graph_centrality.h's IntList. We
 * don't include graph_centrality.h directly because it pulls in
 * sqlite3ext.h which expects SQLITE_EXTENSION_INIT3 — the test runner
 * uses sqlite3_auto_extension instead. */
typedef struct {
    int *items;
    int count;
    int capacity;
} IntList;

extern void intlist_init(IntList *l);
extern void intlist_push(IntList *l, int val);
extern void intlist_clear(IntList *l);
extern void intlist_destroy(IntList *l);
extern int reconstruct_pred_from_dist(const GraphData *g, int source, const double *dist, IntList *pred, int *stack,
                                      int *stack_size, const char *direction, int weighted);

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
extern int sssp_cascade_emit(sqlite3 *db, const char *vt_name, int namespace_id, SsspRebuildStrategy strategy,
                             const int *affected_source_idxs, int n);

/* G5 — sssp_load_or_compute wrapper. Defined in graph_centrality.c so
 * it can call the static sssp_bfs / sssp_dijkstra helpers there. */
extern int sssp_load_or_compute(sqlite3 *db, const char *gii_vt_name, int namespace_id, const GraphData *g, int source,
                                int weighted, double *dist, double *sigma, const char *direction);

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

/* Read a TEXT-typed config value as an integer (matches the convention
 * used elsewhere in graph_adjacency: TEXT key/value, callers cast). */
static sqlite3_int64 config_get_int_text(sqlite3 *db, const char *vt_name, const char *key) {
    char *raw = config_get_text(db, vt_name, key);
    if (!raw) {
        return -1;
    }
    sqlite3_int64 v = (sqlite3_int64)atoll(raw);
    sqlite3_free(raw);
    return v;
}

/* T4.4 — sssp_cascade_emit dispatches per strategy:
 *
 *   REBUILD_SELECTIVE / REBUILD_DELTA_FLUSH: append the supplied
 *     affected source_idx values to <vt>_sssp_delta (INSERT OR IGNORE
 *     on PK conflict). Generation unchanged. _sssp untouched.
 *
 *   REBUILD_FULL: physically DELETE all rows from <vt>_sssp (no
 *     per-row generation, so logical invalidation isn't enough),
 *     DELETE all rows from <vt>_sssp_delta (irrelevant once everything
 *     is stale), and bump generation in <vt>_config. The
 *     affected_source_idxs argument is ignored.
 *
 * The plan describes SELECTIVE and DELTA_FLUSH as differing in WHICH
 * indices they emit (a block-level subset vs the entire namespace),
 * but the primitive itself takes the precomputed list — separating
 * "what to emit" from "how to emit". */
TEST(test_g4_cascade_emit_per_strategy) {
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

    /* Pre-populate _sssp with two rows so the FULL-rebuild clear has
     * something to remove. dist[] values don't matter for this test. */
    double dist[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_put(db, "g", 0, 1, dist, NULL, 5));
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_put(db, "g", 0, 2, dist, NULL, 5));
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));

    /* Pre-populate _sssp_delta with one entry so we can observe whether
     * SELECTIVE/DELTA_FLUSH appends (and FULL clears). */
    rc = sqlite3_exec(db, "INSERT INTO g_sssp_delta(namespace_id, source_idx) VALUES (0, 99)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta"));

    sqlite3_int64 gen_before = config_get_int_text(db, "g", "generation");
    ASSERT(gen_before >= 0);

    /* (1) SELECTIVE: append 3 affected source indices. */
    const int affected_a[3] = {1, 5, 9};
    rc = sssp_cascade_emit(db, "g", 0, REBUILD_SELECTIVE, affected_a, 3);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    /* _sssp_delta now has the prior (0,99) plus (0,1), (0,5), (0,9) — 4 total. */
    ASSERT_EQ_INT(4, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta"));
    /* Generation untouched. */
    ASSERT_EQ_INT(gen_before, config_get_int_text(db, "g", "generation"));
    /* _sssp untouched. */
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));

    /* (2) DELTA_FLUSH: same emit shape as SELECTIVE — appends without
     * touching _sssp or generation. INSERT OR IGNORE means a repeat of
     * source_idx=1 (already in delta from step 1) doesn't double up. */
    const int affected_b[2] = {1, 7}; /* 1 collides with prior selective emit */
    rc = sssp_cascade_emit(db, "g", 0, REBUILD_DELTA_FLUSH, affected_b, 2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    /* +1 net (source_idx=7 is new; source_idx=1 is a PK conflict). */
    ASSERT_EQ_INT(5, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta"));
    ASSERT_EQ_INT(gen_before, config_get_int_text(db, "g", "generation"));
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));

    /* (3) FULL: clears both _sssp and _sssp_delta and bumps generation.
     * The affected_source_idxs argument is ignored. */
    rc = sssp_cascade_emit(db, "g", 0, REBUILD_FULL, NULL, 0);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta"));
    ASSERT(config_get_int_text(db, "g", "generation") > gen_before);

    sqlite3_close(db);
}

/* T4.5 — namespace isolation. Writes scoped to one namespace_id are
 * invisible to another, including under cascade emit. Walks every
 * SSSP API surface that takes a namespace_id (put / get / clear_delta /
 * cascade_emit per strategy) so a missing WHERE filter on any one
 * surface fails with a localized assertion. */
TEST(test_g4_namespace_isolation) {
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

    /* Distinct dist[] payloads per namespace so we can detect cross-
     * namespace contamination via memcmp. */
    const double dist_ns0[3] = {1.0, 2.0, 3.0};
    const double dist_ns1[3] = {10.0, 20.0, 30.0};

    /* (1) put isolates by namespace — same source_idx written under
     * two namespaces produces two distinct rows. */
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_put(db, "g", 0, /*source_idx=*/5, dist_ns0, NULL, 3));
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_put(db, "g", 1, /*source_idx=*/5, dist_ns1, NULL, 3));
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));

    /* (2) get returns the namespace-correct payload. */
    double *out0 = NULL, *out1 = NULL, *junk = NULL;
    int n0 = 0, n1 = 0;
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_get(db, "g", 0, 5, &out0, &junk, &n0));
    ASSERT_EQ_INT(3, n0);
    ASSERT(memcmp(out0, dist_ns0, sizeof(double) * 3) == 0);
    free(out0);
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_get(db, "g", 1, 5, &out1, &junk, &n1));
    ASSERT_EQ_INT(3, n1);
    ASSERT(memcmp(out1, dist_ns1, sizeof(double) * 3) == 0);
    free(out1);

    /* (3) cascade emit SELECTIVE in namespace 0 leaves namespace 1's
     * _sssp_delta untouched. */
    int affected[2] = {7, 8};
    ASSERT_EQ_INT(SQLITE_OK, sssp_cascade_emit(db, "g", 0, REBUILD_SELECTIVE, affected, 2));
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 0"));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 1"));

    /* (4) clear_delta scoped to namespace 0 doesn't touch namespace 1.
     * Pre-populate namespace 1's delta queue first. */
    rc = sqlite3_exec(db, "INSERT INTO g_sssp_delta(namespace_id, source_idx) VALUES (1, 99)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 1"));
    ASSERT_EQ_INT(SQLITE_OK, sssp_shadow_clear_delta(db, "g", 0, 7));
    /* Namespace 0's (0,7) gone; (0,8) remains; namespace 1's (1,99) intact. */
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 0 AND source_idx = 7"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 0"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 1 AND source_idx = 99"));

    /* (5) cascade emit FULL on namespace 0 wipes namespace 0 entirely
     * but leaves namespace 1's _sssp + _sssp_delta intact. */
    ASSERT_EQ_INT(SQLITE_OK, sssp_cascade_emit(db, "g", 0, REBUILD_FULL, NULL, 0));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp WHERE namespace_id = 0"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp WHERE namespace_id = 1"));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 0"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp_delta WHERE namespace_id = 1"));

    sqlite3_close(db);
}

/* T4.6 — threshold defaults match the documented empirical sweep.
 *
 * The defaults seeded by adjacency_create_sssp_tables (graph_adjacency.c)
 * are theta_selective=0.05 and theta_full=0.30. Per
 * docs/plans/adv-centrality-filtering.md G4 ("Threshold-based rebuild
 * strategy"), these are the empirical optimum from the kg_perf
 * benchmark sweep — banding the change ratio at 5% and 30% minimizes
 * total rebuild work across the observed workload distribution.
 *
 * This test is the lockstep contract: if a future sweep re-tunes the
 * thresholds, both the seed in graph_adjacency.c AND the EXPECTED_*
 * constants below must be updated together. Drift between code and
 * documentation otherwise becomes a silent regression.
 *
 * The actual sweep harness lives outside the unit-test gate (in
 * benchmarks/kg_perf/). What lives here is the floor: defaults match
 * what the docs claim; band ordering invariants hold. */
TEST(test_g4_threshold_defaults_match_sweep) {
    const double EXPECTED_THETA_SELECTIVE = 0.05;
    const double EXPECTED_THETA_FULL = 0.30;

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

    char *theta_sel_text = config_get_text(db, "g", "theta_selective");
    char *theta_full_text = config_get_text(db, "g", "theta_full");
    ASSERT(theta_sel_text != NULL);
    ASSERT(theta_full_text != NULL);

    double theta_sel = atof(theta_sel_text);
    double theta_full = atof(theta_full_text);

    /* Numeric match against the documented sweep optimum. 1e-9
     * tolerance covers atof-roundtrip slop for short decimal
     * fractions. */
    ASSERT(fabs(theta_sel - EXPECTED_THETA_SELECTIVE) < 1e-9);
    ASSERT(fabs(theta_full - EXPECTED_THETA_FULL) < 1e-9);

    /* Band-ordering invariants — theta_selective < theta_full
     * enforces the three-band scheme; 0 < theta_selective makes the
     * SELECTIVE band non-empty; theta_full < 1 keeps the FULL band
     * from collapsing to "all edges changed". */
    ASSERT(theta_sel > 0.0);
    ASSERT(theta_sel < theta_full);
    ASSERT(theta_full < 1.0);

    /* Cross-check against the classifier: a ratio just below
     * theta_selective must dispatch SELECTIVE; just above theta_full
     * must dispatch FULL. Locks the boundaries against accidental
     * drift in either the constants or the classifier. */
    ASSERT_EQ_INT(REBUILD_SELECTIVE,
                  (int)sssp_classify_rebuild(/*delta=*/(int)((theta_sel - 0.001) * 1000),
                                             /*total=*/1000, theta_sel, theta_full));
    ASSERT_EQ_INT(REBUILD_FULL, (int)sssp_classify_rebuild(/*delta=*/(int)((theta_full + 0.001) * 1000),
                                                           /*total=*/1000, theta_sel, theta_full));

    sqlite3_free(theta_sel_text);
    sqlite3_free(theta_full_text);
    sqlite3_close(db);
}

/* T5.1 — sssp_load_or_compute writes back on cache miss.
 *
 * Tiny graph 'a' → 'b' → 'c' lets BFS fill in well-known distances
 * (0, 1, 2). Cache starts empty. The first wrapper call must compute
 * via sssp_bfs and persist the result via sssp_shadow_put — verified
 * by counting _sssp rows. The second call reads back exactly the same
 * dist[]/sigma[] bytes (cache-hit fast path). */
TEST(test_g5_load_or_compute_writes_back_on_miss) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_exec(db,
                      "CREATE TABLE edges(src TEXT, dst TEXT);"
                      "INSERT INTO edges(src, dst) VALUES ('a','b'),('b','c');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* GraphData mirrors the rows graph_adjacency just ingested. Both
     * scan `edges` in the same rowid order so node indices align. */
    GraphData graph;
    graph_data_init(&graph);
    GraphLoadConfig config = {.edge_table = "edges",
                              .src_col = "src",
                              .dst_col = "dst",
                              .weight_col = NULL,
                              .direction = "forward",
                              .timestamp_col = NULL,
                              .time_start = NULL,
                              .time_end = NULL};
    char *errmsg = NULL;
    rc = graph_data_load(db, &config, &graph, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(3, graph.node_count);

    /* Cache empty before the first call. */
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_sssp"));

    int N = graph.node_count;
    double *dist = (double *)malloc((size_t)N * sizeof(double));
    double *sigma = (double *)malloc((size_t)N * sizeof(double));
    ASSERT(dist != NULL && sigma != NULL);

    /* MISS: must compute and write back. */
    rc = sssp_load_or_compute(db, "g", 0, &graph, /*source=*/0, /*weighted=*/0, dist, sigma, "forward");
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Exactly one row keyed by (namespace=0, source_idx=0). */
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp "
                                    "WHERE namespace_id = 0 AND source_idx = 0"));

    /* BFS sanity: source distance is 0 (anything else means we're not
     * actually running the SSSP). */
    ASSERT(dist[0] == 0.0);

    /* HIT: returns the same bytes; cache row count unchanged. */
    double *dist2 = (double *)malloc((size_t)N * sizeof(double));
    double *sigma2 = (double *)malloc((size_t)N * sizeof(double));
    ASSERT(dist2 != NULL && sigma2 != NULL);
    rc = sssp_load_or_compute(db, "g", 0, &graph, 0, 0, dist2, sigma2, "forward");
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT(memcmp(dist, dist2, sizeof(double) * (size_t)N) == 0);
    ASSERT(memcmp(sigma, sigma2, sizeof(double) * (size_t)N) == 0);
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM g_sssp "
                                    "WHERE namespace_id = 0 AND source_idx = 0"));

    free(dist);
    free(sigma);
    free(dist2);
    free(sigma2);
    graph_data_destroy(&graph);
    sqlite3_close(db);
}

/* T5.2 — reconstruct_pred_from_dist parity against known-good values
 * for a deterministic 3-node path graph 'a' → 'b' → 'c'.
 *
 * Forward direction from source=0 ('a'):
 *   dist[0]=0, dist[1]=1, dist[2]=2
 *   pred[0]=∅, pred[1]={0}, pred[2]={1}
 *   stack=[0,1,2]  (reachable indices, dist-ascending)
 *
 * The test loads the graph + cached dist[] via sssp_load_or_compute
 * (which exercises T5.1's miss-path), then runs reconstruction and
 * checks pred[] sets and stack[] composition + ordering against the
 * hardcoded expectations above. */
TEST(test_g5_pred_reconstruction_parity) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_exec(db,
                      "CREATE TABLE edges(src TEXT, dst TEXT);"
                      "INSERT INTO edges(src, dst) VALUES ('a','b'),('b','c');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='sssp')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    GraphData graph;
    graph_data_init(&graph);
    GraphLoadConfig config = {.edge_table = "edges",
                              .src_col = "src",
                              .dst_col = "dst",
                              .weight_col = NULL,
                              .direction = "forward",
                              .timestamp_col = NULL,
                              .time_start = NULL,
                              .time_end = NULL};
    char *errmsg = NULL;
    rc = graph_data_load(db, &config, &graph, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(3, graph.node_count);

    /* Get cached dist[] via the wrapper (miss-path computes + writes back). */
    int N = graph.node_count;
    double *dist = (double *)malloc((size_t)N * sizeof(double));
    double *sigma = (double *)malloc((size_t)N * sizeof(double));
    rc = sssp_load_or_compute(db, "g", 0, &graph, /*source=*/0, /*weighted=*/0, dist, sigma, "forward");
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT(dist[0] == 0.0);
    ASSERT(dist[1] == 1.0);
    ASSERT(dist[2] == 2.0);

    /* Reconstruct pred[] / stack[] from dist[]. */
    IntList *pred = (IntList *)calloc((size_t)N, sizeof(IntList));
    int *stack = (int *)malloc((size_t)N * sizeof(int));
    int stack_size = 0;
    for (int i = 0; i < N; i++) {
        intlist_init(&pred[i]);
    }

    rc = reconstruct_pred_from_dist(&graph, 0, dist, pred, stack, &stack_size, "forward", /*weighted=*/0);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* pred[0] empty (source has no predecessor). */
    ASSERT_EQ_INT(0, pred[0].count);
    /* pred[1] == {0}. */
    ASSERT_EQ_INT(1, pred[1].count);
    ASSERT_EQ_INT(0, pred[1].items[0]);
    /* pred[2] == {1}. */
    ASSERT_EQ_INT(1, pred[2].count);
    ASSERT_EQ_INT(1, pred[2].items[0]);

    /* stack covers all 3 reachable nodes in non-decreasing dist order.
     * For a 3-node path with unique distances, the order is exactly
     * [0, 1, 2]. */
    ASSERT_EQ_INT(3, stack_size);
    ASSERT(dist[stack[0]] <= dist[stack[1]]);
    ASSERT(dist[stack[1]] <= dist[stack[2]]);
    /* All three node indices appear (set membership check). */
    int seen[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        ASSERT(stack[i] >= 0 && stack[i] < 3);
        seen[stack[i]] = 1;
    }
    ASSERT(seen[0] && seen[1] && seen[2]);

    /* Cleanup. */
    for (int i = 0; i < N; i++) {
        intlist_destroy(&pred[i]);
    }
    free(pred);
    free(stack);
    free(dist);
    free(sigma);
    graph_data_destroy(&graph);
    sqlite3_close(db);
}

void test_gii_sssp_shadow(void) {
    RUN_TEST(test_g4_schema_creates_with_feature_flag);
    RUN_TEST(test_g4_blob_round_trip);
    RUN_TEST(test_g4_threshold_dispatch);
    RUN_TEST(test_g4_cascade_emit_per_strategy);
    RUN_TEST(test_g4_namespace_isolation);
    RUN_TEST(test_g4_threshold_defaults_match_sweep);
    RUN_TEST(test_g5_load_or_compute_writes_back_on_miss);
    RUN_TEST(test_g5_pred_reconstruction_parity);
}
