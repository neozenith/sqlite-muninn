/*
 * test_gii_communities_shadow.c — GII communities shadow-table tests (G6).
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g6` → ./build/test_runner --filter=test_g6_
 */
#include "test_common.h"
#include "graph_load.h"
#include <math.h>
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>

extern int adjacency_register_module(sqlite3 *db);

/* G6 T6.2 cache state machine — re-declared layout-compatibly so the
 * test doesn't have to include graph_community.h (which pulls in
 * sqlite3ext.h). Values must match graph_community.h exactly. */
typedef enum {
    COMM_CACHE_HIT = 0,
    COMM_CACHE_WARM_START = 1,
    COMM_CACHE_COLD_START = 2
} CommCacheState;

extern CommCacheState check_communities_cache(sqlite3 *db, const char *vtab_name, double requested_resolution);
extern int config_set_double(sqlite3 *db, const char *name, const char *key, double value);
extern double config_get_double(sqlite3 *db, const char *name, const char *key, double def);
extern double run_leiden(const GraphData *g, int *community, double resolution, const char *direction);
extern double run_leiden_warm(const GraphData *g, int *community, double resolution, const char *direction,
                              const int *changed_nodes, int n_changed);
extern int leiden_shadow_put(sqlite3 *db, const char *vt_name, int namespace_id, const int *community, int n,
                             double resolution, double modularity, sqlite3_int64 generation);
extern int leiden_shadow_get(sqlite3 *db, const char *vt_name, int namespace_id, int **out_community, int *out_n);

/* count_rows helper — duplicate of the one in test_provenance.c. */
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

/* Set a TEXT-typed config value via direct UPDATE so the test can
 * simulate every state-machine input combination without driving
 * Leiden. */
static void set_config(sqlite3 *db, const char *vt_name, const char *key, const char *value) {
    char *sql = sqlite3_mprintf("INSERT OR REPLACE INTO \"%w_config\"(key, value) VALUES (?, ?)", vt_name);
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, value, -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    sqlite3_free(sql);
}

/* T6.2 — check_communities_cache truth table.
 *
 * Plan section "Cache state machine" (line 1295) defines four
 * decision branches:
 *
 *   G_comm < 0                              → COLD_START (never computed)
 *   resolution mismatch (>= 1e-10)          → COLD_START
 *   G_comm < G_adj, resolution matches       → WARM_START
 *   G_comm == G_adj, resolution matches      → HIT
 *
 * The test plants each combination directly into _config (bypassing
 * Leiden) and asserts the enum returned. */
TEST(test_g6_cache_state_truth_table) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* (1) Default state: communities_generation = -1 (sentinel). Any
     * resolution → COLD_START because 'never computed' wins. */
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 1.0));
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 0.5));

    /* (2) Cached at resolution=1.0, generation=5, current G_adj=5 →
     * HIT for resolution=1.0. */
    set_config(db, "g", "generation", "5");
    set_config(db, "g", "communities_generation", "5");
    set_config(db, "g", "communities_resolution", "1.0");
    ASSERT_EQ_INT(COMM_CACHE_HIT, (int)check_communities_cache(db, "g", 1.0));

    /* (3) Resolution mismatch (≥ 1e-10) → COLD_START. */
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 1.5));
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 0.999999));

    /* (4) Within-tolerance resolution (< 1e-10 difference) → still HIT. */
    ASSERT_EQ_INT(COMM_CACHE_HIT, (int)check_communities_cache(db, "g", 1.0 + 1e-12));

    /* (5) G_adj advanced past G_comm (e.g. CSR rebuilt) but resolution
     * matches → WARM_START. */
    set_config(db, "g", "generation", "7"); /* G_adj > G_comm = 5 */
    ASSERT_EQ_INT(COMM_CACHE_WARM_START, (int)check_communities_cache(db, "g", 1.0));

    /* (6) Generation moved AND resolution mismatched → COLD_START
     * (resolution check wins; partition is unrecoverable for this
     * gamma regardless). */
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 2.0));

    sqlite3_close(db);
}

/* T6.3 — resolution storage round-trips with bit-equal precision and
 * the 1e-10 epsilon comparison classifies near-equal as match.
 *
 * The plan section 1283 specifies %.17g formatting because
 * lower-precision formats (e.g. %g's default 6 digits) drop bits and
 * a re-read no longer compares bit-equal to the original double. The
 * 1e-10 tolerance does hide differences for non-bit-equal values, but
 * exact match for a previously-stored gamma must survive the
 * round-trip cleanly — otherwise check_communities_cache would
 * sometimes drop into WARM_START / COLD_START even when the user
 * asked for the same gamma they cached at. */
TEST(test_g6_resolution_round_trip) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Tricky doubles — values whose decimal representation requires
     * 17 significant digits to survive a string round-trip:
     *   0.1 + 0.2  → 0.30000000000000004 (classic IEEE 754 example)
     *   1.0 / 3.0  → repeating binary
     *   M_PI       → irrational
     *   1e-10      → boundary value for the comparison tolerance
     */
    const double tricky[] = {0.1 + 0.2, 1.0 / 3.0, 3.14159265358979323846, 1e-10, -1.234567890123456789e-15};
    int n_tricky = (int)(sizeof(tricky) / sizeof(tricky[0]));

    for (int i = 0; i < n_tricky; i++) {
        rc = config_set_double(db, "g", "communities_resolution", tricky[i]);
        ASSERT_EQ_INT(SQLITE_OK, rc);
        double readback = config_get_double(db, "g", "communities_resolution", -999.0);
        /* Bit-exact equality. fabs(diff) > 0 means at least one ULP
         * was lost. */
        ASSERT(readback == tricky[i]);
    }

    /* Tolerance behavior: gammas within 1e-10 of cached → HIT,
     * beyond → COLD_START. Plant a known gamma into _config and
     * exercise the boundary. */
    rc = config_set_double(db, "g", "communities_resolution", 1.0);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    /* communities_generation must be non-sentinel for the resolution
     * check to be reached. */
    sqlite3_exec(db,
                 "INSERT OR REPLACE INTO g_config(key, value) "
                 "VALUES ('communities_generation', '5'), ('generation', '5')",
                 NULL, NULL, NULL);

    /* Within tolerance (< 1e-10): HIT. */
    ASSERT_EQ_INT(COMM_CACHE_HIT, (int)check_communities_cache(db, "g", 1.0));
    ASSERT_EQ_INT(COMM_CACHE_HIT, (int)check_communities_cache(db, "g", 1.0 + 9e-11));
    ASSERT_EQ_INT(COMM_CACHE_HIT, (int)check_communities_cache(db, "g", 1.0 - 9e-11));

    /* Beyond tolerance (≥ 1e-10): COLD_START. */
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 1.0 + 1e-9));
    ASSERT_EQ_INT(COMM_CACHE_COLD_START, (int)check_communities_cache(db, "g", 1.0 - 1e-9));

    sqlite3_close(db);
}

/* T6.4 — run_leiden_warm with all-singleton init must produce a
 * partition with modularity equivalent (within 0.01) to a cold-start
 * run_leiden. Two 4-cliques + bridge gives clear community structure
 * so the modularity is non-trivial — a stub that doesn't run Leiden
 * lands far outside the tolerance. */
TEST(test_g6_warm_with_singletons_equivalent_to_cold) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Two 4-cliques (a-b-c-d and e-f-g-h) plus a single bridge d-e.
     * Clear community structure: cold Leiden should find Q ≈ 0.4 - 0.5. */
    rc = sqlite3_exec(db,
                      "CREATE TABLE edges(src TEXT, dst TEXT);"
                      "INSERT INTO edges(src, dst) VALUES "
                      "  ('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d'),"
                      "  ('e','f'),('e','g'),('e','h'),('f','g'),('f','h'),('g','h'),"
                      "  ('d','e');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    GraphData graph;
    graph_data_init(&graph);
    GraphLoadConfig config = {.edge_table = "edges",
                              .src_col = "src",
                              .dst_col = "dst",
                              .weight_col = NULL,
                              .direction = "both",
                              .timestamp_col = NULL,
                              .time_start = NULL,
                              .time_end = NULL};
    char *errmsg = NULL;
    rc = graph_data_load(db, &config, &graph, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(8, graph.node_count);

    int N = graph.node_count;
    int *community_cold = (int *)malloc((size_t)N * sizeof(int));
    int *community_warm = (int *)malloc((size_t)N * sizeof(int));

    /* Cold-start: run_leiden initializes from singletons internally. */
    double Q_cold = run_leiden(&graph, community_cold, 1.0, "both");
    ASSERT(Q_cold > 0.0); /* Non-trivial structure → non-zero modularity. */

    /* Warm-start with all-singleton init and no changed-nodes hint:
     * equivalent to cold per the plan section 1322 ("When n_changed
     * == 0, equivalent to run_leiden() but skips singleton init"). */
    for (int i = 0; i < N; i++) {
        community_warm[i] = i;
    }
    double Q_warm = run_leiden_warm(&graph, community_warm, 1.0, "both", NULL, 0);

    /* Modularity tolerance per plan section 1258: within 0.01.
     * Leiden has tiebreak nondeterminism so two runs can land at
     * slightly different local optima — 0.01 absorbs that without
     * being so loose a broken stub passes. */
    ASSERT(fabs(Q_cold - Q_warm) < 0.01);

    free(community_cold);
    free(community_warm);
    graph_data_destroy(&graph);
    sqlite3_close(db);
}

/* T6.5 — store_communities atomicity. The DELETE + INSERT loop +
 * config metadata writes must commit as a unit so external readers
 * see either the OLD partition or the NEW partition, never a partial
 * mix. True multi-connection isolation isn't tractable in a
 * single-threaded runner; this test verifies the strongest
 * single-process equivalent: end-state consistency across all
 * artifacts (partition rows + four config keys) after each put,
 * including rapid overwrites that would surface stale-row leaks if
 * the DELETE didn't fire as part of the same transaction. */
TEST(test_g6_concurrent_read_during_write) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = adjacency_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db, "CREATE TABLE edges(src TEXT, dst TEXT)", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_exec(db,
                      "CREATE VIRTUAL TABLE g USING graph_adjacency("
                      "  edge_table=edges, src_col=src, dst_col=dst, features='communities')",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Initial put: 5 nodes, all in community 1. */
    const int initial[5] = {1, 1, 1, 1, 1};
    rc = leiden_shadow_put(db, "g", 0, initial, 5, /*res=*/1.0, /*mod=*/0.30, /*gen=*/1);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Round-trip: read back the same partition. */
    int *out = NULL;
    int n = 0;
    rc = leiden_shadow_get(db, "g", 0, &out, &n);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(5, n);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_INT(1, out[i]);
    }
    free(out);

    /* All four config keys updated to match the put inputs. */
    ASSERT(config_get_double(db, "g", "communities_resolution", -1.0) == 1.0);
    ASSERT(fabs(config_get_double(db, "g", "communities_modularity", -1.0) - 0.30) < 1e-9);
    ASSERT_EQ_INT(1, atoi(config_get_text(db, "g", "communities_generation")));
    ASSERT_EQ_INT(1, atoi(config_get_text(db, "g", "num_communities"))); /* one distinct id */

    /* Overwrite: 5 nodes, 3 communities. Distinct count = 3. */
    const int updated[5] = {2, 2, 3, 3, 4};
    rc = leiden_shadow_put(db, "g", 0, updated, 5, /*res=*/0.5, /*mod=*/0.55, /*gen=*/2);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* No leftover rows from the initial partition (DELETE fired
     * inside the same transaction as the new INSERTs). */
    ASSERT_EQ_INT(5, count_rows(db, "SELECT COUNT(*) FROM g_communities WHERE namespace_id = 0"));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM g_communities WHERE community_id = 1"));

    /* Round-trip new partition. */
    rc = leiden_shadow_get(db, "g", 0, &out, &n);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(5, n);
    ASSERT_EQ_INT(2, out[0]);
    ASSERT_EQ_INT(2, out[1]);
    ASSERT_EQ_INT(3, out[2]);
    ASSERT_EQ_INT(3, out[3]);
    ASSERT_EQ_INT(4, out[4]);
    free(out);

    /* Config keys updated atomically alongside the partition. */
    ASSERT(config_get_double(db, "g", "communities_resolution", -1.0) == 0.5);
    ASSERT(fabs(config_get_double(db, "g", "communities_modularity", -1.0) - 0.55) < 1e-9);
    ASSERT_EQ_INT(2, atoi(config_get_text(db, "g", "communities_generation")));
    ASSERT_EQ_INT(3, atoi(config_get_text(db, "g", "num_communities"))); /* {2, 3, 4} */

    sqlite3_close(db);
}

void test_gii_communities_shadow(void) {
    RUN_TEST(test_g6_schema_and_config_keys);
    RUN_TEST(test_g6_cache_state_truth_table);
    RUN_TEST(test_g6_resolution_round_trip);
    RUN_TEST(test_g6_warm_with_singletons_equivalent_to_cold);
    RUN_TEST(test_g6_concurrent_read_during_write);
}
