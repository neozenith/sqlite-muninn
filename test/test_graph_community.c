/*
 * test_graph_community.c — Unit tests for run_conductance()
 *
 * Builds GraphData in memory (no SQLite needed) using the two-triangle
 * graph from docs/centrality-community.md: alice/bob/carol/dave and
 * eve/frank/grace, joined by the single bridge dave -> eve.
 */
#include "test_common.h"
#include "graph_load.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declared here rather than via graph_community.h: that header pulls in
 * sqlite3ext.h, whose macros redirect every sqlite3_* call through the
 * extension API table, which is not what test code should use. */
typedef struct {
    long size;
    double internal;
    double cut;
    double vol;
    double phi;
} Conductance;
extern int run_conductance(const GraphData *g, const int *group, int k, const char *direction, Conductance *out);
extern double run_leiden(const GraphData *g, int *community, double resolution, const char *direction);
extern int community_register_tvfs(sqlite3 *db);
extern int adjacency_register_module(sqlite3 *db);

/* Node indices in insertion order */
enum { ALICE = 0, BOB, CAROL, DAVE, EVE, FRANK, GRACE, N_NODES };

static void add_edge(GraphData *g, const char *src, const char *dst, double w) {
    int si = graph_data_find_or_add(g, src);
    int di = graph_data_find_or_add(g, dst);
    graph_data_add_edge(g, si, di, w, 1, 1); /* direction = "both" layout */
}

/* Two triangles + bridge, all weights 1.0 */
static void build_two_triangles(GraphData *g) {
    graph_data_init(g);
    const char *names[N_NODES] = {"alice", "bob", "carol", "dave", "eve", "frank", "grace"};
    for (int i = 0; i < N_NODES; i++)
        graph_data_find_or_add(g, names[i]);
    add_edge(g, "alice", "bob", 1.0);
    add_edge(g, "alice", "carol", 1.0);
    add_edge(g, "bob", "carol", 1.0);
    add_edge(g, "bob", "dave", 1.0);
    add_edge(g, "carol", "dave", 1.0);
    add_edge(g, "dave", "eve", 1.0); /* bridge */
    add_edge(g, "eve", "frank", 1.0);
    add_edge(g, "eve", "grace", 1.0);
    add_edge(g, "frank", "grace", 1.0);
}

/* ─── Two-group split along the bridge ─────────────────────── */

TEST(test_conductance_two_groups) {
    GraphData g;
    build_two_triangles(&g);
    int group[N_NODES] = {0, 0, 0, 0, 1, 1, 1};
    Conductance out[2];

    ASSERT_EQ_INT(run_conductance(&g, group, 2, "both", out), 0);

    /* left: 5 internal edges, 1 cut, vol 11 */
    ASSERT_EQ_INT((int)out[0].size, 4);
    ASSERT_EQ_FLOAT(out[0].internal, 5.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].vol, 11.0, 1e-9);
    /* total_vol = 18, so min(11, 7) = 7 */
    ASSERT_EQ_FLOAT(out[0].phi, 1.0 / 7.0, 1e-6);

    /* right: 3 internal edges, 1 cut, vol 7 */
    ASSERT_EQ_INT((int)out[1].size, 3);
    ASSERT_EQ_FLOAT(out[1].internal, 3.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].cut, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].vol, 7.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].phi, 1.0 / 7.0, 1e-6);

    graph_data_destroy(&g);
}

/* ─── Ungrouped nodes still contribute to cut and total volume ── */

TEST(test_conductance_ungrouped_node) {
    GraphData g;
    build_two_triangles(&g);
    int group[N_NODES] = {0, 0, 0, 0, 1, 1, -1}; /* grace ungrouped */
    Conductance out[2];

    run_conductance(&g, group, 2, "both", out);

    /* left unchanged */
    ASSERT_EQ_FLOAT(out[0].internal, 5.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, 1.0 / 7.0, 1e-6);

    /* right = {eve, frank}: internal 1 (eve-frank);
     * cut = dave-eve + eve-grace + frank-grace = 3; vol 5 */
    ASSERT_EQ_INT((int)out[1].size, 2);
    ASSERT_EQ_FLOAT(out[1].internal, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].cut, 3.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].vol, 5.0, 1e-9);
    /* total_vol still 18 (grace's edges count), min(5, 13) = 5 */
    ASSERT_EQ_FLOAT(out[1].phi, 3.0 / 5.0, 1e-6);

    graph_data_destroy(&g);
}

/* ─── Degenerate cases ──────────────────────────────────────── */

TEST(test_conductance_singleton_no_internal_edges_is_one) {
    GraphData g;
    build_two_triangles(&g);
    int group[N_NODES] = {-1, -1, -1, 0, -1, -1, -1}; /* {dave} alone */
    Conductance out[1];

    run_conductance(&g, group, 1, "both", out);

    ASSERT_EQ_INT((int)out[0].size, 1);
    ASSERT_EQ_FLOAT(out[0].internal, 0.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 3.0, 1e-9); /* bob, carol, eve */
    ASSERT_EQ_FLOAT(out[0].vol, 3.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, 1.0, 1e-9);

    graph_data_destroy(&g);
}

TEST(test_conductance_whole_graph_is_zero) {
    GraphData g;
    build_two_triangles(&g);
    int group[N_NODES] = {0, 0, 0, 0, 0, 0, 0};
    Conductance out[1];

    run_conductance(&g, group, 1, "both", out);

    ASSERT_EQ_INT((int)out[0].size, N_NODES);
    ASSERT_EQ_FLOAT(out[0].internal, 9.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 0.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].vol, 18.0, 1e-9);
    /* min(vol, total_vol - vol) = 0 -> phi defined as 0.0, not NaN */
    ASSERT_EQ_FLOAT(out[0].phi, 0.0, 1e-9);

    graph_data_destroy(&g);
}

TEST(test_conductance_disconnected_component_is_zero) {
    GraphData g;
    graph_data_init(&g);
    add_edge(&g, "a", "b", 1.0);
    add_edge(&g, "b", "c", 1.0);
    add_edge(&g, "x", "y", 1.0); /* separate component */
    int group[5] = {0, 0, 0, 1, 1};
    Conductance out[2];

    run_conductance(&g, group, 2, "both", out);

    ASSERT_EQ_FLOAT(out[0].cut, 0.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, 0.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].cut, 0.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].phi, 0.0, 1e-9);

    graph_data_destroy(&g);
}

TEST(test_conductance_no_groups) {
    GraphData g;
    build_two_triangles(&g);
    int group[N_NODES] = {-1, -1, -1, -1, -1, -1, -1};
    Conductance out[1] = {{99, 1, 1, 1, 1}};

    ASSERT_EQ_INT(run_conductance(&g, group, 0, "both", out), 0);
    /* k == 0: out untouched */
    ASSERT_EQ_INT((int)out[0].size, 99);

    graph_data_destroy(&g);
}

TEST(test_conductance_empty_graph) {
    GraphData g;
    graph_data_init(&g);
    Conductance out[2];

    ASSERT_EQ_INT(run_conductance(&g, NULL, 2, "both", out), 0);
    ASSERT_EQ_INT((int)out[0].size, 0);
    ASSERT_EQ_FLOAT(out[1].phi, 0.0, 1e-9);

    graph_data_destroy(&g);
}

/* ─── Weights ───────────────────────────────────────────────── */

TEST(test_conductance_weighted) {
    GraphData g;
    graph_data_init(&g);
    add_edge(&g, "a", "b", 2.0);
    add_edge(&g, "b", "c", 4.0);
    add_edge(&g, "c", "d", 0.5); /* cut edge */
    add_edge(&g, "d", "e", 3.0);
    int group[5] = {0, 0, 0, 1, 1};
    Conductance out[2];

    run_conductance(&g, group, 2, "both", out);

    /* total_vol = 2 * (2 + 4 + 0.5 + 3) = 19 */
    ASSERT_EQ_FLOAT(out[0].internal, 6.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 0.5, 1e-9);
    ASSERT_EQ_FLOAT(out[0].vol, 12.5, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, 0.5 / 6.5, 1e-6); /* min(12.5, 6.5) */

    ASSERT_EQ_FLOAT(out[1].internal, 3.0, 1e-9);
    ASSERT_EQ_FLOAT(out[1].cut, 0.5, 1e-9);
    ASSERT_EQ_FLOAT(out[1].vol, 6.5, 1e-9);
    ASSERT_EQ_FLOAT(out[1].phi, 0.5 / 6.5, 1e-6);

    graph_data_destroy(&g);
}

/* ─── Direction: reverse uses in[] and matches both/forward ──── */

TEST(test_conductance_direction_reverse) {
    GraphData g;
    graph_data_init(&g);
    /* Emulate direction = "reverse": loader fills in[] only */
    int a = graph_data_find_or_add(&g, "a");
    int b = graph_data_find_or_add(&g, "b");
    int c = graph_data_find_or_add(&g, "c");
    graph_data_add_edge(&g, a, b, 1.0, 0, 1);
    graph_data_add_edge(&g, b, c, 1.0, 0, 1);
    int group[3] = {0, 0, 1};
    Conductance out[2];

    run_conductance(&g, group, 2, "reverse", out);

    ASSERT_EQ_FLOAT(out[0].internal, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 1.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].vol, 3.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, 1.0, 1e-9); /* min(3, 1) = 1 */
    ASSERT_EQ_FLOAT(out[1].phi, 1.0, 1e-9);

    graph_data_destroy(&g);
}

TEST(test_conductance_forward_matches_both) {
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "a");
    int b = graph_data_find_or_add(&g, "b");
    int c = graph_data_find_or_add(&g, "c");
    graph_data_add_edge(&g, a, b, 1.0, 1, 0); /* forward only */
    graph_data_add_edge(&g, b, c, 1.0, 1, 0);
    int group[3] = {0, 0, 1};
    Conductance fwd[2], both[2];

    run_conductance(&g, group, 2, "forward", fwd);
    run_conductance(&g, group, 2, "both", both);

    ASSERT_EQ_FLOAT(fwd[0].vol, both[0].vol, 1e-9);
    ASSERT_EQ_FLOAT(fwd[0].phi, both[0].phi, 1e-9);
    ASSERT_EQ_FLOAT(fwd[0].phi, 1.0, 1e-9);

    graph_data_destroy(&g);
}

/* ─── run_leiden ───────────────────────────────────────────── */

/* Two triangles A,B,C and D,E,F plus bridge C-D, stored bidirectionally */
static void build_barbell(GraphData *g) {
    graph_data_init(g);
    const char *pairs[][2] = {{"A", "B"}, {"A", "C"}, {"B", "C"}, {"D", "E"}, {"D", "F"}, {"E", "F"}, {"C", "D"}};
    for (int i = 0; i < 7; i++) {
        add_edge(g, pairs[i][0], pairs[i][1], 1.0);
        add_edge(g, pairs[i][1], pairs[i][0], 1.0);
    }
}

TEST(test_leiden_barbell_two_communities) {
    GraphData g;
    build_barbell(&g);
    int comm[6];
    double q = run_leiden(&g, comm, 1.0, "both");
    ASSERT(q > 0.0);
    int a = graph_data_find(&g, "A"), c = graph_data_find(&g, "C");
    int d = graph_data_find(&g, "D"), f = graph_data_find(&g, "F");
    ASSERT_EQ_INT(comm[a], comm[c]);
    ASSERT_EQ_INT(comm[d], comm[f]);
    ASSERT(comm[a] != comm[d]);
    /* contiguous ids */
    int max = 0;
    for (int i = 0; i < 6; i++)
        if (comm[i] > max)
            max = comm[i];
    ASSERT_EQ_INT(max, 1);
    graph_data_destroy(&g);
}

TEST(test_leiden_conductance_of_found_partition) {
    GraphData g;
    build_barbell(&g);
    int comm[6];
    run_leiden(&g, comm, 1.0, "both");
    Conductance out[2];
    run_conductance(&g, comm, 2, "both", out);
    /* each side: 3 internal edge rows x2 directions = 6, cut = 2 rows (C-D, D-C) */
    ASSERT_EQ_FLOAT(out[0].internal, 6.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].cut, 2.0, 1e-9);
    ASSERT_EQ_FLOAT(out[0].phi, out[1].phi, 1e-9);
    graph_data_destroy(&g);
}

TEST(test_leiden_high_resolution_splits_more) {
    GraphData g;
    build_barbell(&g);
    int lo[6], hi[6];
    run_leiden(&g, lo, 0.1, "both");
    run_leiden(&g, hi, 5.0, "both");
    int nlo = 0, nhi = 0;
    for (int i = 0; i < 6; i++) {
        if (lo[i] + 1 > nlo)
            nlo = lo[i] + 1;
        if (hi[i] + 1 > nhi)
            nhi = hi[i] + 1;
    }
    ASSERT(nhi >= nlo);
    graph_data_destroy(&g);
}

TEST(test_leiden_empty_and_edgeless) {
    GraphData g;
    graph_data_init(&g);
    int comm[2];
    ASSERT_EQ_FLOAT(run_leiden(&g, comm, 1.0, "both"), 0.0, 1e-9);
    graph_data_find_or_add(&g, "lonely");
    graph_data_find_or_add(&g, "alone");
    ASSERT_EQ_FLOAT(run_leiden(&g, comm, 1.0, "both"), 0.0, 1e-9);
    ASSERT_EQ_INT(comm[0], 0);
    ASSERT_EQ_INT(comm[1], 1);
    graph_data_destroy(&g);
}

TEST(test_leiden_forward_direction) {
    /* direction="forward" only walks out[]; on this acyclic graph Leiden keeps
     * everything in one community with Q == 0. The contract asserted here is
     * the output shape (contiguous ids, finite Q), not the partition. */
    GraphData g;
    build_two_triangles(&g);
    int comm[N_NODES];
    double q = run_leiden(&g, comm, 1.0, "forward");
    ASSERT(q >= 0.0 && q <= 1.0);
    int max = 0;
    for (int i = 0; i < N_NODES; i++) {
        ASSERT(comm[i] >= 0);
        if (comm[i] > max)
            max = comm[i];
    }
    ASSERT(max < N_NODES);
    graph_data_destroy(&g);
}

/* ─── SQL: graph_leiden and graph_conductance TVFs ─────────── */

static sqlite3 *open_db(void) {
    sqlite3 *db = NULL;
    sqlite3_open(":memory:", &db);
    adjacency_register_module(db);
    community_register_tvfs(db);
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

static sqlite3 *open_two_triangles_db(void) {
    sqlite3 *db = open_db();
    run_sql(db, "CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0, ts TEXT);"
                "INSERT INTO edges VALUES"
                " ('alice','bob',1.0,'2026-01-10'), ('alice','carol',1.0,'2026-01-10'),"
                " ('bob','carol',1.0,'2026-01-10'), ('bob','dave',1.0,'2026-02-10'),"
                " ('carol','dave',1.0,'2026-02-10'), ('dave','eve',1.0,'2026-03-10'),"
                " ('eve','frank',1.0,'2026-04-10'), ('eve','grace',1.0,'2026-04-10'),"
                " ('frank','grace',1.0,'2026-04-10');"
                "CREATE TABLE assignment (node TEXT, team TEXT);"
                "INSERT INTO assignment VALUES ('alice','left'),('bob','left'),('carol','left'),"
                " ('dave','left'),('eve','right'),('frank','right'),('grace','right');");
    return db;
}

#define CND_WHERE                                                                                                      \
    " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst' AND direction = 'both'"                       \
    "   AND membership_table = 'assignment' AND group_col = 'team' AND member_col = 'node'"

TEST(test_sql_leiden_two_triangles) {
    sqlite3 *db = open_two_triangles_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_leiden WHERE edge_table='edges'"
                                      " AND src_col='src' AND dst_col='dst' AND direction='both'"),
                  7);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(DISTINCT community_id) FROM graph_leiden WHERE "
                                      "edge_table='edges' AND src_col='src' AND dst_col='dst' AND resolution=1.0"),
                  2);
    ASSERT(scalar_double(db, "SELECT modularity FROM graph_leiden WHERE edge_table='edges'"
                             " AND src_col='src' AND dst_col='dst' AND weight_col='weight' LIMIT 1") > 0.0);
    /* temporal filter: only the first triangle */
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_leiden WHERE edge_table='edges'"
                                      " AND src_col='src' AND dst_col='dst' AND timestamp_col='ts'"
                                      " AND time_start='2026-01-01' AND time_end='2026-01-31'"),
                  3);
    sqlite3_close(db);
}

TEST(test_sql_leiden_errors_and_empty) {
    sqlite3 *db = open_two_triangles_db();
    ASSERT(run_sql(db, "SELECT * FROM graph_leiden WHERE edge_table='nope' AND src_col='src' AND dst_col='dst'") !=
           SQLITE_OK);
    ASSERT(run_sql(db, "SELECT * FROM graph_leiden WHERE edge_table='edges' AND src_col='x;' AND dst_col='dst'") !=
           SQLITE_OK);
    /* no constraints at all: empty result, not an error */
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_leiden"), 0);
    run_sql(db, "CREATE TABLE empty_edges (src TEXT, dst TEXT)");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_leiden WHERE edge_table='empty_edges'"
                                      " AND src_col='src' AND dst_col='dst'"),
                  0);
    sqlite3_close(db);
}

TEST(test_sql_leiden_over_adjacency_vt) {
    sqlite3 *db = open_two_triangles_db();
    run_sql(db, "CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst')");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(DISTINCT community_id) FROM graph_leiden"
                                      " WHERE edge_table='g' AND src_col='src' AND dst_col='dst'"),
                  2);
    sqlite3_close(db);
}

TEST(test_sql_conductance_two_triangles) {
    sqlite3 *db = open_two_triangles_db();
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_conductance" CND_WHERE), 2);
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT size FROM graph_conductance" CND_WHERE " AND group_id='left'"), 4);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT internal FROM graph_conductance" CND_WHERE " AND group_id='left'"), 5.0,
                    1e-9);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT cut FROM graph_conductance" CND_WHERE " AND group_id='right'"), 1.0,
                    1e-9);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT vol FROM graph_conductance" CND_WHERE " AND group_id='right'"), 7.0,
                    1e-9);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT phi FROM graph_conductance" CND_WHERE " AND group_id='left'"), 1.0 / 7.0,
                    1e-6);
    /* weighted + temporal constraints are accepted */
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT cut FROM graph_conductance WHERE edge_table='edges' AND src_col='src'"
                                      " AND dst_col='dst' AND weight_col='weight' AND timestamp_col='ts'"
                                      " AND time_start='2026-01-01' AND time_end='2026-02-28'"
                                      " AND membership_table='assignment' AND group_col='team'"
                                      " AND member_col='node' AND group_id='left'"),
                    0.0, 1e-9);
    sqlite3_close(db);
}

TEST(test_sql_conductance_leiden_composition_and_vt) {
    sqlite3 *db = open_two_triangles_db();
    run_sql(db, "CREATE TEMP TABLE discovered AS SELECT node, community_id FROM graph_leiden"
                " WHERE edge_table='edges' AND src_col='src' AND dst_col='dst' AND direction='both';"
                "CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst');");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_conductance WHERE edge_table='g'"
                                      " AND src_col='src' AND dst_col='dst' AND membership_table='discovered'"
                                      " AND group_col='community_id' AND member_col='node'"),
                  2);
    ASSERT_EQ_FLOAT(scalar_double(db, "SELECT MAX(phi) FROM graph_conductance WHERE edge_table='g'"
                                      " AND src_col='src' AND dst_col='dst' AND membership_table='discovered'"
                                      " AND group_col='community_id' AND member_col='node'"),
                    1.0 / 7.0, 1e-6);
    sqlite3_close(db);
}

TEST(test_sql_conductance_errors) {
    sqlite3 *db = open_two_triangles_db();
    /* missing membership triple */
    ASSERT(run_sql(db, "SELECT * FROM graph_conductance WHERE edge_table='edges' AND src_col='src'"
                       " AND dst_col='dst'") != SQLITE_OK);
    /* bad edge table */
    ASSERT(run_sql(db, "SELECT * FROM graph_conductance WHERE edge_table='nope' AND src_col='src' AND dst_col='dst'"
                       " AND membership_table='assignment' AND group_col='team' AND member_col='node'") != SQLITE_OK);
    /* injection in membership identifiers */
    ASSERT(run_sql(db, "SELECT * FROM graph_conductance WHERE edge_table='edges' AND src_col='src' AND dst_col='dst'"
                       " AND membership_table='assignment; DROP TABLE edges' AND group_col='team'"
                       " AND member_col='node'") != SQLITE_OK);
    /* missing membership table */
    ASSERT(run_sql(db, "SELECT * FROM graph_conductance WHERE edge_table='edges' AND src_col='src' AND dst_col='dst'"
                       " AND membership_table='ghost' AND group_col='team' AND member_col='node'") != SQLITE_OK);
    /* NULL rows and unknown members are skipped; empty membership gives no rows */
    run_sql(db, "INSERT INTO assignment VALUES (NULL,'left'), ('zed',NULL), ('zed','left')");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT size FROM graph_conductance" CND_WHERE " AND group_id='left'"), 4);
    run_sql(db, "DELETE FROM assignment");
    ASSERT_EQ_INT((int)scalar_int(db, "SELECT COUNT(*) FROM graph_conductance" CND_WHERE), 0);
    sqlite3_close(db);
}

/* ─── Entry point ──────────────────────────────────────────── */

void test_graph_community(void) {
    RUN_TEST(test_conductance_two_groups);
    RUN_TEST(test_conductance_ungrouped_node);
    RUN_TEST(test_conductance_singleton_no_internal_edges_is_one);
    RUN_TEST(test_conductance_whole_graph_is_zero);
    RUN_TEST(test_conductance_disconnected_component_is_zero);
    RUN_TEST(test_conductance_no_groups);
    RUN_TEST(test_conductance_empty_graph);
    RUN_TEST(test_conductance_weighted);
    RUN_TEST(test_conductance_direction_reverse);
    RUN_TEST(test_conductance_forward_matches_both);
    RUN_TEST(test_leiden_barbell_two_communities);
    RUN_TEST(test_leiden_conductance_of_found_partition);
    RUN_TEST(test_leiden_high_resolution_splits_more);
    RUN_TEST(test_leiden_empty_and_edgeless);
    RUN_TEST(test_leiden_forward_direction);
    RUN_TEST(test_sql_leiden_two_triangles);
    RUN_TEST(test_sql_leiden_errors_and_empty);
    RUN_TEST(test_sql_leiden_over_adjacency_vt);
    RUN_TEST(test_sql_conductance_two_triangles);
    RUN_TEST(test_sql_conductance_leiden_composition_and_vt);
    RUN_TEST(test_sql_conductance_errors);
}
