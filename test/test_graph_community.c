/*
 * test_graph_community.c — Unit tests for run_conductance()
 *
 * Builds GraphData in memory (no SQLite needed) using the two-triangle
 * graph from docs/centrality-community.md: alice/bob/carol/dave and
 * eve/frank/grace, joined by the single bridge dave -> eve.
 */
#include "test_common.h"
#include "graph_community.h"

#include <string.h>

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
}
