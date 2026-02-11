/*
 * test_graph_load.c — Unit tests for the graph_load hash map and adjacency list
 *
 * Tests the in-memory data structure only (no SQLite needed).
 */
#include "test_common.h"
#include "graph_load.h"

#include <stdio.h>
#include <string.h>

/* ─── Hash map basics ──────────────────────────────────────── */

TEST(test_init_destroy) {
    GraphData g;
    graph_data_init(&g);
    ASSERT_EQ_INT(g.node_count, 0);
    ASSERT(g.map_capacity >= 16);
    graph_data_destroy(&g);
}

TEST(test_find_empty) {
    GraphData g;
    graph_data_init(&g);
    ASSERT_EQ_INT(graph_data_find(&g, "nonexistent"), -1);
    graph_data_destroy(&g);
}

TEST(test_find_or_add_single) {
    GraphData g;
    graph_data_init(&g);
    int idx = graph_data_find_or_add(&g, "Alice");
    ASSERT_EQ_INT(idx, 0);
    ASSERT_EQ_INT(g.node_count, 1);
    ASSERT(strcmp(g.ids[0], "Alice") == 0);

    /* Second lookup returns same index */
    int idx2 = graph_data_find_or_add(&g, "Alice");
    ASSERT_EQ_INT(idx2, 0);
    ASSERT_EQ_INT(g.node_count, 1);

    graph_data_destroy(&g);
}

TEST(test_find_or_add_multiple) {
    GraphData g;
    graph_data_init(&g);

    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");

    ASSERT_EQ_INT(a, 0);
    ASSERT_EQ_INT(b, 1);
    ASSERT_EQ_INT(c, 2);
    ASSERT_EQ_INT(g.node_count, 3);

    /* Verify find works */
    ASSERT_EQ_INT(graph_data_find(&g, "A"), 0);
    ASSERT_EQ_INT(graph_data_find(&g, "B"), 1);
    ASSERT_EQ_INT(graph_data_find(&g, "C"), 2);
    ASSERT_EQ_INT(graph_data_find(&g, "D"), -1);

    graph_data_destroy(&g);
}

TEST(test_hash_map_resize) {
    /* Insert enough nodes to trigger multiple rehashes */
    GraphData g;
    graph_data_init(&g);

    char buf[32];
    for (int i = 0; i < 500; i++) {
        snprintf(buf, sizeof(buf), "node_%d", i);
        int idx = graph_data_find_or_add(&g, buf);
        ASSERT_EQ_INT(idx, i);
    }
    ASSERT_EQ_INT(g.node_count, 500);

    /* Verify all lookups still work after rehashing */
    for (int i = 0; i < 500; i++) {
        snprintf(buf, sizeof(buf), "node_%d", i);
        ASSERT_EQ_INT(graph_data_find(&g, buf), i);
    }

    graph_data_destroy(&g);
}

TEST(test_adjacency_lists_init_empty) {
    GraphData g;
    graph_data_init(&g);
    graph_data_find_or_add(&g, "X");

    ASSERT_EQ_INT(g.out[0].count, 0);
    ASSERT_EQ_INT(g.in[0].count, 0);

    graph_data_destroy(&g);
}

/* ─── Entry point ──────────────────────────────────────────── */

void test_graph_load(void) {
    RUN_TEST(test_init_destroy);
    RUN_TEST(test_find_empty);
    RUN_TEST(test_find_or_add_single);
    RUN_TEST(test_find_or_add_multiple);
    RUN_TEST(test_hash_map_resize);
    RUN_TEST(test_adjacency_lists_init_empty);
}
