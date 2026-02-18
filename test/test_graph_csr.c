/*
 * test_graph_csr.c — Unit tests for CSR build, query, and delta merge
 */
#include "test_common.h"
#include "graph_csr.h"
#include "graph_load.h"

#include <stdlib.h>
#include <string.h>

/* Helper: manually add an edge to out[src] and in[dst] */
static void add_edge(GraphData *g, int src, int dst, double weight) {
    GraphAdjList *out = &g->out[src];
    if (out->count >= out->capacity) {
        int nc = out->capacity == 0 ? 4 : out->capacity * 2;
        out->edges = (GraphEdge *)realloc(out->edges, (size_t)nc * sizeof(GraphEdge));
        out->capacity = nc;
    }
    out->edges[out->count++] = (GraphEdge){.target = dst, .weight = weight};

    GraphAdjList *in = &g->in[dst];
    if (in->count >= in->capacity) {
        int nc = in->capacity == 0 ? 4 : in->capacity * 2;
        in->edges = (GraphEdge *)realloc(in->edges, (size_t)nc * sizeof(GraphEdge));
        in->capacity = nc;
    }
    in->edges[in->count++] = (GraphEdge){.target = src, .weight = weight};

    g->edge_count++;
}

/* ── Build + Query ──────────────────────────────────────────── */

TEST(csr_empty_graph) {
    GraphData g;
    graph_data_init(&g);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    ASSERT_EQ_INT(fwd.node_count, 0);
    ASSERT_EQ_INT(fwd.edge_count, 0);
    ASSERT_EQ_INT(rev.node_count, 0);
    ASSERT_EQ_INT(rev.edge_count, 0);

    /* Out-of-range queries should be safe */
    ASSERT_EQ_INT(csr_degree(&fwd, 0), 0);
    ASSERT_EQ_INT(csr_degree(&fwd, -1), 0);
    int32_t cnt;
    ASSERT(csr_neighbors(&fwd, 0, &cnt) == NULL);
    ASSERT_EQ_INT(cnt, 0);

    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
}

TEST(csr_triangle) {
    /* Build a triangle: A→B, B→C, C→A */
    GraphData g;
    graph_data_init(&g);

    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");

    add_edge(&g, a, b, 1.0);
    add_edge(&g, b, c, 1.0);
    add_edge(&g, c, a, 1.0);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    ASSERT_EQ_INT(fwd.node_count, 3);
    ASSERT_EQ_INT(fwd.edge_count, 3);
    ASSERT_EQ_INT(rev.node_count, 3);
    ASSERT_EQ_INT(rev.edge_count, 3);

    /* Check forward CSR: each node has out-degree 1 */
    ASSERT_EQ_INT(csr_degree(&fwd, 0), 1); /* A→B */
    ASSERT_EQ_INT(csr_degree(&fwd, 1), 1); /* B→C */
    ASSERT_EQ_INT(csr_degree(&fwd, 2), 1); /* C→A */

    /* Check A's forward neighbor is B */
    int32_t cnt2;
    const int32_t *nbrs = csr_neighbors(&fwd, 0, &cnt2);
    ASSERT_EQ_INT(cnt2, 1);
    ASSERT_EQ_INT(nbrs[0], b);

    /* Check reverse CSR: each node has in-degree 1 */
    ASSERT_EQ_INT(csr_degree(&rev, 0), 1); /* A←C */
    ASSERT_EQ_INT(csr_degree(&rev, 1), 1); /* B←A */
    ASSERT_EQ_INT(csr_degree(&rev, 2), 1); /* C←B */

    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
}

/* ── Serialization Round-Trip ──────────────────────────────── */

TEST(csr_serialize_roundtrip) {
    /* Build a simple graph, serialize offsets/targets, deserialize */
    GraphData g;
    graph_data_init(&g);

    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");

    add_edge(&g, a, b, 1.0);
    add_edge(&g, a, a, 1.0); /* self-loop */

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    /* Deserialize from the raw arrays (simulating BLOB storage) */
    CsrArray fwd2;
    int offsets_bytes = (fwd.node_count + 1) * (int)sizeof(int32_t);
    int targets_bytes = fwd.edge_count * (int)sizeof(int32_t);
    ASSERT(csr_deserialize(&fwd2, fwd.offsets, offsets_bytes, fwd.targets, targets_bytes, NULL, 0) == 0);

    ASSERT_EQ_INT(fwd2.node_count, fwd.node_count);
    ASSERT_EQ_INT(fwd2.edge_count, fwd.edge_count);

    /* Compare offsets */
    for (int i = 0; i <= fwd.node_count; i++)
        ASSERT_EQ_INT(fwd2.offsets[i], fwd.offsets[i]);

    /* Compare targets */
    for (int i = 0; i < fwd.edge_count; i++)
        ASSERT_EQ_INT(fwd2.targets[i], fwd.targets[i]);

    csr_destroy(&fwd2);
    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
}

/* ── Delta Merge ───────────────────────────────────────────── */

TEST(csr_delta_insert) {
    /* Start with A→B, apply delta: insert A→C */
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");

    add_edge(&g, a, b, 1.0);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    /* Original: A has 1 out-neighbor (B) */
    ASSERT_EQ_INT(csr_degree(&fwd, a), 1);

    /* Apply delta: insert A→C */
    CsrDelta delta = {.src_idx = (int32_t)a, .dst_idx = (int32_t)c, .weight = 1.0, .op = 1};
    CsrArray new_fwd;
    ASSERT(csr_apply_delta(&fwd, &delta, 1, 3, &new_fwd) == 0);

    /* Now A should have 2 neighbors */
    ASSERT_EQ_INT(csr_degree(&new_fwd, a), 2);
    ASSERT_EQ_INT(csr_degree(&new_fwd, b), 0);
    ASSERT_EQ_INT(csr_degree(&new_fwd, c), 0);

    csr_destroy(&fwd);
    csr_destroy(&rev);
    csr_destroy(&new_fwd);
    graph_data_destroy(&g);
}

TEST(csr_delta_delete) {
    /* Start with A→B and A→C, apply delta: delete A→B */
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");

    add_edge(&g, a, b, 1.0);
    add_edge(&g, a, c, 1.0);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    ASSERT_EQ_INT(csr_degree(&fwd, a), 2);

    /* Apply delta: delete A→B */
    CsrDelta delta = {.src_idx = (int32_t)a, .dst_idx = (int32_t)b, .weight = 1.0, .op = 2};
    CsrArray new_fwd;
    ASSERT(csr_apply_delta(&fwd, &delta, 1, 3, &new_fwd) == 0);

    /* Now A should have 1 neighbor (C) */
    ASSERT_EQ_INT(csr_degree(&new_fwd, a), 1);
    int32_t cnt;
    const int32_t *nbrs = csr_neighbors(&new_fwd, a, &cnt);
    ASSERT_EQ_INT(cnt, 1);
    ASSERT_EQ_INT(nbrs[0], c);

    csr_destroy(&fwd);
    csr_destroy(&rev);
    csr_destroy(&new_fwd);
    graph_data_destroy(&g);
}

TEST(csr_delta_add_new_node) {
    /* Start with A→B, apply delta: insert A→D where D is a new node */
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");

    add_edge(&g, a, b, 1.0);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    /* CSR has 2 nodes. Delta introduces node index 2 (D) */
    CsrDelta delta = {.src_idx = (int32_t)a, .dst_idx = 2, .weight = 1.0, .op = 1};
    CsrArray new_fwd;
    ASSERT(csr_apply_delta(&fwd, &delta, 1, 3, &new_fwd) == 0);

    ASSERT_EQ_INT(new_fwd.node_count, 3);
    ASSERT_EQ_INT(csr_degree(&new_fwd, a), 2); /* A→B and A→D */
    ASSERT_EQ_INT(csr_degree(&new_fwd, b), 0);
    ASSERT_EQ_INT(csr_degree(&new_fwd, 2), 0); /* D has no outgoing */

    csr_destroy(&fwd);
    csr_destroy(&rev);
    csr_destroy(&new_fwd);
    graph_data_destroy(&g);
}

/* ── Blocked CSR (Phase 3) ─────────────────────────────────── */

TEST(csr_block_count_basic) {
    /* 10 nodes, block_size=4 → 3 blocks (4+4+2) */
    ASSERT_EQ_INT(csr_block_count(10, 4), 3);
    /* Exact multiple */
    ASSERT_EQ_INT(csr_block_count(8, 4), 2);
    /* Smaller than one block */
    ASSERT_EQ_INT(csr_block_count(3, 4), 1);
    /* Edge cases */
    ASSERT_EQ_INT(csr_block_count(0, 4), 0);
    ASSERT_EQ_INT(csr_block_count(10, 0), 0);
}

TEST(csr_extract_block_single) {
    /* Build 4-node chain: A→B→C→D, extract block [0,2) and [2,4) */
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");
    int d = graph_data_find_or_add(&g, "D");

    add_edge(&g, a, b, 1.0);
    add_edge(&g, b, c, 2.0);
    add_edge(&g, c, d, 3.0);
    (void)d; /* suppress unused */

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    /* Block 0: nodes [0, 2) = A, B */
    CsrArray blk0;
    ASSERT(csr_extract_block(&fwd, 0, 2, &blk0) == 0);
    ASSERT_EQ_INT(blk0.node_count, 2);
    ASSERT_EQ_INT(csr_degree(&blk0, 0), 1); /* A→B (target is global: 1) */
    ASSERT_EQ_INT(csr_degree(&blk0, 1), 1); /* B→C (target is global: 2) */

    /* Verify targets are global indices */
    int32_t cnt;
    const int32_t *nbrs = csr_neighbors(&blk0, 0, &cnt);
    ASSERT_EQ_INT(cnt, 1);
    ASSERT_EQ_INT(nbrs[0], b); /* global index of B */

    /* Block 1: nodes [2, 4) = C, D */
    CsrArray blk1;
    ASSERT(csr_extract_block(&fwd, 2, 2, &blk1) == 0);
    ASSERT_EQ_INT(blk1.node_count, 2);
    ASSERT_EQ_INT(csr_degree(&blk1, 0), 1); /* C→D */
    ASSERT_EQ_INT(csr_degree(&blk1, 1), 0); /* D has no outgoing */

    csr_destroy(&blk0);
    csr_destroy(&blk1);
    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
}

TEST(csr_block_roundtrip) {
    /* Build graph, split into blocks, merge back, verify identical */
    GraphData g;
    graph_data_init(&g);
    int a = graph_data_find_or_add(&g, "A");
    int b = graph_data_find_or_add(&g, "B");
    int c = graph_data_find_or_add(&g, "C");
    int d = graph_data_find_or_add(&g, "D");
    int e = graph_data_find_or_add(&g, "E");

    add_edge(&g, a, b, 1.0);
    add_edge(&g, a, c, 2.0);
    add_edge(&g, b, d, 3.0);
    add_edge(&g, c, e, 4.0);
    add_edge(&g, d, e, 5.0);

    CsrArray fwd, rev;
    ASSERT(csr_build(&g, &fwd, &rev) == 0);

    /* Split with block_size=2: blocks 0,1,2 covering (2+2+1) nodes */
    int bs = 2;
    int nblk = csr_block_count(fwd.node_count, bs);
    ASSERT_EQ_INT(nblk, 3);

    CsrArray blocks[3];
    for (int i = 0; i < nblk; i++) {
        int32_t start = (int32_t)(i * bs);
        int32_t count = (int32_t)bs;
        if (start + count > fwd.node_count)
            count = fwd.node_count - start;
        ASSERT(csr_extract_block(&fwd, start, count, &blocks[i]) == 0);
    }

    /* Merge back */
    CsrArray merged;
    ASSERT(csr_merge_blocks(blocks, nblk, bs, fwd.node_count, &merged) == 0);

    /* Verify identical to original */
    ASSERT_EQ_INT(merged.node_count, fwd.node_count);
    ASSERT_EQ_INT(merged.edge_count, fwd.edge_count);
    for (int i = 0; i <= fwd.node_count; i++)
        ASSERT_EQ_INT(merged.offsets[i], fwd.offsets[i]);
    for (int i = 0; i < fwd.edge_count; i++)
        ASSERT_EQ_INT(merged.targets[i], fwd.targets[i]);

    for (int i = 0; i < nblk; i++)
        csr_destroy(&blocks[i]);
    csr_destroy(&merged);
    csr_destroy(&fwd);
    csr_destroy(&rev);
    graph_data_destroy(&g);
}

/* ── Entry Point ───────────────────────────────────────────── */

void test_graph_csr(void) {
    RUN_TEST(csr_empty_graph);
    RUN_TEST(csr_triangle);
    RUN_TEST(csr_serialize_roundtrip);
    RUN_TEST(csr_delta_insert);
    RUN_TEST(csr_delta_delete);
    RUN_TEST(csr_delta_add_new_node);
    RUN_TEST(csr_block_count_basic);
    RUN_TEST(csr_extract_block_single);
    RUN_TEST(csr_block_roundtrip);
}
