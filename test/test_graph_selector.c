/*
 * test_graph_selector.c — Unit tests for selector parser and evaluator
 */
#include "test_common.h"
#include "graph_load.h"
#include "graph_selector_eval.h"
#include "graph_selector_parse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Parser tests
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_parse_bare_node) {
    char *err = NULL;
    SelectorNode *n = selector_parse("my_node", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_NODE);
    ASSERT(strcmp(n->value, "my_node") == 0);
    ASSERT_EQ_INT(n->depth_up, -1);
    ASSERT_EQ_INT(n->depth_down, -1);
    selector_free(n);
}

TEST(test_parse_ancestors) {
    char *err = NULL;
    SelectorNode *n = selector_parse("+node_a", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_ANCESTORS);
    ASSERT(strcmp(n->value, "node_a") == 0);
    ASSERT_EQ_INT(n->depth_up, -1);
    selector_free(n);
}

TEST(test_parse_descendants) {
    char *err = NULL;
    SelectorNode *n = selector_parse("node_b+", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_DESCENDANTS);
    ASSERT(strcmp(n->value, "node_b") == 0);
    ASSERT_EQ_INT(n->depth_down, -1);
    selector_free(n);
}

TEST(test_parse_both) {
    char *err = NULL;
    SelectorNode *n = selector_parse("+node_c+", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_BOTH);
    ASSERT(strcmp(n->value, "node_c") == 0);
    ASSERT_EQ_INT(n->depth_up, -1);
    ASSERT_EQ_INT(n->depth_down, -1);
    selector_free(n);
}

TEST(test_parse_depth_limited_ancestors) {
    char *err = NULL;
    SelectorNode *n = selector_parse("2+my_node", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_ANCESTORS);
    ASSERT(strcmp(n->value, "my_node") == 0);
    ASSERT_EQ_INT(n->depth_up, 2);
    selector_free(n);
}

TEST(test_parse_depth_limited_descendants) {
    char *err = NULL;
    SelectorNode *n = selector_parse("my_node+3", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_DESCENDANTS);
    ASSERT(strcmp(n->value, "my_node") == 0);
    ASSERT_EQ_INT(n->depth_down, 3);
    selector_free(n);
}

TEST(test_parse_depth_limited_both) {
    char *err = NULL;
    SelectorNode *n = selector_parse("2+build_step+3", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_BOTH);
    ASSERT(strcmp(n->value, "build_step") == 0);
    ASSERT_EQ_INT(n->depth_up, 2);
    ASSERT_EQ_INT(n->depth_down, 3);
    selector_free(n);
}

TEST(test_parse_closure) {
    char *err = NULL;
    SelectorNode *n = selector_parse("@critical", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_CLOSURE);
    ASSERT(strcmp(n->value, "critical") == 0);
    selector_free(n);
}

TEST(test_parse_union) {
    char *err = NULL;
    SelectorNode *n = selector_parse("+a +b", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_UNION);
    ASSERT(n->left != NULL);
    ASSERT(n->right != NULL);
    ASSERT_EQ_INT(n->left->type, SEL_ANCESTORS);
    ASSERT_EQ_INT(n->right->type, SEL_ANCESTORS);
    selector_free(n);
}

TEST(test_parse_intersection) {
    char *err = NULL;
    SelectorNode *n = selector_parse("+a,+b", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_INTERSECT);
    ASSERT(n->left != NULL);
    ASSERT(n->right != NULL);
    ASSERT_EQ_INT(n->left->type, SEL_ANCESTORS);
    ASSERT_EQ_INT(n->right->type, SEL_ANCESTORS);
    selector_free(n);
}

TEST(test_parse_complement) {
    char *err = NULL;
    SelectorNode *n = selector_parse("not @root", &err);
    ASSERT(n != NULL);
    ASSERT_EQ_INT(n->type, SEL_COMPLEMENT);
    ASSERT(n->left != NULL);
    ASSERT_EQ_INT(n->left->type, SEL_CLOSURE);
    selector_free(n);
}

TEST(test_parse_empty_error) {
    char *err = NULL;
    SelectorNode *n = selector_parse("", &err);
    ASSERT(n == NULL);
    ASSERT(err != NULL);
    free(err);
}

/* ═══════════════════════════════════════════════════════════════
 * Evaluator tests — build a small graph and test selections
 *
 * Graph:
 *   A → C → D
 *   B → C → E → F
 *   X → Y → E
 * ═══════════════════════════════════════════════════════════════ */

/* Helper: add an edge to the graph */
static void add_edge(GraphData *g, const char *src, const char *dst) {
    int si = graph_data_find_or_add(g, src);
    int di = graph_data_find_or_add(g, dst);

    /* Grow out[si] */
    GraphAdjList *out = &g->out[si];
    if (out->count >= out->capacity) {
        int nc = out->capacity == 0 ? 4 : out->capacity * 2;
        out->edges = (GraphEdge *)realloc(out->edges, (size_t)nc * sizeof(GraphEdge));
        out->capacity = nc;
    }
    out->edges[out->count].target = di;
    out->edges[out->count].weight = 1.0;
    out->count++;

    /* Grow in[di] */
    GraphAdjList *in = &g->in[di];
    if (in->count >= in->capacity) {
        int nc = in->capacity == 0 ? 4 : in->capacity * 2;
        in->edges = (GraphEdge *)realloc(in->edges, (size_t)nc * sizeof(GraphEdge));
        in->capacity = nc;
    }
    in->edges[in->count].target = si;
    in->edges[in->count].weight = 1.0;
    in->count++;

    g->edge_count++;
}

static void build_test_graph(GraphData *g) {
    graph_data_init(g);
    /* Pre-add all nodes so indices are deterministic:
     * A=0, B=1, C=2, D=3, E=4, F=5, X=6, Y=7 */
    graph_data_find_or_add(g, "A");
    graph_data_find_or_add(g, "B");
    graph_data_find_or_add(g, "C");
    graph_data_find_or_add(g, "D");
    graph_data_find_or_add(g, "E");
    graph_data_find_or_add(g, "F");
    graph_data_find_or_add(g, "X");
    graph_data_find_or_add(g, "Y");

    /* Edges: A→C, B→C, C→D, C→E, E→F, X→Y, Y→E */
    add_edge(g, "A", "C");
    add_edge(g, "B", "C");
    add_edge(g, "C", "D");
    add_edge(g, "C", "E");
    add_edge(g, "E", "F");
    add_edge(g, "X", "Y");
    add_edge(g, "Y", "E");
}

/* Helper: check if a node name is in results */
static int result_has_node(const SelectResults *r, const GraphData *g, const char *name) {
    for (int i = 0; i < r->count; i++) {
        if (strcmp(g->ids[r->rows[i].node_idx], name) == 0)
            return 1;
    }
    return 0;
}

TEST(test_eval_bare_node) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("C", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(results.count, 1);
    ASSERT(result_has_node(&results, &g, "C"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_descendants) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("C+", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* C+ = {C, D, E, F} */
    ASSERT_EQ_INT(results.count, 4);
    ASSERT(result_has_node(&results, &g, "C"));
    ASSERT(result_has_node(&results, &g, "D"));
    ASSERT(result_has_node(&results, &g, "E"));
    ASSERT(result_has_node(&results, &g, "F"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_ancestors) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("+C", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* +C = {A, B, C} */
    ASSERT_EQ_INT(results.count, 3);
    ASSERT(result_has_node(&results, &g, "A"));
    ASSERT(result_has_node(&results, &g, "B"));
    ASSERT(result_has_node(&results, &g, "C"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_both) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("+C+", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* +C+ = {A, B, C, D, E, F} */
    ASSERT_EQ_INT(results.count, 6);
    ASSERT(result_has_node(&results, &g, "A"));
    ASSERT(result_has_node(&results, &g, "B"));
    ASSERT(result_has_node(&results, &g, "C"));
    ASSERT(result_has_node(&results, &g, "D"));
    ASSERT(result_has_node(&results, &g, "E"));
    ASSERT(result_has_node(&results, &g, "F"));
    /* X and Y should NOT be included */
    ASSERT(!result_has_node(&results, &g, "X"));
    ASSERT(!result_has_node(&results, &g, "Y"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_closure) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("@C", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* @C = descendants of C + all ancestors of those descendants
     * C→D, C→E→F, E also has ancestor Y→E and X→Y
     * Descendants: {C, D, E, F}
     * Ancestors of D: {A, B, C}
     * Ancestors of E: {A, B, C, X, Y}
     * Ancestors of F: {A, B, C, X, Y, E}
     * Total: {A, B, C, D, E, F, X, Y} = all 8 nodes */
    ASSERT_EQ_INT(results.count, 8);

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_depth_limited) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("C+1", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* C+1 = {C, D, E} (only 1 hop downstream) */
    ASSERT_EQ_INT(results.count, 3);
    ASSERT(result_has_node(&results, &g, "C"));
    ASSERT(result_has_node(&results, &g, "D"));
    ASSERT(result_has_node(&results, &g, "E"));
    ASSERT(!result_has_node(&results, &g, "F")); /* 2 hops away */

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_union) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("A B", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    /* A union B = {A, B} */
    ASSERT_EQ_INT(results.count, 2);
    ASSERT(result_has_node(&results, &g, "A"));
    ASSERT(result_has_node(&results, &g, "B"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_intersection) {
    GraphData g;
    build_test_graph(&g);

    /* +D = ancestors of D = {A, B, C, D}
     * +E = ancestors of E = {A, B, C, E, X, Y}
     * Intersection = {A, B, C} */
    SelectorNode *ast = selector_parse("+D,+E", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(results.count, 3);
    ASSERT(result_has_node(&results, &g, "A"));
    ASSERT(result_has_node(&results, &g, "B"));
    ASSERT(result_has_node(&results, &g, "C"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_complement) {
    GraphData g;
    build_test_graph(&g);

    /* not C = all nodes except C = {A, B, D, E, F, X, Y} */
    SelectorNode *ast = selector_parse("not C", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(results.count, 7);
    ASSERT(!result_has_node(&results, &g, "C"));
    ASSERT(result_has_node(&results, &g, "A"));

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

TEST(test_eval_node_not_found) {
    GraphData g;
    build_test_graph(&g);

    SelectorNode *ast = selector_parse("nonexistent", NULL);
    ASSERT(ast != NULL);

    SelectResults results;
    char *err = NULL;
    int rc = selector_eval(ast, &g, &results, &err);
    ASSERT(rc != 0);
    ASSERT(err != NULL);
    free(err);

    select_results_destroy(&results);
    selector_free(ast);
    graph_data_destroy(&g);
}

/* ═══════════════════════════════════════════════════════════════
 * NodeSet tests
 * ═══════════════════════════════════════════════════════════════ */

TEST(test_nodeset_basic) {
    NodeSet ns;
    nodeset_init(&ns, 16);

    ASSERT_EQ_INT(nodeset_test(&ns, 0), 0);
    nodeset_set(&ns, 0);
    ASSERT_EQ_INT(nodeset_test(&ns, 0), 1);
    nodeset_set(&ns, 7);
    ASSERT_EQ_INT(nodeset_test(&ns, 7), 1);
    ASSERT_EQ_INT(nodeset_count(&ns, 16), 2);

    nodeset_clear(&ns, 0);
    ASSERT_EQ_INT(nodeset_test(&ns, 0), 0);
    ASSERT_EQ_INT(nodeset_count(&ns, 16), 1);

    nodeset_destroy(&ns);
}

TEST(test_nodeset_complement) {
    NodeSet ns, comp;
    nodeset_init(&ns, 8);
    nodeset_init(&comp, 8);

    nodeset_set(&ns, 0);
    nodeset_set(&ns, 3);
    nodeset_complement(&comp, &ns, 8);

    ASSERT_EQ_INT(nodeset_test(&comp, 0), 0);
    ASSERT_EQ_INT(nodeset_test(&comp, 1), 1);
    ASSERT_EQ_INT(nodeset_test(&comp, 2), 1);
    ASSERT_EQ_INT(nodeset_test(&comp, 3), 0);
    ASSERT_EQ_INT(nodeset_test(&comp, 4), 1);
    ASSERT_EQ_INT(nodeset_count(&comp, 8), 6);

    nodeset_destroy(&ns);
    nodeset_destroy(&comp);
}

/* ═══════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════ */

void test_graph_selector(void) {
    /* Parser tests */
    RUN_TEST(test_parse_bare_node);
    RUN_TEST(test_parse_ancestors);
    RUN_TEST(test_parse_descendants);
    RUN_TEST(test_parse_both);
    RUN_TEST(test_parse_depth_limited_ancestors);
    RUN_TEST(test_parse_depth_limited_descendants);
    RUN_TEST(test_parse_depth_limited_both);
    RUN_TEST(test_parse_closure);
    RUN_TEST(test_parse_union);
    RUN_TEST(test_parse_intersection);
    RUN_TEST(test_parse_complement);
    RUN_TEST(test_parse_empty_error);

    /* NodeSet tests */
    RUN_TEST(test_nodeset_basic);
    RUN_TEST(test_nodeset_complement);

    /* Evaluator tests */
    RUN_TEST(test_eval_bare_node);
    RUN_TEST(test_eval_descendants);
    RUN_TEST(test_eval_ancestors);
    RUN_TEST(test_eval_both);
    RUN_TEST(test_eval_closure);
    RUN_TEST(test_eval_depth_limited);
    RUN_TEST(test_eval_union);
    RUN_TEST(test_eval_intersection);
    RUN_TEST(test_eval_complement);
    RUN_TEST(test_eval_node_not_found);
}
