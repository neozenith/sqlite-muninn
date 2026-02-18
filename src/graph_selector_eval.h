/*
 * graph_selector_eval.h — Evaluator for selector expression ASTs
 *
 * Takes a parsed SelectorNode AST and evaluates it against a GraphData
 * structure, producing a set of selected node indices.
 */
#ifndef GRAPH_SELECTOR_EVAL_H
#define GRAPH_SELECTOR_EVAL_H

#include "graph_load.h"
#include "graph_selector_parse.h"

#include <stdint.h>

/* Bit-vector set over node indices in a GraphData */
typedef struct {
    uint8_t *bits; /* 1 bit per node, rounded up to byte boundary */
    int capacity;  /* = GraphData.node_count, rounded up to multiple of 8 */
} NodeSet;

/* Initialize a NodeSet for a graph with `node_count` nodes (all bits clear). */
void nodeset_init(NodeSet *ns, int node_count);

/* Free the bit vector. */
void nodeset_destroy(NodeSet *ns);

/* Set/clear/test a single bit. */
void nodeset_set(NodeSet *ns, int idx);
void nodeset_clear(NodeSet *ns, int idx);
int nodeset_test(const NodeSet *ns, int idx);

/* Count set bits. */
int nodeset_count(const NodeSet *ns, int node_count);

/* Set operations: dst = a OP b. dst may alias a or b. */
void nodeset_union(NodeSet *dst, const NodeSet *a, const NodeSet *b);
void nodeset_intersect(NodeSet *dst, const NodeSet *a, const NodeSet *b);
void nodeset_complement(NodeSet *dst, const NodeSet *a, int node_count);

/* Per-node evaluation result (for output columns) */
typedef struct {
    int node_idx;          /* index into GraphData.ids[] */
    int depth;             /* distance from anchor, 0 = self */
    const char *direction; /* "self", "ancestor", "descendant", or "closure" */
} SelectResult;

typedef struct {
    SelectResult *rows;
    int count;
    int capacity;
} SelectResults;

/*
 * Evaluate a selector AST against a loaded graph.
 *
 * Returns 0 on success, non-zero on error.
 * On success, results->rows is populated (caller must free via select_results_destroy).
 * On error, *pzErrMsg is set (caller must free).
 */
int selector_eval(const SelectorNode *ast, const GraphData *g, SelectResults *results, char **pzErrMsg);

/* Free results. */
void select_results_destroy(SelectResults *results);

#endif /* GRAPH_SELECTOR_EVAL_H */
