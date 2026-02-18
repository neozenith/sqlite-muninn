/*
 * graph_selector_eval.c — Evaluates selector ASTs against a GraphData
 *
 * Uses bit-vector NodeSets for efficient set operations.
 * BFS traversal is used for ancestor/descendant computation.
 */
#include "graph_selector_eval.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════
 * NodeSet bit-vector operations
 * ═══════════════════════════════════════════════════════════════ */

void nodeset_init(NodeSet *ns, int node_count) {
    int bytes = (node_count + 7) / 8;
    if (bytes == 0)
        bytes = 1;
    ns->bits = (uint8_t *)calloc((size_t)bytes, 1);
    ns->capacity = bytes * 8;
}

void nodeset_destroy(NodeSet *ns) {
    free(ns->bits);
    ns->bits = NULL;
    ns->capacity = 0;
}

void nodeset_set(NodeSet *ns, int idx) {
    if (idx >= 0 && idx < ns->capacity)
        ns->bits[idx / 8] |= (uint8_t)(1 << (idx % 8));
}

void nodeset_clear(NodeSet *ns, int idx) {
    if (idx >= 0 && idx < ns->capacity)
        ns->bits[idx / 8] &= (uint8_t)~(1 << (idx % 8));
}

int nodeset_test(const NodeSet *ns, int idx) {
    if (idx < 0 || idx >= ns->capacity)
        return 0;
    return (ns->bits[idx / 8] >> (idx % 8)) & 1;
}

int nodeset_count(const NodeSet *ns, int node_count) {
    int count = 0;
    for (int i = 0; i < node_count; i++) {
        if (nodeset_test(ns, i))
            count++;
    }
    return count;
}

void nodeset_union(NodeSet *dst, const NodeSet *a, const NodeSet *b) {
    int bytes = (dst->capacity + 7) / 8;
    for (int i = 0; i < bytes; i++)
        dst->bits[i] = a->bits[i] | b->bits[i];
}

void nodeset_intersect(NodeSet *dst, const NodeSet *a, const NodeSet *b) {
    int bytes = (dst->capacity + 7) / 8;
    for (int i = 0; i < bytes; i++)
        dst->bits[i] = a->bits[i] & b->bits[i];
}

void nodeset_complement(NodeSet *dst, const NodeSet *a, int node_count) {
    int bytes = (node_count + 7) / 8;
    for (int i = 0; i < bytes; i++)
        dst->bits[i] = ~a->bits[i];
    /* Clear bits beyond node_count */
    int extra = bytes * 8 - node_count;
    if (extra > 0 && bytes > 0)
        dst->bits[bytes - 1] &= (uint8_t)((1 << (8 - extra)) - 1);
}

/* ═══════════════════════════════════════════════════════════════
 * BFS helpers for graph traversal
 * ═══════════════════════════════════════════════════════════════ */

/* Simple integer queue for BFS */
typedef struct {
    int *data;
    int head, tail, count, capacity;
} IntQueue;

static int iq_init(IntQueue *q, int cap) {
    q->data = (int *)malloc((size_t)cap * sizeof(int));
    if (!q->data)
        return -1;
    q->head = q->tail = q->count = 0;
    q->capacity = cap;
    return 0;
}

static void iq_destroy(IntQueue *q) {
    free(q->data);
}

static int iq_push(IntQueue *q, int val) {
    if (q->count >= q->capacity) {
        int new_cap = q->capacity * 2;
        int *new_data = (int *)malloc((size_t)new_cap * sizeof(int));
        if (!new_data)
            return -1;
        for (int i = 0; i < q->count; i++)
            new_data[i] = q->data[(q->head + i) % q->capacity];
        free(q->data);
        q->data = new_data;
        q->head = 0;
        q->tail = q->count;
        q->capacity = new_cap;
    }
    q->data[q->tail] = val;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    return 0;
}

static int iq_pop(IntQueue *q) {
    int val = q->data[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    return val;
}

/* Depth tracking: parallel array to BFS visited set */
typedef struct {
    int *depths;
    int capacity;
} DepthMap;

static int dm_init(DepthMap *dm, int cap) {
    dm->depths = (int *)malloc((size_t)cap * sizeof(int));
    if (!dm->depths)
        return -1;
    for (int i = 0; i < cap; i++)
        dm->depths[i] = -1;
    dm->capacity = cap;
    return 0;
}

static void dm_destroy(DepthMap *dm) {
    free(dm->depths);
}

/*
 * BFS forward (through out[] adjacency) from a set of seed nodes.
 * Fills `result` with all reachable nodes up to max_depth.
 * If depth_out is non-NULL, records the BFS depth for each node.
 */
static int bfs_forward(const GraphData *g, const NodeSet *seeds, int max_depth, NodeSet *result, DepthMap *depth_out) {
    IntQueue q;
    if (iq_init(&q, 64) != 0)
        return -1;

    for (int i = 0; i < g->node_count; i++) {
        if (nodeset_test(seeds, i)) {
            nodeset_set(result, i);
            iq_push(&q, i);
            if (depth_out)
                depth_out->depths[i] = 0;
        }
    }

    while (q.count > 0) {
        int cur = iq_pop(&q);
        int cur_depth = depth_out ? depth_out->depths[cur] : 0;

        if (max_depth >= 0 && cur_depth >= max_depth)
            continue;

        const GraphAdjList *adj = &g->out[cur];
        for (int e = 0; e < adj->count; e++) {
            int nbr = adj->edges[e].target;
            if (!nodeset_test(result, nbr)) {
                nodeset_set(result, nbr);
                if (depth_out)
                    depth_out->depths[nbr] = cur_depth + 1;
                iq_push(&q, nbr);
            }
        }
    }

    iq_destroy(&q);
    return 0;
}

/*
 * BFS backward (through in[] adjacency) from a set of seed nodes.
 */
static int bfs_backward(const GraphData *g, const NodeSet *seeds, int max_depth, NodeSet *result, DepthMap *depth_out) {
    IntQueue q;
    if (iq_init(&q, 64) != 0)
        return -1;

    for (int i = 0; i < g->node_count; i++) {
        if (nodeset_test(seeds, i)) {
            nodeset_set(result, i);
            iq_push(&q, i);
            if (depth_out)
                depth_out->depths[i] = 0;
        }
    }

    while (q.count > 0) {
        int cur = iq_pop(&q);
        int cur_depth = depth_out ? depth_out->depths[cur] : 0;

        if (max_depth >= 0 && cur_depth >= max_depth)
            continue;

        const GraphAdjList *adj = &g->in[cur];
        for (int e = 0; e < adj->count; e++) {
            int nbr = adj->edges[e].target;
            if (!nodeset_test(result, nbr)) {
                nodeset_set(result, nbr);
                if (depth_out)
                    depth_out->depths[nbr] = cur_depth + 1;
                iq_push(&q, nbr);
            }
        }
    }

    iq_destroy(&q);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * SelectResults management
 * ═══════════════════════════════════════════════════════════════ */

static void sr_init(SelectResults *r) {
    r->rows = NULL;
    r->count = 0;
    r->capacity = 0;
}

static int sr_add(SelectResults *r, int node_idx, int depth, const char *direction) {
    if (r->count >= r->capacity) {
        int new_cap = r->capacity == 0 ? 64 : r->capacity * 2;
        SelectResult *new_rows = (SelectResult *)realloc(r->rows, (size_t)new_cap * sizeof(SelectResult));
        if (!new_rows)
            return -1;
        r->rows = new_rows;
        r->capacity = new_cap;
    }
    r->rows[r->count].node_idx = node_idx;
    r->rows[r->count].depth = depth;
    r->rows[r->count].direction = direction;
    r->count++;
    return 0;
}

void select_results_destroy(SelectResults *results) {
    free(results->rows);
    results->rows = NULL;
    results->count = 0;
    results->capacity = 0;
}

/* ═══════════════════════════════════════════════════════════════
 * AST Evaluation — recursive
 *
 * Each eval function fills a NodeSet and optionally a DepthMap.
 * ═══════════════════════════════════════════════════════════════ */

/* Evaluate a single atom (node lookup). Returns -1 if node not found. */
static int eval_node_lookup(const GraphData *g, const char *name) {
    return graph_data_find(g, name);
}

/*
 * Core recursive evaluator. Returns 0 on success.
 * Fills `ns` (bit vector) and optionally `dm` (depths).
 */
static int eval_ast(const SelectorNode *ast, const GraphData *g, NodeSet *ns, DepthMap *dm, char **pzErrMsg) {
    switch (ast->type) {

    case SEL_NODE: {
        int idx = eval_node_lookup(g, ast->value);
        if (idx < 0) {
            if (pzErrMsg)
                *pzErrMsg = (char *)malloc(256);
            if (pzErrMsg && *pzErrMsg)
                snprintf(*pzErrMsg, 256, "graph_select: node '%s' not found", ast->value);
            return -1;
        }
        nodeset_set(ns, idx);
        if (dm)
            dm->depths[idx] = 0;
        return 0;
    }

    case SEL_ANCESTORS: {
        int idx = eval_node_lookup(g, ast->value);
        if (idx < 0) {
            if (pzErrMsg)
                *pzErrMsg = (char *)malloc(256);
            if (pzErrMsg && *pzErrMsg)
                snprintf(*pzErrMsg, 256, "graph_select: node '%s' not found", ast->value);
            return -1;
        }
        NodeSet seed;
        nodeset_init(&seed, g->node_count);
        nodeset_set(&seed, idx);
        if (dm)
            dm->depths[idx] = 0;
        bfs_backward(g, &seed, ast->depth_up, ns, dm);
        nodeset_destroy(&seed);
        return 0;
    }

    case SEL_DESCENDANTS: {
        int idx = eval_node_lookup(g, ast->value);
        if (idx < 0) {
            if (pzErrMsg)
                *pzErrMsg = (char *)malloc(256);
            if (pzErrMsg && *pzErrMsg)
                snprintf(*pzErrMsg, 256, "graph_select: node '%s' not found", ast->value);
            return -1;
        }
        NodeSet seed;
        nodeset_init(&seed, g->node_count);
        nodeset_set(&seed, idx);
        if (dm)
            dm->depths[idx] = 0;
        bfs_forward(g, &seed, ast->depth_down, ns, dm);
        nodeset_destroy(&seed);
        return 0;
    }

    case SEL_BOTH: {
        int idx = eval_node_lookup(g, ast->value);
        if (idx < 0) {
            if (pzErrMsg)
                *pzErrMsg = (char *)malloc(256);
            if (pzErrMsg && *pzErrMsg)
                snprintf(*pzErrMsg, 256, "graph_select: node '%s' not found", ast->value);
            return -1;
        }
        NodeSet seed;
        nodeset_init(&seed, g->node_count);
        nodeset_set(&seed, idx);

        /* Ancestors */
        DepthMap dm_up;
        dm_init(&dm_up, g->node_count);
        dm_up.depths[idx] = 0;
        bfs_backward(g, &seed, ast->depth_up, ns, &dm_up);

        /* Descendants */
        DepthMap dm_down;
        dm_init(&dm_down, g->node_count);
        dm_down.depths[idx] = 0;
        bfs_forward(g, &seed, ast->depth_down, ns, &dm_down);

        /* Merge depths into dm: prefer minimum */
        if (dm) {
            for (int i = 0; i < g->node_count; i++) {
                if (dm_up.depths[i] >= 0 && dm_down.depths[i] >= 0)
                    dm->depths[i] = dm_up.depths[i] < dm_down.depths[i] ? dm_up.depths[i] : dm_down.depths[i];
                else if (dm_up.depths[i] >= 0)
                    dm->depths[i] = dm_up.depths[i];
                else if (dm_down.depths[i] >= 0)
                    dm->depths[i] = dm_down.depths[i];
            }
        }

        dm_destroy(&dm_up);
        dm_destroy(&dm_down);
        nodeset_destroy(&seed);
        return 0;
    }

    case SEL_CLOSURE: {
        /* @node = descendants of node, then for each descendant, all ancestors */
        int idx = eval_node_lookup(g, ast->value);
        if (idx < 0) {
            if (pzErrMsg)
                *pzErrMsg = (char *)malloc(256);
            if (pzErrMsg && *pzErrMsg)
                snprintf(*pzErrMsg, 256, "graph_select: node '%s' not found", ast->value);
            return -1;
        }

        /* Step 1: get descendants */
        NodeSet seed;
        nodeset_init(&seed, g->node_count);
        nodeset_set(&seed, idx);

        NodeSet descendants;
        nodeset_init(&descendants, g->node_count);
        bfs_forward(g, &seed, -1, &descendants, NULL);

        /* Step 2: get all ancestors of all descendants */
        bfs_backward(g, &descendants, -1, ns, dm);

        /* Also include the descendants themselves */
        nodeset_union(ns, ns, &descendants);

        if (dm) {
            /* Set depth 0 for the anchor */
            dm->depths[idx] = 0;
        }

        nodeset_destroy(&seed);
        nodeset_destroy(&descendants);
        return 0;
    }

    case SEL_UNION: {
        NodeSet left_ns, right_ns;
        nodeset_init(&left_ns, g->node_count);
        nodeset_init(&right_ns, g->node_count);

        int rc = eval_ast(ast->left, g, &left_ns, NULL, pzErrMsg);
        if (rc != 0) {
            nodeset_destroy(&left_ns);
            nodeset_destroy(&right_ns);
            return rc;
        }
        rc = eval_ast(ast->right, g, &right_ns, NULL, pzErrMsg);
        if (rc != 0) {
            nodeset_destroy(&left_ns);
            nodeset_destroy(&right_ns);
            return rc;
        }

        nodeset_union(ns, &left_ns, &right_ns);
        nodeset_destroy(&left_ns);
        nodeset_destroy(&right_ns);
        return 0;
    }

    case SEL_INTERSECT: {
        NodeSet left_ns, right_ns;
        nodeset_init(&left_ns, g->node_count);
        nodeset_init(&right_ns, g->node_count);

        int rc = eval_ast(ast->left, g, &left_ns, NULL, pzErrMsg);
        if (rc != 0) {
            nodeset_destroy(&left_ns);
            nodeset_destroy(&right_ns);
            return rc;
        }
        rc = eval_ast(ast->right, g, &right_ns, NULL, pzErrMsg);
        if (rc != 0) {
            nodeset_destroy(&left_ns);
            nodeset_destroy(&right_ns);
            return rc;
        }

        nodeset_intersect(ns, &left_ns, &right_ns);
        nodeset_destroy(&left_ns);
        nodeset_destroy(&right_ns);
        return 0;
    }

    case SEL_COMPLEMENT: {
        NodeSet child_ns;
        nodeset_init(&child_ns, g->node_count);

        int rc = eval_ast(ast->left, g, &child_ns, NULL, pzErrMsg);
        if (rc != 0) {
            nodeset_destroy(&child_ns);
            return rc;
        }

        nodeset_complement(ns, &child_ns, g->node_count);
        nodeset_destroy(&child_ns);
        return 0;
    }

    default:
        if (pzErrMsg)
            *pzErrMsg = strdup("graph_select: unknown AST node type");
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Direction labeling
 *
 * For leaf AST nodes (SEL_NODE, SEL_ANCESTORS, etc.) we can label
 * directions. For set operations, direction is "selected".
 * ═══════════════════════════════════════════════════════════════ */

static const char *direction_for_type(SelectorType type) {
    switch (type) {
    case SEL_NODE:
        return "self";
    case SEL_ANCESTORS:
        return "ancestor";
    case SEL_DESCENDANTS:
        return "descendant";
    case SEL_BOTH:
        return "both";
    case SEL_CLOSURE:
        return "closure";
    default:
        return "selected";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════ */

int selector_eval(const SelectorNode *ast, const GraphData *g, SelectResults *results, char **pzErrMsg) {
    sr_init(results);

    if (g->node_count == 0)
        return 0;

    NodeSet ns;
    nodeset_init(&ns, g->node_count);

    DepthMap dm;
    dm_init(&dm, g->node_count);

    int rc = eval_ast(ast, g, &ns, &dm, pzErrMsg);
    if (rc != 0) {
        nodeset_destroy(&ns);
        dm_destroy(&dm);
        return rc;
    }

    /* Build result rows in node insertion order (deterministic) */
    const char *dir = direction_for_type(ast->type);

    for (int i = 0; i < g->node_count; i++) {
        if (nodeset_test(&ns, i)) {
            int depth = (dm.depths[i] >= 0) ? dm.depths[i] : 0;
            sr_add(results, i, depth, dir);
        }
    }

    nodeset_destroy(&ns);
    dm_destroy(&dm);
    return 0;
}
