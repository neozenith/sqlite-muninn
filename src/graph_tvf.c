/*
 * graph_tvf.c — Graph table-valued functions
 *
 * Generic graph traversal TVFs that work on any SQLite table with
 * source/target columns. Registered as eponymous-only virtual tables.
 *
 * TVFs:
 *   graph_bfs(edge_table, src_col, dst_col, start_node, max_depth, direction)
 *   graph_dfs(edge_table, src_col, dst_col, start_node, max_depth, direction)
 *   graph_shortest_path(edge_table, src_col, dst_col, start_node, end_node, weight_col)
 *
 * All table/column names are validated via id_validate() to prevent SQL injection.
 */
#include "graph_tvf.h"
#include "graph_common.h"
#include "id_validate.h"
#include "priority_queue.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

SQLITE_EXTENSION_INIT3

/* ═══════════════════════════════════════════════════════════════
 * Common helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Result row for BFS/DFS */
typedef struct {
    char *node;
    int depth;
    char *parent;  /* NULL for start node */
} TraversalRow;

/* Result array */
typedef struct {
    TraversalRow *rows;
    int count;
    int capacity;
} TraversalResults;

static int tr_init(TraversalResults *tr, int cap) {
    tr->rows = (TraversalRow *)malloc((size_t)cap * sizeof(TraversalRow));
    if (!tr->rows) return -1;
    tr->count = 0;
    tr->capacity = cap;
    return 0;
}

static int tr_push(TraversalResults *tr, const char *node, int depth, const char *parent) {
    if (tr->count >= tr->capacity) {
        int new_cap = tr->capacity * 2;
        TraversalRow *new_rows = (TraversalRow *)realloc(tr->rows,
                                                          (size_t)new_cap * sizeof(TraversalRow));
        if (!new_rows) return -1;
        tr->rows = new_rows;
        tr->capacity = new_cap;
    }
    TraversalRow *row = &tr->rows[tr->count++];
    row->node = sqlite3_mprintf("%s", node);
    row->depth = depth;
    row->parent = parent ? sqlite3_mprintf("%s", parent) : NULL;
    return 0;
}

static void tr_destroy(TraversalResults *tr) {
    for (int i = 0; i < tr->count; i++) {
        sqlite3_free(tr->rows[i].node);
        sqlite3_free(tr->rows[i].parent);
    }
    free(tr->rows);
    tr->rows = NULL;
    tr->count = 0;
}

/* Visited set using a simple hash set (open addressing) */
typedef struct {
    char **keys;
    int count;
    int capacity;
} StrHashSet;

static int shs_init(StrHashSet *s, int cap) {
    if (cap < 16) cap = 16;
    /* Round up to power of 2 */
    int c = 1;
    while (c < cap) c *= 2;
    s->keys = (char **)calloc((size_t)c, sizeof(char *));
    if (!s->keys) return -1;
    s->count = 0;
    s->capacity = c;
    return 0;
}

static void shs_destroy(StrHashSet *s) {
    for (int i = 0; i < s->capacity; i++) {
        if (s->keys[i]) sqlite3_free(s->keys[i]);
    }
    free(s->keys);
}

static int shs_contains(const StrHashSet *s, const char *key) {
    unsigned int slot = graph_str_hash(key) & (unsigned int)(s->capacity - 1);
    for (int i = 0; i < s->capacity; i++) {
        int idx = ((int)slot + i) & (s->capacity - 1);
        if (!s->keys[idx]) return 0;
        if (strcmp(s->keys[idx], key) == 0) return 1;
    }
    return 0;
}

static int shs_insert(StrHashSet *s, const char *key) {
    if (s->count * 10 > s->capacity * 7) {
        /* Resize */
        int new_cap = s->capacity * 2;
        char **new_keys = (char **)calloc((size_t)new_cap, sizeof(char *));
        if (!new_keys) return -1;
        for (int i = 0; i < s->capacity; i++) {
            if (s->keys[i]) {
                unsigned int slot = graph_str_hash(s->keys[i]) & (unsigned int)(new_cap - 1);
                for (int j = 0; j < new_cap; j++) {
                    int idx = ((int)slot + j) & (new_cap - 1);
                    if (!new_keys[idx]) { new_keys[idx] = s->keys[i]; break; }
                }
            }
        }
        free(s->keys);
        s->keys = new_keys;
        s->capacity = new_cap;
    }
    unsigned int slot = graph_str_hash(key) & (unsigned int)(s->capacity - 1);
    for (int i = 0; i < s->capacity; i++) {
        int idx = ((int)slot + i) & (s->capacity - 1);
        if (!s->keys[idx]) {
            s->keys[idx] = sqlite3_mprintf("%s", key);
            s->count++;
            return 0;
        }
        if (strcmp(s->keys[idx], key) == 0) return 0;  /* already present */
    }
    return -1;
}

/* ─── BFS Queue ────────────────────────────────────────────── */

typedef struct {
    char *node;
    int depth;
    char *parent;
} QueueItem;

typedef struct {
    QueueItem *items;
    int head, tail, count, capacity;
} Queue;

static int queue_init(Queue *q, int cap) {
    q->items = (QueueItem *)malloc((size_t)cap * sizeof(QueueItem));
    if (!q->items) return -1;
    q->head = q->tail = q->count = 0;
    q->capacity = cap;
    return 0;
}

static void queue_destroy(Queue *q) {
    /* Free any remaining items */
    while (q->count > 0) {
        QueueItem *item = &q->items[q->head];
        sqlite3_free(item->node);
        sqlite3_free(item->parent);
        q->head = (q->head + 1) % q->capacity;
        q->count--;
    }
    free(q->items);
}

static int queue_push(Queue *q, const char *node, int depth, const char *parent) {
    if (q->count >= q->capacity) {
        int new_cap = q->capacity * 2;
        QueueItem *new_items = (QueueItem *)malloc((size_t)new_cap * sizeof(QueueItem));
        if (!new_items) return -1;
        for (int i = 0; i < q->count; i++) {
            new_items[i] = q->items[(q->head + i) % q->capacity];
        }
        free(q->items);
        q->items = new_items;
        q->head = 0;
        q->tail = q->count;
        q->capacity = new_cap;
    }
    QueueItem *item = &q->items[q->tail];
    item->node = sqlite3_mprintf("%s", node);
    item->depth = depth;
    item->parent = parent ? sqlite3_mprintf("%s", parent) : NULL;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    return 0;
}

static QueueItem queue_pop(Queue *q) {
    QueueItem item = q->items[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    return item;
}

/* ═══════════════════════════════════════════════════════════════
 * BFS implementation
 * ═══════════════════════════════════════════════════════════════ */

/*
 * direction: "forward" = follow src→dst, "reverse" = follow dst→src,
 *            "both" = follow both directions
 */
static int run_bfs(sqlite3 *db,
                   const char *edge_table, const char *src_col, const char *dst_col,
                   const char *start_node, int max_depth, const char *direction,
                   TraversalResults *results) {
    /* Validate identifiers */
    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 || id_validate(dst_col) != 0) {
        return SQLITE_ERROR;
    }

    /* Verify table exists */
    char *check_sql = sqlite3_mprintf("SELECT 1 FROM \"%w\" LIMIT 0", edge_table);
    int rc = sqlite3_exec(db, check_sql, NULL, NULL, NULL);
    sqlite3_free(check_sql);
    if (rc != SQLITE_OK) return SQLITE_ERROR;

    int go_forward = (strcmp(direction, "forward") == 0 || strcmp(direction, "both") == 0);
    int go_reverse = (strcmp(direction, "reverse") == 0 || strcmp(direction, "both") == 0);

    /* Prepare queries */
    sqlite3_stmt *fwd_stmt = NULL, *rev_stmt = NULL;

    if (go_forward) {
        char *sql = sqlite3_mprintf("SELECT \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                     dst_col, edge_table, src_col);
        rc = sqlite3_prepare_v2(db, sql, -1, &fwd_stmt, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) return rc;
    }
    if (go_reverse) {
        char *sql = sqlite3_mprintf("SELECT \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                     src_col, edge_table, dst_col);
        rc = sqlite3_prepare_v2(db, sql, -1, &rev_stmt, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) { sqlite3_finalize(fwd_stmt); return rc; }
    }

    tr_init(results, 64);
    StrHashSet visited;
    shs_init(&visited, 64);
    Queue queue;
    queue_init(&queue, 64);

    /* Seed */
    shs_insert(&visited, start_node);
    queue_push(&queue, start_node, 0, NULL);

    while (queue.count > 0) {
        QueueItem item = queue_pop(&queue);

        tr_push(results, item.node, item.depth, item.parent);

        if (item.depth < max_depth) {
            /* Expand neighbors */
            sqlite3_stmt *stmts[] = { fwd_stmt, rev_stmt };
            for (int s = 0; s < 2; s++) {
                if (!stmts[s]) continue;
                sqlite3_bind_text(stmts[s], 1, item.node, -1, SQLITE_STATIC);
                while (sqlite3_step(stmts[s]) == SQLITE_ROW) {
                    const char *neighbor = (const char *)sqlite3_column_text(stmts[s], 0);
                    if (!shs_contains(&visited, neighbor)) {
                        shs_insert(&visited, neighbor);
                        queue_push(&queue, neighbor, item.depth + 1, item.node);
                    }
                }
                sqlite3_reset(stmts[s]);
            }
        }
        sqlite3_free(item.node);
        sqlite3_free(item.parent);
    }

    sqlite3_finalize(fwd_stmt);
    sqlite3_finalize(rev_stmt);
    queue_destroy(&queue);
    shs_destroy(&visited);

    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * DFS implementation
 * ═══════════════════════════════════════════════════════════════ */

/* DFS stack item */
typedef struct {
    char *node;
    int depth;
    char *parent;
} StackItem;

static int run_dfs(sqlite3 *db,
                   const char *edge_table, const char *src_col, const char *dst_col,
                   const char *start_node, int max_depth, const char *direction,
                   TraversalResults *results) {
    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 || id_validate(dst_col) != 0) {
        return SQLITE_ERROR;
    }

    char *check_sql = sqlite3_mprintf("SELECT 1 FROM \"%w\" LIMIT 0", edge_table);
    int rc = sqlite3_exec(db, check_sql, NULL, NULL, NULL);
    sqlite3_free(check_sql);
    if (rc != SQLITE_OK) return SQLITE_ERROR;

    int go_forward = (strcmp(direction, "forward") == 0 || strcmp(direction, "both") == 0);
    int go_reverse = (strcmp(direction, "reverse") == 0 || strcmp(direction, "both") == 0);

    sqlite3_stmt *fwd_stmt = NULL, *rev_stmt = NULL;

    if (go_forward) {
        char *sql = sqlite3_mprintf("SELECT \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                     dst_col, edge_table, src_col);
        rc = sqlite3_prepare_v2(db, sql, -1, &fwd_stmt, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) return rc;
    }
    if (go_reverse) {
        char *sql = sqlite3_mprintf("SELECT \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                     src_col, edge_table, dst_col);
        rc = sqlite3_prepare_v2(db, sql, -1, &rev_stmt, NULL);
        sqlite3_free(sql);
        if (rc != SQLITE_OK) { sqlite3_finalize(fwd_stmt); return rc; }
    }

    tr_init(results, 64);
    StrHashSet visited;
    shs_init(&visited, 64);

    /* Stack */
    int stack_cap = 64;
    StackItem *stack = (StackItem *)malloc((size_t)stack_cap * sizeof(StackItem));
    int stack_top = 0;

    /* Push start */
    stack[stack_top].node = sqlite3_mprintf("%s", start_node);
    stack[stack_top].depth = 0;
    stack[stack_top].parent = NULL;
    stack_top++;

    while (stack_top > 0) {
        stack_top--;
        StackItem item = stack[stack_top];

        if (shs_contains(&visited, item.node)) {
            sqlite3_free(item.node);
            sqlite3_free(item.parent);
            continue;
        }
        shs_insert(&visited, item.node);
        tr_push(results, item.node, item.depth, item.parent);

        if (item.depth < max_depth) {
            sqlite3_stmt *stmts[] = { fwd_stmt, rev_stmt };
            for (int s = 0; s < 2; s++) {
                if (!stmts[s]) continue;
                sqlite3_bind_text(stmts[s], 1, item.node, -1, SQLITE_STATIC);
                while (sqlite3_step(stmts[s]) == SQLITE_ROW) {
                    const char *neighbor = (const char *)sqlite3_column_text(stmts[s], 0);
                    if (!shs_contains(&visited, neighbor)) {
                        /* Grow stack if needed */
                        if (stack_top >= stack_cap) {
                            stack_cap *= 2;
                            stack = (StackItem *)realloc(stack, (size_t)stack_cap * sizeof(StackItem));
                        }
                        stack[stack_top].node = sqlite3_mprintf("%s", neighbor);
                        stack[stack_top].depth = item.depth + 1;
                        stack[stack_top].parent = sqlite3_mprintf("%s", item.node);
                        stack_top++;
                    }
                }
                sqlite3_reset(stmts[s]);
            }
        }
        sqlite3_free(item.node);
        sqlite3_free(item.parent);
    }

    free(stack);
    sqlite3_finalize(fwd_stmt);
    sqlite3_finalize(rev_stmt);
    shs_destroy(&visited);

    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Shortest Path (BFS unweighted, Dijkstra weighted)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *node;
    double distance;
    int path_order;
} PathRow;

typedef struct {
    PathRow *rows;
    int count;
    int capacity;
} PathResults;

static int pr_init(PathResults *pr, int cap) {
    pr->rows = (PathRow *)malloc((size_t)cap * sizeof(PathRow));
    if (!pr->rows) return -1;
    pr->count = 0;
    pr->capacity = cap;
    return 0;
}

static int pr_push(PathResults *pr, const char *node, double distance, int path_order) {
    if (pr->count >= pr->capacity) {
        int new_cap = pr->capacity * 2;
        PathRow *new_rows = (PathRow *)realloc(pr->rows, (size_t)new_cap * sizeof(PathRow));
        if (!new_rows) return -1;
        pr->rows = new_rows;
        pr->capacity = new_cap;
    }
    PathRow *row = &pr->rows[pr->count++];
    row->node = sqlite3_mprintf("%s", node);
    row->distance = distance;
    row->path_order = path_order;
    return 0;
}

static void pr_destroy(PathResults *pr) {
    for (int i = 0; i < pr->count; i++) {
        sqlite3_free(pr->rows[i].node);
    }
    free(pr->rows);
    pr->rows = NULL;
    pr->count = 0;
}

/*
 * BFS shortest path (unweighted).
 * Finds path from start to end using BFS, then traces back via parent pointers.
 */
static int run_shortest_path_bfs(sqlite3 *db,
                                  const char *edge_table, const char *src_col, const char *dst_col,
                                  const char *start_node, const char *end_node,
                                  PathResults *results) {
    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 || id_validate(dst_col) != 0) {
        return SQLITE_ERROR;
    }

    char *sql = sqlite3_mprintf("SELECT \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                 dst_col, edge_table, src_col);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    pr_init(results, 16);

    /* BFS with parent tracking */
    StrHashSet visited;
    shs_init(&visited, 64);

    /* Parent map: parallel arrays */
    int par_cap = 64;
    char **par_nodes = (char **)calloc((size_t)par_cap, sizeof(char *));
    char **par_parents = (char **)calloc((size_t)par_cap, sizeof(char *));
    int par_count = 0;

    Queue queue;
    queue_init(&queue, 64);

    shs_insert(&visited, start_node);
    queue_push(&queue, start_node, 0, NULL);
    par_nodes[par_count] = sqlite3_mprintf("%s", start_node);
    par_parents[par_count] = NULL;
    par_count++;

    int found = 0;

    while (queue.count > 0 && !found) {
        QueueItem item = queue_pop(&queue);

        if (strcmp(item.node, end_node) == 0) {
            found = 1;
            sqlite3_free(item.node);
            sqlite3_free(item.parent);
            break;
        }

        sqlite3_bind_text(stmt, 1, item.node, -1, SQLITE_STATIC);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *neighbor = (const char *)sqlite3_column_text(stmt, 0);
            if (!shs_contains(&visited, neighbor)) {
                shs_insert(&visited, neighbor);
                queue_push(&queue, neighbor, item.depth + 1, item.node);

                /* Store parent */
                if (par_count >= par_cap) {
                    par_cap *= 2;
                    par_nodes = (char **)realloc(par_nodes, (size_t)par_cap * sizeof(char *));
                    par_parents = (char **)realloc(par_parents, (size_t)par_cap * sizeof(char *));
                }
                par_nodes[par_count] = sqlite3_mprintf("%s", neighbor);
                par_parents[par_count] = sqlite3_mprintf("%s", item.node);
                par_count++;
            }
        }
        sqlite3_reset(stmt);
        sqlite3_free(item.node);
        sqlite3_free(item.parent);
    }

    if (found) {
        /* Trace back from end to start */
        int path_cap = 32;
        char **path = (char **)malloc((size_t)path_cap * sizeof(char *));
        int path_len = 0;

        const char *cur = end_node;
        while (cur) {
            if (path_len >= path_cap) {
                path_cap *= 2;
                path = (char **)realloc(path, (size_t)path_cap * sizeof(char *));
            }
            path[path_len++] = sqlite3_mprintf("%s", cur);

            /* Find parent */
            const char *parent = NULL;
            for (int i = 0; i < par_count; i++) {
                if (strcmp(par_nodes[i], cur) == 0) {
                    parent = par_parents[i];
                    break;
                }
            }
            cur = parent;
        }

        /* Reverse and push to results */
        for (int i = path_len - 1; i >= 0; i--) {
            pr_push(results, path[i], (double)(path_len - 1 - i), path_len - 1 - i);
            sqlite3_free(path[i]);
        }
        free(path);
    }

    /* Cleanup */
    for (int i = 0; i < par_count; i++) {
        sqlite3_free(par_nodes[i]);
        sqlite3_free(par_parents[i]);
    }
    free(par_nodes);
    free(par_parents);
    queue_destroy(&queue);
    shs_destroy(&visited);
    sqlite3_finalize(stmt);

    return SQLITE_OK;
}

/*
 * Dijkstra shortest path (weighted).
 * Uses a priority queue with string node IDs.
 */

/* Dijkstra node entry for the "dist" table */
typedef struct {
    char *node;
    double dist;
    char *parent;
} DijkNode;

static int run_shortest_path_dijkstra(sqlite3 *db,
                                       const char *edge_table, const char *src_col, const char *dst_col,
                                       const char *weight_col,
                                       const char *start_node, const char *end_node,
                                       PathResults *results) {
    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 ||
        id_validate(dst_col) != 0 || id_validate(weight_col) != 0) {
        return SQLITE_ERROR;
    }

    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\" WHERE \"%w\" = ?",
                                 dst_col, weight_col, edge_table, src_col);
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    pr_init(results, 16);

    /* Settled set and distance map */
    int dijk_cap = 64;
    DijkNode *dijk_nodes = (DijkNode *)calloc((size_t)dijk_cap, sizeof(DijkNode));
    int dijk_count = 0;
    StrHashSet settled;
    shs_init(&settled, 64);

    /* Priority queue: uses integer IDs mapped to node strings */
    /* We'll use a simple array-based PQ with string keys */
    typedef struct { char *node; double dist; char *parent; } PQEntry;
    int pq_cap = 64;
    PQEntry *pq = (PQEntry *)malloc((size_t)pq_cap * sizeof(PQEntry));
    int pq_size = 0;

    /* Seed */
    pq[pq_size].node = sqlite3_mprintf("%s", start_node);
    pq[pq_size].dist = 0.0;
    pq[pq_size].parent = NULL;
    pq_size++;

    int found = 0;

    while (pq_size > 0 && !found) {
        /* Find minimum (simple linear scan — fine for graph TVFs) */
        int min_idx = 0;
        for (int i = 1; i < pq_size; i++) {
            if (pq[i].dist < pq[min_idx].dist) min_idx = i;
        }
        PQEntry cur = pq[min_idx];
        pq[min_idx] = pq[pq_size - 1];
        pq_size--;

        if (shs_contains(&settled, cur.node)) {
            sqlite3_free(cur.node);
            sqlite3_free(cur.parent);
            continue;
        }
        shs_insert(&settled, cur.node);

        /* Record in settled nodes */
        if (dijk_count >= dijk_cap) {
            dijk_cap *= 2;
            dijk_nodes = (DijkNode *)realloc(dijk_nodes, (size_t)dijk_cap * sizeof(DijkNode));
        }
        dijk_nodes[dijk_count].node = sqlite3_mprintf("%s", cur.node);
        dijk_nodes[dijk_count].dist = cur.dist;
        dijk_nodes[dijk_count].parent = cur.parent ? sqlite3_mprintf("%s", cur.parent) : NULL;
        dijk_count++;

        if (strcmp(cur.node, end_node) == 0) {
            found = 1;
            sqlite3_free(cur.node);
            sqlite3_free(cur.parent);
            break;
        }

        /* Expand neighbors */
        sqlite3_bind_text(stmt, 1, cur.node, -1, SQLITE_STATIC);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *neighbor = (const char *)sqlite3_column_text(stmt, 0);
            double weight = sqlite3_column_double(stmt, 1);
            double new_dist = cur.dist + weight;

            if (!shs_contains(&settled, neighbor)) {
                if (pq_size >= pq_cap) {
                    pq_cap *= 2;
                    pq = (PQEntry *)realloc(pq, (size_t)pq_cap * sizeof(PQEntry));
                }
                pq[pq_size].node = sqlite3_mprintf("%s", neighbor);
                pq[pq_size].dist = new_dist;
                pq[pq_size].parent = sqlite3_mprintf("%s", cur.node);
                pq_size++;
            }
        }
        sqlite3_reset(stmt);
        sqlite3_free(cur.node);
        sqlite3_free(cur.parent);
    }

    if (found) {
        /* Trace back */
        int path_cap = 32;
        char **path = (char **)malloc((size_t)path_cap * sizeof(char *));
        double *dists = (double *)malloc((size_t)path_cap * sizeof(double));
        int path_len = 0;

        const char *cur = end_node;
        while (cur) {
            if (path_len >= path_cap) {
                path_cap *= 2;
                path = (char **)realloc(path, (size_t)path_cap * sizeof(char *));
                dists = (double *)realloc(dists, (size_t)path_cap * sizeof(double));
            }
            path[path_len] = sqlite3_mprintf("%s", cur);
            /* Find distance */
            for (int i = 0; i < dijk_count; i++) {
                if (strcmp(dijk_nodes[i].node, cur) == 0) {
                    dists[path_len] = dijk_nodes[i].dist;
                    cur = dijk_nodes[i].parent;
                    break;
                }
            }
            path_len++;
        }

        /* Reverse and push */
        for (int i = path_len - 1; i >= 0; i--) {
            pr_push(results, path[i], dists[i], path_len - 1 - i);
            sqlite3_free(path[i]);
        }
        free(path);
        free(dists);
    }

    /* Cleanup remaining PQ entries */
    for (int i = 0; i < pq_size; i++) {
        sqlite3_free(pq[i].node);
        sqlite3_free(pq[i].parent);
    }
    free(pq);

    for (int i = 0; i < dijk_count; i++) {
        sqlite3_free(dijk_nodes[i].node);
        sqlite3_free(dijk_nodes[i].parent);
    }
    free(dijk_nodes);
    shs_destroy(&settled);
    sqlite3_finalize(stmt);

    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table: graph_bfs / graph_dfs
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    int is_dfs;  /* 0 = BFS, 1 = DFS */
} GraphTraversalVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    TraversalResults results;
    int current;
    int eof;
} GraphTraversalCursor;

/* Output columns: node, depth, parent */
/* Hidden input columns: edge_table, src_col, dst_col, start_node, max_depth, direction */
#define GTRAV_COL_NODE       0
#define GTRAV_COL_DEPTH      1
#define GTRAV_COL_PARENT     2
#define GTRAV_COL_EDGE_TABLE 3
#define GTRAV_COL_SRC_COL    4
#define GTRAV_COL_DST_COL    5
#define GTRAV_COL_START_NODE 6
#define GTRAV_COL_MAX_DEPTH  7
#define GTRAV_COL_DIRECTION  8

/* No gtrav_create: xCreate=NULL makes graph_bfs/graph_dfs eponymous-only */

static int gtrav_connect(sqlite3 *db, void *pAux, int argc,
                          const char *const *argv, sqlite3_vtab **ppVTab, char **pzErr) {
    (void)argc; (void)argv; (void)pzErr;

    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT,"
        "  depth INTEGER,"
        "  parent TEXT,"
        "  edge_table TEXT HIDDEN,"
        "  src_col TEXT HIDDEN,"
        "  dst_col TEXT HIDDEN,"
        "  start_node TEXT HIDDEN,"
        "  max_depth INTEGER HIDDEN,"
        "  direction TEXT HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    GraphTraversalVtab *vtab = (GraphTraversalVtab *)sqlite3_malloc(sizeof(GraphTraversalVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(GraphTraversalVtab));
    vtab->db = db;
    vtab->is_dfs = (pAux != NULL);  /* pAux is (void*)1 for DFS */

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int gtrav_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

static int gtrav_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    int argv_idx = 1;
    /* We need all hidden columns to be provided */
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        int col = pInfo->aConstraint[i].iColumn;
        if (col >= GTRAV_COL_EDGE_TABLE && col <= GTRAV_COL_DIRECTION &&
            pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
            pInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pInfo->aConstraintUsage[i].omit = 1;
        }
    }
    pInfo->estimatedCost = 1000.0;
    return SQLITE_OK;
}

static int gtrav_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    GraphTraversalCursor *cur = (GraphTraversalCursor *)sqlite3_malloc(sizeof(GraphTraversalCursor));
    if (!cur) return SQLITE_NOMEM;
    memset(cur, 0, sizeof(GraphTraversalCursor));
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int gtrav_close(sqlite3_vtab_cursor *pCursor) {
    GraphTraversalCursor *cur = (GraphTraversalCursor *)pCursor;
    tr_destroy(&cur->results);
    sqlite3_free(cur);
    return SQLITE_OK;
}

static int gtrav_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                         const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxNum; (void)idxStr;
    GraphTraversalCursor *cur = (GraphTraversalCursor *)pCursor;
    GraphTraversalVtab *vtab = (GraphTraversalVtab *)pCursor->pVtab;

    tr_destroy(&cur->results);
    cur->current = 0;
    cur->eof = 1;

    /* Extract parameters from argv (ordered by column index due to xBestIndex) */
    /* We need to handle the fact that constraints may not be in column order */
    const char *edge_table = NULL, *src_col = NULL, *dst_col = NULL;
    const char *start_node = NULL, *direction = "forward";
    int max_depth = 100;

    /* Since xBestIndex assigns argvIndex in constraint order (which is column order
     * for hidden columns with EQ), we can map positionally */
    if (argc >= 1) edge_table = (const char *)sqlite3_value_text(argv[0]);
    if (argc >= 2) src_col = (const char *)sqlite3_value_text(argv[1]);
    if (argc >= 3) dst_col = (const char *)sqlite3_value_text(argv[2]);
    if (argc >= 4) start_node = (const char *)sqlite3_value_text(argv[3]);
    if (argc >= 5) max_depth = sqlite3_value_int(argv[4]);
    if (argc >= 6) direction = (const char *)sqlite3_value_text(argv[5]);

    if (!edge_table || !src_col || !dst_col || !start_node) {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_%s: requires edge_table, src_col, dst_col, start_node parameters",
            vtab->is_dfs ? "dfs" : "bfs");
        return SQLITE_ERROR;
    }

    int rc;
    if (vtab->is_dfs) {
        rc = run_dfs(vtab->db, edge_table, src_col, dst_col,
                     start_node, max_depth, direction, &cur->results);
    } else {
        rc = run_bfs(vtab->db, edge_table, src_col, dst_col,
                     start_node, max_depth, direction, &cur->results);
    }

    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_%s: traversal failed (check table/column names exist)",
            vtab->is_dfs ? "dfs" : "bfs");
        return SQLITE_ERROR;
    }

    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int gtrav_next(sqlite3_vtab_cursor *pCursor) {
    GraphTraversalCursor *cur = (GraphTraversalCursor *)pCursor;
    cur->current++;
    if (cur->current >= cur->results.count) cur->eof = 1;
    return SQLITE_OK;
}

static int gtrav_eof(sqlite3_vtab_cursor *pCursor) {
    return ((GraphTraversalCursor *)pCursor)->eof;
}

static int gtrav_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    GraphTraversalCursor *cur = (GraphTraversalCursor *)pCursor;
    TraversalRow *row = &cur->results.rows[cur->current];

    switch (col) {
        case GTRAV_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT);
            break;
        case GTRAV_COL_DEPTH:
            sqlite3_result_int(ctx, row->depth);
            break;
        case GTRAV_COL_PARENT:
            if (row->parent)
                sqlite3_result_text(ctx, row->parent, -1, SQLITE_TRANSIENT);
            else
                sqlite3_result_null(ctx);
            break;
        default:
            sqlite3_result_null(ctx);
            break;
    }
    return SQLITE_OK;
}

static int gtrav_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((GraphTraversalCursor *)pCursor)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_bfs_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only: no explicit CREATE VIRTUAL TABLE */
    .xConnect = gtrav_connect,
    .xBestIndex = gtrav_best_index,
    .xDisconnect = gtrav_disconnect,
    .xDestroy = gtrav_disconnect,
    .xOpen = gtrav_open,
    .xClose = gtrav_close,
    .xFilter = gtrav_filter,
    .xNext = gtrav_next,
    .xEof = gtrav_eof,
    .xColumn = gtrav_column,
    .xRowid = gtrav_rowid,
};

static sqlite3_module graph_dfs_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only */
    .xConnect = gtrav_connect,
    .xBestIndex = gtrav_best_index,
    .xDisconnect = gtrav_disconnect,
    .xDestroy = gtrav_disconnect,
    .xOpen = gtrav_open,
    .xClose = gtrav_close,
    .xFilter = gtrav_filter,
    .xNext = gtrav_next,
    .xEof = gtrav_eof,
    .xColumn = gtrav_column,
    .xRowid = gtrav_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table: graph_shortest_path
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} GraphSPVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    PathResults results;
    int current;
    int eof;
} GraphSPCursor;

#define GSP_COL_NODE       0
#define GSP_COL_DISTANCE   1
#define GSP_COL_PATH_ORDER 2
#define GSP_COL_EDGE_TABLE 3
#define GSP_COL_SRC_COL    4
#define GSP_COL_DST_COL    5
#define GSP_COL_START_NODE 6
#define GSP_COL_END_NODE   7
#define GSP_COL_WEIGHT_COL 8

/* No gsp_create: xCreate=NULL makes graph_shortest_path eponymous-only */

static int gsp_connect(sqlite3 *db, void *pAux, int argc,
                        const char *const *argv, sqlite3_vtab **ppVTab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;

    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT,"
        "  distance REAL,"
        "  path_order INTEGER,"
        "  edge_table TEXT HIDDEN,"
        "  src_col TEXT HIDDEN,"
        "  dst_col TEXT HIDDEN,"
        "  start_node TEXT HIDDEN,"
        "  end_node TEXT HIDDEN,"
        "  weight_col TEXT HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    GraphSPVtab *vtab = (GraphSPVtab *)sqlite3_malloc(sizeof(GraphSPVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(GraphSPVtab));
    vtab->db = db;

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int gsp_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    int argv_idx = 1;
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        int col = pInfo->aConstraint[i].iColumn;
        if (col >= GSP_COL_EDGE_TABLE && col <= GSP_COL_WEIGHT_COL &&
            pInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
            pInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pInfo->aConstraintUsage[i].omit = 1;
        }
    }
    pInfo->estimatedCost = 1000.0;
    return SQLITE_OK;
}

static int gsp_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    GraphSPCursor *cur = (GraphSPCursor *)sqlite3_malloc(sizeof(GraphSPCursor));
    if (!cur) return SQLITE_NOMEM;
    memset(cur, 0, sizeof(GraphSPCursor));
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int gsp_close(sqlite3_vtab_cursor *pCursor) {
    GraphSPCursor *cur = (GraphSPCursor *)pCursor;
    pr_destroy(&cur->results);
    sqlite3_free(cur);
    return SQLITE_OK;
}

static int gsp_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                       const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxNum; (void)idxStr;
    GraphSPCursor *cur = (GraphSPCursor *)pCursor;
    GraphSPVtab *vtab = (GraphSPVtab *)pCursor->pVtab;

    pr_destroy(&cur->results);
    cur->current = 0;
    cur->eof = 1;

    const char *edge_table = NULL, *src_col = NULL, *dst_col = NULL;
    const char *start_node = NULL, *end_node = NULL, *weight_col = NULL;

    if (argc >= 1) edge_table = (const char *)sqlite3_value_text(argv[0]);
    if (argc >= 2) src_col = (const char *)sqlite3_value_text(argv[1]);
    if (argc >= 3) dst_col = (const char *)sqlite3_value_text(argv[2]);
    if (argc >= 4) start_node = (const char *)sqlite3_value_text(argv[3]);
    if (argc >= 5) end_node = (const char *)sqlite3_value_text(argv[4]);
    if (argc >= 6 && sqlite3_value_type(argv[5]) != SQLITE_NULL)
        weight_col = (const char *)sqlite3_value_text(argv[5]);

    if (!edge_table || !src_col || !dst_col || !start_node || !end_node) {
        vtab->base.zErrMsg = sqlite3_mprintf(
            "graph_shortest_path: requires edge_table, src_col, dst_col, start_node, end_node");
        return SQLITE_ERROR;
    }

    int rc;
    if (weight_col) {
        rc = run_shortest_path_dijkstra(vtab->db, edge_table, src_col, dst_col,
                                         weight_col, start_node, end_node, &cur->results);
    } else {
        rc = run_shortest_path_bfs(vtab->db, edge_table, src_col, dst_col,
                                    start_node, end_node, &cur->results);
    }

    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_shortest_path: search failed");
        return SQLITE_ERROR;
    }

    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int gsp_next(sqlite3_vtab_cursor *pCursor) {
    GraphSPCursor *cur = (GraphSPCursor *)pCursor;
    cur->current++;
    if (cur->current >= cur->results.count) cur->eof = 1;
    return SQLITE_OK;
}

static int gsp_eof(sqlite3_vtab_cursor *pCursor) {
    return ((GraphSPCursor *)pCursor)->eof;
}

static int gsp_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    GraphSPCursor *cur = (GraphSPCursor *)pCursor;
    PathRow *row = &cur->results.rows[cur->current];

    switch (col) {
        case GSP_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT);
            break;
        case GSP_COL_DISTANCE:
            sqlite3_result_double(ctx, row->distance);
            break;
        case GSP_COL_PATH_ORDER:
            sqlite3_result_int(ctx, row->path_order);
            break;
        default:
            sqlite3_result_null(ctx);
            break;
    }
    return SQLITE_OK;
}

static int gsp_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((GraphSPCursor *)pCursor)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_sp_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only */
    .xConnect = gsp_connect,
    .xBestIndex = gsp_best_index,
    .xDisconnect = gtrav_disconnect,
    .xDestroy = gtrav_disconnect,
    .xOpen = gsp_open,
    .xClose = gsp_close,
    .xFilter = gsp_filter,
    .xNext = gsp_next,
    .xEof = gsp_eof,
    .xColumn = gsp_column,
    .xRowid = gsp_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Shared base vtab for graph_components and graph_pagerank
 * (just needs sqlite3_vtab base + db handle)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} GraphVtab;

static int graph_vtab_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * graph_components — Connected components via Union-Find
 *
 * Output: node TEXT, component_id INTEGER, component_size INTEGER
 * Hidden: edge_table, src_col, dst_col
 *
 * Uses Union-Find with path compression + union by rank.
 * ═══════════════════════════════════════════════════════════════ */

/* Union-Find data structure over string node IDs */
typedef struct {
    char **ids;      /* node ID strings, owned */
    int *parent;     /* parent index (self = root) */
    int *rank;       /* union by rank */
    int count;
    int capacity;
} UnionFind;

static void uf_init(UnionFind *uf) {
    uf->capacity = 256;
    uf->count = 0;
    uf->ids = (char **)calloc((size_t)uf->capacity, sizeof(char *));
    uf->parent = (int *)malloc((size_t)uf->capacity * sizeof(int));
    uf->rank = (int *)calloc((size_t)uf->capacity, sizeof(int));
}

static void uf_destroy(UnionFind *uf) {
    for (int i = 0; i < uf->count; i++) free(uf->ids[i]);
    free(uf->ids);
    free(uf->parent);
    free(uf->rank);
}

static int uf_find_or_add(UnionFind *uf, const char *id) {
    for (int i = 0; i < uf->count; i++) {
        if (strcmp(uf->ids[i], id) == 0) return i;
    }
    if (uf->count >= uf->capacity) {
        int new_cap = uf->capacity * 2;
        uf->ids = (char **)realloc(uf->ids, (size_t)new_cap * sizeof(char *));
        uf->parent = (int *)realloc(uf->parent, (size_t)new_cap * sizeof(int));
        uf->rank = (int *)realloc(uf->rank, (size_t)new_cap * sizeof(int));
        for (int i = uf->capacity; i < new_cap; i++) {
            uf->ids[i] = NULL;
            uf->rank[i] = 0;
        }
        uf->capacity = new_cap;
    }
    int idx = uf->count++;
    uf->ids[idx] = strdup(id);
    uf->parent[idx] = idx;  /* self-parent = root */
    uf->rank[idx] = 0;
    return idx;
}

static int uf_find(UnionFind *uf, int x) {
    /* Path compression */
    while (uf->parent[x] != x) {
        uf->parent[x] = uf->parent[uf->parent[x]];  /* halving */
        x = uf->parent[x];
    }
    return x;
}

static void uf_union(UnionFind *uf, int a, int b) {
    int ra = uf_find(uf, a);
    int rb = uf_find(uf, b);
    if (ra == rb) return;
    /* Union by rank */
    if (uf->rank[ra] < uf->rank[rb]) { int tmp = ra; ra = rb; rb = tmp; }
    uf->parent[rb] = ra;
    if (uf->rank[ra] == uf->rank[rb]) uf->rank[ra]++;
}

typedef struct {
    char *node;
    int component_id;
    int component_size;
} ComponentRow;

typedef struct {
    ComponentRow *rows;
    int count;
    int capacity;
} ComponentResults;

static void comp_results_init(ComponentResults *r) {
    r->count = 0;
    r->capacity = 64;
    r->rows = (ComponentRow *)calloc((size_t)r->capacity, sizeof(ComponentRow));
}

static void comp_results_destroy(ComponentResults *r) {
    for (int i = 0; i < r->count; i++) free(r->rows[i].node);
    free(r->rows);
}

static void comp_results_add(ComponentResults *r, const char *node,
                              int comp_id, int comp_size) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (ComponentRow *)realloc(r->rows,
                    (size_t)r->capacity * sizeof(ComponentRow));
    }
    r->rows[r->count].node = strdup(node);
    r->rows[r->count].component_id = comp_id;
    r->rows[r->count].component_size = comp_size;
    r->count++;
}

static int run_components(sqlite3 *db, const char *edge_table,
                           const char *src_col, const char *dst_col,
                           ComponentResults *results) {
    comp_results_init(results);
    UnionFind uf;
    uf_init(&uf);

    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"",
                                 src_col, dst_col, edge_table);
    if (!sql) { uf_destroy(&uf); return SQLITE_NOMEM; }

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) { uf_destroy(&uf); return rc; }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *src = (const char *)sqlite3_column_text(stmt, 0);
        const char *dst = (const char *)sqlite3_column_text(stmt, 1);
        if (!src || !dst) continue;
        int si = uf_find_or_add(&uf, src);
        int di = uf_find_or_add(&uf, dst);
        uf_union(&uf, si, di);
    }
    sqlite3_finalize(stmt);

    /* Count component sizes */
    int *comp_size = (int *)calloc((size_t)uf.count, sizeof(int));
    for (int i = 0; i < uf.count; i++) {
        comp_size[uf_find(&uf, i)]++;
    }

    /* Build results */
    for (int i = 0; i < uf.count; i++) {
        int root = uf_find(&uf, i);
        comp_results_add(results, uf.ids[i], root, comp_size[root]);
    }

    free(comp_size);
    uf_destroy(&uf);
    return SQLITE_OK;
}

/* Column indices for graph_components */
enum {
    GC_COL_NODE = 0,
    GC_COL_COMPONENT_ID,
    GC_COL_COMPONENT_SIZE,
    GC_COL_EDGE_TABLE,    /* hidden */
    GC_COL_SRC_COL,       /* hidden */
    GC_COL_DST_COL,       /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    ComponentResults results;
    int current;
    int eof;
} GraphCompCursor;

static int gc_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                       sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x(node TEXT, component_id INTEGER, component_size INTEGER,"
        " edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN)");
    if (rc != SQLITE_OK) return rc;

    GraphVtab *vtab = (GraphVtab *)sqlite3_malloc(sizeof(GraphVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(GraphVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int gc_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    int arg_mask = 0;
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
        if (!pIdxInfo->aConstraint[i].usable) continue;
        if (pIdxInfo->aConstraint[i].op != SQLITE_INDEX_CONSTRAINT_EQ) continue;
        int col = pIdxInfo->aConstraint[i].iColumn;
        if (col >= GC_COL_EDGE_TABLE && col <= GC_COL_DST_COL) {
            int arg_idx = col - GC_COL_EDGE_TABLE;
            pIdxInfo->aConstraintUsage[i].argvIndex = arg_idx + 1;
            pIdxInfo->aConstraintUsage[i].omit = 1;
            arg_mask |= (1 << arg_idx);
        }
    }
    if (arg_mask == 0x7) {
        pIdxInfo->estimatedCost = 1000.0;
    } else {
        pIdxInfo->estimatedCost = 1e12;
    }
    return SQLITE_OK;
}

static int gc_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    GraphCompCursor *cur = (GraphCompCursor *)calloc(1, sizeof(GraphCompCursor));
    if (!cur) return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int gc_close(sqlite3_vtab_cursor *pCursor) {
    GraphCompCursor *cur = (GraphCompCursor *)pCursor;
    comp_results_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int gc_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxNum; (void)idxStr;
    GraphCompCursor *cur = (GraphCompCursor *)pCursor;
    comp_results_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(ComponentResults));

    if (argc < 3) {
        cur->eof = 1;
        return SQLITE_OK;
    }

    const char *edge_table = (const char *)sqlite3_value_text(argv[0]);
    const char *src_col    = (const char *)sqlite3_value_text(argv[1]);
    const char *dst_col    = (const char *)sqlite3_value_text(argv[2]);

    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 ||
        id_validate(dst_col) != 0) {
        pCursor->pVtab->zErrMsg = sqlite3_mprintf(
            "graph_components: invalid table/column identifier");
        return SQLITE_ERROR;
    }

    GraphVtab *vtab = (GraphVtab *)pCursor->pVtab;

    int rc = run_components(vtab->db, edge_table, src_col, dst_col, &cur->results);
    if (rc != SQLITE_OK) {
        cur->eof = 1;
        return rc;
    }

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int gc_next(sqlite3_vtab_cursor *pCursor) {
    GraphCompCursor *cur = (GraphCompCursor *)pCursor;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int gc_eof(sqlite3_vtab_cursor *pCursor) {
    return ((GraphCompCursor *)pCursor)->eof;
}

static int gc_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    GraphCompCursor *cur = (GraphCompCursor *)pCursor;
    ComponentRow *row = &cur->results.rows[cur->current];
    switch (col) {
        case GC_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT);
            break;
        case GC_COL_COMPONENT_ID:
            sqlite3_result_int(ctx, row->component_id);
            break;
        case GC_COL_COMPONENT_SIZE:
            sqlite3_result_int(ctx, row->component_size);
            break;
        default:
            sqlite3_result_null(ctx);
            break;
    }
    return SQLITE_OK;
}

static int gc_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((GraphCompCursor *)pCursor)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_comp_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only */
    .xConnect = gc_connect,
    .xBestIndex = gc_best_index,
    .xDisconnect = graph_vtab_disconnect,
    .xDestroy = graph_vtab_disconnect,
    .xOpen = gc_open,
    .xClose = gc_close,
    .xFilter = gc_filter,
    .xNext = gc_next,
    .xEof = gc_eof,
    .xColumn = gc_column,
    .xRowid = gc_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * graph_pagerank — Iterative power method PageRank
 *
 * Output: node TEXT, rank REAL
 * Hidden: edge_table, src_col, dst_col, damping, iterations
 *
 * Computes PageRank via iterative updates:
 *   PR(v) = (1-d)/N + d * SUM(PR(u)/out_degree(u)) for all u→v
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *node;
    double rank;
} PRRow;

typedef struct {
    PRRow *rows;
    int count;
    int capacity;
} PRResults;

static void pr_results_init(PRResults *r) {
    r->count = 0;
    r->capacity = 64;
    r->rows = (PRRow *)calloc((size_t)r->capacity, sizeof(PRRow));
}

static void pr_results_destroy(PRResults *r) {
    for (int i = 0; i < r->count; i++) free(r->rows[i].node);
    free(r->rows);
}

static void pr_results_add(PRResults *r, const char *node, double rank) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (PRRow *)realloc(r->rows, (size_t)r->capacity * sizeof(PRRow));
    }
    r->rows[r->count].node = strdup(node);
    r->rows[r->count].rank = rank;
    r->count++;
}

/* Simple adjacency list for PageRank */
typedef struct {
    char **ids;
    int **out_edges;   /* out_edges[i] = array of target indices */
    int *out_count;
    int *out_cap;
    int count;
    int capacity;
} PRAdjList;

static void pr_adj_init(PRAdjList *a) {
    a->capacity = 256;
    a->count = 0;
    a->ids = (char **)calloc((size_t)a->capacity, sizeof(char *));
    a->out_edges = (int **)calloc((size_t)a->capacity, sizeof(int *));
    a->out_count = (int *)calloc((size_t)a->capacity, sizeof(int));
    a->out_cap = (int *)calloc((size_t)a->capacity, sizeof(int));
}

static void pr_adj_destroy(PRAdjList *a) {
    for (int i = 0; i < a->count; i++) {
        free(a->ids[i]);
        free(a->out_edges[i]);
    }
    free(a->ids);
    free(a->out_edges);
    free(a->out_count);
    free(a->out_cap);
}

static int pr_adj_find_or_add(PRAdjList *a, const char *id) {
    for (int i = 0; i < a->count; i++) {
        if (strcmp(a->ids[i], id) == 0) return i;
    }
    if (a->count >= a->capacity) {
        int new_cap = a->capacity * 2;
        a->ids = (char **)realloc(a->ids, (size_t)new_cap * sizeof(char *));
        a->out_edges = (int **)realloc(a->out_edges, (size_t)new_cap * sizeof(int *));
        a->out_count = (int *)realloc(a->out_count, (size_t)new_cap * sizeof(int));
        a->out_cap = (int *)realloc(a->out_cap, (size_t)new_cap * sizeof(int));
        for (int i = a->capacity; i < new_cap; i++) {
            a->ids[i] = NULL;
            a->out_edges[i] = NULL;
            a->out_count[i] = 0;
            a->out_cap[i] = 0;
        }
        a->capacity = new_cap;
    }
    int idx = a->count++;
    a->ids[idx] = strdup(id);
    return idx;
}

static void pr_adj_add_edge(PRAdjList *a, int src, int dst) {
    if (a->out_count[src] >= a->out_cap[src]) {
        int nc = a->out_cap[src] == 0 ? 8 : a->out_cap[src] * 2;
        a->out_edges[src] = (int *)realloc(a->out_edges[src], (size_t)nc * sizeof(int));
        a->out_cap[src] = nc;
    }
    a->out_edges[src][a->out_count[src]++] = dst;
}

static int run_pagerank(sqlite3 *db, const char *edge_table,
                         const char *src_col, const char *dst_col,
                         double damping, int iterations,
                         PRResults *results) {
    pr_results_init(results);
    PRAdjList adj;
    pr_adj_init(&adj);

    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"",
                                 src_col, dst_col, edge_table);
    if (!sql) { pr_adj_destroy(&adj); return SQLITE_NOMEM; }

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) { pr_adj_destroy(&adj); return rc; }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *src = (const char *)sqlite3_column_text(stmt, 0);
        const char *dst = (const char *)sqlite3_column_text(stmt, 1);
        if (!src || !dst) continue;
        int si = pr_adj_find_or_add(&adj, src);
        int di = pr_adj_find_or_add(&adj, dst);
        pr_adj_add_edge(&adj, si, di);
    }
    sqlite3_finalize(stmt);

    int N = adj.count;
    if (N == 0) { pr_adj_destroy(&adj); return SQLITE_OK; }

    /* Initialize ranks to 1/N */
    double *rank_cur = (double *)malloc((size_t)N * sizeof(double));
    double *rank_new = (double *)malloc((size_t)N * sizeof(double));
    if (!rank_cur || !rank_new) {
        free(rank_cur); free(rank_new);
        pr_adj_destroy(&adj);
        return SQLITE_NOMEM;
    }

    double init_rank = 1.0 / N;
    for (int i = 0; i < N; i++) rank_cur[i] = init_rank;

    /* Iterative power method */
    double teleport = (1.0 - damping) / N;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N; i++) rank_new[i] = teleport;

        for (int i = 0; i < N; i++) {
            if (adj.out_count[i] == 0) {
                /* Dangling node: distribute rank evenly */
                double share = damping * rank_cur[i] / N;
                for (int j = 0; j < N; j++) rank_new[j] += share;
            } else {
                double share = damping * rank_cur[i] / adj.out_count[i];
                for (int e = 0; e < adj.out_count[i]; e++) {
                    rank_new[adj.out_edges[i][e]] += share;
                }
            }
        }

        /* Swap */
        double *tmp = rank_cur;
        rank_cur = rank_new;
        rank_new = tmp;
    }

    /* Build results */
    for (int i = 0; i < N; i++) {
        pr_results_add(results, adj.ids[i], rank_cur[i]);
    }

    free(rank_cur);
    free(rank_new);
    pr_adj_destroy(&adj);
    return SQLITE_OK;
}

/* Column indices for graph_pagerank */
enum {
    GPR_COL_NODE = 0,
    GPR_COL_RANK,
    GPR_COL_EDGE_TABLE,   /* hidden */
    GPR_COL_SRC_COL,      /* hidden */
    GPR_COL_DST_COL,      /* hidden */
    GPR_COL_DAMPING,      /* hidden */
    GPR_COL_ITERATIONS,   /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    PRResults results;
    int current;
    int eof;
} GraphPRCursor;

static int gpr_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                         sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x(node TEXT, rank REAL,"
        " edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
        " damping REAL HIDDEN, iterations INTEGER HIDDEN)");
    if (rc != SQLITE_OK) return rc;

    GraphVtab *vtab = (GraphVtab *)sqlite3_malloc(sizeof(GraphVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(GraphVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int gpr_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    int arg_mask = 0;
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
        if (!pIdxInfo->aConstraint[i].usable) continue;
        if (pIdxInfo->aConstraint[i].op != SQLITE_INDEX_CONSTRAINT_EQ) continue;
        int col = pIdxInfo->aConstraint[i].iColumn;
        if (col >= GPR_COL_EDGE_TABLE && col <= GPR_COL_ITERATIONS) {
            int arg_idx = col - GPR_COL_EDGE_TABLE;
            pIdxInfo->aConstraintUsage[i].argvIndex = arg_idx + 1;
            pIdxInfo->aConstraintUsage[i].omit = 1;
            arg_mask |= (1 << arg_idx);
        }
    }
    if ((arg_mask & 0x7) == 0x7) {  /* at least edge_table, src, dst */
        pIdxInfo->estimatedCost = 1000.0;
    } else {
        pIdxInfo->estimatedCost = 1e12;
    }
    return SQLITE_OK;
}

static int gpr_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    GraphPRCursor *cur = (GraphPRCursor *)calloc(1, sizeof(GraphPRCursor));
    if (!cur) return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int gpr_close(sqlite3_vtab_cursor *pCursor) {
    GraphPRCursor *cur = (GraphPRCursor *)pCursor;
    pr_results_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int gpr_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                        const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxNum; (void)idxStr;
    GraphPRCursor *cur = (GraphPRCursor *)pCursor;
    pr_results_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(PRResults));

    if (argc < 3) {
        cur->eof = 1;
        return SQLITE_OK;
    }

    const char *edge_table = (const char *)sqlite3_value_text(argv[0]);
    const char *src_col    = (const char *)sqlite3_value_text(argv[1]);
    const char *dst_col    = (const char *)sqlite3_value_text(argv[2]);

    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 ||
        id_validate(dst_col) != 0) {
        pCursor->pVtab->zErrMsg = sqlite3_mprintf(
            "graph_pagerank: invalid table/column identifier");
        return SQLITE_ERROR;
    }

    /* Optional parameters with defaults */
    double damping = 0.85;
    int iterations = 20;
    if (argc > 3 && sqlite3_value_type(argv[3]) != SQLITE_NULL) {
        damping = sqlite3_value_double(argv[3]);
    }
    if (argc > 4 && sqlite3_value_type(argv[4]) != SQLITE_NULL) {
        iterations = sqlite3_value_int(argv[4]);
    }

    GraphVtab *vtab = (GraphVtab *)pCursor->pVtab;
    int rc = run_pagerank(vtab->db, edge_table, src_col, dst_col,
                           damping, iterations, &cur->results);
    if (rc != SQLITE_OK) {
        cur->eof = 1;
        return rc;
    }

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int gpr_next(sqlite3_vtab_cursor *pCursor) {
    GraphPRCursor *cur = (GraphPRCursor *)pCursor;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int gpr_eof(sqlite3_vtab_cursor *pCursor) {
    return ((GraphPRCursor *)pCursor)->eof;
}

static int gpr_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    GraphPRCursor *cur = (GraphPRCursor *)pCursor;
    PRRow *row = &cur->results.rows[cur->current];
    switch (col) {
        case GPR_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT);
            break;
        case GPR_COL_RANK:
            sqlite3_result_double(ctx, row->rank);
            break;
        default:
            sqlite3_result_null(ctx);
            break;
    }
    return SQLITE_OK;
}

static int gpr_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((GraphPRCursor *)pCursor)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_pr_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only */
    .xConnect = gpr_connect,
    .xBestIndex = gpr_best_index,
    .xDisconnect = graph_vtab_disconnect,
    .xDestroy = graph_vtab_disconnect,
    .xOpen = gpr_open,
    .xClose = gpr_close,
    .xFilter = gpr_filter,
    .xNext = gpr_next,
    .xEof = gpr_eof,
    .xColumn = gpr_column,
    .xRowid = gpr_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Registration
 * ═══════════════════════════════════════════════════════════════ */

int graph_register_tvfs(sqlite3 *db) {
    int rc;

    rc = sqlite3_create_module(db, "graph_bfs", &graph_bfs_module, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_dfs", &graph_dfs_module, (void *)1);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_shortest_path", &graph_sp_module, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_components", &graph_comp_module, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_pagerank", &graph_pr_module, NULL);
    if (rc != SQLITE_OK) return rc;

    return SQLITE_OK;
}
