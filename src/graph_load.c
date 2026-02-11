/*
 * graph_load.c — Shared graph loading for graph algorithms
 *
 * Provides hash-map-based node lookup (O(1) amortized) and adjacency list
 * construction with support for weighted edges and temporal WHERE clauses.
 *
 * When compiled as part of the extension (SQLITE_CORE not defined),
 * sqlite3ext.h redirects all sqlite3_* calls through the extension API
 * function pointer table. The C test runner links libsqlite3 directly
 * and only exercises the hash map (no SQL calls).
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_load.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Hash function (djb2)
 * ═══════════════════════════════════════════════════════════════ */

static unsigned int str_hash(const char *s) {
    unsigned int h = 5381;
    for (; *s; s++) h = h * 33 + (unsigned char)*s;
    return h;
}

/* ═══════════════════════════════════════════════════════════════
 * GraphData lifecycle
 * ═══════════════════════════════════════════════════════════════ */

void graph_data_init(GraphData *g) {
    memset(g, 0, sizeof(GraphData));
    g->node_capacity = 64;
    g->ids = (char **)calloc((size_t)g->node_capacity, sizeof(char *));
    g->out = (GraphAdjList *)calloc((size_t)g->node_capacity, sizeof(GraphAdjList));
    g->in  = (GraphAdjList *)calloc((size_t)g->node_capacity, sizeof(GraphAdjList));

    g->map_capacity = 128;  /* power of 2, > node_capacity * 1.4 */
    g->map_indices = (int *)malloc((size_t)g->map_capacity * sizeof(int));
    for (int i = 0; i < g->map_capacity; i++) g->map_indices[i] = -1;
}

void graph_data_destroy(GraphData *g) {
    for (int i = 0; i < g->node_count; i++) {
        free(g->ids[i]);
        free(g->out[i].edges);
        free(g->in[i].edges);
    }
    free(g->ids);
    free(g->out);
    free(g->in);
    free(g->map_indices);
    memset(g, 0, sizeof(GraphData));
}

/* ═══════════════════════════════════════════════════════════════
 * Hash map operations
 * ═══════════════════════════════════════════════════════════════ */

static void graph_data_rehash(GraphData *g) {
    int new_cap = g->map_capacity * 2;
    int *new_map = (int *)malloc((size_t)new_cap * sizeof(int));
    for (int i = 0; i < new_cap; i++) new_map[i] = -1;

    for (int i = 0; i < g->node_count; i++) {
        unsigned int slot = str_hash(g->ids[i]) & (unsigned int)(new_cap - 1);
        while (new_map[slot] != -1) {
            slot = (slot + 1) & (unsigned int)(new_cap - 1);
        }
        new_map[slot] = i;
    }
    free(g->map_indices);
    g->map_indices = new_map;
    g->map_capacity = new_cap;
}

int graph_data_find(const GraphData *g, const char *id) {
    unsigned int slot = str_hash(id) & (unsigned int)(g->map_capacity - 1);
    for (int probe = 0; probe < g->map_capacity; probe++) {
        int idx = g->map_indices[slot];
        if (idx == -1) return -1;
        if (strcmp(g->ids[idx], id) == 0) return idx;
        slot = (slot + 1) & (unsigned int)(g->map_capacity - 1);
    }
    return -1;
}

int graph_data_find_or_add(GraphData *g, const char *id) {
    /* Check existing */
    int found = graph_data_find(g, id);
    if (found >= 0) return found;

    /* Grow node arrays if needed */
    if (g->node_count >= g->node_capacity) {
        int new_cap = g->node_capacity * 2;
        g->ids = (char **)realloc(g->ids, (size_t)new_cap * sizeof(char *));
        g->out = (GraphAdjList *)realloc(g->out, (size_t)new_cap * sizeof(GraphAdjList));
        g->in  = (GraphAdjList *)realloc(g->in, (size_t)new_cap * sizeof(GraphAdjList));
        for (int i = g->node_capacity; i < new_cap; i++) {
            g->ids[i] = NULL;
            memset(&g->out[i], 0, sizeof(GraphAdjList));
            memset(&g->in[i], 0, sizeof(GraphAdjList));
        }
        g->node_capacity = new_cap;
    }

    int idx = g->node_count++;
    g->ids[idx] = strdup(id);

    /* Rehash at 70% load factor */
    if (g->node_count * 10 > g->map_capacity * 7) {
        graph_data_rehash(g);
    }

    /* Insert into hash map */
    unsigned int slot = str_hash(id) & (unsigned int)(g->map_capacity - 1);
    while (g->map_indices[slot] != -1) {
        slot = (slot + 1) & (unsigned int)(g->map_capacity - 1);
    }
    g->map_indices[slot] = idx;

    return idx;
}

/* ═══════════════════════════════════════════════════════════════
 * Adjacency list helper
 * ═══════════════════════════════════════════════════════════════ */

static void adj_add(GraphAdjList *adj, int target, double weight) {
    if (adj->count >= adj->capacity) {
        int nc = adj->capacity == 0 ? 8 : adj->capacity * 2;
        adj->edges = (GraphEdge *)realloc(adj->edges, (size_t)nc * sizeof(GraphEdge));
        adj->capacity = nc;
    }
    adj->edges[adj->count].target = target;
    adj->edges[adj->count].weight = weight;
    adj->count++;
}

/* ═══════════════════════════════════════════════════════════════
 * Graph loading from SQLite table
 * ═══════════════════════════════════════════════════════════════ */

int graph_data_load(sqlite3 *db, const GraphLoadConfig *config,
                    GraphData *g, char **pzErrMsg) {
    /* Validate identifiers */
    if (id_validate(config->edge_table) != 0 ||
        id_validate(config->src_col) != 0 ||
        id_validate(config->dst_col) != 0) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("invalid table/column identifier");
        return SQLITE_ERROR;
    }
    if (config->weight_col && id_validate(config->weight_col) != 0) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("invalid weight column identifier");
        return SQLITE_ERROR;
    }
    if (config->timestamp_col && id_validate(config->timestamp_col) != 0) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("invalid timestamp column identifier");
        return SQLITE_ERROR;
    }

    /* Build SELECT statement */
    char *sql;
    if (config->weight_col) {
        if (config->timestamp_col) {
            sql = sqlite3_mprintf(
                "SELECT \"%w\", \"%w\", \"%w\" FROM \"%w\""
                " WHERE (\"%w\" >= ?1 OR ?1 IS NULL)"
                " AND (\"%w\" <= ?2 OR ?2 IS NULL)",
                config->src_col, config->dst_col, config->weight_col,
                config->edge_table,
                config->timestamp_col, config->timestamp_col);
        } else {
            sql = sqlite3_mprintf(
                "SELECT \"%w\", \"%w\", \"%w\" FROM \"%w\"",
                config->src_col, config->dst_col, config->weight_col,
                config->edge_table);
        }
    } else {
        if (config->timestamp_col) {
            sql = sqlite3_mprintf(
                "SELECT \"%w\", \"%w\" FROM \"%w\""
                " WHERE (\"%w\" >= ?1 OR ?1 IS NULL)"
                " AND (\"%w\" <= ?2 OR ?2 IS NULL)",
                config->src_col, config->dst_col,
                config->edge_table,
                config->timestamp_col, config->timestamp_col);
        } else {
            sql = sqlite3_mprintf(
                "SELECT \"%w\", \"%w\" FROM \"%w\"",
                config->src_col, config->dst_col,
                config->edge_table);
        }
    }
    if (!sql) return SQLITE_NOMEM;

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("failed to prepare: %s",
                                                    sqlite3_errmsg(db));
        return rc;
    }

    /* Bind temporal parameters if configured */
    if (config->timestamp_col) {
        if (config->time_start) {
            sqlite3_bind_value(stmt, 1, config->time_start);
        } else {
            sqlite3_bind_null(stmt, 1);
        }
        if (config->time_end) {
            sqlite3_bind_value(stmt, 2, config->time_end);
        } else {
            sqlite3_bind_null(stmt, 2);
        }
    }

    /* Determine direction flags */
    int add_forward = 1, add_reverse = 1;
    if (config->direction) {
        if (strcmp(config->direction, "forward") == 0) {
            add_reverse = 0;
        } else if (strcmp(config->direction, "reverse") == 0) {
            add_forward = 0;
        }
        /* "both" or NULL: keep both */
    }

    /* Read edges */
    g->has_weights = (config->weight_col != NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *src = (const char *)sqlite3_column_text(stmt, 0);
        const char *dst = (const char *)sqlite3_column_text(stmt, 1);
        if (!src || !dst) continue;

        double weight = 1.0;
        if (config->weight_col) {
            weight = sqlite3_column_double(stmt, 2);
        }

        int si = graph_data_find_or_add(g, src);
        int di = graph_data_find_or_add(g, dst);

        if (add_forward) adj_add(&g->out[si], di, weight);
        if (add_reverse) adj_add(&g->in[di], si, weight);
        g->edge_count++;
    }
    sqlite3_finalize(stmt);

    return SQLITE_OK;
}
