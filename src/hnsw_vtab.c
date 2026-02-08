/*
 * hnsw_vtab.c — SQLite virtual table wrapping the HNSW index
 *
 * CREATE VIRTUAL TABLE t USING hnsw_index(
 *     dimensions=384, metric='cosine', m=16, ef_construction=200
 * );
 *
 * Columns: rowid (implicit), vector (BLOB), distance (REAL),
 *          k (INTEGER, hidden), ef_search (INTEGER, hidden)
 *
 * Shadow tables: {name}_config, {name}_nodes, {name}_edges
 */
#include "hnsw_vtab.h"
#include "hnsw_algo.h"
#include "vec_math.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

SQLITE_EXTENSION_INIT3

/* ─── Virtual Table Structure ──────────────────────────────── */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *table_name;
    HnswIndex *index;
    int dim;
} HnswVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    HnswVtab *vtab;

    /* Search results */
    HnswSearchResult *results;
    int result_count;
    int current;

    /* Point lookup */
    int64_t lookup_id;
    int is_point_lookup;
    int eof;
} HnswCursor;

/* Column indices */
#define COL_VECTOR    0
#define COL_DISTANCE  1
#define COL_K         2
#define COL_EF_SEARCH 3

/* xBestIndex plan IDs */
#define PLAN_SCAN     0
#define PLAN_KNN      1
#define PLAN_POINT    2

/* ─── Argument Parsing ─────────────────────────────────────── */

typedef struct {
    int dimensions;
    VecMetric metric;
    int m;
    int ef_construction;
} HnswParams;

static void params_defaults(HnswParams *p) {
    p->dimensions = 0;
    p->metric = VEC_METRIC_COSINE;
    p->m = 16;
    p->ef_construction = 200;
}

/*
 * Parse "key=value" pairs from CREATE VIRTUAL TABLE arguments.
 * argv[0] = module name, argv[1] = database name, argv[2] = table name,
 * argv[3..argc-1] = user parameters.
 */
static int parse_params(int argc, const char *const *argv, HnswParams *params, char **errmsg) {
    params_defaults(params);

    for (int i = 3; i < argc; i++) {
        const char *arg = argv[i];

        if (strncmp(arg, "dimensions=", 11) == 0) {
            params->dimensions = atoi(arg + 11);
            if (params->dimensions <= 0) {
                *errmsg = sqlite3_mprintf("hnsw_index: dimensions must be > 0, got '%s'", arg + 11);
                return SQLITE_ERROR;
            }
        } else if (strncmp(arg, "metric=", 7) == 0) {
            /* Strip quotes from metric value */
            const char *val = arg + 7;
            char metric_buf[32];
            int len = (int)strlen(val);
            if (len >= 2 && (val[0] == '\'' || val[0] == '"')) {
                len -= 2;
                if (len >= (int)sizeof(metric_buf)) len = (int)sizeof(metric_buf) - 1;
                memcpy(metric_buf, val + 1, (size_t)len);
                metric_buf[len] = '\0';
                val = metric_buf;
            }
            if (vec_parse_metric(val, &params->metric) != 0) {
                *errmsg = sqlite3_mprintf("hnsw_index: unknown metric '%s' (use 'l2', 'cosine', or 'inner_product')", val);
                return SQLITE_ERROR;
            }
        } else if (strncmp(arg, "m=", 2) == 0) {
            params->m = atoi(arg + 2);
            if (params->m < 2) {
                *errmsg = sqlite3_mprintf("hnsw_index: m must be >= 2, got '%s'", arg + 2);
                return SQLITE_ERROR;
            }
        } else if (strncmp(arg, "ef_construction=", 16) == 0) {
            params->ef_construction = atoi(arg + 16);
            if (params->ef_construction < 1) {
                *errmsg = sqlite3_mprintf("hnsw_index: ef_construction must be >= 1, got '%s'", arg + 16);
                return SQLITE_ERROR;
            }
        } else {
            *errmsg = sqlite3_mprintf("hnsw_index: unknown parameter '%s'", arg);
            return SQLITE_ERROR;
        }
    }

    if (params->dimensions == 0) {
        *errmsg = sqlite3_mprintf("hnsw_index: 'dimensions' parameter is required");
        return SQLITE_ERROR;
    }

    return SQLITE_OK;
}

/* ─── Shadow Table Management ──────────────────────────────── */

static int create_shadow_tables(sqlite3 *db, const char *name) {
    char *sql;
    int rc;

    /* Config table */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_config\" (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Nodes table */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_nodes\" ("
        "  id INTEGER PRIMARY KEY,"
        "  vector BLOB NOT NULL,"
        "  level INTEGER NOT NULL,"
        "  deleted INTEGER NOT NULL DEFAULT 0"
        ")", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Edges table */
    sql = sqlite3_mprintf(
        "CREATE TABLE IF NOT EXISTS \"%w_edges\" ("
        "  source_id INTEGER NOT NULL,"
        "  target_id INTEGER NOT NULL,"
        "  level INTEGER NOT NULL,"
        "  distance REAL NOT NULL,"
        "  PRIMARY KEY (source_id, level, target_id)"
        ") WITHOUT ROWID", name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    /* Reverse index for edge lookups */
    sql = sqlite3_mprintf(
        "CREATE INDEX IF NOT EXISTS \"%w_edges_rev\" ON \"%w_edges\"(target_id, level)",
        name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static int save_config(sqlite3 *db, const char *name, const HnswParams *params,
                       int64_t entry_point, int max_level) {
    char *sql;
    int rc;

    sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w_config\" (key, value) VALUES"
        " ('dimensions', '%d'),"
        " ('metric', '%d'),"
        " ('m', '%d'),"
        " ('ef_construction', '%d'),"
        " ('entry_point', '%lld'),"
        " ('max_level', '%d')",
        name, params->dimensions, (int)params->metric,
        params->m, params->ef_construction,
        (long long)entry_point, max_level);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static int load_config(sqlite3 *db, const char *name, HnswParams *params,
                       int64_t *entry_point, int *max_level) {
    sqlite3_stmt *stmt;
    char *sql;
    int rc;

    params_defaults(params);
    *entry_point = -1;
    *max_level = -1;

    sql = sqlite3_mprintf("SELECT key, value FROM \"%w_config\"", name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *key = (const char *)sqlite3_column_text(stmt, 0);
        const char *val = (const char *)sqlite3_column_text(stmt, 1);
        if (strcmp(key, "dimensions") == 0) params->dimensions = atoi(val);
        else if (strcmp(key, "metric") == 0) params->metric = (VecMetric)atoi(val);
        else if (strcmp(key, "m") == 0) params->m = atoi(val);
        else if (strcmp(key, "ef_construction") == 0) params->ef_construction = atoi(val);
        else if (strcmp(key, "entry_point") == 0) *entry_point = atoll(val);
        else if (strcmp(key, "max_level") == 0) *max_level = atoi(val);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

/* Persist the in-memory index state to shadow tables */
static int persist_node(sqlite3 *db, const char *name, HnswIndex *index, HnswNode *node) {
    sqlite3_stmt *stmt;
    char *sql;
    int rc;

    /* Upsert node */
    sql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO \"%w_nodes\" (id, vector, level, deleted) VALUES (?, ?, ?, ?)",
        name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    sqlite3_bind_int64(stmt, 1, node->id);
    sqlite3_bind_blob(stmt, 2, node->vector, index->dim * (int)sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, node->level);
    sqlite3_bind_int(stmt, 4, node->deleted);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return SQLITE_ERROR;

    /* Delete old edges for this node, then insert current */
    sql = sqlite3_mprintf("DELETE FROM \"%w_edges\" WHERE source_id = %lld",
                          name, (long long)node->id);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    sql = sqlite3_mprintf(
        "INSERT INTO \"%w_edges\" (source_id, target_id, level, distance) VALUES (?, ?, ?, ?)",
        name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    for (int l = 0; l <= node->level; l++) {
        for (int i = 0; i < node->neighbor_count[l]; i++) {
            HnswNode *neighbor = hnsw_get_node(index, node->neighbors[l][i]);
            float dist = neighbor ? index->dist_func(node->vector, neighbor->vector, index->dim) : 0.0f;
            sqlite3_bind_int64(stmt, 1, node->id);
            sqlite3_bind_int64(stmt, 2, node->neighbors[l][i]);
            sqlite3_bind_int(stmt, 3, l);
            sqlite3_bind_double(stmt, 4, (double)dist);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

/* Load all nodes and edges from shadow tables into the in-memory index */
static int load_index_from_shadow(sqlite3 *db, const char *name, HnswIndex *index) {
    sqlite3_stmt *stmt;
    char *sql;
    int rc;

    /* Load nodes */
    sql = sqlite3_mprintf("SELECT id, vector, level, deleted FROM \"%w_nodes\"", name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t id = sqlite3_column_int64(stmt, 0);
        const float *vec = (const float *)sqlite3_column_blob(stmt, 1);
        int level = sqlite3_column_int(stmt, 2);
        int deleted = sqlite3_column_int(stmt, 3);

        /* Resize hash table if needed */
        if (index->node_count * 10 > index->node_capacity * 7) {
            ht_resize(index);
        }

        HnswNode *node = node_create(id, vec, index->dim, level);
        if (!node) { sqlite3_finalize(stmt); return SQLITE_NOMEM; }
        node->deleted = deleted;
        ht_insert(index->nodes, index->node_capacity, node);
        if (!deleted) index->node_count++;
    }
    sqlite3_finalize(stmt);

    /* Load edges */
    sql = sqlite3_mprintf("SELECT source_id, target_id, level FROM \"%w_edges\"", name);
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t src = sqlite3_column_int64(stmt, 0);
        int64_t dst = sqlite3_column_int64(stmt, 1);
        int level = sqlite3_column_int(stmt, 2);

        HnswNode *node = ht_find(index->nodes, index->node_capacity, src);
        if (node && level <= node->level) {
            node_add_neighbor(node, level, dst);
        }
    }
    sqlite3_finalize(stmt);

    return SQLITE_OK;
}

/* ─── Forward declarations for static functions used in module ─ */

/* These are needed because hnsw_algo.c has static ht_* functions.
 * We'll need to either make them non-static or duplicate them.
 * For now, we expose ht_find and ht_insert through the HnswIndex API. */

/* We need access to ht_find, ht_insert, ht_resize, node_create, node_add_neighbor
 * from hnsw_algo.c. Since they're static there, we'll redeclare compatible versions
 * here. In a production build, these would be in hnsw_algo.h. */

/* Workaround: access internal hash table via hnsw_get_node (public API) */

/* ─── Virtual Table Methods ────────────────────────────────── */

static int hnsw_vtab_create(sqlite3 *db, void *pAux, int argc,
                             const char *const *argv,
                             sqlite3_vtab **ppVTab, char **pzErr) {
    (void)pAux;
    HnswParams params;
    int rc = parse_params(argc, argv, &params, pzErr);
    if (rc != SQLITE_OK) return rc;

    /* Declare the virtual table schema */
    rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x(vector BLOB, distance REAL, k INTEGER HIDDEN, ef_search INTEGER HIDDEN)");
    if (rc != SQLITE_OK) return rc;

    /* Create shadow tables */
    rc = create_shadow_tables(db, argv[2]);
    if (rc != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("hnsw_index: failed to create shadow tables");
        return rc;
    }

    /* Create the HNSW index */
    HnswIndex *index = hnsw_create(params.dimensions, params.metric,
                                    params.m, params.ef_construction);
    if (!index) {
        *pzErr = sqlite3_mprintf("hnsw_index: failed to allocate index");
        return SQLITE_NOMEM;
    }

    /* Save config */
    save_config(db, argv[2], &params, -1, -1);

    /* Allocate vtab */
    HnswVtab *vtab = (HnswVtab *)sqlite3_malloc(sizeof(HnswVtab));
    if (!vtab) { hnsw_destroy(index); return SQLITE_NOMEM; }
    memset(vtab, 0, sizeof(HnswVtab));

    vtab->db = db;
    vtab->table_name = sqlite3_mprintf("%s", argv[2]);
    vtab->index = index;
    vtab->dim = params.dimensions;

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int hnsw_vtab_connect(sqlite3 *db, void *pAux, int argc,
                              const char *const *argv,
                              sqlite3_vtab **ppVTab, char **pzErr) {
    (void)pAux;
    (void)argc;
    /* Load existing index from shadow tables */
    HnswParams params;
    int64_t entry_point;
    int max_level;

    int rc = load_config(db, argv[2], &params, &entry_point, &max_level);
    if (rc != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("hnsw_index: failed to load config from shadow tables");
        return rc;
    }

    if (params.dimensions == 0) {
        *pzErr = sqlite3_mprintf("hnsw_index: corrupted config — dimensions is 0");
        return SQLITE_ERROR;
    }

    rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x(vector BLOB, distance REAL, k INTEGER HIDDEN, ef_search INTEGER HIDDEN)");
    if (rc != SQLITE_OK) return rc;

    HnswIndex *index = hnsw_create(params.dimensions, params.metric,
                                    params.m, params.ef_construction);
    if (!index) {
        *pzErr = sqlite3_mprintf("hnsw_index: failed to allocate index");
        return SQLITE_NOMEM;
    }

    /* Restore entry point and max level */
    index->entry_point = entry_point;
    index->max_level = max_level;

    /* Load all nodes and edges */
    rc = load_index_from_shadow(db, argv[2], index);
    if (rc != SQLITE_OK) {
        hnsw_destroy(index);
        *pzErr = sqlite3_mprintf("hnsw_index: failed to load index from shadow tables");
        return rc;
    }

    HnswVtab *vtab = (HnswVtab *)sqlite3_malloc(sizeof(HnswVtab));
    if (!vtab) { hnsw_destroy(index); return SQLITE_NOMEM; }
    memset(vtab, 0, sizeof(HnswVtab));

    vtab->db = db;
    vtab->table_name = sqlite3_mprintf("%s", argv[2]);
    vtab->index = index;
    vtab->dim = params.dimensions;

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int hnsw_vtab_disconnect(sqlite3_vtab *pVTab) {
    HnswVtab *vtab = (HnswVtab *)pVTab;
    hnsw_destroy(vtab->index);
    sqlite3_free(vtab->table_name);
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int hnsw_vtab_destroy(sqlite3_vtab *pVTab) {
    HnswVtab *vtab = (HnswVtab *)pVTab;

    /* Drop shadow tables */
    char *sql;
    sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w_config\"", vtab->table_name);
    sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w_nodes\"", vtab->table_name);
    sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    sql = sqlite3_mprintf("DROP INDEX IF EXISTS \"%w_edges_rev\"", vtab->table_name);
    sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w_edges\"", vtab->table_name);
    sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);

    return hnsw_vtab_disconnect(pVTab);
}

/* ─── xBestIndex ───────────────────────────────────────────── */

static int hnsw_vtab_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    int has_match = -1, has_k = -1, has_rowid = -1, has_ef = -1;
    int arg_idx = 1;

    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;

        int col = pInfo->aConstraint[i].iColumn;
        int op = pInfo->aConstraint[i].op;

        if (col == COL_VECTOR && op == SQLITE_INDEX_CONSTRAINT_MATCH) {
            has_match = i;
        } else if (col == COL_K && op == SQLITE_INDEX_CONSTRAINT_EQ) {
            has_k = i;
        } else if (col == COL_EF_SEARCH && op == SQLITE_INDEX_CONSTRAINT_EQ) {
            has_ef = i;
        } else if (col == -1 && op == SQLITE_INDEX_CONSTRAINT_EQ) {
            /* rowid = ? */
            has_rowid = i;
        }
    }

    if (has_match >= 0 && has_k >= 0) {
        /* KNN search plan */
        pInfo->idxNum = PLAN_KNN;
        pInfo->aConstraintUsage[has_match].argvIndex = arg_idx++;
        pInfo->aConstraintUsage[has_match].omit = 1;
        pInfo->aConstraintUsage[has_k].argvIndex = arg_idx++;
        pInfo->aConstraintUsage[has_k].omit = 1;
        if (has_ef >= 0) {
            pInfo->aConstraintUsage[has_ef].argvIndex = arg_idx++;
            pInfo->aConstraintUsage[has_ef].omit = 1;
        }
        pInfo->estimatedCost = 10.0;
        pInfo->estimatedRows = 10;
    } else if (has_rowid >= 0) {
        /* Point lookup */
        pInfo->idxNum = PLAN_POINT;
        pInfo->aConstraintUsage[has_rowid].argvIndex = 1;
        pInfo->aConstraintUsage[has_rowid].omit = 1;
        pInfo->estimatedCost = 1.0;
        pInfo->estimatedRows = 1;
    } else {
        /* Full scan (expensive) */
        pInfo->idxNum = PLAN_SCAN;
        pInfo->estimatedCost = 1000000.0;
        pInfo->estimatedRows = 1000000;
    }

    return SQLITE_OK;
}

/* ─── Cursor Methods ───────────────────────────────────────── */

static int hnsw_vtab_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    HnswCursor *cur = (HnswCursor *)sqlite3_malloc(sizeof(HnswCursor));
    if (!cur) return SQLITE_NOMEM;
    memset(cur, 0, sizeof(HnswCursor));
    cur->vtab = (HnswVtab *)pVTab;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int hnsw_vtab_close(sqlite3_vtab_cursor *pCursor) {
    HnswCursor *cur = (HnswCursor *)pCursor;
    free(cur->results);
    sqlite3_free(cur);
    return SQLITE_OK;
}

static int hnsw_vtab_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                             const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    HnswCursor *cur = (HnswCursor *)pCursor;
    HnswVtab *vtab = cur->vtab;

    /* Reset cursor state */
    free(cur->results);
    cur->results = NULL;
    cur->result_count = 0;
    cur->current = 0;
    cur->is_point_lookup = 0;
    cur->eof = 1;

    if (idxNum == PLAN_KNN) {
        /* KNN search: argv[0]=query_blob, argv[1]=k, argv[2]=ef_search (optional) */
        const float *query = (const float *)sqlite3_value_blob(argv[0]);
        int query_bytes = sqlite3_value_bytes(argv[0]);
        int k = sqlite3_value_int(argv[1]);
        int ef_search = (argc >= 3) ? sqlite3_value_int(argv[2]) : k * 2;

        int expected_bytes = vtab->dim * (int)sizeof(float);
        if (query_bytes != expected_bytes) {
            vtab->base.zErrMsg = sqlite3_mprintf(
                "hnsw_index: expected %d-dim vector (%d bytes), got %d bytes",
                vtab->dim, expected_bytes, query_bytes);
            return SQLITE_ERROR;
        }

        cur->results = (HnswSearchResult *)malloc((size_t)k * sizeof(HnswSearchResult));
        if (!cur->results) return SQLITE_NOMEM;

        cur->result_count = hnsw_search(vtab->index, query, k, ef_search, cur->results);
        cur->current = 0;
        cur->eof = (cur->result_count == 0);

    } else if (idxNum == PLAN_POINT) {
        /* Point lookup: argv[0]=rowid */
        cur->lookup_id = sqlite3_value_int64(argv[0]);
        cur->is_point_lookup = 1;
        cur->eof = (hnsw_get_node(vtab->index, cur->lookup_id) == NULL);

    } else {
        /* Full scan — not implemented for Phase 1 */
        cur->eof = 1;
    }

    return SQLITE_OK;
}

static int hnsw_vtab_next(sqlite3_vtab_cursor *pCursor) {
    HnswCursor *cur = (HnswCursor *)pCursor;
    if (cur->is_point_lookup) {
        cur->eof = 1;
    } else {
        cur->current++;
        if (cur->current >= cur->result_count) cur->eof = 1;
    }
    return SQLITE_OK;
}

static int hnsw_vtab_eof(sqlite3_vtab_cursor *pCursor) {
    return ((HnswCursor *)pCursor)->eof;
}

static int hnsw_vtab_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    HnswCursor *cur = (HnswCursor *)pCursor;
    if (cur->is_point_lookup) {
        *pRowid = cur->lookup_id;
    } else {
        *pRowid = cur->results[cur->current].id;
    }
    return SQLITE_OK;
}

static int hnsw_vtab_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    HnswCursor *cur = (HnswCursor *)pCursor;
    HnswVtab *vtab = cur->vtab;

    int64_t id;
    if (cur->is_point_lookup) {
        id = cur->lookup_id;
    } else {
        id = cur->results[cur->current].id;
    }

    switch (col) {
        case COL_VECTOR: {
            const float *vec = hnsw_get_vector(vtab->index, id);
            if (vec) {
                sqlite3_result_blob(ctx, vec, vtab->dim * (int)sizeof(float), SQLITE_TRANSIENT);
            } else {
                sqlite3_result_null(ctx);
            }
            break;
        }
        case COL_DISTANCE:
            if (cur->is_point_lookup) {
                sqlite3_result_double(ctx, 0.0);
            } else {
                sqlite3_result_double(ctx, (double)cur->results[cur->current].distance);
            }
            break;
        case COL_K:
        case COL_EF_SEARCH:
            sqlite3_result_null(ctx);
            break;
    }
    return SQLITE_OK;
}

/* ─── xUpdate (INSERT / DELETE) ────────────────────────────── */

static int hnsw_vtab_update(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                             sqlite3_int64 *pRowid) {
    HnswVtab *vtab = (HnswVtab *)pVTab;
    int rc;

    if (argc == 1) {
        /* DELETE: argv[0] = rowid to delete */
        int64_t id = sqlite3_value_int64(argv[0]);
        HnswNode *node = hnsw_get_node(vtab->index, id);
        if (!node) {
            vtab->base.zErrMsg = sqlite3_mprintf("hnsw_index: rowid %lld not found", (long long)id);
            return SQLITE_ERROR;
        }
        rc = hnsw_delete(vtab->index, id);
        if (rc != 0) return SQLITE_ERROR;

        /* Persist delete to shadow tables */
        char *sql = sqlite3_mprintf(
            "UPDATE \"%w_nodes\" SET deleted = 1 WHERE id = %lld",
            vtab->table_name, (long long)id);
        sqlite3_exec(vtab->db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);

        /* Update config */
        save_config(vtab->db, vtab->table_name, &(HnswParams){
            .dimensions = vtab->dim,
            .metric = vtab->index->metric,
            .m = vtab->index->M,
            .ef_construction = vtab->index->ef_construction
        }, vtab->index->entry_point, vtab->index->max_level);

        return SQLITE_OK;
    }

    if (argc > 1 && sqlite3_value_type(argv[0]) == SQLITE_NULL) {
        /* INSERT: argv[1] = rowid (or NULL for auto), argv[2+] = column values */
        int64_t id;
        if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
            /* Auto-generate rowid — use node_count + 1 as simple strategy */
            id = vtab->index->node_count + 1;
            /* Find unused ID */
            while (hnsw_get_node(vtab->index, id) != NULL) id++;
        } else {
            id = sqlite3_value_int64(argv[1]);
        }

        /* argv[2] = vector column */
        if (sqlite3_value_type(argv[2]) != SQLITE_BLOB) {
            vtab->base.zErrMsg = sqlite3_mprintf("hnsw_index: vector must be a BLOB");
            return SQLITE_ERROR;
        }

        const float *vec = (const float *)sqlite3_value_blob(argv[2]);
        int vec_bytes = sqlite3_value_bytes(argv[2]);
        int expected_bytes = vtab->dim * (int)sizeof(float);

        if (vec_bytes != expected_bytes) {
            vtab->base.zErrMsg = sqlite3_mprintf(
                "hnsw_index: expected %d-dim vector (%d bytes), got %d bytes",
                vtab->dim, expected_bytes, vec_bytes);
            return SQLITE_ERROR;
        }

        rc = hnsw_insert(vtab->index, id, vec);
        if (rc != 0) {
            vtab->base.zErrMsg = sqlite3_mprintf("hnsw_index: insert failed (duplicate rowid %lld?)",
                                                  (long long)id);
            return SQLITE_ERROR;
        }

        /* Persist to shadow tables */
        HnswNode *node = hnsw_get_node(vtab->index, id);
        if (node) {
            persist_node(vtab->db, vtab->table_name, vtab->index, node);
        }

        /* Also persist neighbors that may have been modified */
        for (int l = 0; l <= node->level; l++) {
            for (int i = 0; i < node->neighbor_count[l]; i++) {
                HnswNode *neighbor = hnsw_get_node(vtab->index, node->neighbors[l][i]);
                if (neighbor) {
                    persist_node(vtab->db, vtab->table_name, vtab->index, neighbor);
                }
            }
        }

        /* Update config */
        save_config(vtab->db, vtab->table_name, &(HnswParams){
            .dimensions = vtab->dim,
            .metric = vtab->index->metric,
            .m = vtab->index->M,
            .ef_construction = vtab->index->ef_construction
        }, vtab->index->entry_point, vtab->index->max_level);

        *pRowid = id;
        return SQLITE_OK;
    }

    vtab->base.zErrMsg = sqlite3_mprintf("hnsw_index: UPDATE not supported, use DELETE + INSERT");
    return SQLITE_ERROR;
}

/* ─── Module Registration ──────────────────────────────────── */

static sqlite3_module hnsw_module = {
    .iVersion = 0,
    .xCreate = hnsw_vtab_create,
    .xConnect = hnsw_vtab_connect,
    .xBestIndex = hnsw_vtab_best_index,
    .xDisconnect = hnsw_vtab_disconnect,
    .xDestroy = hnsw_vtab_destroy,
    .xOpen = hnsw_vtab_open,
    .xClose = hnsw_vtab_close,
    .xFilter = hnsw_vtab_filter,
    .xNext = hnsw_vtab_next,
    .xEof = hnsw_vtab_eof,
    .xColumn = hnsw_vtab_column,
    .xRowid = hnsw_vtab_rowid,
    .xUpdate = hnsw_vtab_update,
};

int hnsw_register_module(sqlite3 *db) {
    return sqlite3_create_module(db, "hnsw_index", &hnsw_module, NULL);
}
