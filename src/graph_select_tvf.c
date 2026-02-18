/*
 * graph_select_tvf.c — SQLite TVF for dbt-inspired graph node selection
 *
 * Usage:
 *   SELECT node, depth, direction FROM graph_select(
 *       'edge_table', 'src_col', 'dst_col', 'selector_expr'
 *   );
 *
 * The selector expression supports:
 *   node          — bare node
 *   +node         — node + all ancestors
 *   node+         — node + all descendants
 *   +node+        — both directions
 *   N+node+M      — depth-limited
 *   @node         — transitive build closure
 *   A B           — union (space-separated)
 *   A,B           — intersection
 *   not A         — complement
 */
#include "graph_select_tvf.h"
#include "graph_common.h"
#include "graph_load.h"
#include "graph_selector_eval.h"
#include "graph_selector_parse.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table & Cursor
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} GraphSelectVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    /* Loaded graph (owned) */
    GraphData graph;
    int graph_loaded;
    /* Evaluation results (owned) */
    SelectResults results;
    int current;
    int eof;
} GraphSelectCursor;

/* Output columns */
enum {
    GS_COL_NODE = 0,
    GS_COL_DEPTH,
    GS_COL_DIRECTION,
    /* Hidden input columns */
    GS_COL_EDGE_TABLE,
    GS_COL_SRC_COL,
    GS_COL_DST_COL,
    GS_COL_SELECTOR,
};

#define GS_REQUIRED_MASK 0x0F /* all 4 hidden columns required */

/* ═══════════════════════════════════════════════════════════════
 * xConnect (eponymous-only: xCreate = NULL)
 * ═══════════════════════════════════════════════════════════════ */

static int gs_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVTab, char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;

    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x("
                                      "  node TEXT,"
                                      "  depth INTEGER,"
                                      "  direction TEXT,"
                                      "  edge_table TEXT HIDDEN,"
                                      "  src_col TEXT HIDDEN,"
                                      "  dst_col TEXT HIDDEN,"
                                      "  selector TEXT HIDDEN"
                                      ")");
    if (rc != SQLITE_OK)
        return rc;

    GraphSelectVtab *vtab = (GraphSelectVtab *)sqlite3_malloc(sizeof(GraphSelectVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(GraphSelectVtab));
    vtab->db = db;

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int gs_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * xBestIndex — uses the two-pass helper from graph_common.h
 * ═══════════════════════════════════════════════════════════════ */

static int gs_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    return graph_best_index_common(pInfo, GS_COL_EDGE_TABLE, GS_COL_SELECTOR, GS_REQUIRED_MASK, 1000.0);
}

/* ═══════════════════════════════════════════════════════════════
 * xOpen / xClose
 * ═══════════════════════════════════════════════════════════════ */

static int gs_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    GraphSelectCursor *cur = (GraphSelectCursor *)sqlite3_malloc(sizeof(GraphSelectCursor));
    if (!cur)
        return SQLITE_NOMEM;
    memset(cur, 0, sizeof(GraphSelectCursor));
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int gs_close(sqlite3_vtab_cursor *pCursor) {
    GraphSelectCursor *cur = (GraphSelectCursor *)pCursor;
    if (cur->graph_loaded)
        graph_data_destroy(&cur->graph);
    select_results_destroy(&cur->results);
    sqlite3_free(cur);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * xFilter — parse selector, load graph, evaluate
 * ═══════════════════════════════════════════════════════════════ */

static int gs_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    GraphSelectCursor *cur = (GraphSelectCursor *)pCursor;
    GraphSelectVtab *vtab = (GraphSelectVtab *)pCursor->pVtab;

    /* Clean up previous query */
    if (cur->graph_loaded) {
        graph_data_destroy(&cur->graph);
        cur->graph_loaded = 0;
    }
    select_results_destroy(&cur->results);
    cur->current = 0;
    cur->eof = 1;

    /* Check required columns via bitmask */
    if ((idxNum & GS_REQUIRED_MASK) != GS_REQUIRED_MASK || argc < 4) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_select: requires edge_table, src_col, dst_col, selector");
        return SQLITE_ERROR;
    }

    /* Extract parameters (sequential argv from graph_best_index_common) */
    int arg = 0;
    const char *edge_table = NULL, *src_col = NULL, *dst_col = NULL;
    const char *selector_expr = NULL;

    for (int bit = 0; bit < 4; bit++) {
        if (idxNum & (1 << bit)) {
            const char *val = (const char *)sqlite3_value_text(argv[arg++]);
            switch (bit) {
            case 0:
                edge_table = val;
                break;
            case 1:
                src_col = val;
                break;
            case 2:
                dst_col = val;
                break;
            case 3:
                selector_expr = val;
                break;
            }
        }
    }

    if (!edge_table || !src_col || !dst_col || !selector_expr) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_select: NULL parameter not allowed");
        return SQLITE_ERROR;
    }

    /* Validate identifiers */
    if (id_validate(edge_table) != 0 || id_validate(src_col) != 0 || id_validate(dst_col) != 0) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_select: invalid table/column identifier");
        return SQLITE_ERROR;
    }

    /* Parse selector expression */
    char *parse_err = NULL;
    SelectorNode *ast = selector_parse(selector_expr, &parse_err);
    if (!ast) {
        vtab->base.zErrMsg = sqlite3_mprintf("%s", parse_err ? parse_err : "graph_select: parse failed");
        free(parse_err);
        return SQLITE_ERROR;
    }

    /* Load graph */
    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    config.edge_table = edge_table;
    config.src_col = src_col;
    config.dst_col = dst_col;
    config.direction = "both";

    graph_data_init(&cur->graph);
    char *load_err = NULL;
    int rc = graph_data_load(vtab->db, &config, &cur->graph, &load_err);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_select: %s", load_err ? load_err : "failed to load graph");
        sqlite3_free(load_err);
        selector_free(ast);
        return SQLITE_ERROR;
    }
    cur->graph_loaded = 1;

    /* Evaluate selector against graph */
    char *eval_err = NULL;
    rc = selector_eval(ast, &cur->graph, &cur->results, &eval_err);
    selector_free(ast);

    if (rc != 0) {
        vtab->base.zErrMsg = sqlite3_mprintf("%s", eval_err ? eval_err : "graph_select: evaluation failed");
        free(eval_err);
        return SQLITE_ERROR;
    }

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * xNext / xEof / xColumn / xRowid
 * ═══════════════════════════════════════════════════════════════ */

static int gs_next(sqlite3_vtab_cursor *pCursor) {
    GraphSelectCursor *cur = (GraphSelectCursor *)pCursor;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int gs_eof(sqlite3_vtab_cursor *pCursor) {
    return ((GraphSelectCursor *)pCursor)->eof;
}

static int gs_column(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int col) {
    GraphSelectCursor *cur = (GraphSelectCursor *)pCursor;
    SelectResult *row = &cur->results.rows[cur->current];

    switch (col) {
    case GS_COL_NODE:
        sqlite3_result_text(ctx, cur->graph.ids[row->node_idx], -1, SQLITE_TRANSIENT);
        break;
    case GS_COL_DEPTH:
        sqlite3_result_int(ctx, row->depth);
        break;
    case GS_COL_DIRECTION:
        sqlite3_result_text(ctx, row->direction, -1, SQLITE_STATIC);
        break;
    default:
        sqlite3_result_null(ctx);
        break;
    }
    return SQLITE_OK;
}

static int gs_rowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((GraphSelectCursor *)pCursor)->current;
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Module definition and registration
 * ═══════════════════════════════════════════════════════════════ */

static sqlite3_module graph_select_module = {
    .iVersion = 0,
    .xCreate = NULL, /* eponymous-only */
    .xConnect = gs_connect,
    .xBestIndex = gs_best_index,
    .xDisconnect = gs_disconnect,
    .xDestroy = gs_disconnect,
    .xOpen = gs_open,
    .xClose = gs_close,
    .xFilter = gs_filter,
    .xNext = gs_next,
    .xEof = gs_eof,
    .xColumn = gs_column,
    .xRowid = gs_rowid,
};

int graph_select_register_tvf(sqlite3 *db) {
    return sqlite3_create_module(db, "graph_select", &graph_select_module, NULL);
}
