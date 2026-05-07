/*
 * provenance.c — gii_provenance virtual table module.
 *
 * xCreate produces the documented `<name>_provenance` shadow table; later
 * tickets (T1.2..T1.5) attach triggers on event_message_chunks / entities /
 * entity_clusters and (T1.6) participate in the generation counter cascade.
 *
 * The VT itself is read-only over the shadow for now — cursor methods stream
 * rows back via SELECT. Filtering by namespace_id / project_id / canonical
 * lands later when the centrality TVFs and graph_topk_centrality (G2) need
 * push-down constraints.
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "provenance.h"

#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Virtual Table & Cursor Structures
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
    char *vtab_name; /* user-supplied name; shadow tables use "<name>_provenance" */
} ProvVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    sqlite3_stmt *stmt; /* SELECT over <name>_provenance */
    int eof;
    sqlite3_int64 rowid;
} ProvCursor;

/* Output column indices — must match the order used in xFilter's SELECT. */
enum {
    PROV_COL_NAMESPACE_ID = 0,
    PROV_COL_CHUNK_ID,
    PROV_COL_CANONICAL,
    PROV_COL_PROJECT_ID,
    PROV_COL_TIMESTAMP,
    PROV_NUM_COLS
};

/* ═══════════════════════════════════════════════════════════════
 * Shadow Table Management
 * ═══════════════════════════════════════════════════════════════ */

static int prov_create_shadow_tables(sqlite3 *db, const char *name) {
    char *sql;
    int rc;

    sql = sqlite3_mprintf("CREATE TABLE IF NOT EXISTS \"%w_provenance\" ("
                          "namespace_id INTEGER NOT NULL, "
                          "chunk_id INTEGER NOT NULL, "
                          "canonical TEXT NOT NULL, "
                          "project_id TEXT NOT NULL, "
                          "timestamp TEXT NOT NULL, "
                          "PRIMARY KEY (namespace_id, chunk_id, canonical))",
                          name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    sql = sqlite3_mprintf("CREATE INDEX IF NOT EXISTS \"%w_provenance_proj_ts\" "
                          "ON \"%w_provenance\"(namespace_id, project_id, timestamp)",
                          name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    sql = sqlite3_mprintf("CREATE INDEX IF NOT EXISTS \"%w_provenance_canonical\" "
                          "ON \"%w_provenance\"(namespace_id, canonical)",
                          name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static void prov_drop_shadow_tables(sqlite3 *db, const char *name) {
    char *sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w_provenance\"", name);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
}

/* ═══════════════════════════════════════════════════════════════
 * Module Methods
 * ═══════════════════════════════════════════════════════════════ */

static int prov_init(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVTab, char **pzErr,
                     int is_create) {
    (void)pAux;
    (void)argc;

    /* Declare the VT's visible schema. The placeholder table name "x" is
     * ignored by SQLite; only the column list matters. Mirrors the columns
     * of <name>_provenance. */
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x("
                                      "namespace_id INTEGER, "
                                      "chunk_id INTEGER, "
                                      "canonical TEXT, "
                                      "project_id TEXT, "
                                      "timestamp TEXT)");
    if (rc != SQLITE_OK)
        return rc;

    ProvVtab *vtab = (ProvVtab *)sqlite3_malloc(sizeof(ProvVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(ProvVtab));
    vtab->db = db;
    vtab->vtab_name = sqlite3_mprintf("%s", argv[2]);
    if (!vtab->vtab_name) {
        sqlite3_free(vtab);
        return SQLITE_NOMEM;
    }

    if (is_create) {
        rc = prov_create_shadow_tables(db, argv[2]);
        if (rc != SQLITE_OK) {
            *pzErr = sqlite3_mprintf("gii_provenance: failed to create shadow tables");
            sqlite3_free(vtab->vtab_name);
            sqlite3_free(vtab);
            return rc;
        }
    }

    *ppVTab = &vtab->base;
    return SQLITE_OK;
}

static int prov_xCreate(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVTab,
                        char **pzErr) {
    return prov_init(db, pAux, argc, argv, ppVTab, pzErr, 1);
}

static int prov_xConnect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVTab,
                         char **pzErr) {
    return prov_init(db, pAux, argc, argv, ppVTab, pzErr, 0);
}

static int prov_xDisconnect(sqlite3_vtab *pVTab) {
    ProvVtab *vtab = (ProvVtab *)pVTab;
    sqlite3_free(vtab->vtab_name);
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int prov_xDestroy(sqlite3_vtab *pVTab) {
    ProvVtab *vtab = (ProvVtab *)pVTab;
    prov_drop_shadow_tables(vtab->db, vtab->vtab_name);
    return prov_xDisconnect(pVTab);
}

/* xBestIndex: full-scan only for T1.1. Constraint pushdown lands when the
 * centrality TVFs (G2/G7) actually need namespace/project/canonical filters. */
static int prov_xBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
    (void)pVTab;
    pInfo->idxNum = 0;
    pInfo->estimatedCost = 1.0e6;
    return SQLITE_OK;
}

static int prov_xOpen(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    ProvCursor *cur = (ProvCursor *)sqlite3_malloc(sizeof(ProvCursor));
    if (!cur)
        return SQLITE_NOMEM;
    memset(cur, 0, sizeof(ProvCursor));
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int prov_xClose(sqlite3_vtab_cursor *pCursor) {
    ProvCursor *cur = (ProvCursor *)pCursor;
    if (cur->stmt)
        sqlite3_finalize(cur->stmt);
    sqlite3_free(cur);
    return SQLITE_OK;
}

static int prov_xFilter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc,
                        sqlite3_value **argv) {
    (void)idxNum;
    (void)idxStr;
    (void)argc;
    (void)argv;

    ProvCursor *cur = (ProvCursor *)pCursor;
    ProvVtab *vtab = (ProvVtab *)pCursor->pVtab;

    if (cur->stmt) {
        sqlite3_finalize(cur->stmt);
        cur->stmt = NULL;
    }
    char *sql = sqlite3_mprintf("SELECT namespace_id, chunk_id, canonical, project_id, timestamp "
                                "FROM \"%w_provenance\"",
                                vtab->vtab_name);
    if (!sql)
        return SQLITE_NOMEM;
    int rc = sqlite3_prepare_v2(vtab->db, sql, -1, &cur->stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    cur->rowid = 0;
    rc = sqlite3_step(cur->stmt);
    if (rc == SQLITE_ROW) {
        cur->eof = 0;
        cur->rowid = 1;
        return SQLITE_OK;
    }
    if (rc == SQLITE_DONE) {
        cur->eof = 1;
        return SQLITE_OK;
    }
    return rc;
}

static int prov_xNext(sqlite3_vtab_cursor *pCursor) {
    ProvCursor *cur = (ProvCursor *)pCursor;
    int rc = sqlite3_step(cur->stmt);
    if (rc == SQLITE_ROW) {
        cur->eof = 0;
        cur->rowid++;
        return SQLITE_OK;
    }
    if (rc == SQLITE_DONE) {
        cur->eof = 1;
        return SQLITE_OK;
    }
    return rc;
}

static int prov_xEof(sqlite3_vtab_cursor *pCursor) {
    return ((ProvCursor *)pCursor)->eof;
}

static int prov_xColumn(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int N) {
    ProvCursor *cur = (ProvCursor *)pCursor;
    if (!cur->stmt || cur->eof) {
        sqlite3_result_null(ctx);
        return SQLITE_OK;
    }
    int t = sqlite3_column_type(cur->stmt, N);
    if (t == SQLITE_INTEGER) {
        sqlite3_result_int64(ctx, sqlite3_column_int64(cur->stmt, N));
    } else if (t == SQLITE_TEXT) {
        sqlite3_result_text(ctx, (const char *)sqlite3_column_text(cur->stmt, N), sqlite3_column_bytes(cur->stmt, N),
                            SQLITE_TRANSIENT);
    } else {
        sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
}

static int prov_xRowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
    *pRowid = ((ProvCursor *)pCursor)->rowid;
    return SQLITE_OK;
}

static const sqlite3_module gii_provenance_module = {
    .iVersion = 0,
    .xCreate = prov_xCreate,
    .xConnect = prov_xConnect,
    .xBestIndex = prov_xBestIndex,
    .xDisconnect = prov_xDisconnect,
    .xDestroy = prov_xDestroy,
    .xOpen = prov_xOpen,
    .xClose = prov_xClose,
    .xFilter = prov_xFilter,
    .xNext = prov_xNext,
    .xEof = prov_xEof,
    .xColumn = prov_xColumn,
    .xRowid = prov_xRowid,
};

int provenance_register_module(sqlite3 *db) {
    return sqlite3_create_module(db, "gii_provenance", &gii_provenance_module, NULL);
}
