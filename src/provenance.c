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
    if (rc != SQLITE_OK)
        return rc;

    /* Config table — holds the G_prov generation counter (TEXT-typed
     * value column matches graph_adjacency's convention). */
    sql = sqlite3_mprintf("CREATE TABLE IF NOT EXISTS \"%w_config\" "
                          "(key TEXT PRIMARY KEY, value TEXT)",
                          name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    sql = sqlite3_mprintf("INSERT OR IGNORE INTO \"%w_config\"(key, value) "
                          "VALUES ('generation', '0')",
                          name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static void prov_drop_shadow_tables(sqlite3 *db, const char *name) {
    const char *suffixes[] = {"_provenance", "_config"};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        char *sql = sqlite3_mprintf("DROP TABLE IF EXISTS \"%w%s\"", name, suffixes[i]);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Trigger Management
 *
 * Group 1 (T1.2): AFTER INSERT on event_message_chunks back-fills
 * _<name>_provenance for every entity already attached to that chunk.
 * canonical names resolve through entity_clusters when present, fall
 * back to ent.name otherwise. Identifier %w_emc_ai = "emc + after-insert".
 *
 * Group 2 (T1.3): AFTER INSERT/DELETE/UPDATE on entities (_ent_ai,
 * _ent_ad, _ent_au).
 *
 * Group 3 (T1.4): AFTER UPDATE OF canonical on entity_clusters
 * (_ec_au). Canonical rename cascade.
 *
 * Group 4 (T1.5): AFTER INSERT/DELETE on entity_clusters (_ec_ai,
 * _ec_ad). Cluster lifecycle remap.
 *
 * Every trigger body ends with PROV_BUMP_SQL which increments the
 * G_prov generation counter in _<name>_config (T1.6). The counter
 * over-invalidates: it ticks on every trigger fire even when zero
 * rows changed, so caches err on the side of recompute over stale
 * read. Each trigger string therefore takes one extra %w arg for
 * the _<name>_config table.
 * ═══════════════════════════════════════════════════════════════ */

#define PROV_BUMP_SQL                                                                                                  \
    "  UPDATE \"%w_config\" "                                                                                          \
    "  SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "                                                          \
    "  WHERE key = 'generation'; "

static int prov_install_triggers(sqlite3 *db, const char *name) {
    char *sql;
    int rc;

    /* Group 1: AFTER INSERT on event_message_chunks. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_emc_ai\" "
                          "AFTER INSERT ON \"event_message_chunks\" BEGIN "
                          "  INSERT OR IGNORE INTO \"%w_provenance\""
                          "    (namespace_id, chunk_id, canonical, project_id, timestamp) "
                          "  SELECT 0, NEW.chunk_id, COALESCE(ec.canonical, ent.name), "
                          "         e.project_id, e.timestamp "
                          "  FROM entities ent "
                          "  JOIN events e ON e.id = NEW.event_id "
                          "  LEFT JOIN entity_clusters ec ON ec.name = ent.name "
                          "  WHERE ent.chunk_id = NEW.chunk_id; " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 2a: AFTER INSERT on entities. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ent_ai\" "
                          "AFTER INSERT ON \"entities\" BEGIN "
                          "  INSERT OR IGNORE INTO \"%w_provenance\""
                          "    (namespace_id, chunk_id, canonical, project_id, timestamp) "
                          "  SELECT 0, NEW.chunk_id, COALESCE(ec.canonical, NEW.name), "
                          "         e.project_id, e.timestamp "
                          "  FROM event_message_chunks emc "
                          "  JOIN events e ON e.id = emc.event_id "
                          "  LEFT JOIN entity_clusters ec ON ec.name = NEW.name "
                          "  WHERE emc.chunk_id = NEW.chunk_id; " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 2b: AFTER DELETE on entities. Count-aware DELETE: only
     * remove the provenance row if NO OTHER entity in this chunk still
     * resolves to the same canonical (multi-entity-per-canonical
     * scenario surfaced by T1.7's parity test). The OLD row is already
     * gone from `entities` when the body runs, so the NOT EXISTS scan
     * sees only the remaining entities. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ent_ad\" "
                          "AFTER DELETE ON \"entities\" BEGIN "
                          "  DELETE FROM \"%w_provenance\" "
                          "  WHERE namespace_id = 0 "
                          "    AND chunk_id = OLD.chunk_id "
                          "    AND canonical = COALESCE("
                          "      (SELECT canonical FROM entity_clusters WHERE name = OLD.name), "
                          "      OLD.name) "
                          "    AND NOT EXISTS ("
                          "      SELECT 1 FROM entities ent "
                          "      WHERE ent.chunk_id = OLD.chunk_id "
                          "        AND COALESCE("
                          "              (SELECT canonical FROM entity_clusters WHERE name = ent.name), "
                          "              ent.name) "
                          "          = COALESCE("
                          "              (SELECT canonical FROM entity_clusters WHERE name = OLD.name), "
                          "              OLD.name)); " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 2c: AFTER UPDATE on entities — count-aware DELETE-of-OLD
     * (same NOT EXISTS guard as _ent_ad — if NEW.name still resolves
     * to OLD's canonical, or any other entity does, keep the row) +
     * INSERT-of-NEW. Single trigger body so both writes share a
     * transaction with the originating UPDATE. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ent_au\" "
                          "AFTER UPDATE ON \"entities\" BEGIN "
                          "  DELETE FROM \"%w_provenance\" "
                          "  WHERE namespace_id = 0 "
                          "    AND chunk_id = OLD.chunk_id "
                          "    AND canonical = COALESCE("
                          "      (SELECT canonical FROM entity_clusters WHERE name = OLD.name), "
                          "      OLD.name) "
                          "    AND NOT EXISTS ("
                          "      SELECT 1 FROM entities ent "
                          "      WHERE ent.chunk_id = OLD.chunk_id "
                          "        AND COALESCE("
                          "              (SELECT canonical FROM entity_clusters WHERE name = ent.name), "
                          "              ent.name) "
                          "          = COALESCE("
                          "              (SELECT canonical FROM entity_clusters WHERE name = OLD.name), "
                          "              OLD.name)); "
                          "  INSERT OR IGNORE INTO \"%w_provenance\""
                          "    (namespace_id, chunk_id, canonical, project_id, timestamp) "
                          "  SELECT 0, NEW.chunk_id, COALESCE(ec.canonical, NEW.name), "
                          "         e.project_id, e.timestamp "
                          "  FROM event_message_chunks emc "
                          "  JOIN events e ON e.id = emc.event_id "
                          "  LEFT JOIN entity_clusters ec ON ec.name = NEW.name "
                          "  WHERE emc.chunk_id = NEW.chunk_id; " PROV_BUMP_SQL "END",
                          name, name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 3: AFTER UPDATE OF canonical on entity_clusters — canonical
     * rename cascade. Column-scoped so cluster name renames don't fire. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ec_au\" "
                          "AFTER UPDATE OF canonical ON \"entity_clusters\" BEGIN "
                          "  UPDATE \"%w_provenance\" "
                          "  SET canonical = NEW.canonical "
                          "  WHERE namespace_id = 0 AND canonical = OLD.canonical; " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 4a: AFTER INSERT on entity_clusters — raw-name → canonical
     * remap (catches entities inserted before the cluster existed). */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ec_ai\" "
                          "AFTER INSERT ON \"entity_clusters\" BEGIN "
                          "  UPDATE \"%w_provenance\" "
                          "  SET canonical = NEW.canonical "
                          "  WHERE namespace_id = 0 AND canonical = NEW.name; " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK)
        return rc;

    /* Group 4b: AFTER DELETE on entity_clusters — symmetric inverse. */
    sql = sqlite3_mprintf("CREATE TRIGGER IF NOT EXISTS \"%w_ec_ad\" "
                          "AFTER DELETE ON \"entity_clusters\" BEGIN "
                          "  UPDATE \"%w_provenance\" "
                          "  SET canonical = OLD.name "
                          "  WHERE namespace_id = 0 AND canonical = OLD.canonical; " PROV_BUMP_SQL "END",
                          name, name, name);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    sqlite3_free(sql);
    return rc;
}

static void prov_remove_triggers(sqlite3 *db, const char *name) {
    const char *suffixes[] = {"_emc_ai", "_ent_ai", "_ent_ad", "_ent_au", "_ec_au", "_ec_ai", "_ec_ad"};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        char *sql = sqlite3_mprintf("DROP TRIGGER IF EXISTS \"%w%s\"", name, suffixes[i]);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
        sqlite3_free(sql);
    }
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

        /* Install triggers on upstream KG tables. event_message_chunks /
         * entities / entity_clusters / events must already exist — the
         * trigger references them by name and SQLite validates source
         * tables at trigger-creation time. */
        rc = prov_install_triggers(db, argv[2]);
        if (rc != SQLITE_OK) {
            *pzErr = sqlite3_mprintf("gii_provenance: failed to install triggers (%s)", sqlite3_errmsg(db));
            prov_drop_shadow_tables(db, argv[2]);
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
    prov_remove_triggers(vtab->db, vtab->vtab_name);
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
