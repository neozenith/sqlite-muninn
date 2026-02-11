/*
 * muninn.c — Extension entry point for sqlite-muninn
 *
 * Registers all modules and functions with SQLite:
 * - hnsw_index virtual table module (Phase 1)
 * - graph_bfs, graph_dfs, etc. TVFs (Phase 2)
 * - node2vec_train() scalar function (Phase 4)
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "muninn.h"
#include "hnsw_vtab.h"
#include "graph_tvf.h"
#include "node2vec.h"

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_muninn_init(sqlite3 *db, char **pzErrMsg,
                           const sqlite3_api_routines *pApi) {
    SQLITE_EXTENSION_INIT2(pApi);
    int rc;

    rc = hnsw_register_module(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register hnsw_index module");
        return rc;
    }

    rc = graph_register_tvfs(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register graph TVFs");
        return rc;
    }

    rc = node2vec_register_functions(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register node2vec functions");
        return rc;
    }

    return SQLITE_OK;
}
