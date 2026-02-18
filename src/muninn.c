/*
 * muninn.c — Extension entry point for sqlite-muninn
 *
 * Registers all modules and functions with SQLite:
 * - hnsw_index virtual table (HNSW vector index)
 * - graph_bfs, graph_dfs, graph_shortest_path, graph_components, graph_pagerank TVFs
 * - graph_degree, graph_betweenness, graph_closeness centrality TVFs
 * - graph_leiden community detection TVF
 * - graph_adjacency virtual table (persistent CSR adjacency cache)
 * - graph_select TVF (dbt-style node selection)
 * - node2vec_train() scalar function
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "muninn.h"
#include "hnsw_vtab.h"
#include "graph_tvf.h"
#include "graph_centrality.h"
#include "graph_community.h"
#include "graph_adjacency.h"
#include "graph_select_tvf.h"
#include "node2vec.h"

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_muninn_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
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

    rc = centrality_register_tvfs(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register centrality TVFs");
        return rc;
    }

    rc = community_register_tvfs(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register community TVFs");
        return rc;
    }

    rc = adjacency_register_module(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register graph_adjacency module");
        return rc;
    }

    rc = graph_select_register_tvf(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register graph_select TVF");
        return rc;
    }

    rc = node2vec_register_functions(db);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("muninn: failed to register node2vec functions");
        return rc;
    }

    return SQLITE_OK;
}
