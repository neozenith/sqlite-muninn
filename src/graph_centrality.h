/*
 * graph_centrality.h — Centrality measure TVFs
 *
 * Registers graph_degree, graph_node_betweenness, graph_edge_betweenness,
 * and graph_closeness table-valued functions with SQLite.
 */
#ifndef GRAPH_CENTRALITY_H
#define GRAPH_CENTRALITY_H

#include "sqlite3ext.h"

int centrality_register_tvfs(sqlite3 *db);

#endif /* GRAPH_CENTRALITY_H */
