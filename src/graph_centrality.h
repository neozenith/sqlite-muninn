/*
 * graph_centrality.h — Centrality measure TVFs
 *
 * Registers graph_degree, graph_node_betweenness, graph_edge_betweenness,
 * and graph_closeness table-valued functions with SQLite.
 * Also exposes brandes_compute() for programmatic use.
 */
#ifndef GRAPH_CENTRALITY_H
#define GRAPH_CENTRALITY_H

#include "sqlite3ext.h"
#include "graph_load.h"

int centrality_register_tvfs(sqlite3 *db);

/*
 * Brandes betweenness centrality computation.
 *
 * Computes node betweenness into CB[N] and optionally edge betweenness
 * into EB[N*N]. Pass EB=NULL to skip edge computation.
 *
 * Args:
 *   g             — loaded GraphData
 *   direction     — "forward", "reverse", or "both"
 *   auto_approx   — 0 = exact; N > 0 = approximate if node_count > N
 *   normalized    — 1 = normalize by (N-1)(N-2)/2 for undirected
 *   CB            — out: node betweenness, size N, caller must calloc
 *   EB            — out: edge betweenness, size N*N, caller must calloc; NULL to skip
 *
 * Returns SQLITE_OK or SQLITE_NOMEM.
 */
int brandes_compute(const GraphData *g, const char *direction,
                    int auto_approx, int normalized,
                    double *CB, double *EB);

#endif /* GRAPH_CENTRALITY_H */
