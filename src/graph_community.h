/*
 * graph_community.h — Community detection TVFs
 *
 * Registers graph_leiden table-valued function with SQLite.
 * Also exposes run_leiden() for programmatic use by other subsystems.
 */
#ifndef GRAPH_COMMUNITY_H
#define GRAPH_COMMUNITY_H

#include "sqlite3ext.h"
#include "graph_load.h"

int community_register_tvfs(sqlite3 *db);

/*
 * Run Leiden community detection on a loaded graph.
 *
 * Args:
 *   g          — loaded GraphData with out[] and in[] adjacency
 *   community  — caller-allocated int[g->node_count]; filled with community IDs
 *   resolution — modularity resolution (1.0 = standard)
 *   direction  — "forward", "reverse", or "both" (use "both" for undirected)
 *
 * Returns modularity Q.
 */
double run_leiden(const GraphData *g, int *community, double resolution,
                  const char *direction);

#endif /* GRAPH_COMMUNITY_H */
