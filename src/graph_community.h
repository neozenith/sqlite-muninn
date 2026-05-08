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
double run_leiden(const GraphData *g, int *community, double resolution, const char *direction);

/* Communities-cache state machine (G6 T6.2).
 *
 *   COMM_CACHE_HIT         G_comm == G_adj AND resolution matches:
 *                          read partition from <vt>_communities.
 *   COMM_CACHE_WARM_START  G_comm < G_adj AND resolution matches:
 *                          warm-start Leiden from cached partition.
 *   COMM_CACHE_COLD_START  Never computed OR resolution mismatched:
 *                          cold-start (singletons or component-seeded).
 *
 * Resolution comparison uses 1e-10 tolerance — gamma round-trips
 * through TEXT-typed config values via %.17g formatting.
 */
typedef enum {
    COMM_CACHE_HIT = 0,
    COMM_CACHE_WARM_START = 1,
    COMM_CACHE_COLD_START = 2
} CommCacheState;

CommCacheState check_communities_cache(sqlite3 *db, const char *vtab_name, double requested_resolution);

#endif /* GRAPH_COMMUNITY_H */
