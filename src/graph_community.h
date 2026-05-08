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
#include <stdint.h>

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

/* Warm-start Leiden (G6 T6.4).
 *
 * Args:
 *   g              Loaded GraphData.
 *   community      IN:  initial partition. For each node, singleton (idx)
 *                       for changed nodes; cached community_id otherwise.
 *                  OUT: final partition after refinement.
 *   resolution     gamma parameter (modularity resolution).
 *   direction      "forward" / "reverse" / "both".
 *   changed_nodes  Optional array of node indices whose 1-hop
 *                  neighborhoods changed since G_comm. NULL or
 *                  n_changed == 0 falls back to a cold-start run.
 *   n_changed      Number of entries in changed_nodes.
 *
 * Returns: final modularity Q.
 */
double run_leiden_warm(const GraphData *g, int *community, double resolution, const char *direction,
                       const int *changed_nodes, int n_changed);

/* Communities cache I/O (G6 T6.5).
 *
 * leiden_shadow_put: atomically replace the cached partition for one
 *   namespace and update the four communities_* config keys. Wraps the
 *   DELETE + INSERT loop + config writes inside a SAVEPOINT so any
 *   mid-batch failure rolls back the whole operation.
 *
 *     community[i]   community_id assigned to node i (i ∈ [0, n))
 *     resolution     gamma used to compute this partition (stored via
 *                    %.17g for round-trip precision)
 *     modularity     final Q
 *     generation     G_adj at which the partition was computed
 *
 *   Returns SQLITE_OK on success, SQLite errcode on failure.
 *
 * leiden_shadow_get: read the cached partition for one namespace.
 *   *out_community is malloc'd by the callee; caller frees with free().
 *   Returns:
 *     SQLITE_OK        partition loaded into *out_community / *out_n
 *     SQLITE_NOTFOUND  no rows for that (vt, namespace)
 *     other            SQLite errcode on prepare/step failure
 */
int leiden_shadow_put(sqlite3 *db, const char *vt_name, int namespace_id, const int *community, int n,
                      double resolution, double modularity, int64_t generation);
int leiden_shadow_get(sqlite3 *db, const char *vt_name, int namespace_id, int **out_community, int *out_n);

#endif /* GRAPH_COMMUNITY_H */
