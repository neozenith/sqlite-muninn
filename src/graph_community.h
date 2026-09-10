/*
 * graph_community.h — Community detection and partition-quality TVFs
 *
 * Registers graph_leiden and graph_conductance table-valued functions
 * with SQLite. Also exposes run_leiden() and run_conductance() for
 * programmatic use by other subsystems.
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

/* Per-group conductance statistics (Kannan, Vempala & Vetta, 2004). */
typedef struct {
    long size;       /* nodes in the group */
    double internal; /* summed weight, both endpoints inside */
    double cut;      /* summed weight, exactly one endpoint inside */
    double vol;      /* 2 * internal + cut */
    double phi;      /* cut / min(vol, total_vol - vol); 0.0 when the denominator is 0 */
} Conductance;

/*
 * Score any grouping of a loaded graph.
 *
 * Args:
 *   g         — loaded GraphData with out[] and in[] adjacency
 *   group     — int[g->node_count]; group index per node, -1 = ungrouped
 *   k         — number of distinct groups (group values are in 0..k-1)
 *   direction — "forward", "reverse", or "both" (use "both" for undirected)
 *   out       — caller-allocated Conductance[k]
 *
 * Each edge row in the loaded graph is counted exactly once. Edges touching an
 * ungrouped node still count toward the cut of the grouped endpoint and toward
 * the total volume.
 *
 * Returns SQLITE_OK.
 */
int run_conductance(const GraphData *g, const int *group, int k, const char *direction, Conductance *out);

#endif /* GRAPH_COMMUNITY_H */
