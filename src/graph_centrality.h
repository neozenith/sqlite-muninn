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
int brandes_compute(const GraphData *g, const char *direction, int auto_approx, int normalized, double *CB, double *EB);

/*
 * SSSP load-or-compute wrapper (G5 T5.1).
 *
 * Tries to read dist[]/sigma[] for the given (namespace, source) from the
 * GII VT's _sssp shadow table (created by features='sssp', see G4). On
 * cache miss, runs the same Dijkstra/BFS the centrality TVFs would have
 * run anyway, then writes the result back via sssp_shadow_put before
 * returning. On cache hit, copies the cached BLOB into dist[]/sigma[].
 *
 * Pre-allocated by caller, written by callee:
 *   dist[g->node_count]  shortest distances; -1.0 sentinel for unreachable
 *   sigma[g->node_count] shortest-path counts (used by Brandes; ignored
 *                         by closeness-only consumers but populated anyway)
 *
 * Index contract: `source` indexes into g->ids[] AND is used directly as
 * source_idx in the cache. Caller must ensure GraphData and GII VT use
 * matching node indexing (true when both load from the same edges table
 * without ORDER BY divergence). Cache invalidation on rebuild handles
 * the cross-rebuild instability.
 *
 * pred[]/stack[] reconstruction is T5.2's territory and not produced
 * here. Returns SQLITE_OK on success or a SQLite/standard error code.
 */
int sssp_load_or_compute(sqlite3 *db, const char *gii_vt_name, int namespace_id, const GraphData *g, int source,
                         int weighted, double *dist, double *sigma, const char *direction);

/* Dynamic int list used by Brandes pred[] entries. Promoted to the
 * header so external consumers (G5 T5.2 reconstruction, tests) can
 * pre-allocate / iterate / free pred[] entries without re-implementing
 * the helpers. Treat fields as opaque — use the functions below. */
typedef struct IntList {
    int *items;
    int count;
    int capacity;
} IntList;

void intlist_init(IntList *l);
void intlist_push(IntList *l, int val);
void intlist_clear(IntList *l);
void intlist_destroy(IntList *l);

/* Predecessor and stack reconstruction (G5 T5.2).
 *
 * Given a cached dist[] from a previous SSSP run and the same GraphData,
 * rebuild the pred[] adjacency and stack[] visit order needed by Brandes
 * back-propagation. For each w with dist[w] >= 0, u is added to pred[w]
 * iff there's an edge (u, w) with dist[u] + weight(u, w) == dist[w].
 * stack[] is filled with all reachable node indices sorted by dist
 * ascending; back-propagation pops from the end (largest dist first).
 *
 * pred must be pre-initialized via intlist_init for each entry; the
 * function calls intlist_clear before populating. stack must hold at
 * least node_count entries.
 *
 * Returns SQLITE_OK on success, SQLITE_NOMEM if scratch allocation
 * fails, SQLITE_MISUSE for null inputs.
 */
int reconstruct_pred_from_dist(const GraphData *g, int source, const double *dist, IntList *pred, int *stack,
                               int *stack_size, const char *direction, int weighted);

#endif /* GRAPH_CENTRALITY_H */
