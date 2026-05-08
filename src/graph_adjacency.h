/*
 * graph_adjacency.h — Persistent graph adjacency index virtual table
 *
 * Registers the "graph_adjacency" module with SQLite. Usage:
 *
 *   CREATE VIRTUAL TABLE g USING graph_adjacency(
 *       edge_table='edges', src_col='src', dst_col='dst',
 *       weight_col='weight'   -- optional
 *   );
 *
 * Columns: node, node_idx, in_degree, out_degree,
 *          weighted_in_degree, weighted_out_degree
 *
 * Shadow tables: _config, _nodes, _degree, _csr_fwd, _csr_rev, _delta
 */
#ifndef GRAPH_ADJACENCY_H
#define GRAPH_ADJACENCY_H

#include "sqlite3ext.h"
#include "graph_load.h"

/* Register the graph_adjacency virtual table module with db. */
int adjacency_register_module(sqlite3 *db);

/* G4 SSSP shadow API — read/write/clear one source's dist[]+sigma[] BLOBs.
 * Only meaningful when the VT was created with features='sssp'. See
 * docs/plans/adv-centrality-filtering.md G4 for the BLOB contract. */
int sssp_shadow_put(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx, const double *dist,
                    const double *sigma, int n);
int sssp_shadow_get(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx, double **out_dist,
                    double **out_sigma, int *out_n);
int sssp_shadow_clear_delta(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx);

/*
 * Check if a name corresponds to a graph_adjacency virtual table.
 * Returns 1 if yes, 0 if no.
 */
int is_graph_adjacency(sqlite3 *db, const char *name);

/*
 * Load a GraphData from a graph_adjacency VT's shadow tables.
 * If the CSR cache is stale (delta table has rows), falls back to
 * loading from the original edge table. When fresh, loads from
 * shadow table BLOBs (much faster than SQL scan).
 *
 * Returns SQLITE_OK on success. Caller must call graph_data_destroy().
 */
int graph_data_load_from_adjacency(sqlite3 *db, const char *vtab_name, GraphData *g, char **pzErrMsg);

#endif /* GRAPH_ADJACENCY_H */
