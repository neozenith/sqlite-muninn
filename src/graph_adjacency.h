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
#include <stdint.h>

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

/* Returns 1 if (namespace_id, source_idx) is present in <vt>_sssp_delta
 * (the source's cached SSSP is stale and must be recomputed before
 * use), 0 if absent, or a negative SQLite error code on query failure.
 * G5's read path consults this before trusting a cache hit. */
int sssp_shadow_is_stale(sqlite3 *db, const char *vt_name, int namespace_id, int source_idx);

/* Cross-file accessors for the GII <vt>_config shadow table. G6+'s
 * cache state machine reads communities_generation/_resolution from
 * here without duplicating the prepare/step boilerplate. */
int64_t config_get_int64_public(sqlite3 *db, const char *name, const char *key, int64_t def);
double config_get_double(sqlite3 *db, const char *name, const char *key, double def);

/* Write a double to the GII <vt>_config shadow with %.17g formatting
 * (IEEE 754 binary64 round-trip precision). Used by G6 store_communities
 * for resolution / modularity values where bit-equal round-trip matters
 * (the 1e-10 tolerance only hides 'close' divergence — exact match of
 * a previously-stored gamma must round-trip cleanly). */
int config_set_double(sqlite3 *db, const char *name, const char *key, double value);

/* G4 threshold-based rebuild dispatch. Pure classifier — given the
 * change ratio |delta|/total_edges and the two configured thresholds,
 * pick which rebuild strategy to run. The actual cascade behavior is
 * wired in T4.4. */
typedef enum {
    REBUILD_SELECTIVE = 0,   /* ratio < theta_selective: per-block rebuild */
    REBUILD_DELTA_FLUSH = 1, /* ratio in [theta_selective, theta_full): namespace flush */
    REBUILD_FULL = 2         /* ratio >= theta_full or total==0: full rebuild + gen++ */
} SsspRebuildStrategy;

SsspRebuildStrategy sssp_classify_rebuild(int delta_count, int total_edges, double theta_selective, double theta_full);

/* G4 cascade emit. Per the rebuild strategy:
 *   REBUILD_SELECTIVE / REBUILD_DELTA_FLUSH:
 *     INSERT OR IGNORE each affected_source_idxs[i] into <vt>_sssp_delta;
 *     do not touch _sssp; do not bump generation.
 *   REBUILD_FULL:
 *     DELETE all rows from <vt>_sssp and <vt>_sssp_delta; bump
 *     generation in <vt>_config. affected_source_idxs is ignored. */
int sssp_cascade_emit(sqlite3 *db, const char *vt_name, int namespace_id, SsspRebuildStrategy strategy,
                      const int *affected_source_idxs, int n);

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
