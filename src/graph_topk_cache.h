/*
 * graph_topk_cache.h — Top-K result cache TVF (G2)
 *
 * Currently exposes only the signature-computation primitive (G7 T7.4
 * forward-compat). The full cache machinery (xCreate, lookup, write-back,
 * generation invalidation) lands in G2.
 */
#ifndef GRAPH_TOPK_CACHE_H
#define GRAPH_TOPK_CACHE_H

#include "sqlite3ext.h"
#include <stdint.h>

/* Stable cache-key signature for a top-K centrality query.
 *
 * Every parameter that would change the output participates in the
 * hash input. G7 T7.4 enforces that community_filter and
 * community_resolution participate too — without them, two queries
 * that differ only on community filter would collide and read each
 * other's cached results.
 *
 * The hashing primitive is DJB2 today (32-bit), to be upgraded to
 * xxh3 (128-bit) by G2 T2.1 — at which point this signature changes
 * to a hex-string output. For T7.4's contract, the property under
 * test is "community_filter / community_resolution influence the
 * output," not the primitive's collision strength.
 *
 * Pass NULL strings as the empty string. For "no community filter,"
 * pass community_filter = 0 (sentinel — distinguished from a real
 * community 0 by additional context the caller carries; G2 will
 * formalize this).
 */
unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric, int top_k,
                            int depth, int min_degree, int64_t g_adj, int64_t g_prov, int community_filter,
                            double community_resolution);

/* Top-K result cache shadow table (G2 T2.2).
 *
 * Schema per plan section 689:
 *   CREATE TABLE _gii_topk_cache (
 *     signature        TEXT PRIMARY KEY,
 *     seeds_json       TEXT,
 *     nodes_json       TEXT,
 *     edges_json       TEXT,
 *     edge_generation  INTEGER,
 *     prov_generation  INTEGER,
 *     cached_at        TIMESTAMP
 *   );
 *
 * topk_cache_put: INSERT OR REPLACE the row keyed by signature.
 *   Lazily creates the cache table on first call. Returns SQLITE_OK
 *   or a SQLite errcode.
 *
 * topk_cache_get: SELECT seeds_json, nodes_json, edges_json WHERE
 *   signature = ? AND edge_generation = ? AND prov_generation = ?.
 *   Lazy compare-on-read per the plan's ADR — a row whose generations
 *   don't match the caller's current values is invisible (treated as
 *   miss; T2.3 verifies invalidation).
 *
 *   On hit: out_seeds_json / out_nodes_json / out_edges_json are
 *     malloc'd and populated; caller frees with free().
 *   On miss (no row OR stale generations): returns SQLITE_NOTFOUND
 *     and out_* are set to NULL.
 *   On error: returns SQLite errcode; out_* are set to NULL.
 *
 *   Signature parameter is unsigned int (DJB2-derived); rendered as
 *   8-char lowercase hex for the TEXT PRIMARY KEY column. The xxh3
 *   swap (T2.4 follow-up) widens the column to 32-char hex without
 *   changing the read/write API.
 */
int topk_cache_put(sqlite3 *db, unsigned int signature, const char *seeds_json, const char *nodes_json,
                   const char *edges_json, int64_t edge_generation, int64_t prov_generation);
int topk_cache_get(sqlite3 *db, unsigned int signature, int64_t edge_generation, int64_t prov_generation,
                   char **out_seeds_json, char **out_nodes_json, char **out_edges_json);

#endif /* GRAPH_TOPK_CACHE_H */
