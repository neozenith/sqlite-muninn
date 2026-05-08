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

#endif /* GRAPH_TOPK_CACHE_H */
