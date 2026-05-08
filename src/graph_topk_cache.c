/*
 * graph_topk_cache.c — Top-K result cache TVF (G2)
 *
 * Currently only implements the signature primitive (G7 T7.4
 * forward-compat). The full cache machinery — xCreate, lookup,
 * write-back, generation invalidation — lands when G2's tickets ship.
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_topk_cache.h"
#include "graph_common.h"

#include <stdio.h>
#include <string.h>

/* Canonical-string-then-hash signature.
 *
 * Every parameter that would change the output joins the canonical
 * string with a '|' separator; community_filter and
 * community_resolution participate at the end (G7 T7.4) so cached
 * top-K rows for different communities don't collide.
 *
 * community_resolution uses %.17g (IEEE 754 binary64 round-trip
 * precision) so a re-issued query with the same gamma — even one
 * that round-tripped through the user's frontend with float-edge
 * precision — produces the same signature. Lower precision (%g, %f)
 * would silently corrupt cache lookups for callers that re-use the
 * exact double they cached at.
 *
 * Hash primitive is DJB2 via graph_str_hash for now; G2 T2.1 swaps
 * this to xxh3 with proper JSON canonicalization of filter_predicate.
 * The contract under test in T7.4 is "community fields influence the
 * output," not collision strength (T2.4's territory).
 */
unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric, int top_k,
                            int depth, int min_degree, int64_t g_adj, int64_t g_prov, int community_filter,
                            double community_resolution) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s|%s|%s|%d|%d|%d|%lld|%lld|%d|%.17g", provenance_table ? provenance_table : "",
             filter_predicate ? filter_predicate : "", metric ? metric : "", top_k, depth, min_degree, (long long)g_adj,
             (long long)g_prov, community_filter, community_resolution);
    return graph_str_hash(buf);
}
