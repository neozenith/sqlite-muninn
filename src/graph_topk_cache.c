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

/* T7.4 RED stub: hashes everything EXCEPT community_filter and
 * community_resolution. T7.4 GREEN extends the canonical-string
 * builder to include both. */
unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric, int top_k,
                            int depth, int min_degree, int64_t g_adj, int64_t g_prov, int community_filter,
                            double community_resolution) {
    (void)community_filter;
    (void)community_resolution;

    char buf[512];
    snprintf(buf, sizeof(buf), "%s|%s|%s|%d|%d|%d|%lld|%lld", provenance_table ? provenance_table : "",
             filter_predicate ? filter_predicate : "", metric ? metric : "", top_k, depth, min_degree, (long long)g_adj,
             (long long)g_prov);
    return graph_str_hash(buf);
}
