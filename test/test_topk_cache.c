/*
 * test_topk_cache.c — Top-K result cache tests (G2).
 *
 * Tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g2` → ./build/test_runner --filter=test_g2_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric,
                                   int top_k, int depth, int min_degree, sqlite3_int64 g_adj, sqlite3_int64 g_prov,
                                   int community_filter, double community_resolution);

/* T2.1 — JSON canonicalization makes the signature stable under key
 * reordering. Same logical content with different key order must
 * produce the same hash; otherwise cosmetic JSON drift busts the
 * cache and serves stale results.
 *
 * Same numerical content, different reordering — three orderings
 * of the same 3-key object. All must hash identically. */
TEST(test_g2_signature_stable_under_json_reordering) {
    const char *prov = "_gii_provenance";
    const char *metric = "node_betweenness";
    const sqlite3_int64 g_adj = 42;
    const sqlite3_int64 g_prov = 17;

    const char *order_a = "{\"days\":7,\"min_degree\":3,\"project\":\"acme\"}";
    const char *order_b = "{\"project\":\"acme\",\"days\":7,\"min_degree\":3}";
    const char *order_c = "{\"min_degree\":3,\"project\":\"acme\",\"days\":7}";

    unsigned int sig_a = topk_signature(prov, order_a, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    unsigned int sig_b = topk_signature(prov, order_b, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    unsigned int sig_c = topk_signature(prov, order_c, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);

    /* All three reorderings → identical signature. */
    ASSERT_EQ_INT((int)sig_a, (int)sig_b);
    ASSERT_EQ_INT((int)sig_a, (int)sig_c);

    /* Sanity: different content (different value for "days") still
     * produces a different signature. Canonicalization must NOT
     * collapse semantically-distinct queries. */
    const char *order_a_modified = "{\"days\":30,\"min_degree\":3,\"project\":\"acme\"}";
    unsigned int sig_a_modified = topk_signature(prov, order_a_modified, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    ASSERT(sig_a != sig_a_modified);

    /* Nested object — sorting must recurse. */
    const char *nested_a = "{\"window\":{\"days\":7,\"start\":\"2026-05-01\"},\"project\":\"acme\"}";
    const char *nested_b = "{\"project\":\"acme\",\"window\":{\"start\":\"2026-05-01\",\"days\":7}}";
    unsigned int nested_sig_a = topk_signature(prov, nested_a, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    unsigned int nested_sig_b = topk_signature(prov, nested_b, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    ASSERT_EQ_INT((int)nested_sig_a, (int)nested_sig_b);

    /* Arrays are NOT reordered — order is semantically meaningful. */
    const char *arr_a = "{\"tags\":[\"a\",\"b\",\"c\"]}";
    const char *arr_b = "{\"tags\":[\"c\",\"b\",\"a\"]}";
    unsigned int arr_sig_a = topk_signature(prov, arr_a, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    unsigned int arr_sig_b = topk_signature(prov, arr_b, metric, 100, 2, 3, g_adj, g_prov, 0, 1.0);
    ASSERT(arr_sig_a != arr_sig_b);
}

void test_topk_cache(void) {
    RUN_TEST(test_g2_signature_stable_under_json_reordering);
}
