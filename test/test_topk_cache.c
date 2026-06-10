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
extern int topk_cache_put(sqlite3 *db, unsigned int signature, const char *seeds_json, const char *nodes_json,
                          const char *edges_json, sqlite3_int64 edge_generation, sqlite3_int64 prov_generation);
extern int topk_cache_get(sqlite3 *db, unsigned int signature, sqlite3_int64 edge_generation,
                          sqlite3_int64 prov_generation, char **out_seeds_json, char **out_nodes_json,
                          char **out_edges_json);
extern sqlite3_int64 config_get_int64_public(sqlite3 *db, const char *name, const char *key, sqlite3_int64 def);

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

/* T2.2 — put/get round-trip stores and retrieves the JSON payloads.
 *
 * Lazy table creation: the cache table doesn't exist until the first
 * put. Reads on a never-written cache return SQLITE_NOTFOUND.
 *
 * Lookup keyed by (signature, edge_generation, prov_generation) —
 * the generation columns participate in the WHERE clause so future
 * T2.3 tests verify that bumping either generation invalidates the
 * row (lazy compare-on-read per the plan ADR). For T2.2 we only
 * verify the happy path: put stores, get returns. */
TEST(test_g2_cache_hit_returns_stored_rows) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Cache empty before any write — get must report NOTFOUND. */
    char *seeds = (char *)0xdead;
    char *nodes = (char *)0xdead;
    char *edges = (char *)0xdead;
    rc = topk_cache_get(db, /*sig=*/12345u, /*edge_gen=*/5, /*prov_gen=*/2, &seeds, &nodes, &edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);
    ASSERT(seeds == NULL); /* defensive: implementation must reset out-pointers. */
    ASSERT(nodes == NULL);
    ASSERT(edges == NULL);

    /* Put a row. */
    const char *seeds_in = "{\"top_k\":[\"a\",\"b\",\"c\"]}";
    const char *nodes_in = "[{\"id\":\"a\",\"degree\":3},{\"id\":\"b\",\"degree\":2}]";
    const char *edges_in = "[{\"src\":\"a\",\"dst\":\"b\",\"w\":1.0}]";
    rc = topk_cache_put(db, /*sig=*/12345u, seeds_in, nodes_in, edges_in,
                        /*edge_gen=*/5, /*prov_gen=*/2);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Get with matching generations → hit, returns stored payloads. */
    rc = topk_cache_get(db, /*sig=*/12345u, /*edge_gen=*/5, /*prov_gen=*/2, &seeds, &nodes, &edges);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT(seeds != NULL);
    ASSERT(nodes != NULL);
    ASSERT(edges != NULL);
    ASSERT_EQ_INT(0, strcmp(seeds, seeds_in));
    ASSERT_EQ_INT(0, strcmp(nodes, nodes_in));
    ASSERT_EQ_INT(0, strcmp(edges, edges_in));
    free(seeds);
    free(nodes);
    free(edges);

    /* Wrong signature → NOTFOUND. */
    rc = topk_cache_get(db, /*sig=*/99999u, /*edge_gen=*/5, /*prov_gen=*/2, &seeds, &nodes, &edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);
    ASSERT(seeds == NULL);

    /* Re-put with same signature, different payload → INSERT OR REPLACE
     * overwrites cleanly. */
    const char *seeds_in_v2 = "{\"top_k\":[\"x\",\"y\"]}";
    rc = topk_cache_put(db, /*sig=*/12345u, seeds_in_v2, nodes_in, edges_in, /*edge_gen=*/5, /*prov_gen=*/2);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = topk_cache_get(db, /*sig=*/12345u, /*edge_gen=*/5, /*prov_gen=*/2, &seeds, &nodes, &edges);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, strcmp(seeds, seeds_in_v2));
    free(seeds);
    free(nodes);
    free(edges);

    sqlite3_close(db);
}

/* T2.3 — generation invalidation on G_adj / G_prov tick. Reading with
 * either bumped generation must return NOTFOUND, even though the row
 * is still physically in the table. Property follows from T2.2's
 * WHERE-on-generations design (lazy compare-on-read); T2.3 locks it
 * in place against future regression. */
TEST(test_g2_generation_invalidation) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Plant: signature 0xABCD at (edge_gen=5, prov_gen=2). */
    const char *seeds = "[\"a\",\"b\"]";
    const char *nodes = "[]";
    const char *edges = "[]";
    rc = topk_cache_put(db, 0xABCDu, seeds, nodes, edges, /*edge_gen=*/5, /*prov_gen=*/2);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Baseline: matching generations → HIT. */
    char *out_seeds = NULL, *out_nodes = NULL, *out_edges = NULL;
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/5, /*prov_gen=*/2, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    free(out_seeds);
    free(out_nodes);
    free(out_edges);

    /* edge_generation bumped → MISS. */
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/6, /*prov_gen=*/2, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);
    ASSERT(out_seeds == NULL);

    /* prov_generation bumped → MISS. */
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/5, /*prov_gen=*/3, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);
    ASSERT(out_seeds == NULL);

    /* Both bumped → MISS. */
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/6, /*prov_gen=*/3, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);

    /* The original row is still physically present (lazy strategy —
     * the read just doesn't see it). Verified by direct SQL. */
    sqlite3_stmt *stmt = NULL;
    /* Signatures are 8-char zero-padded hex: 0xABCD → "0000abcd". */
    rc = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM _gii_topk_cache WHERE signature = '0000abcd'", -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(SQLITE_ROW, sqlite3_step(stmt));
    int physical_count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    ASSERT_EQ_INT(1, physical_count); /* row still in table — staleness-driven sweep is a separate concern. */

    /* Re-put at the new generation overwrites the stale row (PK is
     * signature alone, not (sig, gen)). Get with new generation now
     * succeeds. */
    rc = topk_cache_put(db, 0xABCDu, "[\"updated\"]", nodes, edges, /*edge_gen=*/6, /*prov_gen=*/3);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/6, /*prov_gen=*/3, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, strcmp(out_seeds, "[\"updated\"]"));
    free(out_seeds);
    free(out_nodes);
    free(out_edges);

    /* Old generation now also misses — the new row replaced the old. */
    rc = topk_cache_get(db, 0xABCDu, /*edge_gen=*/5, /*prov_gen=*/2, &out_seeds, &out_nodes, &out_edges);
    ASSERT_EQ_INT(SQLITE_NOTFOUND, rc);

    sqlite3_close(db);
}

/* qsort comparator for unsigned int — stable order so sequential-pass
 * collision detection works. */
static int compare_uint(const void *a, const void *b) {
    unsigned int ua = *(const unsigned int *)a;
    unsigned int ub = *(const unsigned int *)b;
    if (ua < ub)
        return -1;
    if (ua > ub)
        return 1;
    return 0;
}

/* T2.4 — 10K-signature uniqueness sweep. Generate 10K distinct
 * (provenance, predicate, metric, top_k, ...) tuples and assert the
 * resulting signatures collide zero times.
 *
 * With DJB2's 32-bit space, birthday-paradox expected collisions at
 * 10K is ~0.012 per run — most runs see 0, some see 1-2. If the
 * empirical count is non-zero, the primitive needs a wider hash
 * (FNV-1a-64 inline, or vendor xxh3 per the plan ADR). For T2.4 to
 * pass reliably, we need 0 collisions on every run. */
TEST(test_g2_no_collisions_over_10k) {
    const int N = 10000;
    unsigned int *sigs = (unsigned int *)malloc((size_t)N * sizeof(unsigned int));
    ASSERT(sigs != NULL);

    /* Vary one parameter (top_k) so each call hashes a distinct
     * canonical string. Other inputs constant. */
    for (int i = 0; i < N; i++) {
        sigs[i] = topk_signature("prov", "{\"k\":\"v\"}", "node_betweenness",
                                 /*top_k=*/i, /*depth=*/0, /*min_degree=*/0,
                                 /*g_adj=*/0, /*g_prov=*/0,
                                 /*community_filter=*/0, /*community_resolution=*/1.0);
    }

    /* Sort + sequential-pass collision detection. O(N log N) once,
     * O(N) for the dup check. */
    qsort(sigs, (size_t)N, sizeof(unsigned int), compare_uint);
    int collisions = 0;
    for (int i = 1; i < N; i++) {
        if (sigs[i] == sigs[i - 1]) {
            collisions++;
        }
    }
    free(sigs);

    /* Per plan ADR (line 825): the documented goal is 0 collisions at
     * 10K. If this fires, T2.4 GREEN must upgrade the hash primitive. */
    ASSERT_EQ_INT(0, collisions);
}

/* T2.5 — external generation bump must be visible immediately.
 *
 * Plan section 832 documents the trap: a long-lived prepared
 * sqlite3_stmt that reads _config.value DOES NOT re-execute when the
 * underlying row changes. A naive implementation that caches the
 * statement returns the stale snapshot forever. config_get_int64_public
 * avoids the trap by re-preparing on every call.
 *
 * Test (property-locking, like T2.3 / T2.4):
 *   - Set up a mock _config table.
 *   - Read generation = 1.
 *   - UPDATE generation = 2 (simulating a counter bump from another
 *     statement / connection / write path).
 *   - Re-read; must see 2, not 1.
 *   - Bump 50 more times in a loop; each read must see the latest. */
TEST(test_g2_external_generation_bump) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* config_get_int64_public reads from <name>_config; create one
     * for a mock VT name. */
    rc = sqlite3_exec(db,
                      "CREATE TABLE mock_config(key TEXT PRIMARY KEY, value TEXT);"
                      "INSERT INTO mock_config(key, value) VALUES ('generation', '1');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Initial read. */
    sqlite3_int64 g = config_get_int64_public(db, "mock", "generation", -1);
    ASSERT_EQ_INT(1, (int)g);

    /* External bump → read must reflect the new value, not a cached
     * snapshot. */
    rc = sqlite3_exec(db, "UPDATE mock_config SET value = '2' WHERE key = 'generation'", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    g = config_get_int64_public(db, "mock", "generation", -1);
    ASSERT_EQ_INT(2, (int)g);

    /* Stress: 50 bump-and-read cycles, every read must be fresh. */
    for (int i = 3; i <= 52; i++) {
        char *upd = sqlite3_mprintf("UPDATE mock_config SET value = '%d' WHERE key = 'generation'", i);
        rc = sqlite3_exec(db, upd, NULL, NULL, NULL);
        sqlite3_free(upd);
        ASSERT_EQ_INT(SQLITE_OK, rc);
        g = config_get_int64_public(db, "mock", "generation", -1);
        ASSERT_EQ_INT(i, (int)g);
    }

    /* Default-on-missing: deleting the row must restore the default
     * sentinel. Future code that relies on the -1 sentinel for
     * 'never-computed' state needs this to keep working. */
    rc = sqlite3_exec(db, "DELETE FROM mock_config WHERE key = 'generation'", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    g = config_get_int64_public(db, "mock", "generation", -1);
    ASSERT_EQ_INT(-1, (int)g);

    sqlite3_close(db);
}

void test_topk_cache(void) {
    RUN_TEST(test_g2_signature_stable_under_json_reordering);
    RUN_TEST(test_g2_cache_hit_returns_stored_rows);
    RUN_TEST(test_g2_generation_invalidation);
    RUN_TEST(test_g2_no_collisions_over_10k);
    RUN_TEST(test_g2_external_generation_bump);
}
