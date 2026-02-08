/*
 * test_hnsw_algo.c — Tests for core HNSW algorithms
 */
#include "test_common.h"
#include "hnsw_algo.h"
#include <stdlib.h>
#include <string.h>

/* Helper: brute force KNN for comparison */
static void brute_force_knn(const float *query, const float *vectors, int n, int dim,
                             VecDistanceFunc dist_func, int k,
                             int64_t *out_ids, float *out_dists) {
    /* Simple O(nk) selection */
    float *dists = (float *)malloc((size_t)n * sizeof(float));
    int *used = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < n; i++) {
        dists[i] = dist_func(query, vectors + i * dim, dim);
    }
    for (int j = 0; j < k && j < n; j++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            if (best < 0 || dists[i] < dists[best]) best = i;
        }
        if (best >= 0) {
            out_ids[j] = best;
            out_dists[j] = dists[best];
            used[best] = 1;
        }
    }
    free(dists);
    free(used);
}

TEST(test_hnsw_create_destroy) {
    HnswIndex *idx = hnsw_create(4, VEC_METRIC_L2, 16, 200);
    ASSERT(idx != NULL);
    ASSERT_EQ_INT(idx->dim, 4);
    ASSERT_EQ_INT(idx->M, 16);
    ASSERT_EQ_INT(idx->node_count, 0);
    hnsw_destroy(idx);
}

TEST(test_hnsw_insert_one) {
    HnswIndex *idx = hnsw_create(3, VEC_METRIC_L2, 4, 10);
    float v[] = {1.0f, 2.0f, 3.0f};

    ASSERT_EQ_INT(hnsw_insert(idx, 42, v), 0);
    ASSERT_EQ_INT(idx->node_count, 1);
    ASSERT(idx->entry_point == 42);

    const float *got = hnsw_get_vector(idx, 42);
    ASSERT(got != NULL);
    ASSERT_EQ_FLOAT(got[0], 1.0f, 1e-7f);
    ASSERT_EQ_FLOAT(got[1], 2.0f, 1e-7f);
    ASSERT_EQ_FLOAT(got[2], 3.0f, 1e-7f);

    hnsw_destroy(idx);
}

TEST(test_hnsw_insert_duplicate) {
    HnswIndex *idx = hnsw_create(3, VEC_METRIC_L2, 4, 10);
    float v[] = {1.0f, 2.0f, 3.0f};

    ASSERT_EQ_INT(hnsw_insert(idx, 1, v), 0);
    ASSERT_EQ_INT(hnsw_insert(idx, 1, v), -1);  /* duplicate should fail */

    hnsw_destroy(idx);
}

TEST(test_hnsw_search_exact) {
    HnswIndex *idx = hnsw_create(2, VEC_METRIC_L2, 4, 20);
    hnsw_seed_rng(idx, 12345);

    /* Insert 3 distinct points */
    float v0[] = {0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f};
    float v2[] = {0.0f, 1.0f};
    hnsw_insert(idx, 0, v0);
    hnsw_insert(idx, 1, v1);
    hnsw_insert(idx, 2, v2);

    /* Search for nearest to origin — should be v0 */
    float query[] = {0.1f, 0.1f};
    HnswSearchResult results[3];
    int found = hnsw_search(idx, query, 3, 10, results);

    ASSERT_EQ_INT(found, 3);
    /* Closest should be node 0 */
    ASSERT(results[0].id == 0);

    hnsw_destroy(idx);
}

TEST(test_hnsw_knn_recall_small) {
    /* Insert 50 random 8-dim vectors, verify top-5 matches brute force */
    int n = 50, dim = 8, k = 5;
    HnswIndex *idx = hnsw_create(dim, VEC_METRIC_L2, 8, 50);
    hnsw_seed_rng(idx, 42);

    float *vectors = (float *)malloc((size_t)(n * dim) * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < n * dim; i++) {
        /* Simple LCG for deterministic test data */
        seed = seed * 1103515245 + 12345;
        vectors[i] = (float)(seed % 10000) / 10000.0f;
    }

    for (int i = 0; i < n; i++) {
        ASSERT_EQ_INT(hnsw_insert(idx, i, vectors + i * dim), 0);
    }

    /* Search */
    float query[8];
    seed = 9999;
    for (int i = 0; i < dim; i++) {
        seed = seed * 1103515245 + 12345;
        query[i] = (float)(seed % 10000) / 10000.0f;
    }

    HnswSearchResult hnsw_results[5];
    int found = hnsw_search(idx, query, k, 64, hnsw_results);
    ASSERT_EQ_INT(found, k);

    /* Brute force comparison */
    int64_t bf_ids[5];
    float bf_dists[5];
    brute_force_knn(query, vectors, n, dim, vec_l2_distance, k, bf_ids, bf_dists);

    /* Check recall: how many of HNSW top-5 are in brute-force top-5 */
    int recall = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (hnsw_results[i].id == bf_ids[j]) { recall++; break; }
        }
    }
    ASSERT(recall >= 4);  /* at least 80% recall */

    free(vectors);
    hnsw_destroy(idx);
}

TEST(test_hnsw_delete) {
    HnswIndex *idx = hnsw_create(2, VEC_METRIC_L2, 4, 10);
    hnsw_seed_rng(idx, 100);

    float v0[] = {0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f};
    float v2[] = {2.0f, 0.0f};
    hnsw_insert(idx, 0, v0);
    hnsw_insert(idx, 1, v1);
    hnsw_insert(idx, 2, v2);

    ASSERT_EQ_INT(hnsw_delete(idx, 1), 0);
    ASSERT(hnsw_get_node(idx, 1) == NULL);  /* deleted */

    /* Search should not return deleted node */
    float query[] = {1.0f, 0.0f};  /* closest to deleted v1 */
    HnswSearchResult results[2];
    int found = hnsw_search(idx, query, 2, 10, results);
    ASSERT_EQ_INT(found, 2);
    /* Neither result should be node 1 */
    ASSERT(results[0].id != 1);
    ASSERT(results[1].id != 1);

    hnsw_destroy(idx);
}

TEST(test_hnsw_search_empty) {
    HnswIndex *idx = hnsw_create(2, VEC_METRIC_L2, 4, 10);
    float query[] = {0.0f, 0.0f};
    HnswSearchResult results[5];
    int found = hnsw_search(idx, query, 5, 10, results);
    ASSERT_EQ_INT(found, 0);
    hnsw_destroy(idx);
}

TEST(test_hnsw_cosine_metric) {
    HnswIndex *idx = hnsw_create(2, VEC_METRIC_COSINE, 4, 10);
    hnsw_seed_rng(idx, 55);

    /* Insert unit-ish vectors in different directions */
    float v0[] = {1.0f, 0.0f};   /* right */
    float v1[] = {0.0f, 1.0f};   /* up */
    float v2[] = {-1.0f, 0.0f};  /* left (opposite of right) */
    hnsw_insert(idx, 0, v0);
    hnsw_insert(idx, 1, v1);
    hnsw_insert(idx, 2, v2);

    /* Query slightly right — should match v0 first */
    float query[] = {0.9f, 0.1f};
    HnswSearchResult results[3];
    int found = hnsw_search(idx, query, 3, 10, results);
    ASSERT_EQ_INT(found, 3);
    ASSERT(results[0].id == 0);  /* right is closest direction */

    hnsw_destroy(idx);
}

void test_hnsw_algo(void) {
    RUN_TEST(test_hnsw_create_destroy);
    RUN_TEST(test_hnsw_insert_one);
    RUN_TEST(test_hnsw_insert_duplicate);
    RUN_TEST(test_hnsw_search_exact);
    RUN_TEST(test_hnsw_knn_recall_small);
    RUN_TEST(test_hnsw_delete);
    RUN_TEST(test_hnsw_search_empty);
    RUN_TEST(test_hnsw_cosine_metric);
}
