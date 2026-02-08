/*
 * test_vec_math.c — Tests for distance functions
 */
#include "test_common.h"
#include "vec_math.h"
#include <string.h>

TEST(test_l2_identical) {
    float a[] = {1.0f, 2.0f, 3.0f};
    ASSERT_EQ_FLOAT(vec_l2_distance(a, a, 3), 0.0f, 1e-7f);
}

TEST(test_l2_known) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    /* squared distance = 1 + 1 = 2 */
    ASSERT_EQ_FLOAT(vec_l2_distance(a, b, 3), 2.0f, 1e-7f);
}

TEST(test_l2_single_dim) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    ASSERT_EQ_FLOAT(vec_l2_distance(a, b, 1), 16.0f, 1e-7f);
}

TEST(test_cosine_identical) {
    float a[] = {1.0f, 2.0f, 3.0f};
    ASSERT_EQ_FLOAT(vec_cosine_distance(a, a, 3), 0.0f, 1e-6f);
}

TEST(test_cosine_orthogonal) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    ASSERT_EQ_FLOAT(vec_cosine_distance(a, b, 2), 1.0f, 1e-6f);
}

TEST(test_cosine_opposite) {
    float a[] = {1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    ASSERT_EQ_FLOAT(vec_cosine_distance(a, b, 2), 2.0f, 1e-6f);
}

TEST(test_cosine_zero_vector) {
    float a[] = {0.0f, 0.0f};
    float b[] = {1.0f, 0.0f};
    /* zero vector should return max distance */
    ASSERT_EQ_FLOAT(vec_cosine_distance(a, b, 2), 1.0f, 1e-6f);
}

TEST(test_inner_product_known) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    /* dot = 4+10+18 = 32, negated = -32 */
    ASSERT_EQ_FLOAT(vec_inner_product_distance(a, b, 3), -32.0f, 1e-6f);
}

TEST(test_inner_product_orthogonal) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    ASSERT_EQ_FLOAT(vec_inner_product_distance(a, b, 2), 0.0f, 1e-6f);
}

TEST(test_metric_lookup) {
    VecDistanceFunc f;
    f = vec_get_distance_func(VEC_METRIC_L2);
    ASSERT(f == vec_l2_distance);
    f = vec_get_distance_func(VEC_METRIC_COSINE);
    ASSERT(f == vec_cosine_distance);
    f = vec_get_distance_func(VEC_METRIC_INNER_PRODUCT);
    ASSERT(f == vec_inner_product_distance);
}

TEST(test_parse_metric) {
    VecMetric m;
    ASSERT_EQ_INT(vec_parse_metric("l2", &m), 0);
    ASSERT_EQ_INT((int)m, (int)VEC_METRIC_L2);
    ASSERT_EQ_INT(vec_parse_metric("cosine", &m), 0);
    ASSERT_EQ_INT((int)m, (int)VEC_METRIC_COSINE);
    ASSERT_EQ_INT(vec_parse_metric("inner_product", &m), 0);
    ASSERT_EQ_INT((int)m, (int)VEC_METRIC_INNER_PRODUCT);
    ASSERT_EQ_INT(vec_parse_metric("invalid", &m), -1);
}

void test_vec_math(void) {
    RUN_TEST(test_l2_identical);
    RUN_TEST(test_l2_known);
    RUN_TEST(test_l2_single_dim);
    RUN_TEST(test_cosine_identical);
    RUN_TEST(test_cosine_orthogonal);
    RUN_TEST(test_cosine_opposite);
    RUN_TEST(test_cosine_zero_vector);
    RUN_TEST(test_inner_product_known);
    RUN_TEST(test_inner_product_orthogonal);
    RUN_TEST(test_metric_lookup);
    RUN_TEST(test_parse_metric);
}
