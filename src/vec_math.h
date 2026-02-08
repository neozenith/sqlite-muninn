/*
 * vec_math.h — Vector distance functions for HNSW
 *
 * Supports L2 (squared Euclidean), cosine, and inner product distances.
 * All functions operate on float32 arrays and return smaller values for
 * "more similar" vectors, enabling a uniform min-heap interface.
 */
#ifndef VEC_MATH_H
#define VEC_MATH_H

#include <stdint.h>

typedef enum {
    VEC_METRIC_L2 = 0,
    VEC_METRIC_COSINE = 1,
    VEC_METRIC_INNER_PRODUCT = 2
} VecMetric;

/* Distance function signature: (a, b, dimensions) -> distance */
typedef float (*VecDistanceFunc)(const float *a, const float *b, int dim);

/* Squared Euclidean distance (no sqrt — monotonic for ranking) */
float vec_l2_distance(const float *a, const float *b, int dim);

/* Cosine distance: 1 - cos(a, b). Returns 0 for identical, 2 for opposite. */
float vec_cosine_distance(const float *a, const float *b, int dim);

/* Negative inner product (negated so smaller = more similar) */
float vec_inner_product_distance(const float *a, const float *b, int dim);

/* Look up distance function by metric enum */
VecDistanceFunc vec_get_distance_func(VecMetric metric);

/* Parse metric name string. Returns -1 on invalid input. */
int vec_parse_metric(const char *name, VecMetric *out);

#endif /* VEC_MATH_H */
