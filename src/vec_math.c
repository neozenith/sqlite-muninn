/*
 * vec_math.c — Vector distance functions for HNSW
 *
 * Provides L2, cosine, and inner product distance with SIMD acceleration.
 * SIMD paths: ARM NEON (Apple Silicon, ARM64), x86 SSE (Intel/AMD).
 * Falls back to portable scalar code when no SIMD is available.
 */
#include "vec_math.h"
#include <math.h>
#include <string.h>

/* ─── ARM NEON ──────────────────────────────────────────────── */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

float vec_l2_distance(const float *a, const float *b, int dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vmlaq_f32(sum_vec, diff, diff);
    }
    float sum = vaddvq_f32(sum_vec);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float vec_cosine_distance(const float *a, const float *b, int dim) {
    float32x4_t dot_vec = vdupq_n_f32(0.0f);
    float32x4_t na_vec  = vdupq_n_f32(0.0f);
    float32x4_t nb_vec  = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot_vec = vmlaq_f32(dot_vec, va, vb);
        na_vec  = vmlaq_f32(na_vec, va, va);
        nb_vec  = vmlaq_f32(nb_vec, vb, vb);
    }
    float dot    = vaddvq_f32(dot_vec);
    float norm_a = vaddvq_f32(na_vec);
    float norm_b = vaddvq_f32(nb_vec);
    for (; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-30f) return 1.0f;
    return 1.0f - (dot / denom);
}

float vec_inner_product_distance(const float *a, const float *b, int dim) {
    float32x4_t dot_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot_vec = vmlaq_f32(dot_vec, va, vb);
    }
    float dot = vaddvq_f32(dot_vec);
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;
}

/* ─── x86 SSE ───────────────────────────────────────────────── */
#elif defined(__SSE__) || defined(__SSE2__)
#include <xmmintrin.h>

float vec_l2_distance(const float *a, const float *b, int dim) {
    __m128 sum_vec = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
    }
    /* Horizontal sum of 4 floats */
    float tmp[4];
    _mm_storeu_ps(tmp, sum_vec);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float vec_cosine_distance(const float *a, const float *b, int dim) {
    __m128 dot_vec = _mm_setzero_ps();
    __m128 na_vec  = _mm_setzero_ps();
    __m128 nb_vec  = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(va, vb));
        na_vec  = _mm_add_ps(na_vec, _mm_mul_ps(va, va));
        nb_vec  = _mm_add_ps(nb_vec, _mm_mul_ps(vb, vb));
    }
    float td[4], ta[4], tb[4];
    _mm_storeu_ps(td, dot_vec);
    _mm_storeu_ps(ta, na_vec);
    _mm_storeu_ps(tb, nb_vec);
    float dot    = td[0] + td[1] + td[2] + td[3];
    float norm_a = ta[0] + ta[1] + ta[2] + ta[3];
    float norm_b = tb[0] + tb[1] + tb[2] + tb[3];
    for (; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-30f) return 1.0f;
    return 1.0f - (dot / denom);
}

float vec_inner_product_distance(const float *a, const float *b, int dim) {
    __m128 dot_vec = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, dot_vec);
    float dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;
}

/* ─── Scalar fallback ───────────────────────────────────────── */
#else

float vec_l2_distance(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float vec_cosine_distance(const float *a, const float *b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-30f) return 1.0f;
    return 1.0f - (dot / denom);
}

float vec_inner_product_distance(const float *a, const float *b, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;
}

#endif /* SIMD selection */

VecDistanceFunc vec_get_distance_func(VecMetric metric) {
    switch (metric) {
        case VEC_METRIC_L2:             return vec_l2_distance;
        case VEC_METRIC_COSINE:         return vec_cosine_distance;
        case VEC_METRIC_INNER_PRODUCT:  return vec_inner_product_distance;
    }
    return vec_l2_distance;  /* unreachable if enum is correct */
}

int vec_parse_metric(const char *name, VecMetric *out) {
    if (strcmp(name, "l2") == 0) {
        *out = VEC_METRIC_L2;
        return 0;
    } else if (strcmp(name, "cosine") == 0) {
        *out = VEC_METRIC_COSINE;
        return 0;
    } else if (strcmp(name, "inner_product") == 0) {
        *out = VEC_METRIC_INNER_PRODUCT;
        return 0;
    }
    return -1;
}
