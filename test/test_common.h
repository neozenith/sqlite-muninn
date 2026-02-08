/*
 * test_common.h — Shared test macros
 */
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <math.h>

extern void test_begin(const char *name);
extern void test_pass(void);
extern void test_fail(const char *file, int line, const char *expr);
extern int test_failed_flag(void);

#define ASSERT(expr) do { \
    if (!(expr)) { test_fail(__FILE__, __LINE__, #expr); return; } \
} while (0)

#define ASSERT_EQ_INT(a, b) do { \
    int _a = (a), _b = (b); \
    if (_a != _b) { \
        test_fail(__FILE__, __LINE__, #a " == " #b); \
        return; \
    } \
} while (0)

#define ASSERT_EQ_FLOAT(a, b, eps) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (eps)) { \
        test_fail(__FILE__, __LINE__, #a " ≈ " #b); \
        return; \
    } \
} while (0)

#define RUN_TEST(fn) do { test_begin(#fn); fn(); } while (0)

/* Wrapper that calls test_pass() only if no ASSERT failed */
#define TEST(fn) static void fn##_impl(void); \
    static void fn(void) { fn##_impl(); if (!test_failed_flag()) test_pass(); } \
    static void fn##_impl(void)

#endif /* TEST_COMMON_H */
