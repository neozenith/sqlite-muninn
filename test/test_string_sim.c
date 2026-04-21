/*
 * test_string_sim.c — Tests for Jaro-Winkler string similarity
 */
#include "test_common.h"
#include "string_sim.h"

#define ASSERT_JW(s1, s2, expected, eps)                                                                               \
    do {                                                                                                               \
        double _jw = jaro_winkler(s1, s2);                                                                             \
        if (fabs(_jw - (expected)) > (eps)) {                                                                          \
            test_fail(__FILE__, __LINE__, #s1 " ~ " #s2 " ≈ " #expected);                                              \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

TEST(test_jw_identical) {
    ASSERT_JW("hello", "hello", 1.0, 0.001);
}

TEST(test_jw_empty) {
    ASSERT_JW("", "", 1.0, 0.001);
    ASSERT_JW("hello", "", 0.0, 0.001);
    ASSERT_JW("", "hello", 0.0, 0.001);
}

TEST(test_jw_completely_different) {
    ASSERT_JW("abc", "xyz", 0.0, 0.001);
}

TEST(test_jw_similar_strings) {
    /* "martha" vs "marhta" — classic Jaro example */
    double jw = jaro_winkler("martha", "marhta");
    ASSERT(jw > 0.95); /* Should be ~0.961 */
    ASSERT(jw <= 1.0);
}

TEST(test_jw_prefix_bonus) {
    /* Winkler prefix bonus: shared prefix increases score */
    double no_prefix = jaro_winkler("abcxyz", "abcxzy");
    double with_prefix = jaro_winkler("abcxyz", "abcxzy");
    /* Both have "abc" prefix — bonus applies */
    ASSERT(with_prefix > 0.9);
    (void)no_prefix;
}

TEST(test_jw_product_names) {
    /* Entity resolution use case: product name variants */
    double jw1 = jaro_winkler("sony vaio vpceb15fm", "sony vaio vpc-eb15fm");
    ASSERT(jw1 > 0.9);

    double jw2 = jaro_winkler("canon powershot sd1200", "canon sd1200 powershot");
    ASSERT(jw2 > 0.7);

    /* Completely different products should be low */
    double jw3 = jaro_winkler("sony vaio laptop", "canon powershot camera");
    ASSERT(jw3 < 0.7);
}

TEST(test_jw_symmetry) {
    /* Jaro-Winkler should be symmetric */
    double ab = jaro_winkler("hello", "hallo");
    double ba = jaro_winkler("hallo", "hello");
    ASSERT(fabs(ab - ba) < 0.0001);
}

TEST(test_jw_single_char) {
    ASSERT_JW("a", "a", 1.0, 0.001);
    ASSERT_JW("a", "b", 0.0, 0.001);
}

void test_string_sim(void) {
    RUN_TEST(test_jw_identical);
    RUN_TEST(test_jw_empty);
    RUN_TEST(test_jw_completely_different);
    RUN_TEST(test_jw_similar_strings);
    RUN_TEST(test_jw_prefix_bonus);
    RUN_TEST(test_jw_product_names);
    RUN_TEST(test_jw_symmetry);
    RUN_TEST(test_jw_single_char);
}
