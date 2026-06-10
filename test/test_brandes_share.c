/*
 * test_brandes_share.c — G3 un-defer trigger threshold tests.
 *
 * Tagged for the per-gap test gate:
 *   `make test-g3` → ./build/test_runner --filter=test_g3_
 */
#include "test_common.h"
#include <math.h>

extern double muninn_brandes_share_threshold(void);

/* T3.3 — MUNINN_BRANDES_SHARE_THRESHOLD is documented and accessible.
 *
 * The constant is the un-defer gate per plan ADR (line 909): when the
 * kg_perf harness sees brandes_share = centrality / total exceeding
 * this value for 3 consecutive runs on the leading strategy, G3
 * (filter-aware Brandes) becomes worth implementing. Future un-defer
 * decisions re-derive the value from a fresh sweep
 * (benchmarks/kg_perf/sweeps/g3_brandes_share.py — T3.2).
 *
 * Test verifies:
 *   - The accessor exists and returns a valid double.
 *   - The default is 0.30 (aligned with G4's theta_full rebuild
 *     threshold; reasonable empirical inflection per project history).
 *   - Value is in (0, 1) — valid ratio range.
 */
TEST(test_g3_threshold_default_documented) {
    double threshold = muninn_brandes_share_threshold();

    /* Documented default: 0.30 (30% of pipeline time spent in
     * centrality call). */
    ASSERT(fabs(threshold - 0.30) < 1e-12);

    /* Invariant: must be a valid ratio. */
    ASSERT(threshold > 0.0);
    ASSERT(threshold < 1.0);
}

void test_brandes_share(void) {
    RUN_TEST(test_g3_threshold_default_documented);
}
