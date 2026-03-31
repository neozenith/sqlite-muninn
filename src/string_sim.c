/*
 * string_sim.c — String similarity functions
 *
 * Jaro-Winkler similarity for entity resolution matching cascade.
 */
#include "string_sim.h"

#include <stdlib.h>
#include <string.h>

double jaro_winkler(const char *s1, const char *s2) {
    int len1 = (int)strlen(s1);
    int len2 = (int)strlen(s2);

    if (len1 == 0 && len2 == 0)
        return 1.0;
    if (len1 == 0 || len2 == 0)
        return 0.0;
    if (strcmp(s1, s2) == 0)
        return 1.0;

    int match_dist = (len1 > len2 ? len1 : len2) / 2 - 1;
    if (match_dist < 0)
        match_dist = 0;

    /* Stack-allocate for small strings, heap for large */
    int stack1[256], stack2[256];
    int *s1_matched = len1 <= 256 ? stack1 : (int *)calloc((size_t)len1, sizeof(int));
    int *s2_matched = len2 <= 256 ? stack2 : (int *)calloc((size_t)len2, sizeof(int));
    if (len1 <= 256)
        memset(s1_matched, 0, (size_t)len1 * sizeof(int));
    if (len2 <= 256)
        memset(s2_matched, 0, (size_t)len2 * sizeof(int));

    int matches = 0;
    int transpositions = 0;

    for (int i = 0; i < len1; i++) {
        int lo = i - match_dist;
        if (lo < 0)
            lo = 0;
        int hi = i + match_dist + 1;
        if (hi > len2)
            hi = len2;
        for (int j = lo; j < hi; j++) {
            if (s2_matched[j] || s1[i] != s2[j])
                continue;
            s1_matched[i] = 1;
            s2_matched[j] = 1;
            matches++;
            break;
        }
    }

    if (matches == 0) {
        if (len1 > 256)
            free(s1_matched);
        if (len2 > 256)
            free(s2_matched);
        return 0.0;
    }

    /* Count transpositions */
    int k = 0;
    for (int i = 0; i < len1; i++) {
        if (!s1_matched[i])
            continue;
        while (!s2_matched[k])
            k++;
        if (s1[i] != s2[k])
            transpositions++;
        k++;
    }

    if (len1 > 256)
        free(s1_matched);
    if (len2 > 256)
        free(s2_matched);

    double jaro = ((double)matches / len1 + (double)matches / len2 +
                   ((double)matches - transpositions / 2.0) / matches) /
                  3.0;

    /* Winkler prefix bonus (up to 4 chars) */
    int prefix = 0;
    int max_prefix = len1 < len2 ? len1 : len2;
    if (max_prefix > 4)
        max_prefix = 4;
    for (int i = 0; i < max_prefix; i++) {
        if (s1[i] == s2[i])
            prefix++;
        else
            break;
    }

    return jaro + prefix * 0.1 * (1.0 - jaro);
}
