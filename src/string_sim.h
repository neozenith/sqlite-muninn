/*
 * string_sim.h — String similarity functions
 *
 * Provides Jaro-Winkler similarity for entity resolution matching.
 */
#ifndef STRING_SIM_H
#define STRING_SIM_H

/*
 * Jaro-Winkler string similarity (0.0 to 1.0).
 *
 * Returns 1.0 for identical strings, 0.0 for completely different strings.
 * The Winkler prefix bonus (up to 4 chars) rewards strings that share a
 * common prefix.
 */
double jaro_winkler(const char *s1, const char *s2);

#endif /* STRING_SIM_H */
