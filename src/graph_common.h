/*
 * graph_common.h — Shared utilities for graph TVF modules.
 *
 * Contains common helpers used across graph_tvf.c, graph_load.c,
 * graph_centrality.c, and graph_community.c. Extracted to avoid
 * static function name collisions in the single-file amalgamation.
 *
 * Include AFTER sqlite3ext.h in each .c file. The sqlite3_api pointer
 * is forward-declared here so the SQLite extension macros resolve correctly.
 */
#ifndef GRAPH_COMMON_H
#define GRAPH_COMMON_H

#include "sqlite3ext.h"

#include <stddef.h>

/* Forward-declare the extension API pointer (also declared by SQLITE_EXTENSION_INIT3).
 * Multiple extern declarations of the same symbol are valid C. */
#ifndef SQLITE_CORE
extern const sqlite3_api_routines *sqlite3_api;
#endif

/* Suppress unused-function warnings: each file uses a subset of these. */
#if defined(__GNUC__) || defined(__clang__)
#define GRAPH_UNUSED __attribute__((unused))
#else
#define GRAPH_UNUSED
#endif

/* ── DJB2 string hash ────────────────────────────────────── */

static GRAPH_UNUSED unsigned int graph_str_hash(const char *s) {
    unsigned int h = 5381;
    for (; *s; s++) h = h * 33 + (unsigned char)*s;
    return h;
}

/* ── Safe text extraction from sqlite3_value ─────────────── */

static GRAPH_UNUSED const char *graph_safe_text(sqlite3_value *v) {
    if (!v || sqlite3_value_type(v) == SQLITE_NULL) return NULL;
    return (const char *)sqlite3_value_text(v);
}

/*
 * Two-pass xBestIndex helper for TVFs with optional hidden columns.
 *
 * SQLite requires contiguous argvIndex values (1, 2, 3, ...) with no gaps.
 * When optional hidden params are omitted, column-based argvIndex creates gaps.
 *
 * Pass 1: Scan constraints and record which hidden columns have EQ constraints.
 * Pass 2: Assign sequential argvIndex values in column order.
 *
 * This guarantees argv[] is always in column order in xFilter, and
 * idxNum bitmask tells which columns are present.
 *
 * good_cost: estimatedCost when required columns are present.
 */
static GRAPH_UNUSED int graph_best_index_common(sqlite3_index_info *pIdxInfo,
                                   int first_hidden, int last_hidden,
                                   int required_mask, double good_cost) {
    int n_cols = last_hidden - first_hidden + 1;

    /* Pass 1: map each hidden column to its constraint index (or -1) */
    int constraint_for[32];  /* max 32 hidden columns */
    for (int j = 0; j < n_cols && j < 32; j++) constraint_for[j] = -1;

    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
        if (!pIdxInfo->aConstraint[i].usable) continue;
        if (pIdxInfo->aConstraint[i].op != SQLITE_INDEX_CONSTRAINT_EQ) continue;
        int col = pIdxInfo->aConstraint[i].iColumn;
        if (col >= first_hidden && col <= last_hidden) {
            constraint_for[col - first_hidden] = i;
        }
    }

    /* Pass 2: assign sequential argvIndex in column order */
    int argv_idx = 1;
    int idx_num = 0;
    for (int j = 0; j < n_cols && j < 32; j++) {
        if (constraint_for[j] >= 0) {
            pIdxInfo->aConstraintUsage[constraint_for[j]].argvIndex = argv_idx++;
            pIdxInfo->aConstraintUsage[constraint_for[j]].omit = 1;
            idx_num |= (1 << j);
        }
    }

    pIdxInfo->idxNum = idx_num;
    pIdxInfo->estimatedCost = ((idx_num & required_mask) == required_mask)
                                  ? good_cost
                                  : 1e12;
    return SQLITE_OK;
}

#endif /* GRAPH_COMMON_H */
