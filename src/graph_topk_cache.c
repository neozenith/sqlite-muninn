/*
 * graph_topk_cache.c — Top-K result cache TVF (G2)
 *
 * Currently only implements the signature primitive (G7 T7.4
 * forward-compat). The full cache machinery — xCreate, lookup,
 * write-back, generation invalidation — lands when G2's tickets ship.
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "graph_topk_cache.h"
#include "graph_common.h"
#include "yyjson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Recursive canonicalizer: parse src into a yyjson_doc, walk the tree
 * depth-first, sort each object's keys alphabetically, build a new
 * mutable doc, serialize. Arrays preserve order — only objects sort.
 *
 * Returns a malloc'd string (caller frees). NULL on parse failure or
 * allocation failure. Empty/NULL input passes through unchanged so
 * callers don't need pre-checks. */
typedef struct {
    const char *key;
    yyjson_val *value;
} sortable_kv;

static int compare_keys(const void *a, const void *b) {
    const sortable_kv *ka = (const sortable_kv *)a;
    const sortable_kv *kb = (const sortable_kv *)b;
    return strcmp(ka->key, kb->key);
}

static yyjson_mut_val *canonicalize_into(yyjson_val *src, yyjson_mut_doc *mut) {
    yyjson_type t = yyjson_get_type(src);
    if (t == YYJSON_TYPE_OBJ) {
        size_t n = yyjson_obj_size(src);
        sortable_kv *kvs = NULL;
        if (n > 0) {
            kvs = (sortable_kv *)malloc(n * sizeof(sortable_kv));
            if (!kvs)
                return NULL;
        }
        yyjson_obj_iter iter;
        yyjson_obj_iter_init(src, &iter);
        size_t i = 0;
        yyjson_val *k;
        while ((k = yyjson_obj_iter_next(&iter)) != NULL && i < n) {
            kvs[i].key = yyjson_get_str(k);
            kvs[i].value = yyjson_obj_iter_get_val(k);
            i++;
        }
        if (n > 0) {
            qsort(kvs, n, sizeof(sortable_kv), compare_keys);
        }
        yyjson_mut_val *obj = yyjson_mut_obj(mut);
        for (i = 0; i < n; i++) {
            yyjson_mut_val *child = canonicalize_into(kvs[i].value, mut);
            if (!child) {
                free(kvs);
                return NULL;
            }
            yyjson_mut_obj_add(obj, yyjson_mut_strcpy(mut, kvs[i].key), child);
        }
        free(kvs);
        return obj;
    }
    if (t == YYJSON_TYPE_ARR) {
        yyjson_mut_val *arr = yyjson_mut_arr(mut);
        yyjson_arr_iter iter;
        yyjson_arr_iter_init(src, &iter);
        yyjson_val *v;
        while ((v = yyjson_arr_iter_next(&iter)) != NULL) {
            yyjson_mut_val *child = canonicalize_into(v, mut);
            if (!child)
                return NULL;
            yyjson_mut_arr_append(arr, child);
        }
        return arr;
    }
    /* Scalar leaf — deep-copy as-is. */
    return yyjson_val_mut_copy(mut, src);
}

static char *canonicalize_json(const char *src) {
    if (!src || !*src) {
        return NULL;
    }
    yyjson_doc *doc = yyjson_read(src, strlen(src), 0);
    if (!doc) {
        return NULL;
    }
    yyjson_mut_doc *mut = yyjson_mut_doc_new(NULL);
    if (!mut) {
        yyjson_doc_free(doc);
        return NULL;
    }
    yyjson_mut_val *root = canonicalize_into(yyjson_doc_get_root(doc), mut);
    if (!root) {
        yyjson_mut_doc_free(mut);
        yyjson_doc_free(doc);
        return NULL;
    }
    yyjson_mut_doc_set_root(mut, root);

    size_t out_len = 0;
    char *yy_canonical = yyjson_mut_write(mut, 0, &out_len);
    char *result = NULL;
    if (yy_canonical) {
        /* yyjson allocates with its own allocator; copy to malloc so
         * caller can use plain free(). */
        result = (char *)malloc(out_len + 1);
        if (result) {
            memcpy(result, yy_canonical, out_len);
            result[out_len] = '\0';
        }
        free(yy_canonical);
    }
    yyjson_mut_doc_free(mut);
    yyjson_doc_free(doc);
    return result;
}

/* Canonical-string-then-hash signature.
 *
 * Every parameter that would change the output joins the canonical
 * string with a '|' separator; community_filter and
 * community_resolution participate at the end (G7 T7.4) so cached
 * top-K rows for different communities don't collide.
 *
 * community_resolution uses %.17g (IEEE 754 binary64 round-trip
 * precision) so a re-issued query with the same gamma — even one
 * that round-tripped through the user's frontend with float-edge
 * precision — produces the same signature. Lower precision (%g, %f)
 * would silently corrupt cache lookups for callers that re-use the
 * exact double they cached at.
 *
 * Hash primitive is DJB2 via graph_str_hash for now; G2 T2.1 swaps
 * this to xxh3 with proper JSON canonicalization of filter_predicate.
 * The contract under test in T7.4 is "community fields influence the
 * output," not collision strength (T2.4's territory).
 */
unsigned int topk_signature(const char *provenance_table, const char *filter_predicate, const char *metric, int top_k,
                            int depth, int min_degree, int64_t g_adj, int64_t g_prov, int community_filter,
                            double community_resolution) {
    /* G2 T2.1: canonicalize filter_predicate JSON before hashing so
     * cosmetic key reorderings don't bust the cache. yyjson roundtrip
     * sorts each object's keys alphabetically and preserves array
     * order. NULL / non-JSON input falls through to the original
     * string (still hashable; just no canonicalization benefit). */
    char *canonical = NULL;
    const char *predicate_for_hash = filter_predicate ? filter_predicate : "";
    if (filter_predicate && *filter_predicate) {
        canonical = canonicalize_json(filter_predicate);
        if (canonical) {
            predicate_for_hash = canonical;
        }
    }

    char buf[2048];
    snprintf(buf, sizeof(buf), "%s|%s|%s|%d|%d|%d|%lld|%lld|%d|%.17g", provenance_table ? provenance_table : "",
             predicate_for_hash, metric ? metric : "", top_k, depth, min_degree, (long long)g_adj, (long long)g_prov,
             community_filter, community_resolution);
    unsigned int h = graph_str_hash(buf);
    free(canonical);
    return h;
}

/* G2 T2.2 — cache put/get. */

/* The signature is currently 32-bit unsigned; the hex column width is
 * 8 chars + null. T2.4's xxh3 swap will widen this to 32 chars without
 * changing the read/write API. */
#define TOPK_SIG_HEX_LEN 8

static const char *kTopkCreateSql = "CREATE TABLE IF NOT EXISTS _gii_topk_cache ("
                                    "  signature       TEXT PRIMARY KEY,"
                                    "  seeds_json      TEXT,"
                                    "  nodes_json      TEXT,"
                                    "  edges_json      TEXT,"
                                    "  edge_generation INTEGER,"
                                    "  prov_generation INTEGER,"
                                    "  cached_at       TEXT"
                                    ")";

static void format_signature(unsigned int sig, char out[TOPK_SIG_HEX_LEN + 1]) {
    snprintf(out, TOPK_SIG_HEX_LEN + 1, "%08x", sig);
}

int topk_cache_put(sqlite3 *db, unsigned int signature, const char *seeds_json, const char *nodes_json,
                   const char *edges_json, int64_t edge_generation, int64_t prov_generation) {
    if (!db) {
        return SQLITE_MISUSE;
    }

    /* Lazy CREATE TABLE — first put bootstraps the cache schema. */
    int rc = sqlite3_exec(db, kTopkCreateSql, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }

    char sig_hex[TOPK_SIG_HEX_LEN + 1];
    format_signature(signature, sig_hex);

    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db,
                            "INSERT OR REPLACE INTO _gii_topk_cache"
                            "(signature, seeds_json, nodes_json, edges_json,"
                            " edge_generation, prov_generation, cached_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                            -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }
    sqlite3_bind_text(stmt, 1, sig_hex, TOPK_SIG_HEX_LEN, SQLITE_TRANSIENT);
    if (seeds_json) {
        sqlite3_bind_text(stmt, 2, seeds_json, -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 2);
    }
    if (nodes_json) {
        sqlite3_bind_text(stmt, 3, nodes_json, -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 3);
    }
    if (edges_json) {
        sqlite3_bind_text(stmt, 4, edges_json, -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 4);
    }
    sqlite3_bind_int64(stmt, 5, edge_generation);
    sqlite3_bind_int64(stmt, 6, prov_generation);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/* Copy a TEXT column into a malloc'd string. NULL column → out is NULL.
 * Returns 0 on success, non-zero on alloc failure. */
static int copy_text_column(sqlite3_stmt *stmt, int col, char **out) {
    if (sqlite3_column_type(stmt, col) == SQLITE_NULL) {
        *out = NULL;
        return 0;
    }
    const unsigned char *text = sqlite3_column_text(stmt, col);
    int bytes = sqlite3_column_bytes(stmt, col);
    if (!text || bytes < 0) {
        *out = NULL;
        return 0;
    }
    *out = (char *)malloc((size_t)bytes + 1);
    if (!*out) {
        return -1;
    }
    memcpy(*out, text, (size_t)bytes);
    (*out)[bytes] = '\0';
    return 0;
}

int topk_cache_get(sqlite3 *db, unsigned int signature, int64_t edge_generation, int64_t prov_generation,
                   char **out_seeds_json, char **out_nodes_json, char **out_edges_json) {
    if (out_seeds_json)
        *out_seeds_json = NULL;
    if (out_nodes_json)
        *out_nodes_json = NULL;
    if (out_edges_json)
        *out_edges_json = NULL;
    if (!db || !out_seeds_json || !out_nodes_json || !out_edges_json) {
        return SQLITE_MISUSE;
    }

    char sig_hex[TOPK_SIG_HEX_LEN + 1];
    format_signature(signature, sig_hex);

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db,
                                "SELECT seeds_json, nodes_json, edges_json "
                                "FROM _gii_topk_cache "
                                "WHERE signature = ? "
                                "  AND edge_generation = ? "
                                "  AND prov_generation = ?",
                                -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        /* "no such table" before any put: cache is logically empty.
         * Treat as miss rather than propagating the prepare error. */
        return SQLITE_NOTFOUND;
    }
    sqlite3_bind_text(stmt, 1, sig_hex, TOPK_SIG_HEX_LEN, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, edge_generation);
    sqlite3_bind_int64(stmt, 3, prov_generation);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return SQLITE_NOTFOUND;
    }
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return rc;
    }

    int alloc_ok = (copy_text_column(stmt, 0, out_seeds_json) == 0 &&
                    copy_text_column(stmt, 1, out_nodes_json) == 0 &&
                    copy_text_column(stmt, 2, out_edges_json) == 0);
    sqlite3_finalize(stmt);
    if (!alloc_ok) {
        free(*out_seeds_json);
        free(*out_nodes_json);
        free(*out_edges_json);
        *out_seeds_json = NULL;
        *out_nodes_json = NULL;
        *out_edges_json = NULL;
        return SQLITE_NOMEM;
    }
    return SQLITE_OK;
}
