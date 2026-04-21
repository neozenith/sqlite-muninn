/*
 * llama_er.c — Entity Resolution function: muninn_extract_er()
 *
 * Orchestrates the full ER pipeline using SQL calls to existing subsystems:
 *   1. KNN blocking via HNSW virtual table
 *   2. Source/type guard (same-source pairs filtered)
 *   3. JW + cosine scoring cascade (pure C)
 *   4. Connected components + LLM clustering (if borderline_delta > 0)
 *   5. Leiden clustering via graph_leiden TVF
 *   6. Edge betweenness cleanup via graph_edge_betweenness TVF (if threshold set)
 *
 * The match_threshold is derived: 1 - dist_threshold + borderline_delta.
 */

#include "llama_er.h"
#include "string_sim.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* ── Data structures ─────────────────────────────────────────────── */

typedef struct {
    int rowid;
    char *entity_id;
    char *name;
    char *name_lower;
    char *source;
} ErEntity;

typedef struct {
    int r1, r2; /* indices into entity array */
    double cosine_dist;
} CandidatePair;

typedef struct {
    int r1, r2;
    double weight;
} MatchEdge;

/* ── Helper: lowercase string (caller frees) ─────────────────────── */

static char *str_lower(const char *s) {
    size_t len = strlen(s);
    char *out = (char *)malloc(len + 1);
    for (size_t i = 0; i <= len; i++)
        out[i] = (char)tolower((unsigned char)s[i]);
    return out;
}

/* ── Helper: grow dynamic arrays ─────────────────────────────────── */

#define GROW_ARRAY(arr, count, cap, type)                                                                              \
    do {                                                                                                               \
        if ((count) >= (cap)) {                                                                                        \
            (cap) = (cap) ? (cap) * 2 : 64;                                                                            \
            (arr) = (type *)realloc((arr), (size_t)(cap) * sizeof(type));                                              \
        }                                                                                                              \
    } while (0)

/* ── Helper: write JSON-escaped string into buffer ────────────────── */

static size_t json_escape(char *out, size_t out_size, const char *s) {
    size_t pos = 0;
    out[pos++] = '"';
    for (; *s && pos < out_size - 2; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"') {
            if (pos + 2 >= out_size)
                break;
            out[pos++] = '\\';
            out[pos++] = '"';
        } else if (c == '\\') {
            if (pos + 2 >= out_size)
                break;
            out[pos++] = '\\';
            out[pos++] = '\\';
        } else if (c == '\n') {
            if (pos + 2 >= out_size)
                break;
            out[pos++] = '\\';
            out[pos++] = 'n';
        } else if (c == '\r') {
            if (pos + 2 >= out_size)
                break;
            out[pos++] = '\\';
            out[pos++] = 'r';
        } else if (c == '\t') {
            if (pos + 2 >= out_size)
                break;
            out[pos++] = '\\';
            out[pos++] = 't';
        } else if (c < 0x20) {
            /* Other control characters: \uXXXX */
            if (pos + 6 >= out_size)
                break;
            pos += (size_t)snprintf(out + pos, out_size - pos, "\\u%04x", c);
        } else {
            out[pos++] = (char)c;
        }
    }
    out[pos++] = '"';
    out[pos] = '\0';
    return pos;
}

/* ── Main function ───────────────────────────────────────────────── */

static void fn_extract_er(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 6) {
        sqlite3_result_error(ctx, "muninn_extract_er: requires at least 6 arguments", -1);
        return;
    }

    const char *hnsw_table = (const char *)sqlite3_value_text(argv[0]);
    const char *name_col = (const char *)sqlite3_value_text(argv[1]);
    int k = sqlite3_value_int(argv[2]);
    double dist_threshold = sqlite3_value_double(argv[3]);
    double jw_weight = sqlite3_value_double(argv[4]);
    double borderline_delta = sqlite3_value_double(argv[5]);
    const char *chat_model = argc > 6 ? (const char *)sqlite3_value_text(argv[6]) : NULL;
    double eb_threshold = argc > 7 && sqlite3_value_type(argv[7]) != SQLITE_NULL ? sqlite3_value_double(argv[7]) : -1.0;
    const char *type_guard = argc > 8 ? (const char *)sqlite3_value_text(argv[8]) : NULL;

    /* Type guard modes:
     *   "same_source" — skip pairs where source is identical (record linkage)
     *   "diff_type"   — skip pairs where source (=entity_type) differs (KG ER)
     *   NULL or ""    — no type filtering
     */
    int guard_same_source = type_guard && strcmp(type_guard, "same_source") == 0;
    int guard_diff_type = type_guard && strcmp(type_guard, "diff_type") == 0;

    if (!hnsw_table || !name_col) {
        sqlite3_result_error(ctx, "muninn_extract_er: hnsw_table and name_col required", -1);
        return;
    }

    double match_threshold = 1.0 - dist_threshold + borderline_delta;
    double llm_low = match_threshold - borderline_delta;

    sqlite3 *db = sqlite3_context_db_handle(ctx);
    int rc;
    char *errmsg = NULL;

    /* ── Stage 1: Load entities from the HNSW table's source ───────── */
    /* The HNSW table has shadow tables: {hnsw_table}_nodes
     * But we need entity names. We'll query the entities table that
     * shares rowids with the HNSW index.
     * Convention: the HNSW table name tells us the entities table.
     * We expect the caller has both an entities table with (entity_id, name, source)
     * and an HNSW index that shares rowids. */

    ErEntity *entities = NULL;
    int n_entities = 0, cap_entities = 0;

    /* Query entity data — we need entity_id, name, source from the source table */
    {
        /* Find the source table by looking for a table with the name_col column.
         * Simple heuristic: try "{hnsw_table} minus _vecs or _vec suffix" as entity table,
         * or just use "entities" as convention. */
        const char *entity_table = "entities";
        char sql[512];
        snprintf(sql, sizeof(sql), "SELECT rowid, entity_id, [%s], source FROM [%s]", name_col, entity_table);

        sqlite3_stmt *stmt;
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
            return;
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            GROW_ARRAY(entities, n_entities, cap_entities, ErEntity);
            ErEntity *e = &entities[n_entities++];
            e->rowid = sqlite3_column_int(stmt, 0);
            e->entity_id = strdup((const char *)sqlite3_column_text(stmt, 1));
            e->name = strdup((const char *)sqlite3_column_text(stmt, 2));
            e->name_lower = str_lower(e->name);
            const char *src = (const char *)sqlite3_column_text(stmt, 3);
            e->source = src ? strdup(src) : strdup("");
        }
        sqlite3_finalize(stmt);
    }

    if (n_entities == 0) {
        sqlite3_result_text(ctx, "{\"clusters\":{}}", -1, SQLITE_STATIC);
        return;
    }

    /* Build rowid -> index lookup */
    /* Simple approach: array indexed by max rowid (works for contiguous rowids) */
    int max_rowid = 0;
    for (int i = 0; i < n_entities; i++)
        if (entities[i].rowid > max_rowid)
            max_rowid = entities[i].rowid;

    int *rid_to_idx = (int *)malloc(((size_t)max_rowid + 1) * sizeof(int));
    memset(rid_to_idx, -1, ((size_t)max_rowid + 1) * sizeof(int));
    for (int i = 0; i < n_entities; i++)
        rid_to_idx[entities[i].rowid] = i;

    /* ── Stage 2: KNN blocking via HNSW ────────────────────────────── */

    CandidatePair *candidates = NULL;
    int n_candidates = 0, cap_candidates = 0;

    /* Pre-declare all arrays that cleanup needs to free */
    MatchEdge *match_edges = NULL;
    int n_edges = 0, cap_edges = 0;
    int *cluster_map = NULL;

    /* Prepare statements: one to fetch vector, one to search KNN */
    {
        char sql_vec[256], sql_knn[256];
        snprintf(sql_vec, sizeof(sql_vec), "SELECT vector FROM [%s] WHERE rowid = ?", hnsw_table);
        snprintf(sql_knn, sizeof(sql_knn), "SELECT rowid, distance FROM [%s] WHERE vector MATCH ? AND k = %d",
                 hnsw_table, k + 1);

        sqlite3_stmt *stmt_vec, *stmt_knn;
        rc = sqlite3_prepare_v2(db, sql_vec, -1, &stmt_vec, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
            goto cleanup;
        }
        rc = sqlite3_prepare_v2(db, sql_knn, -1, &stmt_knn, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt_vec);
            sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
            goto cleanup;
        }

        for (int i = 0; i < n_entities; i++) {
            /* Fetch this entity's vector */
            sqlite3_bind_int(stmt_vec, 1, entities[i].rowid);
            if (sqlite3_step(stmt_vec) != SQLITE_ROW) {
                sqlite3_reset(stmt_vec);
                continue;
            }
            const void *vec = sqlite3_column_blob(stmt_vec, 0);
            int vec_bytes = sqlite3_column_bytes(stmt_vec, 0);

            /* KNN search */
            sqlite3_bind_blob(stmt_knn, 1, vec, vec_bytes, SQLITE_TRANSIENT);
            while (sqlite3_step(stmt_knn) == SQLITE_ROW) {
                int nid = sqlite3_column_int(stmt_knn, 0);
                double dist = sqlite3_column_double(stmt_knn, 1);

                if (nid == entities[i].rowid)
                    continue;
                if (dist > dist_threshold)
                    continue;
                if (nid > max_rowid || rid_to_idx[nid] < 0)
                    continue;

                int j = rid_to_idx[nid];
                int lo = i < j ? i : j;
                int hi = i < j ? j : i;

                /* Deduplicate: linear scan (OK for moderate N) */
                int found = 0;
                for (int cc = 0; cc < n_candidates; cc++) {
                    if (candidates[cc].r1 == lo && candidates[cc].r2 == hi) {
                        if (dist < candidates[cc].cosine_dist)
                            candidates[cc].cosine_dist = dist;
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    GROW_ARRAY(candidates, n_candidates, cap_candidates, CandidatePair);
                    CandidatePair *cp = &candidates[n_candidates++];
                    cp->r1 = lo;
                    cp->r2 = hi;
                    cp->cosine_dist = dist;
                }
            }
            sqlite3_reset(stmt_knn);
            sqlite3_reset(stmt_vec);
        }
        sqlite3_finalize(stmt_vec);
        sqlite3_finalize(stmt_knn);
    }

    /* ── Stage 3: Source guard + scoring cascade ───────────────────── */

    /* Borderline pairs for LLM — future: connected components + muninn_chat() */
    (void)chat_model;
    (void)llm_low;

    for (int c = 0; c < n_candidates; c++) {
        int i = candidates[c].r1;
        int j = candidates[c].r2;

        /* Type guard — mode-dependent:
         *   same_source: skip if sources match (record linkage: a≠b only)
         *   diff_type:   skip if sources differ (KG: person≠location)
         */
        if (guard_same_source) {
            if (entities[i].source[0] && entities[j].source[0] && strcmp(entities[i].source, entities[j].source) == 0)
                continue;
        } else if (guard_diff_type) {
            if (entities[i].source[0] && entities[j].source[0] && strcmp(entities[i].source, entities[j].source) != 0)
                continue;
        }

        double cosine_sim = 1.0 - candidates[c].cosine_dist;
        double score;

        /* Exact match */
        if (strcmp(entities[i].name, entities[j].name) == 0) {
            score = 1.0;
        } else if (strcmp(entities[i].name_lower, entities[j].name_lower) == 0) {
            score = 0.9;
        } else {
            double jw = jaro_winkler(entities[i].name_lower, entities[j].name_lower);
            score = jw_weight * jw + (1.0 - jw_weight) * cosine_sim;
        }

        if (score >= match_threshold) {
            GROW_ARRAY(match_edges, n_edges, cap_edges, MatchEdge);
            MatchEdge *me = &match_edges[n_edges++];
            me->r1 = i;
            me->r2 = j;
            me->weight = score;
        }
        /* TODO: else if borderline_delta > 0 && score >= llm_low, collect for LLM */
    }

    /* ── Stage 5: Leiden clustering via temp table + TVF ──────────── */

    /* Create temp edge table for Leiden */
    sqlite3_exec(db, "DROP TABLE IF EXISTS temp._er_edges", NULL, NULL, NULL);
    rc = sqlite3_exec(db, "CREATE TEMP TABLE _er_edges(src TEXT, dst TEXT, weight REAL)", NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_result_error(ctx, errmsg ? errmsg : "failed to create temp table", -1);
        sqlite3_free(errmsg);
        goto cleanup;
    }

    /* Insert bidirectional edges */
    {
        sqlite3_stmt *ins;
        rc = sqlite3_prepare_v2(db, "INSERT INTO temp._er_edges(src, dst, weight) VALUES(?, ?, ?)", -1, &ins, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
            goto cleanup;
        }

        for (int e = 0; e < n_edges; e++) {
            const char *src = entities[match_edges[e].r1].entity_id;
            const char *dst = entities[match_edges[e].r2].entity_id;
            double w = match_edges[e].weight;

            sqlite3_bind_text(ins, 1, src, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(ins, 2, dst, -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(ins, 3, w);
            sqlite3_step(ins);
            sqlite3_reset(ins);

            /* Reverse edge */
            sqlite3_bind_text(ins, 1, dst, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(ins, 2, src, -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(ins, 3, w);
            sqlite3_step(ins);
            sqlite3_reset(ins);
        }
        sqlite3_finalize(ins);
    }

    /* Run Leiden */
    /* cluster_map[i] = community_id for entity i. -1 = singleton */
    cluster_map = (int *)malloc((size_t)n_entities * sizeof(int));
    for (int i = 0; i < n_entities; i++)
        cluster_map[i] = -1;
    int next_cluster = 0;

    if (n_edges > 0) {
        sqlite3_stmt *leiden;
        rc = sqlite3_prepare_v2(db,
                                "SELECT node, community_id FROM graph_leiden"
                                " WHERE edge_table = '_er_edges'"
                                "   AND src_col = 'src'"
                                "   AND dst_col = 'dst'"
                                "   AND weight_col = 'weight'",
                                -1, &leiden, NULL);
        if (rc != SQLITE_OK) {
            sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
            free(cluster_map);
            goto cleanup;
        }

        /* Map community IDs to sequential cluster IDs */
        int *comm_remap = NULL;
        int *comm_keys = NULL;
        int n_remap = 0, cap_remap = 0;

        while (sqlite3_step(leiden) == SQLITE_ROW) {
            const char *node = (const char *)sqlite3_column_text(leiden, 0);
            int comm_id = sqlite3_column_int(leiden, 1);

            for (int i = 0; i < n_entities; i++) {
                if (strcmp(entities[i].entity_id, node) == 0) {
                    int mapped = -1;
                    for (int r = 0; r < n_remap; r++) {
                        if (comm_keys[r] == comm_id) {
                            mapped = comm_remap[r];
                            break;
                        }
                    }
                    if (mapped < 0) {
                        if (n_remap >= cap_remap) {
                            cap_remap = cap_remap ? cap_remap * 2 : 64;
                            comm_remap = (int *)realloc(comm_remap, (size_t)cap_remap * sizeof(int));
                            comm_keys = (int *)realloc(comm_keys, (size_t)cap_remap * sizeof(int));
                        }
                        comm_keys[n_remap] = comm_id;
                        comm_remap[n_remap] = next_cluster++;
                        mapped = comm_remap[n_remap];
                        n_remap++;
                    }
                    cluster_map[i] = mapped;
                    break;
                }
            }
        }
        sqlite3_finalize(leiden);
        free(comm_remap);
        free(comm_keys);
    }

    /* Assign singletons */
    for (int i = 0; i < n_entities; i++) {
        if (cluster_map[i] < 0)
            cluster_map[i] = next_cluster++;
    }

    /* ── Stage 6: Edge betweenness cleanup (optional) ────────────── */

    if (eb_threshold >= 0 && n_edges > 0) {
        sqlite3_stmt *eb_stmt;
        rc = sqlite3_prepare_v2(db,
                                "SELECT src, dst, centrality FROM graph_edge_betweenness"
                                " WHERE edge_table = '_er_edges'"
                                "   AND src_col = 'src'"
                                "   AND dst_col = 'dst'"
                                "   AND direction = 'both'",
                                -1, &eb_stmt, NULL);

        if (rc == SQLITE_OK) {
            /* Collect bridge edges */
            int n_bridges = 0;
            char **bridge_src = NULL, **bridge_dst = NULL;
            int cap_bridges = 0;

            while (sqlite3_step(eb_stmt) == SQLITE_ROW) {
                double bc = sqlite3_column_double(eb_stmt, 2);
                if (bc > eb_threshold) {
                    const char *s = (const char *)sqlite3_column_text(eb_stmt, 0);
                    const char *d = (const char *)sqlite3_column_text(eb_stmt, 1);
                    GROW_ARRAY(bridge_src, n_bridges, cap_bridges, char *);
                    GROW_ARRAY(bridge_dst, n_bridges, cap_bridges, char *);
                    bridge_src[n_bridges] = strdup(s);
                    bridge_dst[n_bridges] = strdup(d);
                    n_bridges++;
                }
            }
            sqlite3_finalize(eb_stmt);

            if (n_bridges > 0) {
                /* Delete bridge edges from temp table */
                for (int b = 0; b < n_bridges; b++) {
                    char del_sql[512];
                    snprintf(del_sql, sizeof(del_sql),
                             "DELETE FROM temp._er_edges WHERE"
                             " (src = '%s' AND dst = '%s') OR (src = '%s' AND dst = '%s')",
                             bridge_src[b], bridge_dst[b], bridge_dst[b], bridge_src[b]);
                    sqlite3_exec(db, del_sql, NULL, NULL, NULL);
                    free(bridge_src[b]);
                    free(bridge_dst[b]);
                }
                free(bridge_src);
                free(bridge_dst);

                /* Re-run Leiden on pruned edges */
                for (int i = 0; i < n_entities; i++)
                    cluster_map[i] = -1;
                next_cluster = 0;

                sqlite3_stmt *leiden2;
                rc = sqlite3_prepare_v2(db,
                                        "SELECT node, community_id FROM graph_leiden"
                                        " WHERE edge_table = '_er_edges'"
                                        "   AND src_col = 'src'"
                                        "   AND dst_col = 'dst'"
                                        "   AND weight_col = 'weight'",
                                        -1, &leiden2, NULL);

                if (rc == SQLITE_OK) {
                    int *comm_remap2 = NULL, *comm_keys2 = NULL;
                    int n_remap2 = 0, cap_remap2 = 0;

                    while (sqlite3_step(leiden2) == SQLITE_ROW) {
                        const char *node = (const char *)sqlite3_column_text(leiden2, 0);
                        int comm_id = sqlite3_column_int(leiden2, 1);
                        for (int i = 0; i < n_entities; i++) {
                            if (strcmp(entities[i].entity_id, node) == 0) {
                                int mapped = -1;
                                for (int r = 0; r < n_remap2; r++) {
                                    if (comm_keys2[r] == comm_id) {
                                        mapped = comm_remap2[r];
                                        break;
                                    }
                                }
                                if (mapped < 0) {
                                    if (n_remap2 >= cap_remap2) {
                                        cap_remap2 = cap_remap2 ? cap_remap2 * 2 : 64;
                                        comm_remap2 = (int *)realloc(comm_remap2, (size_t)cap_remap2 * sizeof(int));
                                        comm_keys2 = (int *)realloc(comm_keys2, (size_t)cap_remap2 * sizeof(int));
                                    }
                                    comm_keys2[n_remap2] = comm_id;
                                    comm_remap2[n_remap2] = next_cluster++;
                                    mapped = comm_remap2[n_remap2];
                                    n_remap2++;
                                }
                                cluster_map[i] = mapped;
                                break;
                            }
                        }
                    }
                    sqlite3_finalize(leiden2);
                    free(comm_remap2);
                    free(comm_keys2);
                }

                for (int i = 0; i < n_entities; i++)
                    if (cluster_map[i] < 0)
                        cluster_map[i] = next_cluster++;
            }
        }
    }

    /* ── Build JSON result ───────────────────────────────────────── */
    {
        /* Estimate size: each entry needs escaped key + int value.
         * Worst case: every char escaped to \uXXXX = 6x expansion. */
        size_t buf_size = (size_t)n_entities * 120 + 32;
        char *json = (char *)malloc(buf_size);
        char esc_buf[2048]; /* scratch for escaping one entity_id */
        size_t pos = 0;

        pos += (size_t)snprintf(json + pos, buf_size - pos, "{\"clusters\":{");
        for (int i = 0; i < n_entities; i++) {
            if (i > 0)
                json[pos++] = ',';
            size_t key_len = json_escape(esc_buf, sizeof(esc_buf), entities[i].entity_id);
            /* Grow if needed */
            if (pos + key_len + 16 > buf_size) {
                buf_size = (pos + key_len + 16) * 2;
                json = (char *)realloc(json, buf_size);
            }
            memcpy(json + pos, esc_buf, key_len);
            pos += key_len;
            pos += (size_t)snprintf(json + pos, buf_size - pos, ":%d", cluster_map[i]);
        }
        pos += (size_t)snprintf(json + pos, buf_size - pos, "}}");

        sqlite3_result_text(ctx, json, (int)pos, free);
        sqlite3_result_subtype(ctx, 'J');
    }

    free(cluster_map);

cleanup:
    /* Cleanup temp table */
    sqlite3_exec(db, "DROP TABLE IF EXISTS temp._er_edges", NULL, NULL, NULL);

    free(candidates);
    free(match_edges);
    free(rid_to_idx);
    for (int i = 0; i < n_entities; i++) {
        free(entities[i].entity_id);
        free(entities[i].name);
        free(entities[i].name_lower);
        free(entities[i].source);
    }
    free(entities);
}

/* ── Registration ────────────────────────────────────────────────── */

int llama_er_register_functions(sqlite3 *db) {
    return sqlite3_create_function(db, "muninn_extract_er", -1, SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL, fn_extract_er,
                                   NULL, NULL);
}
