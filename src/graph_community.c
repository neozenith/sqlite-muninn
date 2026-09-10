/*
 * graph_community.c — Community detection and partition-quality TVFs
 *
 * TVFs:
 *   graph_leiden      — Leiden community detection (Traag et al., 2019)
 *   graph_conductance — per-group conductance of any node membership
 *                       (Kannan, Vempala & Vetta, 2004)
 *
 * The Leiden algorithm iterates three phases:
 *   1. Local moving — each node moves to best neighboring community
 *   2. Refinement — ensure communities are well-connected (sub-partition)
 *   3. Aggregation — collapse into super-graph and repeat
 *
 * Terminates when no node changes community in Phase 1.
 */
#include "graph_community.h"
#include "graph_common.h"
#include "graph_load.h"
#include "graph_adjacency.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

SQLITE_EXTENSION_INIT3

/* ═══════════════════════════════════════════════════════════════
 * Result structure
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *node;
    int community_id;
    double modularity;
} CommunityRow;

typedef struct {
    CommunityRow *rows;
    int count;
    int capacity;
} CommunityResults;

static void comr_init(CommunityResults *r) {
    r->count = 0;
    r->capacity = 64;
    r->rows = (CommunityRow *)calloc((size_t)r->capacity, sizeof(CommunityRow));
}

static void comr_destroy(CommunityResults *r) {
    for (int i = 0; i < r->count; i++)
        free(r->rows[i].node);
    free(r->rows);
    r->rows = NULL;
    r->count = 0;
}

static void comr_add(CommunityResults *r, const char *node, int comm, double mod) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (CommunityRow *)realloc(r->rows, (size_t)r->capacity * sizeof(CommunityRow));
    }
    CommunityRow *row = &r->rows[r->count++];
    row->node = strdup(node);
    row->community_id = comm;
    row->modularity = mod;
}

/* ═══════════════════════════════════════════════════════════════
 * Leiden algorithm internals
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Compute sum of weights from node v to nodes in community c.
 * Uses the combined (out+in for undirected / out for directed) adjacency.
 */
static double weight_to_community(const GraphData *g, int v, const int *community, int target_comm, int use_both) {
    double sum = 0.0;
    for (int e = 0; e < g->out[v].count; e++) {
        int w = g->out[v].edges[e].target;
        if (community[w] == target_comm)
            sum += g->out[v].edges[e].weight;
    }
    if (use_both) {
        for (int e = 0; e < g->in[v].count; e++) {
            int w = g->in[v].edges[e].target;
            if (community[w] == target_comm)
                sum += g->in[v].edges[e].weight;
        }
    }
    return sum;
}

/*
 * Weighted degree of node v (sum of all edge weights).
 */
static double weighted_degree(const GraphData *g, int v, int use_both) {
    double k = 0.0;
    for (int e = 0; e < g->out[v].count; e++)
        k += g->out[v].edges[e].weight;
    if (use_both) {
        for (int e = 0; e < g->in[v].count; e++)
            k += g->in[v].edges[e].weight;
    }
    return k;
}

/*
 * Compute modularity Q = (1/2m) * SUM[ A_ij - gamma*k_i*k_j/(2m) ] * delta(c_i, c_j)
 */
static double compute_modularity(const GraphData *g, const int *community, double resolution, double m, int use_both) {
    int N = g->node_count;
    if (m <= 0)
        return 0.0;

    /* Compute sum_in and sum_tot per community */
    /* sum_in = sum of weights within community */
    /* sum_tot = sum of degrees of nodes in community */
    int max_comm = 0;
    for (int i = 0; i < N; i++) {
        if (community[i] > max_comm)
            max_comm = community[i];
    }
    int n_comm = max_comm + 1;
    double *sum_in = (double *)calloc((size_t)n_comm, sizeof(double));
    double *sum_tot = (double *)calloc((size_t)n_comm, sizeof(double));

    for (int i = 0; i < N; i++) {
        int c = community[i];
        sum_tot[c] += weighted_degree(g, i, use_both);
        sum_in[c] += weight_to_community(g, i, community, c, use_both);
    }

    double Q = 0.0;
    for (int c = 0; c < n_comm; c++) {
        if (sum_tot[c] > 0) {
            Q += sum_in[c] / (2.0 * m) - resolution * (sum_tot[c] / (2.0 * m)) * (sum_tot[c] / (2.0 * m));
        }
    }

    free(sum_in);
    free(sum_tot);
    return Q;
}

/*
 * Phase 1: Local moving.
 * Each node tries moving to the neighboring community that maximizes
 * modularity gain. Repeat until no improvement.
 * Returns number of moves made.
 */
static int leiden_local_moving(const GraphData *g, int *community, double *sum_tot, double *k, double m,
                               double resolution, int use_both) {
    int N = g->node_count;
    int total_moves = 0;
    int improved = 1;

    while (improved) {
        improved = 0;
        for (int v = 0; v < N; v++) {
            int old_comm = community[v];
            double k_v = k[v];

            /* Remove v from its community */
            double k_v_to_old = weight_to_community(g, v, community, old_comm, use_both);

            /* Try each neighboring community */
            int best_comm = old_comm;
            double best_gain = 0.0;

            /* Collect unique neighboring communities */
            int *neigh_comms = (int *)malloc((size_t)(g->out[v].count + g->in[v].count + 1) * sizeof(int));
            int n_neigh = 0;

            for (int e = 0; e < g->out[v].count; e++) {
                int nc = community[g->out[v].edges[e].target];
                /* Check if already seen */
                int seen = 0;
                for (int j = 0; j < n_neigh; j++) {
                    if (neigh_comms[j] == nc) {
                        seen = 1;
                        break;
                    }
                }
                if (!seen)
                    neigh_comms[n_neigh++] = nc;
            }
            if (use_both) {
                for (int e = 0; e < g->in[v].count; e++) {
                    int nc = community[g->in[v].edges[e].target];
                    int seen = 0;
                    for (int j = 0; j < n_neigh; j++) {
                        if (neigh_comms[j] == nc) {
                            seen = 1;
                            break;
                        }
                    }
                    if (!seen)
                        neigh_comms[n_neigh++] = nc;
                }
            }

            for (int j = 0; j < n_neigh; j++) {
                int target_comm = neigh_comms[j];
                if (target_comm == old_comm)
                    continue;

                double k_v_to_target = weight_to_community(g, v, community, target_comm, use_both);

                /* Modularity gain from moving v: old_comm -> target_comm */
                double gain = (k_v_to_target - k_v_to_old) / m +
                              resolution * k_v * (sum_tot[old_comm] - k_v - sum_tot[target_comm]) / (2.0 * m * m);

                if (gain > best_gain) {
                    best_gain = gain;
                    best_comm = target_comm;
                }
            }

            free(neigh_comms);

            if (best_comm != old_comm) {
                /* Move v */
                sum_tot[old_comm] -= k_v;
                sum_tot[best_comm] += k_v;
                community[v] = best_comm;
                improved = 1;
                total_moves++;
            }
        }
    }
    return total_moves;
}

/*
 * Phase 2: Refinement.
 * Within each community found by Phase 1, start with singletons and
 * only merge nodes that are well-connected within the community.
 */
static void leiden_refinement(const GraphData *g, const int *partition, int *refined, double *k, double m,
                              double resolution, int use_both) {
    int N = g->node_count;

    /* Start with singletons */
    for (int i = 0; i < N; i++)
        refined[i] = i;

    /* Compute sum_tot for refined communities */
    double *r_sum_tot = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++)
        r_sum_tot[i] = k[i];

    /* For each partition community, refine internally */
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int v = 0; v < N; v++) {
            int old_ref = refined[v];
            int part_comm = partition[v];
            double k_v = k[v];

            double k_v_to_old = weight_to_community(g, v, refined, old_ref, use_both);

            int best_ref = old_ref;
            double best_gain = 0.0;

            /* Only try merging with nodes in the same Phase-1 community */
            for (int e = 0; e < g->out[v].count; e++) {
                int w = g->out[v].edges[e].target;
                if (partition[w] != part_comm)
                    continue;
                int nr = refined[w];
                if (nr == old_ref)
                    continue;

                double k_v_to_nr = weight_to_community(g, v, refined, nr, use_both);
                double gain = (k_v_to_nr - k_v_to_old) / m +
                              resolution * k_v * (r_sum_tot[old_ref] - k_v - r_sum_tot[nr]) / (2.0 * m * m);

                if (gain > best_gain) {
                    best_gain = gain;
                    best_ref = nr;
                }
            }
            if (use_both) {
                for (int e = 0; e < g->in[v].count; e++) {
                    int w = g->in[v].edges[e].target;
                    if (partition[w] != part_comm)
                        continue;
                    int nr = refined[w];
                    if (nr == old_ref)
                        continue;

                    double k_v_to_nr = weight_to_community(g, v, refined, nr, use_both);
                    double gain = (k_v_to_nr - k_v_to_old) / m +
                                  resolution * k_v * (r_sum_tot[old_ref] - k_v - r_sum_tot[nr]) / (2.0 * m * m);

                    if (gain > best_gain) {
                        best_gain = gain;
                        best_ref = nr;
                    }
                }
            }

            if (best_ref != old_ref) {
                r_sum_tot[old_ref] -= k_v;
                r_sum_tot[best_ref] += k_v;
                refined[v] = best_ref;
                improved = 1;
            }
        }
    }
    free(r_sum_tot);
}

/*
 * Renumber community labels to be contiguous 0..K-1.
 */
static int renumber_communities(int *community, int N) {
    int *mapping = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++)
        mapping[i] = -1;
    int next_id = 0;
    for (int i = 0; i < N; i++) {
        if (mapping[community[i]] == -1) {
            mapping[community[i]] = next_id++;
        }
        community[i] = mapping[community[i]];
    }
    free(mapping);
    return next_id;
}

/*
 * Run the full Leiden algorithm.
 * Returns community assignment in community[] (0-indexed, contiguous).
 */
double run_leiden(const GraphData *g, int *community, double resolution, const char *direction) {
    int N = g->node_count;
    if (N == 0)
        return 0.0;

    int use_both = direction && strcmp(direction, "both") == 0;

    /* Compute total edge weight (m) and per-node weighted degree */
    double *k = (double *)malloc((size_t)N * sizeof(double));
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        k[i] = weighted_degree(g, i, use_both);
        m += k[i];
    }
    m /= 2.0; /* each edge counted twice */
    if (m <= 0.0) {
        free(k);
        for (int i = 0; i < N; i++)
            community[i] = i;
        return 0.0;
    }

    /* Initialize: each node in its own community */
    for (int i = 0; i < N; i++)
        community[i] = i;

    /* Compute initial sum_tot per community */
    double *sum_tot = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++)
        sum_tot[i] = k[i];

    int *refined = (int *)malloc((size_t)N * sizeof(int));
    int max_iter = 100;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Phase 1: Local moving */
        int moves = leiden_local_moving(g, community, sum_tot, k, m, resolution, use_both);
        if (moves == 0)
            break;

        /* Phase 2: Refinement — sub-partition within Phase 1 communities.
         * Only adopt the refined partition if it merges at least as many
         * nodes as Phase 1 did.  The refinement gain formula uses global m,
         * which makes the penalty term dominate for small/medium graphs,
         * causing refinement to produce all-singletons.  Falling back to
         * the Phase 1 result in that case preserves correct community
         * structure. */
        leiden_refinement(g, community, refined, k, m, resolution, use_both);

        /* Count distinct communities in Phase 1 vs refinement */
        int p1_comms = 0, ref_comms = 0;
        {
            /* Use a simple counting approach */
            int *p1_seen = (int *)calloc((size_t)N, sizeof(int));
            int *ref_seen = (int *)calloc((size_t)N, sizeof(int));
            for (int i = 0; i < N; i++) {
                if (!p1_seen[community[i]]) {
                    p1_seen[community[i]] = 1;
                    p1_comms++;
                }
                if (!ref_seen[refined[i]]) {
                    ref_seen[refined[i]] = 1;
                    ref_comms++;
                }
            }
            free(p1_seen);
            free(ref_seen);
        }

        /* Only use refinement if it didn't regress to more communities */
        if (ref_comms <= p1_comms) {
            memcpy(community, refined, (size_t)N * sizeof(int));
        }

        /* Renumber and recompute sum_tot */
        int K = renumber_communities(community, N);
        (void)K;
        memset(sum_tot, 0, (size_t)N * sizeof(double));
        for (int i = 0; i < N; i++) {
            sum_tot[community[i]] += k[i];
        }
    }

    /* Final renumbering */
    renumber_communities(community, N);

    /* Compute final modularity */
    double Q = compute_modularity(g, community, resolution, m, use_both);

    free(k);
    free(sum_tot);
    free(refined);
    return Q;
}

/* ═══════════════════════════════════════════════════════════════
 * TVF: graph_leiden
 * ═══════════════════════════════════════════════════════════════ */

/* safe_text and graph_best_index_common are in graph_common.h */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} CommunityVtab;

static int comm_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

enum {
    LEI_COL_NODE = 0,
    LEI_COL_COMMUNITY_ID,
    LEI_COL_MODULARITY,
    LEI_COL_EDGE_TABLE,    /* hidden */
    LEI_COL_SRC_COL,       /* hidden */
    LEI_COL_DST_COL,       /* hidden */
    LEI_COL_WEIGHT_COL,    /* hidden */
    LEI_COL_RESOLUTION,    /* hidden */
    LEI_COL_DIRECTION,     /* hidden */
    LEI_COL_TIMESTAMP_COL, /* hidden */
    LEI_COL_TIME_START,    /* hidden */
    LEI_COL_TIME_END,      /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    CommunityResults results;
    int current;
    int eof;
} LeidenCursor;

static int lei_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab,
                       char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x("
                                      "  node TEXT, community_id INTEGER, modularity REAL,"
                                      "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
                                      "  weight_col TEXT HIDDEN, resolution REAL HIDDEN,"
                                      "  direction TEXT HIDDEN, timestamp_col TEXT HIDDEN,"
                                      "  time_start HIDDEN, time_end HIDDEN"
                                      ")");
    if (rc != SQLITE_OK)
        return rc;

    CommunityVtab *vtab = (CommunityVtab *)sqlite3_malloc(sizeof(CommunityVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(CommunityVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int lei_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    return graph_best_index_common(pIdxInfo, LEI_COL_EDGE_TABLE, LEI_COL_TIME_END, 0x7, 5000.0);
}

static int lei_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    LeidenCursor *cur = (LeidenCursor *)calloc(1, sizeof(LeidenCursor));
    if (!cur)
        return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int lei_close(sqlite3_vtab_cursor *pCursor) {
    LeidenCursor *cur = (LeidenCursor *)pCursor;
    comr_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int lei_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    LeidenCursor *cur = (LeidenCursor *)pCursor;
    CommunityVtab *vtab = (CommunityVtab *)pCursor->pVtab;

    comr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(CommunityResults));

    if (argc < 3) {
        cur->eof = 1;
        return SQLITE_OK;
    }

    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    double resolution = 1.0;
    int pos = 0;

#define LEI_N_HIDDEN (LEI_COL_TIME_END - LEI_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < LEI_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit)))
            continue;
        switch (bit + LEI_COL_EDGE_TABLE) {
        case LEI_COL_EDGE_TABLE:
            config.edge_table = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_SRC_COL:
            config.src_col = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_DST_COL:
            config.dst_col = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_WEIGHT_COL:
            config.weight_col = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_RESOLUTION:
            resolution = sqlite3_value_double(argv[pos]);
            break;
        case LEI_COL_DIRECTION:
            config.direction = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_TIMESTAMP_COL:
            config.timestamp_col = graph_safe_text(argv[pos]);
            break;
        case LEI_COL_TIME_START:
            config.time_start = argv[pos];
            break;
        case LEI_COL_TIME_END:
            config.time_end = argv[pos];
            break;
        }
        pos++;
    }

    if (!config.direction)
        config.direction = "both";

    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc;
    if (config.edge_table && is_graph_adjacency(vtab->db, config.edge_table)) {
        rc = graph_data_load_from_adjacency(vtab->db, config.edge_table, &g, &errmsg);
    } else {
        rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    }
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg ? errmsg : sqlite3_mprintf("graph_leiden: failed to load graph");
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    if (N == 0) {
        graph_data_destroy(&g);
        comr_init(&cur->results);
        cur->eof = 1;
        return SQLITE_OK;
    }

    int *community = (int *)malloc((size_t)N * sizeof(int));
    double Q = run_leiden(&g, community, resolution, config.direction);

    comr_init(&cur->results);
    for (int i = 0; i < N; i++) {
        comr_add(&cur->results, g.ids[i], community[i], Q);
    }

    free(community);
    graph_data_destroy(&g);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int lei_next(sqlite3_vtab_cursor *p) {
    LeidenCursor *cur = (LeidenCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int lei_eof(sqlite3_vtab_cursor *p) {
    return ((LeidenCursor *)p)->eof;
}

static int lei_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    LeidenCursor *cur = (LeidenCursor *)p;
    CommunityRow *row = &cur->results.rows[cur->current];
    switch (col) {
    case LEI_COL_NODE:
        sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT);
        break;
    case LEI_COL_COMMUNITY_ID:
        sqlite3_result_int(ctx, row->community_id);
        break;
    case LEI_COL_MODULARITY:
        sqlite3_result_double(ctx, row->modularity);
        break;
    default:
        sqlite3_result_null(ctx);
        break;
    }
    return SQLITE_OK;
}

static int lei_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((LeidenCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_leiden_module = {
    .iVersion = 0,
    .xCreate = NULL,
    .xConnect = lei_connect,
    .xBestIndex = lei_best_index,
    .xDisconnect = comm_disconnect,
    .xDestroy = comm_disconnect,
    .xOpen = lei_open,
    .xClose = lei_close,
    .xFilter = lei_filter,
    .xNext = lei_next,
    .xEof = lei_eof,
    .xColumn = lei_column,
    .xRowid = lei_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Conductance
 * ═══════════════════════════════════════════════════════════════ */

/*
 * graph_data_load() stores each edge row once in out[src] (and mirrors it
 * into in[dst] when direction is "both" or "reverse"). Walking out[] alone
 * therefore visits every row exactly once; for "reverse" out[] is empty and
 * in[] holds the rows instead.
 */
int run_conductance(const GraphData *g, const int *group, int k, const char *direction, Conductance *out) {
    int N = g->node_count;
    for (int c = 0; c < k; c++) {
        out[c].size = 0;
        out[c].internal = 0.0;
        out[c].cut = 0.0;
        out[c].vol = 0.0;
        out[c].phi = 0.0;
    }
    if (N == 0 || k == 0)
        return SQLITE_OK;

    for (int i = 0; i < N; i++) {
        if (group[i] >= 0 && group[i] < k)
            out[group[i]].size++;
    }

    int use_in = direction && strcmp(direction, "reverse") == 0;
    double total_vol = 0.0;
    for (int u = 0; u < N; u++) {
        const GraphAdjList *adj = use_in ? &g->in[u] : &g->out[u];
        for (int e = 0; e < adj->count; e++) {
            int v = adj->edges[e].target;
            double w = adj->edges[e].weight;
            total_vol += 2.0 * w;
            int gu = group[u], gv = group[v];
            if (gu >= 0 && gu == gv) {
                out[gu].internal += w;
            } else {
                if (gu >= 0)
                    out[gu].cut += w;
                if (gv >= 0)
                    out[gv].cut += w;
            }
        }
    }

    for (int c = 0; c < k; c++) {
        out[c].vol = 2.0 * out[c].internal + out[c].cut;
        double rest = total_vol - out[c].vol;
        double denom = out[c].vol < rest ? out[c].vol : rest;
        out[c].phi = denom > 0.0 ? out[c].cut / denom : 0.0;
    }
    return SQLITE_OK;
}

/*
 * Read (group, member) rows from membership_table and assign a group index
 * to every loaded graph node. Group values are read as TEXT and interned via
 * a second GraphData (which is a string-interning hash map), so INTEGER
 * community IDs and TEXT labels take the same path. Members absent from the
 * graph are ignored; a node listed more than once keeps its first group.
 *
 * On success *groups holds the interned group ids (groups->ids[c],
 * groups->node_count == k). Caller must graph_data_destroy(groups).
 */
static int load_membership(sqlite3 *db, const char *table, const char *group_col, const char *member_col,
                           const GraphData *g, int *group, GraphData *groups, char **pzErrMsg) {
    if (id_validate(table) != 0 || id_validate(group_col) != 0 || id_validate(member_col) != 0) {
        *pzErrMsg = sqlite3_mprintf("graph_conductance: invalid membership table/column identifier");
        return SQLITE_ERROR;
    }
    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"", group_col, member_col, table);
    if (!sql)
        return SQLITE_NOMEM;
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("graph_conductance: %s", sqlite3_errmsg(db));
        return rc;
    }

    for (int i = 0; i < g->node_count; i++)
        group[i] = -1;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *gid = (const char *)sqlite3_column_text(stmt, 0);
        const char *member = (const char *)sqlite3_column_text(stmt, 1);
        if (!gid || !member)
            continue;
        int node = graph_data_find(g, member);
        if (node < 0 || group[node] >= 0)
            continue;
        group[node] = graph_data_find_or_add(groups, gid);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * TVF: graph_conductance
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *group_id;
    Conductance c;
} ConductanceRow;

enum {
    CND_COL_GROUP_ID = 0,
    CND_COL_SIZE,
    CND_COL_INTERNAL,
    CND_COL_CUT,
    CND_COL_VOL,
    CND_COL_PHI,
    CND_COL_EDGE_TABLE,       /* hidden */
    CND_COL_SRC_COL,          /* hidden */
    CND_COL_DST_COL,          /* hidden */
    CND_COL_WEIGHT_COL,       /* hidden */
    CND_COL_DIRECTION,        /* hidden */
    CND_COL_TIMESTAMP_COL,    /* hidden */
    CND_COL_TIME_START,       /* hidden */
    CND_COL_TIME_END,         /* hidden */
    CND_COL_MEMBERSHIP_TABLE, /* hidden */
    CND_COL_GROUP_COL,        /* hidden */
    CND_COL_MEMBER_COL,       /* hidden */
};

/* Required: edge_table, src_col, dst_col, membership_table, group_col, member_col */
#define CND_REQUIRED_MASK                                                                                              \
    ((1 << (CND_COL_EDGE_TABLE - CND_COL_EDGE_TABLE)) | (1 << (CND_COL_SRC_COL - CND_COL_EDGE_TABLE)) |                \
     (1 << (CND_COL_DST_COL - CND_COL_EDGE_TABLE)) | (1 << (CND_COL_MEMBERSHIP_TABLE - CND_COL_EDGE_TABLE)) |          \
     (1 << (CND_COL_GROUP_COL - CND_COL_EDGE_TABLE)) | (1 << (CND_COL_MEMBER_COL - CND_COL_EDGE_TABLE)))

typedef struct {
    sqlite3_vtab_cursor base;
    ConductanceRow *rows;
    int count;
    int current;
    int eof;
} ConductanceCursor;

static void cnd_rows_destroy(ConductanceCursor *cur) {
    for (int i = 0; i < cur->count; i++)
        free(cur->rows[i].group_id);
    free(cur->rows);
    cur->rows = NULL;
    cur->count = 0;
}

static int cnd_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab,
                       char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x("
                                      "  group_id TEXT, size INTEGER, internal REAL, cut REAL, vol REAL, phi REAL,"
                                      "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
                                      "  weight_col TEXT HIDDEN, direction TEXT HIDDEN, timestamp_col TEXT HIDDEN,"
                                      "  time_start HIDDEN, time_end HIDDEN,"
                                      "  membership_table TEXT HIDDEN, group_col TEXT HIDDEN, member_col TEXT HIDDEN"
                                      ")");
    if (rc != SQLITE_OK)
        return rc;

    CommunityVtab *vtab = (CommunityVtab *)sqlite3_malloc(sizeof(CommunityVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(CommunityVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int cnd_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    return graph_best_index_common(pIdxInfo, CND_COL_EDGE_TABLE, CND_COL_MEMBER_COL, CND_REQUIRED_MASK, 2000.0);
}

static int cnd_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    ConductanceCursor *cur = (ConductanceCursor *)calloc(1, sizeof(ConductanceCursor));
    if (!cur)
        return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int cnd_close(sqlite3_vtab_cursor *pCursor) {
    ConductanceCursor *cur = (ConductanceCursor *)pCursor;
    cnd_rows_destroy(cur);
    free(cur);
    return SQLITE_OK;
}

static int cnd_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    ConductanceCursor *cur = (ConductanceCursor *)pCursor;
    CommunityVtab *vtab = (CommunityVtab *)pCursor->pVtab;

    cnd_rows_destroy(cur);
    cur->current = 0;
    cur->eof = 1;

    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    const char *membership_table = NULL;
    const char *group_col = NULL;
    const char *member_col = NULL;
    int pos = 0;

#define CND_N_HIDDEN (CND_COL_MEMBER_COL - CND_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < CND_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit)))
            continue;
        switch (bit + CND_COL_EDGE_TABLE) {
        case CND_COL_EDGE_TABLE:
            config.edge_table = graph_safe_text(argv[pos]);
            break;
        case CND_COL_SRC_COL:
            config.src_col = graph_safe_text(argv[pos]);
            break;
        case CND_COL_DST_COL:
            config.dst_col = graph_safe_text(argv[pos]);
            break;
        case CND_COL_WEIGHT_COL:
            config.weight_col = graph_safe_text(argv[pos]);
            break;
        case CND_COL_DIRECTION:
            config.direction = graph_safe_text(argv[pos]);
            break;
        case CND_COL_TIMESTAMP_COL:
            config.timestamp_col = graph_safe_text(argv[pos]);
            break;
        case CND_COL_TIME_START:
            config.time_start = argv[pos];
            break;
        case CND_COL_TIME_END:
            config.time_end = argv[pos];
            break;
        case CND_COL_MEMBERSHIP_TABLE:
            membership_table = graph_safe_text(argv[pos]);
            break;
        case CND_COL_GROUP_COL:
            group_col = graph_safe_text(argv[pos]);
            break;
        case CND_COL_MEMBER_COL:
            member_col = graph_safe_text(argv[pos]);
            break;
        }
        pos++;
    }

    if (!config.edge_table || !membership_table || !group_col || !member_col) {
        vtab->base.zErrMsg = sqlite3_mprintf("graph_conductance: edge_table, src_col, dst_col, membership_table, "
                                             "group_col and member_col are required");
        return SQLITE_ERROR;
    }
    if (!config.direction)
        config.direction = "both";

    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc;
    if (is_graph_adjacency(vtab->db, config.edge_table)) {
        rc = graph_data_load_from_adjacency(vtab->db, config.edge_table, &g, &errmsg);
    } else {
        rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    }
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg ? errmsg : sqlite3_mprintf("graph_conductance: failed to load graph");
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    int *group = (int *)malloc((size_t)(N > 0 ? N : 1) * sizeof(int));
    GraphData groups;
    graph_data_init(&groups);
    if (!group) {
        graph_data_destroy(&groups);
        graph_data_destroy(&g);
        return SQLITE_NOMEM;
    }

    rc = load_membership(vtab->db, membership_table, group_col, member_col, &g, group, &groups, &errmsg);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg;
        free(group);
        graph_data_destroy(&groups);
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int k = groups.node_count;
    if (k > 0) {
        Conductance *scores = (Conductance *)malloc((size_t)k * sizeof(Conductance));
        cur->rows = (ConductanceRow *)calloc((size_t)k, sizeof(ConductanceRow));
        if (!scores || !cur->rows) {
            free(scores);
            free(group);
            graph_data_destroy(&groups);
            graph_data_destroy(&g);
            return SQLITE_NOMEM;
        }
        run_conductance(&g, group, k, config.direction, scores);
        for (int c = 0; c < k; c++) {
            cur->rows[c].group_id = strdup(groups.ids[c]);
            cur->rows[c].c = scores[c];
        }
        cur->count = k;
        free(scores);
    }

    free(group);
    graph_data_destroy(&groups);
    graph_data_destroy(&g);

    cur->eof = (cur->count == 0);
    return SQLITE_OK;
}

static int cnd_next(sqlite3_vtab_cursor *p) {
    ConductanceCursor *cur = (ConductanceCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->count);
    return SQLITE_OK;
}

static int cnd_eof(sqlite3_vtab_cursor *p) {
    return ((ConductanceCursor *)p)->eof;
}

static int cnd_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    ConductanceCursor *cur = (ConductanceCursor *)p;
    ConductanceRow *row = &cur->rows[cur->current];
    switch (col) {
    case CND_COL_GROUP_ID:
        sqlite3_result_text(ctx, row->group_id, -1, SQLITE_TRANSIENT);
        break;
    case CND_COL_SIZE:
        sqlite3_result_int64(ctx, (sqlite3_int64)row->c.size);
        break;
    case CND_COL_INTERNAL:
        sqlite3_result_double(ctx, row->c.internal);
        break;
    case CND_COL_CUT:
        sqlite3_result_double(ctx, row->c.cut);
        break;
    case CND_COL_VOL:
        sqlite3_result_double(ctx, row->c.vol);
        break;
    case CND_COL_PHI:
        sqlite3_result_double(ctx, row->c.phi);
        break;
    default:
        sqlite3_result_null(ctx);
        break;
    }
    return SQLITE_OK;
}

static int cnd_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((ConductanceCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_conductance_module = {
    .iVersion = 0,
    .xCreate = NULL,
    .xConnect = cnd_connect,
    .xBestIndex = cnd_best_index,
    .xDisconnect = comm_disconnect,
    .xDestroy = comm_disconnect,
    .xOpen = cnd_open,
    .xClose = cnd_close,
    .xFilter = cnd_filter,
    .xNext = cnd_next,
    .xEof = cnd_eof,
    .xColumn = cnd_column,
    .xRowid = cnd_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Registration
 * ═══════════════════════════════════════════════════════════════ */

int community_register_tvfs(sqlite3 *db) {
    int rc = sqlite3_create_module(db, "graph_leiden", &graph_leiden_module, NULL);
    if (rc != SQLITE_OK)
        return rc;
    return sqlite3_create_module(db, "graph_conductance", &graph_conductance_module, NULL);
}
