/*
 * graph_community.c — Community detection table-valued functions
 *
 * TVFs:
 *   graph_leiden — Leiden community detection (Traag et al., 2019)
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
    for (int i = 0; i < r->count; i++) free(r->rows[i].node);
    free(r->rows);
    r->rows = NULL;
    r->count = 0;
}

static void comr_add(CommunityResults *r, const char *node, int comm, double mod) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (CommunityRow *)realloc(r->rows,
                    (size_t)r->capacity * sizeof(CommunityRow));
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
static double weight_to_community(const GraphData *g, int v, const int *community,
                                  int target_comm, int use_both) {
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
static double compute_modularity(const GraphData *g, const int *community,
                                 double resolution, double m, int use_both) {
    int N = g->node_count;
    if (m <= 0) return 0.0;

    /* Compute sum_in and sum_tot per community */
    /* sum_in = sum of weights within community */
    /* sum_tot = sum of degrees of nodes in community */
    int max_comm = 0;
    for (int i = 0; i < N; i++) {
        if (community[i] > max_comm) max_comm = community[i];
    }
    int n_comm = max_comm + 1;
    double *sum_in  = (double *)calloc((size_t)n_comm, sizeof(double));
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
static int leiden_local_moving(const GraphData *g, int *community,
                               double *sum_tot, double *k,
                               double m, double resolution, int use_both) {
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
                    if (neigh_comms[j] == nc) { seen = 1; break; }
                }
                if (!seen) neigh_comms[n_neigh++] = nc;
            }
            if (use_both) {
                for (int e = 0; e < g->in[v].count; e++) {
                    int nc = community[g->in[v].edges[e].target];
                    int seen = 0;
                    for (int j = 0; j < n_neigh; j++) {
                        if (neigh_comms[j] == nc) { seen = 1; break; }
                    }
                    if (!seen) neigh_comms[n_neigh++] = nc;
                }
            }

            for (int j = 0; j < n_neigh; j++) {
                int target_comm = neigh_comms[j];
                if (target_comm == old_comm) continue;

                double k_v_to_target = weight_to_community(g, v, community, target_comm, use_both);

                /* Modularity gain from moving v: old_comm -> target_comm */
                double gain = (k_v_to_target - k_v_to_old) / m
                    + resolution * k_v * (sum_tot[old_comm] - k_v - sum_tot[target_comm])
                      / (2.0 * m * m);

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
static void leiden_refinement(const GraphData *g, const int *partition,
                              int *refined, double *k, double m,
                              double resolution, int use_both) {
    int N = g->node_count;

    /* Start with singletons */
    for (int i = 0; i < N; i++) refined[i] = i;

    /* Compute sum_tot for refined communities */
    double *r_sum_tot = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++) r_sum_tot[i] = k[i];

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
                if (partition[w] != part_comm) continue;
                int nr = refined[w];
                if (nr == old_ref) continue;

                double k_v_to_nr = weight_to_community(g, v, refined, nr, use_both);
                double gain = (k_v_to_nr - k_v_to_old) / m
                    + resolution * k_v * (r_sum_tot[old_ref] - k_v - r_sum_tot[nr])
                      / (2.0 * m * m);

                if (gain > best_gain) {
                    best_gain = gain;
                    best_ref = nr;
                }
            }
            if (use_both) {
                for (int e = 0; e < g->in[v].count; e++) {
                    int w = g->in[v].edges[e].target;
                    if (partition[w] != part_comm) continue;
                    int nr = refined[w];
                    if (nr == old_ref) continue;

                    double k_v_to_nr = weight_to_community(g, v, refined, nr, use_both);
                    double gain = (k_v_to_nr - k_v_to_old) / m
                        + resolution * k_v * (r_sum_tot[old_ref] - k_v - r_sum_tot[nr])
                          / (2.0 * m * m);

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
    for (int i = 0; i < N; i++) mapping[i] = -1;
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
static double run_leiden(const GraphData *g, int *community,
                         double resolution, const char *direction) {
    int N = g->node_count;
    if (N == 0) return 0.0;

    int use_both = direction && strcmp(direction, "both") == 0;

    /* Compute total edge weight (m) and per-node weighted degree */
    double *k = (double *)malloc((size_t)N * sizeof(double));
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        k[i] = weighted_degree(g, i, use_both);
        m += k[i];
    }
    m /= 2.0;  /* each edge counted twice */
    if (m <= 0.0) {
        free(k);
        for (int i = 0; i < N; i++) community[i] = i;
        return 0.0;
    }

    /* Initialize: each node in its own community */
    for (int i = 0; i < N; i++) community[i] = i;

    /* Compute initial sum_tot per community */
    double *sum_tot = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++) sum_tot[i] = k[i];

    int *refined = (int *)malloc((size_t)N * sizeof(int));
    int max_iter = 100;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Phase 1: Local moving */
        int moves = leiden_local_moving(g, community, sum_tot, k, m,
                                        resolution, use_both);
        if (moves == 0) break;

        /* Phase 2: Refinement */
        leiden_refinement(g, community, refined, k, m, resolution, use_both);

        /* Use refined partition as the community assignment */
        memcpy(community, refined, (size_t)N * sizeof(int));

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

static int lei_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                       sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT, community_id INTEGER, modularity REAL,"
        "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
        "  weight_col TEXT HIDDEN, resolution REAL HIDDEN,"
        "  direction TEXT HIDDEN, timestamp_col TEXT HIDDEN,"
        "  time_start HIDDEN, time_end HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    CommunityVtab *vtab = (CommunityVtab *)sqlite3_malloc(sizeof(CommunityVtab));
    if (!vtab) return SQLITE_NOMEM;
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
    if (!cur) return SQLITE_NOMEM;
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

static int lei_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    LeidenCursor *cur = (LeidenCursor *)pCursor;
    CommunityVtab *vtab = (CommunityVtab *)pCursor->pVtab;

    comr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(CommunityResults));

    if (argc < 3) { cur->eof = 1; return SQLITE_OK; }

    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    double resolution = 1.0;
    int pos = 0;

    #define LEI_N_HIDDEN (LEI_COL_TIME_END - LEI_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < LEI_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit))) continue;
        switch (bit + LEI_COL_EDGE_TABLE) {
            case LEI_COL_EDGE_TABLE:    config.edge_table    = graph_safe_text(argv[pos]); break;
            case LEI_COL_SRC_COL:       config.src_col       = graph_safe_text(argv[pos]); break;
            case LEI_COL_DST_COL:       config.dst_col       = graph_safe_text(argv[pos]); break;
            case LEI_COL_WEIGHT_COL:    config.weight_col    = graph_safe_text(argv[pos]); break;
            case LEI_COL_RESOLUTION:    resolution = sqlite3_value_double(argv[pos]); break;
            case LEI_COL_DIRECTION:     config.direction     = graph_safe_text(argv[pos]); break;
            case LEI_COL_TIMESTAMP_COL: config.timestamp_col = graph_safe_text(argv[pos]); break;
            case LEI_COL_TIME_START:    config.time_start    = argv[pos]; break;
            case LEI_COL_TIME_END:      config.time_end      = argv[pos]; break;
        }
        pos++;
    }

    if (!config.direction) config.direction = "both";

    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg ? errmsg
            : sqlite3_mprintf("graph_leiden: failed to load graph");
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
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT); break;
        case LEI_COL_COMMUNITY_ID:
            sqlite3_result_int(ctx, row->community_id); break;
        case LEI_COL_MODULARITY:
            sqlite3_result_double(ctx, row->modularity); break;
        default:
            sqlite3_result_null(ctx); break;
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
 * Registration
 * ═══════════════════════════════════════════════════════════════ */

int community_register_tvfs(sqlite3 *db) {
    return sqlite3_create_module(db, "graph_leiden", &graph_leiden_module, NULL);
}
