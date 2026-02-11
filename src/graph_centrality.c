/*
 * graph_centrality.c — Centrality measure table-valued functions
 *
 * TVFs:
 *   graph_degree      — in/out/total degree centrality (O(V+E))
 *   graph_betweenness — Brandes (2001) betweenness centrality
 *   graph_closeness   — closeness centrality with Wasserman-Faust normalization
 *
 * All TVFs use the shared graph_load module for graph loading with
 * support for weighted edges and temporal filtering.
 */
#include "graph_centrality.h"
#include "graph_common.h"
#include "graph_load.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

SQLITE_EXTENSION_INIT3

/* ═══════════════════════════════════════════════════════════════
 * Shared result structure for centrality TVFs
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char *node;
    double centrality;
    double in_degree;    /* only used by degree TVF */
    double out_degree;   /* only used by degree TVF */
    double degree;       /* only used by degree TVF */
} CentralityRow;

typedef struct {
    CentralityRow *rows;
    int count;
    int capacity;
} CentralityResults;

static void cr_init(CentralityResults *r) {
    r->count = 0;
    r->capacity = 64;
    r->rows = (CentralityRow *)calloc((size_t)r->capacity, sizeof(CentralityRow));
}

static void cr_destroy(CentralityResults *r) {
    for (int i = 0; i < r->count; i++) free(r->rows[i].node);
    free(r->rows);
    r->rows = NULL;
    r->count = 0;
}

static void cr_add(CentralityResults *r, const char *node, double centrality,
                   double in_deg, double out_deg, double deg) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (CentralityRow *)realloc(r->rows,
                    (size_t)r->capacity * sizeof(CentralityRow));
    }
    CentralityRow *row = &r->rows[r->count++];
    row->node = strdup(node);
    row->centrality = centrality;
    row->in_degree = in_deg;
    row->out_degree = out_deg;
    row->degree = deg;
}

/* ═══════════════════════════════════════════════════════════════
 * Shared vtab structure (db handle only)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} CentralityVtab;

static int cent_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * Helper: parse common graph config from TVF argv
 *
 * Expects argv layout: edge_table, src_col, dst_col, weight_col,
 *   <algo-specific params...>, direction, timestamp_col,
 *   time_start, time_end
 *
 * Each TVF defines its own column enum and calls this to
 * fill a GraphLoadConfig and extract additional params.
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * Double-precision priority queue for Dijkstra/Brandes
 *
 * The shared PriorityQueue uses float (for HNSW beam search).
 * Brandes needs double for accurate shortest-path counting with
 * tie detection (new_dist == d[w]).
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int node;
    double dist;
} DPQEntry;

typedef struct {
    DPQEntry *entries;
    int size;
    int capacity;
} DoublePQ;

static void dpq_init(DoublePQ *pq, int cap) {
    pq->entries = (DPQEntry *)malloc((size_t)cap * sizeof(DPQEntry));
    pq->size = 0;
    pq->capacity = cap;
}

static void dpq_destroy(DoublePQ *pq) {
    free(pq->entries);
}

static void dpq_push(DoublePQ *pq, int node, double dist) {
    if (pq->size >= pq->capacity) {
        pq->capacity *= 2;
        pq->entries = (DPQEntry *)realloc(pq->entries,
                        (size_t)pq->capacity * sizeof(DPQEntry));
    }
    int i = pq->size++;
    pq->entries[i].node = node;
    pq->entries[i].dist = dist;
    /* Sift up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->entries[parent].dist <= pq->entries[i].dist) break;
        DPQEntry tmp = pq->entries[parent];
        pq->entries[parent] = pq->entries[i];
        pq->entries[i] = tmp;
        i = parent;
    }
}

static DPQEntry dpq_pop(DoublePQ *pq) {
    DPQEntry top = pq->entries[0];
    pq->size--;
    if (pq->size > 0) {
        pq->entries[0] = pq->entries[pq->size];
        int i = 0;
        for (;;) {
            int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
            if (left < pq->size && pq->entries[left].dist < pq->entries[smallest].dist)
                smallest = left;
            if (right < pq->size && pq->entries[right].dist < pq->entries[smallest].dist)
                smallest = right;
            if (smallest == i) break;
            DPQEntry tmp = pq->entries[i];
            pq->entries[i] = pq->entries[smallest];
            pq->entries[smallest] = tmp;
            i = smallest;
        }
    }
    return top;
}

/* Epsilon comparison for double shortest-path tie detection */
static int double_eq(double a, double b) {
    return fabs(a - b) < 1e-10 * fmax(1.0, fabs(b));
}

/* ═══════════════════════════════════════════════════════════════
 * Predecessor list (dynamic array of int arrays)
 * Used by Brandes for tracking shortest-path predecessors.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int *items;
    int count;
    int capacity;
} IntList;

static void intlist_init(IntList *l) {
    l->items = NULL;
    l->count = 0;
    l->capacity = 0;
}

static void intlist_push(IntList *l, int val) {
    if (l->count >= l->capacity) {
        l->capacity = l->capacity == 0 ? 4 : l->capacity * 2;
        l->items = (int *)realloc(l->items, (size_t)l->capacity * sizeof(int));
    }
    l->items[l->count++] = val;
}

static void intlist_clear(IntList *l) {
    l->count = 0;
}

static void intlist_destroy(IntList *l) {
    free(l->items);
    l->items = NULL;
    l->count = l->capacity = 0;
}

/* ═══════════════════════════════════════════════════════════════
 * SSSP helpers shared by betweenness and closeness
 * ═══════════════════════════════════════════════════════════════ */

/*
 * BFS-based SSSP (unweighted graphs).
 * Sets dist[v] = shortest distance from source, sigma[v] = number of
 * shortest paths, and pred[v] = predecessor list. Also fills stack
 * in order of non-decreasing distance (for Brandes back-propagation).
 */
static void sssp_bfs(const GraphData *g, int source,
                     double *dist, double *sigma, IntList *pred,
                     int *stack, int *stack_size,
                     const char *direction) {
    int N = g->node_count;
    for (int i = 0; i < N; i++) {
        dist[i] = -1.0;
        sigma[i] = 0.0;
        intlist_clear(&pred[i]);
    }
    dist[source] = 0.0;
    sigma[source] = 1.0;
    *stack_size = 0;

    /* BFS queue */
    int *queue = (int *)malloc((size_t)N * sizeof(int));
    int qhead = 0, qtail = 0;
    queue[qtail++] = source;

    int use_out = !direction || strcmp(direction, "reverse") != 0;
    int use_in  = direction && (strcmp(direction, "reverse") == 0 ||
                                strcmp(direction, "both") == 0);

    while (qhead < qtail) {
        int v = queue[qhead++];
        stack[(*stack_size)++] = v;

        /* Expand neighbors */
        for (int pass = 0; pass < 2; pass++) {
            const GraphAdjList *adj = (pass == 0) ?
                (use_out ? &g->out[v] : NULL) :
                (use_in  ? &g->in[v]  : NULL);
            if (!adj) continue;

            for (int e = 0; e < adj->count; e++) {
                int w = adj->edges[e].target;
                /* First visit? */
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1.0;
                    queue[qtail++] = w;
                }
                /* Is this a shortest path?
                 * Skip if v is already a predecessor of w (duplicate
                 * edge from traversing both out[] and in[]). */
                if (double_eq(dist[w], dist[v] + 1.0)) {
                    if (pred[w].count == 0 ||
                        pred[w].items[pred[w].count - 1] != v) {
                        sigma[w] += sigma[v];
                        intlist_push(&pred[w], v);
                    }
                }
            }
        }
    }
    free(queue);
}

/*
 * Dijkstra-based SSSP (weighted graphs).
 * Same interface as sssp_bfs.
 */
static void sssp_dijkstra(const GraphData *g, int source,
                          double *dist, double *sigma, IntList *pred,
                          int *stack, int *stack_size,
                          const char *direction) {
    int N = g->node_count;
    for (int i = 0; i < N; i++) {
        dist[i] = -1.0;
        sigma[i] = 0.0;
        intlist_clear(&pred[i]);
    }
    dist[source] = 0.0;
    sigma[source] = 1.0;
    *stack_size = 0;

    int use_out = !direction || strcmp(direction, "reverse") != 0;
    int use_in  = direction && (strcmp(direction, "reverse") == 0 ||
                                strcmp(direction, "both") == 0);

    DoublePQ pq;
    dpq_init(&pq, N > 16 ? N : 16);
    dpq_push(&pq, source, 0.0);

    /* Settled flag prevents re-expansion of nodes popped multiple
     * times from the lazy-deletion priority queue. */
    int *settled = (int *)calloc((size_t)N, sizeof(int));

    while (pq.size > 0) {
        DPQEntry top = dpq_pop(&pq);
        int v = top.node;

        if (settled[v]) continue;
        settled[v] = 1;

        stack[(*stack_size)++] = v;

        for (int pass = 0; pass < 2; pass++) {
            const GraphAdjList *adj = (pass == 0) ?
                (use_out ? &g->out[v] : NULL) :
                (use_in  ? &g->in[v]  : NULL);
            if (!adj) continue;

            for (int e = 0; e < adj->count; e++) {
                int w = adj->edges[e].target;
                double new_dist = dist[v] + adj->edges[e].weight;

                if (dist[w] < 0 || new_dist < dist[w] - 1e-10) {
                    /* Shorter path found */
                    dist[w] = new_dist;
                    sigma[w] = sigma[v];
                    intlist_clear(&pred[w]);
                    intlist_push(&pred[w], v);
                    dpq_push(&pq, w, new_dist);
                } else if (double_eq(new_dist, dist[w])) {
                    /* Equally short path — skip if v is already a
                     * predecessor of w (duplicate from out[]+in[]). */
                    if (pred[w].count == 0 ||
                        pred[w].items[pred[w].count - 1] != v) {
                        sigma[w] += sigma[v];
                        intlist_push(&pred[w], v);
                    }
                }
            }
        }
    }
    free(settled);
    dpq_destroy(&pq);
}

/* ═══════════════════════════════════════════════════════════════
 * DEGREE CENTRALITY
 *
 * O(V+E) — iterate adjacency lists, sum weights per node.
 * ═══════════════════════════════════════════════════════════════ */

enum {
    DEG_COL_NODE = 0,
    DEG_COL_IN_DEGREE,
    DEG_COL_OUT_DEGREE,
    DEG_COL_DEGREE,
    DEG_COL_CENTRALITY,
    DEG_COL_EDGE_TABLE,    /* hidden */
    DEG_COL_SRC_COL,       /* hidden */
    DEG_COL_DST_COL,       /* hidden */
    DEG_COL_WEIGHT_COL,    /* hidden */
    DEG_COL_NORMALIZED,    /* hidden */
    DEG_COL_DIRECTION,     /* hidden */
    DEG_COL_TIMESTAMP_COL, /* hidden */
    DEG_COL_TIME_START,    /* hidden */
    DEG_COL_TIME_END,      /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    CentralityResults results;
    int current;
    int eof;
} DegreeCursor;

static int deg_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                       sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT, in_degree REAL, out_degree REAL, degree REAL, centrality REAL,"
        "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
        "  weight_col TEXT HIDDEN, normalized INTEGER HIDDEN,"
        "  direction TEXT HIDDEN, timestamp_col TEXT HIDDEN,"
        "  time_start HIDDEN, time_end HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    CentralityVtab *vtab = (CentralityVtab *)sqlite3_malloc(sizeof(CentralityVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(CentralityVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int deg_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    return graph_best_index_common(pIdxInfo, DEG_COL_EDGE_TABLE, DEG_COL_TIME_END, 0x7, 1000.0);
}

static int deg_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    DegreeCursor *cur = (DegreeCursor *)calloc(1, sizeof(DegreeCursor));
    if (!cur) return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int deg_close(sqlite3_vtab_cursor *pCursor) {
    DegreeCursor *cur = (DegreeCursor *)pCursor;
    cr_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int deg_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    DegreeCursor *cur = (DegreeCursor *)pCursor;
    CentralityVtab *vtab = (CentralityVtab *)pCursor->pVtab;

    cr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(CentralityResults));

    if (argc < 3) { cur->eof = 1; return SQLITE_OK; }

    /* Decode argv using idxNum bitmask — argv is in column order */
    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    int normalized = 0;
    int pos = 0;

    #define DEG_N_HIDDEN (DEG_COL_TIME_END - DEG_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < DEG_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit))) continue;
        switch (bit + DEG_COL_EDGE_TABLE) {
            case DEG_COL_EDGE_TABLE:    config.edge_table    = graph_safe_text(argv[pos]); break;
            case DEG_COL_SRC_COL:       config.src_col       = graph_safe_text(argv[pos]); break;
            case DEG_COL_DST_COL:       config.dst_col       = graph_safe_text(argv[pos]); break;
            case DEG_COL_WEIGHT_COL:    config.weight_col    = graph_safe_text(argv[pos]); break;
            case DEG_COL_NORMALIZED:    normalized = sqlite3_value_int(argv[pos]); break;
            case DEG_COL_DIRECTION:     config.direction     = graph_safe_text(argv[pos]); break;
            case DEG_COL_TIMESTAMP_COL: config.timestamp_col = graph_safe_text(argv[pos]); break;
            case DEG_COL_TIME_START:    config.time_start    = argv[pos]; break;
            case DEG_COL_TIME_END:      config.time_end      = argv[pos]; break;
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
            : sqlite3_mprintf("graph_degree: failed to load graph");
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    cr_init(&cur->results);

    for (int i = 0; i < N; i++) {
        double in_deg = 0, out_deg = 0;
        for (int e = 0; e < g.out[i].count; e++)
            out_deg += g.out[i].edges[e].weight;
        for (int e = 0; e < g.in[i].count; e++)
            in_deg += g.in[i].edges[e].weight;
        double total = in_deg + out_deg;

        double cent = total;
        if (normalized && N > 1) cent = total / (double)(N - 1);

        cr_add(&cur->results, g.ids[i], cent, in_deg, out_deg, total);
    }

    graph_data_destroy(&g);
    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int deg_next(sqlite3_vtab_cursor *p) {
    DegreeCursor *cur = (DegreeCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int deg_eof(sqlite3_vtab_cursor *p) {
    return ((DegreeCursor *)p)->eof;
}

static int deg_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    DegreeCursor *cur = (DegreeCursor *)p;
    CentralityRow *row = &cur->results.rows[cur->current];
    switch (col) {
        case DEG_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT); break;
        case DEG_COL_IN_DEGREE:
            sqlite3_result_double(ctx, row->in_degree); break;
        case DEG_COL_OUT_DEGREE:
            sqlite3_result_double(ctx, row->out_degree); break;
        case DEG_COL_DEGREE:
            sqlite3_result_double(ctx, row->degree); break;
        case DEG_COL_CENTRALITY:
            sqlite3_result_double(ctx, row->centrality); break;
        default:
            sqlite3_result_null(ctx); break;
    }
    return SQLITE_OK;
}

static int deg_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((DegreeCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_degree_module = {
    .iVersion = 0,
    .xCreate = NULL,  /* eponymous-only */
    .xConnect = deg_connect,
    .xBestIndex = deg_best_index,
    .xDisconnect = cent_disconnect,
    .xDestroy = cent_disconnect,
    .xOpen = deg_open,
    .xClose = deg_close,
    .xFilter = deg_filter,
    .xNext = deg_next,
    .xEof = deg_eof,
    .xColumn = deg_column,
    .xRowid = deg_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * BETWEENNESS CENTRALITY — Brandes (2001)
 *
 * Exact: O(VE) unweighted, O(VE + V² log V) weighted
 * Approx: sample ceil(sqrt(N)) sources when N > threshold
 * ═══════════════════════════════════════════════════════════════ */

enum {
    BET_COL_NODE = 0,
    BET_COL_CENTRALITY,
    BET_COL_EDGE_TABLE,    /* hidden */
    BET_COL_SRC_COL,       /* hidden */
    BET_COL_DST_COL,       /* hidden */
    BET_COL_WEIGHT_COL,    /* hidden */
    BET_COL_NORMALIZED,    /* hidden */
    BET_COL_DIRECTION,     /* hidden */
    BET_COL_AUTO_APPROX,   /* hidden */
    BET_COL_TIMESTAMP_COL, /* hidden */
    BET_COL_TIME_START,    /* hidden */
    BET_COL_TIME_END,      /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    CentralityResults results;
    int current;
    int eof;
} BetweennessCursor;

static int bet_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                       sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT, centrality REAL,"
        "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
        "  weight_col TEXT HIDDEN, normalized INTEGER HIDDEN,"
        "  direction TEXT HIDDEN, auto_approx_threshold INTEGER HIDDEN,"
        "  timestamp_col TEXT HIDDEN, time_start HIDDEN, time_end HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    CentralityVtab *vtab = (CentralityVtab *)sqlite3_malloc(sizeof(CentralityVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(CentralityVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int bet_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    return graph_best_index_common(pIdxInfo, BET_COL_EDGE_TABLE, BET_COL_TIME_END, 0x7, 1000.0);
}

static int bet_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    BetweennessCursor *cur = (BetweennessCursor *)calloc(1, sizeof(BetweennessCursor));
    if (!cur) return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int bet_close(sqlite3_vtab_cursor *pCursor) {
    BetweennessCursor *cur = (BetweennessCursor *)pCursor;
    cr_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int bet_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    BetweennessCursor *cur = (BetweennessCursor *)pCursor;
    CentralityVtab *vtab = (CentralityVtab *)pCursor->pVtab;

    cr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(CentralityResults));

    if (argc < 3) { cur->eof = 1; return SQLITE_OK; }

    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    int normalized = 0;
    int auto_approx = 50000;
    int pos = 0;

    #define BET_N_HIDDEN (BET_COL_TIME_END - BET_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < BET_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit))) continue;
        switch (bit + BET_COL_EDGE_TABLE) {
            case BET_COL_EDGE_TABLE:    config.edge_table    = graph_safe_text(argv[pos]); break;
            case BET_COL_SRC_COL:       config.src_col       = graph_safe_text(argv[pos]); break;
            case BET_COL_DST_COL:       config.dst_col       = graph_safe_text(argv[pos]); break;
            case BET_COL_WEIGHT_COL:    config.weight_col    = graph_safe_text(argv[pos]); break;
            case BET_COL_NORMALIZED:    normalized = sqlite3_value_int(argv[pos]); break;
            case BET_COL_DIRECTION:     config.direction     = graph_safe_text(argv[pos]); break;
            case BET_COL_AUTO_APPROX:   auto_approx = sqlite3_value_int(argv[pos]); break;
            case BET_COL_TIMESTAMP_COL: config.timestamp_col = graph_safe_text(argv[pos]); break;
            case BET_COL_TIME_START:    config.time_start    = argv[pos]; break;
            case BET_COL_TIME_END:      config.time_end      = argv[pos]; break;
        }
        pos++;
    }

    if (!config.direction) config.direction = "forward";

    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg ? errmsg
            : sqlite3_mprintf("graph_betweenness: failed to load graph");
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    if (N == 0) {
        graph_data_destroy(&g);
        cr_init(&cur->results);
        cur->eof = 1;
        return SQLITE_OK;
    }

    /* Allocate Brandes working arrays */
    double *CB     = (double *)calloc((size_t)N, sizeof(double));
    double *dist   = (double *)malloc((size_t)N * sizeof(double));
    double *sigma  = (double *)malloc((size_t)N * sizeof(double));
    double *delta  = (double *)malloc((size_t)N * sizeof(double));
    int    *stack  = (int *)malloc((size_t)N * sizeof(int));
    IntList *pred  = (IntList *)calloc((size_t)N, sizeof(IntList));
    for (int i = 0; i < N; i++) intlist_init(&pred[i]);

    /* Determine source set (exact vs approx) */
    int n_sources = N;
    int *sources = (int *)malloc((size_t)N * sizeof(int));
    double scale = 1.0;

    if (auto_approx > 0 && N > auto_approx) {
        /* Approximate: sample ceil(sqrt(N)) evenly spaced sources */
        n_sources = (int)ceil(sqrt((double)N));
        if (n_sources < 1) n_sources = 1;
        int step = N / n_sources;
        if (step < 1) step = 1;
        n_sources = 0;
        for (int i = 0; i < N && n_sources < (int)ceil(sqrt((double)N)); i += step) {
            sources[n_sources++] = i;
        }
        scale = (double)N / (double)n_sources;
    } else {
        for (int i = 0; i < N; i++) sources[i] = i;
    }

    /* Brandes main loop */
    int weighted = g.has_weights;
    for (int si = 0; si < n_sources; si++) {
        int s = sources[si];
        int stack_size = 0;

        if (weighted) {
            sssp_dijkstra(&g, s, dist, sigma, pred, stack, &stack_size,
                          config.direction);
        } else {
            sssp_bfs(&g, s, dist, sigma, pred, stack, &stack_size,
                     config.direction);
        }

        /* Backward accumulation */
        for (int i = 0; i < N; i++) delta[i] = 0.0;
        while (stack_size > 0) {
            int w = stack[--stack_size];
            for (int pi = 0; pi < pred[w].count; pi++) {
                int v = pred[w].items[pi];
                if (sigma[w] > 0)
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s) CB[w] += delta[w];
        }
    }

    /* Scale for approximation */
    if (scale != 1.0) {
        for (int i = 0; i < N; i++) CB[i] *= scale;
    }

    /* For undirected graphs (bidirectional edges), each shortest path
     * s->t is discovered from both endpoints, doubling the accumulation.
     * Halve to correct. */
    int undirected = config.direction && strcmp(config.direction, "both") == 0;
    if (undirected) {
        for (int i = 0; i < N; i++) CB[i] /= 2.0;
    }

    /* Normalize */
    if (normalized && N > 2) {
        double norm_factor;
        if (undirected) {
            norm_factor = (double)(N - 1) * (double)(N - 2) / 2.0;
        } else {
            norm_factor = (double)(N - 1) * (double)(N - 2);
        }
        for (int i = 0; i < N; i++) CB[i] /= norm_factor;
    }

    /* Build results */
    cr_init(&cur->results);
    for (int i = 0; i < N; i++) {
        cr_add(&cur->results, g.ids[i], CB[i], 0, 0, 0);
    }

    /* Cleanup */
    for (int i = 0; i < N; i++) intlist_destroy(&pred[i]);
    free(pred);
    free(CB);
    free(dist);
    free(sigma);
    free(delta);
    free(stack);
    free(sources);
    graph_data_destroy(&g);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int bet_next(sqlite3_vtab_cursor *p) {
    BetweennessCursor *cur = (BetweennessCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int bet_eof(sqlite3_vtab_cursor *p) {
    return ((BetweennessCursor *)p)->eof;
}

static int bet_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    BetweennessCursor *cur = (BetweennessCursor *)p;
    CentralityRow *row = &cur->results.rows[cur->current];
    switch (col) {
        case BET_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT); break;
        case BET_COL_CENTRALITY:
            sqlite3_result_double(ctx, row->centrality); break;
        default:
            sqlite3_result_null(ctx); break;
    }
    return SQLITE_OK;
}

static int bet_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((BetweennessCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_betweenness_module = {
    .iVersion = 0,
    .xCreate = NULL,
    .xConnect = bet_connect,
    .xBestIndex = bet_best_index,
    .xDisconnect = cent_disconnect,
    .xDestroy = cent_disconnect,
    .xOpen = bet_open,
    .xClose = bet_close,
    .xFilter = bet_filter,
    .xNext = bet_next,
    .xEof = bet_eof,
    .xColumn = bet_column,
    .xRowid = bet_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * CLOSENESS CENTRALITY
 *
 * C(v) = reachable / sum_dist, with Wasserman-Faust normalization:
 * C(v) *= reachable / (N - 1)
 * ═══════════════════════════════════════════════════════════════ */

enum {
    CLO_COL_NODE = 0,
    CLO_COL_CENTRALITY,
    CLO_COL_EDGE_TABLE,    /* hidden */
    CLO_COL_SRC_COL,       /* hidden */
    CLO_COL_DST_COL,       /* hidden */
    CLO_COL_WEIGHT_COL,    /* hidden */
    CLO_COL_NORMALIZED,    /* hidden */
    CLO_COL_DIRECTION,     /* hidden */
    CLO_COL_TIMESTAMP_COL, /* hidden */
    CLO_COL_TIME_START,    /* hidden */
    CLO_COL_TIME_END,      /* hidden */
};

typedef struct {
    sqlite3_vtab_cursor base;
    CentralityResults results;
    int current;
    int eof;
} ClosenessCursor;

static int clo_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                       sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux; (void)argc; (void)argv; (void)pzErr;
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  node TEXT, centrality REAL,"
        "  edge_table TEXT HIDDEN, src_col TEXT HIDDEN, dst_col TEXT HIDDEN,"
        "  weight_col TEXT HIDDEN, normalized INTEGER HIDDEN,"
        "  direction TEXT HIDDEN, timestamp_col TEXT HIDDEN,"
        "  time_start HIDDEN, time_end HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    CentralityVtab *vtab = (CentralityVtab *)sqlite3_malloc(sizeof(CentralityVtab));
    if (!vtab) return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(CentralityVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int clo_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    return graph_best_index_common(pIdxInfo, CLO_COL_EDGE_TABLE, CLO_COL_TIME_END, 0x7, 1000.0);
}

static int clo_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    ClosenessCursor *cur = (ClosenessCursor *)calloc(1, sizeof(ClosenessCursor));
    if (!cur) return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int clo_close(sqlite3_vtab_cursor *pCursor) {
    ClosenessCursor *cur = (ClosenessCursor *)pCursor;
    cr_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int clo_filter(sqlite3_vtab_cursor *pCursor, int idxNum,
                      const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    ClosenessCursor *cur = (ClosenessCursor *)pCursor;
    CentralityVtab *vtab = (CentralityVtab *)pCursor->pVtab;

    cr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(CentralityResults));

    if (argc < 3) { cur->eof = 1; return SQLITE_OK; }

    GraphLoadConfig config;
    memset(&config, 0, sizeof(config));
    int normalized = 1;  /* default ON for closeness */
    int pos = 0;

    #define CLO_N_HIDDEN (CLO_COL_TIME_END - CLO_COL_EDGE_TABLE + 1)
    for (int bit = 0; bit < CLO_N_HIDDEN && pos < argc; bit++) {
        if (!(idxNum & (1 << bit))) continue;
        switch (bit + CLO_COL_EDGE_TABLE) {
            case CLO_COL_EDGE_TABLE:    config.edge_table    = graph_safe_text(argv[pos]); break;
            case CLO_COL_SRC_COL:       config.src_col       = graph_safe_text(argv[pos]); break;
            case CLO_COL_DST_COL:       config.dst_col       = graph_safe_text(argv[pos]); break;
            case CLO_COL_WEIGHT_COL:    config.weight_col    = graph_safe_text(argv[pos]); break;
            case CLO_COL_NORMALIZED:    normalized = sqlite3_value_int(argv[pos]); break;
            case CLO_COL_DIRECTION:     config.direction     = graph_safe_text(argv[pos]); break;
            case CLO_COL_TIMESTAMP_COL: config.timestamp_col = graph_safe_text(argv[pos]); break;
            case CLO_COL_TIME_START:    config.time_start    = argv[pos]; break;
            case CLO_COL_TIME_END:      config.time_end      = argv[pos]; break;
        }
        pos++;
    }

    if (!config.direction) config.direction = "forward";

    GraphData g;
    graph_data_init(&g);
    char *errmsg = NULL;
    int rc = graph_data_load(vtab->db, &config, &g, &errmsg);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = errmsg ? errmsg
            : sqlite3_mprintf("graph_closeness: failed to load graph");
        graph_data_destroy(&g);
        return SQLITE_ERROR;
    }

    int N = g.node_count;
    if (N == 0) {
        graph_data_destroy(&g);
        cr_init(&cur->results);
        cur->eof = 1;
        return SQLITE_OK;
    }

    /* Allocate SSSP working arrays (reused across sources) */
    double *dist  = (double *)malloc((size_t)N * sizeof(double));
    double *sigma = (double *)malloc((size_t)N * sizeof(double));
    int    *stack = (int *)malloc((size_t)N * sizeof(int));
    IntList *pred = (IntList *)calloc((size_t)N, sizeof(IntList));
    for (int i = 0; i < N; i++) intlist_init(&pred[i]);

    double *closeness = (double *)calloc((size_t)N, sizeof(double));
    int weighted = g.has_weights;

    for (int s = 0; s < N; s++) {
        int stack_size = 0;

        if (weighted) {
            sssp_dijkstra(&g, s, dist, sigma, pred, stack, &stack_size,
                          config.direction);
        } else {
            sssp_bfs(&g, s, dist, sigma, pred, stack, &stack_size,
                     config.direction);
        }

        /* Sum distances and count reachable nodes */
        double sum_dist = 0.0;
        int reachable = 0;
        for (int i = 0; i < N; i++) {
            if (i != s && dist[i] >= 0) {
                sum_dist += dist[i];
                reachable++;
            }
        }

        if (reachable > 0 && sum_dist > 0) {
            double cc = (double)reachable / sum_dist;
            if (normalized && N > 1) {
                /* Wasserman-Faust: scale by reachable/(N-1) */
                cc *= (double)reachable / (double)(N - 1);
            }
            closeness[s] = cc;
        }
    }

    /* Build results */
    cr_init(&cur->results);
    for (int i = 0; i < N; i++) {
        cr_add(&cur->results, g.ids[i], closeness[i], 0, 0, 0);
    }

    /* Cleanup */
    for (int i = 0; i < N; i++) intlist_destroy(&pred[i]);
    free(pred);
    free(dist);
    free(sigma);
    free(stack);
    free(closeness);
    graph_data_destroy(&g);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int clo_next(sqlite3_vtab_cursor *p) {
    ClosenessCursor *cur = (ClosenessCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int clo_eof(sqlite3_vtab_cursor *p) {
    return ((ClosenessCursor *)p)->eof;
}

static int clo_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    ClosenessCursor *cur = (ClosenessCursor *)p;
    CentralityRow *row = &cur->results.rows[cur->current];
    switch (col) {
        case CLO_COL_NODE:
            sqlite3_result_text(ctx, row->node, -1, SQLITE_TRANSIENT); break;
        case CLO_COL_CENTRALITY:
            sqlite3_result_double(ctx, row->centrality); break;
        default:
            sqlite3_result_null(ctx); break;
    }
    return SQLITE_OK;
}

static int clo_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((ClosenessCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module graph_closeness_module = {
    .iVersion = 0,
    .xCreate = NULL,
    .xConnect = clo_connect,
    .xBestIndex = clo_best_index,
    .xDisconnect = cent_disconnect,
    .xDestroy = cent_disconnect,
    .xOpen = clo_open,
    .xClose = clo_close,
    .xFilter = clo_filter,
    .xNext = clo_next,
    .xEof = clo_eof,
    .xColumn = clo_column,
    .xRowid = clo_rowid,
};

/* ═══════════════════════════════════════════════════════════════
 * Registration
 * ═══════════════════════════════════════════════════════════════ */

int centrality_register_tvfs(sqlite3 *db) {
    int rc;

    rc = sqlite3_create_module(db, "graph_degree", &graph_degree_module, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_betweenness", &graph_betweenness_module, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_create_module(db, "graph_closeness", &graph_closeness_module, NULL);
    if (rc != SQLITE_OK) return rc;

    return SQLITE_OK;
}
