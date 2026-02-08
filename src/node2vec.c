/*
 * node2vec.c — Node2Vec graph embeddings
 *
 * Implements the Node2Vec pipeline:
 *   1. Load edges from any SQLite table → in-memory adjacency list
 *   2. Biased random walks with p,q parameters (Grover & Leskovec 2016)
 *   3. Skip-gram with Negative Sampling (Mikolov et al. 2013)
 *   4. Insert learned embeddings into a target HNSW virtual table
 *
 * Exposed as: SELECT node2vec_train(edge_table, src_col, dst_col,
 *                                    output_table, dimensions, p, q,
 *                                    num_walks, walk_length, window,
 *                                    neg_samples, learning_rate, epochs);
 */
#include "node2vec.h"
#include "id_validate.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

SQLITE_EXTENSION_INIT3

/* ─── PRNG (xorshift32) ─────────────────────────────────────── */

static unsigned int n2v_xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double n2v_rand(unsigned int *state) {
    return (double)n2v_xorshift32(state) / (double)0xFFFFFFFFu;
}

/* ─── Adjacency List ─────────────────────────────────────────── */

typedef struct {
    int *neighbors;   /* indices into the global node array */
    int count;
    int capacity;
} AdjList;

typedef struct {
    char **node_ids;   /* string IDs, owned */
    int node_count;
    int node_capacity;
    AdjList *adj;      /* adj[i] = neighbors of node i */
} Graph;

static void graph_init(Graph *g) {
    g->node_count = 0;
    g->node_capacity = 256;
    g->node_ids = (char **)calloc((size_t)g->node_capacity, sizeof(char *));
    g->adj = (AdjList *)calloc((size_t)g->node_capacity, sizeof(AdjList));
}

static void graph_destroy(Graph *g) {
    for (int i = 0; i < g->node_count; i++) {
        free(g->node_ids[i]);
        free(g->adj[i].neighbors);
    }
    free(g->node_ids);
    free(g->adj);
}

/* Find or insert a node, returning its index */
static int graph_node_index(Graph *g, const char *id) {
    /* Linear scan — fine for graphs up to ~10K nodes */
    for (int i = 0; i < g->node_count; i++) {
        if (strcmp(g->node_ids[i], id) == 0) return i;
    }
    /* Insert new node */
    if (g->node_count >= g->node_capacity) {
        int new_cap = g->node_capacity * 2;
        g->node_ids = (char **)realloc(g->node_ids, (size_t)new_cap * sizeof(char *));
        g->adj = (AdjList *)realloc(g->adj, (size_t)new_cap * sizeof(AdjList));
        for (int i = g->node_capacity; i < new_cap; i++) {
            g->node_ids[i] = NULL;
            g->adj[i].neighbors = NULL;
            g->adj[i].count = 0;
            g->adj[i].capacity = 0;
        }
        g->node_capacity = new_cap;
    }
    int idx = g->node_count++;
    g->node_ids[idx] = strdup(id);
    return idx;
}

static void graph_add_edge(Graph *g, int src_idx, int dst_idx) {
    AdjList *a = &g->adj[src_idx];
    /* Check for duplicate */
    for (int i = 0; i < a->count; i++) {
        if (a->neighbors[i] == dst_idx) return;
    }
    if (a->count >= a->capacity) {
        int new_cap = a->capacity == 0 ? 8 : a->capacity * 2;
        a->neighbors = (int *)realloc(a->neighbors, (size_t)new_cap * sizeof(int));
        a->capacity = new_cap;
    }
    a->neighbors[a->count++] = dst_idx;
}

/* Load edges from SQLite table into Graph */
static int graph_load_edges(sqlite3 *db, Graph *g,
                             const char *edge_table,
                             const char *src_col,
                             const char *dst_col) {
    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\"",
                                 src_col, dst_col, edge_table);
    if (!sql) return SQLITE_NOMEM;

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) return rc;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *src = (const char *)sqlite3_column_text(stmt, 0);
        const char *dst = (const char *)sqlite3_column_text(stmt, 1);
        if (!src || !dst) continue;

        int si = graph_node_index(g, src);
        int di = graph_node_index(g, dst);

        /* Undirected: add both directions */
        graph_add_edge(g, si, di);
        graph_add_edge(g, di, si);
    }
    sqlite3_finalize(stmt);
    return SQLITE_OK;
}

/* ─── Biased Random Walks ────────────────────────────────────── */

/*
 * Node2Vec biased random walk (Grover & Leskovec, KDD 2016).
 *
 * Second-order Markov chain: the transition probability from node u
 * to neighbor x depends on the previous node t:
 *   - 1/p  if x == t         (return to previous)
 *   - 1    if x is neighbor of t  (stay local)
 *   - 1/q  if x is not neighbor of t  (explore far)
 *
 * p controls "return" tendency, q controls "exploration" tendency.
 * p=q=1 reduces to DeepWalk (uniform random walk).
 */
static int is_neighbor(const Graph *g, int node, int target) {
    const AdjList *a = &g->adj[node];
    for (int i = 0; i < a->count; i++) {
        if (a->neighbors[i] == target) return 1;
    }
    return 0;
}

/*
 * Generate one biased random walk starting at start_node.
 * walk: output array of size walk_length (caller allocates).
 * Returns actual walk length (may be < walk_length if node has no neighbors).
 */
static int biased_walk(const Graph *g, int start_node,
                        double p, double q,
                        int walk_length,
                        int *walk,
                        unsigned int *rng) {
    walk[0] = start_node;
    if (g->adj[start_node].count == 0) return 1;

    /* First step: uniform random */
    int idx = (int)(n2v_rand(rng) * g->adj[start_node].count);
    if (idx >= g->adj[start_node].count) idx = g->adj[start_node].count - 1;
    walk[1] = g->adj[start_node].neighbors[idx];

    for (int step = 2; step < walk_length; step++) {
        int cur = walk[step - 1];
        int prev = walk[step - 2];
        const AdjList *cur_adj = &g->adj[cur];

        if (cur_adj->count == 0) return step;

        /* Compute unnormalized transition weights */
        double total_weight = 0.0;
        for (int i = 0; i < cur_adj->count; i++) {
            int x = cur_adj->neighbors[i];
            double w;
            if (x == prev) {
                w = 1.0 / p;       /* return */
            } else if (is_neighbor(g, prev, x)) {
                w = 1.0;           /* stay local */
            } else {
                w = 1.0 / q;       /* explore */
            }
            total_weight += w;
        }

        /* Sample proportional to weights */
        double r = n2v_rand(rng) * total_weight;
        double cumulative = 0.0;
        int chosen = cur_adj->neighbors[0];  /* fallback */
        for (int i = 0; i < cur_adj->count; i++) {
            int x = cur_adj->neighbors[i];
            double w;
            if (x == prev) {
                w = 1.0 / p;
            } else if (is_neighbor(g, prev, x)) {
                w = 1.0;
            } else {
                w = 1.0 / q;
            }
            cumulative += w;
            if (r <= cumulative) {
                chosen = x;
                break;
            }
        }
        walk[step] = chosen;
    }
    return walk_length;
}

/* ─── SGNS Training Kernel ───────────────────────────────────── */

/*
 * Skip-gram with Negative Sampling (Mikolov et al. 2013).
 *
 * Two embedding matrices:
 *   syn0[N][dim] — input (target) embeddings → these are the final output
 *   syn1neg[N][dim] — output (context) embeddings → discarded after training
 *
 * For each (center, context) pair from walks:
 *   1. positive: dot(syn0[center], syn1neg[context]) → sigmoid → gradient
 *   2. negative: sample neg_samples random nodes, push embeddings apart
 *
 * Pre-computed sigmoid LUT for speed.
 */

#define SIGMOID_TABLE_SIZE 1000
#define MAX_SIGMOID 6.0f

static float sigmoid_table[SIGMOID_TABLE_SIZE + 1];
static int sigmoid_table_initialized = 0;

static void init_sigmoid_table(void) {
    if (sigmoid_table_initialized) return;
    for (int i = 0; i <= SIGMOID_TABLE_SIZE; i++) {
        float x = (float)i / (float)SIGMOID_TABLE_SIZE * 2.0f * MAX_SIGMOID - MAX_SIGMOID;
        sigmoid_table[i] = 1.0f / (1.0f + expf(-x));
    }
    sigmoid_table_initialized = 1;
}

static float fast_sigmoid(float x) {
    if (x >= MAX_SIGMOID) return 1.0f;
    if (x <= -MAX_SIGMOID) return 0.0f;
    int idx = (int)((x + MAX_SIGMOID) / (2.0f * MAX_SIGMOID) * SIGMOID_TABLE_SIZE);
    if (idx < 0) idx = 0;
    if (idx > SIGMOID_TABLE_SIZE) idx = SIGMOID_TABLE_SIZE;
    return sigmoid_table[idx];
}

/* Negative sampling table: sample proportional to freq^0.75 */
#define NEG_TABLE_SIZE 100000

typedef struct {
    float *syn0;       /* input embeddings: N * dim */
    float *syn1neg;    /* output embeddings: N * dim */
    int *neg_table;    /* negative sampling distribution */
    int N;             /* number of nodes */
    int dim;           /* embedding dimension */
} SgnsModel;

static void build_neg_table(SgnsModel *model, const Graph *g) {
    /* Total freq^0.75 */
    double total = 0.0;
    for (int i = 0; i < model->N; i++) {
        total += pow((double)(g->adj[i].count + 1), 0.75);
    }

    int idx = 0;
    double cumulative = 0.0;
    for (int i = 0; i < model->N && idx < NEG_TABLE_SIZE; i++) {
        cumulative += pow((double)(g->adj[i].count + 1), 0.75) / total;
        while (idx < NEG_TABLE_SIZE && (double)idx / NEG_TABLE_SIZE < cumulative) {
            model->neg_table[idx++] = i;
        }
    }
    /* Fill remaining (rounding) */
    while (idx < NEG_TABLE_SIZE) {
        model->neg_table[idx++] = model->N - 1;
    }
}

static SgnsModel *sgns_create(int N, int dim, const Graph *g,
                               unsigned int *rng) {
    SgnsModel *m = (SgnsModel *)calloc(1, sizeof(SgnsModel));
    if (!m) return NULL;
    m->N = N;
    m->dim = dim;

    m->syn0 = (float *)malloc((size_t)N * (size_t)dim * sizeof(float));
    m->syn1neg = (float *)calloc((size_t)N * (size_t)dim, sizeof(float));
    m->neg_table = (int *)malloc(NEG_TABLE_SIZE * sizeof(int));
    if (!m->syn0 || !m->syn1neg || !m->neg_table) {
        free(m->syn0); free(m->syn1neg); free(m->neg_table); free(m);
        return NULL;
    }

    /* Initialize syn0 with small random values */
    for (int i = 0; i < N * dim; i++) {
        m->syn0[i] = ((float)n2v_rand(rng) - 0.5f) / (float)dim;
    }

    build_neg_table(m, g);
    return m;
}

static void sgns_destroy(SgnsModel *m) {
    if (!m) return;
    free(m->syn0);
    free(m->syn1neg);
    free(m->neg_table);
    free(m);
}

/*
 * Train one (center, context) pair with negative sampling.
 * Updates syn0[center] and syn1neg[context + negative samples].
 */
static void sgns_train_pair(SgnsModel *m, int center, int context,
                             int neg_samples, float lr,
                             unsigned int *rng) {
    float *vec_center = m->syn0 + (size_t)center * (size_t)m->dim;
    float *neu1e = (float *)calloc((size_t)m->dim, sizeof(float));  /* gradient accumulator */
    if (!neu1e) return;

    for (int s = 0; s <= neg_samples; s++) {
        int target;
        float label;

        if (s == 0) {
            /* Positive sample */
            target = context;
            label = 1.0f;
        } else {
            /* Negative sample */
            target = m->neg_table[n2v_xorshift32(rng) % NEG_TABLE_SIZE];
            if (target == center || target == context) continue;
            label = 0.0f;
        }

        float *vec_context = m->syn1neg + (size_t)target * (size_t)m->dim;

        /* dot product */
        float dot = 0.0f;
        for (int d = 0; d < m->dim; d++) {
            dot += vec_center[d] * vec_context[d];
        }

        /* sigmoid + error */
        float sig = fast_sigmoid(dot);
        float err = (label - sig) * lr;

        /* Accumulate gradient for center */
        for (int d = 0; d < m->dim; d++) {
            neu1e[d] += err * vec_context[d];
        }
        /* Update context embedding */
        for (int d = 0; d < m->dim; d++) {
            vec_context[d] += err * vec_center[d];
        }
    }

    /* Apply accumulated gradient to center */
    for (int d = 0; d < m->dim; d++) {
        vec_center[d] += neu1e[d];
    }
    free(neu1e);
}

/* ─── SQL Function: node2vec_train() ────────────────────────── */

/*
 * node2vec_train(edge_table, src_col, dst_col, output_table,
 *                dimensions, p, q, num_walks, walk_length,
 *                window, neg_samples, learning_rate, epochs)
 *
 * Returns: number of nodes embedded (INTEGER)
 */
static void node2vec_train_func(sqlite3_context *ctx,
                                 int argc, sqlite3_value **argv) {
    if (argc < 13) {
        sqlite3_result_error(ctx, "node2vec_train: requires 13 arguments", -1);
        return;
    }

    /* Parse arguments */
    const char *edge_table = (const char *)sqlite3_value_text(argv[0]);
    const char *src_col    = (const char *)sqlite3_value_text(argv[1]);
    const char *dst_col    = (const char *)sqlite3_value_text(argv[2]);
    const char *out_table  = (const char *)sqlite3_value_text(argv[3]);
    int dim         = sqlite3_value_int(argv[4]);
    double p        = sqlite3_value_double(argv[5]);
    double q        = sqlite3_value_double(argv[6]);
    int num_walks   = sqlite3_value_int(argv[7]);
    int walk_length = sqlite3_value_int(argv[8]);
    int window      = sqlite3_value_int(argv[9]);
    int neg_samples = sqlite3_value_int(argv[10]);
    double lr_init  = sqlite3_value_double(argv[11]);
    int epochs      = sqlite3_value_int(argv[12]);

    /* Validate identifiers (id_validate returns 0 on success) */
    if (!edge_table || id_validate(edge_table) != 0) {
        sqlite3_result_error(ctx, "node2vec_train: invalid edge_table name", -1);
        return;
    }
    if (!src_col || id_validate(src_col) != 0) {
        sqlite3_result_error(ctx, "node2vec_train: invalid src_col name", -1);
        return;
    }
    if (!dst_col || id_validate(dst_col) != 0) {
        sqlite3_result_error(ctx, "node2vec_train: invalid dst_col name", -1);
        return;
    }
    if (!out_table || id_validate(out_table) != 0) {
        sqlite3_result_error(ctx, "node2vec_train: invalid output_table name", -1);
        return;
    }

    /* Validate numeric parameters */
    if (dim <= 0 || dim > 1024) {
        sqlite3_result_error(ctx, "node2vec_train: dimensions must be 1-1024", -1);
        return;
    }
    if (p <= 0.0 || q <= 0.0) {
        sqlite3_result_error(ctx, "node2vec_train: p and q must be > 0", -1);
        return;
    }
    if (num_walks <= 0 || walk_length <= 0) {
        sqlite3_result_error(ctx, "node2vec_train: num_walks and walk_length must be > 0", -1);
        return;
    }
    if (window <= 0 || neg_samples <= 0) {
        sqlite3_result_error(ctx, "node2vec_train: window and neg_samples must be > 0", -1);
        return;
    }
    if (lr_init <= 0.0 || epochs <= 0) {
        sqlite3_result_error(ctx, "node2vec_train: learning_rate and epochs must be > 0", -1);
        return;
    }

    sqlite3 *db = sqlite3_context_db_handle(ctx);
    init_sigmoid_table();

    /* Step 1: Load graph from edge table */
    Graph g;
    graph_init(&g);
    int rc = graph_load_edges(db, &g, edge_table, src_col, dst_col);
    if (rc != SQLITE_OK) {
        graph_destroy(&g);
        sqlite3_result_error(ctx, "node2vec_train: failed to load edges", -1);
        return;
    }

    if (g.node_count == 0) {
        graph_destroy(&g);
        sqlite3_result_int(ctx, 0);
        return;
    }

    /* Step 2: Create SGNS model */
    unsigned int rng = 42;
    SgnsModel *model = sgns_create(g.node_count, dim, &g, &rng);
    if (!model) {
        graph_destroy(&g);
        sqlite3_result_error(ctx, "node2vec_train: out of memory", -1);
        return;
    }

    /* Step 3: Train — generate walks and update embeddings */
    int *walk = (int *)malloc((size_t)walk_length * sizeof(int));
    if (!walk) {
        sgns_destroy(model);
        graph_destroy(&g);
        sqlite3_result_error(ctx, "node2vec_train: out of memory", -1);
        return;
    }

    int total_words = g.node_count * num_walks * walk_length * epochs;
    int word_count = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int w = 0; w < num_walks; w++) {
            for (int n = 0; n < g.node_count; n++) {
                /* Learning rate decay */
                float lr = (float)(lr_init * (1.0 - (double)word_count / (double)total_words));
                if (lr < (float)(lr_init * 0.0001)) lr = (float)(lr_init * 0.0001);

                int wlen = biased_walk(&g, n, p, q, walk_length, walk, &rng);

                /* Skip-gram: for each position, train with context window */
                for (int pos = 0; pos < wlen; pos++) {
                    int center = walk[pos];
                    int ctx_start = pos - window;
                    int ctx_end = pos + window;
                    if (ctx_start < 0) ctx_start = 0;
                    if (ctx_end >= wlen) ctx_end = wlen - 1;

                    for (int c = ctx_start; c <= ctx_end; c++) {
                        if (c == pos) continue;
                        sgns_train_pair(model, center, walk[c],
                                        neg_samples, lr, &rng);
                    }
                    word_count++;
                }
            }
        }
    }

    free(walk);

    /* Step 4: Insert embeddings into the output HNSW table */
    char *insert_sql = sqlite3_mprintf(
        "INSERT INTO \"%w\" (rowid, vector) VALUES (?, ?)", out_table);
    if (!insert_sql) {
        sgns_destroy(model);
        graph_destroy(&g);
        sqlite3_result_error(ctx, "node2vec_train: out of memory", -1);
        return;
    }

    sqlite3_stmt *insert_stmt = NULL;
    rc = sqlite3_prepare_v2(db, insert_sql, -1, &insert_stmt, NULL);
    sqlite3_free(insert_sql);
    if (rc != SQLITE_OK) {
        sgns_destroy(model);
        graph_destroy(&g);
        sqlite3_result_error(ctx, "node2vec_train: failed to prepare INSERT "
                                   "for output table (does it exist?)", -1);
        return;
    }

    int inserted = 0;
    for (int i = 0; i < g.node_count; i++) {
        float *emb = model->syn0 + (size_t)i * (size_t)dim;

        /* L2-normalize the embedding for cosine similarity */
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) norm += emb[d] * emb[d];
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int d = 0; d < dim; d++) emb[d] /= norm;
        }

        sqlite3_bind_int64(insert_stmt, 1, (sqlite3_int64)(i + 1));
        sqlite3_bind_blob(insert_stmt, 2, emb,
                          dim * (int)sizeof(float), SQLITE_TRANSIENT);

        rc = sqlite3_step(insert_stmt);
        if (rc == SQLITE_DONE) {
            inserted++;
        }
        sqlite3_reset(insert_stmt);
    }

    sqlite3_finalize(insert_stmt);
    sgns_destroy(model);
    graph_destroy(&g);

    sqlite3_result_int(ctx, inserted);
}

/* ─── Registration ───────────────────────────────────────────── */

int node2vec_register_functions(sqlite3 *db) {
    return sqlite3_create_function(
        db, "node2vec_train", 13,
        SQLITE_UTF8 | SQLITE_DETERMINISTIC,
        NULL,
        node2vec_train_func,
        NULL, NULL
    );
}
