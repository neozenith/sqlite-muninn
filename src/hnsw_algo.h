/*
 * hnsw_algo.h — Core HNSW data structures and algorithms
 *
 * In-memory HNSW graph with insert, search, and delete operations.
 * The graph is persisted to SQLite shadow tables by the virtual table layer.
 */
#ifndef HNSW_ALGO_H
#define HNSW_ALGO_H

#include "vec_math.h"
#include <stdint.h>

/* Maximum number of layers (geometric distribution makes >30 astronomically unlikely) */
#define HNSW_MAX_LEVELS 32

/* A single node in the HNSW graph */
typedef struct HnswNode {
    int64_t id;
    float *vector;       /* owned: float[dim], allocated by caller */
    int level;           /* max level this node participates in */
    int deleted;         /* soft-delete flag */

    /* Per-level neighbor lists */
    int64_t **neighbors;  /* neighbors[l] = array of node IDs at level l */
    int *neighbor_count;  /* neighbor_count[l] = count at level l */
    int *neighbor_cap;    /* neighbor_cap[l] = allocated capacity at level l */
} HnswNode;

/* Search result: (id, distance) pair */
typedef struct {
    int64_t id;
    float distance;
} HnswSearchResult;

/* The HNSW index */
typedef struct {
    int dim;               /* vector dimensionality */
    int M;                 /* max connections per node per layer (M_max) */
    int M_max0;            /* max connections at layer 0 (typically 2*M) */
    int ef_construction;   /* beam width during insert */
    VecMetric metric;
    VecDistanceFunc dist_func;
    double level_mult;     /* 1/ln(M), for random level generation */

    int64_t entry_point;   /* node ID of entry point (-1 if empty) */
    int max_level;         /* current max level in graph */

    /* Node storage: hash map (id -> node) via simple open-addressing */
    HnswNode **nodes;      /* hash table buckets */
    int node_count;
    int node_capacity;     /* must be power of 2 */
    unsigned int rng_state; /* xorshift32 PRNG state */
} HnswIndex;

/* Create a new empty HNSW index. Returns NULL on failure. */
HnswIndex *hnsw_create(int dim, VecMetric metric, int M, int ef_construction);

/* Free all memory associated with the index. */
void hnsw_destroy(HnswIndex *idx);

/*
 * Insert a vector with the given ID. The vector data is copied internally.
 * Returns 0 on success, -1 on failure.
 */
int hnsw_insert(HnswIndex *idx, int64_t id, const float *vector);

/*
 * Search for the k nearest neighbors of the query vector.
 * results must point to an array of at least k HnswSearchResult.
 * ef_search controls beam width (higher = better recall, slower).
 * Returns the number of results found (may be < k if index has fewer nodes).
 */
int hnsw_search(HnswIndex *idx, const float *query, int k, int ef_search,
                HnswSearchResult *results);

/*
 * Delete a node by ID (soft delete + neighbor reconnection).
 * Returns 0 on success, -1 if node not found.
 */
int hnsw_delete(HnswIndex *idx, int64_t id);

/*
 * Look up a node by ID. Returns NULL if not found or deleted.
 */
HnswNode *hnsw_get_node(HnswIndex *idx, int64_t id);

/*
 * Get the vector for a node. Returns NULL if not found.
 */
const float *hnsw_get_vector(HnswIndex *idx, int64_t id);

/* Seed the PRNG (for reproducible tests) */
void hnsw_seed_rng(HnswIndex *idx, unsigned int seed);

/*
 * Internal functions exposed for the virtual table layer (shadow table I/O).
 * Not part of the public API — do not call from application code.
 */
HnswNode *ht_find(HnswNode **table, int capacity, int64_t id);
int ht_insert(HnswNode **table, int capacity, HnswNode *node);
int ht_resize(HnswIndex *idx);
HnswNode *node_create(int64_t id, const float *vector, int dim, int level);
void node_destroy(HnswNode *node);
int node_add_neighbor(HnswNode *node, int level, int64_t neighbor_id);
void node_remove_neighbor(HnswNode *node, int level, int64_t neighbor_id);

#endif /* HNSW_ALGO_H */
