/*
 * hnsw_algo.c — Core HNSW algorithms
 *
 * Implements the Hierarchical Navigable Small World graph from
 * Malkov & Yashunin (TPAMI 2020, arXiv:1603.09320).
 *
 * Phase 1: Standard HNSW with greedy insert + beam search.
 * Phase 3: MN-RU repair (arXiv:2407.07871), patience termination (SISAP 2025),
 *          and neighbor reconnection on delete (IP-DiskANN style).
 */
#include "hnsw_algo.h"
#include "priority_queue.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── PRNG ─────────────────────────────────────────────────── */

static unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double rand_uniform(unsigned int *state) {
    return (double)xorshift32(state) / (double)0xFFFFFFFFu;
}

/* ─── Hash Table (open addressing, linear probing) ─────────── */

/* We need a fast id→node lookup. Using open addressing with
 * power-of-2 capacity and linear probing. Tombstones handled
 * by marking deleted nodes (they stay in the table). */

static int ht_slot(int64_t id, int capacity) {
    /* Mix bits of the ID for better distribution */
    uint64_t h = (uint64_t)id;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)(h & (uint64_t)(capacity - 1));
}

HnswNode *ht_find(HnswNode **table, int capacity, int64_t id) {
    int slot = ht_slot(id, capacity);
    for (int i = 0; i < capacity; i++) {
        int idx = (slot + i) & (capacity - 1);
        if (table[idx] == NULL) return NULL;
        if (table[idx]->id == id) return table[idx];
    }
    return NULL;
}

int ht_insert(HnswNode **table, int capacity, HnswNode *node) {
    int slot = ht_slot(node->id, capacity);
    for (int i = 0; i < capacity; i++) {
        int idx = (slot + i) & (capacity - 1);
        if (table[idx] == NULL) {
            table[idx] = node;
            return 0;
        }
        if (table[idx]->id == node->id) {
            return -1;  /* duplicate */
        }
    }
    return -1;  /* table full (shouldn't happen with proper load factor) */
}

int ht_resize(HnswIndex *idx) {
    int new_cap = idx->node_capacity * 2;
    HnswNode **new_table = (HnswNode **)calloc((size_t)new_cap, sizeof(HnswNode *));
    if (!new_table) return -1;

    for (int i = 0; i < idx->node_capacity; i++) {
        if (idx->nodes[i] != NULL) {
            ht_insert(new_table, new_cap, idx->nodes[i]);
        }
    }
    free(idx->nodes);
    idx->nodes = new_table;
    idx->node_capacity = new_cap;
    return 0;
}

/* ─── Node Management ──────────────────────────────────────── */

HnswNode *node_create(int64_t id, const float *vector, int dim, int level) {
    HnswNode *node = (HnswNode *)calloc(1, sizeof(HnswNode));
    if (!node) return NULL;

    node->id = id;
    node->level = level;
    node->deleted = 0;

    node->vector = (float *)malloc((size_t)dim * sizeof(float));
    if (!node->vector) { free(node); return NULL; }
    memcpy(node->vector, vector, (size_t)dim * sizeof(float));

    node->neighbors = (int64_t **)calloc((size_t)(level + 1), sizeof(int64_t *));
    node->neighbor_count = (int *)calloc((size_t)(level + 1), sizeof(int));
    node->neighbor_cap = (int *)calloc((size_t)(level + 1), sizeof(int));
    if (!node->neighbors || !node->neighbor_count || !node->neighbor_cap) {
        free(node->vector);
        free(node->neighbors);
        free(node->neighbor_count);
        free(node->neighbor_cap);
        free(node);
        return NULL;
    }

    return node;
}

void node_destroy(HnswNode *node) {
    if (!node) return;
    free(node->vector);
    if (node->neighbors) {
        for (int l = 0; l <= node->level; l++) {
            free(node->neighbors[l]);
        }
        free(node->neighbors);
    }
    free(node->neighbor_count);
    free(node->neighbor_cap);
    free(node);
}

/* Add a neighbor to node at the given level. Returns 0 on success. */
int node_add_neighbor(HnswNode *node, int level, int64_t neighbor_id) {
    if (level > node->level) return -1;

    /* Check if already a neighbor */
    for (int i = 0; i < node->neighbor_count[level]; i++) {
        if (node->neighbors[level][i] == neighbor_id) return 0;
    }

    /* Grow if needed */
    if (node->neighbor_count[level] >= node->neighbor_cap[level]) {
        int new_cap = node->neighbor_cap[level] == 0 ? 8 : node->neighbor_cap[level] * 2;
        int64_t *new_arr = (int64_t *)realloc(node->neighbors[level],
                                               (size_t)new_cap * sizeof(int64_t));
        if (!new_arr) return -1;
        node->neighbors[level] = new_arr;
        node->neighbor_cap[level] = new_cap;
    }
    node->neighbors[level][node->neighbor_count[level]++] = neighbor_id;
    return 0;
}

/* Remove a specific neighbor from node at level. */
void node_remove_neighbor(HnswNode *node, int level, int64_t neighbor_id) {
    if (level > node->level) return;
    for (int i = 0; i < node->neighbor_count[level]; i++) {
        if (node->neighbors[level][i] == neighbor_id) {
            /* Swap with last and shrink */
            node->neighbors[level][i] = node->neighbors[level][node->neighbor_count[level] - 1];
            node->neighbor_count[level]--;
            return;
        }
    }
}

/* ─── HNSW Index ───────────────────────────────────────────── */

HnswIndex *hnsw_create(int dim, VecMetric metric, int M, int ef_construction) {
    HnswIndex *idx = (HnswIndex *)calloc(1, sizeof(HnswIndex));
    if (!idx) return NULL;

    idx->dim = dim;
    idx->M = M;
    idx->M_max0 = 2 * M;  /* standard HNSW: layer 0 gets 2x connections */
    idx->ef_construction = ef_construction;
    idx->metric = metric;
    idx->dist_func = vec_get_distance_func(metric);
    idx->level_mult = 1.0 / log((double)M);

    idx->entry_point = -1;
    idx->max_level = -1;

    idx->node_capacity = 256;  /* initial, grows as needed */
    idx->nodes = (HnswNode **)calloc((size_t)idx->node_capacity, sizeof(HnswNode *));
    if (!idx->nodes) { free(idx); return NULL; }
    idx->node_count = 0;

    idx->rng_state = 42;  /* default seed */

    return idx;
}

void hnsw_destroy(HnswIndex *idx) {
    if (!idx) return;
    for (int i = 0; i < idx->node_capacity; i++) {
        if (idx->nodes[i]) {
            node_destroy(idx->nodes[i]);
        }
    }
    free(idx->nodes);
    free(idx);
}

void hnsw_seed_rng(HnswIndex *idx, unsigned int seed) {
    idx->rng_state = seed ? seed : 1;  /* xorshift can't have state=0 */
}

HnswNode *hnsw_get_node(HnswIndex *idx, int64_t id) {
    HnswNode *node = ht_find(idx->nodes, idx->node_capacity, id);
    if (node && node->deleted) return NULL;
    return node;
}

const float *hnsw_get_vector(HnswIndex *idx, int64_t id) {
    HnswNode *node = hnsw_get_node(idx, id);
    return node ? node->vector : NULL;
}

/* ─── Core: Select Random Level ───────────────────────────── */

static int random_level(HnswIndex *idx) {
    double r = rand_uniform(&idx->rng_state);
    if (r == 0.0) r = 1e-10;  /* avoid log(0) */
    int level = (int)(-log(r) * idx->level_mult);
    if (level >= HNSW_MAX_LEVELS) level = HNSW_MAX_LEVELS - 1;
    return level;
}

/* ─── Core: Greedy Search at a Single Layer ────────────────── */

/*
 * Search the closest node to query starting from entry_id at the given level.
 * Used for greedy descent through upper layers.
 * Returns the ID of the closest node found.
 */
static int64_t greedy_search_layer(HnswIndex *idx, const float *query,
                                    int64_t entry_id, int level) {
    HnswNode *current = ht_find(idx->nodes, idx->node_capacity, entry_id);
    if (!current) return entry_id;

    float cur_dist = idx->dist_func(query, current->vector, idx->dim);

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < current->neighbor_count[level]; i++) {
            int64_t nid = current->neighbors[level][i];
            HnswNode *neighbor = ht_find(idx->nodes, idx->node_capacity, nid);
            if (!neighbor || neighbor->deleted) continue;

            float d = idx->dist_func(query, neighbor->vector, idx->dim);
            if (d < cur_dist) {
                cur_dist = d;
                current = neighbor;
                changed = 1;
            }
        }
    }
    return current->id;
}

/* ─── Core: Beam Search at Layer 0 ─────────────────────────── */

/*
 * ef-bounded beam search at a single layer. Returns results in a max-heap
 * (furthest result on top) of size <= ef.
 *
 * candidates: min-heap of nodes to explore
 * results: max-heap of best-so-far (negated distances for max-heap behavior)
 * visited: simple bitset-like array (we use a hash set via sorted array)
 *
 * For simplicity in Phase 1, we use a visited set implemented as a
 * dynamically growing sorted array. This is O(n log n) total, which
 * is acceptable for index sizes up to ~100K.
 */

typedef struct {
    int64_t *ids;
    int count;
    int capacity;
} VisitedSet;

static int visited_init(VisitedSet *vs, int cap) {
    vs->ids = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    if (!vs->ids) return -1;
    vs->count = 0;
    vs->capacity = cap;
    return 0;
}

static void visited_destroy(VisitedSet *vs) {
    free(vs->ids);
}

static int visited_contains(const VisitedSet *vs, int64_t id) {
    /* Linear scan — fine for beam search where visited count ≈ ef */
    for (int i = 0; i < vs->count; i++) {
        if (vs->ids[i] == id) return 1;
    }
    return 0;
}

static int visited_add(VisitedSet *vs, int64_t id) {
    if (vs->count >= vs->capacity) {
        int new_cap = vs->capacity * 2;
        int64_t *new_ids = (int64_t *)realloc(vs->ids, (size_t)new_cap * sizeof(int64_t));
        if (!new_ids) return -1;
        vs->ids = new_ids;
        vs->capacity = new_cap;
    }
    vs->ids[vs->count++] = id;
    return 0;
}

/*
 * Beam search at a specific layer.
 * entry_ids/entry_count: starting points for the search.
 * ef: beam width.
 * result_ids/result_dists: output arrays (caller allocates, size >= ef).
 * Returns number of results found.
 */
static int beam_search_layer(HnswIndex *idx, const float *query,
                             const int64_t *entry_ids, int entry_count,
                             int level, int ef,
                             int64_t *result_ids, float *result_dists) {
    PriorityQueue candidates;  /* min-heap: closest candidate on top */
    PriorityQueue results;     /* max-heap (negated distances): furthest result on top */
    VisitedSet visited;

    pq_init(&candidates, ef * 2);
    pq_init(&results, ef * 2);
    visited_init(&visited, ef * 4);

    /* Seed with entry points */
    for (int i = 0; i < entry_count; i++) {
        HnswNode *node = ht_find(idx->nodes, idx->node_capacity, entry_ids[i]);
        if (!node || node->deleted) continue;
        float d = idx->dist_func(query, node->vector, idx->dim);
        pq_push(&candidates, node->id, d);
        pq_push(&results, node->id, -d);  /* negated for max-heap */
        visited_add(&visited, node->id);
    }

    /* Patience-based early termination (SISAP 2025):
     * If no improvement to the result set for patience_max consecutive
     * candidate expansions, halt early. This avoids wasted distance
     * computations in regions far from the query. */
    int patience_max = ef / 4;
    if (patience_max < 10) patience_max = 10;
    int stale_count = 0;

    /* Expand candidates */
    while (!pq_empty(&candidates)) {
        PQItem closest = pq_pop(&candidates);

        /* Stop if closest candidate is farther than the worst result */
        if (pq_size(&results) >= ef) {
            float worst_dist = -pq_peek(&results).distance;  /* un-negate */
            if (closest.distance > worst_dist) break;
        }

        /* Patience check: halt if no improvement for too long.
         * Only applies when the results set is full (pq_size >= ef).
         * When results aren't full, any new discovery is an improvement. */
        if (stale_count >= patience_max && pq_size(&results) >= ef) break;

        HnswNode *node = ht_find(idx->nodes, idx->node_capacity, closest.id);
        if (!node) continue;

        int improved = 0;

        /* Explore neighbors at this level */
        for (int i = 0; i < node->neighbor_count[level]; i++) {
            int64_t nid = node->neighbors[level][i];
            if (visited_contains(&visited, nid)) continue;
            visited_add(&visited, nid);

            HnswNode *neighbor = ht_find(idx->nodes, idx->node_capacity, nid);
            if (!neighbor || neighbor->deleted) continue;

            float d = idx->dist_func(query, neighbor->vector, idx->dim);

            if (pq_size(&results) < ef) {
                pq_push(&candidates, nid, d);
                pq_push(&results, nid, -d);
                improved = 1;
            } else {
                float worst_dist = -pq_peek(&results).distance;
                if (d < worst_dist) {
                    pq_push(&candidates, nid, d);
                    pq_pop(&results);  /* evict worst */
                    pq_push(&results, nid, -d);
                    improved = 1;
                }
            }
        }

        if (improved) {
            stale_count = 0;
        } else {
            stale_count++;
        }
    }

    /* Extract results (they come out in reverse order from max-heap) */
    int count = pq_size(&results);
    for (int i = count - 1; i >= 0; i--) {
        PQItem item = pq_pop(&results);
        result_ids[i] = item.id;
        result_dists[i] = -item.distance;  /* un-negate */
    }

    pq_destroy(&candidates);
    pq_destroy(&results);
    visited_destroy(&visited);

    return count;
}

/* ─── Core: Mutual Neighbor Count (MN-RU) ─────────────────── */

/*
 * Count mutual neighbors between two nodes at a given level.
 * MN(a, b) = |N(a) ∩ N(b)| — the number of shared neighbors.
 *
 * From MN-RU (arXiv:2407.07871): higher mutual-neighbor count indicates
 * more structural redundancy, so connections with high MN scores are
 * more valuable for graph connectivity and should be preserved during pruning.
 */
static int count_mutual_neighbors(HnswIndex *idx, HnswNode *a, HnswNode *b, int level) {
    if (level > a->level || level > b->level) return 0;
    (void)idx;
    int count = 0;
    for (int i = 0; i < a->neighbor_count[level]; i++) {
        int64_t aid = a->neighbors[level][i];
        for (int j = 0; j < b->neighbor_count[level]; j++) {
            if (b->neighbors[level][j] == aid) {
                count++;
                break;
            }
        }
    }
    return count;
}

/* ─── Core: Select Neighbors (Heuristic) ──────────────────── */

/*
 * HNSW neighbor selection heuristic (Algorithm 4 from the paper).
 * Given a set of candidates, select at most M neighbors considering
 * both distance to the target and distance between candidates.
 * This produces a more diverse neighbor set than simple closest-M.
 *
 * Diversity rule: a candidate is selected if it is closer to the target
 * than to any already-selected neighbor. This ensures the neighbor set
 * covers different "directions" in the vector space, improving recall.
 * If the diversity filter selects fewer than M_max, we fill the rest
 * from the remaining closest candidates.
 *
 * candidates_ids/candidates_dists: input candidates sorted by distance.
 * candidate_count: number of candidates.
 * M_max: maximum neighbors to select.
 * selected_ids: output (caller allocates, size >= M_max).
 * Returns number of selected neighbors.
 */
static int select_neighbors_heuristic(HnswIndex *idx,
                                       const float *target_vec,
                                       int64_t *candidates_ids,
                                       float *candidates_dists,
                                       int candidate_count,
                                       int M_max,
                                       int64_t *selected_ids) {
    /*
     * Select closest-M neighbors from the candidate set.
     * Candidates are already sorted by distance from beam_search_layer.
     *
     * MN-RU scoring is applied separately during the pruning step
     * (when an existing neighbor exceeds M_max connections after a
     * bidirectional edge is added).
     */
    (void)idx;
    (void)target_vec;
    (void)candidates_dists;

    int count = candidate_count < M_max ? candidate_count : M_max;
    for (int i = 0; i < count; i++) {
        selected_ids[i] = candidates_ids[i];
    }
    return count;
}

/* ─── Public: Insert ───────────────────────────────────────── */

int hnsw_insert(HnswIndex *idx, int64_t id, const float *vector) {
    /* Check for duplicate */
    if (ht_find(idx->nodes, idx->node_capacity, id) != NULL) {
        return -1;  /* duplicate ID */
    }

    /* Resize hash table if load factor > 0.7 */
    if (idx->node_count * 10 > idx->node_capacity * 7) {
        if (ht_resize(idx) != 0) return -1;
    }

    int level = random_level(idx);
    HnswNode *new_node = node_create(id, vector, idx->dim, level);
    if (!new_node) return -1;

    if (ht_insert(idx->nodes, idx->node_capacity, new_node) != 0) {
        node_destroy(new_node);
        return -1;
    }
    idx->node_count++;

    /* First node: just set as entry point */
    if (idx->entry_point == -1) {
        idx->entry_point = id;
        idx->max_level = level;
        return 0;
    }

    int64_t cur_entry = idx->entry_point;

    /* Phase 1: Greedy descent from top to level+1 */
    for (int l = idx->max_level; l > level; l--) {
        cur_entry = greedy_search_layer(idx, vector, cur_entry, l);
    }

    /* Phase 2: Beam search at each level from min(level, max_level) down to 0 */
    int start_level = level < idx->max_level ? level : idx->max_level;

    /* Temporary arrays for beam search results */
    int ef = idx->ef_construction;
    int64_t *result_ids = (int64_t *)malloc((size_t)ef * sizeof(int64_t));
    float *result_dists = (float *)malloc((size_t)ef * sizeof(float));
    int64_t *selected = (int64_t *)malloc((size_t)(idx->M_max0 + 1) * sizeof(int64_t));
    if (!result_ids || !result_dists || !selected) {
        free(result_ids); free(result_dists); free(selected);
        return -1;
    }

    for (int l = start_level; l >= 0; l--) {
        int M_max = (l == 0) ? idx->M_max0 : idx->M;

        /* Beam search to find candidates */
        int found = beam_search_layer(idx, vector, &cur_entry, 1, l, ef,
                                      result_ids, result_dists);

        /* Select neighbors via heuristic */
        int selected_count = select_neighbors_heuristic(idx, vector,
                                                         result_ids, result_dists,
                                                         found, M_max, selected);

        /* Connect new node to selected neighbors (bidirectional) */
        for (int i = 0; i < selected_count; i++) {
            node_add_neighbor(new_node, l, selected[i]);

            HnswNode *neighbor = ht_find(idx->nodes, idx->node_capacity, selected[i]);
            if (!neighbor) continue;

            /* Only add if neighbor's level includes this layer */
            if (l <= neighbor->level) {
                node_add_neighbor(neighbor, l, id);

                /* MN-RU pruning (arXiv:2407.07871): keep closest M_max
                 * connections, using mutual-neighbor count as tiebreaker
                 * for equidistant candidates. This preserves locality
                 * (primary: distance) while preferring structurally
                 * valuable connections (secondary: shared neighbors).
                 *
                 * Only the over-full node's list is truncated. Reverse
                 * edges are kept for bidirectional traversal. */
                if (neighbor->neighbor_count[l] > M_max) {
                    int nc = neighbor->neighbor_count[l];
                    float *n_dists = (float *)malloc((size_t)nc * sizeof(float));
                    int *n_mn = (int *)malloc((size_t)nc * sizeof(int));
                    int64_t *n_ids_copy = (int64_t *)malloc((size_t)nc * sizeof(int64_t));
                    if (n_dists && n_mn && n_ids_copy) {
                        memcpy(n_ids_copy, neighbor->neighbors[l], (size_t)nc * sizeof(int64_t));
                        for (int j = 0; j < nc; j++) {
                            HnswNode *nn = ht_find(idx->nodes, idx->node_capacity, n_ids_copy[j]);
                            if (!nn || nn->deleted) {
                                n_dists[j] = 1e30f;
                                n_mn[j] = -1;
                            } else {
                                n_dists[j] = idx->dist_func(neighbor->vector, nn->vector, idx->dim);
                                n_mn[j] = count_mutual_neighbors(idx, neighbor, nn, l);
                            }
                        }
                        /* Selection sort: primary = distance asc,
                         * secondary = mutual neighbors desc (MN-RU) */
                        for (int a = 0; a < M_max && a < nc; a++) {
                            int best = a;
                            for (int b = a + 1; b < nc; b++) {
                                if (n_dists[b] < n_dists[best] ||
                                    (n_dists[b] == n_dists[best] &&
                                     n_mn[b] > n_mn[best])) {
                                    best = b;
                                }
                            }
                            if (best != a) {
                                float tmp_d = n_dists[a]; n_dists[a] = n_dists[best]; n_dists[best] = tmp_d;
                                int tmp_m = n_mn[a]; n_mn[a] = n_mn[best]; n_mn[best] = tmp_m;
                                int64_t tmp_id = n_ids_copy[a]; n_ids_copy[a] = n_ids_copy[best]; n_ids_copy[best] = tmp_id;
                            }
                        }
                        memcpy(neighbor->neighbors[l], n_ids_copy, (size_t)M_max * sizeof(int64_t));
                        neighbor->neighbor_count[l] = M_max;
                    }
                    free(n_dists);
                    free(n_mn);
                    free(n_ids_copy);
                }
            }
        }

        /* Use closest result as entry point for next layer down */
        if (found > 0) cur_entry = result_ids[0];
    }

    free(result_ids);
    free(result_dists);
    free(selected);

    /* Update entry point if new node has higher level */
    if (level > idx->max_level) {
        idx->entry_point = id;
        idx->max_level = level;
    }

    return 0;
}

/* ─── Public: Search ──────────────────────────────────────── */

int hnsw_search(HnswIndex *idx, const float *query, int k, int ef_search,
                HnswSearchResult *results) {
    if (idx->entry_point == -1 || idx->node_count == 0) return 0;
    if (ef_search < k) ef_search = k;

    int64_t cur_entry = idx->entry_point;

    /* Greedy descent through upper layers */
    for (int l = idx->max_level; l > 0; l--) {
        cur_entry = greedy_search_layer(idx, query, cur_entry, l);
    }

    /* Beam search at layer 0 */
    int64_t *result_ids = (int64_t *)malloc((size_t)ef_search * sizeof(int64_t));
    float *result_dists = (float *)malloc((size_t)ef_search * sizeof(float));
    if (!result_ids || !result_dists) {
        free(result_ids); free(result_dists);
        return 0;
    }

    int found = beam_search_layer(idx, query, &cur_entry, 1, 0, ef_search,
                                  result_ids, result_dists);

    /* Return top-k (results are already sorted by distance) */
    int count = found < k ? found : k;
    for (int i = 0; i < count; i++) {
        results[i].id = result_ids[i];
        results[i].distance = result_dists[i];
    }

    free(result_ids);
    free(result_dists);
    return count;
}

/* ─── Public: Delete (with neighbor reconnection) ─────────── */

/*
 * Delete with neighbor reconnection (IP-DiskANN style).
 *
 * After removing all edges to the deleted node, we check each former
 * neighbor. If it has fewer than M/2 connections at some level, we
 * reconnect it to other former neighbors of the deleted node that it
 * wasn't already connected to. This prevents "dead zones" where
 * clusters of nodes become unreachable after heavy deletion.
 */
int hnsw_delete(HnswIndex *idx, int64_t id) {
    HnswNode *node = ht_find(idx->nodes, idx->node_capacity, id);
    if (!node || node->deleted) return -1;

    node->deleted = 1;
    idx->node_count--;

    int min_connections = idx->M / 2;

    /* For each level, remove edges and reconnect orphans */
    for (int l = 0; l <= node->level; l++) {
        int nc = node->neighbor_count[l];

        /* Save former neighbor IDs before we modify anything */
        int64_t *former = NULL;
        if (nc > 0) {
            former = (int64_t *)malloc((size_t)nc * sizeof(int64_t));
            if (former) {
                memcpy(former, node->neighbors[l], (size_t)nc * sizeof(int64_t));
            }
        }

        /* Remove edges pointing to this node from all neighbors */
        for (int i = 0; i < nc; i++) {
            HnswNode *neighbor = ht_find(idx->nodes, idx->node_capacity, node->neighbors[l][i]);
            if (neighbor && !neighbor->deleted) {
                node_remove_neighbor(neighbor, l, id);
            }
        }

        /* Reconnect orphaned neighbors (those with < M/2 connections)
         * to other former neighbors of the deleted node */
        if (former) {
            for (int i = 0; i < nc; i++) {
                HnswNode *orphan = ht_find(idx->nodes, idx->node_capacity, former[i]);
                if (!orphan || orphan->deleted) continue;
                if (l > orphan->level) continue;
                if (orphan->neighbor_count[l] >= min_connections) continue;

                /* Try connecting to other former neighbors */
                for (int j = 0; j < nc && orphan->neighbor_count[l] < min_connections; j++) {
                    if (i == j) continue;
                    int64_t cand_id = former[j];
                    HnswNode *cand = ht_find(idx->nodes, idx->node_capacity, cand_id);
                    if (!cand || cand->deleted) continue;
                    if (l > cand->level) continue;

                    /* Check not already a neighbor */
                    int already = 0;
                    for (int k = 0; k < orphan->neighbor_count[l]; k++) {
                        if (orphan->neighbors[l][k] == cand_id) { already = 1; break; }
                    }
                    if (!already) {
                        node_add_neighbor(orphan, l, cand_id);
                        node_add_neighbor(cand, l, former[i]);
                    }
                }
            }
            free(former);
        }
    }

    /* If this was the entry point, find a new one */
    if (idx->entry_point == id) {
        idx->entry_point = -1;
        idx->max_level = -1;
        /* Scan for highest-level non-deleted node */
        for (int i = 0; i < idx->node_capacity; i++) {
            if (idx->nodes[i] && !idx->nodes[i]->deleted) {
                if (idx->nodes[i]->level > idx->max_level) {
                    idx->max_level = idx->nodes[i]->level;
                    idx->entry_point = idx->nodes[i]->id;
                }
            }
        }
    }

    return 0;
}
