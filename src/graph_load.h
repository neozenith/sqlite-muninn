/*
 * graph_load.h — Shared graph loading for graph algorithms
 *
 * Provides an O(1) hash-map-based adjacency structure with support for
 * weighted edges and temporal filtering. Used by centrality, community
 * detection, and other graph algorithm TVFs.
 */
#ifndef GRAPH_LOAD_H
#define GRAPH_LOAD_H

#include <sqlite3.h>

/* A single edge in an adjacency list */
typedef struct {
    int target;    /* index into GraphData.ids[] */
    double weight; /* 1.0 for unweighted graphs */
} GraphEdge;

/* Adjacency list for one node */
typedef struct {
    GraphEdge *edges;
    int count;
    int capacity;
} GraphAdjList;

/* Complete in-memory graph with hash-map-based node lookup */
typedef struct {
    char **ids; /* ids[i] = string ID of node i */
    int node_count;
    int node_capacity;
    int *map_indices;  /* open-addressing hash: slot -> node index, -1 = empty */
    int map_capacity;  /* always power of 2 */
    GraphAdjList *out; /* forward adjacency: out[i] = neighbors of node i */
    GraphAdjList *in;  /* reverse adjacency: in[i] = predecessors of node i */
    int has_weights;
    int edge_count;
} GraphData;

/* Configuration for loading a graph from a SQLite table */
typedef struct {
    const char *edge_table;
    const char *src_col;
    const char *dst_col;
    const char *weight_col;    /* NULL = unweighted (all edges weight 1.0) */
    const char *direction;     /* "forward", "reverse", or "both" */
    const char *timestamp_col; /* NULL = no temporal filter */
    sqlite3_value *time_start; /* NULL = no lower bound */
    sqlite3_value *time_end;   /* NULL = no upper bound */
} GraphLoadConfig;

/* Initialize an empty graph. Must be called before any other operations. */
void graph_data_init(GraphData *g);

/* Free all memory owned by the graph. */
void graph_data_destroy(GraphData *g);

/* Look up a node by string ID. Returns index or -1 if not found. */
int graph_data_find(const GraphData *g, const char *id);

/* Look up or insert a node. Returns its index. Auto-resizes hash map at 70% load. */
int graph_data_find_or_add(GraphData *g, const char *id);

/* Add a single edge to an adjacency list. */
void adj_add(GraphAdjList *adj, int target, double weight);

/*
 * Add a weighted edge between two nodes (by index).
 * Adds to out[src] and optionally in[dst]. Increments edge_count.
 */
static inline void graph_data_add_edge(GraphData *g, int src_idx, int dst_idx, double weight, int add_forward,
                                       int add_reverse) {
    if (add_forward)
        adj_add(&g->out[src_idx], dst_idx, weight);
    if (add_reverse)
        adj_add(&g->in[dst_idx], src_idx, weight);
    g->edge_count++;
}

/*
 * Load a graph from a SQLite table according to the given configuration.
 * Validates identifiers via id_validate(). Builds both forward (out) and
 * reverse (in) adjacency lists.
 *
 * Returns SQLITE_OK on success, or an error code.
 * On error, sets *pzErrMsg (caller must sqlite3_free it).
 */
int graph_data_load(sqlite3 *db, const GraphLoadConfig *config, GraphData *g, char **pzErrMsg);

/* G7 T7.3 — community-filter helpers.
 *
 * build_community_mask: allocate and return an int[g->node_count]
 *   where mask[i] = 1 iff partition[i] == target_community_id, else 0.
 *   Caller frees with free(). Returns NULL on allocation failure.
 *
 * induce_subgraph: build a NEW GraphData containing only nodes where
 *   mask[i] != 0. Edges are preserved iff BOTH endpoints are in the
 *   mask. The new graph uses fresh 0..n-1 indexing; node string IDs
 *   are preserved (graph_data_find_or_add carries them). out_to_orig,
 *   if non-NULL, receives a malloc'd int[new_node_count] mapping each
 *   new index back to the original index in g.
 *
 *   has_weights is copied from g. Returns SQLITE_OK on success or a
 *   SQLite errcode on failure. On failure out_g is left in a
 *   destroy-safe state (caller still calls graph_data_destroy).
 */
int *build_community_mask(const GraphData *g, const int *partition, int target_community_id);
int induce_subgraph(const GraphData *g, const int *mask, GraphData *out_g, int **out_to_orig);

/* G7 T7.5 — mask intersection. Returns a fresh malloc'd int[n] where
 * out[i] = 1 iff a[i] != 0 AND b[i] != 0. Both inputs are treated as
 * truthy/falsy (any non-zero counts as "in"). NULL on bad inputs or
 * allocation failure. Caller frees with free().
 *
 * Used to compose G1's provenance filter mask with G7's community
 * filter mask before inducing the centrality subgraph. */
int *intersect_masks(const int *a, const int *b, int n);

#endif /* GRAPH_LOAD_H */
