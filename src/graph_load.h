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
    int target;      /* index into GraphData.ids[] */
    double weight;   /* 1.0 for unweighted graphs */
} GraphEdge;

/* Adjacency list for one node */
typedef struct {
    GraphEdge *edges;
    int count;
    int capacity;
} GraphAdjList;

/* Complete in-memory graph with hash-map-based node lookup */
typedef struct {
    char **ids;            /* ids[i] = string ID of node i */
    int node_count;
    int node_capacity;
    int *map_indices;      /* open-addressing hash: slot -> node index, -1 = empty */
    int map_capacity;      /* always power of 2 */
    GraphAdjList *out;     /* forward adjacency: out[i] = neighbors of node i */
    GraphAdjList *in;      /* reverse adjacency: in[i] = predecessors of node i */
    int has_weights;
    int edge_count;
} GraphData;

/* Configuration for loading a graph from a SQLite table */
typedef struct {
    const char *edge_table;
    const char *src_col;
    const char *dst_col;
    const char *weight_col;      /* NULL = unweighted (all edges weight 1.0) */
    const char *direction;       /* "forward", "reverse", or "both" */
    const char *timestamp_col;   /* NULL = no temporal filter */
    sqlite3_value *time_start;   /* NULL = no lower bound */
    sqlite3_value *time_end;     /* NULL = no upper bound */
} GraphLoadConfig;

/* Initialize an empty graph. Must be called before any other operations. */
void graph_data_init(GraphData *g);

/* Free all memory owned by the graph. */
void graph_data_destroy(GraphData *g);

/* Look up a node by string ID. Returns index or -1 if not found. */
int graph_data_find(const GraphData *g, const char *id);

/* Look up or insert a node. Returns its index. Auto-resizes hash map at 70% load. */
int graph_data_find_or_add(GraphData *g, const char *id);

/*
 * Load a graph from a SQLite table according to the given configuration.
 * Validates identifiers via id_validate(). Builds both forward (out) and
 * reverse (in) adjacency lists.
 *
 * Returns SQLITE_OK on success, or an error code.
 * On error, sets *pzErrMsg (caller must sqlite3_free it).
 */
int graph_data_load(sqlite3 *db, const GraphLoadConfig *config,
                    GraphData *g, char **pzErrMsg);

#endif /* GRAPH_LOAD_H */
