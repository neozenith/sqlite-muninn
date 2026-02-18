/*
 * graph_csr.h — Compressed Sparse Row (CSR) representation for graphs
 *
 * Provides in-memory CSR arrays built from GraphData adjacency lists,
 * plus serialization/deserialization for BLOB storage in shadow tables.
 *
 * CSR layout:
 *   offsets[V+1] — cumulative edge count; node i's neighbors are
 *                  targets[offsets[i] .. offsets[i+1])
 *   targets[E]  — neighbor indices (int32)
 *   weights[E]  — edge weights (double), NULL if unweighted
 */
#ifndef GRAPH_CSR_H
#define GRAPH_CSR_H

#include "graph_load.h"

#include <stdint.h>

/* Default block size for blocked CSR storage (Phase 3).
 * Each block covers this many nodes and is stored as a separate
 * row in the shadow table. Chosen so each block's BLOB is ~16KB
 * (4096 nodes × 4 bytes per offset), fitting in ~4 SQLite pages. */
#define CSR_BLOCK_SIZE 4096

/* In-memory CSR representation for one direction (forward or reverse) */
typedef struct {
    int32_t node_count; /* V */
    int32_t edge_count; /* E */
    int32_t *offsets;   /* [V+1] — offsets[i] = start of node i's neighbors */
    int32_t *targets;   /* [E]   — target node indices */
    double *weights;    /* [E] or NULL if unweighted */
    int has_weights;
} CsrArray;

/* A delta operation for incremental CSR merge */
typedef struct {
    int32_t src_idx;
    int32_t dst_idx;
    double weight;
    int op; /* 1 = INSERT, 2 = DELETE */
} CsrDelta;

/*
 * Build forward and reverse CSR arrays from a GraphData adjacency structure.
 * fwd is built from g->out[], rev from g->in[].
 * Returns 0 on success, -1 on allocation failure.
 * Caller must call csr_destroy() on both arrays.
 */
int csr_build(const GraphData *g, CsrArray *fwd, CsrArray *rev);

/* Free memory owned by a CsrArray. Safe to call on a zeroed struct. */
void csr_destroy(CsrArray *csr);

/* Get the degree (neighbor count) of node idx. Returns 0 if out of range. */
int32_t csr_degree(const CsrArray *csr, int32_t idx);

/*
 * Get pointer to neighbor target indices for node idx.
 * Sets *count to the number of neighbors.
 * Returns pointer into csr->targets, or NULL if idx is out of range.
 */
const int32_t *csr_neighbors(const CsrArray *csr, int32_t idx, int32_t *count);

/*
 * Deserialize a CSR from raw BLOB data (as stored in shadow tables).
 * Makes copies of the input data — caller owns the resulting CsrArray.
 * offsets_blob: int32_t[node_count+1], targets_blob: int32_t[edge_count],
 * weights_blob: double[edge_count] or NULL.
 * Returns 0 on success, -1 on error.
 */
int csr_deserialize(CsrArray *csr, const void *offsets_blob, int offsets_bytes, const void *targets_blob,
                    int targets_bytes, const void *weights_blob, int weights_bytes);

/*
 * Apply delta operations to an existing CSR, producing a new CSR.
 * new_node_count may be larger than old_csr->node_count if new nodes
 * were introduced by the delta.
 * Returns 0 on success, -1 on error.
 */
int csr_apply_delta(const CsrArray *old_csr, const CsrDelta *deltas, int delta_count, int32_t new_node_count,
                    CsrArray *new_csr);

/* ── Phase 3: Blocked CSR ─────────────────────────────────── */

/*
 * Return the number of blocks needed for node_count nodes.
 * Returns ceil(node_count / block_size).
 */
int csr_block_count(int32_t node_count, int block_size);

/*
 * Extract a sub-CSR for one block of nodes.
 * Block covers global nodes [start, start+count). Offsets are rebased
 * to 0 within the block. Targets remain as GLOBAL node indices.
 * Returns 0 on success, -1 on error. Caller must csr_destroy() the block.
 */
int csr_extract_block(const CsrArray *csr, int32_t start, int32_t count, CsrArray *block);

/*
 * Reassemble a monolithic CSR from an array of block CSRs.
 * Each block[i] covers block_size nodes (last block may be smaller).
 * Offsets are rebased to global, targets are already global.
 * Returns 0 on success, -1 on error. Caller must csr_destroy() the result.
 */
int csr_merge_blocks(const CsrArray *blocks, int n_blocks, int block_size, int32_t total_nodes, CsrArray *out);

#endif /* GRAPH_CSR_H */
