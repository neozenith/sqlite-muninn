/*
 * priority_queue.h — Binary min-heap for HNSW beam search and Dijkstra
 *
 * Stores (id, distance) pairs. Smallest distance at top.
 * For max-heap behavior, negate distances before push.
 */
#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int64_t id;
    float distance;
} PQItem;

typedef struct {
    PQItem *items;
    int size;
    int capacity;
} PriorityQueue;

/* Initialize a priority queue with given initial capacity. Returns 0 on success, -1 on alloc failure. */
int pq_init(PriorityQueue *pq, int initial_capacity);

/* Push an item. Grows capacity if needed. Returns 0 on success, -1 on alloc failure. */
int pq_push(PriorityQueue *pq, int64_t id, float distance);

/* Pop the minimum item. Caller must ensure pq->size > 0. */
PQItem pq_pop(PriorityQueue *pq);

/* Peek at the minimum item without removing. Caller must ensure pq->size > 0. */
PQItem pq_peek(const PriorityQueue *pq);

/* Current number of items. */
int pq_size(const PriorityQueue *pq);

/* Returns 1 if empty. */
int pq_empty(const PriorityQueue *pq);

/* Remove all items (keeps allocated memory). */
void pq_clear(PriorityQueue *pq);

/* Free all allocated memory. PQ is unusable after this. */
void pq_destroy(PriorityQueue *pq);

#endif /* PRIORITY_QUEUE_H */
