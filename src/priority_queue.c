/*
 * priority_queue.c — Binary min-heap implementation
 *
 * Standard array-based heap: parent at i/2, children at 2i and 2i+1.
 * We use 1-based indexing internally (items[0] is unused) for cleaner
 * parent/child arithmetic.
 */
#include "priority_queue.h"
#include <stdlib.h>
#include <string.h>

static void swap(PQItem *a, PQItem *b) {
    PQItem tmp = *a;
    *a = *b;
    *b = tmp;
}

static void sift_up(PQItem *items, int idx) {
    while (idx > 1) {
        int parent = idx / 2;
        if (items[parent].distance <= items[idx].distance) break;
        swap(&items[parent], &items[idx]);
        idx = parent;
    }
}

static void sift_down(PQItem *items, int size, int idx) {
    while (1) {
        int smallest = idx;
        int left = 2 * idx;
        int right = 2 * idx + 1;
        if (left <= size && items[left].distance < items[smallest].distance)
            smallest = left;
        if (right <= size && items[right].distance < items[smallest].distance)
            smallest = right;
        if (smallest == idx) break;
        swap(&items[idx], &items[smallest]);
        idx = smallest;
    }
}

int pq_init(PriorityQueue *pq, int initial_capacity) {
    if (initial_capacity < 4) initial_capacity = 4;
    /* +1 because index 0 is unused (1-based indexing) */
    pq->items = (PQItem *)malloc((size_t)(initial_capacity + 1) * sizeof(PQItem));
    if (!pq->items) return -1;
    pq->size = 0;
    pq->capacity = initial_capacity;
    return 0;
}

int pq_push(PriorityQueue *pq, int64_t id, float distance) {
    if (pq->size >= pq->capacity) {
        int new_cap = pq->capacity * 2;
        PQItem *new_items = (PQItem *)realloc(pq->items, (size_t)(new_cap + 1) * sizeof(PQItem));
        if (!new_items) return -1;
        pq->items = new_items;
        pq->capacity = new_cap;
    }
    pq->size++;
    pq->items[pq->size].id = id;
    pq->items[pq->size].distance = distance;
    sift_up(pq->items, pq->size);
    return 0;
}

PQItem pq_pop(PriorityQueue *pq) {
    PQItem top = pq->items[1];
    pq->items[1] = pq->items[pq->size];
    pq->size--;
    if (pq->size > 0) {
        sift_down(pq->items, pq->size, 1);
    }
    return top;
}

PQItem pq_peek(const PriorityQueue *pq) {
    return pq->items[1];
}

int pq_size(const PriorityQueue *pq) {
    return pq->size;
}

int pq_empty(const PriorityQueue *pq) {
    return pq->size == 0;
}

void pq_clear(PriorityQueue *pq) {
    pq->size = 0;
}

void pq_destroy(PriorityQueue *pq) {
    free(pq->items);
    pq->items = NULL;
    pq->size = 0;
    pq->capacity = 0;
}
