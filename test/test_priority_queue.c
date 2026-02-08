/*
 * test_priority_queue.c — Tests for binary min-heap
 */
#include "test_common.h"
#include "priority_queue.h"

TEST(test_pq_empty) {
    PriorityQueue pq;
    pq_init(&pq, 4);
    ASSERT_EQ_INT(pq_empty(&pq), 1);
    ASSERT_EQ_INT(pq_size(&pq), 0);
    pq_destroy(&pq);
}

TEST(test_pq_push_pop) {
    PriorityQueue pq;
    pq_init(&pq, 4);
    pq_push(&pq, 1, 3.0f);
    pq_push(&pq, 2, 1.0f);
    pq_push(&pq, 3, 2.0f);
    ASSERT_EQ_INT(pq_size(&pq), 3);

    /* Min should come out first */
    PQItem item = pq_pop(&pq);
    ASSERT_EQ_INT((int)item.id, 2);
    ASSERT_EQ_FLOAT(item.distance, 1.0f, 1e-7f);

    item = pq_pop(&pq);
    ASSERT_EQ_INT((int)item.id, 3);
    ASSERT_EQ_FLOAT(item.distance, 2.0f, 1e-7f);

    item = pq_pop(&pq);
    ASSERT_EQ_INT((int)item.id, 1);
    ASSERT_EQ_FLOAT(item.distance, 3.0f, 1e-7f);

    ASSERT_EQ_INT(pq_empty(&pq), 1);
    pq_destroy(&pq);
}

TEST(test_pq_peek) {
    PriorityQueue pq;
    pq_init(&pq, 4);
    pq_push(&pq, 10, 5.0f);
    pq_push(&pq, 20, 2.0f);

    PQItem item = pq_peek(&pq);
    ASSERT_EQ_INT((int)item.id, 20);
    ASSERT_EQ_FLOAT(item.distance, 2.0f, 1e-7f);
    ASSERT_EQ_INT(pq_size(&pq), 2);  /* peek doesn't remove */

    pq_destroy(&pq);
}

TEST(test_pq_grow) {
    PriorityQueue pq;
    pq_init(&pq, 4);  /* start small */

    /* Insert 100 items in reverse order */
    for (int i = 100; i > 0; i--) {
        pq_push(&pq, i, (float)i);
    }
    ASSERT_EQ_INT(pq_size(&pq), 100);

    /* Should come out in ascending order */
    for (int i = 1; i <= 100; i++) {
        PQItem item = pq_pop(&pq);
        ASSERT_EQ_INT((int)item.id, i);
    }
    pq_destroy(&pq);
}

TEST(test_pq_clear) {
    PriorityQueue pq;
    pq_init(&pq, 4);
    pq_push(&pq, 1, 1.0f);
    pq_push(&pq, 2, 2.0f);
    pq_clear(&pq);
    ASSERT_EQ_INT(pq_empty(&pq), 1);
    /* Should still work after clear */
    pq_push(&pq, 3, 3.0f);
    ASSERT_EQ_INT(pq_size(&pq), 1);
    pq_destroy(&pq);
}

TEST(test_pq_duplicate_distances) {
    PriorityQueue pq;
    pq_init(&pq, 4);
    pq_push(&pq, 1, 5.0f);
    pq_push(&pq, 2, 5.0f);
    pq_push(&pq, 3, 5.0f);

    /* All should come out (order among equal distances is unspecified) */
    ASSERT_EQ_INT(pq_size(&pq), 3);
    pq_pop(&pq);
    pq_pop(&pq);
    pq_pop(&pq);
    ASSERT_EQ_INT(pq_empty(&pq), 1);
    pq_destroy(&pq);
}

TEST(test_pq_max_heap_pattern) {
    /* Demonstrate max-heap via negated distances */
    PriorityQueue pq;
    pq_init(&pq, 4);
    pq_push(&pq, 1, -3.0f);  /* actual distance 3.0 */
    pq_push(&pq, 2, -1.0f);  /* actual distance 1.0 */
    pq_push(&pq, 3, -2.0f);  /* actual distance 2.0 */

    /* Max should come out first (most negative = largest actual distance) */
    PQItem item = pq_pop(&pq);
    ASSERT_EQ_INT((int)item.id, 1);  /* distance 3.0 is the max */
    ASSERT_EQ_FLOAT(-item.distance, 3.0f, 1e-7f);

    pq_destroy(&pq);
}

void test_priority_queue(void) {
    RUN_TEST(test_pq_empty);
    RUN_TEST(test_pq_push_pop);
    RUN_TEST(test_pq_peek);
    RUN_TEST(test_pq_grow);
    RUN_TEST(test_pq_clear);
    RUN_TEST(test_pq_duplicate_distances);
    RUN_TEST(test_pq_max_heap_pattern);
}
