/*
 * test_id_validate.c — Tests for SQL injection prevention
 */
#include "test_common.h"
#include "id_validate.h"
#include <stddef.h>

TEST(test_valid_identifiers) {
    ASSERT_EQ_INT(id_validate("my_table"), 0);
    ASSERT_EQ_INT(id_validate("edges"), 0);
    ASSERT_EQ_INT(id_validate("col123"), 0);
    ASSERT_EQ_INT(id_validate("A"), 0);
    ASSERT_EQ_INT(id_validate("_private"), 0);
    ASSERT_EQ_INT(id_validate("CamelCase"), 0);
}

TEST(test_invalid_identifiers) {
    ASSERT_EQ_INT(id_validate(NULL), -1);
    ASSERT_EQ_INT(id_validate(""), -1);
    ASSERT_EQ_INT(id_validate("drop table"), -1);       /* space */
    ASSERT_EQ_INT(id_validate("my-table"), -1);          /* hyphen */
    ASSERT_EQ_INT(id_validate("tab;le"), -1);            /* semicolon */
    ASSERT_EQ_INT(id_validate("x' OR 1=1--"), -1);      /* SQL injection */
    ASSERT_EQ_INT(id_validate("\"quoted\""), -1);        /* quotes */
    ASSERT_EQ_INT(id_validate("table.column"), -1);      /* dot */
    ASSERT_EQ_INT(id_validate("func()"), -1);            /* parens */
}

void test_id_validate(void) {
    RUN_TEST(test_valid_identifiers);
    RUN_TEST(test_invalid_identifiers);
}
