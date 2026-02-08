/*
 * id_validate.c — SQL identifier validation
 *
 * This module is used both from extension code and from tests that link
 * directly against libsqlite3, so it uses sqlite3.h rather than sqlite3ext.h.
 */
#include "id_validate.h"
#include <sqlite3.h>
#include <stddef.h>

int id_validate(const char *identifier) {
    if (!identifier || identifier[0] == '\0') return -1;

    for (const char *p = identifier; *p; p++) {
        char c = *p;
        if (!((c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') ||
              c == '_')) {
            return -1;
        }
    }
    return 0;
}

char *id_quote(const char *identifier) {
    if (id_validate(identifier) != 0) return NULL;
    return sqlite3_mprintf("\"%w\"", identifier);
}
