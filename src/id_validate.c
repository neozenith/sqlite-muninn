/*
 * id_validate.c — SQL identifier validation
 *
 * When compiled as part of the extension (SQLITE_CORE not defined),
 * sqlite3ext.h redirects sqlite3_mprintf through the extension API
 * function pointer table. The C test runner links libsqlite3 directly
 * and provides a NULL sqlite3_api pointer (id_quote is not called in tests).
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "id_validate.h"
#include <stddef.h>

int id_validate(const char *identifier) {
    if (!identifier || identifier[0] == '\0')
        return -1;

    for (const char *p = identifier; *p; p++) {
        char c = *p;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            return -1;
        }
    }
    return 0;
}

char *id_quote(const char *identifier) {
    if (id_validate(identifier) != 0)
        return NULL;
    return sqlite3_mprintf("\"%w\"", identifier);
}
