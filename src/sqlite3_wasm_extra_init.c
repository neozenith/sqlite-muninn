/*
 * sqlite3_wasm_extra_init.c — Static extension registration for WASM builds.
 *
 * In WASM, SQLite extensions cannot be dynamically loaded. Instead, this file
 * registers muninn as an auto-extension that initializes with every new
 * database connection.
 *
 * Build:
 *   emcc -O2 -s WASM=1 \
 *       vendor/sqlite3.c dist/muninn.c src/sqlite3_wasm_extra_init.c \
 *       -o dist/muninn_sqlite3.js
 */
#include "sqlite3.h"

extern int sqlite3_muninn_init(sqlite3 *, char **, const sqlite3_api_routines *);

int sqlite3_wasm_extra_init(const char *z) {
    (void)z; /* unused */
    return sqlite3_auto_extension((void (*)(void))sqlite3_muninn_init);
}
