/*
 * provenance.c — STUB for T1.1 RED.
 *
 * provenance_register_module() returns SQLITE_ERROR until T1.1 GREEN replaces
 * this with the real virtual-table module + shadow-table xCreate handler.
 */
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT3

#include "provenance.h"

int provenance_register_module(sqlite3 *db) {
    (void)db;
    return SQLITE_ERROR;
}
