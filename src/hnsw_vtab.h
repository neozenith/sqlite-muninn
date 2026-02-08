/*
 * hnsw_vtab.h — HNSW virtual table for SQLite
 *
 * Registers the "hnsw_index" module with SQLite.
 */
#ifndef HNSW_VTAB_H
#define HNSW_VTAB_H

#include "sqlite3ext.h"

/* Register the hnsw_index virtual table module with db. */
int hnsw_register_module(sqlite3 *db);

#endif /* HNSW_VTAB_H */
