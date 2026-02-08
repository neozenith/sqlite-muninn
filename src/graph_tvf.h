/*
 * graph_tvf.h — Graph table-valued functions (Phase 2)
 */
#ifndef GRAPH_TVF_H
#define GRAPH_TVF_H

#include "sqlite3ext.h"

int graph_register_tvfs(sqlite3 *db);

#endif /* GRAPH_TVF_H */
