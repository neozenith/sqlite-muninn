/*
 * graph_select_tvf.h — Registration for graph_select TVF
 */
#ifndef GRAPH_SELECT_TVF_H
#define GRAPH_SELECT_TVF_H

#include <sqlite3.h>

/* Register the graph_select eponymous-only TVF. */
int graph_select_register_tvf(sqlite3 *db);

#endif /* GRAPH_SELECT_TVF_H */
