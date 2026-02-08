/*
 * node2vec.h — Node2Vec graph embeddings
 *
 * Learns d-dimensional vector embeddings from graph topology using
 * biased random walks (Grover & Leskovec, KDD 2016) and Skip-gram
 * with Negative Sampling (Mikolov et al., 2013).
 *
 * Exposed as a single SQL scalar function: node2vec_train().
 */
#ifndef NODE2VEC_H
#define NODE2VEC_H

#include "sqlite3ext.h"

/*
 * Register the node2vec_train() scalar function with SQLite.
 * Returns SQLITE_OK on success.
 */
int node2vec_register_functions(sqlite3 *db);

#endif /* NODE2VEC_H */
