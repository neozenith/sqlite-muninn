/*
 * llama_er.h — Entity Resolution scalar function
 *
 * Registers muninn_extract_er() with SQLite.
 *
 * SQL:
 *   muninn_extract_er(
 *     hnsw_table TEXT,           -- HNSW virtual table name containing entity vectors
 *     name_col TEXT,             -- column name in the source table containing entity names
 *     k INTEGER,                 -- KNN neighbors per entity
 *     dist_threshold REAL,       -- max cosine distance for candidate pairs
 *     jw_weight REAL,            -- Jaro-Winkler vs cosine weight (0.0-1.0)
 *     borderline_delta REAL,     -- LLM window width (0.0 = no LLM)
 *     chat_model TEXT,           -- chat model name (or NULL if delta=0)
 *     edge_betweenness_threshold REAL, -- bridge removal threshold (or NULL to skip)
 *     type_guard TEXT            -- 'same_source' (skip same-source pairs, for record linkage)
 *                                -- 'diff_type'   (skip different-type pairs, for KG ER)
 *                                -- NULL or ''     (no type filtering)
 *   ) -> TEXT (JSON)
 *
 * Returns JSON object: {"clusters": {"entity_id": cluster_id, ...}}
 *
 * The match_threshold is implicit: 1 - dist_threshold + borderline_delta.
 */
#ifndef LLAMA_ER_H
#define LLAMA_ER_H

#include "sqlite3ext.h"

int llama_er_register_functions(sqlite3 *db);

#endif /* LLAMA_ER_H */
