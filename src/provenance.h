/*
 * provenance.h — `gii_provenance` virtual table module
 *
 * Maintains a denormalized provenance shadow table for the KG filter chain
 * (events → chunks → entities → entity_clusters), collapsing the 4-way join
 * at `claude-code-sessions/.../kg/payload.py:_allowed_canonicals` into a
 * single indexed lookup. See docs/plans/adv-centrality-filtering.md G1.
 *
 * Usage:
 *
 *   CREATE VIRTUAL TABLE _gii USING gii_provenance();
 *
 * Side effect: shadow table "<name>_provenance" is created with schema
 *
 *   CREATE TABLE "<name>_provenance" (
 *       namespace_id INTEGER NOT NULL,
 *       chunk_id     INTEGER NOT NULL,
 *       canonical    TEXT NOT NULL,
 *       project_id   TEXT NOT NULL,
 *       timestamp    TEXT NOT NULL,
 *       PRIMARY KEY (namespace_id, chunk_id, canonical)
 *   );
 *
 * The shadow table is the storage; the VT itself is the query/maintenance
 * surface (later tickets add INSERT/UPDATE/DELETE triggers and generation
 * counter cascade).
 */
#ifndef PROVENANCE_H
#define PROVENANCE_H

#include "sqlite3ext.h"

/* Register the gii_provenance virtual table module with db. */
int provenance_register_module(sqlite3 *db);

#endif /* PROVENANCE_H */
