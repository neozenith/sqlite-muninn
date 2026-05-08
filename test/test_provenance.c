/*
 * test_provenance.c — Unit tests for the gii_provenance virtual table module.
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g1` → ./build/test_runner --filter=test_g1_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

extern int provenance_register_module(sqlite3 *db);

/* Mirror the upstream knowledge-graph schema referenced by the trigger
 * definitions in docs/plans/adv-centrality-filtering.md G1.
 *
 * Only the columns the triggers read are declared here — kept minimal so
 * tests stay focused on provenance behavior rather than schema fidelity. */
static void seed_kg_schema(sqlite3 *db) {
    const char *ddl = "CREATE TABLE events ("
                      "  id INTEGER PRIMARY KEY,"
                      "  project_id TEXT,"
                      "  timestamp TEXT);"
                      "CREATE TABLE event_message_chunks ("
                      "  chunk_id INTEGER PRIMARY KEY,"
                      "  event_id INTEGER);"
                      "CREATE TABLE entities ("
                      "  chunk_id INTEGER,"
                      "  name TEXT);"
                      "CREATE TABLE entity_clusters ("
                      "  name TEXT PRIMARY KEY,"
                      "  canonical TEXT);";
    sqlite3_exec(db, ddl, NULL, NULL, NULL);
}

/* Count rows returned by sql. Returns -1 on prepare/step failure. */
static int count_rows(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    int n = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        n = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return n;
}

/* Symmetric-difference count between _gii_provenance and the
 * hand-rolled 4-way JOIN that mirrors payload.py:_allowed_canonicals.
 * Zero means perfect parity. */
static int parity_diff_count(sqlite3 *db) {
    const char *sql = "SELECT "
                      "(SELECT COUNT(*) FROM ("
                      "  SELECT * FROM _gii_provenance "
                      "  EXCEPT "
                      "  SELECT 0, emc.chunk_id, COALESCE(ec.canonical, ent.name), "
                      "         e.project_id, e.timestamp "
                      "  FROM entities ent "
                      "  JOIN event_message_chunks emc ON emc.chunk_id = ent.chunk_id "
                      "  JOIN events e ON e.id = emc.event_id "
                      "  LEFT JOIN entity_clusters ec ON ec.name = ent.name)) "
                      "+ "
                      "(SELECT COUNT(*) FROM ("
                      "  SELECT 0, emc.chunk_id, COALESCE(ec.canonical, ent.name), "
                      "         e.project_id, e.timestamp "
                      "  FROM entities ent "
                      "  JOIN event_message_chunks emc ON emc.chunk_id = ent.chunk_id "
                      "  JOIN events e ON e.id = emc.event_id "
                      "  LEFT JOIN entity_clusters ec ON ec.name = ent.name "
                      "  EXCEPT "
                      "  SELECT * FROM _gii_provenance))";
    return count_rows(db, sql);
}

/* Wall-clock seconds since some monotonic epoch — for T1.8's overhead
 * measurement. Resolution is sub-millisecond on every platform muninn
 * targets (Linux/macOS/Windows-via-MinGW). */
static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Insert n events + n chunks + 2n entities through prepared statements,
 * wrapped in one transaction. Caller controls whether triggers are
 * active by creating (or not) the gii_provenance VT before calling.
 * Returns elapsed seconds. */
static double time_ingest(sqlite3 *db, int n) {
    int rc;
    sqlite3_stmt *ins_event = NULL;
    sqlite3_stmt *ins_chunk = NULL;
    sqlite3_stmt *ins_entity = NULL;

    /* One cluster matches half the entity names — the other half take
     * the COALESCE fallback. Realistic mix of trigger paths. */
    sqlite3_exec(db,
                 "INSERT INTO entity_clusters(name, canonical) "
                 "VALUES ('Foo', 'FooCo');",
                 NULL, NULL, NULL);

    sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);

    rc = sqlite3_prepare_v2(db, "INSERT INTO events(id, project_id, timestamp) VALUES (?,?,?)", -1, &ins_event, NULL);
    if (rc != SQLITE_OK)
        return -1.0;
    rc = sqlite3_prepare_v2(db, "INSERT INTO event_message_chunks(chunk_id, event_id) VALUES (?,?)", -1, &ins_chunk,
                            NULL);
    if (rc != SQLITE_OK)
        return -1.0;
    rc = sqlite3_prepare_v2(db, "INSERT INTO entities(chunk_id, name) VALUES (?,?)", -1, &ins_entity, NULL);
    if (rc != SQLITE_OK)
        return -1.0;

    double t0 = seconds_now();

    for (int i = 1; i <= n; i++) {
        sqlite3_bind_int(ins_event, 1, i);
        sqlite3_bind_text(ins_event, 2, "proj_a", -1, SQLITE_STATIC);
        sqlite3_bind_text(ins_event, 3, "2026-05-08T10:00:00Z", -1, SQLITE_STATIC);
        sqlite3_step(ins_event);
        sqlite3_reset(ins_event);

        sqlite3_bind_int(ins_chunk, 1, i);
        sqlite3_bind_int(ins_chunk, 2, i);
        sqlite3_step(ins_chunk);
        sqlite3_reset(ins_chunk);

        /* Entity 1: matches the 'Foo' cluster. */
        sqlite3_bind_int(ins_entity, 1, i);
        sqlite3_bind_text(ins_entity, 2, "Foo", -1, SQLITE_STATIC);
        sqlite3_step(ins_entity);
        sqlite3_reset(ins_entity);

        /* Entity 2: no cluster, COALESCE fallback. */
        sqlite3_bind_int(ins_entity, 1, i);
        sqlite3_bind_text(ins_entity, 2, "Bar", -1, SQLITE_STATIC);
        sqlite3_step(ins_entity);
        sqlite3_reset(ins_entity);
    }

    double t1 = seconds_now();

    sqlite3_finalize(ins_event);
    sqlite3_finalize(ins_chunk);
    sqlite3_finalize(ins_entity);

    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    return t1 - t0;
}

/* Read the provenance VT's generation counter from its _config shadow.
 * Returns -1 if the row is missing or the query fails. */
static sqlite3_int64 get_generation(sqlite3 *db, const char *vt_name) {
    char *sql = sqlite3_mprintf("SELECT CAST(value AS INTEGER) FROM \"%w_config\" "
                                "WHERE key = 'generation'",
                                vt_name);
    if (!sql)
        return -1;
    sqlite3_stmt *stmt = NULL;
    sqlite3_int64 g = -1;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            g = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_free(sql);
    return g;
}

/* T1.1 — provenance_register_module + xCreate produces the shadow table
 * with schema (namespace_id, chunk_id, canonical, project_id, timestamp,
 * PRIMARY KEY (namespace_id, chunk_id, canonical)) per
 * docs/plans/adv-centrality-filtering.md "Provenance Shadow Table". */
TEST(test_g1_schema_creates_with_xcreate) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Upstream KG schema must exist before xCreate — the trigger installer
     * (T1.2) references event_message_chunks / entities / events /
     * entity_clusters by name and SQLite validates source tables at
     * trigger-creation time. */
    seed_kg_schema(db);

    char *errmsg = NULL;
    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
    }
    ASSERT_EQ_INT(SQLITE_OK, rc);

    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db,
                            "SELECT sql FROM sqlite_master "
                            "WHERE type = 'table' AND name = '_gii_provenance'",
                            -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);

    const char *sql = (const char *)sqlite3_column_text(stmt, 0);
    ASSERT(sql != NULL);
    ASSERT(strstr(sql, "namespace_id") != NULL);
    ASSERT(strstr(sql, "chunk_id") != NULL);
    ASSERT(strstr(sql, "canonical") != NULL);
    ASSERT(strstr(sql, "project_id") != NULL);
    ASSERT(strstr(sql, "timestamp") != NULL);
    ASSERT(strstr(sql, "PRIMARY KEY") != NULL);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* T1.2 — chunk-INSERT trigger group. Inserting a row into
 * event_message_chunks must populate _gii_provenance for every entity
 * already attached to that chunk_id, resolving canonical names through
 * entity_clusters when present and falling back to ent.name otherwise. */
TEST(test_g1_chunk_insert_populates_provenance) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    /* CREATE VIRTUAL TABLE installs the AFTER INSERT trigger on
     * event_message_chunks (added by T1.2 GREEN). */
    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Seed: one event, one cluster, two entities pointing at chunk 42.
     * The first entity has a cluster (resolved canonical); the second
     * has no cluster (fallback to ent.name). */
    rc = sqlite3_exec(db,
                      "INSERT INTO events(id, project_id, timestamp) "
                      "  VALUES (1, 'proj_a', '2026-05-08T10:00:00Z');"
                      "INSERT INTO entity_clusters(name, canonical) "
                      "  VALUES ('AcmeCorp', 'Acme Corporation');"
                      "INSERT INTO entities(chunk_id, name) "
                      "  VALUES (42, 'AcmeCorp'), (42, 'Bob');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* This insert should fire the trigger and populate two rows. */
    rc = sqlite3_exec(db, "INSERT INTO event_message_chunks(chunk_id, event_id) VALUES (42, 1);", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Verify rows: ordered alphabetically by canonical so the assertions
     * have a stable shape regardless of trigger emission order. */
    sqlite3_stmt *stmt = NULL;
    rc = sqlite3_prepare_v2(db,
                            "SELECT namespace_id, chunk_id, canonical, project_id, timestamp "
                            "FROM _gii_provenance "
                            "WHERE chunk_id = 42 "
                            "ORDER BY canonical",
                            -1, &stmt, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Row 1: cluster-resolved 'Acme Corporation' */
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);
    ASSERT_EQ_INT(0, sqlite3_column_int(stmt, 0));
    ASSERT_EQ_INT(42, sqlite3_column_int(stmt, 1));
    const char *canonical = (const char *)sqlite3_column_text(stmt, 2);
    ASSERT(canonical != NULL && strcmp(canonical, "Acme Corporation") == 0);
    const char *project = (const char *)sqlite3_column_text(stmt, 3);
    ASSERT(project != NULL && strcmp(project, "proj_a") == 0);
    const char *ts = (const char *)sqlite3_column_text(stmt, 4);
    ASSERT(ts != NULL && strcmp(ts, "2026-05-08T10:00:00Z") == 0);

    /* Row 2: name-fallback 'Bob' (no cluster) */
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_ROW, rc);
    canonical = (const char *)sqlite3_column_text(stmt, 2);
    ASSERT(canonical != NULL && strcmp(canonical, "Bob") == 0);

    /* Exactly two rows */
    rc = sqlite3_step(stmt);
    ASSERT_EQ_INT(SQLITE_DONE, rc);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

/* T1.3 — entity INSERT/DELETE/UPDATE trigger group. Mutating the
 * `entities` table after the chunk is already in event_message_chunks
 * must keep _gii_provenance in sync without a manual rebuild. */
TEST(test_g1_entity_mutations_propagate) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Setup: event + cluster + chunk (no entities yet). The chunk-INSERT
     * trigger from T1.2 fires here but emits zero rows — there are no
     * entities to project. */
    rc = sqlite3_exec(db,
                      "INSERT INTO events(id, project_id, timestamp) "
                      "  VALUES (1, 'proj_a', '2026-05-08T10:00:00Z');"
                      "INSERT INTO entity_clusters(name, canonical) "
                      "  VALUES ('AcmeCorp', 'Acme Corp');"
                      "INSERT INTO event_message_chunks(chunk_id, event_id) "
                      "  VALUES (42, 1);",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance"));

    /* INSERT entity → provenance row appears, canonical resolved through
     * entity_clusters. */
    rc = sqlite3_exec(db, "INSERT INTO entities(chunk_id, name) VALUES (42, 'AcmeCorp');", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE chunk_id = 42 AND canonical = 'Acme Corp' "
                                    "  AND project_id = 'proj_a'"));

    /* UPDATE entity name → old provenance row removed, new row appears.
     * The new name 'Bob' has no cluster, so it falls through to ent.name. */
    rc = sqlite3_exec(db, "UPDATE entities SET name = 'Bob' WHERE chunk_id = 42 AND name = 'AcmeCorp';", NULL, NULL,
                      NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0,
                  count_rows(db, "SELECT COUNT(*) FROM _gii_provenance WHERE canonical = 'Acme Corp'"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE chunk_id = 42 AND canonical = 'Bob'"));

    /* DELETE entity → provenance row removed; provenance is back to empty. */
    rc = sqlite3_exec(db, "DELETE FROM entities WHERE chunk_id = 42 AND name = 'Bob';", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance"));

    sqlite3_close(db);
}

/* T1.4 — entity_clusters UPDATE-rename cascade. Renaming the canonical
 * column on a cluster must remap every provenance row that points at
 * the old canonical. Mirrors the realistic ER workflow where a manual
 * canonicalization fix-up needs to propagate without rebuilding the
 * whole shadow table. */
TEST(test_g1_canonical_rename_cascades) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Seed two chunks under different events but the same cluster, so
     * the rename has to touch multiple rows at once. */
    rc = sqlite3_exec(db,
                      "INSERT INTO events(id, project_id, timestamp) VALUES "
                      "  (1, 'proj_a', '2026-05-08T10:00:00Z'),"
                      "  (2, 'proj_a', '2026-05-08T11:00:00Z');"
                      "INSERT INTO entity_clusters(name, canonical) "
                      "  VALUES ('AcmeCorp', 'Acme Corp');"
                      "INSERT INTO entities(chunk_id, name) VALUES "
                      "  (42, 'AcmeCorp'), (99, 'AcmeCorp');"
                      "INSERT INTO event_message_chunks(chunk_id, event_id) VALUES "
                      "  (42, 1), (99, 2);",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Pre-rename baseline: two rows, both under canonical = 'Acme Corp'. */
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'Acme Corp'"));

    /* Rename the cluster's canonical. */
    rc = sqlite3_exec(db,
                      "UPDATE entity_clusters SET canonical = 'Acme Corporation' "
                      "WHERE name = 'AcmeCorp';",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Both provenance rows must now point at the new canonical, none at
     * the old one. */
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'Acme Corp'"));
    ASSERT_EQ_INT(2, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'Acme Corporation'"));

    sqlite3_close(db);
}

/* T1.5 — entity_clusters INSERT/DELETE rebuild cascade. A full ER
 * rebuild that does DELETE-all + INSERT-all of entity_clusters wouldn't
 * fire UPDATE triggers and would silently leave provenance frozen at
 * whatever canonical assignment was active before the rebuild. The
 * INSERT/DELETE triggers fix this by remapping provenance rows on
 * cluster lifecycle events.
 *
 * Round-trip: entity stored with raw-name canonical → cluster INSERT
 * remaps to canonical → cluster DELETE reverts to raw. */
TEST(test_g1_cluster_rebuild_cascade) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Seed event + chunk + entity *without* any cluster, so the entity's
     * provenance row stores the raw name (COALESCE fallback). */
    rc = sqlite3_exec(db,
                      "INSERT INTO events(id, project_id, timestamp) "
                      "  VALUES (1, 'proj_a', '2026-05-08T10:00:00Z');"
                      "INSERT INTO event_message_chunks(chunk_id, event_id) "
                      "  VALUES (42, 1);"
                      "INSERT INTO entities(chunk_id, name) "
                      "  VALUES (42, 'AcmeCorp');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Pre-cluster baseline: provenance has the raw name as canonical. */
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'AcmeCorp'"));

    /* INSERT cluster → remap raw 'AcmeCorp' to canonical 'Acme Corp'. */
    rc = sqlite3_exec(db,
                      "INSERT INTO entity_clusters(name, canonical) "
                      "  VALUES ('AcmeCorp', 'Acme Corp');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'AcmeCorp'"));
    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'Acme Corp'"));

    /* DELETE cluster → revert canonical back to the raw name. */
    rc = sqlite3_exec(db, "DELETE FROM entity_clusters WHERE name = 'AcmeCorp';", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    ASSERT_EQ_INT(1, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'AcmeCorp'"));
    ASSERT_EQ_INT(0, count_rows(db, "SELECT COUNT(*) FROM _gii_provenance "
                                    "WHERE canonical = 'Acme Corp'"));

    sqlite3_close(db);
}

/* T1.6 — G_prov generation tick on every change. Each mutation that
 * fires a provenance trigger must strictly increment the generation
 * counter persisted in _<vt>_config. The counter is the cache-invalidation
 * hook G2 (top-K result cache) and G7 (community filter) read.
 *
 * Walks all seven trigger groups in order so non-monotonic ticks fail at
 * the offending mutation, not at a final aggregate check. */
TEST(test_g1_generation_strictly_increases) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    sqlite3_int64 g0 = get_generation(db, "_gii");
    ASSERT(g0 >= 0); /* xCreate must initialize the counter */

    /* (1) AFTER INSERT on event_message_chunks fires _emc_ai. Generation
     * ticks even though no entities exist yet — over-invalidate over
     * under-invalidate. */
    sqlite3_exec(db,
                 "INSERT INTO events(id, project_id, timestamp) "
                 "  VALUES (1, 'proj_a', '2026-05-08T10:00:00Z');"
                 "INSERT INTO event_message_chunks(chunk_id, event_id) "
                 "  VALUES (42, 1);",
                 NULL, NULL, NULL);
    sqlite3_int64 g1 = get_generation(db, "_gii");
    ASSERT(g1 > g0);

    /* (2) AFTER INSERT on entities fires _ent_ai. */
    sqlite3_exec(db, "INSERT INTO entities(chunk_id, name) VALUES (42, 'AcmeCorp');", NULL, NULL, NULL);
    sqlite3_int64 g2 = get_generation(db, "_gii");
    ASSERT(g2 > g1);

    /* (3) AFTER INSERT on entity_clusters fires _ec_ai (raw → canonical
     * remap). */
    sqlite3_exec(db, "INSERT INTO entity_clusters(name, canonical) VALUES ('AcmeCorp', 'Acme Corp');", NULL, NULL,
                 NULL);
    sqlite3_int64 g3 = get_generation(db, "_gii");
    ASSERT(g3 > g2);

    /* (4) AFTER UPDATE OF canonical on entity_clusters fires _ec_au
     * (canonical rename). */
    sqlite3_exec(db, "UPDATE entity_clusters SET canonical = 'Acme Corporation' WHERE name = 'AcmeCorp';", NULL, NULL,
                 NULL);
    sqlite3_int64 g4 = get_generation(db, "_gii");
    ASSERT(g4 > g3);

    /* (5) AFTER UPDATE on entities fires _ent_au (DELETE-OLD + INSERT-NEW). */
    sqlite3_exec(db, "UPDATE entities SET name = 'Bob' WHERE chunk_id = 42 AND name = 'AcmeCorp';", NULL, NULL, NULL);
    sqlite3_int64 g5 = get_generation(db, "_gii");
    ASSERT(g5 > g4);

    /* (6) AFTER DELETE on entities fires _ent_ad. */
    sqlite3_exec(db, "DELETE FROM entities WHERE chunk_id = 42 AND name = 'Bob';", NULL, NULL, NULL);
    sqlite3_int64 g6 = get_generation(db, "_gii");
    ASSERT(g6 > g5);

    /* (7) AFTER DELETE on entity_clusters fires _ec_ad (canonical → raw
     * revert). */
    sqlite3_exec(db, "DELETE FROM entity_clusters WHERE name = 'AcmeCorp';", NULL, NULL, NULL);
    sqlite3_int64 g7 = get_generation(db, "_gii");
    ASSERT(g7 > g6);

    sqlite3_close(db);
}

/* T1.7 — row parity between the maintained _gii_provenance shadow and
 * the canonical 4-way JOIN over events × event_message_chunks ×
 * entities × entity_clusters. The trigger maintenance must exactly
 * reproduce what a hand-rolled JOIN would compute, otherwise the
 * shadow is silently wrong and any downstream cache (G2, G7) inherits
 * that wrongness.
 *
 * Fixture exercises the multi-entity-per-canonical case deferred
 * during T1.3: two clusters share the same canonical, and two
 * entities under one chunk both resolve through them. PK constraint
 * collapses them to one provenance row (matching 4-way JOIN's set
 * semantics). Deleting one of those entities is what surfaces the
 * simple-DELETE bug — the other entity still resolves to that
 * canonical so the row should remain. */
TEST(test_g1_provenance_parity_with_4way_join) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    rc = provenance_register_module(db);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    seed_kg_schema(db);

    rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* Non-trivial KG fixture across two events / three chunks / five
     * entities / three clusters. Two clusters intentionally share
     * canonical = 'Acme Corp' so chunk 100 has two entities mapping
     * to the same canonical — exposing the count-aware-DELETE
     * requirement. */
    rc = sqlite3_exec(db,
                      "INSERT INTO events(id, project_id, timestamp) VALUES "
                      "  (1, 'proj_a', '2026-05-08T10:00:00Z'),"
                      "  (2, 'proj_b', '2026-05-08T11:00:00Z');"
                      "INSERT INTO entity_clusters(name, canonical) VALUES "
                      "  ('AcmeCorp',    'Acme Corp'),"
                      "  ('AcmeCorpInc', 'Acme Corp'),"
                      "  ('Charlie',     'C-Charlie');"
                      "INSERT INTO event_message_chunks(chunk_id, event_id) VALUES "
                      "  (100, 1), (101, 2), (102, 2);"
                      "INSERT INTO entities(chunk_id, name) VALUES "
                      "  (100, 'AcmeCorp'),"
                      "  (100, 'AcmeCorpInc')," /* same canonical as above */
                      "  (101, 'AcmeCorp'),"
                      "  (101, 'Bob')," /* no cluster — name fallback */
                      "  (102, 'Charlie');",
                      NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);

    /* After all inserts: provenance must match the 4-way JOIN exactly. */
    ASSERT_EQ_INT(0, parity_diff_count(db));

    /* DELETE one of the two entities sharing 'Acme Corp' on chunk 100.
     * The 4-way JOIN still includes (0, 100, 'Acme Corp', ...) because
     * the OTHER entity ('AcmeCorp') still resolves there. The trigger
     * must keep the provenance row alive — count-aware DELETE. */
    rc = sqlite3_exec(db, "DELETE FROM entities WHERE chunk_id = 100 AND name = 'AcmeCorpInc';", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, parity_diff_count(db));

    /* Sanity: deleting the last entity that maps to a canonical does
     * remove the provenance row. */
    rc = sqlite3_exec(db, "DELETE FROM entities WHERE chunk_id = 101 AND name = 'Bob';", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    ASSERT_EQ_INT(0, parity_diff_count(db));

    sqlite3_close(db);
}

/* T1.8 — ingest overhead within 2× of the no-trigger baseline. The
 * provenance shadow buys downstream cache invalidation, but the
 * trigger maintenance must not dominate ingestion cost. Two :memory:
 * databases on the same schema; one has the gii_provenance VT
 * installed (triggers fire), the other does not (bare INSERTs).
 *
 * Test reports the measured ratio when bound is exceeded so a
 * timing-noise failure shows the actual numbers. */
TEST(test_g1_ingest_overhead_bounded) {
    const int N = 2000;

    /* Baseline: schema only, no VT, no triggers. */
    sqlite3 *db_baseline = NULL;
    int rc = sqlite3_open(":memory:", &db_baseline);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    seed_kg_schema(db_baseline);
    double baseline = time_ingest(db_baseline, N);
    sqlite3_close(db_baseline);
    ASSERT(baseline > 0.0);

    /* With triggers: schema + VT installed. */
    sqlite3 *db_triggered = NULL;
    rc = sqlite3_open(":memory:", &db_triggered);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    rc = provenance_register_module(db_triggered);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    seed_kg_schema(db_triggered);
    rc = sqlite3_exec(db_triggered, "CREATE VIRTUAL TABLE _gii USING gii_provenance()", NULL, NULL, NULL);
    ASSERT_EQ_INT(SQLITE_OK, rc);
    double triggered = time_ingest(db_triggered, N);
    sqlite3_close(db_triggered);
    ASSERT(triggered > 0.0);

    double ratio = triggered / baseline;
    if (ratio > 2.0) {
        printf("    (ingest overhead %.2fx — baseline=%.4fs, triggered=%.4fs)\n", ratio, baseline, triggered);
    }
    ASSERT(ratio <= 2.0);
}

void test_provenance(void) {
    RUN_TEST(test_g1_schema_creates_with_xcreate);
    RUN_TEST(test_g1_chunk_insert_populates_provenance);
    RUN_TEST(test_g1_entity_mutations_propagate);
    RUN_TEST(test_g1_canonical_rename_cascades);
    RUN_TEST(test_g1_cluster_rebuild_cascade);
    RUN_TEST(test_g1_generation_strictly_increases);
    RUN_TEST(test_g1_provenance_parity_with_4way_join);
    RUN_TEST(test_g1_ingest_overhead_bounded);
}
