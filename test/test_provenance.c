/*
 * test_provenance.c — Unit tests for the gii_provenance virtual table module.
 *
 * Tests are tagged for the per-gap test gates defined in
 * docs/plans/adv-centrality-filtering.md (Execution Plan, "Test tags"):
 *   `make test-g1` → ./build/test_runner --filter=test_g1_
 */
#include "test_common.h"
#include <sqlite3.h>
#include <string.h>

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

void test_provenance(void) {
    RUN_TEST(test_g1_schema_creates_with_xcreate);
    RUN_TEST(test_g1_chunk_insert_populates_provenance);
    RUN_TEST(test_g1_entity_mutations_propagate);
    RUN_TEST(test_g1_canonical_rename_cascades);
    RUN_TEST(test_g1_cluster_rebuild_cascade);
}
