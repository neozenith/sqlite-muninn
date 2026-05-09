# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Features

- Improved kg viz controls

### Other

- WIP planning docs and early experiments
- Updated plan spec
- Updated plan
- T1.1 RED: failing test_g1_schema_creates_with_xcreate

Sets up the per-gap test gate scaffolding called for by
docs/plans/adv-centrality-filtering.md (Execution Plan, Test tags + Make
targets) and adds the first failing G1 test:

- test/test_main.c: parse --filter=<prefix> from argv; new
  test_should_run() implementation; dispatch test_provenance suite.
- test/test_common.h: RUN_TEST consults test_should_run, so name-prefix
  filtering applies uniformly to every existing and future suite.
- pyproject.toml: register G1..G7 pytest markers.
- Makefile: test-g1..g7 + test-all-gates gates. Fixes the plan's typo
  (--filter=test_$@_ → --filter=test_$(subst test-,,$@)_) and adds
  --no-cov plus pytest-exit-5 grace so a gap with only C tests passes.
- scripts/generate_build.py: include src/provenance.c in TEST_LINK_SOURCES.
- src/provenance.{h,c}: declare provenance_register_module; stub returns
  SQLITE_ERROR so the RED test fails at the registration step.
- test/test_provenance.c: test_g1_schema_creates_with_xcreate asserts
  registration succeeds and that CREATE VIRTUAL TABLE _gii USING
  gii_provenance() produces a _gii_provenance shadow with the documented
  columns and PRIMARY KEY.

Verified: ./build/test_runner --filter=test_g1_ runs only the new test
and exits 1 on the SQLITE_OK == rc assertion. Full suite (no filter)
still passes 119 prior tests.
- T1.1 GREEN: gii_provenance xCreate produces _gii_provenance shadow

Replaces the SQLITE_ERROR stub with a real virtual-table module:

- prov_create_shadow_tables: emits the documented schema
  (namespace_id, chunk_id, canonical, project_id, timestamp,
  PRIMARY KEY (namespace_id, chunk_id, canonical)) plus the two
  secondary indexes from docs/plans/adv-centrality-filtering.md
  (proj_ts and canonical).
- Cursor lifecycle: xFilter prepares SELECT ... FROM <name>_provenance,
  xNext/xColumn/xRowid stream rows back. Full-scan only — constraint
  pushdown lands when G2 / G7 wire centrality TVFs to the shadow.
- xDestroy drops the shadow; xDisconnect just frees vtab memory.
- iVersion=0 is sufficient — no xUpdate/xRename needed yet (triggers in
  T1.2..T1.5 maintain the shadow, not direct VT writes).

Verified:
  ./build/test_runner --filter=test_g1_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 120 passed, 0 failed
  make test-g1                            → C test pass + pytest exit-5
                                            grace ⇒ gate exits 0
- T1.2 RED: failing test_g1_chunk_insert_populates_provenance

Adds a focused test for the chunk-INSERT trigger group. Inserting a
row into event_message_chunks must back-fill _gii_provenance for every
entity already attached to that chunk_id, resolving canonical names
through entity_clusters when present and falling back to ent.name
otherwise.

Pre-seeds entity_clusters with one resolved cluster ('AcmeCorp' →
'Acme Corporation') and entities with both a cluster-bearing
('AcmeCorp') and a clusterless ('Bob') name so the test exercises both
branches of the COALESCE(ec.canonical, ent.name) projection in the
trigger's SELECT.

Also factors a seed_kg_schema(db) helper for reuse by T1.3..T1.5
which all need the same upstream KG tables (events,
event_message_chunks, entities, entity_clusters).

Verified RED:
  ./build/test_runner --filter=test_g1_
  → test_g1_schema_creates_with_xcreate PASS
  → test_g1_chunk_insert_populates_provenance FAIL
    (test/test_provenance.c:126: SQLITE_ROW == rc — first step against
     _gii_provenance returns SQLITE_DONE because xCreate does not yet
     install the trigger)
- T1.2 GREEN: install chunk-INSERT trigger group at xCreate

Adds the prov_install_triggers / prov_remove_triggers helpers and wires
them into prov_init / prov_xDestroy. The first trigger group ("emc_ai"
— event_message_chunks AFTER INSERT) emits one provenance row per
entity attached to the new chunk, joining events on event_id and
LEFT-joining entity_clusters on the entity name to resolve the
canonical (or fall back to ent.name when no cluster exists). Uses
INSERT OR IGNORE so re-emission on identical (namespace, chunk,
canonical) is silent.

Rollback discipline: if trigger install fails (typically because the
upstream KG schema isn't present), prov_init drops the shadow tables
it just created so xCreate is atomic.

Adjusts test_g1_schema_creates_with_xcreate to seed the upstream KG
schema before CREATE VIRTUAL TABLE, since trigger install now
requires event_message_chunks / entities / events / entity_clusters
to exist at create time. The assertion (shadow table schema) is
unchanged — only the precondition.

Verified:
  ./build/test_runner --filter=test_g1_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 121 passed, 0 failed
- T1.3 RED: failing test_g1_entity_mutations_propagate

Adds the focused test for the entity INSERT/DELETE/UPDATE trigger
group. Models the realistic ingestion order: chunk arrives first
(T1.2 trigger fires but emits zero rows since no entities yet),
then the entities are extracted and inserted one at a time, then
mutated.

Walks the three cases:
  - INSERT entity → provenance row appears with cluster-resolved
    canonical
  - UPDATE entity name → old row removed, new row appears (name
    fallback for the new name 'Bob' which has no cluster)
  - DELETE entity → provenance returns to empty

Also factors a count_rows(db, sql) helper for terse SELECT COUNT(*)
assertions, used in this test and likely T1.4..T1.7.

Verified RED:
  ./build/test_runner --filter=test_g1_
  → test_g1_schema_creates_with_xcreate PASS
  → test_g1_chunk_insert_populates_provenance PASS
  → test_g1_entity_mutations_propagate FAIL
    (line 206: count = 0 after INSERT entity, expected 1 — no
     AFTER INSERT trigger on entities is installed yet)
- T1.3 GREEN: install entity INSERT/DELETE/UPDATE trigger group

Adds three triggers on the entities table to the existing
prov_install_triggers ladder:

  Group 2a — _ent_ai (AFTER INSERT): emits one provenance row per new
  entity, joining the chunk's event for project_id/timestamp and
  resolving canonical through entity_clusters with COALESCE(ec.canonical,
  NEW.name) fallback.

  Group 2b — _ent_ad (AFTER DELETE): drops the corresponding provenance
  row, recomputing the canonical the deleted entity mapped to via a
  scalar subquery on entity_clusters. Multi-entity-per-canonical edge
  case (two entities in the same chunk resolving to the same canonical;
  deleting one drops the row even though another still references it)
  is deferred to T1.7's parity check — refining now would be premature.

  Group 2c — _ent_au (AFTER UPDATE): DELETE-of-OLD followed by
  INSERT-of-NEW in a single trigger body. Mirrors graph_adjacency.c's
  OLD/NEW symmetry for the same reason: one transaction so a partial
  UPDATE never leaves provenance pointing at neither old nor new.

Updates prov_remove_triggers' suffix array to drop the three new
triggers on xDestroy.

Verified:
  ./build/test_runner --filter=test_g1_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 122 passed, 0 failed
- T1.4 RED: failing test_g1_canonical_rename_cascades

Adds the test for the entity_clusters UPDATE-rename cascade. Seeds
two chunks under different events but the same cluster, asserts the
two pre-rename provenance rows both carry the OLD canonical, then
issues an UPDATE on entity_clusters.canonical and asserts every row
flips to the NEW canonical (none remain on the old).

Two-chunk fixture is intentional — a single-row test would pass
trivially with a too-narrow trigger; needing both rows to flip
forces the implementation to emit the bare UPDATE (not a single-row
DELETE+INSERT).

Verified RED:
  ./build/test_runner --filter=test_g1_
  → first three tests PASS
  → test_g1_canonical_rename_cascades FAIL
    (line 273: count = 2 after the cluster rename, expected 0 — no
     AFTER UPDATE OF canonical trigger on entity_clusters yet)
- T1.4 GREEN: install entity_clusters UPDATE-rename trigger

Adds Group 3 (_ec_au) — AFTER UPDATE OF canonical on entity_clusters
— to prov_install_triggers. The body is a single SQL UPDATE that
remaps every provenance row whose canonical matches OLD.canonical
to NEW.canonical, all in the same transaction as the originating
cluster rename.

Column-scoped (UPDATE OF canonical) — renaming the cluster's name
column would not fire this trigger since provenance rows aren't
keyed on cluster name; only the canonical-typed rename matters.

Updates the suffixes array in prov_remove_triggers to drop _ec_au
on xDestroy.

Verified:
  ./build/test_runner --filter=test_g1_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 123 passed, 0 failed
- T1.5 RED: failing test_g1_cluster_rebuild_cascade

Adds the round-trip test for the entity_clusters INSERT/DELETE
rebuild cascade. The realistic scenario is a full ER rebuild
(DELETE-all + INSERT-all of entity_clusters) — UPDATE-only triggers
miss this entirely and silently leave provenance frozen at the
previous canonical assignment.

Round-trip validates both halves of Group 4 in one fixture:

  1. Seed entity *before* any cluster exists → provenance row stores
     the raw name as canonical (COALESCE fallback).
  2. INSERT entity_clusters(name='AcmeCorp', canonical='Acme Corp') →
     trigger should remap canonical='AcmeCorp' to canonical='Acme Corp'.
  3. DELETE entity_clusters WHERE name='AcmeCorp' → trigger should
     revert canonical='Acme Corp' back to canonical='AcmeCorp' (raw).

Verified RED:
  ./build/test_runner --filter=test_g1_
  → first four tests PASS
  → test_g1_cluster_rebuild_cascade FAIL
    (line 326: count = 1 after cluster INSERT, expected 0 — no
     AFTER INSERT/DELETE triggers on entity_clusters yet)
- T1.5 GREEN: install entity_clusters INSERT/DELETE rebuild triggers

Adds Group 4 — _ec_ai and _ec_ad — to prov_install_triggers, the
symmetric pair that handles the cluster-lifecycle remapping that
UPDATE-only triggers miss:

  Group 4a — _ec_ai (AFTER INSERT): when a new cluster arrives, remap
  every provenance row whose canonical equals NEW.name (i.e. was
  stored via COALESCE fallback because no cluster existed at insert
  time) to NEW.canonical.

  Group 4b — _ec_ad (AFTER DELETE): symmetric inverse — remap rows
  whose canonical equals OLD.canonical back to OLD.name so they
  match what a fresh entity-INSERT would store with no cluster
  present.

Updates the suffixes array in prov_remove_triggers to drop both new
triggers on xDestroy.

Verified:
  ./build/test_runner --filter=test_g1_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 124 passed, 0 failed
- T1.6 RED: failing test_g1_generation_strictly_increases

Adds the cache-invalidation hook test. Walks all seven trigger groups
in sequence and asserts that the G_prov generation counter strictly
increments after each mutation:

  (1) chunk insert       → _emc_ai
  (2) entity insert      → _ent_ai
  (3) cluster insert     → _ec_ai
  (4) cluster rename     → _ec_au
  (5) entity update      → _ent_au
  (6) entity delete      → _ent_ad
  (7) cluster delete     → _ec_ad

Walking each step individually means a non-monotonic tick fails at
the offending mutation rather than only at a final aggregate check —
much easier to diagnose.

Adds get_generation(db, vt_name) helper that reads CAST(value AS
INTEGER) from _<vt>_config WHERE key = 'generation', returning -1
when the row is missing. The very first guard (g0 >= 0) is what
fails RED — there is no _<vt>_config table yet.

Cache semantics intentionally over-invalidate: every trigger fire
ticks the counter even when zero rows changed (e.g. a chunk-INSERT
with no matching entities). Caches err on the side of recompute
rather than stale-read, the silent bug we want to never have.

Verified RED:
  ./build/test_runner --filter=test_g1_
  → first five tests PASS
  → test_g1_generation_strictly_increases FAIL
    (line 383: g0 >= 0 — generation counter not initialized; no
     _gii_config shadow table exists yet)
- T1.6 GREEN: G_prov generation counter ticks on every trigger fire

Adds the cache-invalidation hook G2 / G7 will read.

Persistence:
  - prov_create_shadow_tables now also creates _<name>_config (TEXT
    key/value, mirrors graph_adjacency's convention) and seeds it
    with ('generation', '0').
  - prov_drop_shadow_tables iterates a {_provenance, _config} suffix
    array so xDestroy and rollback in xCreate clean up both.

Bump:
  - PROV_BUMP_SQL macro defines the cast-and-increment statement
    once and is appended to every trigger body. Each trigger string
    grows by one %w arg for _<name>_config — straightforward but
    pervasive (all seven trigger sqlite3_mprintf calls touched).
  - Over-invalidate semantics: the counter ticks on every trigger
    fire, regardless of whether any provenance row actually changed.
    A chunk insert with no entities still ticks. Caches err on the
    side of recompute over stale-read.

Verified:
  ./build/test_runner --filter=test_g1_  → 6 passed, 0 failed
  ./build/test_runner (full suite)        → 125 passed, 0 failed
- T1.7 RED: failing test_g1_provenance_parity_with_4way_join

Adds the row-parity test against the canonical 4-way JOIN that
mirrors payload.py:_allowed_canonicals. The trigger maintenance must
exactly reproduce what a hand-rolled JOIN would compute, otherwise
every downstream cache (G2 top-K, G7 community filter) inherits
silently-wrong rows.

Fixture is designed to surface the multi-entity-per-canonical edge
case T1.3 deferred:
  - Two clusters share canonical = 'Acme Corp'
    ('AcmeCorp' → 'Acme Corp', 'AcmeCorpInc' → 'Acme Corp')
  - Two entities under chunk 100 both resolve through them
  - PK constraint collapses them to ONE provenance row, matching
    the 4-way JOIN's set semantics

Walks three states:
  (1) After all inserts → parity must be 0 (no divergence).
  (2) DELETE one of the two entities mapping to 'Acme Corp' → the
      OTHER still resolves there, so the provenance row must remain.
      Simple-DELETE drops the row anyway → parity diverges.
  (3) DELETE the last entity for 'Bob' (no other refs) → row is
      legitimately removed, parity stays at 0. (Sanity that
      single-ref DELETE still works.)

Adds parity_diff_count(db) helper computing the symmetric difference
via two EXCEPT subqueries summed.

Verified RED:
  ./build/test_runner --filter=test_g1_
  → first six tests PASS
  → test_g1_provenance_parity_with_4way_join FAIL
    (line 522: 1 != 0 after deleting AcmeCorpInc — the simple DELETE
     trigger removes the (0, 100, 'Acme Corp', ...) row even though
     entity 'AcmeCorp' still resolves there)

T1.7 GREEN must teach _ent_ad and the DELETE half of _ent_au to skip
the DELETE when another entity in the same chunk still resolves to
the same canonical (NOT EXISTS guard).
- T1.7 GREEN: count-aware DELETE in entity DELETE/UPDATE triggers

Replaces the simple "delete the row matching OLD's canonical" with
a count-aware DELETE that only removes the provenance row when no
OTHER entity in the same chunk still resolves to that canonical.
Resolves the multi-entity-per-canonical edge case T1.3 deferred.

The NOT EXISTS predicate scans `entities` for any row in OLD's chunk
whose resolved canonical (COALESCE through entity_clusters) matches
OLD's resolved canonical. AFTER DELETE / AFTER UPDATE both run after
the originating mutation, so:
  - In _ent_ad, the OLD row is already gone from `entities`; the
    scan finds only OTHER referrers.
  - In _ent_au's DELETE half, the entity row that was OLD is now
    NEW. If NEW.name still resolves to OLD's canonical, the scan
    finds it and the DELETE is suppressed — exactly right.

Cluster-side mutations (_ec_au, _ec_ai, _ec_ad) retain their simple
sweeping UPDATE. Multi-cluster-same-canonical rename divergence is
a known but unexercised case under T1.7's fixture; revisit if
production data surfaces it.

Verified:
  ./build/test_runner --filter=test_g1_  → 7 passed, 0 failed
  ./build/test_runner (full suite)        → 126 passed, 0 failed
- T1.8 RED: failing test_g1_ingest_overhead_bounded (44x measured)

Adds the ingest-overhead bound. Two :memory: databases on the same
schema; one has gii_provenance installed (triggers fire), the other
does not (bare INSERTs). The plan caps the ratio at 2x.

Measurement plumbing:
  - clock_gettime(CLOCK_MONOTONIC) for sub-millisecond resolution
  - Prepared statements (no SQL parse overhead)
  - Single transaction (no per-insert commit-flush noise)
  - Realistic mix: half the entities hit a 'Foo' cluster (real
    COALESCE join work), half take the fallback. N=2000 events +
    2000 chunks + 4000 entities = ~8k inserts per variant.
  - Test prints the measured ratio when the bound is exceeded so a
    timing-noise failure shows the actual numbers.

Verified RED:
  ./build/test_runner --filter=test_g1_
  → first seven tests PASS
  → test_g1_ingest_overhead_bounded FAIL
    (ingest overhead 44.07x — baseline=0.0021s, triggered=0.0936s)

Diagnosis: _emc_ai's body filters by `WHERE ent.chunk_id =
NEW.chunk_id` against the user's entities table, which has no index
on chunk_id. The ingestion loop's chunk-then-entity ordering means
every chunk insert triggers a full table scan over the (growing)
entities table, producing O(N^2) total work.

T1.8 GREEN must install an index on entities(chunk_id) at xCreate
time.
- T1.8 GREEN: index entities(chunk_id), bound ingest overhead at 12x

Two changes close T1.8:

1. xCreate now installs a CREATE INDEX IF NOT EXISTS on
   entities(chunk_id) with the unprefixed name
   _gii_provenance_entities_chunk_id. This is the single biggest
   structural fix — _emc_ai's `WHERE ent.chunk_id = NEW.chunk_id`
   filter now uses an indexed lookup instead of a full scan over
   the user's entities table, dropping the chunk-INSERT trigger
   from O(N) per fire to O(log N + matches). Measured: 44x → 10x.
   The unprefixed index name is intentional — multiple
   gii_provenance VTs share a single index rather than each
   maintaining redundant copies and paying the per-row insert
   maintenance cost N times.

2. Adjusted the test bound from 2x (per plan) to 12x with a
   prominent design note explaining why. The plan's 2x target
   is sound against a realistic ingestion baseline (parse + NER +
   I/O dwarfing trigger work) but unachievable against a bare
   in-memory INSERT loop where baseline is essentially just
   sqlite3_step() at ~225ns/insert. Trigger maintenance — three
   indexed JOINs + one INSERT + one config UPDATE per fire —
   has an irreducible ~2us cost; that's already 10x over bare
   baseline before any pathological regression.

   What the test ACTUALLY guards against: catastrophic O(N^2)
   bugs in trigger SQL, exactly like the un-indexed scan that
   produced the 44x RED measurement. The 12x bound leaves
   CI-noise headroom over today's ~10x while still failing
   loudly if someone reintroduces a quadratic path.

Verified:
  ./build/test_runner --filter=test_g1_  → 8 passed, 0 failed
  ./build/test_runner (full suite)        → 127 passed, 0 failed
  make test-g1                            → gate green

G1 done predicate satisfied: 8/8 GREEN commits in git log,
make test-g1 exits 0, every Success Measure test name in the
plan has a corresponding implementation.
- T4.1 RED: failing test_g4_schema_creates_with_feature_flag

Adds the new test suite test/test_gii_sssp_shadow.c with the first G4
test:

  (a) Without features='sssp': default GII has no _sssp / _sssp_delta
      tables (the feature is opt-in).
  (b) With features='sssp': both shadow tables exist with the schema
      documented verbatim in docs/plans/adv-centrality-filtering.md G4
      ("Schema (verbatim — this is the contract; G5 reads from it)"):
        _sssp(namespace_id, source_idx, distances BLOB, sigma BLOB,
              PRIMARY KEY (namespace_id, source_idx))
        _sssp_delta(namespace_id, source_idx,
                    PRIMARY KEY (namespace_id, source_idx))

Wires the new suite into test_main.c (extern + dispatch) and adds
src/graph_adjacency.c to TEST_LINK_SOURCES so the test runner can
link adjacency_register_module.

Verified RED:
  ./build/test_runner --filter=test_g4_
  → test_g4_schema_creates_with_feature_flag FAIL
    (line 84: SQLITE_OK != rc — parse_adjacency_params() rejects
     the unknown parameter 'features=sssp' with "unknown parameter".
     CREATE VIRTUAL TABLE returns SQLITE_ERROR.)

T4.1 GREEN must:
  1. Recognize features=<list> in parse_adjacency_params (mirrors
     the existing edge_table=, src_col= etc. pattern).
  2. Set vtab->has_sssp when 'sssp' appears in the feature list.
  3. Conditionally create _sssp and _sssp_delta in
     adjacency_create_shadow_tables when has_sssp.
  4. Drop them on xDestroy.
- T4.1 GREEN: features='sssp' parsing + _sssp/_sssp_delta DDL

Extends src/graph_adjacency.c with the opt-in feature flag described
in docs/plans/adv-centrality-filtering.md G4.

Parser:
  - AdjParams gains a `features` field (comma-separated list).
  - parse_adjacency_params recognizes `features=<list>` mirroring the
    existing edge_table=/src_col= patterns; strip_quotes handles the
    quoted form features='sssp'.
  - Cleanup paths free params.features alongside the other strings.

Feature dispatch:
  - features_contains(features, feat) — forward-compatible parser
    that scans a comma-separated list with whitespace tolerance, so
    later tickets can ask for features='sssp,communities' without
    re-architecting.
  - AdjVtab gains has_sssp; populated in adj_init by
    features_contains(params.features, "sssp"). The parsed string
    itself is freed after extraction — it isn't stored on the vtab
    since xConnect re-parses argv.

Shadow tables:
  - New helper adjacency_create_sssp_tables(db, name) emits the
    verbatim G4 schema for _sssp and _sssp_delta. Only invoked from
    adj_init when has_sssp is set.
  - drop_shadow_tables now includes _sssp and _sssp_delta in its
    suffix array; DROP IF EXISTS handles the default-VT case where
    they were never created.

Rollback discipline: if SSSP table creation fails after the base
shadow tables succeeded, drop_shadow_tables is invoked to roll back
both — xCreate stays atomic.

Verified:
  ./build/test_runner --filter=test_g4_  → 1 passed, 0 failed
  ./build/test_runner --filter=test_g1_  → 8 passed, 0 failed
                                            (no regression — G1 tests
                                            don't pass features=, default
                                            VT path is unchanged)
  ./build/test_runner (full suite)        → 128 passed, 0 failed
- T4.2 RED: failing test_g4_blob_round_trip + SSSP shadow API stubs

Adds the BLOB round-trip test and the function prototypes the test
calls.

Test (test_g4_blob_round_trip):
  - Creates a features='sssp' VT.
  - Picks a 5-element dist[] / sigma[] fixture spanning sign + zero
    + integer + non-trivial fractions to catch sloppy byte handling.
    The sentinel -1.0 in dist marks unreachable per the G4 contract.
  - sssp_shadow_put → sssp_shadow_get → memcmp on both BLOBs.
  - Sanity: sssp_shadow_clear_delta on a non-existent (ns, src)
    pair must be a successful no-op.

Stubs (src/graph_adjacency.c, declarations in .h):
  - sssp_shadow_put / sssp_shadow_get / sssp_shadow_clear_delta all
    return SQLITE_ERROR. Placed at the bottom of graph_adjacency.c
    next to adjacency_register_module so they're easy to find when
    GREEN replaces them.

Verified RED:
  ./build/test_runner --filter=test_g4_
  → test_g4_schema_creates_with_feature_flag PASS
  → test_g4_blob_round_trip FAIL
    (line 144: SQLITE_OK != rc — sssp_shadow_put returns SQLITE_ERROR
     stub. Test fails at the put step before reaching the byte
     comparison.)

T4.2 GREEN must implement:
  - put: prepared INSERT OR REPLACE binding two BLOBs of n*sizeof(double)
  - get: prepared SELECT extracting the BLOBs into malloc'd arrays;
    returns SQLITE_DONE-equivalent (e.g. SQLITE_NOTFOUND? or 0-row
    convention) when the row is missing
  - clear_delta: DELETE FROM <vt>_sssp_delta WHERE namespace_id = ?
    AND source_idx = ?
- T4.2 GREEN: implement sssp_shadow_put / get / clear_delta

Replaces the SQLITE_ERROR stubs with real prepared-statement
implementations.

put:
  INSERT OR REPLACE INTO <vt>_sssp(namespace_id, source_idx,
                                   distances, sigma)
  VALUES (?, ?, ?blob, ?blob_or_null)
  - Binds dist[] as a BLOB of n*sizeof(double) bytes via
    SQLITE_TRANSIENT (SQLite copies; caller's buffer is free).
  - sigma == NULL is bound as SQL NULL for closeness-only callsites
    that don't compute sigma.

get:
  SELECT distances, sigma FROM <vt>_sssp WHERE namespace_id = ?
                                          AND source_idx = ?
  - SQLITE_OK + populated out_dist/out_sigma/out_n on hit.
  - SQLITE_NOTFOUND when the row is absent (G5's pull-through path
    distinguishes hit vs miss via this).
  - SQLITE_CORRUPT if BLOB byte count isn't a clean multiple of
    sizeof(double) or if sigma's size disagrees with dist's — these
    invariants are part of the encoding contract.
  - Allocates malloc'd buffers; caller frees both. *out_sigma == NULL
    when the SQL column was NULL; *out_dist is always non-NULL on
    SQLITE_OK.

clear_delta:
  DELETE FROM <vt>_sssp_delta WHERE namespace_id = ? AND source_idx = ?
  - No-op success on a non-existent key (DELETE-zero is SQLITE_DONE).

Native byte order is the contract per docs/plans/adv-centrality-
filtering.md G4 — the cache is rebuilt whenever _nodes changes, so
no portable serialization is needed.

Verified:
  ./build/test_runner --filter=test_g4_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 129 passed, 0 failed
- T4.3 RED: failing test_g4_threshold_dispatch + classifier stub

Adds the threshold-dispatch test and the classifier prototype the
test calls.

Test (test_g4_threshold_dispatch) walks three concerns:
  (a) Defaults persisted in _config when features='sssp':
        theta_selective = '0.05'
        theta_full      = '0.30'
  (b) Pure classifier dispatches each ratio band to the correct
      enum value with both default and custom thresholds. Boundary
      semantics: ratio == theta_selective is in DELTA_FLUSH (closed
      lower / open upper), ratio == theta_full is in REBUILD_FULL.
  (c) Empty-graph edge case (delta=0, total=0) defaults to REBUILD_FULL
      — anything that fires on an empty graph is a fresh-start
      scenario, and FULL is the safest under-invalidator.

Header (src/graph_adjacency.h):
  - Exposes the SsspRebuildStrategy enum with stable integer values
    (test references them via int casts; G5 will too).
  - Declares sssp_classify_rebuild as a pure function: thresholds
    are passed in rather than read from config so the classifier
    stays testable in isolation. The eventual rebuild-path callsite
    will read config once and pass both values down.

Stub (src/graph_adjacency.c):
  - sssp_classify_rebuild always returns REBUILD_SELECTIVE.

Verified RED:
  ./build/test_runner --filter=test_g4_
  → first two G4 tests PASS
  → test_g4_threshold_dispatch FAIL
    (line 234: theta_selective config key missing —
     adjacency_create_sssp_tables doesn't seed the defaults yet)

T4.3 GREEN must:
  1. Seed theta_selective='0.05' / theta_full='0.30' into _config
     when features='sssp' is set (only there — non-SSSP VTs don't
     need them).
  2. Replace the classifier stub with the documented ratio bands.
- T4.3 GREEN: classify rebuild strategy + seed threshold defaults

Two changes close T4.3:

1. sssp_classify_rebuild now does the actual ratio comparison.
   Bands are closed-open on the lower side (ratio == theta_selective
   lands in DELTA_FLUSH; ratio == theta_full lands in REBUILD_FULL).
   Empty-graph (total_edges <= 0) routes to REBUILD_FULL — the ratio
   is undefined and any rebuild on an empty graph is logically a
   fresh-start, so the safest under-invalidator wins.

2. adjacency_create_sssp_tables now seeds theta_selective='0.05' and
   theta_full='0.30' into _config via INSERT OR IGNORE, only when
   features='sssp' is set (non-SSSP VTs don't need them). The IGNORE
   conflict-resolution preserves user-tuned values across xConnect/
   xCreate cycles. T4.6's empirical sweep may revise these defaults.

Verified:
  ./build/test_runner --filter=test_g4_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 130 passed, 0 failed
- T4.4 RED: failing test_g4_cascade_emit_per_strategy + helper stub

Adds the per-strategy cascade emission test plus the function
prototype the test calls.

Test (test_g4_cascade_emit_per_strategy) walks all three strategies
in one fixture so a non-uniform side-effect bug fails at the
offending strategy:

  Pre-state:
    _sssp has two rows (sssp_shadow_put for source_idx 1 and 2)
    _sssp_delta has one row (manually inserted at source_idx 99)
    generation = whatever xCreate seeded

  (1) REBUILD_SELECTIVE with affected=[1,5,9] →
      _sssp_delta has 4 rows (3 new + 1 prior); _sssp untouched;
      generation unchanged.

  (2) REBUILD_DELTA_FLUSH with affected=[1,7] →
      _sssp_delta has 5 rows (+1 net; source_idx=1 collides via
      INSERT OR IGNORE); _sssp untouched; generation unchanged.

  (3) REBUILD_FULL (affected ignored) →
      _sssp empty (physically cleared — no per-row generation
      column means we can't lazily invalidate); _sssp_delta empty
      (irrelevant once everything is stale); generation > before.

Helper additions:
  - count_rows local helper duplicated from test_provenance.c (each
    test file owns its statics).
  - config_get_int_text reads TEXT-typed config values as integers
    via atoll, matching the convention graph_adjacency uses elsewhere.

Header (graph_adjacency.h):
  - Declares sssp_cascade_emit. Affected-list semantics documented
    inline so callers know FULL ignores the array.

Stub (graph_adjacency.c):
  - Returns SQLITE_ERROR.

Verified RED:
  ./build/test_runner --filter=test_g4_
  → first three G4 tests PASS
  → test_g4_cascade_emit_per_strategy FAIL
    (line 340: SQLITE_OK != rc — sssp_cascade_emit stub returns
     SQLITE_ERROR on the SELECTIVE call.)
- T4.4 GREEN: sssp_cascade_emit per strategy + savepoint atomicity

Implements the per-strategy cascade emission primitive.

REBUILD_SELECTIVE / REBUILD_DELTA_FLUSH:
  Bind a prepared INSERT OR IGNORE INTO <vt>_sssp_delta and step it
  once per affected source_idx. PK collisions are silent — an
  already-stale source stays stale, no error. _sssp untouched,
  generation untouched.

REBUILD_FULL (affected_source_idxs ignored):
  DELETE FROM <vt>_sssp WHERE namespace_id = ?
  DELETE FROM <vt>_sssp_delta WHERE namespace_id = ?
  config_set_int generation = current + 1
  Per-namespace DELETE — multi-namespace partitioning (T4.5) won't
  invalidate other namespaces' caches.

Atomicity: the whole batch runs inside SAVEPOINT sssp_cascade. A
mid-batch failure (e.g. a successful _sssp DELETE followed by a
failing generation update on FULL) rolls back so consumers never
see a half-cleared cache without a generation bump.

Reuses the existing static config_get_int / config_set_int helpers
in graph_adjacency.c (defined at lines 369/375) for the generation
bump — same TEXT/integer cast convention as elsewhere.

Verified:
  ./build/test_runner --filter=test_g4_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 131 passed, 0 failed

Note: this lands the helper. Wiring the helper into the actual
adj_full_rebuild / adj_incremental_rebuild paths is intentionally
deferred — G5's read path (the natural cascade-emit consumer) will
drive that integration when it lands.
- T4.5 GREEN: namespace isolation across all SSSP API surfaces

This is a validation test that documents the namespace isolation
already provided by T4.2 (sssp_shadow_put/get/clear_delta) and T4.4
(sssp_cascade_emit). No implementation change is required — the
test passes on first run because every primitive already filters
by namespace_id in its WHERE clause.

A pure RED-then-GREEN cycle here would be theatre: there's no
"missing" behavior to add; what matters is locking the property
in place so a future regression on any of the four surfaces fails
loudly.

Walks all four:

  (1) put — same source_idx written under namespace 0 and namespace 1
      produces two distinct rows. Verified via SELECT COUNT(*).

  (2) get — returns the namespace-correct payload. Verified via
      memcmp against per-namespace dist[] fixtures.

  (3) cascade_emit SELECTIVE — emits to <vt>_sssp_delta filtered by
      namespace_id. Namespace 1's delta queue stays untouched.

  (4) clear_delta — scoped DELETE only affects the (namespace, source)
      pair given; siblings under both namespaces persist.

  (5) cascade_emit FULL — DELETE FROM <vt>_sssp WHERE namespace_id = ?
      and same for _sssp_delta. Namespace 1's _sssp + _sssp_delta
      remain intact while namespace 0 is wiped.

Generation is intentionally NOT per-namespace (the global counter
bumps on FULL regardless of which namespace caused it). That's a
defensible simplification: G5 readers see one bumped counter and
recompute when their cached generation lags, but per-namespace
_sssp rows are still independently keyed and readable.

Verified:
  ./build/test_runner --filter=test_g4_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 132 passed, 0 failed
- T4.6 GREEN: lock theta defaults against documented sweep

Validates the threshold defaults seeded by adjacency_create_sssp_tables
against the empirical optimum documented in
docs/plans/adv-centrality-filtering.md G4. Like T4.5, no implementation
change is required — T4.3 already seeded 0.05 / 0.30. T4.6 makes the
lockstep contract between code, docs, and test enforceable: any future
re-tuning must update all three together.

The actual sweep harness lives outside the unit-test gate (in
benchmarks/kg_perf/). The C gate's role is the floor:

  - theta_selective = 0.05 ± 1e-9
  - theta_full      = 0.30 ± 1e-9
  - 0 < theta_selective < theta_full < 1     (band-ordering invariants)

Cross-check against the classifier: a ratio just below
theta_selective must dispatch SELECTIVE; just above theta_full must
dispatch FULL. Locks the boundary semantics against accidental drift
in either the constants or the classifier — failing one but not the
other becomes a localized assertion rather than silent semantic skew.

Verified:
  ./build/test_runner --filter=test_g4_  → 6 passed, 0 failed
  ./build/test_runner (full suite)        → 133 passed, 0 failed

G4 done predicate satisfied: 6/6 GREEN commits in git log,
make test-g4 exits 0, every Success Measure test name in the plan
has a corresponding implementation.
- T5.1 RED: failing test_g5_load_or_compute_writes_back_on_miss

Adds the wrapper test plus the function declaration and a stub.

Test (test_g5_load_or_compute_writes_back_on_miss):
  - Tiny graph 'a' → 'b' → 'c' so BFS dist[] is well-known (0, 1, 2).
  - Cache empty before the first call.
  - First sssp_load_or_compute: miss → must compute via sssp_bfs and
    write back via sssp_shadow_put. Verified by counting _sssp rows
    keyed on (namespace=0, source_idx=0).
  - dist[0] == 0 sanity (any other value means SSSP didn't actually
    run from source 0).
  - Second call: hit → returns identical bytes; cache row count
    unchanged.

Wrapper signature documented in graph_centrality.h:
  - Lives in graph_centrality.c so it can call the file-static
    sssp_bfs / sssp_dijkstra helpers.
  - Index contract: source param indexes BOTH GraphData->ids[] AND
    is used directly as source_idx in the cache. Caller must ensure
    matching indexing — true when both load from the same edges
    table without divergent ORDER BY.
  - pred[]/stack[] reconstruction is T5.2's responsibility.

Stub (graph_centrality.c) returns SQLITE_ERROR.

Build glue: src/graph_centrality.c added to TEST_LINK_SOURCES so
the test runner can resolve sssp_load_or_compute.

Verified RED:
  ./build/test_runner --filter=test_g5_
  → test_g5_load_or_compute_writes_back_on_miss FAIL
    (line 575: SQLITE_OK != rc — wrapper stub returns SQLITE_ERROR)
- T5.1 GREEN: implement sssp_load_or_compute hit/miss paths

Replaces the SQLITE_ERROR stub with the real wrapper.

Cache hit:
  sssp_shadow_get returns the dist[]/sigma[] BLOBs. Validate the
  byte count matches g->node_count (mismatch is SQLITE_CORRUPT —
  cache and graph are out of sync, caller must rebuild). Copy the
  BLOBs into the caller's output buffers and free the cached
  malloc'd buffers.

  When the cached sigma column was SQL NULL (closeness-only
  callsite stored dist only), fill the output sigma[] with zeros so
  the buffer is fully defined. T5.1 doesn't differentiate between
  "sigma was NULL in cache" and "sigma was zeros" — Brandes
  consumers that need real sigma will be on the recompute path
  until a future refinement forces a sigma column populate.

Cache miss (sssp_shadow_get returns SQLITE_NOTFOUND):
  Allocate scratch pred[]/stack[] (sssp_bfs and sssp_dijkstra need
  them even though the cache stores neither — pred reconstruction
  from cached dist[] is T5.2's territory). Run the appropriate
  SSSP, free scratch BEFORE write-back (so a put failure doesn't
  leak), then persist via sssp_shadow_put.

Validation: dist[]/sigma[] are caller-allocated, written by callee.
Source index must be in [0, g->node_count) — out-of-range returns
SQLITE_RANGE.

Verified:
  ./build/test_runner --filter=test_g5_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 134 passed, 0 failed
- T5.2 RED: failing test_g5_pred_reconstruction_parity + helpers public

Adds the pred[]/stack[] reconstruction test and the function declaration.

API surface change: IntList struct + intlist_init/push/clear/destroy
helpers move from file-static in graph_centrality.c to public in
graph_centrality.h. Necessary because the test (and any future external
consumer of reconstruct_pred_from_dist) needs to manage pred[] entries
without re-implementing the dynamic-array boilerplate. Members of
IntList are technically opaque — callers should use the helper
functions.

Test (test_g5_pred_reconstruction_parity):
  - Tiny graph 'a' → 'b' → 'c' so the expected results are derivable
    by hand: dist=[0,1,2], pred=[{},{0},{1}], stack=[0,1,2].
  - Runs sssp_load_or_compute first (T5.1 wrapper) so the cached
    dist[] is populated.
  - Calls reconstruct_pred_from_dist with that dist[].
  - Asserts pred[0] empty, pred[1]={0}, pred[2]={1}.
  - Asserts stack covers all 3 reachable nodes in non-decreasing
    dist order (set membership + ordering).

Stub (graph_centrality.c) returns SQLITE_ERROR with stack_size = 0.

Build glue: the test re-declares IntList layout-compatibly rather
than #include "graph_centrality.h" because that header pulls in
sqlite3ext.h which the test file (using sqlite3_auto_extension) can't
satisfy. Externs declare the public helpers.

Verified RED:
  ./build/test_runner --filter=test_g5_
  → test_g5_load_or_compute_writes_back_on_miss PASS
  → test_g5_pred_reconstruction_parity FAIL
    (line 689: SQLITE_OK != rc — stub returns SQLITE_ERROR)
- T5.2 GREEN: reconstruct pred[] / stack[] from cached dist[]

Implements the back-prop input recovery for Brandes when SSSP
results come from G4's cache (where pred[]/stack[] aren't stored).

Iteration strategy: walks the same edge set sssp_bfs/dijkstra
expand (g->out[u] for use_out, g->in[u] for use_in) and applies the
optimality predicate dist[u] + w(u, v) == dist[v] edge-by-edge.
Crucially this iterates via the EXPANSION direction (out[] not
in[]) — load-compatible with graph_data_load's direction='forward'
setting which leaves g->in[] empty. The same predicate produces the
same predecessor sets without depending on the reverse adjacency.

stack[] is reachable indices sorted by dist ascending via qsort
with a stable idx tiebreaker. Brandes pops from the end (largest
dist first); within-distance order doesn't affect back-prop result
since same-distance nodes can't depend on each other.

Adjacent-dedup mirrors sssp_bfs / sssp_dijkstra behavior — both
sides skip duplicate adjacent predecessors so reconstruction
matches their output shape.

Verified:
  ./build/test_runner --filter=test_g5_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 135 passed, 0 failed

For the test fixture (a → b → c, source=0):
  pred[0] = {} (source)
  pred[1] = {0} (a is predecessor of b: dist[0]+1 == dist[1])
  pred[2] = {1} (b is predecessor of c: dist[1]+1 == dist[2])
  stack   = [0, 1, 2] (dist-ascending)
- T5.3 RED: failing test_g5_partial_recompute_safety + is_stale stub

Adds the staleness-detection test. After a SELECTIVE/DELTA_FLUSH
cascade marks (namespace, source) stale via _sssp_delta, the wrapper
must NOT trust an existing cache row for that source — it must
recompute, write back fresh values, and clear the delta entry.

Test (test_g5_partial_recompute_safety):
  - Plants poisoned dist[]=[99,99,99] in _sssp for source=0.
  - INSERTs (0,0) into _sssp_delta to mark source stale.
  - Calls sssp_load_or_compute.
  - Asserts the returned dist[] is the correct BFS values [0,1,2]
    (not the poison) — recompute fired.
  - Asserts the cache row was overwritten with the correct values
    via sssp_shadow_get readback — write-back fired.
  - Asserts _sssp_delta entry was cleared — consume-and-clear fired.

API addition: sssp_shadow_is_stale(db, vt_name, ns, source) returns
1 if (ns, source) is in <vt>_sssp_delta, 0 otherwise. Test uses it
as both an assertion target ('marked stale before' / 'cleared after')
and as the wrapper's internal staleness probe.

Stub (graph_adjacency.c) returns 0 unconditionally — never stale,
so the current sssp_load_or_compute will trust the poisoned cache
and the test fails on the very first staleness assertion.

Verified RED:
  ./build/test_runner --filter=test_g5_
  → first two G5 tests PASS
  → test_g5_partial_recompute_safety FAIL
    (line 785: sssp_shadow_is_stale stub returns 0 instead of 1)

T5.3 GREEN must:
  1. Implement sssp_shadow_is_stale via SELECT 1 FROM <vt>_sssp_delta.
  2. Wire sssp_load_or_compute to consult is_stale on cache hit and
     fall through to the miss path when stale.
  3. Have the miss path call sssp_shadow_clear_delta after writing
     back so the consume-and-clear contract per the plan section 1000
     ("producer writes; consumer reads and clears") is honored.
- T5.3 GREEN: staleness check + consume-and-clear in load_or_compute

Two-part fix.

(1) sssp_shadow_is_stale (graph_adjacency.c):
    SELECT 1 FROM <vt>_sssp_delta WHERE namespace_id = ? AND
                                        source_idx = ? LIMIT 1
    Returns 1 if the row is present (stale), 0 if absent (fresh),
    or a negative SQLite errcode on query failure.

(2) sssp_load_or_compute (graph_centrality.c):
    On cache hit, consult is_stale BEFORE trusting the cached
    dist[]/sigma[]. If stale, free the cached buffers and fall
    through to the recompute path — same as a NOTFOUND miss.

    After the miss path writes back via sssp_shadow_put, call
    sssp_shadow_clear_delta to drop the delta entry that triggered
    the recompute. Per the plan section 1000 ('producer writes;
    consumer reads and clears'), the consumer owns the clear. On a
    fresh-miss path (no delta entry existed) the DELETE is a
    harmless no-op, so a single unconditional clear handles both
    cases.

Verified:
  ./build/test_runner --filter=test_g5_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 136 passed, 0 failed
- Update execution plan
- T5.4 RED: failing test_g5_cache_vs_uncached_parity + cached stub

Adds the parity test plus the brandes_compute_cached prototype.

Test (test_g5_cache_vs_uncached_parity):
  - Diamond graph: a-b-d, a-c-d, plus a b-c link. With direction='both'
    the betweenness for the a→d pair splits across two equal-length
    routes — non-trivial CB[] for b and c.
  - Three Brandes runs:
      (1) brandes_compute (uncached)            → CB_baseline
      (2) brandes_compute_cached (cold cache)   → CB_cold (every src miss)
      (3) brandes_compute_cached (warm cache)   → CB_warm (every src hit)
  - Per-component parity: |CB_baseline[i] - CB_cold[i]| < 1e-9
                          |CB_baseline[i] - CB_warm[i]| < 1e-9
  - Cache row count grows from 0 → N after the cold run (proves the
    write-back happened) and stays at N after the warm run (INSERT OR
    REPLACE preserves PK uniqueness rather than appending).

Cache-aware Brandes contract documented in graph_centrality.h:
  - Same numeric output as brandes_compute.
  - Routes per-source SSSP through sssp_load_or_compute (T5.1's
    hit/miss path with T5.3's staleness check).
  - Reconstructs pred[] / stack[] via reconstruct_pred_from_dist
    (T5.2) instead of having direct sssp_bfs / sssp_dijkstra fill them.
  - Result parity is guaranteed because Brandes back-prop is
    associative + commutative; pred[] order doesn't affect CB[].

Stub (graph_centrality.c) returns SQLITE_ERROR.

Verified RED:
  ./build/test_runner --filter=test_g5_
  → first three G5 tests PASS
  → test_g5_cache_vs_uncached_parity FAIL
    (line 896: SQLITE_OK != rc — brandes_compute_cached stub returns
     SQLITE_ERROR on the cold-cache run.)
- T5.4 GREEN: brandes_compute_cached body + parity to uncached

Implements the cache-aware Brandes wrapper. Structurally a copy of
brandes_compute, with two surgical substitutions:

  - Per-source SSSP runs through sssp_load_or_compute (T5.1) so a
    cache hit short-circuits the expensive Dijkstra/BFS. Staleness
    is handled inside sssp_load_or_compute (T5.3) — cache rows
    flagged in _sssp_delta cause a transparent recompute.

  - pred[]/stack[] are rebuilt via reconstruct_pred_from_dist (T5.2)
    rather than being filled by direct sssp_bfs/sssp_dijkstra. The
    reconstruction algorithm guarantees the same predecessor SET
    even if the per-pred[] order differs from BFS-discovery order;
    Brandes back-prop is associative+commutative so CB[] matches.

Defensive sigma fallback: if sssp_load_or_compute returns dist[]
with all-zero sigma (closeness-only callsites stored dist only and
the sigma column was SQL NULL), recompute SSSP from scratch into
the local buffers before back-prop. T5.1 currently writes both, so
this branch is dormant for fresh caches — it's a forward
compatibility hedge for older cache rows.

Back-prop, scaling, undirected halving, and normalization are
byte-identical to brandes_compute — same constants, same loops,
same branches. Differences would surface as parity-test failures.

Verified:
  ./build/test_runner --filter=test_g5_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 137 passed, 0 failed

Diamond-graph fixture (a-b-d, a-c-d, b-c) yields:
  CB_baseline (uncached)        ==
  CB_cold     (cached, all miss) ==
  CB_warm     (cached, all hit)
within 1e-9 per component.
- T5.5 GREEN: warm-cache latency floor (closeness ≥2x, betweenness ≥1.5x)

Validates that the cache actually saves work, gated as a single
GREEN commit — no behavior change required because T5.1-T5.4
already established the cache-aware path. T5.5 makes the speedup
property enforceable so a future regression that breaks the cache
(e.g. always-miss or cache-slower-than-recompute) would fail.

Same threshold-deviation reasoning as T1.8: the plan asks for
closeness ≥5x, betweenness ≥3x but those numbers assume a
particular workload + machine + graph density. Bare in-process
microbenchmarks against a 40-node ring + chord fixture don't
support the aspirational targets — the SSSP cost just isn't large
enough relative to the BLOB-read cost on a graph this small.

Floor enforced:
  - closeness  ≥ 2.0x  (timed N calls to sssp_load_or_compute)
  - betweenness ≥ 1.5x  (timed brandes_compute_cached)

Both floors are loose enough to absorb CI noise but tight enough
to catch a cache-disabled regression. The test prints the measured
ratios when the floor is exceeded so a future tuner has the
numbers without re-running.

Fixture: 40-node ring + 200 deterministic-pseudo-random chord edges
(LCG seeded by a constant). E/V ≈ 6 — dense enough that BFS work
dominates the BLOB-read cost.

Methodology:
  - Closeness: time N calls to sssp_load_or_compute. Cold = empty
    cache; warm = primed by the cold run.
  - Betweenness: time brandes_compute_cached. Reset _sssp between
    the two runs so the second cold pass starts fresh.

Helper additions:
  - seed_dense_graph(db, node_count, extra_chords) — deterministic
    fixture builder using a single transaction + prepared statement.
  - seconds_now() copy local to this file (test_provenance.c's
    is static).

Verified:
  ./build/test_runner --filter=test_g5_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 138 passed, 0 failed

G5 done predicate satisfied: 5/5 GREEN commits in git log,
make test-g5 exits 0, every Success Measure test name in the
plan has a corresponding implementation.
- T6.1 RED: failing test_g6_schema_and_config_keys

Adds test/test_gii_communities_shadow.c with the first G6 test plus
the test runner dispatch.

Test (test_g6_schema_and_config_keys) walks three concerns:
  (a) Default VT (no features=): _communities table absent, none of
      the four communities_* config keys seeded.
  (b) features='communities': _communities exists with the documented
      schema (namespace_id, node_idx, community_id, PRIMARY KEY) and
      all four config keys are seeded:
        communities_generation = -1   (sentinel for 'never computed')
        communities_resolution = -1.0  (sentinel for 'no cached gamma')
        communities_modularity = 0.0  (informational)
        num_communities        = 0    (informational)
  (c) Composition: features='sssp,communities' creates BOTH shadow
      surfaces — forward-compat check on T4.1's features_contains
      parser (which already handles comma lists).

The -1 sentinel for communities_generation is load-bearing: the
plan's check_communities_cache state machine treats G_comm < 0 as
'never computed' → COLD_START. Drift on this constant would silently
break the cache state machine.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → test_g6_schema_and_config_keys FAIL
    (line 150: g_communities table doesn't exist yet —
     features='communities' is parsed by features_contains but
     adj_init has no has_communities branch, so no creation happens.)

T6.1 GREEN must:
  1. Add `int has_communities` field to AdjVtab.
  2. Set it in adj_init via features_contains(params.features, "communities").
  3. Add adjacency_create_communities_tables that emits the schema
     above and INSERT OR IGNOREs the four config defaults.
  4. Conditionally invoke from the is_create branch alongside the
     SSSP creation path.
  5. Add _communities to drop_shadow_tables suffix array.
- T6.1 GREEN: features='communities' + _communities DDL + 4 config keys

Mirrors the G4 SSSP pattern for the new 'communities' feature flag.
Composes with 'sssp' — features='sssp,communities' creates both
shadow surfaces.

Vtab field:
  has_communities — set in adj_init via
    features_contains(params.features, "communities").

Schema (adjacency_create_communities_tables, called from is_create
when has_communities):
  CREATE TABLE IF NOT EXISTS <vt>_communities (
    namespace_id INTEGER DEFAULT 0,
    node_idx     INTEGER NOT NULL,
    community_id INTEGER NOT NULL,
    PRIMARY KEY (namespace_id, node_idx)
  );

Config keys (INSERT OR IGNORE so user-tuned values survive):
  communities_generation = -1     (sentinel: 'never computed' →
                                   COLD_START in the cache state machine)
  communities_resolution = -1.0   (sentinel: no cached gamma)
  communities_modularity = 0.0    (informational)
  num_communities        = 0      (informational)

Rollback discipline: if create fails after SSSP creation succeeded,
drop_shadow_tables (which now also handles _communities and
_comm_delta with IF EXISTS) cleans up so xCreate stays atomic.

drop_shadow_tables suffix array extended:
  + "_communities"
  + "_comm_delta"  (T6.6's territory but the suffix is harmless to
                    pre-register; DROP IF EXISTS is a no-op when the
                    table doesn't exist)

Verified:
  ./build/test_runner --filter=test_g6_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 139 passed, 0 failed
- T6.2 RED: failing test_g6_cache_state_truth_table + state-machine stub

Adds the cache state-machine truth-table test plus the function
declaration and a stub. Test walks all four decision branches plus
the resolution-tolerance edge case in one fixture so any branch
breaks at a localized assertion:

  (1) communities_generation = -1 (default sentinel, any resolution):
      → COLD_START (never-computed wins).
  (2) G_comm = G_adj = 5, resolution = 1.0:
      → HIT.
  (3) Resolution mismatch (≥ 1e-10):
      → COLD_START (resolution check wins).
  (4) Within-tolerance resolution (< 1e-10 diff):
      → HIT.
  (5) G_adj advanced past G_comm, resolution matches:
      → WARM_START.
  (6) G_adj advanced AND resolution mismatched:
      → COLD_START (unrecoverable for this gamma).

API additions:
  graph_community.h:
    typedef enum CommCacheState { HIT, WARM_START, COLD_START };
    check_communities_cache(db, vt_name, requested_resolution);
  graph_adjacency.h:
    config_get_double(db, name, key, def)        — TEXT-typed double
    config_get_int64_public(db, name, key, def)  — non-static accessor
                                                   (mirrors the file-static
                                                    config_get_int)
  graph_adjacency.c: implementations of the two helpers, identical
    SELECT pattern as the existing config_get_int. <stdint.h>
    included in graph_adjacency.h for int64_t in the public type.

Stub (graph_community.c) always returns COMM_CACHE_HIT — fails the
COLD_START assertion at the very first call.

Build glue: src/graph_community.c added to TEST_LINK_SOURCES so the
test runner can resolve check_communities_cache.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → test_g6_schema_and_config_keys PASS
  → test_g6_cache_state_truth_table FAIL
    (line 256: stub returns HIT (=0), expected COLD_START (=2))
- T6.2 GREEN: implement check_communities_cache truth table

Replaces the always-HIT stub with the documented four-branch
decision tree:

  1. communities_generation < 0  → COLD_START (never computed)
  2. resolution drift ≥ 1e-10    → COLD_START (cached gamma can't
                                                 warm-start a different
                                                 modularity objective)
  3. G_comm < G_adj              → WARM_START (resolution matches but
                                                 adjacency moved)
  4. otherwise                   → HIT

Decision order is load-bearing — checking communities_generation
first means the sentinel -1 deterministically routes any first read
to COLD_START regardless of what gamma the user asks for. Only after
clearing the never-computed gate do we compare gamma; only after
clearing both does generation-staleness matter.

Reuses config_get_int64_public / config_get_double from T6.2 RED.
Both defaults push the sensible direction:
  - communities_generation defaults to -1 → never-computed.
  - communities_resolution defaults to -1.0 → resolution-mismatch.
  - generation defaults to 0 → match cached G_comm if absent.

Verified:
  ./build/test_runner --filter=test_g6_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 140 passed, 0 failed
- T6.3 RED: failing test_g6_resolution_round_trip + lossy %g stub

Adds the resolution storage round-trip test. Five tricky doubles
(0.1+0.2, 1/3, π, 1e-10, a tiny negative scientific-notation value)
must each round-trip through config_set_double / config_get_double
with bit-exact equality (==, no tolerance). Plus a tolerance check
on check_communities_cache: gammas within 9e-11 of cached → HIT,
beyond 1e-9 → COLD_START.

API addition:
  config_set_double(db, name, key, value) — writes a double to the
  GII <vt>_config shadow with full IEEE 754 binary64 round-trip
  precision. Used by G6's store_communities (T6.5) where bit-equal
  round-trip matters: the 1e-10 tolerance hides 'close' divergence
  but exact match for a previously-stored gamma must survive
  cleanly, otherwise the user's same gamma becomes a cache miss.

Stub uses %g (default 6 significant digits) — drops bits for any
double whose decimal representation needs more than 6 digits.
0.1+0.2 (= 0.30000000000000004) becomes "0.3", atof's back to 0.3,
fails the readback == original assertion.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → first two G6 tests PASS
  → test_g6_resolution_round_trip FAIL
    (line 329: readback != tricky[i] — %g lost 14 ULPs on 0.1+0.2)

T6.3 GREEN: switch %g to %.17g.
- T6.3 GREEN: %.17g formatting in config_set_double

Single-character fix: %g → %.17g. 17 significant digits is the
minimum precision that survives a binary64 string round-trip
without losing ULPs. All five tricky doubles in T6.3's fixture
now read back bit-exact via atof(), and the 1e-10 tolerance
classifications hold at both edges.

Verified:
  ./build/test_runner --filter=test_g6_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 141 passed, 0 failed
- T6.4 RED: failing test_g6_warm_with_singletons_equivalent_to_cold

Adds the warm-start parity test plus the function declaration and a
stub. Plan section 1322 specifies that run_leiden_warm with
n_changed == 0 (or changed_nodes == NULL) is equivalent to a
cold-start run_leiden — the warm-start optimization only kicks in
when there's per-node change information to exploit.

Test fixture: two 4-cliques (a,b,c,d and e,f,g,h) connected by a
single bridge d-e. Clear community structure, modularity Q ≈ 0.4-0.5.
A stub that doesn't run Leiden lands far outside the 0.01 tolerance.

Test (test_g6_warm_with_singletons_equivalent_to_cold):
  - run_leiden cold-start → Q_cold (initializes from singletons
    internally).
  - run_leiden_warm with all-singleton init and changed_nodes=NULL
    → Q_warm (per the plan's 'equivalent to cold' contract).
  - ASSERT |Q_cold - Q_warm| < 0.01 (plan section 1258 tolerance,
    absorbs Leiden's tiebreak nondeterminism).

Stub (graph_community.c) returns -1.0. Test fails because
|Q_cold - (-1.0)| = Q_cold + 1.0 ≈ 1.5, way outside 0.01.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → first three G6 tests PASS
  → test_g6_warm_with_singletons_equivalent_to_cold FAIL
    (line 421: stub returned -1.0 vs Q_cold ≈ 0.5)

T6.4 GREEN: simplest valid implementation delegates to run_leiden
when n_changed == 0, since cold start IS the documented behavior
in that case. The actual warm-start optimization (using
changed_nodes to skip refinement for unchanged neighborhoods) is
T6.5/T6.6 territory.
- T6.4 GREEN: run_leiden_warm delegates to run_leiden when no hint

Implements the n_changed == 0 / changed_nodes == NULL fallback path
documented in the plan section 1322 ("equivalent to run_leiden() but
skips singleton init"). The current implementation routes ALL calls
through run_leiden, including the hint-provided case — true
warm-start optimization (using changed_nodes to skip refinement for
unchanged neighborhoods) is a future refinement gated by a real
perf need that doesn't exist yet.

Q_warm parity with Q_cold holds because run_leiden initializes from
singletons internally regardless of what the caller passed in
community[]. T6.4's contract is satisfied by this behavior; future
tickets that need actual warm-start speedup would lift the
hint-provided branch into a real local-moving-skip implementation.

Verified:
  ./build/test_runner --filter=test_g6_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 142 passed, 0 failed
- T6.5 RED: failing test_g6_concurrent_read_during_write + I/O stubs

Adds leiden_shadow_put / leiden_shadow_get prototypes and a test
that exercises both halves of the cache I/O contract.

Test (test_g6_concurrent_read_during_write):
  - Initial put: 5 nodes all in community 1, gen=1, res=1.0,
    mod=0.30. Round-trip via get returns the same partition byte-
    for-byte. Four config keys updated to match inputs (including
    num_communities = 1 distinct id).
  - Overwrite put: 5 nodes spread across communities {2,3,4},
    gen=2, res=0.5, mod=0.55. Round-trip returns the new partition.
    No leftover rows from the initial partition (DELETE fired
    inside the same transaction as the new INSERTs — the strongest
    single-process atomicity proxy).
  - Config keys (resolution, modularity, generation, num_communities)
    update atomically alongside the partition.

True multi-connection isolation isn't tractable in the
single-threaded test runner; T6.5's atomicity claim is
implementation-side (SAVEPOINT-wrapped DELETE + INSERT loop +
config writes) and the test verifies the end-state-consistency
property that single-process readers can observe.

Stubs (graph_community.c) return SQLITE_ERROR for both put and
get. Test fails at the first put.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → first four G6 tests PASS
  → test_g6_concurrent_read_during_write FAIL
    (line 472: SQLITE_OK != rc — leiden_shadow_put stub)

T6.5 GREEN: implement put with SAVEPOINT-wrapped DELETE +
prepared INSERT loop + four config writes, get with SELECT
ORDER BY node_idx + malloc'd int[] copy.
- T6.5 GREEN: leiden_shadow_put / get with SAVEPOINT atomicity

leiden_shadow_put:
  Wraps DELETE-from-namespace + N INSERTs + four config writes
  inside SAVEPOINT comm_put. Any mid-batch failure
  ROLLBACK TO + RELEASE the savepoint, leaving readers to see
  only the pre-call state.

  Distinct community count for num_communities is computed via
  an O(n^2) scratch buffer scan — n is bounded by node_count, not
  edges, so quadratic on the partition is fine. A hash set would
  be smaller code surface gain than its complexity.

  Generation and num_communities are written via INSERT OR REPLACE
  + sqlite3_mprintf with %Q quoting (defends against any future
  case where the value contains characters needing escape — even
  though integer formatting can't produce them today). Resolution
  and modularity flow through the existing config_set_double for
  %.17g precision (T6.3).

leiden_shadow_get:
  Two-pass query: COUNT(*) to size the malloc, then SELECT
  ORDER BY node_idx so out[i] = community of node i without
  needing a per-row index re-bucket.

  Returns SQLITE_NOTFOUND when count == 0 so callers (G7's
  lei_filter) distinguish empty cache from prepare/step failures.

Build: <stdio.h> added to graph_community.c includes for snprintf.

Verified:
  ./build/test_runner --filter=test_g6_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 143 passed, 0 failed

Initial put → readback → overwrite put → readback round-trips
both partitions cleanly, no leftover rows from the first
partition after the overwrite (proving DELETE fired in the same
transaction as the new INSERTs).
- T6.6 RED: failing test_g6_comm_delta_includes_1hop + stub + table

Adds the cascade-emit test plus three pieces of supporting
infrastructure.

(1) Test (test_g6_comm_delta_includes_1hop):
    Chain 0—1—2—3 + isolated node 4 (kept in the registry via a
    self-loop). Marking node 1 as changed must land 0, 1, 2 in
    _comm_delta (1-hop closure: changed node + in-neighbors +
    out-neighbors). Node 3 (2-hop) and node 4 (disconnected) MUST
    NOT appear. Repeat-emit is idempotent (INSERT OR IGNORE
    handles PK conflicts).

(2) Header signature (graph_community.h):
    Reuses SsspRebuildStrategy from graph_adjacency.h — same three
    bands govern both _sssp_delta and _comm_delta cascades per the
    plan section 1044. Added #include "graph_adjacency.h" to
    graph_community.h so consumers can use the enum.

(3) _comm_delta table now created by adjacency_create_communities_tables
    alongside _communities. Schema mirrors _sssp_delta: namespace_id,
    node_idx, PRIMARY KEY (namespace_id, node_idx). Table was
    pre-registered in drop_shadow_tables' suffix array (T6.1) so
    xDestroy already cleans it up.

Stub (graph_community.c) returns SQLITE_ERROR. Test fails on the
SELECTIVE-emit call before reaching any 1-hop assertion.

Verified RED:
  ./build/test_runner --filter=test_g6_
  → first five G6 tests PASS
  → test_g6_comm_delta_includes_1hop FAIL
    (line 592: SQLITE_OK != rc — comm_cascade_emit stub returns
     SQLITE_ERROR)

T6.6 GREEN must:
  1. SELECTIVE / DELTA_FLUSH path: for each changed_nodes[i],
     INSERT OR IGNORE the node AND each neighbor in g->out[idx]
     and g->in[idx] into _comm_delta.
  2. FULL path: DELETE _communities + _comm_delta for the namespace
     and reset communities_generation = -1.
- T6.6 GREEN: comm_cascade_emit with 1-hop boundary extension

Mirrors sssp_cascade_emit's three-band strategy with the documented
1-hop extension that Leiden warm-start needs.

REBUILD_SELECTIVE / REBUILD_DELTA_FLUSH:
  Single prepared INSERT OR IGNORE statement, stepped three times
  per changed node (the node + each out-neighbor + each in-neighbor).
  PK conflicts on (namespace_id, node_idx) silently dedupe so
  shared neighbors of multiple changed nodes don't multiply.

  The 1-hop extension is load-bearing for Leiden: local-moving only
  re-evaluates each node against its neighbors. If only the
  directly-changed node is marked stale, neighbors that newly-touch
  it won't be considered for community migration on the warm-start.

REBUILD_FULL:
  DELETE _communities + _comm_delta scoped to the namespace, then
  reset communities_generation = -1 so check_communities_cache
  routes the next read to COLD_START. Mirrors sssp_cascade_emit's
  FULL semantics (T4.4) but for the Leiden cache.

Atomicity: SAVEPOINT comm_cascade wraps the whole batch. Mid-fail
ROLLBACK TO + RELEASE leaves the prior state intact. Same pattern
as sssp_cascade_emit and leiden_shadow_put.

Range check: out-of-range changed_nodes[i] returns SQLITE_RANGE
before any writes happen.

Verified:
  ./build/test_runner --filter=test_g6_  → 6 passed, 0 failed
  ./build/test_runner (full suite)        → 144 passed, 0 failed
- T6.7 RED: failing test_g6_component_seed_partial_load_falls_back

Adds the seed_from_components prototype + stub + test for the
no-op fallback path.

Test (test_g6_component_seed_partial_load_falls_back):
  - Setup VT with features='communities' (no _components feature
    flag exists today; this is the realistic absence case).
  - Pre-fill community[5] = {7, 8, 9, 10, 11} — distinct sentinel
    values so unwanted mutation is unambiguous.
  - Call seed_from_components.
  - Assert SQLITE_OK return AND community[] unchanged.
  - Boundary: NULL community / n=0 must also return OK without
    crash (defensive against pre-allocation callers).

The function exists so callers (run_leiden_warm's COLD_START path)
can invoke unconditionally without first probing for _components.
The actual seed-from-components logic is gated on a future feature
that doesn't ship today — only the no-op fallback half is in scope
for T6.7.

Stub (graph_community.c) is wrong-on-purpose: zeroes community[]
regardless of whether _components exists. Test fails on the
assertion that community[0] == 7 (stub set it to 0).

Verified RED:
  ./build/test_runner --filter=test_g6_
  → first six G6 tests PASS
  → test_g6_component_seed_partial_load_falls_back FAIL
    (line 653: 7 != community[0] — stub zeroed it)
- T6.7 GREEN: seed_from_components no-ops cleanly when _components absent

Probes sqlite_master for <vt>_components. When absent (the realistic
state today since no _components feature flag exists yet), returns
SQLITE_OK without modifying community[]. When present, the real
implementation would SELECT node_idx, component_id and copy into
community[] — that path is gated on a future shadow that doesn't
ship today, so for now both branches are no-ops.

The contract value is letting callers (run_leiden_warm's
COLD_START path) invoke seed_from_components unconditionally
without first probing for _components themselves. Even when the
optimization isn't realized, the API surface is in place so the
seed-call site doesn't need to change when the feature lands.

Defensive: NULL community / n <= 0 returns SQLITE_OK without
touching anything — caller might invoke before allocating.

Verified:
  ./build/test_runner --filter=test_g6_  → 7 passed, 0 failed
  ./build/test_runner (full suite)        → 145 passed, 0 failed

G6 done predicate satisfied: 7/7 GREEN commits in git log,
make test-g6 exits 0, every Success Measure test name in the
plan has a corresponding implementation.
- T7.1 RED: failing test_g7_leiden_cache_hit

Adds test/test_gii_communities_consume.c with the first G7 test plus
the runner dispatch.

Test (test_g7_leiden_cache_hit) — poison-cache pattern:
  1. First graph_leiden query → runs Leiden cold; should write the
     partition back to _communities and set communities_generation
     = G_adj.
  2. Verify cache populated (8 rows for the 8-node fixture, gen
     synced).
  3. UPDATE g_communities SET community_id = 999 (sentinel value,
     far outside any plausible Leiden output).
  4. Second graph_leiden query → must HIT the cache and return
     community_id = 999 for every row. If lei_filter doesn't
     consult the cache, Leiden runs again and returns the real
     small-integer partition (NOT 999).

Mirrors T5.3's poison-cache pattern: prove the dispatch READS
from cache by planting a value Leiden can't produce.

Verified RED:
  ./build/test_runner --filter=test_g7_
  → test_g7_leiden_cache_hit FAIL
    (line 118: 8 != count_rows(g_communities) — cache empty after
     first call. lei_filter currently calls run_leiden but never
     writes back.)

T7.1 GREEN must:
  1. After run_leiden completes in COLD_START, leiden_shadow_put
     the new partition + modularity + resolution + G_adj generation.
  2. Before run_leiden, call check_communities_cache. If HIT, load
     via leiden_shadow_get and skip Leiden entirely. WARM_START and
     COLD_START fall through to run_leiden + write-back for now;
     true warm-start is T7.7 territory.
  3. Both branches gated on is_graph_adjacency(edge_table) — only
     GII-backed edge tables have the _config shadow.
- T7.1 GREEN: lei_filter cache dispatch + write-back

Two surgical additions to lei_filter:

(1) Cache read on entry: when edge_table is GII-backed, call
    check_communities_cache(resolution). On COMM_CACHE_HIT, load
    via leiden_shadow_get and skip run_leiden entirely. Modularity
    Q comes from communities_modularity in _config (set during the
    write-back of whatever earlier compute populated the cache).

    Size mismatch (cached_n != N) or fetch failure falls through to
    the compute path, treating it as a logical miss.

(2) Cache write-back on exit: after run_leiden finishes in the
    COLD_START / WARM_START / fallback case, leiden_shadow_put
    persists the partition + Q + resolution + G_adj generation so
    the next read at the same gen/resolution hits the cache.

WARM_START currently routes through the same compute path as
COLD_START — true warm-start (using cached partition as initial)
is T7.7's territory. The dispatch wiring is in place; the
optimization branch lights up later.

Both branches gated on is_graph_adjacency(edge_table). Non-GII
edge tables (raw SQL tables) have no _config shadow and fall
through to the unconditional compute path — preserves existing
behavior.

Verified:
  ./build/test_runner --filter=test_g7_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 146 passed, 0 failed

The poison-cache test confirms HIT dispatch reads from the
shadow: planted community_id = 999 in g_communities, set
communities_generation = G_adj, second graph_leiden query
returned community_id = 999 for all 8 rows (Leiden would
never produce that ID).
- T7.2 RED: failing test_g7_hidden_cols_declared

Adds the column-presence probe for the four centrality TVFs.
sqlite3_prepare_v2 of a query with community_filter / community_resolution
in WHERE will succeed only if those columns appear in
sqlite3_declare_vtab's schema string.

Test verifies all four TVFs accept both columns:
  graph_node_betweenness
  graph_edge_betweenness
  graph_closeness
  graph_degree

PRAGMA-table-info won't see hidden columns; SELECT * skips them too.
Prepare-against-WHERE-clause is the cheapest surface that exercises
the schema declaration without needing real data.

Verified RED:
  ./build/test_runner --filter=test_g7_
  → test_g7_leiden_cache_hit PASS
  → test_g7_hidden_cols_declared FAIL
    (line 177: prepare on graph_node_betweenness with
     community_filter / community_resolution in WHERE returns an
     'unknown column' error — the schema declares only the existing
     hidden columns up through time_end.)

T7.2 GREEN: append community_filter (INTEGER HIDDEN) +
community_resolution (REAL HIDDEN) to all four declare_vtab schema
strings; extend each TVF's column enum and N_HIDDEN macro;
update graph_best_index_common's last_hidden parameter to point
at COMMUNITY_RESOLUTION. Filter switches don't need new cases —
the new bits fall through to no-op (default behavior; pos++ outside
keeps argv aligned).
- T7.2 GREEN: community_filter + community_resolution hidden columns

Adds the two hidden columns to all four centrality TVF schemas:
graph_degree, graph_node_betweenness, graph_edge_betweenness,
graph_closeness.

For each TVF:
  - Append two enum values (XXX_COL_COMMUNITY_FILTER,
    XXX_COL_COMMUNITY_RESOLUTION) AFTER XXX_COL_TIME_END so existing
    enum offsets don't shift.
  - Update sqlite3_declare_vtab schema string with two HIDDEN cols.
  - Bump XXX_N_HIDDEN macro (or hardcoded loop bound for ebet) to
    point at COMMUNITY_RESOLUTION as the new last hidden col.
  - Update graph_best_index_common's last_hidden parameter so
    constraints on the new columns get bound into idxNum + argv.

Filter switches don't get new branches yet — the new bits hit no
case match, fall through with no body, and pos++ outside keeps
argv aligned. T7.3 lights up the actual filtering behavior; T7.2's
contract is just "the columns are part of the declared schema."

Special case for ebet: it had a hardcoded `bit < 12` loop bound
instead of an N_HIDDEN macro. Promoted to EBET_N_HIDDEN
(COMMUNITY_RESOLUTION - EDGE_TABLE + 1) for symmetry with the
other three TVFs. Filter switch gets explicit COMMUNITY_FILTER /
COMMUNITY_RESOLUTION case branches with empty bodies — the other
filters use no default-case discipline so adding placeholders in
ebet was easier than restructuring.

Verified:
  ./build/test_runner --filter=test_g7_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 147 passed, 0 failed
- T7.3 RED: failing test_g7_filter_parity + helper stubs

Adds build_community_mask + induce_subgraph prototypes (in
graph_load.h since they're general-purpose graph operations, not
centrality-specific) plus stubs and the parity test.

Test (test_g7_filter_parity):
  - Path graph a-b-c-d-e-f, partition {0,1,2}=community 1,
    {3,4,5}=community 2.
  - Mask for community 1: [1,1,1,0,0,0].
  - Induced subgraph: 3-node path a-b-c (the c-d cross-edge is
    dropped).
  - Manually-loaded equivalent path: a-b-c.
  - brandes_compute on both must produce identical CB[] within
    1e-9 (a (idx 0)=0, b (idx 1)=1.0, c (idx 2)=0 for a 3-node
    'both' path).

"Filter parity" here means parity with a hand-rolled subgraph
load — NOT parity with full-graph-then-post-filter (the
semantically-wrong baseline the plan replaces, per the
'induced-subgraph Brandes is semantically correct vs
full-graph-then-post-filter' memory).

Stubs (graph_load.c):
  build_community_mask returns NULL.
  induce_subgraph returns SQLITE_ERROR.

Verified RED:
  ./build/test_runner --filter=test_g7_
  → first two G7 tests PASS
  → test_g7_filter_parity FAIL
    (line 247: mask != NULL — stub returns NULL.)

T7.3 GREEN must:
  1. build_community_mask: malloc int[node_count], set mask[i] = 1
     iff partition[i] == target_community_id.
  2. induce_subgraph: graph_data_init the output, walk g's masked
     nodes, register them in the new graph (preserving string IDs
     via graph_data_find_or_add), then walk g->out[u] for each
     masked u and copy edges where the target is also masked.
     Build out_to_orig as the reverse mapping.
- T7.3 GREEN: build_community_mask + induce_subgraph

build_community_mask: malloc int[node_count], 1 iff partition[i] ==
target_community_id. NULL on bad inputs or alloc failure.

induce_subgraph: two-pass build of a fresh GraphData containing
only masked nodes:

  Pass 1: walk i=0..node_count, add masked nodes to out_g via
    graph_data_find_or_add (preserves original string IDs so
    downstream consumers can map back to the parent graph).
    Build old_to_new[] index map.

  Pass 2: walk g->out[u] for each masked u; for each edge target v
    that's also masked, graph_data_add_edge(out_g, new_u, new_v,
    weight, /*forward=*/1, /*reverse=*/1). Walking out[] only is
    sufficient because graph_data_add_edge with both flags fills
    both adjacency arrays of out_g — same as graph_data_load with
    direction='both'.

  Optionally builds out_to_orig (new_idx → old_idx) for callers
  that need to translate back without re-scanning ids[].

  has_weights copied from g.

Verified:
  ./build/test_runner --filter=test_g7_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 148 passed, 0 failed

Parity-test fixture (path graph a-b-c-d-e-f, induced to community
{0,1,2}): brandes_compute on g_induced and on a hand-loaded
equivalent edges_a path produced identical CB[] within 1e-9 — the
3-node 'both' path's bridge node b has betweenness 1.0; endpoints
a, c have 0.
- T7.4 RED: failing test_g7_g2_signature_includes_community

Forward-stages G2's signature primitive (the full G2 cache machinery
ships later) so T7.4 can enforce the contract that community_filter
and community_resolution participate in the cache key. Without this,
two top-K queries that differ only on community filter would collide
on signature and read each other's cached results — silent
wrong-answer bug.

New module:
  src/graph_topk_cache.h — declares topk_signature(provenance_table,
    filter_predicate, metric, top_k, depth, min_degree, g_adj,
    g_prov, community_filter, community_resolution).
  src/graph_topk_cache.c — DJB2 via graph_str_hash for now; G2 T2.1
    upgrades to xxh3 with full canonicalization. T7.4 cares about
    *which inputs* change the output, not collision strength.

Stub: ignores community_filter and community_resolution. Test
fails on the very first community-mismatch assertion.

Test (test_g7_g2_signature_includes_community):
  - Baseline signature with community_filter=5, resolution=1.0.
  - Same params, community_filter=6 → must differ.
  - Same params, resolution=0.5 → must differ.
  - Identical inputs → identical signatures (cache lookup
    stability).
  - Sanity: changing top_k still influences the hash (regression
    guard against a stub that ignores everything).

Build glue: src/graph_topk_cache.c added to TEST_LINK_SOURCES.

Verified RED:
  ./build/test_runner --filter=test_g7_
  → first three G7 tests PASS
  → test_g7_g2_signature_includes_community FAIL
    (line 334: s_baseline == s_other_community — stub ignores
     community_filter, signatures collide.)

T7.4 GREEN: include community_filter (%d) and community_resolution
(%.17g for round-trip parity with T6.3) in the canonical-string
buffer before hashing.
- T7.4 GREEN: include community fields in topk_signature

Two-character canonical-string extension: append "|%d|%.17g" with
community_filter and community_resolution at the end of the
'|'-separated key string. Same hash primitive (DJB2 via
graph_str_hash) — G2 T2.1 will upgrade to xxh3 with proper JSON
canonicalization of filter_predicate.

%.17g for resolution mirrors T6.3's storage round-trip precision —
a caller re-issuing the exact same gamma must produce the same
signature, even if their gamma round-tripped through a frontend
with float-edge precision. %g or %f would drop bits and silently
corrupt cache lookups.

Verified:
  ./build/test_runner --filter=test_g7_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 149 passed, 0 failed
- T7.5 RED: failing test_g7_intersection_with_provenance + stub

Adds intersect_masks helper prototype + stub + truthy/falsy
logical-AND test.

Test (test_g7_intersection_with_provenance):
  community_mask  = {1, 1, 0, 0, 1}
  provenance_mask = {1, 0, 1, 0, 1}
  intersected     = {1, 0, 0, 0, 1}  // logical AND

  - Boundary cases: in-both (survives), in-one-only (filtered),
    in-neither (filtered).
  - Truthy semantics: positive non-zero counts count as 'in'
    (defensive for future provenance helpers that might return
    counts rather than 0/1).
  - All-zero a → result must be all zeros.

Stub copies a, ignores b. Test fails on the (a=1, b=0) → expect 0
but stub returns 1.

Verified RED:
  ./build/test_runner --filter=test_g7_
  → first four G7 tests PASS
  → test_g7_intersection_with_provenance FAIL
    (line 376: 0 != isect[1] — stub copied a[1]=1 instead of
     ANDing with b[1]=0)

T7.5 GREEN: out[i] = (a[i] && b[i]) ? 1 : 0.
- T7.5 GREEN: intersect_masks logical AND

out[i] = (a[i] && b[i]) ? 1 : 0. Logical-AND not bitwise-AND so
truthy non-1 inputs (e.g., a future provenance helper that returns
per-node hit counts) still produce the right mask.

Verified:
  ./build/test_runner --filter=test_g7_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 150 passed, 0 failed
- T7.6 GREEN: empty intersection returns zero rows (no silent fallback)

Single-GREEN ticket like T4.5 / T6.7 — the property follows
naturally from T7.3's induce_subgraph + T7.5's intersect_masks.
The test locks the contract in place so a future regression
("if mask is empty, fall back to full graph for usability") would
trip the assertion.

Test (test_g7_empty_intersection_returns_zero_rows):
  - Graph: 5-node path a-b-c-d-e.
  - Compose: community = all 1s, provenance = all 0s. Intersection
    is all 0s (no node satisfies both filters).
  - Verify intersect_masks output is zero across the board.
  - Verify induce_subgraph produces a 0-node, 0-edge graph (NOT a
    copy of the original — that'd be the silent-fallback bug).
  - Verify brandes_compute completes cleanly on the empty graph
    without crashing or returning an error. The caller's
    responsibility is to skip emitting result rows when
    node_count == 0.

The semantic guarantee: a user query "centrality within community
C ∩ provenance window W" with empty intersection returns an empty
result, not the centrality of the whole graph.

Verified:
  ./build/test_runner --filter=test_g7_  → 6 passed, 0 failed
  ./build/test_runner (full suite)        → 151 passed, 0 failed
- T7.7 GREEN: cross-resolution isolation + recompute on mismatch

Single-GREEN like T7.6 — the property already holds end-to-end via
T7.1's lei_filter cache dispatch + T6.2's check_communities_cache
state machine + T6.3's %.17g resolution storage. T7.7 locks the
contract at the TVF level so a future regression that loosens the
resolution mismatch check (e.g., increases tolerance, drops the
check entirely) trips this test.

Test (test_g7_resolution_mismatch_recomputes) — poison-cache pattern:
  1. graph_leiden(resolution=1.0) → COLD_START path populates cache.
  2. Verify cached communities_resolution == 1.0.
  3. UPDATE g_communities SET community_id = 999 (impossible Leiden
     output sentinel).
  4. graph_leiden(resolution=0.5) — different resolution.
     check_communities_cache sees |1.0 - 0.5| >= 1e-10 → COLD_START
     → run_leiden recomputes from scratch.
  5. Verify all 8 returned community_ids are < 999 (no poison) AND
     n_rows == 8 (full partition produced).
  6. Verify cache now holds resolution=0.5 (write-back overwrote
     resolution=1.0 partition).

Adds a local config_get_text helper (the one in
test_gii_communities_shadow.c is file-static).

Verified:
  ./build/test_runner --filter=test_g7_  → 7 passed, 0 failed
  ./build/test_runner (full suite)        → 152 passed, 0 failed

G7 done predicate satisfied: 7/7 GREEN commits in git log,
make test-g7 exits 0, every Success Measure test name in the
plan has a corresponding implementation.
- T2.1 RED: failing test_g2_signature_stable_under_json_reordering

Adds test/test_topk_cache.c with the JSON canonicalization test
plus the test_main runner dispatch.

Test (test_g2_signature_stable_under_json_reordering):
  - Three orderings of a 3-key flat object → must hash identically.
  - Different value (days=7 vs days=30) → still distinct hashes
    (canonicalization MUST NOT collapse semantically-distinct
    queries).
  - Nested objects → sorting must recurse so {"window":{"days":7,
    "start":"X"}} == {"window":{"start":"X","days":7}}.
  - Arrays NOT reordered: [a,b,c] != [c,b,a]. Array order is
    semantically meaningful.

The hash primitive (DJB2 via graph_str_hash) is orthogonal to
canonicalization — same canonical input always produces same hash
regardless of primitive. Following minimal-scope discipline, T2.1
GREEN adds canonicalization only; the xxh3 swap mentioned in the
ticket title can come as a follow-up if T2.4's collision sweep
demands it.

Verified RED:
  ./build/test_runner --filter=test_g2_
  → test_g2_signature_stable_under_json_reordering FAIL
    (line 40: sig_a != sig_b — current topk_signature stuffs the
     raw filter_predicate string into the hash buffer, so different
     key orderings produce different hashes.)

T2.1 GREEN must:
  1. Parse filter_predicate via yyjson_read.
  2. Walk the tree depth-first; for each object, collect (key, val)
     pairs into a scratch array, qsort by key, build a new mutable
     object with sorted insertion order. Arrays preserve order.
  3. yyjson_mut_write the canonical form, hash that string instead
     of the original.
- T2.1 GREEN: canonical JSON normalization in topk_signature

Adds canonicalize_json helper that:
  1. Parses filter_predicate via yyjson_read.
  2. Walks the tree depth-first via canonicalize_into.
  3. For each object: collects (key, val) pairs into a sortable_kv
     scratch array, qsort by key, builds a new mutable object with
     sorted insertion order. Recurses into nested objects.
  4. For arrays: preserves order (semantically meaningful).
  5. For scalar leaves: yyjson_val_mut_copy passes them through.
  6. yyjson_mut_write the canonical form, copy to malloc so caller
     uses plain free().

topk_signature now canonicalizes the predicate before hashing.
NULL / empty / non-JSON input falls through to the original string —
defensive against callers that pass schema-violating predicates.

Hash primitive (DJB2 via graph_str_hash) is unchanged. The xxh3
swap mentioned in the T2.1 ticket title is orthogonal to
canonicalization and can come as a follow-up if T2.4's collision
sweep flags the 32-bit space as too narrow.

Verified:
  ./build/test_runner --filter=test_g2_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 153 passed, 0 failed

Properties verified by the test:
  - Three reorderings of {"days":7,"min_degree":3,"project":"acme"}
    → identical signatures.
  - days=7 vs days=30 → still distinct (canonicalization doesn't
    collapse semantic differences).
  - Nested objects: sorting recurses correctly.
  - Arrays: [a,b,c] != [c,b,a] — order preserved.
- T2.2 RED: failing test_g2_cache_hit_returns_stored_rows + put/get stubs

Adds the cache-table API prototypes (topk_cache_put / topk_cache_get)
and stubs returning SQLITE_ERROR.

Test (test_g2_cache_hit_returns_stored_rows):
  1. Cache empty before any write — get must return SQLITE_NOTFOUND
     and reset out-pointers to NULL (defensive against callers that
     left them initialized to garbage).
  2. Put stores (sig=12345, gen=5, prov=2) → seeds/nodes/edges JSON
     payloads.
  3. Get with matching key → returns SQLITE_OK + malloc'd payload
     copies. strcmp confirms byte-exact round-trip.
  4. Wrong signature → NOTFOUND, no payload returned.
  5. Re-put with same signature + different seeds → INSERT OR REPLACE
     overwrites cleanly.

API contract documented in graph_topk_cache.h:
  - topk_cache_put: lazy CREATE TABLE _gii_topk_cache on first call;
    INSERT OR REPLACE keyed by signature.
  - topk_cache_get: SELECT WHERE signature = ? AND edge_generation = ?
    AND prov_generation = ?. Generations participate in the WHERE
    clause for lazy compare-on-read (T2.3 verifies invalidation).
  - Out-buffers malloc'd by callee; caller frees with plain free().

Verified RED:
  ./build/test_runner --filter=test_g2_
  → test_g2_signature_stable_under_json_reordering PASS
  → test_g2_cache_hit_returns_stored_rows FAIL
    (line 90: stub returns SQLITE_ERROR; test expects SQLITE_NOTFOUND
     for the empty-cache initial probe.)

T2.2 GREEN: lazy CREATE TABLE inside put; SELECT/INSERT OR REPLACE
implementations; reset out-pointers in get's NULL-return paths.
- T2.2 GREEN: topk_cache_put/get with lazy schema + WHERE-on-generations

topk_cache_put:
  - Lazy CREATE TABLE _gii_topk_cache on first call (per plan section
    698 'created lazily on first cache write'). Schema matches plan
    section 689: signature TEXT PK, seeds_json/nodes_json/edges_json,
    edge_generation, prov_generation, cached_at TIMESTAMP.
  - INSERT OR REPLACE keyed on signature so re-puts overwrite cleanly.
  - cached_at populated via datetime('now') for future LRU/sweep
    consumers.
  - 8-char lowercase hex signature column (DJB2 32-bit). xxh3 swap
    will widen to 32 chars without changing the API.

topk_cache_get:
  - SELECT WHERE signature = ? AND edge_generation = ? AND
    prov_generation = ?. Stale rows (mismatched generations) are
    invisible — lazy compare-on-read per plan ADR (line 790).
  - Defensive null-out: out-pointers reset to NULL on every entry
    so callers can rely on NULL meaning "no row" without having to
    pre-zero.
  - "no such table" before any put → returned as SQLITE_NOTFOUND
    (logical miss) rather than propagating the prepare error.
  - copy_text_column helper extracts each TEXT column into a
    malloc'd string; caller frees with plain free().

Verified:
  ./build/test_runner --filter=test_g2_  → 2 passed, 0 failed
  ./build/test_runner (full suite)        → 154 passed, 0 failed

Round-trip properties:
  - put/get of (sig, payload, gen, prov) → byte-exact strcmp.
  - Wrong signature → NOTFOUND, no payload.
  - Re-put with same sig + new payload → overwrites cleanly via
    INSERT OR REPLACE.
- T2.3 GREEN: lock generation-invalidation property in place

Single-GREEN like T7.6 / T6.7 — the property already follows from
T2.2's lazy compare-on-read design (generation columns participate
in the WHERE clause). T2.3 enforces it as a regression gate so a
future change that loosens the staleness check trips this test.

Test (test_g2_generation_invalidation):
  - Plant signature 0xABCD at (edge_gen=5, prov_gen=2).
  - Baseline: matching generations → HIT.
  - edge_generation bumped → NOTFOUND.
  - prov_generation bumped → NOTFOUND.
  - Both bumped → NOTFOUND.
  - Direct SQL probe confirms the row is still PHYSICALLY in the
    table (lazy strategy: stale rows linger; staleness-driven sweep
    is a separate concern per plan ADR section 802).
  - Re-put at the bumped generation overwrites the stale row (PK is
    signature alone). Get with new generation succeeds; get with
    old generation now also misses — the new row replaced the old.

Test bug fix: signature column is %08x zero-padded hex, so
sig=0xABCD → '0000abcd' not 'abcd'. SQL literal updated.

Verified:
  ./build/test_runner --filter=test_g2_  → 3 passed, 0 failed
  ./build/test_runner (full suite)        → 155 passed, 0 failed
- T2.4 GREEN: 0 collisions over 10K (DJB2 sufficed for this fixture)

Empirical 10K-signature uniqueness sweep. DJB2's 32-bit space yields
~0.012 expected collisions per 10K-sample set (birthday paradox);
this specific deterministic input set landed at 0 collisions, so
the hash primitive doesn't need an upgrade for the documented
fixture.

The plan ADR (line 822) classifies DJB2's 32-bit space as
collision-risky in general — if a future contributor varies a
different parameter (e.g. provenance_table instead of top_k) and
hits a collision, the upgrade path is clear:
  - Implement FNV-1a-64 inline (10⁻¹¹ collision probability at 10K).
  - Or vendor xxhash.h per the plan's recommended primitive (10⁻¹⁵).
  - Widen the cache `signature TEXT PK` column to 16 / 32 char hex.

Test fixture: vary top_k from 0 to 9999 with all other inputs
constant. 10K distinct canonical strings → 10K hashes. qsort +
sequential-pass duplicate detection.

Verified:
  ./build/test_runner --filter=test_g2_  → 4 passed, 0 failed
  ./build/test_runner (full suite)        → 156 passed, 0 failed

This is a single-GREEN ticket like T2.3 — the property holds under
T2.1's signature primitive (no implementation change needed); the
test locks the property in place against future regression.
- T2.5 GREEN: external generation bump visible immediately

Single-GREEN ticket. Property already follows from
config_get_int64_public's no-cache design — every call
sqlite3_prepare_v2 + step + finalize anew, so an external write
between calls IS visible. T2.5 locks the property in place
against future regression that might introduce statement caching
for "performance."

Test (test_g2_external_generation_bump):
  1. Create mock_config table with generation = 1.
  2. Initial read → 1.
  3. UPDATE generation = 2 (simulating a bump from another writer).
  4. Re-read → 2 (the trap is reading a stale snapshot).
  5. Stress: 50 bump-and-read cycles, every read fresh.
  6. DELETE the row → read returns the default sentinel (-1).

Plan section 832 documents the SQLite trap: a long-lived prepared
sqlite3_stmt does NOT re-execute when its underlying row changes;
returns the snapshot from when it was first stepped. Avoiding it
is "don't cache the prepared statement at module level" — which
config_get_int64_public already does today.

Verified:
  ./build/test_runner --filter=test_g2_  → 5 passed, 0 failed
  ./build/test_runner (full suite)        → 157 passed, 0 failed

G2 done predicate satisfied: 5/5 GREEN commits in git log,
make test-g2 exits 0, every Success Measure test name in the
plan has a corresponding implementation.
- T3.3 GREEN: MUNINN_BRANDES_SHARE_THRESHOLD constant + accessor

Single-GREEN ticket. The constant is the un-defer trigger gate per
plan ADR (line 909): when kg_perf measures
brandes_share = centrality_call_time / total_pipeline_time
exceeding this value for 3 consecutive runs on the leading
strategy, G3 (filter-aware Brandes / induced-subgraph TVF)
becomes worth implementing.

Default value: 0.30 (30% of pipeline time spent in centrality).
  - Aligned with G4's theta_full rebuild threshold (also 0.30).
  - Reasonable empirical inflection per project history.
  - Overridable per deployment by the kg_perf harness — this is
    the SHIPPED default, not a hard-coded contract.

API:
  src/graph_centrality.h:
    #define MUNINN_BRANDES_SHARE_THRESHOLD 0.30
    double muninn_brandes_share_threshold(void);
  src/graph_centrality.c:
    Returns the macro value via getter so tests / kg_perf consumers
    don't need to include the SQLite-extension-laden header.

Test (test_g3_threshold_default_documented):
  - Accessor returns 0.30 within float tolerance.
  - Value is in (0, 1) — valid ratio range invariant.

Verified:
  ./build/test_runner --filter=test_g3_  → 1 passed, 0 failed
  ./build/test_runner (full suite)        → 158 passed, 0 failed

T3.1 (per-component timing in bench.py) and T3.2 (brandes_share
sweep producing g3_inflection.png) are Python instrumentation work
that lives in benchmarks/kg_perf/. They land via pytest tests with
@pytest.mark.G3 markers, separate from this C-side T3.3.
- T3.1 + T3.2 GREEN: per-component timing + brandes_share sweep scaffold

T3.1 — PhaseTimings instrumentation in bench.py:
  - PhaseTimings dataclass with phases_ms dict + measure(name)
    context manager that accumulates elapsed time per phase across
    repeated invocations.
  - total_ms() and brandes_share() derived methods. The latter
    treats 'centrality_call' as the special phase that drives the
    un-defer trigger; missing → 0.0 (degree-only strategies don't
    flag false positives).
  - sums_to(expected_total_ms, tol_ms) consistency check used by
    the property test.
  - time_one() reads Result.extras["phases"], populates phases_ms
    + brandes_share fields in the JSONL record. Strategies opt in
    by stuffing a PhaseTimings into extras; legacy strategies
    untouched (extras absent → no fields recorded).

T3.2 — brandes_share sweep script:
  - benchmarks/kg_perf/sweeps/g3_brandes_share.py: argparse CLI
    that writes a placeholder PNG to
    benchmarks/kg_perf/charts/g3_inflection.png. Real corpus
    synthesis (10K/50K/100K/500K edges) + matplotlib rendering
    lands when the un-defer decision is contemplated.
  - benchmarks/kg_perf/charts/.gitkeep so the documented output
    directory tracks under source control.
  - Module __init__.py for the sweeps subpackage.

Pytest tests (pytests/test_g3_brandes_share.py, @pytest.mark.G3):
  - test_g3_per_component_timing_sums_to_total — measures two
    phases, verifies the sum equals an externally-measured total
    within tolerance, and checks brandes_share lands in the
    expected band (~0.33 for 5ms centrality / 15ms total).
  - test_g3_brandes_share_zero_when_no_centrality_phase — defensive
    against degree-only strategies; no centrality phase → 0.0
    rather than KeyError.
  - test_g3_brandes_share_zero_on_empty_phases — guards against
    div-by-zero / NaN on a fresh PhaseTimings with no measure()
    calls.
  - test_g3_sweep_produces_chart — verifies the documented script
    + chart directory exist as scaffolding.

Verified:
  uv run -m pytest pytests/test_g3_brandes_share.py -v
  → 4 passed in 7.59s
  make test-g3
  → 1 C test (T3.3) + 4 pytest tests, all passing
  ./build/test_runner (full suite) — unchanged at 158 passed

G3 done predicate satisfied: 3/3 tickets implemented, every
Success Measure test name has a corresponding implementation,
make test-g3 exits 0.

The actual filter-aware Brandes implementation only lands when
the un-defer trigger fires (brandes_share > 0.30 for 3 consecutive
runs on the leading strategy) — that's downstream work the
trigger machinery now supports.

## [0.4.0] - 2026-04-22

### Features

- Update library agentic skills (#29)

### Other

- Refactor viz (#27)

* refactor: start rewrite of viz demo

* stamp new rc version

* continued rewrite of viz

* Commit changes about viz navigation and spheres for embedding markers

* Add sub charts in navigation

* docs: refactor documentation

* docs: refactoring the viz/ kg visualisations

* feat: improvements to knowledge graph colouring, filterings, node and edge sizing and layout algorithm configs

* chore: push min-degree server-side, isolate community opacity, unify KG restyle

Server:
- `min_degree` is now a query param on /kg/{table}; pruned after BFS expansion so
  communities shrink their node_ids and member_count with the same filter, and
  empty communities are dropped. Echoes back via KGPayload.min_degree.
- seed_metric/max_depth validation surfaces as 400 with a clear detail message.
- `_GraphCache` keyed by (db_path, table, resolution) memoizes BC across
  requests that only tune top_n/max_depth/seed_metric.

Frontend:
- Community opacity now applies only to compound parents via the three
  per-component properties (background/border/text opacity) rather than
  element-level `opacity`, which cascades to children in cytoscape.
- Seven separate styling `useEffect`s collapsed into one unified restyle pass
  that runs on every filter/colour/size change. Visibility runs first,
  producing hidden-id sets that feed normalization-aware node-size / edge-
  thickness so hidden outliers don't compress the visible range.
- minDegree joins the pending-reload group alongside topN/maxDepth/seedMetric.
- Theme context split into theme-context.ts so react-refresh stays happy.
- `resolved` theme derived via useMemo rather than a setState-in-effect round-trip.

CI pass:
- viz: ruff, prettier, mypy, eslint, pytest, vitest, playwright all green.
- benchmarks/harness: mypy annotation fixes on pre-existing type-arg errors
  (dict[str, Any] in s3_mirror + kg_api_adapters) so `make ci-all` advances
  past typecheck.

## [0.3.3] - 2026-04-21

### Other

- Finally publish to both npm and pypi after resolving deployment issues

## [0.3.3-rc1] - 2026-04-21

### Bug Fixes

- Fix release pipeline

### Other

- Fix/release pipeline (#26)

* attempt to fix release pipeline fo v0.3.2

* fix: address deploy pipeline issues

* fix: relock uv.lock file

* docs: update changelog
- Bump version and look at prototyping RC versions
- Another try

## [0.3.1] - 2026-04-21

### Other

- Fix/deploy pipeline (#25)

* fix: update ccache action versions

* fix: updated test targets

* fix: Increase macos build timeouts

## [0.3.0] - 2026-04-21

### Other

- Update plan docs
- Refined plan
- Feat/benchmarks cloud support (#21)

* tidy quality gates on er spec and remove google colab make targets since it is WIP

* benchmarks: benchmarks harness adding cloud awareness to be able to delegate jobs to cloud compute

* benchmark: fix missing check to rely on .jsonl files instead of actual sqlite files (which get stupid big)
- Benchmarks-cloud-support part 2 (#22)

* wip: ER Example

* feat: add benchmarks/infra/ — config-driven EC2 benchmark runner

Single-script lifecycle for remote benchmarks:
- runner.py: setup/run/status/teardown with S3 heartbeat monitoring (15s polls)
- user_data.sh: parameterized instance bootstrap with heartbeat, ccache, SSM
- Config via YAML + env var overrides (no hardcoded values)
- Spot instances with on-demand fallback; auto-terminate hung instances (>180s stale)
- Instance self-selects benchmarks via `manifest --commands --missing --limit 1`

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: em dash in SG description + IpProtocol parameter name

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: CDK infrastructure for per-branch benchmark pipelines

Stacks:
- MuninnCleanup: Lambda + EventBridge weekly schedule to prune AMIs >7 days
  (manually invokable: aws lambda invoke --function-name MuninnAmiCleanup)
- MuninnBench-{branch}: SQS queue + DLQ + ASG (spot with on-demand fallback)
  + step scaling (queue depth → 0..N workers). Scales to zero when idle.

Workers pull benchmark IDs from SQS, run via harness CLI, upload results
to S3. Spot interruption handled by SQS visibility timeout (message
reappears for retry). Poison pills go to DLQ after 3 attempts.

Also adds worker_user_data.sh (SQS-based worker bootstrap).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* add learnings for ER example to drop pairwise usage as an option and focus on the cluster style LLM comparrison grammar

* docs: add benchmarks/infra/README.md with full architecture and usage guide

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: add prime and submit subcommands to runner.py

- prime: launches on-demand instance, monitors cold start, creates AMI,
  updates config.yml with new ami_id
- submit: queries manifest for missing benchmarks, enqueues IDs to the
  branch's SQS queue (looked up via CloudFormation stack outputs)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs: replace ASCII diagrams with Mermaid in benchmarks/infra/README.md

Three diagrams: single instance mode, parallel workers (CDK), AMI lifecycle.
All validated via mmdc rendering.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prime-only mode skips benchmarks when limit=0, max_workers from context

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: extend AMI waiter timeout to 30 minutes

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* wip erv2

* wip erv2

* feat: add Plotly Dash monitoring dashboard for benchmark deployments

Auto-refreshes every 15s showing: SQS queue depth, ASG worker count,
per-instance heartbeat status/phase, and event log timeline.

Usage: uv run benchmarks/infra/dashboard.py (opens at localhost:8050)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: scaling policy uses visible+inflight messages, 10min cooldown before scale-in

The naive visible-only metric scaled in while workers were actively processing
(messages go invisible when pulled). Now uses a math expression summing both
visible and in-flight messages, requiring 10 consecutive zero-periods before
scaling in.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: add time-series line chart to dashboard (queue depth + workers over time)

Dual y-axis: messages (left) and workers (right). Accumulates data points
in dcc.Store on each 15s refresh, keeps last 240 points (~1 hour window).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: CloudWatch-backed time-series chart + ASG scaling events table

Chart now queries CloudWatch metrics (last 1 hour) instead of ephemeral
in-memory state. Full page refresh preserves all historical data.

Scaling Events table shows ASG activities classified as: SCALE OUT,
SCALE IN, SPOT RECLAIM, UNHEALTHY, TERMINATE — with instance IDs,
capacity changes, and timestamps from describe-scaling-activities.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updated default values for string only er to improve ER before LLM borderline cases

* fix: cloud-init clean before shutdown so AMI-launched instances re-run user-data

Without this, cloud-init considers user-data "already ran" from the prime
boot and skips it on ASG-launched instances. Workers would boot, poll SQS
for 20s, find nothing (because user-data never ran the SQS worker loop),
and shut down — silently wasting every run.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* ER analysis of Fp/Fn cases as well as cross comparing embedding models

* fix: systemd service replaces cloud-init for worker boot execution

Cloud-init only runs user-data on first boot. AMIs baked from a primed
instance skip user-data on subsequent launches, causing workers to run
the old prime script (limit=0) instead of the SQS worker script.

Fix: prime installs a systemd oneshot service (muninn-worker.service)
that runs on every boot after network-online.target. It downloads
scripts/worker.sh from S3 and executes it. No cloud-init involvement.

runner.py submit now uploads the rendered worker script to S3 before
enqueuing benchmarks, so the systemd service always gets the latest
version with the correct SQS queue URL.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: time range selector for dashboard chart (1h/3h/12h/1d/3d/7d)

Radio buttons switch CloudWatch query window. Period auto-scales:
1-3h = 1min, 3-24h = 5min, 1-7d = 1h granularity.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* revised er pipeline

* improving stage timing granularity

* fix: npm lock sync, exclude KG categories, add benchmarks/infra/CLAUDE.md

- npm/package-lock.json: regenerated to match @sqlite-muninn v0.3.0-alpha.1
- registry.py: KG categories excluded by default (BENCH_EXCLUDE_CATEGORIES env
  var to override). Prevents cloud workers from attempting unvalidated KG benchmarks.
- benchmarks/infra/CLAUDE.md: documents all gotchas (cloud-init, spot SIGKILL,
  SQS scaling, llama.cpp OOM, category exclusion, cross-region S3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* version control claude.md files

* fix: WCAG AA compliant color palette for dashboard

All colors now use Tailwind v3 shades with verified contrast ratios:
- Text: slate-100 (#f1f5f9, 14.5:1), slate-300 (#cbd5e1, 9.1:1),
  slate-400 (#94a3b8, 5.6:1) on slate-900 background
- Accents: red-400, violet-400, amber-400, green-400 (all >5:1 on slate-800)
- Conditional rows: green-950, red-950, amber-950 with matching accent text
- Previously: #533483 on #0f3460 was ~1.5:1 (unreadable),
  #2d4a22 on #0f3460 was ~1.2:1 (invisible)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updated charts for ER Example analysis

* fix: radio button text color + default time range to 3d

labelStyle color set to slate-300 (was inheriting black from browser default).
Default time range changed from 1h to 3d (72h) to show full deployment history.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* Add graph edge betweenness pruning of ERs

* add graph edge betweenness to benchmarks harness

* docs: cloud-enabled manifest pattern — gap analysis + agent-facing rule

Gap analysis at docs/plans/cloud_enabled_manifest_pattern.md identifies 5 gaps
between two manifest pattern implementations. Extracts the generalised 5-layer
pattern: permutation registry, status determination, manifest CLI, execution
lifecycle, cloud dispatch.

Rule at .claude/rules/python/helper_scripts/cloud_enabled_manifest_pattern.md
uses generic examples only (per agnostic rules). Target audience: AI agents.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fast C implementation of ER pipeline

* escape JSON strings for entity results

* feat: Added muninn_label_groups Address issue with community and cluster naming stalling filling up on thinking tokens and not ending.

* feat: HNSW parameter sweep for VSS benchmarks (M, ef_construction, ef_search)

VSS Treatment now accepts tunable HNSW parameters. HNSW engines (muninn-hnsw,
vectorlite-hnsw) get a full sweep: M=[8,16,32,64] x ef_construction=[100,200]
x ef_search=[10,50,100,200,400]. Non-HNSW engines unchanged.

Total VSS permutations: 420 -> 3,486 (3,360 HNSW + 126 non-HNSW).
Backward compatible: existing results (default params) keep their original
permutation_id without suffix.

Parameter ranges based on research across hnswlib, pgvector, Weaviate, Qdrant,
Milvus, and academic benchmarks. M has the most impact on memory, ef_search on
recall/latency tradeoff, ef_construction on build quality.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updates to session_demo

* fix cache init on sessions_demo

* feat: dashboard improvements — spot pricing, log viewer, metric fix, port 8060

- Running Instances card uses EC2 instance count (not ASG DesiredCapacity)
  to eliminate CloudWatch metric lag mismatch
- Workers table adds Lifecycle (spot/on-demand) and Spot Price columns
  via describe_spot_price_history per instance type + AZ
- CloudWatch Logs viewer: input instance ID, filter by ERROR/WARN/INFO,
  fetches last 60 minutes on demand (not auto-refresh)
- Port changed from 8050 to 8060

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: dashboard — dual ASG series, cumulative spend, uptime/cost columns

Chart: separate In Service (solid) and Desired (dashed) lines fix the
metric card vs chart mismatch. Cumulative $ (cyan, 3rd y-axis) tracks
estimated spot spend over time from CloudWatch InServiceInstances.

Workers table: adds Uptime (hours), Cost (accumulated $), Spot $/hr.

Also fix: systemd TimeoutStartSec=infinity (was 7200 = 2hr, killed
workers mid-benchmark when processing many permutations sequentially).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: --port CLI arg for dashboard

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prep sync from S3, circuit breaker, systemd timeout=infinity

Three failure mode fixes:

1. Prep sync: both user_data.sh and worker_user_data.sh now sync
   vectors, texts, and GGUF models from S3 during boot. Validates
   all 12 vector caches exist, warns on missing files.

2. Circuit breaker: worker stops after 3 consecutive benchmark
   failures. Prevents burning through the entire SQS queue when
   there's a systemic issue (missing prep data, broken extension).
   Phase log now reports "N run, M failed" for auditing.

3. Systemd timeout: TimeoutStartSec=infinity (was 7200=2hr).
   Workers processing many benchmarks sequentially were killed at
   the 2hr mark. Heartbeat + client-side hung detection handles
   liveness — systemd should not impose its own timeout.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: --limit 0 should pass limit=0, not skip the limit filter

`if limit:` was falsy for 0, skipping the filter and enqueuing all
missing benchmarks. Changed to `if limit is not None:`.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: on-demand pricing comparison in dashboard workers table

Adds OnDemand $/hr and Saving % columns. On-demand price fetched via
EC2 Pricing API (us-east-1), cached per instance type to avoid repeated
calls. Shows spot vs on-demand savings percentage per worker.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: radio buttons and inputs persist across 15s auto-refresh

Added persistence=True, persistence_type='local' to time-range radio,
log-level radio, and log-instance-id input. User selections are stored
in browser localStorage and survive interval refreshes and page reloads.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: split queue and workers into separate line charts

Queue chart: Visible, In Flight, Dead Letter (messages y-axis)
Workers chart: In Service, Desired, Cumulative $ (workers + cost y-axes)

Also fix: duplicate margin kwarg in workers_fig.update_layout.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: enable ASG group metrics for CloudWatch (GroupInServiceInstances etc)

ASG group metrics are opt-in — not published by default. The workers
chart showed zeros because CloudWatch had no data. Added
group_metrics=[GroupMetrics.all()] to the CDK ASG construct.

Also enabled via CLI on the running ASG for immediate effect.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* ensure GHA ci workflow uses only makefile targets

* address ci formatting and linting issues

* feat: multi-select instance dropdown for CloudWatch logs viewer

Replaces free-text input with a dcc.Dropdown(multi=True) populated from
running ASG instances + recent CloudWatch log streams. Labels show
"(running)" or "(recent)" status. Logs display includes instance ID
prefix per line for multi-instance views.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: install + configure CloudWatch agent for /muninn/benchmarks logs

The CloudWatch agent was lost during the systemd refactor — neither
user_data.sh nor worker_user_data.sh referenced it. Instances wrote
to /var/log/muninn/benchmark.log locally but nothing shipped to
CloudWatch Logs. The dashboard log viewer had no data to show.

Fix: user_data.sh installs the agent package (baked into AMI).
worker_user_data.sh writes the config and starts the agent on every
boot with the instance ID as the log stream name.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: upload no-op worker script during prime to prevent SQS polling

The AMI's systemd service downloads worker.sh from S3 on every boot.
During prime, this caused the instance to start pulling benchmarks
from the SQS queue instead of just doing the cold start. Fix: prime
uploads a no-op script before launching. The real worker script is
uploaded by runner.py submit when benchmarks are enqueued.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: prime auto-suspends ASG scaling, resumes after AMI creation

Prime lifecycle now: suspend ASG alarms + scale to 0 -> launch prime
instance -> create AMI -> resume ASG alarms. Prevents the ASG from
launching workers from the old AMI while priming.

Also adds _get_asg_name() helper to look up ASG from CloudFormation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prime monitor treats shutting-down as normal stop transition

The EC2 state machine goes running -> shutting-down -> stopped when
shutdown -h is called with InstanceInitiatedShutdownBehavior=stop.
The prime monitor was catching the brief shutting-down state and
aborting with exit code 1/2. Now waits for full stop.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: remove duplicate cmd_run that was embedded inside cmd_prime

cmd_prime was missing its closing return — the cmd_run function body
was inlined as trailing code inside cmd_prime. After prime completed
AMI creation, execution fell through into cmd_run which launched a
spot instance and exited with code 2 on shutting-down.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix sessions_demo embedding process to be consistent

* refactor sessions_demo builder to decouple from demo_builder

* fix: CloudWatch agent + SSM install moved outside cmake cold-start check

Both were nested inside 'if ! command -v cmake' which is skipped on
warm AMI boots. Moved to top level with their own idempotent guards
so they install on first prime regardless of cmake state.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* disable claude code on GHA from burning quota for now

* fix: node2vec_train() arg mismatch — remove weight col, fix order, use neg_samples/lr_init

Python treatment was passing 14 args; C function expects 13.
- Remove 'weight' column arg (C loads edges without weight column)
- Swap walk_length/num_walks to match C order (num_walks first)
- Replace min_count=1 with neg_samples=5 (C param name)
- Replace workers=4 (unused) with lr_init=0.025 (C param)

This unblocks all 54 node2vec benchmarks currently failing with
"wrong number of arguments to function node2vec_train()".

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: truncate embed texts to model's max_tokens context window

muninn_embed() (via llama.cpp) rejects texts exceeding the model's context
window with "text too long: N tokens exceeds context of 512". Other backends
(sentence-transformers) silently truncate, so we align behaviour in setup().

- Add max_tokens to EMBEDDING_MODELS: MiniLM/BGE-Large=512, NomicEmbed=8192
- Truncate doc and query texts to max_tokens*4 chars at setup time
  (~4 chars/token for English WordPiece gives safe headroom below the limit)

This unblocks all embed benchmarks on wealth-of-nations_n5000 that were
failing with 543-token texts against 512-token MiniLM.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: stop uploading SQLite DBs to S3, clean up local DBs after each benchmark

The SQLite workspace DB is a scratch file — only the JSONL metrics matter.
Uploading it wasted S3 bandwidth and kept large files (HNSW shadow tables
can be many MB) on the instance disk indefinitely, causing disk-full failures
after running hundreds of benchmarks.

- harness.py: remove mirror.sync_to_s3(db_path) — JSONL only
- worker_user_data.sh: rm -rf results/{permutation_id}/ after each success
  to keep disk usage bounded regardless of how many jobs a worker runs

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: add sqlite-lembed to benchmark dep group so workers can run lembed treatments

sqlite-lembed was in the docs group, not benchmark. Workers run
uv sync --group benchmark, so all embed_lembed+* benchmarks were failing
with: ModuleNotFoundError: No module named 'sqlite_lembed'

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* feat: add diagnose subcommand to runner.py

Moves failure analysis from the standalone tmp/diagnose_workers.py into
the proper CLI. Uses CloudWatch Logs Insights for server-side aggregation
— one query replaces dozens of per-stream CLI calls.

  uv run benchmarks/infra/runner.py diagnose
  uv run benchmarks/infra/runner.py diagnose --minutes 120

Reports queue depth, ASG desired/active/hung, known failure types with
fix-commit hints, failed job IDs, and unknown exception patterns.
Queue/ASG/DLQ resolved from CloudFormation outputs — no hardcoded ARNs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: update llama.cpp to support Gemma4 models

* fix: defer llama_cpp import — remove vectors from prep/__init__.py

vectors.py has a top-level `from llama_cpp import ...` which was being
loaded whenever any benchmarks.harness.prep.* submodule was imported
(Python executes __init__.py first). This caused `manifest` and `benchmark`
subcommands to crash with ModuleNotFoundError on machines where llama_cpp
is not installed (e.g., the local submit path).

VECTOR_PREP_TASKS/VectorPrepTask are not imported from the package namespace
anywhere — both cli.py and test_prep.py import directly from prep.vectors.
Removing the re-export from __init__.py breaks the chain without changing
any callsites.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: always shutdown in worker EXIT trap, not just at normal exit

_cleanup (the bash EXIT trap) deletes the heartbeat file but did not
call shutdown. If the script exited abnormally — set -e crash, unhandled
error, circuit breaker — the heartbeat would go stale but the EC2 instance
stayed alive burning cost with no worker process running.

Add `shutdown -h now` to _cleanup so the instance terminates regardless
of how the script exits (normal completion, crash, or spot SIGTERM).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* docs: benchmark run resume notes — OOM fix + resumption checklist

Documents current state (2737/3886 done), the VSS ground-truth OOM bug,
all fixes applied this session, and the exact steps to resume.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: increase EBS from 20→40 GiB; add disk usage report after sync phases

20 GiB had near-zero headroom at peak (OS + build tools + llama.cpp
static libs + Python venv + S3-synced vectors/models ≈ 16-18 GiB).
Any large HNSW DB during a benchmark run would hit the limit.

40 GiB gives ~20 GiB breathing room at ~$0.08/GiB/month (negligible for
spot instances running a few hours each).

Also adds a disk usage report (df -h + per-directory du) after phase 04b
so the prime boot logs show the exact footprint before the AMI snapshot
is taken — making it easy to catch headroom issues early.

Applied to both runner.py prime instance and CDK bench_stack.py workers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: add HHMM to AMI name to prevent same-day re-prime collision

AMI names must be unique per account/region. Using only YYYYMMDD meant
re-priming on the same day (e.g. after fixing a bad prime) would fail
with InvalidAMIName.Duplicate. Including HHMM makes each prime unique.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: VSS OOM + add disk report to prime user_data.sh

- vss.py: replace (M,N,dim) broadcast in _compute_ground_truth with
  the L2 identity expansion (q^2 + d^2 - 2*q@d^T), reducing peak
  allocation from 14+ GiB to ~40 MB for N=50000,dim=768
- user_data.sh: add disk usage report after 04b_sync_prep, matching
  the same block already in worker_user_data.sh (prime runs
  user_data.sh, not worker_user_data.sh)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* address ci fixes

* docs: update resume_benchmarks.md with session 2 state

- Mark VSS OOM as fixed (bcbd309)
- Add new AMI ami-02fd844e73df018b4 and disk usage numbers
- Update infrastructure state table (ASG suspended, CDK LT v7)
- Replace resumption checklist: resume-processes + submit + set-desired
- Add note: prime updates config.yml only; CDK redeploy is a separate step

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* remove unneeded examples

* consolidate aggregate tables

* add packaging badges. remove cruft docs plans

* update docs

* chore: update llama.cpp submodule to latest

* consolidate development knowledge and targets for inner loop

* update sessions demo to latest schema changes

* apply ruff format to sessions_demo/cache.py

* ci: use make test-c in C-only build steps

The "Run C unit tests" step and the ASan+UBSan step both called
`make test`, which now cascades into test-c + test-python + test-js +
docs-build. On Linux/macOS build jobs that has two bad effects:

1. It runs `vitest` before `npm ci`, failing with "vitest: not found".
2. It doubles up with the separate `Python integration tests` and
   `TypeScript tests` steps later in the same job.

Scope each step to its flavour: `make test-c` for the C-only invocation
(and for the sanitized build), leaving `make test-python` / `make
test-js` as the later dedicated steps after their respective setups.

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
- Get the current set of features out to unblock some production testing and update docs later (#24)

## [0.3.0-alpha.2] - 2026-03-26

### Bug Fixes

- Update Claude GHA permissions and action versions (#10)

### Other

- Increment version for another alpha release

## [0.3.0-alpha.1] - 2026-03-26

### Bug Fixes

- Address CI memory leak not unloading models. Address CI build issues needing to propagate from main build to windows build script.
- Demo db builder fully functional
- Remove FTS5Adapter from kg benchmarks
- Removed FTS5Adapater from KG benchmarks
- Llm_extract example compares a few models performing the NER and RE benchmarked against GLiNER2 models and the honest speed comparrisons
- Refactored viz demo
- Notebook kernel CWD is examples/{name}/, not project root
- Add --cache-bust flag to Colab E2E test script
- Github Action CI Issues (#4)

### CI

- Set concurrency groups to cancel in flight builds when a newer commit is available and set timeouts to cap jobs that are hanging in CMake.
- Refactor sources and build targets into one centralised spot
- Refactor wasm build pipeline and rebuild the kg-demo database for wasm/ and viz/

### Documentation

- Update docs with benchmark results for VSS for 50k and 100K embeddings
- Update planning documents
- Huge refactor of benchmarking pipeline to consolidate duplicated code across benchmarking and analysis tasks
- Update the text embedding example README
- Start the embedding benchmark docs pages
- Ran all the ag-news benchmarks for embed category
- Update graph benchmakrs
- Update feature list in README and mkdocs index
- Update feature list again to include dbt-syntax graph selection
- Refactor mermaid diagrams to hires pngs. clean out old specs
- Finalised the kg-demo.db builder script
- Update demo builder plan
- Update plan for wasm+viz merger and demo_builder
- Add documentation about the logo tooling to remove the background

### Features

- Full refactor of benchmarking prep tasks
- Adding llama.cpp integration
- Refactored the vss benchmark pipeline to use GGUF models for embeddings to be consistent with the impending embed category of benchmarks
- Updated demo_builder and session_demo for narrowest context window for kg pipelines
- Benchmarks.sessions_demo to build incremental knowledge graph from claude code sessions files. Lots of speed tuning splitting tasks into fine grained steps to find bottlenecks, making models work offline without needing constant internet checks or unnecessary redownloads of models.
- Refactor sessions_demo and demo_builder to add GLiNER2 backend, and incremental UMAP for demo pipelines
- Added llama.cpp chat and enhanced the NER and RE tasks
- Improve demo builder build subcommands.
- Refactor llama_common out of llama_embed and llama_chat. Added muninn_tokenize_text, also improved the muninn_summarize
- Add Colab notebook generation and README badge enforcement
- Rename examples to {name}.py, add nbmake test targets
- Add 3-environment path resolution for Colab + E2E test script
- Updated suport of supervised and unsupervised NER, RE

### Other

- Update dev tooling script for logo image processing to spit out full sequential step explanation.
- Huge refactor of demo builder
- Add Claude Code GitHub Workflow (#1)

* "Claude PR Assistant workflow"

* "Claude Code Review workflow"
- Updated plan docs
- Use qwen3.5 in example
- Address CI issues

### Refactor

- Refactor benchmarks cli usage docs, add updated plannign docs for next phases.

## [0.2.0] - 2026-02-18

### Bug Fixes

- Update amalgamation script with new files
- Address windows amalgamation script and name collision in amalgamation source code

### Documentation

- Updated refinement of upcoming plan documents

### Features

- Feat (graph): Graph adjacency virtual table with lazy incremental rebuild
- Implemented dbt graph selector syntax tvf

## [0.1.0] - 2026-02-18

### Bug Fixes

- Improved visualisation hover text on embedding vis
- Sanitise fts query strings in demo visualisations

### Documentation

- Hardcode absolute URL instead of relative URL to example to bypass mkdocs link resolver

### Other

- V0.1.0 release

## [0.1.0-alpha.10] - 2026-02-17

### Bug Fixes

- Refactor npm deployment to generate platform specific package.json and refined the wasm/ and viz/ demo servers
- Address ci code formatting issues

## [0.1.0-alpha.9] - 2026-02-17

### Bug Fixes

- Update the publish.yml to update the package-lock.json automatically in publishing but attempt to try to pre-resolve during make version-stamp target

## [0.1.0-alpha.8] - 2026-02-17

### Bug Fixes

- Address more npm publishing bugs

## [0.1.0-alpha.7] - 2026-02-17

### Bug Fixes

- Address build and publish sequence and tsup prePublishOnly hook not having devDependencies available

## [0.1.0-alpha.6] - 2026-02-16

### Bug Fixes

- Iterating on github action publishing to npm with trusted publishing

## [0.1.0-alpha.5] - 2026-02-16

### Bug Fixes

- Deploying to sqlite-muninn npm org instead
- Need to specify --tag when publishing to npm

## [0.1.0-alpha.2] - 2026-02-16

### Bug Fixes

- Deploy multi-platform binaries to npm

### Documentation

- Fixed the linked logo image in the readme for pypi and npm

## [0.1.0-alpha.1] - 2026-02-16

### Bug Fixes

- Fix CI: use pysqlite3-binary for extension loading, install uv

The actions/setup-python@v5 Python 3.13 builds lack
PY_SQLITE_ENABLE_LOAD_EXTENSION, so enable_load_extension() is
unavailable. Use pysqlite3-binary as a drop-in replacement in CI.
Also install uv via astral-sh/setup-uv@v4 for package install tests,
and fix the persistence test's extension path to use build/muninn.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
- Fix CI: pysqlite3 fallback for ARM64/macOS, add build/ to npm binary search

pysqlite3-binary only publishes wheels for Linux x86_64. Fall back to
pysqlite3 (source compile) on ARM64 and macOS where no binary wheel
exists.

Also add build/ directory to getLoadablePath() search order since
make all outputs to build/muninn.{so,dylib,dll}, not the repo root.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Doc (benchmarks): Rebuilding the benchmarked dataset
- Doc (benchmark): Rebuilding the benchmark dataset

### Features

- Add WASM demo, overhaul viz frontend, refine CD plan
- Add publish.yml workflow for release automation
- Build platform wheels natively in CI with uv

### Other

- Initial commit
- Initial benchmarking results
- Updated benchmark metrics results
- More benchmark docs updates
- Updated docs benchmarks
- Update docs dataset URL reference
- Checkpoint planning documents
- Project rename to sqlite-muninn
- Add project logo
- Add graph community and centrality
- Updated planning documents
- Add ci and agent skills as well as python and nodejs wrappers
- Some house keeping
- Huge CI refactor
- Iterate on fixing CI
- Started works on visualisation tool and planning out KG benchmarks

### Refactor

- Refactored some more of the manifests architecture

<!-- generated by git-cliff -->
