# Canonical Race Collection Phase 1 foundation

## Scope and ownership

Phase 1 adds an isolated `race_collection` package. It is not imported by the
deployed daemon, Flask application, scrapers, prediction services, model
registry, or training code. No service or timer is changed.

The SQLite operations database is the sole transactional authority for the new
workflow. The existing racing database remains an input to future adapters.
Reports and runtime JSON are projections, never workflow authority. Immutable
payload bytes live in the artifact store; operations rows refer to their
`sha256:<lowercase hex>` identities.

## Operations schema

`race_collection/migrations/0001_operations.sql` is an explicit, checksum-bound
migration. `SQLiteOperationsStore.migrate()` enables WAL and foreign keys and
is safe to repeat. A changed already-applied migration fails closed.
Forward-only Phase 2 migration `0005_exact_checksum_contracts.sql` hardens every
artifact-checksum column to the exact artifact identity grammar without
rewriting earlier migrations.

| Table | Ownership and invariant |
| --- | --- |
| `schema_migrations` | Applied migration version and exact SHA-256. |
| `operations` | Globally idempotent operation identity, kind, and intent hash. Reuse with different intent is rejected. |
| `racing_days` | Official local date, IANA timezone, aware opening instant, and durable close barrier. |
| `races` | Internal `RaceId`, owning Racing Day, and one persisted forward-only lifecycle state. |
| `lifecycle_events` | Append-only operation-bound transition history. |
| `race_aliases` | Source IDs, URLs, filenames, and source keys; `(source, alias)` is globally unique. |
| `dogs` / `dog_aliases` | Internal `DogId` and uniquely owned source aliases. |
| `dog_runs` | One Dog Run per `(DogId, official local racing date)`. |
| `run_observations` | Any number of provenance-bearing accounts of that Dog Run; repeated observations do not create repeated runs, starts, or wins. |
| `quarantines` | Explicit reason and stage, committed atomically with the corresponding terminal lifecycle transition. |
| `supersessions` | Append-only correction edge; the prior record is not updated backward or deleted. |

Every repository mutation uses `BEGIN IMMEDIATE`, records its operation in the
same transaction, and either commits all domain/event rows or rolls them all
back. Read connections continue to see the prior committed WAL snapshot while
a writer is open.

The day may close only when every expected race is awaiting day close. A race
may enter prediction only after closure. Result collection may start only after
every race in the day has committed a prediction, entered prediction
quarantine, or moved beyond that barrier. Only `prediction_committed` can move
to `result_pending`.

## Artifact layout and checksum contract

`LocalArtifactStore` publishes each byte string at:

```text
<root>/sha256/<hex[0:2]>/<hex[2:4]>/<64-character SHA-256 hex>
```

The checksum covers the exact stored bytes; media type is descriptive and does
not change identity. A caller-supplied checksum must match before any directory
or file is written. Publication writes an `.incoming-*` file in the destination
directory, flushes and fsyncs it, verifies its bytes, atomically replaces the
final path, then fsyncs the directory. Existing content is verified and reused
without changing its inode or modification time. Reads recompute the checksum
and fail closed on corruption. Derived digest-only paths, root containment, and
symlink checks prevent path traversal.

## Existing-to-domain mapping

| Existing concept observed in legacy code | Phase 1 domain concept |
| --- | --- |
| `state.json`, `odds_capture_state.json`, manifests, and status reports | Non-authoritative projections; later adapters must submit stable operations. |
| Filename/date/venue/race-number identities and `canonical_race_identity()` | Source aliases attached to a newly assigned immutable `RaceId`. |
| Form-guide history row | `RunObservation`, optionally supporting one provisional `DogRun`. |
| Official result runner row | Authoritative `RunObservation` of the same Dog Run; it supersedes provisional evidence without duplicating participation. |
| Protected paths and output-manifest hashes | Content-addressed evidence artifacts plus transactional references. |
| Script locks and oneshot timer overlap | Not adopted. A future single Race Collection Service owns durable operations and scheduling. |
| Dry-run/readiness/report status gates | Forward-only race state and Racing Day barriers in the operations store. |
| Shadow prediction snapshot | Future Deferred Snapshot Prediction artifact after the day barrier; no prediction behavior exists in Phase 1. |

The checked-in systemd units point at a dated resource-isolation worktree, not
the canonical repository path. The isolated worktree daemon is byte-identical
to the unit's invoked script, while the canonical repository daemon differs.
Consequently checked-in repository paths alone cannot establish live authority.

## Read-only discovery record

No runtime artifact was needed as a fixture and none was copied. Only exact
files named by the deployed unit were inspected read-only for shape, size, and
checksum:

| Source | Size | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| Canonical `scripts/shadow_autopilot_daemon.py` | 141,658 bytes | `e58b4915b13fc6afcdadf0f92258cfe676c3dd5127d587311fcaa16058b69f23` | Prove it differs from the deployed unit target. |
| Unit-targeted `scripts/shadow_autopilot_daemon.py` | 609,861 bytes | `df263b229a7fc291c90517066580ff2d6faa279743396622c236531c33fc526c` | Reconcile actual daemon code with the isolated worktree. |
| Unit-targeted `state.json` | 382,084 bytes | `17cb54a64490042e4e53dbcc01fee30b145f77a1e1d7d96be0126da02fc6955c` | Confirm broad report/projection shape rather than transactional state. |
| Unit-targeted `odds_capture_state.json` | 10,187 bytes | `5b4e61a999a20ad2ac06f0a0c85a55d467250e082db1f8d3ce671b24d1de5be3` | Confirm capture/report projection shape. |

No credentials, environment files, databases, models, logs, or broad runtime
directories were read or copied. No service manager command was issued.

## Deferred to Phase 2

- Expected programme source and hard-cutoff policy, including how a day closes
  with unresolved inventory.
- Source-adapter payloads and stable operation-ID derivation.
- Race alias reconciliation and tiered DogId resolution/confidence policy.
- Field-specific authority and the exact provisional-to-authoritative Dog Run
  supersession policy.
- Adaptive odds attempt/capture records, source timing, freeze authority, and
  runner/box conflict quarantine.
- Raw and normalized Sealed Race Evidence schemas and operations-to-artifact
  reference indexes.
- Racing Day reconciliation projections and administrative command surface.
- PostgreSQL implementation details; the store protocol remains the seam.

Phase 2 must continue to treat the new operations store as isolated and must
not assume authority over deployed collection until later cutover gates.
