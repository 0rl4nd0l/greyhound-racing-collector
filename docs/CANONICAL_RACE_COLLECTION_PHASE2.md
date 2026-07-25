# Canonical Race Collection Phase 2

## Scope

Phase 2 extends the isolated `race_collection` package with programme inventory,
identity resolution, collection observations, evidence sealing, and Racing Day
reconciliation. It is observation-only: no deployed daemon, timer, scraper,
prediction path, model, or production database imports this package.

Phase 1 boundaries remain unchanged. SQLite is workflow authority, artifact
bytes are immutable and content-addressed, and reports are projections.
Prediction and result orchestration remain deferred to Phase 3.

## Inventory and identity

`ProgrammeAdapter` accepts already-fetched source bytes. The adapter never
performs network access. Discovery assigns a stable internal `RaceId`, or
attaches inventory to an already-discovered or explicitly reconciled race in
the same Racing Day, and transactionally records the expected race, source
alias, programme checksum, and initial lifecycle event when applicable.
Existing expected inventory is immutable; later programme observations are
appended and conflicts fail closed. A conflicting observation and durable
`expected_inventory_conflict` quarantine commit together; exact replay raises
the same error without duplication. Reconciliation therefore compares collected
races with an independent expected population; zero races is not complete.

The `expected_races` population joined to `races` is authoritative for
reconciliation, day closure, prediction, and result barriers. Zero inventory
fails closed; directly discovered non-inventory races cannot block closure or
enter prediction. Unquarantined expected races block closure, while expected
races at `awaiting_day_close` or durably collection-quarantined are terminal.
Collection-terminal expected races never require predictions or labels.

Dog identities use three persisted tiers:

1. Authoritative registration alias.
2. High-confidence, source-scoped provisional name with one candidate.
3. Ambiguous, with no `DogId` assignment and identity quarantine required.

One `(DogId, local racing date)` remains one Dog Run. An authoritative
observation upgrades a same-source, equal-name provisional run in place and
appends another Run Observation; it does not add another participation, start,
or win.
The upgrade is persisted as a provisional-to-canonical DogId alias, so later
callers may submit either identity. Ambiguous decisions are recorded in the
authoritative identity quarantine table in the same transaction as the
append-only decision.

## Odds collection and feature freeze

The initial adaptive cadence is 30 minutes beyond three hours, 10 minutes from
one to three hours, five minutes from 10–60 minutes, and one minute in the final
10 minutes. Scheduling stops when jump is due. Failures use exponential retry
backoff from 30 seconds capped at five minutes; execution belongs to the future
service runner. Every success and failure is appended to `odds_attempts`, so a
failed retry cannot replace the last valid capture.

Sealing selects the latest successful observation strictly before authoritative
actual jump. Without actual-jump proof it selects strictly before the scheduled
jump persisted in expected inventory minus the caller's versioned conservative
buffer. A caller/persisted scheduled-jump mismatch is durably quarantined. The
absence of a qualifying capture fails closed. Runner mapping checksums are
mandatory for successful captures.

## Sealed Race Evidence and source authority

The raw manifest lists source artifact checksums and every odds attempt. The
persisted expected programme source and checksum are loaded through a typed
operations-store read, verified independently of caller mappings, and retained
as a distinct `programme_artifact` member. Missing, corrupt, malformed, or
checksum-mismatched programme provenance is durably quarantined and cannot seal.
The normalized package contains versioned normalized fields, all field-level source
authority observations, and the freeze decision. Both JSON documents are
canonicalized and written through `ArtifactStore` before one transaction
records their checksums and advances `collecting_odds` to `evidence_sealed`.
Stable operation replay returns the authoritative persisted seal before reading
or verifying retained inputs or quarantine state, and performs no writes.
Before a successful seal, every source, odds-capture, and runner-mapping
artifact referenced by the retained raw manifest is verified. Missing or
corrupt retained evidence blocks the seal. A race with no qualifying
pre-freeze odds is durably quarantined before sealing returns failure.

Authority is field-specific. Identity, runner set, jump time, and result order
are intrinsically critical and cannot be declassified by an adapter. Official
programme/jump evidence outranks official cards, source cards, market evidence,
and embedded form evidence. Conflicting
values at the highest authority for a critical field (identity, runner set,
jump time, or result order when later applicable) create a durable collection
quarantine and block sealing. No latest-write-wins rule is used.
Evidence fields use the closed typed `EvidenceField` registry, with criticality
as intrinsic metadata. Unknown or misspelled fields and criticality
declassification fail at the observation boundary. Any collection quarantine
blocks a fresh seal at both sealer and transactional-store boundaries; exact
replay of an already committed seal remains idempotent.

## Reconciliation

`CollectionRepository.reconcile()` queries expected inventory, lifecycle, and
collection quarantine tables in the operations database. Its counts and
unresolved identities are a regenerated projection. An unresolved race becomes
terminal for collection reconciliation only through an explicit quarantine
such as a versioned hard cutoff; a partial or empty scrape cannot report
complete. Phase 2 does not grant this projection authority over existing
outputs or deployed processes.

## Migrations 0002 through 0005

Migration `0002_collection_and_sealing.sql` adds expected races, versioned dog
identity decisions, append-only odds attempts, field evidence indexes, sealed
evidence references, and collection quarantine. Applied migration bytes remain
checksum-bound and repeat migration remains safe. Migration
`0003_identity_aliases.sql` adds durable provisional-to-canonical DogId aliases
and authoritative identity quarantine. Migration statements execute inside one
explicit transaction without `sqlite3.executescript`, so a mid-migration error
leaves neither schema objects nor a migration record committed.
Migration `0004_internal_identity_and_provenance.sql` adds checksum-bound
programme observation provenance while preserving forward-only migration and
replay behavior.
Migration `0005_exact_checksum_contracts.sql` enforces exactly `sha256:` plus 64
lowercase hexadecimal characters on INSERT and UPDATE for every Phase 1/2
artifact-checksum column, including nullable odds checksums, and validates
existing rows during migration.

## Read-only discovery and copied artifacts

The Phase 1 discovery record remains applicable. Phase 2 inspected only the
checked-in daemon's odds scheduling and report-shape code and did not read a
production database, model, credential, environment file, log, or broad runtime
directory. No ignored runtime artifact was required or copied, so there is no
source/destination/size/checksum copy ledger entry for this phase.

The observed legacy fixed-window/report mechanism is evidence only, not adopted
as authority. The new adapter accepts source bytes and the new repository
records attempts without invoking or changing that mechanism.

## Deferred to Phase 3

- Read-only discovery and explicit `legacy-origin` import of the active model.
- Racing Day model/release/policy pinning.
- Deferred Snapshot Prediction execution after closure.
- Per-race prediction commit/quarantine and the result-access barrier.
- Bounded official-result retries and immutable Training Example joins.
- Separation of On-demand Forecast storage from evaluation forecasts.
