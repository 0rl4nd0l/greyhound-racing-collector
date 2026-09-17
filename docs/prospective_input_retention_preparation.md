# Prospective input retention in the existing collector

Status: IMPLEMENTED_AND_SYNTHETICALLY_VERIFIED, DEFAULT_OFF. No merge,
deployment, restart, live prediction, protected-history access or research
activation is authorized by this PR. The original historical study remains
blocked/closed. The 1,000-race proposal is separate and unapplied.

## Exact call path

Both existing daemon commands (`run-once`, `run-odds-capture-once`) and their
existing service generators accept optional `--input-retention-config PATH`.
The option passes through `shadow_autopilot_v1.run_autopilot` and
`autonomous_live_odds_capture_command` into `autonomous_live_odds_capture.main`.
Omitting it leaves the existing command/service and capture behavior unchanged.
No new service, timer, scheduler, acquisition source or prediction pipeline.

`autonomous_live_odds_capture.execute_capture_plan` invokes
`ScheduledInputRetention` immediately after a successful WIN append and
`publish_scheduled_capture_receipts` returns PUBLISHED, before optional
research-corpus admission. It passes the exact plan, unmodified sealed capture
attempt and receipt publication result. Retention does not require or activate
a research cohort. It never invokes a scorer or changes the append result.

The existing `_scheduled_handoff` authenticates the exact receipt, report,
form and sidecar hashes and race/runner binding. The worker additionally retains
original raw form and primary-page bytes bound to their receipts. A mirrored
odds sidecar may point to its original raw_exports lane under the same evidence
root; those original bytes are preserved and checked, not reconstructed.

## Snapshot and reproducible features

The existing `seal_history_database` obtains a verified copy of one checkpointed
SQLite image. It checks filesystem identity and hashes across two source reads,
rejects nonempty WAL/journal files and concurrent source changes, and reads the
copy through immutable SQLite. It does not checkpoint, query or write the live
canonical DB through SQLite. This is fail-closed copying, not a WAL-aware backup.

Retention opts into a runner-scoped projection using the actual feature loader's
`clean_name` semantics. Target, same-date and future DB rows remain excluded by
the existing date policy. Original columns—including historical source/url
columns when present—are retained. The original helper's default return schema
and behavior are unchanged, preserving prediction-bundle verification.
The temporary consistent source copy still contains the whole database and its
history; filtering does not make that initial access runner-scoped.

The sealed DB, exact raw and normalized forms, adjacent sidecar, primary-page
bytes/receipt, exact WIN receipt/report, frozen model/manifest/config/schema,
source ZIP, environment lock and replay-worker bytes are hash bound. Capture
start, data-copy completion, checked manifest-seal time and prediction cutoff
are recorded. Source observation must precede retention; retention and parent
acceptance must precede cutoff, which precedes jump in the configured local
race timezone. This is possession/availability evidence for the snapshot;
it does not invent upstream provenance or establish historical completeness.

Feature generation calls the existing `build_live_feature_rows` on that exact
sealed database and colocated form/sidecar. It uses the retained schema,
generator source ZIP and model's feature order. The retained worker runs in an
isolated interpreter, checks its declared Python/package environment, denies
network/subprocess operations and permits feature-time data reads only from
those retained files. All 16 values and nulls are privately stored in canonical
`feature_values.json`, keyed by generated race/runner identity. There are no
probabilities, target labels, tuning, model scores or feature-route comparisons.
Only this frozen model's 16 inputs are certified by the replay comparison;
runner projection need not preserve unrelated generator features.

`replay_retained_inputs(bundle)` verifies every byte binding, executes the
retained worker and generator in a fresh interpreter and requires identical
canonical feature bytes. It never resolves original paths to obtain data.
`original_path` is provenance only. Scheduled bundles additionally require a
parent `terminal.json` with RETAINED status, matching manifest/config hashes and
pre-cutoff acceptance time; a child's completion marker alone is insufficient.
No packet is described as qualified merely because its bytes can be replayed.
A future approved prediction must consume these sealed inputs/features and the
matched retained WIN receipt. Re-reading the then-current canonical history DB
would forfeit this reproducibility evidence. Scorer execution is not added here.

## Authority and failure behavior

Without the option, no retention object is constructed. With the option, the
configuration must specify an approved scope/reference, exact authorized race
IDs, exact source DB path and validity interval before any history/form payload
access or even history file stat by the callback. The required scope token is
`complete_checkpointed_database_and_captured_sources_machine_only`. This flag
records a separately reviewed owner authorization; it is not itself permission,
a signature or permission to use protected outcomes for arbitrary purposes.
No such real-data authorization/configuration was created in this PR.

The scope must explicitly cover temporary copying of the full named database,
including any protected records it contains, machine filtering of earlier
histories, and use of the resulting prior-history rows and captured form/page
histories solely for input preservation and feature replay. No target outcome
join or outcome-derived diagnostic is emitted. If this scope is not granted,
the callback returns HISTORY_ACCESS_NOT_AUTHORIZED before reading payloads.
Date filtering alone never authorizes protected earlier results. Do not silently
remove protected earlier rows and call the resulting feature route unchanged.

At most one expensive retention attempt runs per collector invocation; further
callbacks report RUN_RETENTION_BUDGET_EXHAUSTED. A durable claim permits only one
attempt per exact race and approved configuration across invocations. Failure
consumes that attempt: no retries, substitution or later-history reconstruction.

The worker and its feature child share a process group. The parent waits at most
`min(max_seconds, cutoff-now)`, terminates that group on timeout, rejects late
completion and removes unsuccessful private bundle/scratch directories. A
finite outcome-free terminal reason is retained. Active WAL/source mutation,
hash mismatch, source-size and bundle-size limits are failures, not missing
feature values. Safe underlying HISTORY_DATABASE_BUSY/CHANGED codes survive;
source exceptions and raw worker stdout/stderr do not reach collector logs.

Odds already appended and their receipts remain intact. Retention failure does
not turn them into a failed odds capture or stop later normal captures; it
prevents claiming a reproducible retained input. Reports distinguish reported,
retained and rejected counts alongside existing capture denominators. A missing
terminal after parent interruption remains unusable; no automatic recovery is
introduced. Operator cleanup of such private scratch remains a separate action.

## Storage and capture-time cost

Only filesystem metadata was inspected for the real canonical DB: size
191,963,136 bytes (about 183 MiB) at the final September 17 metadata-only check.
WAL, journal and SHM siblings were absent at that instant; this does not establish
their state during any future capture. No DB contents were opened. Consistency
verification reads at least twice that size from the source (about 366 MiB),
plus SQLite reads of the temporary snapshot. It temporarily stores a full source
copy and loads earlier rows for projection. Persistent history is reduced to
the active runners, rather than retaining a whole-database copy for every race.
Actual source IO speed, peak RAM and retained-history size remain unmeasured.

With source size S and final data-bundle size B, extra disk peak is approximately
S+B plus small control files; persistent storage is B plus those controls. A
256 MiB source limit and 16 MiB bundle limit are proposed acceptance ceilings,
not live measurements. At the latter ceiling, 1,000 bundles could retain
15.625 GiB plus manifests/claims; no deduplication or automatic deletion is
claimed. Static source/model/environment files are copied into each packet.

A proposed 15-second worker budget caps expensive added work at approximately
15 seconds per collector invocation, plus bounded metadata/cleanup overhead,
because the callback permits one expensive attempt per invocation. This is not
proof of acceptable capture latency. The normal run's outer timeout and other
capture windows still matter. Acceptance must measure elapsed time, temporary
and persistent bytes, source-journal availability and impact on the next due
capture before recurring enablement is recommended.

## Configuration and static artifact preparation

Configuration schema: `scheduled_input_retention_v1`. Required fields are
`authority` (approved, scope, approval_reference, race_ids, history_source,
not_before, expires_at), `output_root`, positive `max_seconds`, positive integer
`cutoff_seconds_before_jump`, `max_history_source_bytes`, `max_bundle_bytes`,
and `static_files`. The seven static roles are model, model_manifest,
configuration, feature_schema, generator_source_archive, environment_lock and
feature_replay_worker. Each has an absolute path and SHA-256. Source/control
files must be reviewed and pinned before configuring any real operation.

Use unchanged market_form_residual_v1/full_strength and manual-default.json;
model SHA-256 is
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`;
configuration SHA-256 is
`f8a3c321dca12321a38a4d12a08f4f43461e1c1e73100eda871fd60252ed1820`.
The source ZIP must contain the approved generator dependency closure, including
scripts/__init__.py, scripts/utils.py, run_shadow_non_tgr_rf_evaluation.py,
run_feature_recovery_execution_v1.py and the imported utils/config modules.
`generator_files` in the synthetic test demonstrates complete archive assembly.
The environment lock declares exact Python and dependency versions; actual
runtime dependency closure and reproducibility must be reviewed at acceptance.
The replay worker itself is retained and executed from the bundle, so a later
checkout cannot silently replace its projection or read boundary.

## Fixture evidence and remaining acceptance

Fixtures exercise the existing service/command plumbing, real scheduled receipt
producer/verifier, snapshot and archived feature generator. A mixed synthetic
DB-plus-form history produces identical values and missingness for all 16
features before and after runner projection, and reproduces after originals are
removed. Other fixtures reject missing/rejected/mismatched parent terminals,
manifest writes crossing cutoff, unapproved scope before source access, changed
bytes, missing inputs, duplicate attempts and worker timeouts. Existing history
seal tests protect the unchanged manual-prediction default. This is fixture
proof only; it does not establish source completeness, real throughput or live
collector success.

Before any real replay, approve exactly one future race/first authenticated
capture outside protected study populations; one snapshot of the named canonical
DB (`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db`)
and the seven dynamic source artifacts bound to that receipt; only the static
reviewed source/model/config/environment artifacts; and narrowly scoped machine-
only prior-history processing as specified above. Keep feature/history payloads
private, allow only control hashes/counts/timing/failure codes in output, and
permit no retries, predictions, target-label evaluation or population amendments.
If whole-source copying/earlier protected feature use is not approved, stop before
payload access and report the unavailable authority; no substitute data route.

Merge, deployment/restart, enabling the option and that real-data acceptance
remain separate approvals. The rollout decision is to retain this reviewed PR
as default-off preparation; it is not ready for recurring live activation based
on fixtures alone. Statistical evaluation is separately assessed in
`prospective_residual_evaluation_assessment.md` and is not recommended for
adoption yet.
