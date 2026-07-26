# Phase 7 operational cutover and hardening

Phase 7 provides capability only. It did not install units, start or restart a
service, activate a release, disable the legacy path, access external services,
or touch production runtime data.

`RaceCollectionService` is the sole unattended dispatcher. Every phase is
fenced by the single durable scheduler lease. Manual tools and dashboards must
call the same `OperationalAuthority` application commands. There are no
workflow-owning timers. The service orders programme discovery, card/form and
adaptive append-only odds collection, closure and sealing, deferred prediction,
post-prediction results, immutable joins, reconciliation, then training requests.
It can request training but contains no training, tuning, or promotion command.
An explicitly versioned `result-blind-observation-v1` runtime input retains the
immutable nine-command plan but bounds execution at deferred prediction. It
omits result and training-example inputs and produces only the contiguous
receipt prefix 1--5; no result, join, reconciliation or training-request
handler is executed.
Adapters submit only closed typed commands. A `ClosedCommandDispatcher` is
assembled once at the composition root with exactly one trusted Phase 1--6
handler for every phase; missing, duplicate, and unknown bindings fail closed.
Handlers cannot supply workflow results. Receipts are derived only from
exhaustive durable postconditions after the handler succeeds. Result collection
is one Racing-Day command, and training requests bind an immutable request ID
and producer operation to exactly one reconciled Racing Day.

The operations database is workflow truth. Operation IDs give exact replay or
conflict semantics. Durable barriers, releases, audit records, alerts, scoped
pauses, reconciliation, cutover evidence, probation evidence, backups, and
restore drills are transactional records. Reports and exit status are merely
projections. Direct SQL repair is unsupported; operators use audited commands
with actor, reason, and operation ID. Critical alerts pause affected downstream
work while collection remains available. Resume and probation reset are explicit
audited operations.

Every Racing Day has one immutable nine-command plan. Planning is authorized
transactionally by the exact live lease token and generation at fresh trusted
authority time. A later generation may use either an ordinary or migration-v27
plan only after an append-only adoption transaction revalidates all nine command
identities, original provenance, the canonical contiguous receipt/progress
prefix, and any interrupted claim. Command execution and progress insertion
both require that exact token-fenced adoption. Progress uses trusted authority
time read inside its `BEGIN IMMEDIATE` transaction, so caller time cannot carry
a receipt across lease expiry. One public monotonic composition timestamp also
drives polling, `next_cycle`, lease acquisition and renewal. A
result-before-prediction attempt receives a distinct deterministic rejection
operation, leaving the immutable planned result-command identity available for
audited resume and retry. Attempt audit details contain closed error codes and
exception class names only; the service entrypoint maps arbitrary exceptions to
a text-free unavailable exit.

Release manifests are immutable content-addressed documents binding an exact
commit, typed config checksum, exactly schema 29, artifact contract, policy, supported
bundle versions, and a stable absolute service root. Unit generation emits one
generic service and no independent workflow timer. Generation and validation do
not install or enable it. Generated user-service content requires an explicitly
verified Python 3.11 executable and uses the valid user target
`default.target`.

That unit's `ExecStart` is backed by the checked-in
`bin/race-collection-service` composition root. The canonical
`phase7-config-v1` document binds one explicit `module:factory`
`runtime_adapter` and one immutable content-addressed runtime-input manifest.
The checked-in `race_collection.runtime_adapters:checked_in` factory rehydrates
exact durable cycle identities and invokes the existing Phase 1--6 public
authorities. It never selects inputs from environment variables, registries, or
filesystem recency; absent or changed live inputs fail closed. Startup validates
the active immutable release's existing Git commit, exact clean tree, resolved
in-root service source, and nonsymlink executable bytes and mode before importing
the adapter or acquiring a durable lease. It then validates the complete
nine-handler closed dispatcher; an absent or
incomplete adapter exits unavailable without advancing a Racing Day. The same
entrypoint provides deterministic `--once` and signal-aware `--continuous`
modes and closes adapter-owned resources on every completion path. It can emit
a training request through the accepted application API but contains no
training, tuning, or promotion executor.

Reconciliation accounts for expected/discovered races, exact runner/box evidence,
adaptive scheduled odds attempts, successes, retry failures and the final valid
pre-freeze snapshot, checksum-verified seals, predictions/quarantines, results
and provenance, joins, day-scoped retries/supersessions/failures, and the exact
active release chain. Any unexplained mismatch remains incomplete and blocks
cutover, backup, probation, training and promotion. Champion/challenger coverage
comes only from the day assignment and that day's registered forecast cohort,
with bundle and component artifacts verified; historical bundles are not
implicitly required. The append-only day cohort is operational coverage
authority: it authenticates the assigned champion, already-registered
challengers, exact bundle/component checksums, and exact forecast-service
commands before forecast work begins. It is not an evaluation verdict,
promotion evidence, or a substitute for a genuinely sealed, paired
500-or-more-race `phase6_trusted_evaluations` record. Day-cohort forecasts remain
ordinary authenticated forecast-service artifacts that a later independent
Phase 6 evaluation may consume. Ordering evidence includes only an immutable application
receipt for a real result command rejected before prediction; an alert or
counter is not barrier evidence.

Adaptive odds timing policy `adaptive-odds-timing-v1` stores the canonical
scheduled due time separately from actual attempt time. Actual attempts may be
zero through five seconds late, inclusive. Cadence is derived from due time so
latency never shifts later scheduling; early, missing, duplicate, wrong-policy,
post-cutoff, and excessive-late attempts fail closed. Official-result timing
policy `official-result-timing-v1` requires
`published <= observed <= attempted <= trusted_command` and bounds publication
through attempt at five minutes, inclusive.

Observation is explicit durable authority, not a consequence of registration.
While the main pointer remains intact legacy authority, an audited append-only
authorization selects exactly one verified candidate; revocation immediately
fences its composition and cycles. Eligibility needs the two exact complete
scheduled Racing Days immediately preceding the prospective boundary, and both
must be recorded after that authorization. Activation atomically consumes the
observation authorization while switching the pointer. Active and observation
composition use the same immutable release configuration; mode is durable
authority state, not configuration. Activation is a prospective Racing
Day-boundary transaction and
keeps legacy authority intact as the exact rollback target. Failures before the
pointer transaction cannot change authority. Rollback restores that durable
target but never silently reauthorizes observation. Legacy retirement is only a
later eligibility capability. After the exact sealed fourteen-day generation,
an audited append-only command may record eligibility bound to the active
candidate, activation operation, sealed probation, and preserved legacy
rollback target. It cannot disable, archive, delete, or otherwise mutate the
legacy service. Nothing here performs retirement.

Probation independently authenticates fourteen further scheduled Racing Days,
strictly after candidate activation, against their expected-programme identity
and immutable reconciliation, restart, ordering, and determinism artifacts.
Restart proof binds a durable later-generation plan adoption to an exact
contiguous earlier-generation prefix. Determinism proof binds two distinct
durable executions over the same immutable input and release before comparing
their outputs. Critical failures pause it; audited reset
starts a new generation. A generation cannot be sealed twice. At exactly
fourteen successes it seals the manifest
and writes the pre-existing Phase 6 input contract. Database triggers reject
unauthenticated Phase 6 rows. Only Phase 6 can consume that contract to promote;
Phase 7 never promotes.

Backups use SQLite's transactional backup API only after complete reconciliation,
replicate the explicit schema-versioned set of referenced immutable artifacts by
checksum, and record an inventory. Unknown future schemas fail closed.
An exact replay never rewrites the snapshot: it returns the recorded checksum
only when the isolated path is still a regular non-symlink file with identical
bytes.
The inventory is hostile restore input: it must be canonical JSON containing a
sorted unique list of exact checksums, and it must equal a fresh inventory
computed from the restored read-only snapshot before any replica object is
accepted. Schema 29 includes both determinism input and output artifacts plus
the versioned scheduled-due and bounded-late odds-attempt authority. An
isolated restore drill verifies snapshot checksum, `integrity_check`, the exact
inventory, every artifact, and an application query, and closes the restored
database on every path.

The checked-in determinism runner accepts only an immutable input identity,
reads those bytes itself, and stores its own output. Each execution binds the
Racing Day, release manifest, typed configuration, supported-bundle authority,
runner identity and implementation version, input, output, and distinct
operation identity. Evidence requires two distinct executions with identical
bindings and output bytes.

The Phase 7-to-Phase 6 probation authority relationally binds the probation ID
and generation, candidate release, activation and seal operations, state
checksum, and all fourteen accepted scheduled Racing Days. Retirement
eligibility is a separate append-only audited record over that chain and the
preserved legacy rollback release; it never mutates either release authority.

Command success without every check records a failed drill and proves nothing.
Tests use temporary directories and synthetic immutable fixtures only. They do
not establish deployment, live-data collection, training, promotion, service
installation, or runtime readiness.

The bounded operator interface, separate-database preconditions, result-blind
input contract, recovery commands and rollback boundaries are specified in
[Forecasting observation-canary operator contract](FORECASTING_OBSERVATION_CANARY.md).
