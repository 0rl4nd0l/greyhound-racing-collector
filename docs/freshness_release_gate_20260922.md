# Freshness release gate — September 22

Status: **HOLD — diagnostic measured, dependable timing unproven**. No candidate
deployment, result ingestion or research activation. Historical contracts and
outcomes below do not authorize further capture/overlap execution.

## Restricted source-capability probe: September 22, 20:08 AEST

One authorized date-page request returned HTTP 200 but triggered the frozen
access/challenge guard. No schedule projection was released; the second exact URL
was not requested. No retries, redirect, live refresh or timer operation followed.
This is an ambiguous guard stop, not proof of server denial or missing source
timestamps. Bulk-discovery replacement remains unsupported; no acquisition change
was made. See the [actual extraction finding and implementation decision](freshness_schedule_extraction_20260922.md)
and [earlier boundary-incident account](freshness_boundary_incident_20260922.md).
The probe is consumed. Existing 65/300-second limits and historical failures remain.

## Latest diagnostic execution: September 22, 18:59 AEST

The corrected passive supervisor observed production for 1090.359 seconds before
natural quiescence, completed a checked 0.033863-second timer/lock handoff, and ran
the single newly authorized source refresh. Native elapsed time was 48.250756
seconds. All 14 selected races were attempted; 13 CSVs were accepted and 9 races
met strict index requirements. Timers were restored, installed units/R3 unchanged,
and only the isolated private index was published. No capture or overlap test ran.

Request tracing places 23.675 seconds in discovery and 23.147 in the download pool.
The faster discovery versus the failed observation remains variable, not a proven
fix. The chosen next step is fewer-request exact URL-bound discovery, with unchanged
coverage, 65-second budget and 300-second R3 freshness. See the
[complete measurement and concrete design requirements](freshness_quiescent_measurement_20260922.md).
Evidence: `/home/l4nd0/greyhound-freshness-quiescent-diagnostic-20260922/`.
This execution is consumed; neither supervisor may be rerun.

## Prior diagnostic execution: September 22, 18:16 AEST

**HOLD — DRAIN_TIMEOUT_NO_ACQUISITION.** The separately approved instrumented
diagnostic was invoked once. Existing full collector PID 316981 did not drain
within 600 seconds; no candidate worker, refresh request or trace began. Both
timers were restored active/enabled without errors, all five unit hashes matched,
and R3 remained PID 149626. Source pins were unchanged. There is no new latency or
workload evidence and no supported performance correction from this attempt.

Retained assessment:
`/home/l4nd0/greyhound-freshness-diagnostic-20260922-prepared/REHEARSAL.md` and
`assessment.json`, alongside the unchanged prepared/started/result records.
This attempt is consumed despite zero acquisition. Do not execute that supervisor
again. The missing request/worker measurement remains necessary; another unchanged
live rerun, larger drain budget, deployment or capture is not authorized.

## Retained preparation before that execution

Latest offline continuation: **HOLD — acquisition design needs meaningful margin**.
See [the measured diagnosis and alternative](freshness_diagnosis_20260922.md).
Only opt-in request/worker instrumentation was added; no speedup or new live
success is claimed. Both historical refresh outcomes remain unchanged. A separate
[one-refresh diagnostic contract](freshness_diagnostic_rehearsal_20260922.md)
is prepared for review and needs fresh approval. Zero capture attempts in the
failed rehearsal does not authorize reuse of its supervisor or another run.

## September22 bounded capture/overlap execution

**Latest renewed execution: FAILED refresh budget.** Under subsequent user approval,
the17:28 rehearsal in
`/home/l4nd0/greyhound-capture-overlap-20260922-approved-1728/` drained production
in115.339 seconds and executed one candidate refresh with zero lock wait. Source
acquisition took65.128 seconds; the complete native phase took66.422 seconds,
exceeding65. Discovery/browser startup consumed38.920 seconds and downloads26.086.
Of14 selected races,12 passed metadata selection and2 were excluded; publication
was rejected because the source acquisition budget failed. No capture or full-lane
overlap started. Both timers were restored; installed hashes, R3 PID and canonical
index remained unchanged. See its `REHEARSAL.md` for assessed failure, not just the
supervisor completion status. No retry or deployment occurred. Release now requires
a narrower native refresh critical-path correction with demonstrated timing margin;
simply finding an idle window no longer addresses the demonstrated blocker.

### Earlier execution: drain timeout

Final outcome: **HOLD — DRAIN_TIMEOUT_NO_ACQUISITION**. The supervisor
finished at17:21:42.392729 AEST after600.765712 seconds. Existing production
full-service PID104485 did not finish within the600-second natural-drain gate.
No candidate lane launched, no race/window was bound, and no acquisition attempt
started. This is a failed rehearsal prerequisite, not a candidate timing failure
or a successful overlap test. The capture attempt remains unconsumed; the
one-shot supervisor must not be rerun or its budget enlarged automatically.

Both previously active/enabled timers were restored without errors. All five
service/timer definition hashes match the before snapshot; R3 remained PID149626.
At17:24 AEST both timers were active and R3 was running. There were no remaining
candidate workers. No deployment, canonical index replacement, retention/journal
activation or historical-record mutation was performed by this rehearsal.
Exact retained outcome: `result.json` and `REHEARSAL.md` in the directory below.

Release remains held: live capture/overlap, sustained270-second freshness and
cross-midnight load are unproven. No commit/PR or deployment approval is requested.
A further bounded rehearsal needs renewed authorization and a naturally idle
production window, not a larger drain budget or forced termination.

The one-shot rehearsal was started at17:11:41 AEST under the existing authorization.
Contract, source/harness hashes, admission guards and live status are retained in
`/home/l4nd0/greyhound-capture-overlap-20260922.BYw05Q/`.
`started.json` prevents rerunning it. Consult `result.json` if present; absence
means it is still in progress, not permission to launch another rehearsal.

The supervisor temporarily pauses timer triggers and allows existing work to drain
for at most600 seconds. It never kills healthy work or deletes a held lock. Its
normal candidate-work gate is240 seconds, total admission/observation ceiling2700,
refresh65, capture50 and source age270. A late native worker is recorded pending
natural drain, not treated as success. Prior timer states are restored on exit.

Preflight found and corrected a concrete invocation defect: the opt-in full lane
queued capture despite missing enable/execute/allow flags. It now requires all
three; production full-unit flags already supply all three. The competing rehearsal
full request supplies none and is refresh-only, so the shared one-attempt cap is
not bypassed by a second lane. Native regressions exercise each missing flag.
Final relevant candidate suite:63 passed; isolated guard/wiring suite:16 passed.
Independent review checked failed/no-row attempts, truncated progress, unresolved
manual claims, race/window freezing and repeat-attempt exclusion.

No live success or release readiness is claimed from this stopped run.

## R3 consumer correction — implemented offline

The subsequent "fix" request is implemented in the canonical consumer and native
collector coordinator. This supersedes the idle-health blocker recorded below.

Completed odds evidence now expires at the earlier of **completion + timer gap
+ accuracy** and **original source observation +300 seconds**. The old predicate
incorrectly charged acquisition time against the next idle scheduling interval.
It could expire a65-second observation at age75, only10 seconds after completion.
The corrected all-minute case remains ready at age76 and through its next due
time (age140); if no new lifecycle arrives, it is stale immediately afterwards.
A fresh terminal report cannot rejuvenate source evidence past300 seconds. The
270-second collector target, acquisition budgets and provenance checks are unchanged.

The phase coordinator also emits the existing native RUNNING lifecycle after
acquiring the shared lock, before acquisition. Final timing validation stays
RUNNING rather than briefly claiming a terminal failure. Only validated completion
becomes ready; measured failures still become failures. Native R3 requires matching
running-unit/PID evidence for ACTIVE and rejects a running report with an inactive
unit. Upcoming-source freshness remains independent of process activity.

Regression development reproduced both defects before the correction: two consumer
failures and12 missing-running-report failures. The first corrected consumer,
coordinator and complete adapter suite passed289 tests; additional native producer
to consumer checks cover ACTIVE versus inactive-unit divergence. No live collection,
capture, timer mutation, data write, prediction or research activation was performed.

Final targeted integration run: **306 passed in16.44s**, retained at
`/tmp/greyhound-r3-freshness-fixed-final.log`. It includes the complete native adapter
suite, coordinator timing/failure regressions, calendar overlap, timer generation,
budgets and checkpoint recovery. Independent spec and standards reviews found no
material issue in this narrow delta. These are offline results, not live acceptance.

Release preparation must now include a **new reviewed R3 source pin/package**, not
authority-only regeneration of unchanged5013ff03. The consumer file at candidate
base64c71c56 matches accepted5013ff03 exactly before this one-line correction;
no unrelated R3 code or model/configuration change is required. Installed R3 remains
5013ff03. Both collector units and the opt-in odds timer remain offline candidates.
The single live rehearsal and90-minute observation are still unexecuted; consumed
window reconciliation and cross-midnight live timing remain release prerequisites.

## Earlier continuation: timing correction implemented; release still held

The native coordinator now charges startup, verification, terminal writes and
release to its aggregate overhead, rather than declaring success immediately
after acquisition. Lock wait excludes only recorded retry sleeps; uncontended
lock/marker I/O is overhead. Failed verification is measured too. Native CLI
startup uses Linux process-start ticks, including interpreter/module startup
(tick-resolution uncertainty remains). External supervision still measures
stdout, exit and systemd dispatch; it must not treat the coordinator's terminal
assessment as process completion. Persisted timing counters describe the snapshot
before the terminal artifact write; the following acceptance check also charges
that write. They are not exact process-exit totals.

State retains its native state schema. Initial terminal persistence is explicitly
pending, not successful. Immutable phase results remain unchanged; a work-complete
checkpoint points to a separate post-release `terminal-timing.json` verdict.
Late overruns correct the native report and, under the existing lock without
stealing it, the matching current state. If another owner prevents correction,
the failed report and retained failure record remain authoritative; do not infer
success from an older state alone. Timing diagnostics do not redate source data.

The original eight regressions now pass. The intermediate full coordinator run
was **25 passed** (`/tmp/greyhound-terminal-release-20260922.log`). Expanded native
timing, checkpoint, timer/consumer, publication and calendar checks returned
**55 passed in107.89s** (`/tmp/greyhound-freshness-offline-final-20260922.log`).
This includes late lock-release failure correction and Linux CLI startup coverage.
These are offline checks, not live reliability or release approval.

Final combined check after reserving the residual5-second polling delay and
adding the50-second capture boundary: **56 passed in14.94s**. Exact command,
modeled timeline summaries and source SHA256s are retained in
`/tmp/greyhound-freshness-final-boundary-20260922.log`. This run uses
`--noconftest` with self-contained fixtures, avoiding unrelated application
initialization. Formatting checks for the changed coordinator/budget/tests and
`git diff --check` pass.

The deterministic calendar driver uses real native entrypoints, file locks and
checkpoints with virtual clocks/network acquisition. At source acquisition55s
(native refresh55.25s), it completed3 full and14 odds cycles; its maximum source
age was151.25s. At source acquisition64.75s (native refresh65s), it completed3 full
and13 odds cycles, with maximum188.75s. Both exercised ignored active-service
triggers, full wait/odds deferral and subsequent work by both lanes. Capture was
mocked at40s in these cases. These are modeled traces, not executed systemd timing
or live capture attempts, and do not exhaust all activation orderings.

The final max-phase case (native refresh65s, capture50s) completed **3/3 full
activations and10/20 odds activations**, with ten odds deferrals (one full-handoff,
nine lock-held),31 ignored ticks while the odds service was active, and modeled
maximum age **189.0s**. Both lanes performed mocked captures. This demonstrates
bounded progress in that deterministic ordering, not source/network timing proof.

The previously red65s source case is preserved as a rejection case: its extra
0.25s verification makes the native refresh65.25s. It produces zero successful
cycles and explicit phase-budget failures; no budget was increased to admit it.
The exact generated all-minute timer passes both native R3 parsing and local
`systemd-analyze calendar`; the default odds timer stays byte-identical.

### Necessary offline scheduling correction

The unchanged calendar cannot support the full declared load envelope:
65 seconds of allowed refresh already exceeds its 60-second permissible yield
age, before terminal overhead. Repeating another 65-second refresh does not fix
that inequality. This is a design counterexample, not an observed live outage.

The opt-in proposal removes only the four excluded calendar minutes (02,17,32,47)
for the freshness profile, preserving AccuracySec=15, locking, the full timer,
and both ordinary default generators. The resulting next-opportunity bound is
75 seconds; active-service ticks remain coalesced rather than new workers.
Completion is capped at source age75: 75 +75 +65 +10 = **225 seconds**,
45 below the target270 and75 below R3's300. Work admission is age130, reserving
50 work +65 refresh +20 combined overhead across two owners +5 residual polling
=270. Full lock
retry is80 (65 refresh +10 owner overhead +5 polling margin), not the previous70.
All are supported-load assumptions, not hard real-time guarantees.

### Separate unresolved R3 consumer boundary

Independent review identified a concrete integration blocker beyond index age.
`src/operator_ui/live_adapters.py` caps a completed odds report at its refresh
source time plus the timer gap/accuracy. With the proposed calendar this is75
seconds. A source aged65 at completion can therefore make the collector panel
stale after10 seconds, before the next timer activation and replacement. ACTIVE
reporting alone does not cover this idle interval. The upcoming index policy is
independently300 seconds; collector health is not the prediction admission gate.
Neither consumer policy nor prediction configuration is changed in this repair.
An offline native-consumer regression demonstrates the boundary exactly: source
age65 at completion is RECEIPT_READY; at age76 it is STALE while the independently
verified upcoming index remains AVAILABLE/FRESH. The exact-timer suite passed7
tests (`/tmp/greyhound-live-freshness-timer-exact-20260922.PBsIaX.log`), including
that expected failure classification, not a claim that the health gate passed.

Therefore the timer proposal is **not** an unchanged-R3-health solution. The
mandatory native-readiness release gate remains unsatisfied. Do not consume the
single capture rehearsal, commit a purported release, publish a draft release PR,
or request deployment on the strength of the timing tests alone. Consumed-window
reconciliation, native capture timing, sustained timing and two-date load also
remain unverified. No runtime, timer, canonical data, retention or research changes
have been made in this continuation.

Final read-only runtime check: both installed units still use the retention-lanes
release; neither enables live-freshness, retention or baseline activation. Both
timers remain active/enabled. Collector source is64c71c56; R3 source is5013ff03,
active PID149626 with Restart=no. No files were installed, timers paused, captures
attempted, results read or canonical records modified by this continuation.
Candidate remains uncommitted above64c71c56; there is no release commit/draft PR.

## Retained earlier preflight outcome: HOLD, no capture attempt

The arithmetic correction is implemented in `race_collection/live_phase_budget.py`.
Independent review agrees with yield 60/work 145 and unchanged 270/65/50/135. However,
`scripts/live_collection_cycle.py` still measures its phase after pre-checks and
before terminal persistence/release. It never calls `overhead_exceeded`. Arithmetic
reservation alone does not fix that integration defect.

Focused native tests now expose it in both lanes: eleven seconds of startup or
excessive finalization still returns LIVE_COLLECTION_COMPLETE. A50-second terminal
write can leave source age over105 seconds at release after an otherwise compliant
55-second refresh. Combined with the allowed135-second calendar gap and65-second
next refresh, this admits an age over300, not merely a missed270 target. These are
offline injected-delay counterexamples, not a claim that such a write was observed
in production. Full calendar/overlap integration proof remains absent.

Latest focused run: **20 passed, 8 failed**. The failures are retained regression
gates, not xfailed, removed or called passing:
`/tmp/greyhound-reserve10-native-overhead-red-20260922.rnp6Q6.log`.
Two failures expose missing timing diagnostics; six expose false successful
completion after excessive overhead. Existing successful source evidence remains
valid for its narrower boundary.

The finalization-only replay pins start 11:59:12.5 UTC and completion 12:01:00,
with the minute01 trigger ordered before completion, minute02 omitted and the
next allowed activation at12:03:15. Its asserted predecessor age is
**105.5 +135 +65 =305.5 seconds**. This is conditional allowed calendar ordering,
not an executed systemd/calendar proof. Both lanes still incorrectly return
success; both timeline assertions pass. Focused replay: **2 failed,23 deselected**;
`/tmp/greyhound-finalization50-calendar-aligned-red-20260922.JxbH1T.log`.

Planck independently identified the timing gap and supplied native repro tests;
Halley independently reviewed budget arithmetic, weather relevance, consumption
boundaries and this observation contract. Neither approved release. No reviewed
release commit or PR exists: HEAD remains `64c71c568fbd3ef6fdc972e5fcb7640cc3b7c70c`
with the candidate uncommitted. Conditional commit/PR authorization is not reached.

The one authorized capture/overlap rehearsal was **not executed**: zero new capture
attempts, zero acquisitions, no timer pauses, no installed changes and no canonical
data writes in this preflight. Next implementation boundary is finalization timing
and truthful terminal completion, then timer-driven overlap proof and reconciled
one-attempt admission. Do not request deployment approval before those gates pass.

## Runtime verification

Both installed collector ExecStart paths still target the retention-lanes release
at `64c71c568fbd3ef6fdc972e5fcb7640cc3b7c70c`. R3 source remains
`5013ff039fda418f47a59373041d7bc7c4124f07`, running PID 149626 when inspected.
Neither collector has live-freshness, input-retention or baseline-experiment flags.
Both timers are active/enabled; this preparation did not stop either timer.

Retained authority/recovery: `/home/l4nd0/greyhound-runtime-recovery-20260921.MCpbyV/RECOVERY.md`.
Latest isolated source proof:
`/home/l4nd0/greyhound-live-timing-recheck-20260922.Tya0VW/REHEARSAL.md`.

## Earlier unchanged-timer model (superseded by the proposal above)

The full timer uses OnUnitInactiveSec=15min, AccuracySec=30s. It cannot maintain
R3's 300-second freshness by itself. The odds calendar omits minutes 02, 17, 32
and 47; its longest scheduled gap is 120 seconds plus 15-second accuracy.
Events occurring while the same service is active are not independent queued
workers. The proof therefore uses the next opportunity after a completed yield,
not an assumed one-minute start cadence.

Both native entrypoints dispatch to `run_live_collection_cycle`. Every admitted
cycle starts with a genuine refresh. Full uses its 20–160 minute selection window;
odds uses its configured 0–60 minute window. Each publishes through the existing
strict publisher to the shared operational index. A narrower odds packet does
not preserve every race in the preceding full packet.

Within a cycle, the coordinator queues at most one eligible capture and, for an
explicitly configured full lane, one ordinary result observation. The latter is
NOT authorized in this rehearsal. Before additional work it checks source age;
if necessary it refreshes first. Before returning to timer scheduling it checks
whether the current source is young enough; if not it performs a final refresh.

The old yield condition allowed age 70 + trigger 135 + refresh 65 = 270, leaving
no allowance for process startup, verification, checkpoints and terminal I/O.
The corrected acceptance model reserves **10 seconds in aggregate**, not per
operation: yield age 60 + timer gap 135 + refresh 65 + overhead 10 = 270. Work admission
similarly tightens from age 155 to 145: age 145 + work 50 + refresh 65 + overhead 10 = 270.
These are reduced admission ages, not enlarged acquisition budgets.

Example without contention: a genuine source observation at t=0 publishes and
yields by t=60. A next odds opportunity occurs by t=195; startup/other omitted
costs plus refreshing must publish the successor by t=270. The predecessor's age
immediately before replacement is the important maximum. The successor age starts
at its own source observation, never at publication.

With contention, full waits at most 70 seconds in five-second polls for an odds
owner and writes a full-wait marker. Odds refuses a new phase while that marker
is active. Odds does not wait on a full owner; it defers to another calendar
opportunity. A successful intervening refresh resets the age calculation, so
adding every wait to the old packet is incorrect. Conversely, a skip is not lane
progress: timer-driven overlap validation must show actual subsequent refresh
and eligible work for each lane, not merely successful lock rejection.

The 281-second drain in the prior rehearsal was draining the old deployed batch
before migration/testing. It does not meet, validate, or replace recurring
candidate phase-wait bounds.

An overrun stops further candidate work and remains a failure even if source data
was published. Healthy acquisition drains under the existing outer timeout path;
there is no kill at 65/50, timestamp reset or fallback to the collector's 1,200-second
reader limit. Thus 270 is a supported-load acceptance target, not an unconditional
network/systemd scheduling guarantee. Startup before the measured entrypoint and
post-release output must be included by the external rehearsal supervisor.

## Capture admission must cover failed attempts

The existing DB check reads `live_odds` for race_id/capture_mode. It skips complete
captures but permits some incomplete/stale groups and cannot see attempts that
inserted no rows. Therefore the canonical DB alone cannot establish eligibility
under this rehearsal's stricter no-retry authorization.

Before any acquisition, under the real shared lock after natural drain:

1. Reconcile the production capture reports, attempt/progress records and any
   previous rehearsal records with the canonical DB for the relevant race/window.
   Include manual request/claim/attempt/response chains and capture-phase
   checkpoints; response consumption is not capture-window consumption.
   Include ambiguous started acquisitions, not just successful appends. Missing
   or unreadable accounting is a stop, not proof of absence.
2. Freeze the eligible set from the native plan and deterministic ordering:
   earliest scheduled jump, then stable race ID, then capture window. Exclude
   every previously consumed race/window regardless of inserted-row count.
3. Freeze exactly one race/window and a cap of **one acquisition attempt total**.
   Record its identity, source packet hash and accounting evidence. Once bound,
   changed identity, closed window or failure ends the rehearsal; no substitute.
   Overlapping requests may demonstrate refresh/lock progress but must not
   initiate a second acquisition under this shared one-attempt cap.
4. Record the maximum wall-clock duration and timeout/cleanup policy before
   executing. The capture harness is not yet approved by a passing offline test;
   no numeric duration is represented here as an executed contract.

Use the actual DB and ordinary collector authority, isolated candidate outputs,
no forward corpus/result-observer arguments, no journal or retention. Output
isolation must never reset consumption. Capture/overlap authorization remains
unused until this reconciliation and timing validation are complete.

## Weather-test relevance

Independent review identifies a pre-existing coverage defect: the canonical-page
weather extractor passes weather labels through the track-condition whitelist,
which discards `Overcast`. The tests are not obsolete and must not be deleted or
marked passing. Candidate and deployed browser source match at this boundary.
The source download also invokes the existing weather forecast collector when
weather remains absent; strict index admission still excludes missing required
metadata. This is not evidence of weakened provenance, but fallback success and
latency cannot be assumed for every race. Keep it visible in release review;
changing weather normalization is separate from this timing correction.

## Cross-midnight scope

The afternoon 55.784-second source proof applies to one discovered date. Full's
160-minute window reaches tomorrow from 21:20 Australia/Melbourne; odds' 60-minute
window reaches tomorrow from 23:00 in that timezone. Require the host/discovery
timezone and observation clock to match. At those boundaries the code correctly
retains two dates.
There is no live two-date timing proof. Do not advertise 24-hour freshness on the
strength of the afternoon result. A future limited daytime deployment observation
must end before 21:20 Australia/Melbourne, including rollback allowance, and must
verify actual one-date discovery throughout (a clock/date mismatch can retain a
wider horizon regardless of cutoff). General release requires
separate two-date timing evidence without expanding budgets or changing race scope.

## Subsequent 90-minute production observation (not executed)

Start only after an approved paired deployment and native R3 authority check.
Use append-only observation output, monotonic and UTC sample-start/end times, a
two-second target period, and a five-second maximum sample gap. Require at least
three completed full cycles and six completed odds cycles, plus actual lock-handoff
and eligible-work evidence. Absence of eligible capture must be reported as a
validation limitation, not fabricated progress or authority to substitute a race.

At each sample record the canonical packet hash, unchanged original source time,
native integrity/readiness result, verified race count, actual R3 upcoming and
collector lane status, completed report identities, process state, lock owner,
wait/phase/publication/release timestamps. Health 200 alone is insufficient. Do not
submit prediction jobs or read results. Reuse native consumer verification; do not
invent a parallel freshness predicate.

For adjacent valid samples, conservatively bound predecessor age at replacement
by its last measured age plus the elapsed interval to the next sample's read-end,
including read duration and any detected wall/monotonic discrepancy. Report this
upper bound separately from the sampled maximum; never call a two-second sampled
maximum exact. Require a complete retained publication chain with nondecreasing
original source timestamps, not merely increasing publication times. Otherwise
calculate each intervening packet's conservative age or fail the interval.
If packet chronology regresses, publication evidence is incomplete, clock steps
occur, a sample is missing/invalid, or a gap exceeds 5 seconds, the interval is
unverified/failing, not omitted. Sampling cannot rule out an unrecorded transient
integrity failure between reads; state that limitation explicitly.

Accept only if the conservative age upper bound is at most 270 seconds and native
R3 passes its native authority, integrity,300-second source freshness and lane-readiness
contract, with neither lane starved. Explicitly classify empty/no-eligible periods;
they are not evidence of usable race discovery or capture progress. Preserve overrun/error
samples and stop on a failed gate; do not extend the observation until green.

## Conditional deployment/rollback scope

No deployment is authorized by this document. Once gates support release, pin
and independently review the exact source, commit it and prepare the requested
draft PR. Generate both existing collector units from that pin with live-freshness
enabled, preserving Python, DB/evidence/state/lock paths, security and the exact
full timer. Include the opt-in all-minute odds timer as an explicit reviewed
schedule change; save both installed timer files and their active/enabled states.
No new service or boot-policy change. The R3 consumer correction also requires
its own reviewed source pin and package, using the accepted pinned environment
and unchanged model/configuration hashes. Do not patch the installed release in place.

Back up exact installed paired units and R3 binding/package/environment immediately
before an approved rollout. Pause triggers, naturally drain, install/verify both
units, obtain genuinely completed evidence from both lanes and regenerate authority
for the reviewed corrected R3 source. Verify the generated package, native startup,
unit identities and syntax before installation. Start/reconcile R3 only as explicitly approved, then restore
prior timer states and run the observation above. Keep journal/retention off.

Rollback restores the saved paired units, both exact timers and matching R3 files after natural drain;
leave R3 inactive if its saved authority is incompatible. Never restore an old DB,
rewrite attempts, undo appended evidence or alter the Wentworth Park acceptance.
The October trial and 1,000-race study are not part of this scope.
