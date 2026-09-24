# Collector metadata workload and host schedule — 24 September 2026

## Evidence boundary and source identities

This is an offline source/operational-metadata assessment. It reads no real race
payload, historical outcome or database and makes no provider request. The
inspected old package is `release-candidate-6e7fa6e4`, execution commit
`6e7fa6e423655df22a6769e524b18c975359b2f6`; this working source starts at PR #184
head `fe3d984a13a7446ca0af301dd736fed20bab0b96`. PR #185 is excluded. A changed
execution must receive its own package, source archive and plan identities; the
old package and campaign ledger are not reset or relabelled.

Retained `docs/collector_sportsbet_operating_policy_20260923.md` establishes 15
Sportsbet Python operations across an earlier launch and two completed refreshes
exporting six and nine CSVs. It does **not** retain their per-phase timestamps or
prove that all 15 calls occurred within one minute. Six plus nine downloads is
therefore an evidence-supported synthetic workload, not a reconstructed live
request timeline. Six plus sixteen covers the configured pair's selected-race
limits. One race alone cannot test this defect.

## Actual demand path

Both prepared units invoke `run_freshness_service.py`, then the existing daemon
and `live_collection_cycle.py`. Each refresh phase invokes
`shadow_autopilot_v1.py`, then `refresh_prejump_upcoming.py` with two workers.
The full lane takes its primary refresh (limit **6**) and returns that collection
phase; the odds lane skips the primary refresh and takes its odds refresh (limit
**16**). These are separate subprocesses and refresh snapshots. The full timer
is `OnActiveSec=15min`, `OnUnitInactiveSec=15min`; the odds timer is
`OnCalendar=*:*`, `AccuracySec=15s`. These trigger definitions do not guarantee
completed refresh throughput.

The old two-worker implementation makes a fresh `UpcomingRaceBrowser` for every
task, despite reusing worker processes. Its instance-local NextEvents cache is
therefore ineffective across races. Each download missing safe track condition
can issue a new metadata request. The one-worker path already reuses a browser
for the entire refresh. Worker process reuse does not repair the two-worker path.

The collector phase lock coordinates the lanes, and the source gate serializes
Sportsbet operations across workers/processes. Serialization prevents overlapping
source operations; it does not space them by six seconds or reset their rolling
count. Six calls followed by nine sufficiently quick calls can therefore reach
the shared 11th-operation STOP even when no source calls overlap. The 16-race
odds refresh can exceed the ten-operation/minute ceiling alone.

The minimal correction is a snapshot scoped to **one refresh**, shared by its
worker tasks. This reduces six plus nine (or six plus sixteen) potential metadata
acquisitions to at most two, one per refresh, while still validating every race
separately. No cross-refresh or cross-lane cache is required. The two-worker path eagerly acquires once for every nonempty refresh, even
when race pages would themselves provide track conditions; budget it accordingly. A failed snapshot must be shared
as a failure; workers must not convert it into independent fallback acquisitions.

`live_collection_cycle.py` can refresh again before pending work when freshness
requires it, and once more before yielding. Thus one snapshot per refresh is not
one snapshot per service invocation. Every repeated refresh remains charged.
The prepared `bounded80-v1` profile allows **80 seconds** per refresh and 90
seconds completion age; the non-profile default's 65 seconds is not this
package's bound. Publication target270 seconds, R3 stale rejection300 seconds,
all source gates and campaign limits remain unchanged.

## Snapshot validation and timing

The payload is suitable for reuse inside its bounded refresh: NextEvents contains
multiple events, and `collect_sportsbet_track_metadata` already accepts a snapshot
and applies matching independently per requested race. Preserve its observed-at
time and canonical payload SHA256 in every matching sidecar. Do not timestamp the
same payload again at each worker or reuse it in a later refresh.

Existing metadata matching checks greyhound type, normalized venue, race number,
local race date and nearest jump within the existing20-minute tolerance; tied
matches fail. This is **not** an exact jump-equality test. Missing events, ambiguous
matches and missing/placeholder track status remain rejections. The downloader's
race/runner checks and subsequent CSV/sidecar/current-index validation remain
responsible for their existing exact identity contracts. Sharing must not copy
one race's result to another, replace matching with a simple venue lookup, or
loosen any tolerance. Snapshots are provenance-bearing inputs, not proof of WIN
odds or receipts. Browser capture, market/runner validation, append and receipt
verification are separate stages and require their existing tests.

## Cumulative and rolling capacity

The real shared gate records each Python or browser operation before transport.
It accepts at most10 Python and1 browser operation in a rolling60 seconds, at
most2 explicit navigations per browser operation, and at most512 total operations
across this continuation. Operation513 is rejected; no reset on worker exit,
restart, new refresh or new package is permitted. Browser navigations and the
campaign's mixed logical-request count are distinct accounting units.

The pre-incident inspection found one source operation. Subsequent read-only
inspection found `OPEN`, no active operation, recovery consumed1/1 and **3/512
source operations**. Two unintended metadata requests occurred during an older
fixture run at01:07:42 UTC (11:07:42 AEST), before that invocation had a kernel
network guard. This violated the no-extra-probe instruction. No capture occurred;
HTTP status and usable response validity were not retained. The original calls
and denial/recovery history remain preserved.

The campaign ledger still records attempts1/12, logical requests405/48000 and
charged seconds721.058807/10800; it does not include these two calls. The separate
incident record proposes reserving two additional logical attempts and2.21 seconds
(the entire failed test invocation), giving effective accounting407 requests and
723.268807 seconds if approved. The ledger has not been rewritten. Live launch is
held pending explicit incident/accounting reconciliation. All later tests install
kernel network denial before imports and use temporary source gates.

Let R be future refreshes that acquire NextEvents and B future browser operations.
With no other new Sportsbet operations, the exact cumulative condition is
`3 + R + B <= 512`. At the remaining maximum11 capture attempts, R must be at most
498. This is a remaining budget, not an eligibility promise.

For scale only, a synthetic90-minute scenario with90 odds refreshes and6 full
refreshes would formerly demand `90*16 + 6*6 = 1476` metadata operations at full
selection. Sharing reduces that to96; with11 browser operations and the prior3,
the cumulative total is110/512. The illustrative six-plus-nine selection produces
846 versus96 metadata operations over the same dispatch counts. These are
arithmetic scenarios, not observed completed dispatches or proven maxima. Extra
freshness-driven refreshes must be added to R. The correction does not prevent a
STOP from11 distinct refreshes within60 seconds, a second browser operation
within60 seconds, or eventual exhaustion of512. Those remain valid failures,
never grounds to raise a ceiling or hide a request.

## Actual host schedule and startup/cleanup

Read-only host inspection reported `Timezone=Australia/Melbourne`,
`NTPSynchronized=yes`, `LocalRTC=no`; the clock was2026-09-24 01:02:17 UTC,
which is11:02:17 AEST (UTC+10), not AEDT. The pinned old plan's explicit offsets
are authoritative:

| Stage | Local host time | UTC |
| --- | --- | --- |
| Admission and preflight open | 24 Sep11:30 AEST | 24 Sep01:30 |
| Observation and timer start | 24 Sep12:00 AEST | 24 Sep02:00 |
| Observation ends; cleanup begins | 24 Sep13:30 AEST | 24 Sep03:30 |
| Existing cleanup allowance ends | 24 Sep14:01 AEST | 24 Sep04:01 |

11:30 is **preflight/admission**, not a live observation start. The executor
verifies source/runtime, pauses and drains existing triggers, reconciles under
the real lock and prepares the exact paired units during admission. It then
waits until12:00, rejects a start dispatch over5 seconds late, opens the charged
campaign lease and starts timers. The first-index deadline is180 seconds from
observation start; no startup warmup is silently moved before12:00.

Observation90 minutes plus cleanup31 minutes reserves7260 seconds, within the
read-only remaining10078.941193 seconds, leaving2818.941193 seconds. Cleanup
stops triggers, naturally drains owned work, waits for consumed capture-window
closure and restores the original files. It does not kill workers to obtain a
pass. Drain/closure failure produces `RESTORATION_PENDING`; arithmetic alone does
not prove timely cleanup. The existing admission check's latest practical11:45
recommendation is a preparation margin, not permission to move observation or
shorten its90 minutes.

A newly pinned replacement can use this same future window only if packaging,
meaningful offline proof and exact preflight finish in time. A late repair must
hold; it must not backdate12:00, extend13:30 or overwrite the old plan. Live
observation remains conditional on validated replacement identity, current
quiescence, unchanged remaining allowance and the existing supervised executor.


## Offline proof and release boundary

The immutable old execution package reproduced STOP on the attempted eleventh
Python operation through real discovery, downloader, parser, CSV/sidecar and
current-index publication with two spawned workers. Six-plus-nine and six-plus-
sixteen cases used invented HTTP payloads at the transport adapter; kernel IPv4
and IPv6 networking was denied. The old package remained unchanged.

The correction admits every selected race in both workloads with the sequence
Python metadata, browser operation, Python metadata across the shared gate.
Each refresh preserves a single payload hash and original observation timestamp.
A wrong-race event excludes exactly that race. Malformed/429 responses are acquired
once, then rejected across all workers without fallback acquisition. A gate with
511 invented persisted operations accepts512, rejects the next lane and remains
STOP across restart. The final CLI invocation includes the prepared80-second
refresh budget. These are synthetic orchestration results, not live capacity,
coverage, freshness, or successful prediction evidence.

Independent Standards and Spec reviews found no production blocker. Their concrete
requests were addressed: conservative per-refresh demand, incident-adjusted
capacity and the explicit80-second test branch. The capture fixture additionally
uses the real operating policy: an initial capture/append/receipt and same-claim
cross-lane handoff must pass; a different race's second browser operation within
60 seconds must STOP with the consumed claim and first receipt preserved.

Execution changes belong to a new collector-only package pinned from this branch.
PR #185 is excluded. The previous6e7fa6e4 package must not launch; it remains retained
for identity and failure evidence. Package generation does not install units,
arm timers or authorize a new observation window. The exact replacement identity,
validation results and explicit supersession are recorded separately after export.


Executed validation before final export:21 bounded-refresh/worker tests passed
in14.79 seconds against the184ea2bf execution export. With real policy, four
mismatch/denial capture cases passed in the128.03-second five-case invocation;
the canonical case initially failed because its accounting expectation omitted
three correctly charged metadata requests. After correcting the expectation,
that canonical packaged capture case passed in42.86 seconds. The earlier
no-policy expectation that an immediate second browser operation should succeed
also failed and was corrected to require STOP. These were fixture expectation
failures, not changes to production policy. The immutable old package's two
representative workload cases still fail at the source gate with the explicit
80-second budget (5.72 seconds); that is the intended defect reproduction.
No entire-repository suite or live observation was executed for this repair.
