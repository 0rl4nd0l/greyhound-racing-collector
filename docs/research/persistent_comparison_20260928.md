# Persistent four-way comparison: prepared, not activated

This implementation continues PR #194 without changing models, training data,
feature definitions or the scientific endpoint. It integrates the operational
collector through `8b5552c78222f9a0b5dfa1fa7bb30003bb375922`, including its final
shutdown/receipt correction. Production comparison remains opt-in. No service
installation, provider requests or target-outcome access occurred in this work.

The deployment and provider owner accepted preparation responsibility in the
operational thread `01a0e56c-5136-7672-9af9-be7a0ecfd780`, following its new user
instruction. Its deployment helpers and runbook are integrated here. This is an
accepted preparation handoff, not permission to start collecting outcomes.

The proposed allocation is October 5, 2026 at noon Melbourne time through
January 25, 2027 at noon, followed by closure on February 8 at noon. These are
future, unactivated proposal dates. If readiness or approval misses that start,
regenerate the entire packet with a later future start before activation. The
approved endpoint never moves because of performance or missingness. Existing
112-day inference, four-contrast multiplicity, calibration and worst-case
missing-result bounds remain as specified in the September 28 science plan.
The October successor and September 16 residual proposals still need explicit
deferral in the allocation approval; this task does not amend their protocols.

## Executing software

| Responsibility | Component | Durable state / schedule |
| --- | --- | --- |
| Fixed session admission and supervision | `run_comparison_schedule` around existing `prepare_freshness_rehearsal` / `run_freshness_rehearsal` | `sessions/slots`, scheduler flock, five-minute systemd timer |
| Predictions | Existing collector, retention, operational prediction worker and packaged subprocess | New prospective NVMe `operational-predictions` root, authenticated by programme authority; old paths unchanged |
| Official result acquisition | Existing `autonomous_official_result_capture` with required comparison guard | Exact admitted field and canonical race URL only; no browser, meeting discovery, redirect or alternative source |
| Queue, next-day and weekly repair | `run_comparison_result_queue` | Private SQLite queue/events/request charges, 20-minute persistent timer |
| Final closure | Same result worker calls existing byte-copy closure sealer | Immutable snapshot and hash receipt at end + 14 days; no evaluation |
| Monitoring | `check_comparison_health` | Five-minute timer; durable structural alert and systemd journal; no outcomes or comparative metrics |

No chat process owns runtime work. The operational owner responds to exceptional
holds; ordinary scheduled sessions do not require attendance. The first session
is the unattended canary. Later admissions require its completed session,
verified prediction job and a retained result for that exact job. A CLOSED
record from another job cannot satisfy this gate. Result collection continues
when admission is paused or its endpoint has passed, until fixed closure.

Each weekday session is fixed at 13:00–14:30 Melbourne time. Preparation is due
12:50–12:55; missed slots are retained and skipped, never shifted or made up.
Schedules use local calendar days across daylight-saving transitions. The
existing supervisor controls launch, freshness, shutdown and restoration.
No competing collector loop is introduced. The foreground operating mode stays
available and the tested operational release can deploy independently.

## Budgets and scope

These are engineering ceilings, not provider limits or projected consumption:

* 80 slots × 90 minutes = 120 observation hours. Session request ceiling 16,000;
  programme prediction requests 1,280,000. At most 1,000 additional capture
  attempts. Maximum new reserved operating/cleanup time 580,800 seconds.
* Shared campaign absolute ceilings 1,128 attempts, 1,400,000 logical requests,
  624,000 seconds, plus independent new-programme counters. Old records and
  consumption remain. Results have a distinct 24,000-request allocation.
* Source renewals: at most 80 × 192 operations, three hours per lease; renew only
  while OPEN, idle, and with unchanged denial/recovery/policy evidence. An expired
  OPEN allocation may renew within this approved total; STOP/COOLDOWN never does.
* Results: at most eight races/cycle, one GET per attempt, 24 attempts per race,
  1,000 races, 24,000 requests. First due T+15 minutes; next T+2 hours, next day,
  then weekly. Missed cycles do one overdue attempt, not a catch-up burst.
* Result response ≤1 MiB; retained original bytes and rejected envelopes remain
  private. Each child has a 40-second parent timeout, the cycle stops dispatching
  at 300 seconds, service runtime ≤360 seconds. Locks and holds can delay capture;
  prompt retention is an acceptance question, not assumed source availability.
* Proposed NVMe volume has approximately 602 GiB free at assessment. Reserve
  ≥100 GiB before installation. Session evidence ≤40 GiB, result evidence ≤32
  GiB, prospective predictions provisioned for 20 GiB; admission free floor 10
  GiB, result emergency floor 2 GiB. Nothing is deleted to recover capacity.
  Missing or changed mount identity stops before creating fallback directories.

Every transport attempt first consumes the shared campaign request budget and a
private request record. Unknown/interrupted requests remain consumed. Caller and
transport both enforce exact authority, source hold and field membership. Direct
use of the comparison result CLI cannot bypass the transport guard. Normal
comparison-off collector entrypoints preserve their existing behavior.

## Recovery and inferential consequences

Queue transactions use SQLite synchronous FULL. Logs, response bytes, parsed
records and quarantines stay private (0700 directories/0600 files under UMask0077).
A restart revalidates retained evidence before any new request. Valid dead heats
are retained with original tied positions; no scoring occurs during collection.
Changed or missing source runner identities, ambiguous fields and unsupported
terminal evidence cannot become successful closure. Cancellations/scratches
without supported evidence remain unresolved for the prespecified bounds.

Source denial or retry instructions create a durable shared campaign hold and
retain timing guidance. Expiry of a delay is not permission to clear that hold.
Lock contention defers work without consuming a provider attempt. A killed
request is uncertain and stays charged. A SIGTERM releases an owned result lock
through the guard; an unclean kill leaving an ownership marker requires explicit
operator reconciliation. It is never silently stolen.

Missed systemd timer events are checked after reboot. Pending result work resumes
where source authority and locks permit. Interrupted prediction sessions run the
original restore-only path before any later admission, including interruptions
between restoration and campaign closure. If reboot changes R3 PID or leaves
unknown worker/lock ownership, existing strict restoration deliberately holds
for the operational owner. Automatic safe abandonment after arbitrary reboot is
**not demonstrated**; this exception does not require routine session attendance.

Closure publication is an atomic directory rename. If a crash occurs after that
rename but before queue updates, restart verifies the snapshot hash and completes
queue terminalization once. Incomplete staging directories remain evidence.
Collection stops at its fixed deadline. Closure does not grant evaluation
permission. Unresolved common races remain in denominators and sensitivity
bounds; complete-case performance remains descriptive when selection bias cannot
be ruled out. No live source availability or prediction advantage is claimed.

## Narrow scheduler state convention

ADR0004's transactional storage preference is followed for the result queue.
The session wrapper deliberately retains the collector's existing immutable
create-once JSON claims under one kernel flock instead of introducing a second
job scheduler/store. Claims are consumed before preparation, fsynced by the
existing create-once primitive, and never reclaimed. On crash, an incomplete
claim is classified or held before the next slot. SQLite is not substituted for
existing collector receipts or campaign ledgers. This documented engineering
exception changes no scientific protocol or existing evidence.

## Prepared approval and acceptance

Use `prepare_persistent_comparison` to produce inactive science, campaign,
source-schedule and result-authority files. `authorize_persistent_comparison`
materializes approval-bound files only after the consolidated decision; it does
not install services or write shared campaign authority. The operational runbook
contains the exact deployment, preflight, monitoring and rollback sequence.

Approve the allocation/deferrals, strictly earlier machine-only histories,
scoped machine-only official result retention, finite shared budgets, and
installation/activation together. Do not open target results or comparative
performance during the canary. Verify through structural counts and hashes that
the first scheduled session survives a disconnected client, seals predictions,
restores collector units, discovers exact owed jobs, and retains at least one
valid official result. Then test a process restart without duplicate admission or
closure. A provider hold, missing result beyond next-day repair, failed cleanup
or failed canary blocks later admissions; results already owed remain queued.

## Executed verification and concrete pins

Runnable integration: `869fca1c66a6ee7c557facdb55e6f7592f2992cb` at
`/home/l4nd0/greyhound-persistent-release-869fca1c`. The research PR also carries
later documentation/evidence commits; deployment uses this tested integration.
The [closeout manifest](persistent_comparison_20260928_evidence/closeout.json)
records complete artifact, interpreter, package and approval-packet hashes.

* Exported source archive SHA-256:
  `54205f738737ae2b71f8c40fa794bbd58960840997db6caf74e324e7b74fbbdd`.
* Staged deployment manifest SHA-256:
  `79640db81b72dbdbeb5024d3f08882ca70d5e34f44add5c432929653f0ffefd5`.
* Inactive approval packet SHA-256:
  `d5abb20433db531a655909d9b671a8b73209b8e28aee1c5d2b4dd5b5aa776327`.
* Frozen registry: `938f433133057591e4ebd17a1ceef4fb25da252b5e3d6aba903194eb54b96ca9`.
* Residual plus box: `51639ddada362b1a14110461ef258dbff852eb7b2797ea074cd85f22381a4d32`.
* Half-strength residual: `e827e8f29c756c8995de08e15f8a6c25ca92f0709693cb55ad6bbeb9e16716e0`.

Thirty-one tests passed from the exported source (90.82 seconds). The full
export/replay/process proof took 101.46 seconds elapsed, 98.61 child CPU seconds,
and 283.5 MiB maximum child RSS. One additional fixed-package boundary test
(0.55 seconds) calls the actual scheduler `prepare_session`, preparer, execution
contract constructor and `FreshnessContract`; environment/services are stubbed.
Earlier focused controls passed (117 before final boundary fixes; 97 after the
preparer/contract and allocation changes). These are overlapping checks, not an
inflated count of independent acceptance cases.

All four probabilities replayed, and production predictions were byte-equivalent
in the comparison-off/on fixture. Comparison work was 41.4 ms elapsed / 29.5 ms
CPU. The paired whole-process times were 1.671 s off and 1.668 s on; that noisy
single pair does not establish a speedup. Sampled process RSS increased by
1.64 MiB. The historical 40–44 ms comparison estimate remains consistent.

The real collector process retained an initially unavailable synthetic result on
a later cycle; repeat processes neither refetched closed results nor duplicated
closure. Both lock types defer; shared source denial holds survive restart and
block the direct comparison collector too. Changed field identities quarantine,
valid dead heats close, and closure recovery survives interruption both before
and after atomic publication. Monitoring releases structural status only. No
provider traffic is possible in these tests because kernel network denial is
inherited by their subprocesses.

The former changed-runner-name acceptance was a demonstrated bug in the scoped
result path and is fixed before any real acquisition. Wrapper reconciliation-map
serialization and the new prediction-root execution contract were also repaired.
The original comparison-off behavior and immutable models remain unchanged.

Systemd unit syntax and read-only deployment preflight passed for all six staged
units. The actual unattended timer → supervisor → provider boundary remains the
first approved canary; synthetic HTTP responses and stubbed systemd controls do
not prove live result availability or reboot restoration after unknown ownership.
There are zero actual study members, official target results, or performance
scores from this task.

The consolidated scientific allocation must explicitly defer
`docs/forward_overround_successor_protocol.md` and
`/home/l4nd0/greyhound-prospective-readiness-20260916/PROSPECTIVE_EVALUATION_PROPOSAL.md`
for the proposed window. The [September 28 reservation manifest](future_comparison_20260928_evidence/reservation_review.json)
contains their identities; refresh allocation existence checks before approval.
No previous operational race becomes a member retrospectively.
