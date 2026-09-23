# Bounded80 scheduling candidate and deciding rehearsal

Status: offline candidate, **release held**. This implements the accepted
[architecture decision](freshness_architecture_decision_20260922.md). It supersedes
that document's implementation-pending status, not its evidence or the failures in
the [release gate](freshness_release_gate_20260922.md). Bulk investigation and
historical recovery remain closed. No live execution occurred in this work.

## Exact candidate

The opt-in `--live-freshness --live-freshness-profile bounded80-v1
--live-freshness-contract <absolute-path>` reaches both generated services,
daemon entrypoints, the existing phase coordinator, native refresh and capture.
Without those flags, service/timer generation and the previous 65-second phase
allocation retain their defaults. Invalid flag combinations stop before work.

Both existing lanes use the same real collector lock: bounded refresh, at most
one eligible capture, refresh again when needed. The profile excludes model,
prediction, result-observer, retention, corpus and heavy maintenance execution.
The full lane's progress means completion of its operational refresh/capture cycle;
it does not mean completion of the ordinary full research/maintenance workload.
No parallel acquisition pipeline or new source capability is introduced.

| Setting | Candidate |
| --- | --- |
| Complete refresh / capture / overhead | 80 / 50 / 10 seconds |
| Completed yield age / capture admission age | 90 / 115 seconds |
| Target / unchanged R3 rejection age | 270 / 300 seconds |
| Full timer | OnActiveSec=15min; OnUnitInactiveSec=15min; AccuracySec=30s; Persistent=true |
| Odds timer | OnCalendar=*:*; AccuracySec=15s; Persistent=true |
| Refresh concurrency | Existing two private spawned download workers |
| Full / odds selected-race caps | 6 / 16; existing time windows 20–160 / 0–60 minutes |
| Scope | One Melbourne date; 90 minutes; end plus 20-minute cleanup no later than 21:20 |
| Acquisition allowance | At most one shared, newly reconciled capture attempt; 24,000 logical refresh HTTP calls |

**The installed odds service currently uses cap 4.** The proposed cap 16 is an
explicit rehearsal workload increase, not a claim to preserve installed request
volume. All source dates needed within the admitted window, native timestamps,
selection/exclusion metadata, publication coverage and strict runner validation
remain intact. The finite one-date admission rejects two-date/cross-midnight
operation; it is not an unattended service profile.

## Accounting, publication and recovery

Quiescent admission inventories scheduled progress and terminal reports, manual
claims and attempts, phase checkpoints, previous rehearsal reservations, and only
`race_id,capture_mode` from the live-odds table. It does not select odds values,
results, histories or models. Missing roots, unreadable/truncated records, unknown
attempt formats and ambiguous identities reject admission. No-row failures consume
their windows. Old records lacking a window conservatively consume all four.
Reconciliation is deliberately **pending until the approved, owned-lock admission**;
an offline document cannot certify tomorrow's unused allowance.

The shared lock-relative ledger reserves the single allowance before child launch,
then reserves each race/window alias. O_EXCL creation, file and directory fsync
make a partial reservation consumed. A separate fetch marker verifies reservation,
window, jump time and hashed input files immediately before network work. Failures,
timeouts, zero-row outcomes and interruption do not free the reservation. The
profile stops instead of choosing a substitute after a consumed attempt. A restart
may reconcile an unstarted phase; an ambiguous started phase cannot be replayed.
Old evidence is retained. Normal future deployment would need this same ledger
guard in all acquisition entrypoints; the rehearsal pauses the two installed
lanes and admits no other acquisition owner.

Publication uses the existing strict native index, publication and lifecycle
checks, preserves original source time and rejects non-increasing generations.
An immutable hash-linked publication event follows successful lifecycle publication.
Native producer JSON now uses the pretty/sorted serialization required by R3;
the tests consume actual written bytes rather than reserializing fixtures first.
Multi-file publication remains fail-closed rather than transactional: a reader
observing an inconsistent generation records unavailable/invalid evidence. The
rehearsal counts that failure; rejection alone is not availability.

R3 still enforces 300 seconds. The candidate adds the bounded `WAITING_FOR_PEER`
collector status only when the deferred report names the actual cooperating run,
the two services declare this profile, the peer is fresh, and the native index is
fresh within 270 seconds. Stale evidence or another peer is not promoted.

Changing a collector timer or service invalidates the **installed** R3 deployment
binding. This is tested and is not suppressed. The rehearsal does not rewrite its
manifest or restart R3. A separate operational-only measurement process runs the
candidate's native upcoming/collector/system readers against the exact candidate
commit, source tree and actual temporary installed unit bytes. It reads no model,
prediction, research or result sources. Candidate readiness is not installed R3
readiness. Any release still needs a separately reviewed coordinated R3 binding.

## Budget and availability claim

The complete idle recurrence is `90 + 75 + 80 + 10 = 255 seconds`, leaving 15
seconds to target and a separate 30 seconds to R3 rejection. Capture admission is
`age + 50 + 80 + 2*10 + 5 <= 270`, hence age <=115. A waiting full lane has a
95-second allowance for one odds phase plus overhead and polling. Source age is
never reset by publication, completion or lock acquisition.

These are conditional engineering allocations. Systemd AccuracySec permits
coalescing; it is not a dispatch latency guarantee under overload. Active oneshot
services do not queue every elapsed calendar tick. The full timer is inactivity
relative, not a wall-clock quarter-hour schedule. Its first activation can also
be due immediately from the prior inactive time; the OnActiveSec arm still exists.
The calendar tests exercise initial full offsets 0, 15, 45, 94.75, 900 and 930
seconds, both timer arms, ignored active-service ticks and lock handoff. Overrun
of 80.25 seconds fails and closes the shared scope. No test rewrites the previous
65-second failure into success.

The rehearsal captures systemd timer trigger/next-due observations, start/exit
monotonic timestamps and phase timing,
so dispatch, interpreter startup, final stdout and actual exit must fit the same
10-second overhead, not an extra unbudgeted tail. Timer dispatch is added to
process overhead; calendar delay beyond the allocated 15 seconds also consumes
overhead. Actual host Timer properties render durations such as `1d 4h 52min
52.401544s`, while service timestamps are numeric microseconds; both are parsed
and tested. It measures both lanes' actual
activations, wait durations and completed cycles. A sampled gap over five seconds,
clock discontinuity over 250ms, unaccounted publication or conservative age above
270 fails. Between samples, source age is bounded by the preceding observed age
plus elapsed monotonic time, with every intervening publication checked for
non-decreasing original source time. This is finite measured evidence, not a
guarantee about future host/source latency or sub-sample consumer availability.
Calendar accounting distinguishes observed activations, triggers during a known
active oneshot, coverage of nominal timer windows, and unverified missing trigger/
exit observations. An unobserved exit is never treated as an indefinitely active
service. Both full-timer next-due fields are retained rather than inventing a
quarter-hour activation denominator.

There is no quantitative availability SLO established yet. Admission/drain and
initial index construction are unavailable time. The first usable index is due
within 180 seconds; both lanes/native candidate readiness must be fresh by 20
minutes (to allow the full timer's first trigger), then remain so. Warmup samples
remain in the availability denominator. Initial load is not hidden as steady state.
At least three full and six odds cycles and one observed lock wait are required;
absence of contention or capture is inconclusive, not success.

## Source load and capture throughput

The sanitized existing observation records 249 logical calls for 14 selections:
141 discovery (one date plus 140 race pages) and 108 download-side calls. The
discovery cost is paid again even when few races are selected. Using that single
trace as a cost illustration, 27 refreshes in 51 modeled minutes is about 31.8
refreshes/hour, 7,909 logical calls/hour or 11,864 in 90 minutes. This is a workload
calculation, not a measured forecast. A short-refresh upper scheduling illustration
of 91 odds starts, six full starts and one additional post-capture refresh is 98
refreshes, or 24,402 calls at the same cost. Extrapolating download calls linearly
from 14 to 16 selections gives about 264.4 per refresh and 25,914 calls; date-page
population, failures and retries make either extrapolation uncertain.

The shared 24,000-call ceiling is a finite experiment ceiling for approval, **not
provider rate permission** or a wire-request limit. Every logical requests.Session
call is counted before it starts across spawned refresh workers. Timing traces
separate endpoint classes. Browser navigation attempts for the one capture are
recorded separately; browser subresources and transport retries remain unmeasured.
They must not be reported as zero or folded into a falsely complete HTTP total.
The all-minute schedule can substantially increase load relative to installed
cap 4 and reserved-minute scheduling. Approval must accept this proposed ceiling;
observed load and source behavior decide whether a lower cadence/cap is needed.

An 80-second refresh already skips minute ticks. Each capture adds up to 50 seconds
and can require another refresh. More refreshes protect index age but can consume
time needed for narrow capture windows. The shared single-capture allowance proves
at most one race/window's reservation, native acquisition/append/receipt and
post-capture freshness under real lane coexistence. It proves no burst capacity,
all-window coverage, sustainable multi-capture throughput or unattended capacity.
The window ledger reports observed eligible, attempted, excluded, missed and still
pending windows separately. Windows outside refreshed coverage remain unassessed.

## One prepared rehearsal

`scripts/prepare_freshness_rehearsal.py` packages committed operational code and
configuration only. It excludes retained datasets, models, artifacts and test
fixtures. Its plan pins commit/tree, per-file source hashes, archive hash, Python
binary hash, runtime environment digest, all four candidate unit hashes and all
five baseline unit hashes (including R3). The output is a local review artifact,
not an installation.

September 23 package verification found that the original plan's collector Python
environment cannot import Flask, required by the native R3 reader package. Its
successful supervisor `--help` check did not cover those deferred imports. That
unstarted package remains preserved at
`/home/l4nd0/greyhound-freshness-rehearsal-20260923-final/`; it is superseded and
must not be used for the deciding rehearsal.

The final review package is
`/home/l4nd0/greyhound-freshness-rehearsal-20260923-review-ready/` and uses the
existing `/home/l4nd0/greyhound_racing_collector/.venv/bin/python`. Preparation
now imports the actual collector entrypoints, acquisition dependencies and native
readiness readers under that interpreter with network, database and retained-data
access denied. It records interpreter prefix/version, imported module hashes and
installed distribution versions/RECORD hashes in `runtime-identity.json`. Admission
repeats this check before touching services and requires the supervisor to use that
same Python environment. This detects missing imports or changed recorded package
identity; it does not execute a browser or prove acquisition success. No packages
were installed or runtime environments modified.

The intermediate `greyhound-freshness-rehearsal-20260923-runtime-verified` package
also remains unstarted and is superseded by the final review package. Final
documentation verification corrected the table's odds-window typo from 2–160 to
the actual native 0–60-minute defaults; no selector implementation changed.

Proposed fixed time: **September 23, 2026, 12:00–13:30 Australia/Melbourne**;
passive admission 11:30–12:00; natural cleanup deadline 13:50. A late start does
not move the experiment to another time/date. The final local package's plan hash
is the approval identity. Execute only its pinned `source/scripts/run_freshness_rehearsal.py`
with that plan, digest and explicit approval ID. A one-shot started marker is
consumed even if admission fails; there is no retry/resume experiment command.

1. Compare baseline unit bytes/drop-ins; record timer active/enabled state and R3
   PID. Observe both services naturally inactive, MainPID zero, empty cgroups and
   no lock. Pause both triggers, recheck, acquire without stealing, then recheck.
2. Reconcile the frozen operational root inventory under that owned lock. Bind
   the fresh single-attempt allowance and exact source/date/time contract. On
   ambiguity stop with no source request. Install only the four approved collector
   files temporarily; full/odds evidence and current index use a new private root.
3. Start the exact timers at the fixed start; sample every two seconds. Retain
   actual native freshness/readiness, publication chain, source/coverage metadata,
   both lanes' service state/exit timing, locks, waits, request counts, phase reports
   and window accounting. One earliest eligible reconciled window may be reserved;
   it is not a promise that an eligible race exists in the fixed interval.
4. Stop on scope/phase/overhead violation, unknown lock owner, changed source or
   units/R3 process, exhausted request budget, consumed/changed capture identity,
   native integrity/authority failure, age/gap/clock failure, or readiness failure
   after warmup. At 90 minutes stop even if no capture occurred.
5. Close scope, stop both triggers, naturally drain descendants, preserve DB
   appends, receipts, failed attempts, private publications and ledger. Restore
   exact four baseline files/modes and previous timer activity/enablement; verify
   all five hashes and unchanged R3 PID. Never kill a live owner or remove its lock.
   Baseline file bytes and their directory are fsynced before the restoration
   manifest and any installation. After natural drain, classify the final capture
   reservation and wait for its native window to close before restarting the old
   collectors, which cannot honor the new ledger. This also covers a reservation
   written by an already-admitted child during shutdown. If drain or window closure
   exceeds the same 20-minute cleanup ceiling, or the claim is unclassifiable,
   leave triggers paused and emit
   `RESTORATION_PENDING`; the same approved package supports **restore-only**, never
   an acquisition retry. Power loss requires this restoration procedure on return.

No installed current-index replacement, prediction or retention activation is
part of this rehearsal. A successful measurement supports this restricted
scheduling candidate only; release stays held pending review of the measured
availability/load, coordinated R3 authority and capture-capacity limitations.

## Validation and local-change separation

Targeted synthetic tests cover generated CLI/defaults, actual native publication
and consumer serialization, monotonic publication, cooperative R3 readiness,
stale binding rejection, scope/date closure, consumed/partial attempts, request
ceilings, native capture window recheck, checkpoint restart, process workers,
calendar contention, external exit overhead and exact restoration after partial
timer pause or observation failure. Network and retained-artifact access are
blocked by the offline test runner; subprocess workers use synthetic browser code.

Validation: **147 distinct targeted checks passed**, across the focused runs;
the final changed coordinator/calendar/profile group passed 78 checks. Six
successful named-profile calendar orderings reached maximum modeled source age
210.75 seconds, with three or four full cycles, 22 or 23 odds cycles and exactly
one synthetic capture. The 80.25-second overrun case stops as expected. Native
receipt reuse and tamper rejection passed separately. The checked-in
`scripts/check_freshness_candidate_offline.py` provides the same read/network
guard for reproduction with the listed test modules and selected legacy seams.
No broad research, historical recovery or outcome-dependent suite was run.

The candidate incorporates the previously uncommitted bounded coordinator,
refresh workers, completion-time correction and their relevant tests. The
pre-existing official-result observer changes/tests and closed bulk-probe code/
tests are excluded from the commit. Historical diagnostic documents, timing
fixtures and unused diagnostic tests remain untouched outside it, except the
three user-named sanitized evidence/decision documents included as references.
No historical verdict or consumed attempt is edited.

September 23 continuation reused the retained scheduling/worker evidence and ran
39 focused candidate/timer checks successfully. Three new runtime-preflight checks
passed, including missing reader dependency rejection, environment identity drift,
and rejection before service access; the three affected restoration fault cases
also passed. No scheduling or acquisition logic changed in this correction. The
local follow-up commit includes only package preparation/admission, its dependency
probe, these tests and this candidate document. Unrelated dirty files remain outside
the candidate.

## Standards review

One material finding was corrected: restoring the old collectors could repeat a
consumed no-row capture. Restoration now drains before inspecting the final
reservation, waits for native window closure and preserves a pending state on
ambiguity/deadline. A delayed-reservation fault test covers the shutdown race.
The standards reviewer confirmed the correction; no other material findings.

## Specification review

Three findings were corrected: missing timer dispatch/activation accounting,
window-less exclusions that could also be counted missed, and backups not fsynced
before installation. The reviewer confirmed these fixes, including the follow-up
test that an unobserved old exit cannot classify later ticks as active forever.
Review totals: standards 1 corrected; specification 3 corrected. Sustained live
availability and multi-capture throughput remain unproven, not review passes.
