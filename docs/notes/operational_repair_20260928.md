# Collector-to-prediction operational repair, 28 September 2026

This continues PR #192 and the existing campaign. The September 24 observation
remains failed at shutdown despite nine verified predictions; no prior attempt,
STOP, missed preparation, or charge is relabelled. See odds_recovery_20260924.md.

Final outcome: candidate `7bd839f0` completed the 90-minute observation with eight
verified predictions and successful planned shutdown/restoration. Collection is
PAUSED. The release decision is GO for approval of finite, foreground-supervised
daytime collector/predictor sessions; unattended operation and an R3 upgrade are
not included. The final evidence and limitations are recorded below.

## Verified baseline

PR head c60ffb65ec1e156ee03df1122cef013aa49f500c and applicable CI passed.
The recovery worktree was clean. Installed collector files point to the original
retention release; both timers are disabled/inactive, workers and cgroups empty,
shared collector lock absent. R3 PID 149626 and binding SHA256
c39dd4bed98d6003a2e459cdb864b42cd0e6ae2eacbdf8b8c2b3af9e00e921c0 are unchanged.
Source state is OPEN with no owner, 241 operations and four retained denials.
Campaign consumption is 43 capture attempts, 39,100 logical requests and
16,674.500646 charged seconds; all leases closed.

Evidence root: /home/l4nd0/greyhound-collector-campaign-20260923/recovery-20260928.
The unrelated original worktree remains untouched.

## Narrow changes and validation

Since live candidate 40c5bc40, the earlier commits add authenticated planned
admission closure and bounded short diagnostics. Native STALE remains STALE;
only matching, reaped closure evidence with healthy source/index/peer authority
qualifies for planned shutdown. The capture and frozen scoring path is unchanged.

Commit 30a57dfe adds a relative launch option chosen after export, dependency
checks and retention preparation. Canonical create-once serialization is used;
sealed plans are never rescheduled. Cleanup must stay on the source date.
New cumulative authority is appended through hash-linked immutable extensions,
not edits to the original authority or prior amendment. This session selects
128 capture attempts, 96,000 logical requests and 43,200 charged seconds cumulatively.
These finite engineering ceilings accommodate the requested 90-minute run and
31-minute cleanup while retaining all historical charges. Source rates and denial
controls are unchanged; one 192-operation diagnostic allocation lasts at most
three hours. One accounted schedule metadata operation showed naturally useful
Healesville races, without denial; it is not a capture or prediction.

The first targeted offline run retained one failure: a lifecycle test expected
an older invocation's report to raise, contrary to the already documented
unmeasured-overhead behavior. The corrected test still rejects a wrong child
within the matching invocation. New tests went red before adding scheduling and
extension support, then passed. Restart tests establish durable non-reclamation
of dispatched/partially written or started race identities; they do not claim
that crashed inference is automatically retried.

Standards review: no findings. Specification review: no code findings; report
restart nonduplication separately from full recovery. Independent reviewers
performed no provider or runtime actions.

## Supervised observation

Package live-30a57dfe uses commit 30a57dfe86596dcdee7c262a08c12f310472801c.
Plan SHA256: 29d711ee14267e7147f614181d332f2f7f6ece07fe91be4b52ff42a00952a6ef.
Source archive SHA256: 364ac263948909a1c3ad68af2ecb62da14fcaa63adb72c2b72d4b33651a2b745.
Prepared window: 10:48:44.441864–12:18:44.441864 AEST, cleanup reserved through
12:49:44.441864. Baseline/source/budget/package/interpreter preflight passed.
Live outcome will be appended after observation and verified restoration.

### First attempt: timer-accounting failure, fully restored

88 selected packaged checks, the real paired-lock test and five worker/API
restart checks passed before launch. Current GitHub checks also passed.
At 10:49:56.690108 AEST the observation stopped after 72 seconds on
`dispatch_plus_process_overhead_exceeded`. Two refresh phases completed in
31.547288–34.965658 seconds; the initial index contained three Healesville races.
No eligible capture was attempted and no prediction was generated. Source
operations increased by two, with no new denial. Original units were restored
at 10:49:57.343156; collector timers held, R3 unchanged.

Exact replay reproduces the exception on retained sample 000036. Persistent
startup fired at 10:48:44 and the first refresh remained active beyond 10:49:15.
The next calendar activation triggered at 10:49:22.184158, with actual dispatch
0.051487 seconds and process overhead 3.196098 seconds. The observer incorrectly
charged the prior service's active time as additional timer lateness, totaling
10.431743 seconds. This is a local accounting error, not source staleness.

Correction 1be9bfcb retains raw calendar delay, credits only an earlier service's
observed lifetime across the nominal tick, and charges all remaining idle delay.
Persistent startup uses scope admission as its earliest possible trigger.
Ten-second dispatch/process-overhead ceilings, 270-second observed source-age
bound and 300-second R3 gate are unchanged. Two regression cases failed before
the fix; all six focused timer cases and all 37 original live samples passed
afterward. Negative tests still reject genuine idle delay and process overhead.
Both review axes reported no findings. A new package/attempt is required;
the first observation remains failed and charged.

### Second attempt: verified prediction, controlled interruption

Package live-1be9bfcb used commit 1be9bfcbe8c0ad68cab57a2189989283cce05146,
plan 71d039c685e03cf14ffb95bdabcbbc30eadcebc35f76cc18e1480357cdac934e,
archive 9e12d1892edb57a5b9d5768b330897a0b7913efea7ed0602fbcbcdd109df450d.
It started 10:59:32.589194 AEST. Five odds cycles refreshed inputs in
33.2267–44.7654 seconds. Initial weather unavailability recovered on later
observations. One eligible race/window was attempted and Healesville Race 3
produced PREDICTION_READY, verified 349.90312 seconds before jump. Prediction
processing took 4.706634 seconds; capture took 14.483016 seconds. Independent
chain verification passed all 28 checks without target-result access.

Four Open-Meteo 503 responses exposed a gap: the previous guard recorded retry
guidance only for explicit 401/403/429 responses. Headers from those actual 503s
were not retained, so neither presence nor absence of Retry-After is known.
Timers were stopped, current work drained naturally, and the supervisor was
interrupted deliberately. It correctly records rehearsal_interrupted. Restoration
completed at 11:06:00.995070 with timers held, R3 unchanged and no new Sportsbet
denial. This remains an interrupted observation, not a sustained pass.

Correction eebd63c3 records sanitized retry guidance from every Python error
response. Explicit retry guidance on an error creates a durable campaign-wide
source hold, honored by later requests and launches, as do 401/403/429 errors.
A bare 503 remains unavailable input. Holds do not auto-clear when their minimum
wait expires. Sportsbet's separate gate remains authoritative for its operations.
The original cumulative ledger, diagnostic history and consumed attempts remain.
Focused guidance/hold tests passed (20). The broader packaged run passed 31 and
found four old tests expecting busy-owner rejection at service admission; service
queuing now permits an active owner while actual transport still excludes it.
The tests now check nonzero completion and no transport under the busy owner,
then retain their actual denial, persistent stop and empty-capture assertions.

The joined exported-service weather-guidance regression passed under kernel
network denial. A second test run retained four failures caused by the busy-owner
probe already stopping the same test scope; ownership exclusion and actual denial
now use independent packages. Dedicated unconditional HTTP/browser fixture
markers prove exclusion, with a positive transport assertion on genuine denial.
The independent spec reviewer found no further issue after this correction.

### Compatible R3 state reader

The pinned collector observation remains commit 35b375ba. A separate compatibility
review found that installed and candidate R3 gave odds_state only 256 KiB and 4 KiB
strings, despite retaining the same embedded HTTP provenance as odds_report.
The second September 28 attempt retained a 552,067-byte state with a 59,520-byte
maximum string. The September 24 final state was 326,658 bytes / 58,680-byte string.
These exceed R3's limits although the rehearsal's explicit 2 MiB / 128 KiB limits
accept them. A rehearsal pass alone therefore did not prove configured R3 reading.

The R3-only correction applies the existing 2 MiB / 128 KiB odds provenance envelope
to odds_state. Full state/report retain 512 KiB / 4 KiB, other sources remain unchanged,
as do identity/hash checks, source timestamps and the 300-second freshness gate.
A generated-package regression failed under the old byte limit. 34 checks then
passed across actual package generation, startup and configured collector reads,
including exact limits, over-limit rejection, depth/items, startup/runtime state
growth, external-refresh divergence and absence of raw bodies in public output.
The state parametrization's tamper case still targets the external refresh; it
is a consistency test, not independent state-file tampering. Both review axes
reported no blocking findings. Installed R3 remains unchanged; deployment requires
compatible candidate code as well as new authority binding, because 5013ff03 also
lacks candidate lock-wait/deferred-lock handling. Model/config/schema bytes remain
identical. This UI-only change does not alter the running exported collector.

A further network-denied exported prediction test completed the real synthetic
capture/retention/frozen-score chain, then launched a fresh predictor process
with the same claim. The second process rejected the existing race directory,
preserved claim and terminal bytes, and left exactly one consumed verified job.
This passed in 30.18 seconds. It establishes completed-handoff nonduplication;
a crash after dispatch remains consumed and can require operator reconciliation.
The earlier controlled tests cover partial dispatch and interrupted capture too.

Live R3-limit control at 11:24:07 rejected the actual current odds state under
old limits as INVALID/INTEGRITY_FAILED while its index and authority were fresh.
The corrected-limit native read accepted that state. Neither probe started or
changed installed R3; the full lane was still waiting for its first 15-minute tick.

### Operational-only R3 authority

Preparing an enabled legacy R3 authority would require opening corpus inventories
and scorecards unrelated to this operational repair. A new exact authority schema,
`operator_ui_operational_authority_v1`, requires all existing collector, deployment
and model inputs while excluding corpus inputs entirely. The legacy full schema
keeps its complete input set. Generation and startup both reject missing collector
sources, extra corpus sources and unknown schemas. No scientific protocol or
model configuration changes. The corpus panel returns unavailable without a read.

The generated-package test failed before implementation with corpus fixture files
removed. The broad affected-module run then passed 461 checks and retained one old
odds-state 256 KiB assertion; its corrected source-specific boundary cases both pass.
Review found that nonempty unavailable corpus data violated the public API rule.
A real authenticated API route test reproduced NON_OPERATIONAL/PROVIDER_ERROR;
the empty unavailable response fixed it. All five operational binding cases pass.
The running collector package remains 35b375ba; this separate R3 package is uninstalled.

### Coverage timestamp correction

Independent measurement review found that refresh report `generated_at` is set
before discovery begins. It is the original source-age timestamp, not proof of
when each race became known. Interim `coverage_metrics.py` reports incorrectly
called overlapping schedule races "prospectively discovered". Those reports remain
retained. Corrected `coverage_metrics_v2.py` separately reports retained schedule
races with overlapping T-10 windows, consumer-index publication before each window
closed, planner readiness, attempts and verified predictions. It does not change
any original timestamp or acceptance threshold.

The broader denominator still includes Healesville Race 4: observation began at
11:20:42.698 AEST, its T-10 window closed at 11:21, and the first index publication
completed at 11:21:17.522. That race was excluded as past_or_too_close and never
indexed. Its exact discovery time is absent. It remains a startup coverage miss;
removing it would make the coverage claim misleading. Undiscovered races remain
unassessed. All later coverage uses the corrected definition.

### Evidence boundaries retained for release review

- `committed-packaged-tests.log`: exported runtime and actual subprocess boundaries,
  including empty startup, input transition, planned shutdown and genuine failure
  controls. Kernel network denial prevents a synthetic rehearsal contacting providers.
- `paired-lock-tests.log`: the actual two exported services contend for and hand
  off the same lock. `timer-overlap-replay-green.json` replays every retained sample
  from the failed first attempt against the bounded accounting correction.
- `ownership-denial-separated-tests.log` and `guidance-joined-tests.log`: actual
  guarded transport and exported weather-guidance stop behavior; busy-owner exclusion
  uses a separate package so its stop cannot masquerade as a provider-denial test.
- `restart-worker-tests.log` and `restart-completed-package.log`: interruption and
  fresh-process nonduplication. Durable consumption remains authoritative; automatic
  retry of a crashed prediction is not established or proposed.
- `transition/r3-operational-startup-v3.json`: actual configured R3 startup against
  preserved live input bytes, network denied, no database or corpus access. This
  establishes compatible startup reading, not installation or continuous panel refresh.
- `transition/OPERATING_PLAN.md` and `APPROVED_ACTIVATION_COMMAND.md`: exact pinned
  collector mode, preparation/preflight/execution commands, foreground monitoring,
  finite authority and coordinated rollback. Independent specification review found
  its activation, monitoring and scope gaps resolved. Approval remains conditional
  on the final live result and verified cleanup.

The immediate proposal excludes installing the staged R3 upgrade. Existing R3
continues to use its unchanged authority. Supervisor records and verified campaign
outputs are the operational interface for this collector mode; no campaign job
import into R3's separate operations store is claimed.

### Third attempt: released-lock handoff gap, fully restored

Pinned package `live-35b375ba` started at 11:20:42.698 AEST and failed at
12:23:37.223 on `native_integrity_or_authority_failed`. It is not a 90-minute
pass. Three predictions were verified, from three attempts, four observed-ready
T-10 races and five schedule races with overlapping T-10 windows. Race 4 was the
startup miss; Race 8 became ready but remained pending when the run stopped.
Three full cycles and 60 completed odds cycles were counted; a further odds
refresh completed before yielding its pending capture. All 64 refresh phases
took 29.83–44.53 seconds. The three predictions completed 535.76–540.14 seconds
before jump. All 142 recorded capture browser responses were HTTP 200, with no
recorder drops. No new source denial occurred.

At the failure, odds had published a fresh index and deliberately yielded to the
full daemon's live wait marker. Its report was SKIPPED_FULL_DAEMON_LOCK_HANDOFF;
its expected deferred exit code was 2. The full child was between five-second
lock polls, so the shared lock was briefly absent. The native reader supported
an active peer and a completed successful peer, but rejected this authenticated
yielding peer. The supervisor then stopped the scope, causing the waiting full
child to exit without another provider request. No producer-side ownership defect
was established. Restoration completed at 12:23:41.892. Independent restoration
verification passed with no findings: collector timers disabled, no workers or
locks, unchanged R3 PID/binding, preserved historical accounting, source OPEN.

The correction recognizes only a reciprocal handoff matching both run IDs,
collector child PIDs and service invocation IDs, with the yielding peer's fresh
original publication. Its allowance expires after the next poll plus the existing
ten-second overhead allowance, bounded by the original wait deadline. It does
not invent lock ownership or classify the skipped odds invocation as completed
collection. A focused regression failed before the correction; all 37 selected
wait/handoff checks then passed, including stale-index, wrong-recipient,
wrong-invocation, expired-gap and dead-waiter rejection.

An additional generated-package regression uses actual wrappers and a fresh
exported observer process. It reproduces the same DIVERGENT gap with a fabricated
pending T-10 capture under kernel network denial. The initial fixture also exposed
that pending checkpoints resume their original run identity; that fixture failure
is retained separately. The focused handoff scenario now ends after full-lane
progress; the existing no-pending scenario still covers reverse cooperation.
A changed package and a new observation are required. The staged 35b375ba release
proposal remains unapproved and is superseded by this failure.

### Fourth attempt: 90-minute observation completed and restored

The exported-observer handoff regression first reproduced DIVERGENT with the old
reader. The corrected actual-package cases and focused controls passed 36 checks
in 76.13 seconds; affected reader/lifecycle checks passed 317 checks in 7.09 seconds.
These ran with kernel network denial. Both review axes found no unresolved issue.
Applicable CI passed before launch. No runtime dependency installation was needed.

The unchanged live package was:

- Commit: `7bd839f0aa22d5a44dd0be8648e0f7fc1ee47d8a`.
- Tree: `f60bcdca424cf34c0f5b397bcd14ef1c1dbce1ee`.
- Package: `recovery-20260928/live-7bd839f0` under the existing campaign.
- Plan SHA256: `1ae6a6f120d9ba6021ec174ed14347c855893e35a6fd20a2d954ea68e8061836`.
- Source archive SHA256: `3ce661aaa0248de879aa9dddf420569e0dea40368fc29f836ba44e1cc69e1e17`.
- Sealed observation: 12:39:42.529578–14:09:42.529578 AEST, 28 September 2026.

Preparation generated the future canonical plan after export and runtime checks.
Preflight passed in 1.598 seconds. The prospectively recorded OPEN source allocation
covered three hours from 12:34:14, with 192 operations and unchanged source rates.
It did not clear a denial, reset a recovery counter or extend the cumulative campaign
ceiling again. The supervisor exited 0; restoration completed at 14:09:43.654073,
1.124495 seconds after the planned end. Campaign ownership closed at 14:09:43.670195.
No final drain request occurred after the last observation sample. Final workers
were reaped and ownership released. There is no failure.json for this candidate.

The supervisor retained 2,700 samples across the 90-minute contract; the final
sample ended at 14:09:41.065558. Its original status remains
REHEARSAL_MEASURED_NOT_RELEASED. Six full and 83 odds cycles completed, with 89
refresh phases and eight captures. No refresh failed or exceeded its phase budget.
Planned shutdown completed naturally. This particular run needed zero special
planned_shutdown sample overrides; the prior false-failure branch and its genuine
stale/source/process negative controls remain supported by the exported offline
rehearsal, not claimed as newly exercised live. The same distinction applies to
the specific released-lock polling gap: actual exported controls prove that repair,
while this observation proves repeated live lane cooperation and completion.

### Coverage, readiness and latency

The operational profile admits one T-10 attempt per race. Retained schedules
contained 115 distinct races, of which eight had T-10 windows overlapping this
observation. Seven windows were fully contained; Healesville Race 9 overlapped
startup. All eight were published before window closure, observed ready, attempted,
captured and independently verified through frozen prediction: 8/8 at each of those
stages. No T-10 opportunity remained pending or missed in the observed accounting.
All eight original jump times remained unchanged. Undiscovered races and capacity
beyond the selected races remain unassessed, so this is not national race coverage.

T-60 and T-30 readiness also appeared in planner evidence (nine and eight unique
windows respectively), but the existing single-T-10 profile intentionally excludes
those capture windows. Its 289 repeated non-T-10 exclusion observations are not
289 missed races. A further 22 observations skipped already-complete captures.
Twelve unique races were selected and index-eligible during the run; the four
additional races had no T-10 overlap. Ballarat Race 5's T-10 window opened at 14:12,
after observation ended. No required metadata was unavailable in the retained
selected-input observations. Index eligibility alone was not counted as a prediction.

Initial index unavailability lasted 40.002197 seconds, with first available sample
at 12:40:22.680917. The aggregate paired collector became available at 12:41:38.688553,
after 116.009833 seconds. There were no later unavailable intervals in either stream.
Authority remained available throughout. These are sampled consumer states, distinct
from source age and per-race paired-market readiness.

| Measurement | Observed range / maximum |
| --- | --- |
| Original source age | 31.917–126.923 seconds |
| Conservative adjacent-sample age bound | 128.998 seconds; unchanged 270-second supervisor / 300-second R3 limits |
| Refresh phase | 31.359–55.258 seconds |
| Source timestamp to index publication | 30.309–54.338 seconds |
| Publication operation itself | 0.0143–0.0297 seconds |
| Capture phase | 12.264–14.447 seconds |
| Retention/data preparation | 2.720–2.920 seconds |
| Machine-only history snapshot | 1.496–1.713 seconds |
| Retained feature-generation subprocess | 0.594–0.607 seconds |
| Frozen prediction subprocess | 1.050–1.167 seconds |
| Pure model inference | 0.00254–0.00301 seconds |
| Prediction operation, including retention and verification | 4.315–4.634 seconds |
| Observed price to original verification | 19.497–21.719 seconds |
| Original verified prediction lead before jump | 436.161–542.387 seconds |
| Shared-lock wait | maximum 25 seconds |
| Native sample read | maximum 0.155 seconds |

Prediction-stage terminal lead times are slightly later than the original verification
events; the lead range above uses the original events, without timestamp rewriting.
The scorer's separately named feature_generation_seconds (0.0166–0.0198 seconds)
does not replace the retained feature-generation subprocess measurement. Timer
dispatch was at most 0.0881 seconds; complete process overhead was at most 4.0946
seconds. Raw startup calendar delay remains recorded as 42.559851 seconds; only
0.030273 seconds was unblocked delay. No timer threshold was relaxed for this run.

### Source accounting and retained failures

This owner performed 89 Python Sportsbet operations and eight browser operations.
Session Python accounting recorded 13,348 started calls: 12,916 TheDogs, 89 Sportsbet,
343 Open-Meteo. Sixteen browser navigation attempts bring the campaign charge for
this session to 13,364 logical requests. Browser instrumentation recorded 421
provider requests, 853 other requests and 382 responses, all HTTP 200, with zero
recorder drops. These categories have different scopes; recorded browser responses
are not a complete wire-traffic census. Wire retries remain unmeasured; the configured
HTTP transport has zero implicit retries. No new denial, response-error record or
campaign source hold occurred. The four historical denials remain preserved.

Independent review corrected two reporting risks without changing the live package.
summarize_v3.py attributes global source operations to actual campaign ownership,
not the planned end of an earlier failed run; it reads final session counters after
drain and retains the last-sample counters separately. Generated-at is the original
refresh start, not individual discovery completion. Coverage v3 reports both actual
and planned intervals for early failures. Earlier reports are retained.

In particular, failed 35b375ba had three predictions from five overlapping schedule
races before termination, and seven over its originally planned interval. Two windows
opened after that failure. This later candidate's predictions do not backfill them.
Its actual ownership interval accounts for 64 Python and three browser operations,
9,454 Python calls, and no new denial. The earlier timer failure, interrupted weather
diagnostic, consumed market failure from September 24 and all other attempts remain
failed/interrupted/consumed as originally recorded; candidates are not pooled into
a single successful observation.

### Verified runtime and release decision

restoration-7bd839f0-final.json independently verifies original unit hashes, disabled
inactive collector timers, nonrunning services, empty cgroups, no orphan workers,
released collector/source/campaign ownership, no open lease, unchanged R3 PID 149626
and unchanged R3 binding. Collection is PAUSED; source is OPEN. Original legacy
collector triggers remain held because their source coordination is not established.
No legacy service was restarted. The verified preflight ledger prefix preserved all
47 earlier capture attempts and 25 earlier launches; its reconstruction exactly matches the
independently retained preflight receipt hash. It is labelled as a later reconstruction,
not an original snapshot. All 317 earlier source operations and four denials remain.

The odds unit retains systemd ActiveState=failed, MainPID=0 and exit status 2 from
14:09:03. The retained journal shows OPERATING_SCOPE_CLOSED/SKIPPED at 14:09:02.
This expected final admission refusal made no new request. Its failed flag has not
been cleared; verified paused cleanup does not mean every service is labelled healthy.
See final-scope-closure-7bd839f0.log. No special native-status override was needed.

Final cumulative accounting: 55 capture attempts, 62,956 logical requests, 26,316.325333
charged seconds, 414 source operations, four historical denials. Remaining campaign
allowance is 73 capture attempts, 33,044 requests and 16,883.674667 seconds. Future sessions
still need a current finite source allocation and sufficient allowance; these numbers
are not provider permission or a sustainable throughput guarantee.

Release decision: GO for user approval of this exact collector/automatic frozen
predictor release in finite, foreground-supervised 90-minute daytime sessions.
NO GO for unattended, all-day or cross-midnight operation, peak national coverage,
automatic denial recovery, or a continuously refreshed R3 monitoring claim.
The prepared transition is in transition-7bd839f0/OPERATING_PLAN.md and
APPROVED_ACTIVATION_COMMAND.md. Its four collector unit files exactly match the
tested bytes; each future session generates a new path-bound finite contract using
the same pinned code. Expired test units must never be enabled indefinitely.

The separate compatible R3 operational binding and startup are prepared and tested,
but uninstalled. Its snapshot panels age, and its separate operations store does not
import campaign prediction jobs. The immediate proposal therefore excludes the R3
upgrade. Existing installed R3 remains unchanged; verified campaign bundles and the
supervisor provide this mode's operational evidence. Model/configuration and all 41
feature-generator members remain unchanged from the September 24 candidate.

The demonstrated operating period is 12:39–14:09 AEST at this race density. The
profile selects at most four races per refresh. Same-date cleanup is enforced;
midnight rollover is unsupported. Startup time varies with timer state. Genuine
denials and explicit Retry-After produce durable holds requiring operator disposition.
Controlled interruption/restart proves nonduplication and non-reclamation; it does
not prove automatic completion of a job that crashed after dispatch. A foreground
operator and retained-evidence disk monitoring remain required (29 GB free at the
last live check). No target results, accuracy evaluation, betting or scientific
protocol changes were needed for this release decision.

Final evidence under the stated evidence root:

- final-7bd839f0-summary.json and final-7bd839f0-coverage.json: full metrics/denominators.
- final-7bd839f0-audit.json: eight complete chains, 28 independent checks each.
- restoration-7bd839f0-final.json: runtime and cumulative-history verification.
- verified-predictions-7bd839f0.md: links to each sealed prediction and original times.
- observation-7bd839f0-final.png / .svg: original source age, readiness and lead times.
- release-equivalence-7bd839f0.json: frozen model/configuration/generator equivalence.
- release-manifest-7bd839f0.json: reviewed package, proposed units, binding and rollback hashes.

The 48-file release manifest SHA256 is
`1575f738e0ed84c61a1ae7f68fec0da7343adec51e9d86711612ed5fd56ff30e`.

Permanent deployment and merge remain unperformed and require the user's approval.
