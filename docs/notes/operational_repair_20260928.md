# Collector-to-prediction operational repair, 28 September 2026

This continues PR #192 and the existing campaign. The September 24 observation
remains failed at shutdown despite nine verified predictions; no prior attempt,
STOP, missed preparation, or charge is relabelled. See odds_recovery_20260924.md.

## Verified baseline

PR head c60ffb65ec1e156ee03df1122cef013aa49f500c and applicable CI passed.
The recovery worktree was clean. Installed collector files point to the original
retention release; both timers are disabled/inactive, workers and cgroups empty,
shared collector lock absent. R3 PID 149626 and binding SHA256
c39dd4bed98d6003a2e459cdb864b42cd0e6ae2eacbdf8b8c2b3af9e00e921c0 are unchanged.
Source state is OPEN with no owner, 241 operations and four retained denials.
Campaign consumption is 43 captures, 39,100 logical requests and
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
128 captures, 96,000 logical requests and 43,200 charged seconds cumulatively.
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

The pinned collector observation remains commit35b375ba. A separate compatibility
review found that installed and candidate R3 gave odds_state only256KiB and4KiB
strings, despite retaining the same embedded HTTP provenance as odds_report.
The second September28 attempt retained a552,067-byte state with a59,520-byte
maximum string. The September24 final state was326,658bytes/58,680-byte string.
These exceed R3's limits although the rehearsal's explicit2MiB/128KiB limits
accept them. A rehearsal pass alone therefore did not prove configuredR3 reading.

The R3-only correction applies the existing2MiB/128KiB odds provenance envelope
to odds_state. Full state/report retain512KiB/4KiB, other sources remain unchanged,
as do identity/hash checks, source timestamps and the300-second freshness gate.
A generated-package regression failed under the old byte limit.34 checks then
passed across actual package generation, startup and configured collector reads,
including exact limits, over-limit rejection, depth/items, startup/runtime state
growth, external-refresh divergence and absence of raw bodies in public output.
The state parametrization's tamper case still targets the external refresh; it
is a consistency test, not independent state-file tampering. Both review axes
reported no blocking findings. InstalledR3 remains unchanged; deployment requires
compatible candidate code as well as new authority binding, because5013ff03 also
lacks candidate lock-wait/deferred-lock handling. Model/config/schema bytes remain
identical. This UI-only change does not alter the running exported collector.

A further network-denied exported prediction test completed the real synthetic
capture/retention/frozen-score chain, then launched a fresh predictor process
with the same claim. The second process rejected the existing race directory,
preserved claim and terminal bytes, and left exactly one consumed verified job.
This passed in30.18seconds. It establishes completed-handoff nonduplication;
a crash after dispatch remains consumed and can require operator reconciliation.
The earlier controlled tests cover partial dispatch and interrupted capture too.

Live R3-limit control at11:24:07 rejected the actual current odds state under
old limits as INVALID/INTEGRITY_FAILED while its index and authority were fresh.
The corrected-limit native read accepted that state. Neither probe started or
changed installedR3; the full lane was still waiting for its first15-minute tick.

### Operational-only R3 authority

Preparing an enabled legacy R3 authority would require opening corpus inventories
and scorecards unrelated to this operational repair. A new exact authority schema,
`operator_ui_operational_authority_v1`, requires all existing collector, deployment
and model inputs while excluding corpus inputs entirely. The legacy full schema
keeps its complete input set. Generation and startup both reject missing collector
sources, extra corpus sources and unknown schemas. No scientific protocol or
model configuration changes. The corpus panel returns unavailable without a read.

The generated-package test failed before implementation with corpus fixture files
removed. The broad affected-module run then passed461checks and retained one old
odds-state256KiB assertion; its corrected source-specific boundary cases both pass.
Review found that nonempty unavailable corpus data violated the public API rule.
A real authenticated API route test reproduced NON_OPERATIONAL/PROVIDER_ERROR;
the empty unavailable response fixed it. All five operational binding cases pass.
The running collector package remains35b375ba; this separate R3 package is uninstalled.
