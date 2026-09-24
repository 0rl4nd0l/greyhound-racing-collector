# Offline repair of the September 23 integration candidate

This repairs executed source `9095fbde`, whose failure is retained in outcome
commit `3bda8e45` and the immutable 13:50 launch directory. It does not resume that
run. Its MURR R9, September 23, 30-minute attempt remains consumed, including all
aliases and no-row failure evidence. Defaults and the scheduling architecture
remain unchanged. No live acquisition, installed changes, protected history,
prediction, result access or retention activation is part of this repair.

## Defects and corrections

The index and reservation used canonical `MURR`; the CSV planner used
`MURRAY-BRIDGE-STRAIGHT`. The capture planner also rebuilt the item before fetch
and again before append. Both rebuilds could lose canonical identity. The new
binding authenticates the reservation, exactly one planner row, recorded aliases,
race number/date, exact CSV/sidecar hashes, native source race ID, source URL,
jump, capture window and runner set. It then carries canonical `race_id` and
explicit `planner_race_id` through both time gates. Exact reservation equality
remains mandatory. The child receives the immutable reservation path in its
actual arguments. Append stores canonical IDs; alias-indexed receipts retain
canonical source plan/attempt/append evidence and pass the existing consumer's
hash validation. Window observations use canonical IDs without double-counting
native aliases.

The `auto` capture command selected `uv` because `webdriver_manager` was absent
from the installed interpreter. The bounded profile now selects `sys.executable`
directly. Its Chrome factory supplies the installed Chrome and ChromeDriver
paths explicitly and never enters either driver-manager installer. The generated
services, daemon and capture child verify the pinned runtime. The admitted
contract binds the runtime manifest digest. Preparation runs both actual generated
commands through the capture preflight in a minimal service environment with
kernel-denied IPv4/IPv6, imports required dependencies, and executes the pinned
browser/driver version commands. Missing dependencies and incompatible browser
major versions fail preflight. Installer subprocesses are rejected, with
`UV_OFFLINE=1` and `PIP_NO_INDEX=1` as additional controls. Legacy defaults retain
their previous command-selection behavior.

The old timing comparison assumed the kernel process birth tick could not
precede systemd ExecMainStart. A service wrapper now records a conservative kernel
birth bound, systemd invocation ID, wrapper/worker PIDs, and completion after all
owned descendants exit. Phase execution adopts and reaps even detached descendants
before releasing its lock. The external measurement binds invocation and PIDs and
charges through systemd's final exit, including startup, final output and cleanup.
It does not discard negative intervals or waive the 10-second overhead test.
Interruption stops the scope, drains children, records a failed terminal step and
charges the whole interrupted interval to its phase. It leaves the reservation spent;
a stopped scope cannot append after returning from source acquisition.

Python HTTP accounting now separates provider calls, permitted weather calls and
blocked unexpected hosts. Browser navigation counts and CDP request observations
are separate, including non-provider host counts and log errors. Neither the
24,000 logical-call ceiling nor CDP events claim a complete wire-request count:
transport retries, Chrome startup/background activity and incomplete event logs
remain explicit measurement limitations. The installer failure cannot be hidden
as a small provider request count.

## Faithful offline boundary

`tests/test_freshness_capture_e2e.py` prepares an exported source package and runs
its generated full service command. Fabricated TheDogs acquisition responses
supply a schedule and CSV/sidecar; a fabricated WebDriver supplies Sportsbet DOM
responses. Production refresh selection, metadata validation, publication, planner,
reservation, Python process launch, DOM odds parsing, time gates, SQLite append,
receipt publication and receipt consumer all execute normally. WebDriver transport
and its unused TCP-port allocation are replaced; Chrome source interaction is not
claimed tested. Kernel IPv4/IPv6 denial is inherited by all acquisition children.
All database/evidence files belong to fresh isolated fixtures.

The observed MURR alias case must append eight rows (four WIN, four PLACE) with
canonical IDs and permit verified receipt lookup by canonical and planner alias.
The real generated odds lane then refreshes successfully without acquiring a
second capture. A source response identifying R8 instead of R9 must append no
rows while preserving its spent claim. SIGTERM during source acquisition must
keep the lock until child cleanup, append nothing and preserve STOP/consumption.
Separate real-process checks cover detached-child cleanup and timeout boundaries.

The existing packaged supervisor/reader/monitor fixtures are reused for empty
startup, first publication, expected unavailability, stale and malformed evidence,
stop and restoration. Empty startup remains explicitly unavailable. Focused tests
also retain R3 deployment-binding divergence, 300-second native freshness,
request-ceiling enforcement, failed/no-row attempt persistence and legacy defaults.
No full repository suite is required for these changes.

## Fresh rehearsal to request, not execute

Prepare one new package for **September 24, 2026, 12:00–13:30 AEST**, with admission
from 11:30, unchanged natural-quiescence requirements and a fixed cleanup deadline
of 13:50. Exact commit/tree/archive/runtime/unit/plan digests are in that package's
`plan.json`, `SOURCE_IDENTITY.json`, `runtime-identity.json` and `service-preflight.json`.
The prior 13:50 launch is explicitly added to the reconciliation inventory. A new
attempt is available only after the existing quiescent reconciliation proves it;
no prior claim/window may be reset, retried, backfilled or substituted.

Retain all-minute odds activations with 15-second timer accuracy and the full
lane's OnActiveSec=15min plus OnUnitInactiveSec=15min, 30-second accuracy. Retain
80-second refresh, 50-second capture, 10-second overhead, 270-second measured
freshness target, 300-second R3 threshold, one reconciled capture, 24,000 guarded
logical HTTP calls, and the enforced single Melbourne-date operating cutoff.
The conditional yield calculation remains 90 + 75 + 80 + 10 = 255 seconds; actual
dispatch, skipped activations, lock waits, startup and cleanup must be observed.
This arithmetic is not a scheduling guarantee.

At the previous observed 191 calls per refresh, 90 nominal odds starts plus six
full starts would imply 18,336 logical calls. This is an illustrative estimate,
not an activation or load bound: active timers skip starts, capture can require
another refresh, and per-refresh discovery volume varies. The shared ceiling
stops excess guarded calls. Frequent refresh and lock occupancy may still starve
capture. One permitted acquisition cannot measure sustained capture throughput,
multi-race capacity, starvation under a busy schedule or unattended reliability.

The prepared supervisor retains two-second freshness/native-readiness samples,
startup gaps, both lanes' progress, positive handoff waits, timer trigger/start
accounting, request categories, and eligible/attempted/excluded/missed windows.
Fixed original stops and complete paired restoration apply on every exit; R3 is
measured using candidate-native readers without changing its installed binding.
A fresh approval must name this new package and window. The old authorization is
spent and provides no relaunch allowance for this package.

Remaining live assumptions are source metadata/DOM behavior, installed browser
transport compatibility, actual systemd invocation/timer/lock behavior and sustained
90-minute freshness/readiness. Offline evidence establishes the corrected boundary,
not production capacity or predictive accuracy.
