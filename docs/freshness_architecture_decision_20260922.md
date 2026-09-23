# Decision: bounded serial collection with a complete age budget

Status: **recommended for implementation review; release held**. Bulk-discovery
investigation is closed. No further extraction, provider contact, live experiment,
deployment, timer operation, protected-history access or prediction is authorized.

## Decision and scope

Keep the existing two collector entrypoints, current per-race source path, native
publisher and shared lock. Use the existing opt-in phase coordinator in both lanes:
refresh first, at most one eligible capture, then refresh before yielding whenever
the source is too old. Release the lock between phases. Keep heavy batch processing
out of this freshness profile. Use the already prepared all-minute odds calendar
with its 15-second accuracy allowance; keep the full timer unchanged. Both lanes
must move together: an old full batch can still monopolize the shared lock.

Prepare an **80-second complete native refresh allocation**, a **90-second maximum
source age at completed yield**, and a **115-second work-admission age**, with the
derivation below. These are proposed operating limits, not measured source bounds.
Production defaults remain 65 seconds. R3 remains 300 seconds. The proposal spends
some scheduling margin to accommodate the current source; it does not convert any
previous failed attempt into success.

This is the smallest plausible current-source design because phase isolation,
early publication, two private download workers, checkpoints and the alternate
calendar already exist in this worktree. The evidence does not establish a safe
small acquisition speedup. Do not build a second pipeline or an unverified bulk
adapter. Do not remove fresh runner-page requests, change source concurrency, drop
dates/races, reuse stale schedule data, or relax metadata admission to meet timing.

The recommended design is conditional on the dependencies below. No runtime code
was changed in this reassessment: the useful new implementation is an offline
native-scheduler evaluation of the proposed allocation, not a release candidate
with an arbitrarily enlarged constant.

## Evidence boundary

Only these operational accounts and relevant source/test code inform this decision:

- [Restricted extraction](freshness_schedule_extraction_20260922.md): one guarded
  response, no released schedule projection; no source-capability conclusion.
- [Sanitized incident](freshness_boundary_incident_20260922.md): inherited exposure
  includes outcomes; a fresh context does not make affected artifacts admissible.
- [Quiescent measurement](freshness_quiescent_measurement_20260922.md): 48.250756s
  native refresh, 14 selected/attempted, 13 accepted CSVs, nine index-eligible;
  23.675s discovery and 23.147s download pool. Earlier 55.784s and 66.422s observations
  have different workloads or incomplete comparable traces. They are not a latency
  distribution. The reported 1006.503s production step is a separate batch-lock issue.
- [Release gate](freshness_release_gate_20260922.md): scheduling, terminal timing,
  consumer readiness, attempt reconciliation and two-date load remain material.

No raw traces, historical fixtures, source bodies, research artifacts or production
DB were opened. Operational numbers above are report-derived, not refreshed live.
Earlier rejected evidence and consumed diagnostic/extraction attempts stay unchanged.

## Why scheduling and locking are coupled

The ordinary `run_once` and `run_odds_capture_once` entrypoints in
[`shadow_autopilot_daemon.py`](../scripts/shadow_autopilot_daemon.py) acquire the
shared lock around an autopilot subprocess. That subprocess combines refresh,
publication and capture, and the ordinary full path can perform additional work.
Publishing early in the subprocess helps only until the packet ages out: it does
not free the other lane to refresh during the rest of a long lock hold. Timer
activations while a service is active do not queue independent workers.

[`shadow_autopilot_v1.py`](../scripts/shadow_autopilot_v1.py),
`run_autopilot`, checks `scheduled_collector_authority`; collection phases require
an owned collector lock. Its refresh commands run discovery and downloads and then
`publish_current_race_index_after_refresh`. Thus the exclusive acquisition boundary
is inherited from the orchestration/authority interface, not a requirement of HTTP.

[`live_collection_cycle.py`](../scripts/live_collection_cycle.py),
`run_live_collection_cycle`, already reduces this to one lock per refresh/capture
phase. It starts a checkpoint before acquisition, rechecks the current packet before
work, checks timing after descendants finish, and releases its own lock in `finally`.
The full wait marker gives a waiting full lane a handoff; odds defers. This limits
batch interference only if both lanes use the phase path and meet its durations.

| Operation | Actual exclusion requirement |
| --- | --- |
| Date/race-page discovery and exact-time selection | Source acquisition and private computation; no canonical DB/index write requires the global lock. Shared source admission/rate control still matters. |
| Selected CSV, metadata, native identity acquisition and validation | Can operate in a unique private generation. `refresh_prejump_upcoming.py` uses separate spawned workers and `UPCOMING_RACES_DIR`; raw exports, quarantines and sidecars are local writes, not write-free work. Existing-file reuse makes unique directories essential. |
| Canonical index, publication receipt and lifecycle update | Serialize writers and bind exact retained bytes, source report, runner identities and lifecycle. Atomic replacement of each file is not a multi-file transaction. |
| Capture selection, consumed-attempt check, claim/start marker, DB append and receipt | One coordinated admission/commit authority is necessary. SQLite transaction serialization alone does not prevent two acquisitions or preserve no-row attempt consumption. |
| Capture HTTP/browser acquisition | Does not inherently need a DB lock, but the current function combines acquisition with persistence. Unlocking it safely requires a durable reservation and a separate validated commit. |
| Shared state, checkpoint recovery, manual request claims | Serialize ownership and recovery. A dead or missing worker is not evidence that an attempt was unused. |
| Private trace/report construction | Can occur outside exclusive ownership; canonical state updates and terminal corrections still require ownership checks. |

The strict publisher in
[`synchronous_manual_capture.py`](../race_collection/synchronous_manual_capture.py)
validates retained files, normalizes/seals runner rows, replaces the packet, then
the publication record; the caller subsequently writes lifecycle evidence. Readers
can reject a transient mixed generation. That is correct rejection, but the rejected
interval must count against availability. The publisher does not itself compare
the candidate observation time with the currently published generation.

## Alternatives assessed

1. **Recommended: change the operating schedule and bounded phase allocation.**
   Maintain source acquisition, provenance, caps and concurrency. Keep capture
   bounded and refresh after it when needed. This directly removes the long-batch
   scheduling obstacle and funds realistic refresh duration within 270 seconds.
   Reordering publication earlier alone is insufficient; it already precedes capture.
2. **Stage refresh outside the global lock, then commit under it.** Technically
   eligible, but not the smallest proven correction. It reduces interference with
   capture, not the refresh's source-age cost or a long owner's publication wait.
   Two unconstrained lanes could double source fan-out and finish out of order.
   A safe version needs: one durable refresh reservation across lanes; a unique
   immutable staging directory; original request/source times; full selection and
   exclusion accounting; a bounded commit wait; under-lock revalidation of source
   age, identities and current generation; rejection of an older candidate; existing
   strict publication and lifecycle; and crash reconciliation without replaying a
   started acquisition. A losing/expired candidate remains retained, never redated.
   The current owned-lock authority check and combined publisher call must be split.
   Capture stays exclusive until its acquisition/persistence interface is separated.
   This is a larger change, with no demonstrated need if the serial envelope works.
3. **Faster discovery/download tweaks.** The measured workers were balanced and
   request time dominated. Even the optimistic discovery balancing calculation in
   the report saved about 3.7s on that one run. Neither duplicate-call removal nor a
   bulk source has been established. Increasing concurrency or pruning coverage is
   not justified by this evidence.

## Complete proposed age budget

Age is always measured from the packet's original `source_generated_at`, which the
current refresh receives before acquisition. Publication, release and completion
must not reset it. A stale predecessor keeps aging until a valid successor is
actually usable, including integrity/lifecycle evidence.

Let A=270 be the target, R the complete refresh phase, W=50 the single work phase,
G=75 the next calendar opportunity (60+15), H=10 aggregate non-phase overhead per
owner/cycle, and P=5 residual handoff polling. H includes startup, verification not
already in R, checkpoint/report writes, finalization and release; external process
exit/dispatch must be measured within these allowances, not omitted or double-counted.

For idle recurrence, finish a fresh cycle by source age R+H. Then the predecessor's
age when its successor is usable is bounded by:

`(R + H) + G + (R + H) = 2R + 95`.

Reserve an additional 15 seconds below A for this path: `2R + 95 + 15 <= 270`, so
**R <= 80**. This derives 80 from the full schedule, not from rounding 66.422 upward.
The absolute algebraic ceiling without that reserve is 87.5; it is not recommended.
The 15-second reserve is an engineering choice for review, not a statistical tail
estimate. There remains a separate 30 seconds from target 270 to R3's limit 300.

| Path / control | Proposed budget |
| --- | --- |
| Completed yield age Y | R+H = 90s; later yields require another refresh or failure |
| Idle recurrence | Y+G+R+H = 90+75+80+10 = **255s** |
| Work admission at packet age a | a+W+R+2H+P <=270, hence **a<=115s** |
| Work after a fresh phase, allowing age 90 | 90+50+80+20+5 = **245s** |
| Full wait allowance for one odds phase | max(R,W)+H+P = **95s** |
| Unchanged legacy calendar | 90+135+80+10 = **315s**: unacceptable |

The candidate's existing wait calculation derives from R, so it becomes 95 in the
test-injected profile. An intervening successful refresh resets the calculation
to its own original observation. A deferral without such a publication does not.
The full handoff marker prevents the odds owner from starting another phase ahead
of the waiting full lane; phase completion and polling costs still count.

At maximum phases, refresh then capture reaches at least source age 130; it cannot
yield at 90 or admit another 50-second task at 115. It must refresh. This makes
useful work and freshness compatible rather than spending every activation on
refresh-only retries. Overruns stop further work and remain failures; healthy
in-flight acquisitions drain under the existing cleanup policy, so no hard network
latency guarantee follows from these admission numbers.

Changing R alone is insufficient. A future implementation must carry this profile
through native phase acceptance, the child acquisition deadline, completion age,
handoff and work admission, generated timer configuration and consumer integration.
Keep 65-second historical verdicts and the unchanged 300-second reader policy.

## Availability and unresolved dependencies

The design can conditionally keep a usable index within 270 seconds under the
declared load envelope, while allowing capture progress. It cannot yet promise a
percentage uptime, every race/window, cross-midnight service or a prediction stream.

- **Latency/load:** all relevant refreshes must fit R=80 and complete source coverage
  accounting. None of the three observations establishes a tail bound, cap-16
  performance or two-date performance. Source failure/guard stops remain failures.
- **Capture capacity:** one race/window per cycle may be inadequate. At maximum
  refresh/capture/final-refresh costs, an uncontended cycle can consume 220 seconds
  plus up to 75 until the next activation: approximately one opportunity per 295s.
  This is an illustrative budgeted cadence, not guaranteed throughput or a capacity
  promise; contention, source failures and short T2 windows can make it worse.
  A fresh packet without eligible captures completed on time is not a usable service.
- **Coverage:** retain full 20–160-minute/cap-6 and odds 0–60-minute/cap-16 selectors,
  exclusions and date horizons. They replace the same packet; an odds packet does
  not preserve all full-lane races. Current task revalidation can drop work when a
  race disappears from the narrower packet. Report those losses; do not union old
  rows under a new timestamp. If the requirement is continuous 0–160 coverage, this
  design does not satisfy it: a reviewed per-row provenance/selection change would
  be required. The proposed scope preserves existing coverage semantics only.
- **Attempt integrity:** `queue_work` uses DB row evidence, which cannot prove that
  a failed/no-row capture was never attempted. Phase checkpoints retain STARTED and
  failed boundaries, but a common cross-lane/manual consumed race/window decision
  remains necessary. Before live testing, reconcile all existing accounting and
  persist a shared single-attempt reservation; output isolation cannot reset it.
- **Consumer/publication availability:** retain the prepared completion-based R3
  idle-health correction, original-source 300-second cap and running-unit authority.
  Test actual native readiness through full ownership, odds deferral and sequential
  publication/lifecycle writes. The scheduler model stubs the verified view; it does
  not establish this contract. Native packet freshness and lane health are separate.

Define availability over the entire authorized observation interval after explicit
warm-up: valid native R3 index-and-lane-ready seconds / total observed seconds.
Report missing/unverified time, longest outage and 270/300-second violations
separately, never omit failed samples. Also report discovered, selected, attempted,
CSV-accepted and index-eligible counts; every exclusion; due unconsumed race/windows,
attempted windows and completed valid captures; misses by cause; per-lane completed
cycles and maximum wait. Empty periods are distinct from usable coverage.

## Concrete preparation and offline result

Extended [the native calendar test](../tests/test_live_collection_calendar.py)
with a review-only `LiveBudget(refresh_seconds=80, completion_age_seconds=90)`
injection. Runtime defaults and all four previous timing cases are unchanged.
Added budget-boundary assertions for yield, work admission and the legacy calendar.

The modeled 51-minute timer sequence uses native entrypoints, lock files and
checkpoints, synthetic identities, mocked network/DB-facing work and a virtual
clock. At native refresh **80.0s** and capture **50s**, it completed **3 full cycles
and 10 odds cycles**, with 12 odds deferrals and 29 ignored active-service ticks.
Both lanes performed mocked captures; maximum modeled source age was **215.75s**.
At native refresh **80.25s**, neither lane completed successfully and no capture ran.
The original **65.25s against 65s** rejection remains. These are deterministic
schedule results for one ordering, not new source observations or a statistical SLO.
The mocked command can publish before the outer phase detects overrun; that modeled
publication does not mean the real strict publisher accepts an over-budget report.

Validation uses the existing collector venv, `PYTHONDONTWRITEBYTECODE=1`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, pytest `--noconftest -p no:cacheprovider`, and
only this test file. A Python audit hook rejects network connections/DNS, SQLite
connections and opens under retained artifacts/fixtures or database filenames.
No protected fixtures or actual source acquisitions are part of the test.
The calendar suite passed **9 tests in 13.49s**; the subsequently added budget
boundary test passed **1 test in 0.74s**. System `python3` lacked pytest, so validation
used the existing venv without installing anything.

## Minimum deciding experiment and approval

First complete offline integration of the proposed profile, consumed-attempt
reservation and native consumer/publication checks. Sweep relative timer starts
and handoff order, terminal overhead, capture expiry/identity changes, no-row
failure/crash recovery and overlapping publications using synthetic inputs. The
current passing calendar case is necessary evidence, not that sweep.

Then seek **fresh, bounded operational rehearsal authorization**, not another isolated
refresh. Use the existing natural-quiescence admission and native entrypoints for a
fixed **90-minute** timer-driven candidate interval, with at least **three completed
full cycles and six completed odds cycles**, observed handoff and one genuinely
eligible capture under a pre-reconciled shared **one-acquisition cap**. No substitutes
or retry after the bound attempt fails. Subsequent cycles may demonstrate refresh
and handoff but cannot initiate more captures. If no eligible window occurs, capture
availability remains undecided; do not extend until green. One capture validates
coexistence, not busy-period capture capacity.

Use isolated candidate outputs and approved ordinary capture authority. Record
complete source/publication/lock/phase/exit timestamps and immutable hashes; expose
only operational projections, never bodies or protected history. Measure the native
R3 verification/readiness contract against those outputs in the supervisor without
rebinding the installed R3. Keep the existing 2-second target / 5-second maximum
sampling gap and conservative predecessor-age bound from the release gate. Reject
missing chronology, source-time regression, >270s conservative age, non-progress,
invalid publication or out-of-budget phases. Stop further admissions on failure;
retain and naturally drain any already-started work. Freeze workload, wall-clock
ceiling, cleanup and timer restoration before authorization.

For an initial one-date claim, finish before the full lane's 21:20 Melbourne
two-date boundary with drain/restoration allowance. A general release additionally
requires equivalent two-date evidence; alternatively approve that load in the first
bounded rehearsal. Do not extrapolate a daytime pass to 24-hour availability.

Remaining approval: review the proposed schedule/allocation and scope, then authorize
the precisely prepared rehearsal (including any temporary timer operations and its
single capture). Deployment of both collector units, the odds timer and corrected
R3 package remains a separate later approval after evidence and review. Predictions,
research activation, provider contact and further bulk discovery remain excluded.
