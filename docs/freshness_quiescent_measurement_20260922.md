# Instrumented refresh measured; release remains held

The newly authorized diagnostic acquired the real lock after natural quiescence
and ran once, from 18:59:20 to 19:00:08 AEST on September 22. The native boundary
took **48.250756 seconds**, including descendant completion. This is one timing
observation with **14 selected/attempted races, 13 accepted CSVs and 9 index-eligible
races**. It is not dependable timing, full coverage or operational acceptance.
Refresh remains 65 seconds and R3 freshness remains 300 seconds.

Decision: retain the corrected experiment admission and prepare a discovery
redesign that reduces request fan-out while preserving exact URL-bound times.
The trace does not justify a small acquisition patch as a dependable margin fix.
No unchanged live rerun, capture, overlap test or deployment follows this run.

## Evidence and preservation

New evidence directory:
`/home/l4nd0/greyhound-freshness-quiescent-diagnostic-20260922/`.
Read `result.json` together with `measurement-assessment.json`,
`admission-assessment.json`, `trace-analysis-classified.json` and `CONTRACT.md`.
The raw supervisor status `ASSESS_RETAINED_EVIDENCE` is not release acceptance.
The assessed status is **DIAGNOSTIC_MEASURED_RELEASE_HOLD**.

Verified collector/candidate base:
`64c71c568fbd3ef6fdc972e5fcb7640cc3b7c70c`.
Verified installed R3 source:
`5013ff039fda418f47a59373041d7bc7c4124f07`.
R3 remained PID 149626. All five installed service/timer hashes matched before
and after. Both timers ended active/enabled. The canonical index hash under the
owned lock was unchanged:
`79ee5549a3edf2ba5ab0011eafaae2a7a16b28fc2731758776e5d822c077d094`.
The candidate published only its isolated private nine-race packet. Its unused DB
remained absent, no prediction ran, and candidate capture attempts were zero.

The original 55.784-second and 66.422-second refresh reports remain unchanged;
their fixture hashes still match. The earlier 600-second admission failure in
`/home/l4nd0/greyhound-freshness-diagnostic-20260922-prepared/` is also retained
unchanged. Its supervisor was neither edited nor rerun. New supervision was
separately pinned. All prepared acquisition Python hashes matched before source
launch and after completion. Subsequent Python changes are confined to the
offline timing test; acquisition code remains as measured.

## What blocked the earlier admission

The old production PID 316981 was a supervising parent, sleeping while child
331938 worked. Child CPU increased across observations. The real shared lock
named PID 316981, rather than a dead or unrelated owner. Operational step metadata
records start 18:16:07.215285 and natural finish 18:32:53.718719, duration
**1006.503422 seconds**, return code zero, without timeout, against a configured
1680-second step allowance. The old 600-second drain expired during this step.
This establishes long-running, progressing work rather than a stuck sleeping
parent. It does not establish the semantic correctness of any downstream work;
no result bodies, protected histories or database records were inspected.

The new passive observation began at 18:41:10.031330. Odds PID 403171 initially
owned the lock while full PID 412412 was already service-active and waiting.
The full PID first appears as owner at 18:42:17.559, releases briefly, and owns
the lock again with payload start 18:42:28.158247. Two samples during that gap
show no lock but an active full-service PID: these were correctly refused.
Subsequent process samples show the full child advancing through acquisition
stages and later accumulating CPU. Short ordinary odds-service activations
while full holds the lock do not establish concurrent acquisition.

At **18:59:20.385899**, both services were inactive, MainPID zero, cgroups empty,
and the lock absent. This was the first admissible sample, after **1090.359
seconds** (18 minutes 10 seconds), within the 1800-second ceiling. There were
1083 samples; maximum sampling gap was 1.029269 seconds. Sampling cannot rule out
shorter unobserved gaps. The subsequent locked rechecks establish actual admission.

Installed full-timer configuration includes both OnActiveSec=15min and
OnUnitInactiveSec=15min. Restarting it after the previous attempt scheduled the
18:41 activation; assuming only completion-plus-15-minutes was incorrect.
Timer definitions were not changed. These long full-cycle lock holds are a
scheduling constraint independent of the refresh's own elapsed time.

## Admission correction, separate from acquisition

The new supervisor observes with timers running normally for at most 30 minutes.
Only a jointly idle service/cgroup/lock observation permits a brief trigger pause.
It rechecks, uses the native exclusive-create shared-lock protocol, then rechecks
service state again. Existing or partially written competitor locks fail closed;
the supervisor disables stale-lock reclamation locally for its acquisition call.
Races restore timers immediately and return to the same observation deadline.

Admission is bounded to 10 seconds with three-second systemctl calls and bounded
restoration. Actual handoff, including stopping and restoring both triggers,
was **0.033863 seconds**, with no restoration errors. Timers were restored before
the source invocation; normally triggered collectors still had to respect the
owned lock. No competing collector was explicitly started and none was killed.
The supervisor released only its own lock after native descendants completed.

Scope remained 0–60 minutes, cap 16, existing three discovery threads and two
spawned download workers, and one effective discovery date under the existing
daytime rule. This matches the failed 14-race odds refresh. The older six-race
observation used the different full-refresh window 20–160 minutes/cap 6.
No selector or concurrency setting was silently changed.

The original 1200-second native cleanup backstop and 2700-second candidate
supervision ceiling were retained, separate from passive observation. They do
not replace the 65-second rejection gate. Exclusive started/source-started markers
prevent reuse and a second source invocation. Failure/partial traces are retained.

## Measured acquisition and critical path

| Boundary | Seconds |
|---|---:|
| Native phase, including descendant completion | 48.250756 |
| Inner refresh | 46.961917 |
| Browser import / constructor | 0.015360 / 0.000150 |
| Discovery | 23.674680 |
| Selection | 0.012978 |
| Download pool, including startup and cleanup | 23.146774 |
| Sidecar validation | 0.107634 |
| Index metadata selection | 0.000720 |
| Strict private publication | 0.024893 |
| Native-minus-inner residual | 1.288839 |

These boundaries overlap. The last residual includes process startup, report
handling, publication and exit; it is not all interpreter startup. Worker-side
parsing, runner validation and persistence are measured together as non-request
residuals, not individually timed stages.

Discovery fetched one date page and **140 distinct race-page endpoints across
12 venues**, returning 140 entries. Existing venue/race/date-key deduplication
left 139; the browser returned 71 after its time filtering. Selection then had
14 in-window, 48 later and 9 past/too-close entries. All 14 selected races were
attempted and produced raw exports. One failed the existing incomplete-runner-set
check and was quarantined. Thirteen CSVs/sidecars were accepted. Strict index
metadata excluded five of the fourteen: the quarantined race, three lacking safe
weather/track metadata, and one lacking native identity. Nine were privately
published; exclusions were not hidden or relaxed.

The discovery requests occupied 58.573486 summed request seconds across three
threads, with a union of **23.406776 seconds** within the 23.674680-second phase.
The date request alone was 0.697786 seconds. Venue work is serial within each
thread assignment; Mandurah was the final venue, finishing 2.170 seconds after
the penultimate venue. The union means at least one request was active, not that
all other time was idle or that all occupancy was wire latency. Logical calls
include existing client retries and server/transport waits.

Both download workers performed seven tasks. The slower path, PID 473013, used
22.562851 seconds inside downloads: **20.762651 request seconds** and **1.800200
non-request seconds**. The other used 22.512294 / 20.620536 / 1.891758 seconds.
Initial spawn-to-browser-import delays were 0.421854 and 0.433706 seconds; first
download starts were 0.437 and 0.449 seconds after pool start. Later tasks queue
behind prior tasks on those same two workers; this is bounded busy-worker queueing,
not a new lock wait. The final worker completions differed by only 0.039488 seconds.
Pool exit after the final download added 0.139009 seconds of cleanup/residual.

| Download request family | Calls | Summed seconds across workers |
|---|---:|---:|
| Fresh primary race page | 14 | 7.142024 |
| Sportsbet metadata | 14 | 0.647447 |
| Weather fallback | 11 | 7.492469 |
| Expert Form page and export-form submission | 28 | 11.677595 |
| Export download | 14 | 7.768357 |
| Native identity page | 14 | 4.874615 |
| Native identity API | 13 | 1.780680 |

These sums are not additive wall time. All **249 logical requests** across
discovery and downloads ended with HTTP 200; this does not assert no transparent
client retries. Fifteen trace files have no unmatched spans or malformed records.
The initial aggregate classifier grouped API paths ending in `/odds` with identity
pages; the corrected derived analysis separates the 13 API calls. Both derived
outputs and all raw traces are retained; no source request was repeated.

The two Expert Form calls per race are not demonstrated duplicate acquisition:
the source uses the first response for metadata/export discovery and the second
submits export form parameters. Trace query strings are intentionally omitted.
The second calls total 1.831681 seconds; the metadata page calls total 9.845914.
The primary race pages are reacquired after discovery for fresh runner validation.
Removing them would change the observation boundary and requires separate proof.
Sportsbet sharing alone has less than 0.648 summed seconds available in this run.

## What scales, and what remains uncertain

Compared with the failed 14-race observation, startup/discovery fell from
38.919860 to 23.690611 seconds (**15.229249**), selection/downloads from 26.085599
to 23.160249 (**2.925350**). The two runs have equal selected counts but different
races and exclusions; discovery attempted 139 versus 140 race pages. A smaller
discovery request count cannot explain the faster latest discovery.

Measured now: request-path time dominates both phases; workers are balanced;
startup, metadata validation and cleanup are small. Inference: full-inventory
discovery cost grows with the date/venue/race inventory, while download cost grows
with selected races and their conditional request paths. These three observations
do not establish linear scaling, a network latency distribution, or a dependable
cap-16/two-date bound. The older uninstrumented runs cannot separate server/network
variation, retries and CPU scheduling. The fast run therefore does not resolve
the historical overrun by itself.

Even an optimistic resource calculation with this run's request durations gives
0.697786 + (58.573486 - 0.697786)/3 = **19.989686 seconds** for discovery, assuming
perfect scheduling and zero remaining processing. That is only about 3.685 seconds
below this discovery phase. It is a conditional scheduling calculation, not a
live lower bound, and cannot justify the 11.422-second improvement needed to move
the failed 66.422 run to the proposed 55-second engineering target.

## Concrete next design, prepared offline

Replace the per-race discovery time fan-out with an exhaustive date/meeting
inventory carrying **exact canonical-URL-bound jump timestamps**. Preserve the
existing full requested date horizon, aliases, source bytes/hash and original
response observation time. The interface must report every discovered URL as
resolved, ambiguous or missing; it cannot silently drop unresolved races. Existing
selection ordering/window/cap and fresh per-selected-race runner/metadata validation
then operate unchanged. Never stamp a retained schedule or runner page with a new
observation time. A fallback remains inside the same 65-second budget and reports
failure truthfully when coverage or time is insufficient.

Engineering allocation for evaluation: at most **20 seconds for discovery across
the complete requested horizon**, not per date. Combined with the failed run's
26.086-second download phase, 0.123-second metadata phase and 1.294-second outer
residual, this models approximately **47.503 seconds**, leaving about 7.5 seconds
to the 55-second margin target. This is a design target, not a performance claim
or changed gate. Rate limits and current concurrency remain requirements.

No retained input here proves that the provider offers the required complete,
exact bulk timestamp mapping. Therefore no speculative parser or stale-cache
shortcut is implemented. The smallest next evidence step is inspection of an
already-retained pre-jump date/meeting response, if one exists, for that exact
mapping; otherwise a separately authorized, fixed-scope discovery-source inspection
is needed, rather than another full refresh or drain. Any implementation must
first compare URL coverage, timestamps, selection and failure behavior offline
against per-race discovery. No new live authorization is implied by this report.

The all-minute scheduling argument remains conditional on bounded phase handoff
and a dependable <=65-second refresh. Its arithmetic (225/270 seconds in the prior
proposal) is unchanged, but long installed full-cycle lock ownership and the retained
66.422-second failure leave those premises unproven. Experiment admission succeeded;
production freshness/coexistence has not been demonstrated.

## Focused offline validation

Fourteen supervisor tests passed before execution: passive timeout, active-service
versus free-lock distinction, cgroup descendants, raced/partial locks, post-acquire
rechecks, interruption, restoration, unchanged scope, one-shot refusal and truthful
66.5-second overrun. Three interval/classification checks passed after measurement.

The new aggregate-only fixture preserves the latest report hash and timing. The
existing historical fixture was not modified. Thirteen request-timing tests passed,
including both historical replays, the new measured replay, failed-publication and
interruption accounting. A labelled stress projection repeats the new combined
discovery/startup cost plus the existing 0.5-second inter-date delay: **71.152528
seconds** before outer overhead, with all 14 selected attempts retained and native
publication rejected. This is deterministic stress evidence, not measured two-date
performance. Synthetic validation data in the replay is not coverage proof.

Command: `PYTHONDONTWRITEBYTECODE=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/.venv/bin/python -m pytest --noconftest -q tests/test_refresh_request_timing.py`.
The generic repository conftest imports the Flask app and was unavailable because
`flask_compress` is absent; the focused tests have their own network-denial fixture
and passed without that unrelated app setup. No dependencies were installed.
`git diff --check` passed. No acquisition implementation, freshness/coverage rule,
source timestamp or production setting was changed after measurement.
