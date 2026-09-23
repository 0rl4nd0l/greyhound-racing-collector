# September 23 integration outcome: failed capture identity, restored

Executed source: `9095fbde47249b58b8d5f68b6d89e12a9250dcbf`, with startup fix
`a22dcc2f`. The actual packaged supervisor/reader/monitor/restoration passed five
isolated scenarios before execution. Existing validation was reused.

The 13:05 and 13:20 launches missed natural-quiescence admission with no candidate
acquisition or runtime mutation. Both are preserved. The final authorized relaunch
reached natural quiescence, preserved 42 consumed race/window records, and began
its fixed 13:50–15:20 AEST observation. The two permitted relaunches are exhausted.

One full-lane refresh completed in 47.692108 seconds and published four of six
selected races. Two lacked safe native identity metadata. The source guard counted
191 logical refresh calls: 141 discovery and 45 download calls have endpoint traces;
five further guarded calls lack endpoint attribution. Wire retries are unmeasured.

The single capture allowance was consumed, then `capture_reservation_identity_changed`
blocked it before browser fetch, append or receipt. The reservation used canonical
`Race 9 - MURR - 2026-09-23`; the native CSV plan used its recorded
`MURRAY-BRIDGE-STRAIGHT` alias. The guard requires exact ID equality. No claim was
reset, capture retried, race substituted or observation clock restarted.

The supervisor stopped at 13:50:54.152838 after 27 samples. Initial index
unavailability lasted until the first confirmed fresh observation at 48.187351
seconds. Three samples observed a fresh index, maximum source age 51.183276 seconds;
collector readiness remained unavailable throughout. Candidate-native authority
was fresh, but installed R3 was not rebound. Odds deferred on the shared lock; no
positive wait, successful handoff or sustained two-lane progress was proven.

Four capture windows were observed eligible: one attempted/failed and three
unattempted at abort. No unattempted window had closed by that early stop; the
unobserved remainder of the 90 minutes is unassessed. The detailed assessment maps
the attempt through immutable aliases rather than double-counting it as missed.

Two additional preparation gaps are retained. The capture child selected `uv`
because the pinned environment lacks `webdriver_manager`, downloading dependencies
outside the runtime pin and refresh request counter. The external-completion helper
cannot bind actual process-start ticks that precede systemd ExecMainStart by small
measured offsets. Neither finding authorizes weaker integrity or timing checks.

Restoration completed at **14:07:00.265184 AEST**, after the consumed 30-minute
window closed. All five original unit hashes were independently verified, both
timers are active/enabled, R3 remains PID 149626, and the supervisor has exited.
The failed claim, alias markers, prior exclusions, STOP and all launch evidence
remain retained. No permanent collector deployment, prediction, research activation
or protected-outcome access occurred. Dependency-cache downloads remain recorded.

The 90-minute proof is incomplete. Immediate repair requires consistent canonical
identity propagation across reservation and native planning, plus an actually pinned
capture child and correct measured timing-boundary binding. No further live launch
is available under this consumed authorization. Source load, capture capacity,
unattended reliability and predictive accuracy are not established.

Complete retained outcome and identities:
`/home/l4nd0/greyhound-freshness-integration-20260923-authorized/greyhound-integration-20260923-1350-launch-3/OUTCOME.md`
and sibling `terminal-assessment.json`. The task root retains both earlier failed
launches and the superseded, unstarted generic-ID preparation.
