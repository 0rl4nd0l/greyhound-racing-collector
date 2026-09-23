# Durable Sportsbet coordination — 23 September 2026

**BLOCKED on the access basis; PR #184 stays draft and permanent rollout stays held.**
This is the continuation of `collector-integration-20260923-01a0ccec`, not a new
campaign. The engineering repair below is implemented and tested offline. No new
provider request, live launch, capture attempt or allowance was created.

## Access finding and next executable action

Scoped existing campaign records and source documentation do not establish an
applicable Sportsbet permission, explicit prohibition, or retry/access policy for
the existing NextEvents metadata and browser fixed-odds routes. The acquisition
plan's restrictions for Betfair, TAB, GRV and additional bookmakers are not treated
as a Sportsbet ban. HTTP 429 proves neither permission nor permanent prohibition.
The original response headers were not retained: whether Retry-After was supplied
remains unknown. Earlier ordinary metadata responses do not establish recovery.

The precise outstanding question is whether applicable Sportsbet conditions permit
automated pre-race collection through those two existing routes and append-only
storage, and what source-wide concurrency/retry conditions apply. An existing
terms/licence/permission record can resolve this; provider clarification is needed
only if those records cannot answer it. No inquiry was sent. The prior report's
inquiry remains a draft, now with the additional fact that both collector triggers
are paused. This is an evidence gap, not a finding of an explicit prohibition.

Next action: reconcile that route-specific access basis from an applicable record.
If permitted, update the existing gate's basis under its lock with the evidence
reference, retaining its denial history, deadline and recovery counter. Repin the
review package to the earliest suitable future window and install both guarded
units under the existing campaign and shared collector lock. The old consumed
Taree R9 T-30 remains spent. A permitted basis does not guarantee recovery; a
renewed denial stops acquisition again. No additional campaign approval is needed.

## Implemented behavior

- Both generated lane units pin the same durable state at
  `/home/l4nd0/.local/state/greyhound/sportsbet-access.json`. Their network-free
  ExecCondition and direct service wrapper reject held work before reservation.
  Python requests and browser creation also claim an exclusive source operation
  through the same OS lock. Missing/corrupt state fails closed.
- Source admission and recovery consumption are fsynced before transport. A new
  process cannot clear a cooldown, interrupted operation, or consumed recovery.
  Simultaneous admissions fail closed instead of queuing behind service deadlines.
- Sportsbet Python adapter retries are disabled and redirects are surfaced.
  Browser denials and instructed HTTP errors are observed through an independent
  CDP WebSocket even during pending WebDriver navigation. Already queued responses
  are processed before another navigation. A denial blocks source URLs and stops
  page loading; observer loss persists STOP and terminates only verified owned
  browser processes, including retained descendants after parent exit.
- Allowlisted retry headers are retained without response bodies, cookies or
  credentials. Numeric and HTTP-date Retry-After extend the deadline, including
  conservative server-clock handling. Unclassified reset guidance requires
  reconciliation rather than guessed units. 401/403 and instructed non-429 errors
  stop automatic recovery.
- When access is permitted but Retry-After is absent/unusable, **our engineering
  policy** applies: 30-minute floor, exponential fallback capped at two hours, at
  most one durably consumed recovery operation across lanes/processes. Provider
  instructions may extend beyond that cap. Recovery is one Python request or one
  browser session with at most two explicit navigations under existing lifecycle
  deadlines; browser subresources are not claimed to be one physical request.
  Success does not reset the recovery counter. Failure, inconclusive recovery,
  renewed denial or interrupted ownership requires reconciliation.
- Restoration verifies original unit bytes and unchanged R3. It leaves collector
  triggers inactive/disabled while the gate is held or the restored baseline has
  not been verified to enforce shared coordination. It records that exception
  explicitly instead of resuming legacy Sportsbet traffic.

## Offline evidence and limitations

Execution code is pinned at `7b194e4f` (later report/test-only commits do not change
it). The actual exported generated services exercise the planner, reservation,
capture subprocess, append, native receipt, monitor and restoration with fabricated
transport and kernel-denied outbound networking. Both lane commands reject
concurrent admission and survive denial/restart without consuming new windows.
Browser and Python denial cases cover both lanes. The recovery case performs valid
synthetic appends across lanes, preserves duplicate prevention, consumes exactly
one source recovery, then blocks renewed denial. No fixture grants live access.

Five packaged coordination scenarios passed with the campaign's exact recorded
interpreter path, `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/.venv/bin/python`
(125.92 seconds); the earlier equivalent-path check also passed all five. Focused tests passed (55 checks), and 16 coordination
regressions cover pending navigation, queued denial, background retry guidance,
observer loss, parent-exit cleanup, restart and deadline handling. The pinned
Selenium WebSocket adapter source matches the inspected runtime version.

Material findings corrected during this continuation: denial accounting could
skip browser cleanup; a disabled timer's expected systemctl exit 1 was treated as
an error; WebDriver polling could miss a denial during navigation; queued denial,
background retry guidance and reparented descendants needed explicit handling.
An additional attempt to package with the *legacy* autonomous collector interpreter
was rejected before launch because it lacks Flask. The campaign's recorded
interpreter is a different existing environment; no dependency installation was
performed. Failed test output is retained alongside the passing evidence.

This proves fabricated packaged behavior, not provider recovery, 90 minutes of
live progress, source capacity, or installed R3 readiness. Actual browser/provider
behavior remains a live validation gap. Installed R3 still needs explicit collector
binding reconciliation; candidate-reader readiness cannot satisfy that dependency.
Source serialization is deliberately conservative and has no burst-capacity proof.

## Runtime and unchanged campaign totals

Only `shadow-autopilot.timer` and `shadow-autopilot-odds-capture.timer` were stopped
at **17:39:21 AEST**, then disabled to preserve the hold across reboot. Their workers
drained naturally; by **17:56:57 AEST** both services were inactive with MainPID 0.
No ordinary worker was killed. Their original four unit files remain unchanged,
and R3 remains active at PID 149626. Both timers remain inactive/disabled because
the installed legacy source cannot enforce this gate. The interruption remains
open; it affects both lanes' ordinary scheduled collection. Exact previous states
(active/enabled) and unit hashes are retained for conditional restoration. No new
collector code or candidate units have been installed.

The existing durable state was seeded once with the retained 429, unknown original
observation time, a separately labelled recording time, and `access_basis=unresolved`.
Elapsed fallback time cannot lift that basis hold. Do not reinitialize this state
or re-enable unguarded legacy triggers. Permanent rollout and merge are not approved.

Campaign ledger remains **1/12 consumed attempts, 404/48,000 guarded logical
requests, 720.899931/10,800 charged live seconds**. Its SHA-256 remains
`bd372029a1f9d4c73bb25d2f264265b28b355bc1116124e159740dc4ce846aa0`.
Thus 11 attempts, 47,596 requests and 10,079.100069 seconds remain. The earlier live
run still has zero appended rows/receipts, eight observed eligible windows (one
failed/consumed, two missed, five pending at stop), and no 90-minute success.
The paused interval adds no freshly observed race/window inventory; none is inferred.

The 404 logical calls are not Sportsbet's observed total: they include other
providers and omit ordinary traffic, hidden earlier retries/redirects and complete
browser wire activity. Separate CDP request events overlap navigation counts and
must not be added as independent physical requests. No cause or scope of the rate
limit is inferred from these incomplete denominators.

Operational evidence is retained under the existing campaign's
`coordination-20260923/`, including pause/baseline, access finding, test logs, review
package and final state verification. Earlier run artifacts and attempts remain
unchanged. The permanent-rollout decision remains **hold** until the access basis,
resumed live acceptance and installed binding reconciliation are satisfied.
