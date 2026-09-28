# September 28 operational reliability repair

This is offline engineering against installed release
`551ae3c03ba8c731baeb5e41c712ecd672f02a4b`. The October 1 scientific
programme, models, allocation, schedule, queues and consumed attempts are not
modified. No racing/weather acquisition, target-result access or fitting is
authorized here. Existing owner session `01a0e56c-5136-7672-9af9-be7a0ecfd780`
handed off idle; preflight found no collector owner or open campaign lease.

## Scope and acceptance specification

The user's September 28 instruction requires a demonstrated startup cause,
offline reproduction through exported scheduler/supervisor/worker boundaries,
an accounting of five named exclusions, narrow repairs and a ready deployment
proposal. Preserve stale/inconsistent/failed-input rejection, source holds,
duplicate claims, planned shutdown and restoration. Use retained authorized
pre-jump material and synthetic transport only. Do not change frozen feature
semantics or scientific membership. Q Straight R6 was unattempted because the
three-forecast target was reached, and is not a missing-data case.

## Demonstrated startup cause

The original evidence is under
`/home/l4nd0/greyhound-collector-campaign-20260923/tonight-operational-20260928`.
All times below are Melbourne time (AEST).

| Time | Retained evidence |
| --- | --- |
| 19:28:13.693881 | Short operational session began: full timer 1 second, readiness 180 seconds. |
| 19:28:16.025188 | Odds cycle `20260928T192815+1000_odds_capture` began refresh 0 and subsequently yielded its queued AP R4 capture to full. |
| 19:29:21.524684 | Full captured the AP R4 quote. |
| 19:29:35.625644 | Full report completed `DAEMON_READY`. |
| 19:29:43.228129 | Independent AP R4 forecast verification completed, before the 19:37 jump. |
| 19:30:02.992761 | Resumed odds refresh 1; the queued AP R4 task was excluded as `shared_capture_allowance_consumed`. |
| 19:31:03.915746 | The same unfinished odds cycle resumed again, refresh 2. |
| 19:31:13.890119 | Sample `000090`: index and authority fresh, source age 70.895277 seconds, full `RECEIPT_READY`, odds `DATA_MISSING`. |
| 19:31:13.913649 | Supervisor rejected `native_readiness_failed`. The later refresh failure follows the stop; it is not evidence of an earlier provider denial. |

Full service wrapper/child PIDs were 3414135/3414199, invocation
`6a24c10943564f67988e5fe54dc4615f`. At rejection, odds was activating with
wrapper/child 3423341/3423401, invocation
`1fa0ddeaa36f4c219a840519c30287af`. The preceding odds invocation
`9ca596fa9aac4fd38a07b61abbfd975f` (3420042/3420101) had exited zero.
Thus successful process exit was insufficient evidence of finalized lane state.

Two producer defects combine. Active/deferred cold lanes publish reports but
no initial state; the native reader legitimately rejects the absent state.
More decisively, consuming the last queued task via the shared-attempt exclusion
returns without `finish_checkpoint()`. A successful refresh followed by that
exclusion leaves a `RUNNING` checkpoint, no completed state, and the same cycle
is resumed on subsequent timer starts. The analogous expired/too-close final
task path has the same omission. The frozen prediction can complete from the
full lane's independently valid receipt before this session-level failure.

The patch publishes the actual initial lifecycle under the existing publication
lock, preserving existing or unreadable state and another invocation's state.
It finalizes empty excluded queues under collector ownership only while source
freshness is valid. It neither makes a rejected capture successful nor refunds
an attempt. No timeout, freshness threshold, reader rejection or source hold is
relaxed.

`tests/test_scheduled_peer_handoff.py` runs exported real service wrappers,
children, locks, checkpoints and the native observer. Its cold tests exercise
both 45-minute and 90-minute packages. Real supervisor sampling/monitoring at
the existing warmup boundary rejects missing initial state, then passes with
the producer's valid state. The consumed-queue case was red with an empty queue
still `RUNNING`, and checks preserved claim bytes and finalized state after repair.
Only invented HTTP, systemd observations and monitor elapsed time are simulated.

The installed scheduler calls the same preparer with 90 minutes: full starts
after 15 minutes and readiness is checked after 20. This reduces initial
contention but does not repair the shared exclusion/finalization branch.
The installed path is exposed; this does not establish that October 1 would
necessarily fail. No scheduler interval or study admission changes are needed.

## Fixed exclusion set

| Race | Exact loss and retained evidence | Disposition |
| --- | --- | --- |
| Launceston R1, 19:40 | CSV downloaded and active runners aligned. Header explicitly said `Maiden 515m`. Grade parsed, then rejected by inconsistent `LCTN`/`LAU`/`LAUNCESTON` provenance identities; CSV quarantined as `target_metadata_not_verified:missing_target_grade`. Identity lookup separately rejected `scratched_runner_has_active_price`. | Exact grade recovered from retained header. Still unavailable: native identity conflict and unacquired weather/track evidence. |
| Q Straight R5, 19:47 | Form and seven active native runner IDs present, but native race ID absent. Odds API normalization rejected `scratched_runner_has_active_price`; exact straight venue alias already matched. | Correctly unavailable under existing native identity contract. No price contradiction is discarded, no native race ID inferred. |
| Maitland R3, 19:50 | Form/expert metadata and native race 1275997 verified. `sportsbet_venue_timezone_unmapped` and `weather_venue_not_mapped` prevented track matching and forecast acquisition. | Geographic mapping repaired. Eligible only with actual valid provider evidence; synthetic full path succeeds. No retained missing forecast invented. |
| Grafton R4, 20:00 | Same mapping failure; form/expert metadata and native race 1276355 verified. | Same repair and condition as Maitland. |
| Launceston R2, 20:06 | CSV downloaded and native race 1275685 verified. Header explicitly said `6th Grade 515m`; same grade alias rejection as R1. Once repaired, the index handoff also incorrectly compared non-idempotent legacy venue labels. Weather mapping was absent too. | Grade recovered; exact source identity used at publication; geographic mapping repaired. Synthetic full path succeeds only with complete evidence. |

The absent published form, expert form, runner timing and native IDs in the
Launceston readiness summary were downstream effects of quarantining the CSV,
not evidence that discovery or download never happened. R1's native identity
conflict remains an independent blocker. Failed identity API response bodies
were not retained by the successful-evidence-only contract; the precise bad
quote cannot be reconstructed from the retained rejection reason alone.

Grade is required by feature provenance and the two same-grade history inputs.
Runner identities and their freshness are receipt/integrity requirements.
Weather and track condition are acquisition/readiness requirements: the frozen
production scorer has 16 form features plus its frozen missingness indicators,
and does not consume those weather/track fields. Existing feature construction
can represent absent weather as null, but that does not override the installed
`--require-safe-refresh-metadata` scientific collection contract or AGENTS.md's
source-safe sidecar requirement. This patch preserves that gate. A proposal to
remove it must explicitly version the collection contract and demonstrate all
four frozen comparison inputs and downstream receipt validation unchanged;
it is not applied here. No default weather/track value is introduced.

Exact source identities now keep Q Straight, Q1 Lakeside and Q2 Parklands
distinct even though the legacy historical-feature venue mapping groups them.
That historical feature mapping is unchanged. Q Straight's actual exclusion
is not claimed as recovered by this stricter identity validation.

Recovery accounting: **0/5 fully recoverable from retained material**, **2/5
retained grade fields recovered**, **3/5 exclusion scenarios become fully
eligible through synthetic acquisition and the frozen prediction path**
(Maitland R3, Grafton R4, Launceston R2). R1 remains partially repaired; Q
Straight R5 remains correctly rejected. These counts are not live coverage
estimates and do not create retrospective pre-jump forecasts.

## Verification and evidence

Private outputs are retained under
`/home/l4nd0/greyhound-operational-reliability-output-20260928`.
`baseline.json` records installed units, release, source/campaign accounting
and 23 protected programme file hashes. Red and green logs remain separate.
`tests/fixtures/replay_operational_20260928.py` hashes the original bundles and
replays all three known forecasts from sealed features/odds through the frozen
scorer, without opening SQLite. Probabilities and ranks agree to 1e-12.
The replay emits diagnostics only and preserves original timestamps.

Focused checks cover native stale/integrity/lifecycle rejection, real packaged
capture/retention/scoring, missing metadata, source holds, duplicate prediction
claims, supervisor restoration and planned shutdown. Source transport and
child processes inherit kernel network denial. Static geographic documentation
was consulted separately; no racing or weather provider acquisition occurred.

New geographic lookup coordinates are sourced from the existing mapped venue
locations, not observed weather: [Maitland Greyhound Track](https://mapcarta.com/W1159117752),
[Grafton Greyhound Racing Club](https://mapcarta.com/N1811177458), and
[Mowbray Racecourse](https://mapcarta.com/W71883600). These describe the venues'
locations; provider matching, pre-jump timestamps and missing-value rejection
remain mandatory. The full excluded-race input set was not retrospectively
completed from these geographic references.

## Deployment boundary

An installed source update is necessary for these fixes to affect October 1.
The installed release remains 551ae3c during this task. The existing source
pin and scheduler configuration hash must be migrated together; merely
replacing service code would fail `schedule_source_changed` or
`schedule_changed`. The prepared deployment packet must preserve the old
configuration and identity bytes, every slot and attempt, all source accounting,
all frozen models and result bindings. No new schedule state directory or
parallel collector is needed. Approval is required before installing the new
runtime or making the proposed smallest supervised live check.
