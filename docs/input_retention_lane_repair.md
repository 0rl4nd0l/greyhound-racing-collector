# Input-retention capture-lane repair

Scope: offline correction following the consumed Wentworth Park Race 1 acceptance
(`1273521`, 17 September 2026). Its immutable closeout is
`/home/l4nd0/greyhound-input-retention-deploy-20260917/RETENTION_ONLY_CLOSEOUT.json`.
No backfill, retry, replacement race, protected-history access, model prediction,
research activation, deployment or service operation is authorized by this repair.

## Failure and actual call paths

The natural T30 capture published its exact receipt at 18:27:13 AEST through the
full daemon. No retention claim or history snapshot was created. Default-off was
restored at 18:29:41. The prior approval is consumed.

Both paths already exist in merged `89d22067`:

1. `shadow-autopilot.service` -> `shadow_autopilot_daemon.py run-once` ->
   `run_once`'s `autopilot_cycle` command -> `shadow_autopilot_v1.py run_autopilot`.
2. `shadow-autopilot-odds-capture.service` ->
   `shadow_autopilot_daemon.py run-odds-capture-once` ->
   `odds_capture_only_autopilot_command` -> the same `run_autopilot`.
3. Both use `autonomous_live_odds_capture_command` ->
   `autonomous_live_odds_capture.py main` -> `execute_capture_plan`.
4. After successful append, `publish_scheduled_capture_receipts` publishes the
   exact receipt. Only its `PUBLISHED` result calls `ScheduledInputRetention`.

The preparer's `build_package.py` hard-coded only
`write-odds-capture-service-files`, one service and one timer. It deployed that
lane to `89d22067`, and later armed only that lane's option. The full daemon
remained on `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260825-9f5c2409`
without the option. Earlier tests proved each renderer's optional argument and
the retention callback separately, not that the assembled deployment covered
every existing capture owner. This was a preparation/deployment coverage error,
not a missing second retention callback.

## Smallest correction

`scripts/prepare_input_retention_services.py` stages the two existing services
together using their existing generators. It requires both installed units,
checks their supported command shape, and preserves each lane's settings.
The only default-off command changes are the code checkout paths. Both optional
retention units receive the same absolute config path. Interpreter, history DB,
evidence root, collector lock and odds state must agree across lanes.

The tool writes a fresh staging directory only: default-off units, paired
retention-configured templates, exact original rollback units/timers, and hashes.
It never creates an approved configuration, installs units, starts services,
opens the history DB, or creates another capture scheduler. Unsupported settings,
a missing lane, an already-armed unit, or split ownership fail before output.

The shared receipt-to-retention boundary and frozen feature route are unchanged.
The existing exclusive claim directory is keyed by race ID plus the exact shared
configuration bytes' SHA-256, under that configuration's shared output root.
No lane, process or run ID enters the claim. It persists across restarts and a
failed worker; another observer cannot start a second worker. Keep configuration
bytes, output root and claims immutable throughout an authorization. Creating
a different configuration/output directory is not a permitted retry.

## Required offline integration evidence

The new fixtures enter through the generated service ExecStart arguments and
the actual daemon/autopilot parsers and orchestration, then execute real receipt
publication, retention claims, child workers, history sealing and feature
generation. Both lanes must pass. Synthetic source fetch, append response,
ownership and wall time replace external boundaries; the fixture stops directly
after capture so ordinary downstream scoring/result operations cannot run.

Adversarial duplicate/concurrent tests deliberately let both synthetic observers
reach the shared boundary despite ordinary collector serialization/deduplication.
They must produce exactly one worker claim. A new daemon invocation must preserve
that claim. Disabled, expired and absent config cases must not stat a nonexistent
history source. Worker failure must retain the ordinary APPENDED/PUBLISHED result,
record a terminal failure and reject a later observation.

These are offline fixture claims, not a new live acceptance or proof of upstream
history completeness. Existing unchanged generator/replay evidence is reused.

## Deployment scope for approval

Stage one pinned source release for **both** services. Pause both existing timers,
allow all active owners of the shared collector lock to finish, then replace both
service definitions with the paired default-off definitions. Do not kill or steal
a lock. Reload definitions and restore the timers' prior active states; leave
their bytes/schedules unchanged. Verify both loaded commands point to the pinned
source and neither has retention configured. On partial installation failure,
restore both original service definitions before resuming timers.

Rollback is the exact two prior service definitions, not one common old checkout:
the odds lane previously used `89d22067`, while the full daemon used `9f5c2409`.
The full daemon source upgrade is part of this proposed deployment and requires
explicit approval; unchanged command options do not imply unchanged full-daemon
implementation. Preserve its existing model, result-observer, state and timeout
settings. This repair adds no result or prediction authority.

Any later bounded acceptance needs a fresh approval/configuration and must arm
both paired templates together, use one output/claim root, and restore both
default-off units afterward. Deployment approval alone does not authorize that
activation or history access. No new race is selected by this repair, and the
1,000-race proposal and all scientific cohorts remain separate and unchanged.
