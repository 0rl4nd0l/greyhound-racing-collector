# Persistent prediction and result execution: coordinated deployment preparation

Status: PREPARED, NOT AUTHORIZED OR INSTALLED. This task prepares one coordinated
proposal for PR #194. It makes no provider request, activates no service and reads
no scientific target result. The tested collector baseline remains
`8b5552c78222f9a0b5dfa1fa7bb30003bb375922`; its live evidence is reused rather than
repeating the successful 90-minute observation.

The secondary implementation owner supplies the persistent session wrapper and
durable result queue. Operational integration owns compatibility review, exact
installation/monitoring/rollback and an executable acceptance handoff. Execution
belongs to persistent software components, not to an open chat session.

## Components and operating boundaries

The proposed `run_comparison_schedule` service reads 80 explicit preannounced
90-minute slots (five per week for 16 weeks), invokes the existing
`prepare_freshness_rehearsal` and exported `run_freshness_rehearsal`, and preserves
its paired collector services, source gate, shared campaign owner and collector
lock. It is not another collector. Missed slots remain missed; retries do not
reuse consumed packages, race claims or scientific membership. Same-date cleanup
and a fixed programme endpoint remain required. The scheduler is `greyhound-comparison-schedule.service` and its five-minute
timer. Systemd invokes the pinned Python module, independent of Codex.

The proposed `greyhound-comparison-results.service` and timer invoke
`run_comparison_result_queue`, which uses the existing
`autonomous_official_result_capture` entry point for authenticated programme
members. It owns the same campaign and collector locks during provider work,
charges the shared request ledger and retains genuine denials/Retry-After holds.
It cannot run concurrently with an active prediction session. Target response
bodies, result databases and detailed logs remain private machine inputs; public
monitoring exposes structural state only. Fixed-deadline result closure performs
no evaluation and creates no outcome-reading permission.

The result timer runs every 20 minutes and persists missed timer notifications;
the queue retains individual due times and attempt limits. Timer persistence
must not mean repeated unchanged retries into denial, arbitrary catch-up
prediction sessions, or reclaimed uncertain requests.

## Verified baseline and storage

The read-only baseline receipt is
`/home/l4nd0/greyhound-collector-campaign-20260923/persistent-preparation-20260928/baseline.json`.
Both collector timers are disabled/inactive; no collector worker, collector lock,
source owner or open campaign lease exists. Installed R3 remains PID149626 with
its original service and binding. The source ledger retains539 operations and
four historical denials; campaign totals are69 consumed attempts,79,896 logical
requests and33,353.560854 charged seconds. No counter is reset by preparation.

User lingering is already enabled (`loginctl show-user l4nd0 -p Linger` gives
`Linger=yes`), so enabled user-systemd units can execute without an open Codex or
login session. Reboot safety additionally requires persistent runtime recovery;
lingering alone does not prove that boundary.

The root volume has about27 GiB available. Recent 90-minute packages occupy
376–402 MiB each; 80 comparable packages alone project29–32 GiB before future
prediction/result retention. That is a storage estimate, not a guaranteed bound.
The existing `/mnt/tenn-nvme2` volume has about602 GiB free. Proposed new programme,
session-package and private result files belong beneath
`/mnt/tenn-nvme2/tenn/greyhound-persistent-comparison-20261005`.
New programme prediction outputs also use that NVMe root, explicitly bound by
the programme authority. Historical campaign and prediction paths remain unchanged; nothing is moved,
symlinked, deleted or silently rebound. The mount must be present and its device
identity checked before any worker writes. New units require that mount, and
runtime disk checks must pause admission before exhaustion while preserving
already owed result retention where resources allow.

## Proposed finite resource envelope

These are proposed engineering allocations requiring the final approval, not
provider permissions or changes already applied. Preserve the existing effective
campaign maxima128 attempts /96,000 logical requests /43,200 seconds and append
a hash-linked programme amendment. Suggested additional limits are1,000 capture
attempts,1,280,000 prediction requests (16,000 per scheduled session),24,000 result
requests and580,800 session/cleanup seconds (80×(5,400+1,860)). This yields total
campaign ceilings1,128 attempts /1,400,000 requests /624,000 seconds. The wrapper
also enforces the80-slot programme and separate per-programme totals; old unused
allowance must not enlarge the schedule or scientific allocation.

The proposed Sportsbet allowance is at most15,360 additional operations across
80 sessions, at most192 operations and three hours per lease. Preserve10 Python
operations/minute, one browser operation/minute and two navigations/browser
operation. A new lease requires the exact approved programme, unchanged source
basis/policy, OPEN/no-active-owner state, unchanged denial history, no campaign
hold and remaining global allowance. Only an expired OPEN lease may be replaced.
STOP, denial or Retry-After never automatically clears. Every appended lease binds
the preceding source-state hash and retains all history.

The secondary result worker proposes at most1,000 races,24 attempts/race,
24,000 requests, eight races/cycle and360 seconds/cycle; bounded exact canonical
GET only, no redirect, alternate route, browser or transport retry. Its storage
ceiling is 32 GiB; session packages have a 40 GiB ceiling and prediction storage
a 20 GiB ceiling. Installation needs 100 GiB free; new prediction admission stops
below 10 GiB, while already owed result retention has a separate 2 GiB floor.

## Compatibility and recovery checks to complete

- Exact candidate includes the tested allocation and final timer-closure repairs;
  production model/configuration/feature computation stay frozen.
- One owner hierarchy: programme lock for wrapper exclusion, short campaign
  preflight ownership, then existing supervisor ownership. Never retain a parent
  campaign flock while launching a child that must acquire the same lock.
- Result work defers when prediction/campaign/collector ownership is busy. No
  polling loop creates provider traffic during contention.
- A new scheduled package is sealed only after runtime/export preparation, with
  a viable future admission window and the fixed slot endpoint. A late package
  fails before acquisition; it does not move the scientific schedule.
- Service stop and shutdown first pause timer admission, then permit existing
  collector/prediction children to drain. A short default systemd stop timeout
  must not kill the existing supervised cleanup path.
- Same-boot restart retains current PID/lifetime identity checks. Changed-boot
  recovery needs sealed boot identity, unchanged unit/package/binding hashes,
  no current worker/cgroup/lock owner, and append-only interruption records.
  No old lifecycle is rewritten as reaped, and no consumed prediction is replayed.
  The current `r3_process_changed` guard and unknown-child lifetime guard must
  have explicitly tested reboot handling or produce a concrete operator hold.
- Unknown or corrupt recovery evidence, active provider hold, missing mount,
  exhausted resource bounds or permission failure must stop new admission and
  remain visible in structural monitoring.

## Acceptance scope

The later approval request will contain one finite live acceptance, using the
installed persistent units with no open Codex session required. It must prove a
scheduled fresh package, both lanes, an eligible verified prediction, machine-only
result queuing/retention when due, restart persistence, shared-owner deferral and
planned cleanup. Offline fault tests cover reboot/PID reuse, interruption,
no-duplicate claims, budget exhaustion, denial, malformed authority and disk/mount
failure. No extra 90-minute trial is required solely because this task is new.
The proposed first scheduled slot is 5 October 2026, 13:00–14:30 Melbourne
time. Preparation runs at 12:50–12:55; a late preparation is a missed slot, not
a shifted experiment. That first regular slot is the operational canary, with
no additional repeat trial. Later slots require a verified first-slot prediction
and its authenticated result closure, plus fresh result-worker health. Prior
passed/failed evidence remains distinct. Scientific activation and result acquisition require the explicit
consolidated approval, not this preparation task.

Exact installation, monitoring, rollback and first-slot acceptance commands are
complete in [the executable handoff](persistent_execution_commands_20260928.md).
Runtime is pinned to `869fca1c66a6ee7c557facdb55e6f7592f2992cb`; the tested
collector baseline remains `8b5552c7`. Both independent review axes have zero
remaining blocking findings after the focused correction review. The actual
export passed network-denied prediction/result subprocess checks; native
systemd unit syntax and read-only host preflight passed. None of the new units
or approved authority files is installed.


## Monitoring and practical limits

`greyhound-comparison-health.service` runs every five minutes. It reads only
structural health JSON, service status, disk availability and shared source/campaign
holds. It emits a durable private monitor receipt and journal alerts. Routine
scheduler health expires after 15 minutes, result health after 45 minutes; an
active session may take 135 minutes including preparation and natural drain.
A `SESSION_RUNNING` report without an active service is an alert immediately.
No notification destination is configured: the journal and monitor receipt are
local monitoring, not a promise that an absent operator receives a message.

Fixed weekday daytime slots do not establish continuous, cross-midnight or
all-race coverage. A missed slot stays missed. Source denial requires explicit
review and a prospective disposition; elapsed backoff alone never reopens it.
Mount loss, disk pressure, corrupt authority, unresolved lifetime or stale lock
holds new work. After an interrupted reboot, the existing R3 PID and process
lifetime checks may require operator reconciliation. The proposal does not claim
automatic continuation through an unknown process lifetime, hardware outage or
source denial. Existing evidence and no-steal lock semantics take precedence.

## Pause, drain and rollback procedure

The final package supplies absolute paths for every command below. These are
post-approval procedures, not commands executed during this preparation.

1. Stop future admission with `systemctl --user disable --now
   greyhound-comparison-schedule.timer`, then create the schedule root's
   `PAUSE_ADMISSIONS` marker. Do not delete slots, source leases, prediction
   attempts, queue requests or campaign authority.
2. Let an active schedule service finish naturally. For an explicitly requested
   interruption, `systemctl --user stop greyhound-comparison-schedule.service`
   sends SIGTERM to the wrapper, which waits for the existing supervisor to
   restore. Its 2,400-second stop timeout allows the bounded drain. Verify its
   session `restored.json`, process exits, released locks, closed campaign lease,
   original paired unit hashes and unchanged R3 binding before removal.
3. Keep the result timer and health timer for already owed machine-only result
   retention until the fixed closure deadline, provided their authority and
   source controls remain valid. Stopping new predictions does not erase owed
   results. A complete emergency pause may disable both additional timers and
   naturally stop the result worker; unresolved queue members remain explicit.
4. Only after quiescence remove the six installed comparison unit files whose
   hashes exactly match the deployment manifest, then reload the user manager.
   Preserve the release checkout, approval packet, programme authority, all
   receipts and result queue. Leave both legacy collector timers disabled.
   Never reactivate a legacy unit to bypass an active source hold.
5. An unresolved stale lock or changed-boot identity is a hold, not permission
   to delete the lock or fabricate a reaped lifecycle. Use the pinned supervisor's
   `--restore-only` entry point for a valid same-boot package. If that guard fails,
   retain the diagnostic and keep admission paused pending a focused repair.
