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
and a fixed programme endpoint remain required. The final pinned unit names and
commands will be supplied by the implementation package before approval.

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
ceiling and free-space reserve will be reconciled with the selected volume and
prediction retention before sealing the deployment package.

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
The smallest live duration and exact date will be fixed once the new package and
natural opportunity requirements are reviewed; prior passed/failed evidence will
remain distinct. Scientific activation and result acquisition require the explicit
consolidated approval, not this preparation task.

Installation, monitoring, rollback and acceptance commands are completed against
the eventual pinned package before this document is presented for approval.
