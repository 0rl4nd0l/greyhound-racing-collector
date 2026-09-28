# October 1 prospective scheduling amendment

The user explicitly authorized bringing the wholly unconsumed programme forward
from October 5 to October 1, plus routine installation and monitoring changes.
This amendment preserves the original September 28 activation receipt, frozen
models, study rules, source controls, cumulative budgets and first-session gate.
It authorizes no model selection, betting or interim outcome reporting.

The schedule is 80 weekdays at 13:00–14:30 Australia/Melbourne, October 1, 2026
through January 20, 2027 inclusive. Admission is October 1 noon (AEST, 02:00 UTC)
to January 21 noon (AEDT, 01:00 UTC), exactly 112 local calendar days. Result
closure is February 4 noon AEDT (01:00 UTC), 14 calendar days later. On October 4
at 02:00 AEST the clock advances to 03:00 AEDT; October 1–2 sessions start at
03:00 UTC, subsequent sessions at 02:00 UTC. Preparation starts ten minutes before
each slot. Weekday slots include public holidays; selection is calendar-only.
Missed/failed slots stay consumed; there are no replacement races or makeup slots.

The current reservation authorities and documented successor locations were
rechecked before preparation. Both inactive overlapping proposals remain deferred
under the revised exclusive allocation. Original approved protected populations
remain protected; no target outcomes were decoded for this amendment.

`amend_comparison_schedule` requires zero slot claims, membership/prediction files,
queue jobs/events/requests, official-result rows and result evidence, unchanged
source bytes/campaign consumption and no holds, live leases or collector lock.
It prepares new immutable authorities, preserving all superseded files. The
campaign gains a hash-linked **date-only** amendment; budgets, initial counters,
programme identity and prediction root cannot change. The October 5-named parent
and programme identifier remain stable labels, not admission dates.

The original empty `sessions` and `results` stores remain intact. New
`sessions-october1` and `results-october1` stores bind the replacement authorities.
This is permitted only by the recorded zero-consumption proof, not a way to reset
attempts. Installation repeats that proof under all three ownership locks after
stopping the comparison timers/workers, preserves old units, replaces the six
units using their existing generator, and requires installed preflight before
arming. A failed installation leaves timers stopped and evidence intact.

The existing five-minute health worker now publishes a concise outcome-blind
status view (`--human`): next slot/release, worker/timer state, observed capture
opportunities and attempts, verified prediction counts, retained slot failures,
source age, unavailable observation samples and unverified timer intervals,
outstanding-result counts/age, source phase and disk usage. Out-of-session
intervals remain unassessed. Inactive oneshot workers are normal between ticks.
A failed slot remains an alert even after a later `NO_SLOT_DUE` cycle.

Optional notification delivery uses a separate `notification.APPROVED.json` next
to the schedule; its endpoint file must be private and HTTPS. Only service name,
structural status and classified alert codes are sent, with no race identities,
features, predictions or outcomes. Redirects are disabled, requests time out,
attempts are persisted before transport, and failed deliveries retry no faster
than every 15 minutes. Duplicate successful statuses are suppressed. No destination
was configured at preparation; local receipts/journals are **not delivered alerts**.
An authorized destination can be supplied without changing the scientific plan.

Example notification configuration after the user names/authorizes an endpoint:

```json
{"status":"AUTHORIZED_OPERATIONAL_ALERTS","authority_reference":"ACTUAL_USER_DESTINATION_DECISION","endpoint_file":"/absolute/private/endpoint-file"}
```

The endpoint file contains the complete HTTPS URL and must be mode 0600 or more
restrictive. Never put the URL/token into documentation, shell history or reports.
The generic webhook expects a 2xx acknowledgement; live delivery must be checked
with the chosen recipient before claiming notifications work.

Runtime owns scheduled preparation/launch, clean shutdown, exact-job result work
and structural monitoring without an open chat. Later admissions stay gated by
first-slot completion, a verified prediction and an authenticated retained result
for that same job. Existing lock/PID uncertainty, source denial or failed cleanup
requires operator intervention; nothing here clears those holds.

Pause admission by disabling the schedule timer and creating
`sessions-october1/PAUSE_ADMISSIONS`. Keep result and monitoring timers running to
retain already owed results. Do not restore the superseded schedule or delete
stores to recover a consumed slot. Full emergency pause may stop all comparison
timers but must preserve owed work, authorities, denials and immutable records.

## Installed and verified

Installed at September 28 **19:01:00 AEST**, with release
`551ae3c03ba8c731baeb5e41c712ecd672f02a4b` in
`/home/l4nd0/greyhound-persistent-release-october1`.
The primary operational owner confirmed no concurrent writes and received the
installed handoff. R3 binding, baseline collector units, production routing and
frozen candidate bytes are unchanged.

Current immutable control directory:
`/mnt/tenn-nvme2/tenn/greyhound-persistent-comparison-20261005/control-october1-v2-20260928`.
Authority amendment receipt SHA-256:
`71b86ebf16216ead6d29a5a1296ce038bc4a68d00e72860c6ab7eb5838f240af`.
It binds the old and new packet identities. The campaign's appended date-only
amendment is `programme-schedule-amendments/0001.json`; the original programme
file remains unchanged. Complete identities are in
[identities.json](../research/comparison_october1_evidence/identities.json).

The installed verification passed **21 checks**: all three timers enabled/active,
workers exited successfully, ownership locks released, source and campaign ledger
byte-identical to the pre-amendment proof, original approvals preserved, and zero
slots/membership/predictions/result requests. The canary remains unaccepted.
Startup briefly reported missing health while the workers initialized; this was
retained, then the monitor ran successfully after both workers published health.
An empty result-worker restart succeeded. This is not in-flight live result proof.

The exported release passed 46 focused tests with kernel networking denied;
three real synthetic prediction/result subprocess tests also passed (39.16s).
Existing live collector and packaged model equivalence evidence was reused.
No provider requests, scientific observations or target-result decoding occurred.
The initial staging packet remains preserved and uninstalled: its reservation
review serialization hash was corrected before the final packet was installed.

Status command (read-only; no provider requests or notification delivery):

```bash
~/.local/bin/greyhound-comparison-status
```

The command shows current structural worker status plus the two collector service
states. Outside a session the collector is stopped. A retained old service exit
can show `failed` while no PID is running; it is not evidence of a new failed
study session. During a session the supervisor owns collector failures/restoration;
failed or missed study slots remain visible in the monitor. Source-age `None`
means no study input has been observed, not fresh inputs.

The monitor runs every five minutes, as does the due-session scheduler. The result
worker runs every twenty minutes, retaining first eligible T+15m observations and
next-day/weekly repair work. User-systemd linger is enabled. No open chat is needed.
The next scheduled preparation window is **October 1, 12:50–12:55 AEST**.
Successful source access and official-result availability remain live acceptance
uncertainties; a failure must remain visible and keep later admission gated.

Exact pause of new admission, preserving results/monitoring and owed cleanup:

```bash
systemctl --user disable --now greyhound-comparison-schedule.timer
touch /mnt/tenn-nvme2/tenn/greyhound-persistent-comparison-20261005/sessions-october1/PAUSE_ADMISSIONS
```

Do not interrupt an active session merely to pause later admission. If required,
stop the schedule service with its existing 2,400-second graceful-stop timeout.
Never remove an ownership lock, clear a denial or reclaim a slot. Recovery from
ambiguous reboot/PID ownership requires an explicit operator disposition.

Notifications currently remain **local only**. No approved destination exists;
a destination was requested from the user. The optional transport is prepared but
no message has been sent or acknowledged. This does not block the armed first
session, but it does mean nobody receives unattended exception alerts remotely.
