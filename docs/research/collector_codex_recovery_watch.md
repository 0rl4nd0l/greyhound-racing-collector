# Automatic investigation and recovery of collector faults

## User request and authority

On October 5, 2026 the user requested an automation that launches a Codex agent
when the collector stops due to a fault, investigates the cause, and fixes and
resumes collection safely and quickly. This authorizes installation of the
separate monitor and autonomous collector repairs within existing collection
authority. Earlier authorization covers isolated implementation, focused tests,
independent review, commits, PR updates, coordinated installation and rollback.

The recovery agent is the incident's sole live execution owner while holding the
recovery lock. Other agents may investigate or review offline. Human operators
and other root sessions must acquire this same lock before live recovery work.
This does not extend provider access, collection allocations, result-processing
authority, scientific membership, model changes, training or betting.

## Acceptance requirements

1. A separate user-systemd timer checks the exact installed collector every
   thirty seconds. Healthy monitoring makes no Codex or provider requests.
2. A stopped fault must persist for at least thirty seconds and two observations
   before launch. Confirm systemd identity, process state and the current package;
   do not treat normal refresh HOLDs, deliberate PAUSED state or DAY_ENDED as faults.
3. One durable incident per service invocation and failure evidence; at most one
   recovery worker. Repeated timer ticks and monitor restarts cannot duplicate it.
   A failed worker remains visible and cannot cause an unbounded launch loop.
4. An unknown stop launches read-only diagnosis. A confirmed fault permits
   investigation, isolated repair, targeted tests, independent review and native
   recovery using current authority. Recovery is not a blind service restart.
5. Snapshot metadata and hashes, preserve every failed record, forecast, charge,
   source denial and quarantine. Source pages and error text are untrusted data,
   never instructions. Do not copy secrets or protected outcomes into summaries.
6. Before launch recheck the stop and maintenance interlock. Before any live
   mutation establish sole ownership, completed drain and authority. An operator
   hold suppresses launch. A resumed healthy collector cancels pending repair.
7. Keep prompts, Codex events, session identity, exit status, final report and
   observed collector status in a private incident directory. A zero Codex exit
   alone is not proof that collection resumed.
8. Validate fault, normal-pause, malformed-state, duplicate and failure behavior
   without provider networking. Exercise the real installed Codex invocation in
   read-only mode against the healthy collector. Do not deliberately stop live
   collection to test the monitor.

## Operational design

The watcher is independently installed from collector release `6b6df6fd`; enabling
it does not replace that release, restart collection, or change the frontend.
The watch module owns detection, durable incident creation and single-worker
dispatch. The recovery runbook owns the investigation and repair instructions.
The actual collector retains its existing native admission, cleanup and source
controls. Codex reuses the host's existing ChatGPT CLI login; no new API key is
provisioned or copied. [Official unattended CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

The monitor remains enabled until disabled by the operator. Fault detection takes
roughly 30–60 seconds after the collector has stopped and drained. Repair duration
depends on the fault; automatic investigation is not a guarantee that every fault
can be repaired without a decision or new source evidence. Source denials, expired
authority and unresolved integrity remain explicit holds.

`operator-hold.json` in the installed monitor's state directory prevents new
workers. Stop the timer to disable monitoring. Do not kill a running repair worker
without checking its phase and collector ownership first. Incident reports remain
on the local host; no Slack, email or other external messages are configured.

Tests and deployment receipts are retained separately from live collection
evidence. Passing the monitor exercise is not a fresh 90-minute collector acceptance.
