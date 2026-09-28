# Persistent comparison activation: service startup observed

The operational owner executed the explicitly approved deployment on September
28, 2026. This note changes documentation only. Runtime remains pinned to
`869fca1c66a6ee7c557facdb55e6f7592f2992cb`; candidates, production routing and the
scientific window are unchanged. Promotion, betting and performance evaluation
remain excluded.

The approval receipt is
`/home/l4nd0/greyhound-collector-campaign-20260923/persistent-activation-20260928/approval.json`
(SHA-256 `dafdea7bd7fadb8d008cace8e82ad36ff2815b9813662a57cf67d4f22dbe1ef2`).
It binds the exact deployment, scientific allocation and restricted result
retention with the first-session gate intact. The earlier preparation report and
inactive packet remain historical evidence; this note supersedes their
"not installed/activated" descriptions of current service state.

## Observed startup and restart

The activation command completed successfully at **18:09:07 AEST**. All six
service/timer files were installed and the three comparison timers enabled.
The first scheduler cycle reported `NO_SLOT_DUE`; the result worker reported
`CYCLE_COMPLETE` with an empty queue and zero request attempts.

The monitor initially reported missing worker health: it ran before the workers
published their first status files. That alert was retained. The next observed
monitor timer cycle became `HEALTHY` at **18:10:13 AEST**, 64.92 seconds later,
without changing code, units or acceptance criteria.

The owner stopped and restarted the real installed result service at
**18:10:23–18:10:25 AEST**. It exited successfully, released its process, and
preserved the empty queue, zero request attempts and all structural counters.
This establishes installed-service startup/restart with no due work. It does
**not** establish recovery during an in-flight provider request or live official
result availability.

## Native timer observation completed

The owner retained **20 structural observation samples**, from **18:11:18 to
18:20:45 AEST**. The native calendar result cycle ran successfully at 18:20.
Across startup and that observation, the scheduler executed four times, the
result worker three times (including the deliberate restart), and the monitor
four times: one retained initial alert followed by three healthy observations.

The [operation-verification receipt](persistent_comparison_20260928_evidence/operation-verification.json)
reports `PASS`; all **11 named checks** are true. Source state and the campaign
ledger remained byte-identical during observation, timers remained enabled and
active, latest service exits succeeded, and the scheduler/result/campaign locks
were released. There were **zero provider request attempts, slot claims, future
predictions and fetched official results**. This is successful operation of the
installed persistent components before the approved scientific start.

Receipt SHA-256:
`f6ff252a1b7fd03b6b537bf3c64d7b7420051a5e26aa328dbe60cb07027d8c6d`.
The receipt explicitly records `first_session_gate_accepted: false`. No target
outcome or prediction-performance evidence was opened to prepare this note.

## Scientific admission remains future and gated

The first fixed slot remains **October 5, 2026, 13:00–14:30 Melbourne time**;
preparation is due 12:50–12:55. Starting the services early does not admit past
operational races or create study observations. The endpoint remains January 25,
2027 at noon, with result closure February 8 at noon.

Later slots require a successful first session, a verified pre-jump prediction
and restricted retention of the official result for that same first-slot job.
Only structural status may be released. Missing/failed canary evidence keeps
later admission closed; already owed results stay queued. No interim scoring is
authorized.

The remaining live acceptance boundary is the first scheduled collection session
and its exact-member result acquisition. Existing offline tests remain the
interruption/idempotency evidence until actual due work can be observed under the
approved scope. The [operational commands](../operations/persistent_execution_commands_20260928.md)
remain the installation, monitoring, pause and recovery reference.

The operational owner sealed the final `activation-manifest.json` in the same
evidence directory, SHA-256
`e836aca32deeb1b0ccf0e1205db0513c9badff55ac46e9a58cc519e0648ffb4d`.
Its final installed preflight passed. Runtime and the first-session gate remain
unchanged; this documentation update performed no service or provider actions.
