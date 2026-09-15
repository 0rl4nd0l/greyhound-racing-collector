# R3 worker cleanup gate: measurement correction

Scope: offline PR #180 successor to `c052df8e`. No production worker,
JobStore, configured timeout, research protocol or runtime change.

## Requirement and original failure

`docs/operator_ui_v1/TICKETS.md`, GHU-031, requires one fixed predictor,
durable terminal status and no duplicate invocation after restart. Its validation
includes "Duplicate/restart/timeout/terminal-status tests"; it specifies no
three-second whole-job latency SLA. `CONTRACTS.md` section 7 retains terminal
timeout/cancellation, reaping and no retries/substitution.

The existing decision `DEC-GHU-035C4-REJECTED-C5-DETERMINISTIC-DEADLINE-EVIDENCE`
in `docs/operator_ui_v1/DECISIONS.md` explicitly distinguishes the same worker's
cleanup budget from whole-job time: "The rejected assertions instead included
store setup, audit transitions, persistence, and scheduling in total `run_once`
wall time." Its C6 successor requires decreasing remainders from one absolute
cleanup deadline. That deterministic blocked-close test remains unchanged.

The matrix's three-second assertion originated in `dee082e954038c9ac4bf48d48bbe3901879310b8`.
Its fixture sets a 90-second process timeout and two-second cleanup grace.
Timing all of `run_once` therefore does not measure the documented cleanup
requirement. No claim is made that the entire durable worker completes within
three seconds, or that source/read bounds guarantee whole-job latency.

The retained c052 full pytest run was **784 passed, 1 failed**. The stall case
took 3.174544 seconds and failed before its pipe-close assertions. Its retained
fixture records one attempt and durable FAILED / PROCESS_OUTPUT_INVALID with
incomplete stdout; history is not reset or reclassified as a passing run.

## CPU attribution under pytest

The original assertion also failed in `diagnostic-store-operations-c052.xml`
at 3.164615 seconds. Instrumentation surrounds real operations, excludes fixture
setup and distinguishes inclusive from nested/self timings. It does not replace
any validator or audit confirmation. Cleanup: 2.003051 seconds wall,
0.003571 seconds CPU. Whole worker: 1.140551 seconds CPU.

| Operation inside run_once | Calls | Inclusive CPU seconds |
| --- | ---: | ---: |
| JobStore `_verify_db` | 10 | 0.544048 |
| Its `_schema_valid` | 10 | 0.151522 |
| JobStore `_connect` path/separation/connection checks | 10 | 0.145140 |
| `resolve_audit_confirmation` | 8 | 0.285081 |
| Its `_validate_confirmation_preimage` | 8 | 0.223310 |
| `canonical` JSON serialization across these operations | 238 | 0.269202 |
| `_rows_hash` | 14 | 0.079477 |
| `_mutation_intent` | 4 | 0.016555 |

Rows overlap; **do not add inclusive times**. The outer calls are one initial
get, one claim and three transitions (ATTEMPT_STARTED, RESPONSE_RECORDED,
FAILED). Their nested reads total six gets. Four mutations each resolve the
exact audit proposal twice: auditor and independently validating store.
Each of ten validations rebuilds an in-memory expected schema, compares live
schema/foreign keys, validates the full store/event/attempt hash chain, and
reconstructs immutable input and transition identities. Four seals additionally
rehash all rows. These are CPU costs outside cleanup, not extra cleanup waits.

Repeated reference-schema construction is a possible optimization; validation
after mutation and dual audit confirmation also repeat work intentionally.
No measured minimal optimization is necessary to satisfy this cleanup
requirement, and no cache or skipped durability check is introduced. Earlier
standalone/profiled faster runs do not establish an environmental cause.

## Corrected evidence boundary

Keep real `run_once`, JobStore, exact audit validation and all matrix faults.
Observe real cleanup without replacing its clock, and apply the unchanged
three-second test guard to cleanup only. Keep two seconds configured cleanup
grace. Observe actual reader/closer threads, closure attempts and truthful
close failure, reaping, durable failure and refusal to consume another attempt.
An added slow-audit case demonstrates why unrelated pre-spawn audit latency
must not consume the cleanup measurement. Whole-job elapsed time is retained
as diagnostic evidence, not silently discarded.

Final exact-commit pytest, review, package and CI evidence is appended to the
host ledger `greyhound-r3-180-correction-evidence-20260915.ik8sFN/REVIEW_AND_VALIDATION.md`
and summarized on PR #180. This note grants no release or live authority.
The September proposal stays separate; 48 hours remains only a proposed
explicit observation cap, not an approved value or availability guarantee.
