# Observation authority mode boundary repair

## Objective

Bind observation release authority to the explicit result-blind runtime mode
and the exact cycle prefix ending at deferred prediction.

## Current state

`DONE_WITH_RISK`

Fresh remote `master` is
`c989b149acc06c8de727662802c1cb58eb5f0654`, tree
`c839ee74e82f4406e68a21e29de5e6fe7c2afcd2`. The launch checkout's unrelated
untracked evidence remains untouched; work is isolated in this clean sibling.
The reviewed implementation is commit
`6aa51b97bcfb2ac257867716a9dc935e890314a0`, tree
`1fffed933452f147c41b69147fc3e39ddd7fb59e`.
Draft PR #67 is open at
`https://github.com/0rl4nd0l/greyhound-racing-collector/pull/67`.

## Constraints and unsafe actions

No deployment, activation, installed service, timer, runtime input, database,
model, prediction, training, promotion, betting, live-data, or merge action is
authorized.

## Evidence used

- Owner `/goal` request dated 2026-07-26.
- Fresh `origin/master` and merged PR #65 identity.
- PR #65 task, decisions, review, validation, adapter, service, schema, and
  focused tests.

## Files touched

- `race_collection/service.py`
- `race_collection/runtime_adapters.py`
- `tests/race_collection/test_phase7_runtime_adapter.py`
- `tests/race_collection/test_phase7_operational.py`
- Two existing observation-contract documents
- Task card and report bundle

## Files intentionally not touched

- Production operational, recovery, operator, migration, schema,
  service-generation, deployment, runtime, database, model, and live-data
  paths.

## Approvals needed

None for the bounded repository repair, validation, commit, push, and draft PR.

## Blocked items and DATA_MISSING

Runtime and activation remain `DATA_MISSING` and out of scope.

## Validation status

Focused adapter, operator, operational, recovery, service-generation, schema,
formatting, lint, compile, allowlist, and diff checks are green. The fresh
read-only exact-diff review returned `SUCCESS` with no findings. The local
90-minute suite was intentionally not run; broad validation belongs to GitHub
CI on the draft PR.

## Docs impact

- docs_impact: `DOCS_UPDATED`
- docs_checked: `docs/CANONICAL_RACE_FORECASTING_PHASE7.md`,
  `docs/FORECASTING_OBSERVATION_CANARY.md`
- docs_changed: both checked files
- docs_followup: none
- reason: both documents now state the mechanically enforced authority/mode
  boundary

## Unsafe actions avoided

No deployment, activation, service-manager, runtime, database, model,
prediction, training, promotion, betting, or live-data command ran. No merge
is authorized.

## Ignored and untracked artifact note

The launch checkout's untracked `AGENTS.md` and two race-evidence inventory
directories are unrelated and remain untouched.

## Remaining risk

Broad CI is pending on draft PR #67. Observation-canary activation remains
blocked until the repair is merged and a separate activation preflight passes.
