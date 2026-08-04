---
job_id: operator_ui_invalid_collector_suppression_v1_20260804
lane: Operator UI
owner: Codex
mutation_mode: safe_extension
approval_required: true
approval_id: USER_PROCEED_20260804
production_data_access: false
timeout_seconds: 7200
allowed_files:
  - docs/agent_tasks/operator_ui_invalid_collector_suppression_v1_20260804.md
  - src/operator_ui/api.py
  - tests/operator_ui/test_api.py
---

# Operator UI invalid collector suppression repair

## Objective

Restore the fail-closed API contract that rejects a non-empty collector payload
when its aggregate evidence is `INVALID/INTEGRITY_FAILED`.

## Boundaries

- Do not change collector, runtime, deployment, model, corpus, prediction, or
  betting behavior.
- Do not weaken validation or remove the existing regression test.
- Keep the finite empty invalid response for genuinely empty provider data.

## Regression adjudication

- Canonical base: `38a3c992669523a0cc34d370e4039ecc658c0fdf`.
- Old contract and permanent gate: commit `d5e1c986`,
  `test_invalid_collector_payload_remains_suppressed`.
- Introducing change: commit `271f0026` allowed finite invalid collector
  resources to return `{}` even when the provider supplied non-empty data.
- Classification: `TRUE_REGRESSION`.
- Runtime functionality proof: not required; this is a pure read-API safety
  repair with no live-state mutation or runtime claim.

## Required validation

- Run the failing regression test before and after the patch.
- Run `tests/operator_ui/test_api.py` under an owner-controlled `TMPDIR` whose
  ancestry is not group/world writable.
- Run fatal Ruff, `py_compile`, and `git diff --check` for the changed Python
  surface.

## Definition of done

- Non-empty invalid collector data produces the audited provider-error path.
- Empty invalid collector data retains the finite empty response.
- No files outside this card's allowlist change.
