---
job_id: manual_scheduled_protocol_chain_v1_20260804
lane: Forecasting Core
supporting_lanes:
  - Provenance
  - Manual Prediction
owner: Codex
mutation_mode: safe_extension
approval_required: true
approval_id: USER_PROCEED_20260804
production_data_access: false
timeout_seconds: 10800
allowed_files:
  - docs/agent_tasks/manual_scheduled_protocol_chain_v1_20260804.md
  - race_collection/manual_prediction_collector_request.py
  - scripts/predict_race_now.py
  - src/predictor/on_demand.py
  - tests/test_predict_race_now.py
---

# Scheduled receipt sealed protocol-chain repair

## Objective

Restore supported reuse of a current scheduled exact receipt while the canonical
collector lock is busy, without weakening descriptor-retained provenance or the
sealed prediction-bundle verifier.

## Boundaries

- Preserve the existing manual-request protocol-chain schema and validation.
- Add a distinct, exact scheduled-receipt chain; do not fabricate a manual
  request, claim, attempt, response, receipt, or consume lifecycle.
- Snapshot the scheduled receipt and all referenced source members using
  no-follow retained descriptors and verify identities again before return.
- Do not change capture authority, browser/lock behavior, append-only writes,
  model/features, corpus, deployment, promotion, EV, or betting behavior.
- No live runtime claim is authorized by unit/integration validation.

## Regression adjudication

- Canonical base: `38a3c992669523a0cc34d370e4039ecc658c0fdf`.
- Old fix: commit `da68ef39` and
  `test_scheduled_exact_receipt_reuses_while_capture_authority_is_busy`.
- Introducing change: commits `21e7b02e` through `90502a36` required every
  selected handoff to have the manual-request protocol chain, although scheduled
  exact receipts intentionally have their own authenticated lifecycle.
- Classification: `TRUE_REGRESSION`.
- Permanent gate: the existing scheduled-reuse test plus scheduled sealed-chain
  verifier tests added by this repair.
- Runtime functionality proof: `DATA_MISSING`; this task makes no live-runtime
  success claim and performs no collector acquisition.

## Required validation

- Preserve the red result: one failure, 62 passes in `tests/test_predict_race_now.py`.
- Run focused scheduled snapshot, scheduled reuse, and sealed verifier tests.
- Run the complete manual prediction module and sealed-bundle verifier module.
- Run fatal Ruff, `py_compile`, and `git diff --check` on changed Python files.
- Run the full forecasting suite once through exact-head CI; do not repeat it
  locally.

## Definition of done

- A valid current scheduled receipt produces a prediction-ready sealed bundle
  with an independently validated scheduled protocol chain.
- Mutation, path, hash, schema, identity, or membership mismatch still fails
  closed.
- Existing manual-request sealed bundles remain valid without schema changes.
- No file outside this card's allowlist changes.
