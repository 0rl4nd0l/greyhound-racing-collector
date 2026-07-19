---
job_id: odds_capture_latency_v1_20260719
lane: Evaluation
supporting_lanes:
  - Runtime Proof
  - Provenance
owner: Codex
approval_required: true
approval_source: The owner approved the decision-complete PC latency remediation plan and requested implementation.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/odds_capture_latency_v1_20260719
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Tests may use copied or synthetic SQLite and evidence fixtures only. No production database, runtime evidence, service, timer, lock, model pointer, or installed unit may be mutated.
live_service_mutation_allowed: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: odds_capture_latency_v1_20260719
proof_question: Can the odds-only cycle avoid full reporting scans, repeated history loads, and repeated browser startups while preserving strict pre-jump capture, append-only odds, exact provenance, early residual scoring before lock release, and the full daemon contract?
hypothesis_id: explicit_lightweight_odds_child_and_cycle_scoped_reuse_v1
program_track: offline_development
entry_state: live odds-only cycles run the full reporting tail and repeat database and browser setup per captured race
target_transition: one reviewed stacked implementation provides a separate lightweight child contract, one history load and browser session per batch, and durable low-priority service controls without changing capture semantics
exit_predicate: Focused differential and contract tests pass; odds-only mode performs zero global reporting builders, at most one history load and one browser setup absent recovery, preserves per-race outcomes and lock order, and full mode remains unchanged.
source_class: pr48_exact_head_plus_synthetic_and_copied_no_write_fixtures
dataset_version: pr48_f776bfd142b1e8acd3befca330eee36f490402ed_20260719
evidence_hash: sha256:825ff52e31f72afd1c653e7348e80adc588f19cb517a5728b65fe0b670a47d44
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
resume_only_if: Stop if PR 48 head changes, another active task overlaps an allowed code path, production data or runtime mutation becomes necessary, capture timing or runner validation would be weakened, early residual scoring cannot remain before lock release, or full-mode artifacts would become optional.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: None if the implementation and generated unit contract match the updated operator documentation.
reason: The change adds CLI and artifact contracts, batching behavior, browser lifecycle behavior, and service resource controls.
task_tier: large
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: The change spans a time-sensitive browser collector, append-only persistence, hash-bound feature provenance, shared-lock ordering, process batching, and durable service generation.
worker_model_allowed: false
worker_decision_limit: No delegation; one agent owns the coupled capture, feature, scoring, service, and regression-test surfaces.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/odds_capture_latency_v1_20260719.md
  - scripts/shadow_autopilot_v1.py
  - scripts/shadow_autopilot_daemon.py
  - scripts/run_shadow_non_tgr_rf_evaluation.py
  - scripts/predict_market_form_residual.py
  - scripts/autonomous_live_odds_capture.py
  - odds_auto_integrator.py
  - tests/test_shadow_autopilot_v1.py
  - tests/test_shadow_autopilot_daemon.py
  - tests/test_run_shadow_non_tgr_rf_evaluation.py
  - tests/test_predict_market_form_residual.py
  - tests/test_autonomous_live_odds_capture.py
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
  - ops/systemd/shadow-autopilot-odds-capture.service
  - reports/agent_jobs/odds_capture_latency_v1_20260719/STATE.md
  - reports/agent_jobs/odds_capture_latency_v1_20260719/DECISIONS.md
  - reports/agent_jobs/odds_capture_latency_v1_20260719/VALIDATION.md
  - reports/agent_jobs/odds_capture_latency_v1_20260719/RUNTIME_PROOF.md
  - reports/agent_jobs/odds_capture_latency_v1_20260719/CODE_REVIEW.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/RUN_OUTCOME.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/DECISION_ENTRY.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/status.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/validation.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/diff-check.json
  - reports/agent_jobs/odds_capture_latency_v1_20260719/commands.log
---

# Odds-capture latency remediation v1

Implement the approved lightweight odds-child, cycle-scoped feature/history
batching, browser-session reuse, and durable service resource controls on a new
branch from PR #48 exact head.

## Hard boundaries

- Preserve the 16-race cap, timer frequency, capture windows, source fallback,
  WIN and PLACE capture, runner and URL validation, append-only odds writes,
  shared lock, and early residual stage before lock release.
- Keep the full daemon output contract unchanged.
- Do not inspect outcomes, train, promote, bet, activate, migrate the database,
  touch production data, install units, reload systemd, restart services, push,
  merge, or mutate GitHub.
- Use synthetic or copied no-write fixtures for validation.

## Required validation

- RED/GREEN tests for the lightweight contract and zero heavy builders.
- Differential tests for per-race feature and residual results.
- Tests proving one DB history load and one browser setup per batch, with
  restart-on-failure and per-race isolation.
- Tests proving early residual work completes before lock release.
- Tests proving the full output contract and existing single-race CLIs remain
  compatible.
- Generated systemd service assertions and `git diff --check`.
