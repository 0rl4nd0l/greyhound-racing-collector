---
job_id: early_residual_shadow_activation_v1_20260716
lane: Evaluation
supporting_lanes:
  - Evaluation
  - Provenance
  - Runtime Proof
owner: Codex
approval_required: true
approval_source: The owner said "It's not going to fix itself. Make it start its run earlier" after two prospective predictions missed their scheduled cutoffs because the shared collection lock remained continuously owned.
allow_unapproved_safe_extension: false
timeout_seconds: 7200
output_dir: reports/agent_jobs/early_residual_shadow_activation_v1_20260716
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: The new residual stage may open the configured production SQLite database only through the existing reviewed score-live mode=ro feature path. It must not write SQL, schemas, odds, results, labels, snapshots, model pointers, or registries.
live_service_mutation_allowed: true
closeout_scope: runtime
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: early_residual_shadow_activation_v1_20260716
proof_question: Can the reviewed PR 47 feature handoff be activated inside the existing odds-only lock owner so a successful strict pre-jump capture is feature-scored and appended to one fail-closed frozen residual shadow JSONL before jump, without interrupting the active daemon or changing capture windows, model parameters, production predictions, outcomes, betting, promotion, merge, or canonical DB bytes?
hypothesis_id: early_residual_scoring_inside_odds_lock_owner_v1
program_track: prospective_readiness
entry_state: reviewed_pr47_handoff_not_live_and_external_prediction_starved_by_continuous_lock_handoff
target_transition: first_live_append_only_frozen_residual_shadow_record_materialized_prejump_inside_odds_capture_cycle
exit_predicate: A clean descendant of PR 47 adds a tested early residual stage after successful odds capture; the exact generated and installed user units point to that committed worktree; daemon-reload completes without stopping or restarting the active service; the next scheduled odds-only cycle either appends an outcome-free normalized frozen residual shadow record before jump or fails closed with an exact blocker; DB bytes attributable to the residual stage and production model/service semantics remain unchanged.
source_class: strict_prejump_sportsbet_capture_plus_verified_thedogs_form_plus_pr47_hash_bound_features_no_outcomes
dataset_version: live_early_residual_shadow_activation_v1_20260716
evidence_hash: sha256:b54c9db90cde6b5d5fc686a1a5c65ae48528c739987af96bd02e50d567096497
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - RUNTIME_CHANGE
  - PUBLISH
resume_only_if: Stop if PR 47 source ancestry changes, the active daemon would need interruption, the live lock would need bypass or deletion, the DB cannot remain read-only for the residual stage, strict capture or feature provenance is incomplete, outcomes would be inspected, or scoring cannot fail closed. Merge, betting, promotion, cohort assignment, production model replacement, result writes, and timer-frequency changes remain forbidden.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: None if the installed unit and live proof match the documented early shadow-only stage.
reason: Activating append-only shadow persistence and changing the runtime workdir alter operator behavior and require durable documentation.
task_tier: critical
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: The change crosses reviewed Git ancestry, live systemd generation and installation, a shared-lock owner, read-only production data access, pre-jump provenance, append-only persistence, and runtime proof.
worker_model_allowed: false
worker_decision_limit: No delegation; the primary agent owns the live runtime, service-generation, provenance, and release gates.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/early_residual_shadow_activation_v1_20260716.md
  - scripts/predict_market_form_residual.py
  - scripts/shadow_autopilot_daemon.py
  - tests/test_predict_market_form_residual.py
  - tests/test_shadow_autopilot_daemon.py
  - docs/manual_live_market_form_residual_prediction.md
  - AGENTS.md
  - ops/systemd/shadow-autopilot.service
  - ops/systemd/shadow-autopilot.timer
  - ops/systemd/shadow-autopilot-odds-capture.service
  - ops/systemd/shadow-autopilot-odds-capture.timer
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/STATE.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/DECISIONS.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/VALIDATION.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/LIVE_PROOF.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/CODE_REVIEW.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/PR_REVIEW.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/REGRESSION_ADJUDICATION.md
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/status.json
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/validation.json
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/diff-check.json
  - reports/agent_jobs/early_residual_shadow_activation_v1_20260716/commands.log
runtime_write_paths:
  - /home/l4nd0/.config/systemd/user/shadow-autopilot.service
  - /home/l4nd0/.config/systemd/user/shadow-autopilot.timer
  - /home/l4nd0/.config/systemd/user/shadow-autopilot-odds-capture.service
  - /home/l4nd0/.config/systemd/user/shadow-autopilot-odds-capture.timer
  - /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_early_residual_*
  - /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/market_form_residual_shadow_predictions_v1.jsonl
---

# Early residual shadow activation v1

Activate the reviewed PR #47 handoff as an early stage inside the existing
odds-only daemon while that process legitimately owns the shared lock. For each
new successful strict pre-jump capture, materialize exact hash-bound features,
score the frozen full and half residual variants, and idempotently append one
outcome-free shadow record before releasing the lock.

Do not change timer frequency or capture windows. Do not interrupt, stop, or
restart the active daemon. Install only generated unit files for later runs and
use `systemctl --user daemon-reload` without starting a service manually.

Do not inspect prospective outcomes, write results or labels, change the
production DB, replace a model, promote, bet, assign a cohort cutoff, merge, or
touch PR #47's branch.
