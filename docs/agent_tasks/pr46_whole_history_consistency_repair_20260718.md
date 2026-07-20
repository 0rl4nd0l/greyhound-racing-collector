---
job_id: pr46_whole_history_consistency_repair_20260718
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_whole_history_consistency_repair_20260718.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/STATE.md
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/VALIDATION.md
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/CODE_REVIEW.json
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/status.json
  - reports/agent_jobs/pr46_whole_history_consistency_repair_20260718/WAIT_RESULT.json
approval_required: true
approval_source: The owner's 2026-07-18 request authorizes one narrow whole-history consistency repair commit and normal push to existing draft PR 46 from exact head 9151789431d83bca7aa608bcc8f8889022f64464.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_whole_history_consistency_repair_20260718
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_whole_history_consistency_repair_20260718
proof_question: Can append_shadow_record reject every repeated stable identity across otherwise valid v2 history before append or replay while preserving the exact public v2, migration, frozen-model, serialization, and no-runtime boundaries?
hypothesis_id: pr46_whole_history_stable_identity_uniqueness_v1
program_track: offline_development
entry_state: exact_pr46_91517894_draft_unmerged_blocked_on_cross_row_history_consistency
target_transition: exact_pr46_one_descendant_whole_history_consistency_repair_ready_for_fresh_independent_review
exit_predicate: At most one normal commit whose sole parent is 9151789431d83bca7aa608bcc8f8889022f64464 is pushed to the existing PR 46 branch only if locked whole-history stable-identity uniqueness, candidate conflict/replay semantics, byte-preserving rejection, every requested regression and validation gate, unchanged artifacts, exact ancestry, and GitHub CI all pass while the PR remains draft and unmerged.
source_class: exact_pr46_operational_integrity_v2_bundle_plus_independent_cross_row_failure_fixture_and_review
dataset_version: greyhound_pr46_master_c1dfd464_head_91517894_whole_history_consistency_repair_20260718
evidence_hash: sha256:d7cfff72746766c3fe02e74dd29cdb35470d14115144938f6eb622d8ce46c6af
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: PR 46 head remains 9151789431d83bca7aa608bcc8f8889022f64464, live master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, and no overlapping append_shadow_record whole-history repair is active.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
docs_changed:
  - src/predictor/market_form_residual.py
  - docs/agent_tasks/pr46_whole_history_consistency_repair_20260718.md
docs_followup: NONE
reason: The writer's deterministic rejection semantics for repeated valid history identities change and are documented at the source API and task-contract surfaces.
---

# PR 46 whole-history stable-identity consistency repair

Continue only the existing draft PR 46 branch from exact head
`9151789431d83bca7aa608bcc8f8889022f64464`. Publish at most one normal
descendant commit; do not rebase, force-push, create another PR, mark ready,
merge, deploy, or migrate data.

Inside the existing `append_shadow_record` file lock, fully validate every
historical v2 row with the current canonical, checksum, provenance, and frozen
model rescoring rules, then enforce uniqueness of the stable duplicate identity
across the complete validated history. Repeated prior identities must reject
before candidate comparison whether their canonical bytes are identical or
different. A candidate may replay only one identical prior row, conflicts with
one prior identity must reject, and a new identity may append only over valid,
unique history. Every rejection must preserve file bytes.

Preserve `history_migration_required` for v1, mixed, and insufficient-input
history and every malformed, truncated, invalid-UTF8, noncanonical, unsupported,
checksum, provenance, rescoring, deterministic-serialization, deep-freeze,
fixed-output, model-hash, manifest-hash, runtime, and migration boundary already
approved at `91517894`. Do not add signing or authentication requirements.

Use only temporary synthetic fixtures and an isolated validation environment.
Do not read production SQLite, live history, or outcomes; do not refit, adapt a
runtime caller, touch services/timers, deploy, activate, or migrate anything.

Task routing: `large`, high-reasoning primary model, no worker delegation because
the lock scope, history state machine, error semantics, regressions, ancestry,
and release are one coupled lane.
