---
job_id: pr46_atomic_shadow_append_repair_20260719
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_atomic_shadow_append_repair_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/README.md
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/STATE.md
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/DECISIONS.md
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/VALIDATION.md
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/CODE_REVIEW.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/status.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/WAIT_RESULT.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/exploit.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/failure-matrix.json
  - reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719/validation.log
approval_required: true
approval_source: The owner's 2026-07-19 goal authorizes one narrow atomic-shadow-append descendant commit and normal push to existing draft PR 46 from exact head 2c595d27ac748d3df8e4031d5491c76606c5be89.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_atomic_shadow_append_repair_20260719
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_atomic_shadow_append_repair_20260719
proof_question: Can append_shadow_record publish a fully staged and fsynced JSONL replacement under a stable sidecar lock so every rejected append preserves target existence and exact bytes while every visible committed append returns APPENDED even if parent-directory fsync or cleanup fails?
hypothesis_id: pr46_atomic_shadow_append_transaction_v1
program_track: offline_development
entry_state: exact_pr46_2c595d27_draft_unmerged_blocked_on_rejected_fsync_append_publishing_bytes
target_transition: exact_pr46_one_descendant_atomic_shadow_append_ready_for_fresh_independent_review
exit_predicate: At most one normal commit whose sole parent is 2c595d27ac748d3df8e4031d5491c76606c5be89 is pushed to existing PR 46 only if staged write flush and fsync precede atomic publication under a stable sidecar lock, all pre-commit failures preserve exact target state, all post-publication failures report success without overstating durability, every requested regression and validation passes, frozen artifacts remain byte-exact, CI passes, and PR 46 remains draft and unmerged.
source_class: owner_exact_head_goal_plus_independent_fsync_failure_fixture
dataset_version: greyhound_pr46_master_63866bf4_head_2c595d27_atomic_shadow_append_20260719
evidence_hash: sha256:ac2383c01c7eb5809e83f9e00d1f842e51bb32920a6dd95dab88ac42ba740065
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: PR 46 head remains 2c595d27ac748d3df8e4031d5491c76606c5be89 with sole parent 9151789431d83bca7aa608bcc8f8889022f64464, remains open draft and unmerged, and no overlapping atomic append repair is active.
docs_impact: DOCS_UPDATED
docs_checked:
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - docs/agent_tasks/pr46_atomic_shadow_append_repair_20260719.md
docs_changed:
  - src/predictor/market_form_residual.py
  - docs/agent_tasks/pr46_atomic_shadow_append_repair_20260719.md
docs_followup: NONE
reason: The writer transaction boundary, stable lock path, atomic publication, and post-publication durability semantics are public safety behavior and are documented in the source API and this task contract.
---

# PR 46 atomic shadow append repair

Continue only the existing draft PR 46 branch from exact head
`2c595d27ac748d3df8e4031d5491c76606c5be89`, whose sole parent must remain
`9151789431d83bca7aa608bcc8f8889022f64464`. Publish at most one normal
descendant commit; do not rebase, force-push, create another PR, mark ready,
merge, deploy, or migrate data.

Implement the narrowest transactional correction to `append_shadow_record`.
Use a stable same-directory sidecar lock across full-history read, validation,
staging, publication, and cleanup. Build a complete replacement in a
same-directory temporary file, preserve existing target permissions where
applicable, flush and fsync it, and atomically publish it. The publication is
the commit point: every exception before it must preserve target existence and
exact bytes; once it succeeds, the call must return `APPENDED` even if
directory fsync or cleanup cannot confirm durability. Never claim durability
that was not obtained.

Preserve locked full-history validation, canonical/rescored identities,
`duplicate_shadow_history_identity`, `conflicting_shadow_duplicate`, one-row
`EXACT_REPLAY`, migration and malformed-history failures, exact byte
preservation, fixed outputs, and artifact hashes. Test absent/existing targets,
permissions, staged write/flush/fsync, publication, directory fsync, cleanup,
concurrency, retry, and leftover artifacts with target existence, bytes, SHA,
row-count, and retry assertions.

Use only temporary synthetic JSONL files. Do not inspect or mutate production
SQLite, live history, outcomes, callers, services, timers, runtime bindings,
artifacts, fits, migrations, dependencies, PR 47, PR 48, or unrelated code.

Task routing: `large`, high-reasoning primary model, no worker delegation
because transaction semantics, locking, failure injection, implementation,
ancestry, and release are one tightly coupled lane.
