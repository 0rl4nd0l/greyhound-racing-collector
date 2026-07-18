---
job_id: pr46_operational_integrity_v2_20260718
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_operational_integrity_v2_20260718.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/STATE.md
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/VALIDATION.md
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/CODE_REVIEW.json
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/status.json
  - reports/agent_jobs/pr46_operational_integrity_v2_20260718/WAIT_RESULT.json
approval_required: true
approval_source: The owner's 2026-07-18 request authorizes one narrow operational-integrity descendant commit and push to existing draft PR 46 from exact head f57c78000a5ba9565f5dd9bd518387c1ee92ff7f under the explicitly bounded non-malicious-host threat model.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_operational_integrity_v2_20260718
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_operational_integrity_v2_20260718
proof_question: Can exact PR 46 head f57c7800 receive one deterministic v2 JSONL contract that re-scores complete stored inputs, validates every row, detects corruption and inconsistent construction, rejects legacy history, and preserves append/replay without claiming protection against coordinated host-level rewriting?
hypothesis_id: pr46_owner_approved_operational_integrity_v2
program_track: offline_development
entry_state: exact_pr46_f57c7800_draft_unmerged_blocked_by_impossible_hostile_rewrite_gate
target_transition: exact_pr46_one_descendant_operational_integrity_v2_ready_for_fresh_independent_review
exit_predicate: One normal commit whose sole parent is f57c7800 is pushed to existing PR 46 only if v2 rows contain complete canonical score inputs and provenance, writer-side rescoring and full-content identity checks pass all requested regressions and validations, frozen artifacts remain byte-exact, and PR 46 remains draft and unmerged.
source_class: owner_approved_operational_integrity_boundary_plus_completed_pr46_review_and_blocked_repair_packets
dataset_version: greyhound_pr46_base_c1dfd464_head_f57c7800_operational_integrity_contract_20260718
evidence_hash: sha256:9a395209e8ea3415292f8c42daff2259b392bb355e035c7d074e055b2d6295eb
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: PR 46 head remains f57c78000a5ba9565f5dd9bd518387c1ee92ff7f before commit, master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, and no overlapping scorer/writer integrity repair is active.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
docs_changed:
  - src/predictor/market_form_residual.py
  - docs/agent_tasks/pr46_operational_integrity_v2_20260718.md
docs_followup: NONE
reason: The writer's public JSONL schema, validation behavior, migration error, and checksum threat boundary change and are documented at the source API and task-contract surfaces.
---

# PR 46 owner-approved operational-integrity v2

Continue only the existing PR 46 author branch from exact head
`f57c78000a5ba9565f5dd9bd518387c1ee92ff7f`. Publish at most one normal
descendant commit; do not rebase, force-push, create another PR, mark ready, or
merge.

Implement the smallest deterministic v2 history contract in
`market_form_residual.py` and its focused tests. Every row must carry complete
canonical score-affecting runner inputs and provenance, be re-scored with the
frozen model before write or replay, and bind full canonical content to its
checksum and record identity. Existing history must fail closed on malformed,
truncated, noncanonical, unsupported, inconsistent, v1, mixed-version, or
insufficient-input content. V1, mixed-version, and insufficient-input history
must raise exactly `history_migration_required`; no inference, upgrade, rewrite,
or migration is authorized.

The approved threat model covers buggy callers, forged caller fields,
accidental corruption, incomplete writes, stale/mixed schemas, writable aliases,
and prediction/input inconsistency. A malicious actor with host/filesystem
access who rewrites a complete canonical row and recomputes every checksum and
identifier is explicitly out of scope. Do not add an external signer, manage
keys, or claim cryptographic authentication. Preserve the prior hostile-rewrite
fixture as documented out-of-scope evidence.

Do not access production SQLite or live history, inspect outcomes, refit, adapt
or deploy the runtime caller, change services/timers, migrate history, or alter
frozen artifacts. Use only temporary synthetic fixtures and an isolated
validation environment when needed.

Task routing: `large`, high-reasoning primary model, no worker delegation because
schema design, regression behavior, implementation, ancestry, and release are
one tightly coupled lane.
