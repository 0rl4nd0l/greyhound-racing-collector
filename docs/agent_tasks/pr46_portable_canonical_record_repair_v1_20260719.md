---
job_id: PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
  - Runtime Compatibility
owner: Codex
approval_required: true
approval_source: The owner explicitly requested /goal PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1, including one bounded exact-head descendant, cross-runtime validation, and fast-forward publication to existing draft PR 46.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Synthetic and committed fixed fixtures only. No production database, outcomes, live prediction, services, timers, activation, deployment, betting action, model refit, or frozen artifact mutation.
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1
proof_question: Can one versioned deterministic numerical canonicalization rule make complete sealed residual records byte-identical and exactly replayable across supported Python 3.11.15 and 3.13.12 runtimes without changing frozen artifact bytes or weakening integrity semantics?
hypothesis_id: final_prediction_decimal_canonicalization_removes_cross_runtime_last_bit_noise_v1
program_track: offline_development
entry_state: exact draft PR 46 head 813d700c emits different complete v2 records under Python 3.11.15 and 3.13.12 with NumPy 1.26.4 and rejects bidirectional replay
target_transition: one published PR 46 descendant emits byte-identical portable v3 records and exact bidirectional replay while PR 46 remains draft unmerged and PR 53 remains untouched
exit_predicate: The exact failure is reproduced before edits; one output-contract version bump binds the smallest quantified canonicalization rule into effective state; both runtimes emit identical bytes, SHA, checksum and key and return EXACT_REPLAY across histories; fixed and broader fixtures preserve ordering, ranks, winners, normalization and meaningful values; all scorer/writer, lock/resource, lint, compile, regeneration, hardening, V2 and GitHub gates pass; one normal child commit fast-forwards PR 46; its body is current; PR 46 remains draft unmerged; PR 53 remains exact and untouched.
source_class: exact_pr46_813d700c_plus_committed_fixed_fixture_plus_deterministic_synthetic_cross_runtime_fixtures
dataset_version: pr46_813d700c_portable_record_contract_v1_20260719
evidence_hash: sha256:40371bbaaa6684c537d010b65c6af562e6e99c86a642c95c861610ed5bc7087b
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Stop if live PR 46 changes from 813d700c622f5a0cb424b297242c611ccd753578 before push, master changes from c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa in a way that invalidates review, PR 53 changes from 2deb5aec454fb9314a22f30a30169aa05b2261c5, the worktree gains unrelated dirt, another non-stale claim overlaps source/test/CI paths, frozen model/manifest/fit bytes must change, tolerance comparison or checksum bypass becomes necessary, live runtime mutation is required, or the repair cannot be one normal descendant commit.
docs_impact: DOCS_UPDATED
docs_checked:
  - docs/manual_market_form_residual_prediction.md
  - AGENTS.md
docs_changed:
  - docs/manual_market_form_residual_prediction.md
docs_followup: Update only the frozen output-record portability and explicit migration contract; do not document activation.
reason: The change alters canonical prediction record representation, schema migration semantics, effective-state identity, and supported-runtime reproducibility.
task_tier: critical
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: The repair couples floating-point canonicalization, cryptographic identity, history replay, frozen artifact invariants, dual-runtime CI, transactional writer hardening, and exact-head publication.
worker_model_allowed: false
worker_decision_limit: No delegation; one agent owns reproduction, numerical analysis, implementation, validation, review, and exact-head publication.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/pr46_portable_canonical_record_repair_v1_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - tests/test_market_form_residual_portability.py
  - .github/workflows/market-form-residual-portability.yml
  - docs/manual_market_form_residual_prediction.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/README.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/STATE.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/DECISIONS.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/VALIDATION.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/CODE_REVIEW.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/RUN_OUTCOME.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/DECISION_ENTRY.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/status.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/validation.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/guard-preflight.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/guard-final.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/diff-check.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/repro.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/canonicalization-analysis.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/cross-runtime-proof.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/hash-proof.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/suite-results.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/remote-pr.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/github-checks.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/final-refs.json
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/pr-body.md
  - reports/agent_jobs/PR46_PORTABLE_CANONICAL_RECORD_REPAIR_V1/commands.log
---

# PR46 portable canonical record repair v1

Make v2-era sealed records portable across the repository's supported Python
runtimes with one explicit, deterministic numerical output contract. Bump the
record representation rather than reinterpreting prior bytes. Preserve every
frozen artifact and safety boundary, publish exactly one child of `813d700c`,
and leave PR 46 draft/unmerged and PR 53 untouched.

## Non-negotiable gates

- No tolerance-based replay, runtime branches, checksum bypass, omitted fields,
  Python 3.11-only environment, or old-byte reinterpretation.
- Quantify the chosen boundary against rejected narrower and coarser rules.
- Pin full record bytes, SHA, checksum, key, and bidirectional exact replay under
  Python 3.11.15 and 3.13.12 with NumPy 1.26.4.
- Cover signed values, zero, rounding boundaries, near-cap values, ties,
  normalization, repeated execution, fixed predictions, and broader fixtures.
- Preserve model, manifest, fit-population bytes, outcome exclusion,
  provenance, forgery/history rejection, lock admission, descriptor ownership,
  publication, and durability semantics.
