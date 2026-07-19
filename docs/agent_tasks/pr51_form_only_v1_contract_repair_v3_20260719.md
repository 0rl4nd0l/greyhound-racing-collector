---
job_id: PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3
title: Repair PR 51 FORM_ONLY_V1 independent-acceptance blockers
lane: Provenance
supporting_lanes:
  - Data Engineering
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-19 /goal explicitly authorizes a new V3 repair from exact PR 51 head 7ca2d55393c2bc5273b3c90c93ff49fbc06658c8, a fresh worktree/card, local validation, one normal descendant commit, a non-force push to the existing draft PR branch, and an accurate PR-body update.
allow_unapproved_safe_extension: false
allow_audit_code_changes: true
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Read only the immutable pre-race evidence and outcome-unopened Jul 11 through Aug 9 input freeze bound by PR51_FORM_ONLY_V1_REPAIR_INDEPENDENT_ACCEPTANCE_V1. Do not open outcomes, read or write databases, touch runtime or services, fit or evaluate models, create market cohorts, claim edge, or mutate PRs 46 through 48.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/pr51_form_only_v1_contract_repair_v3_20260719.md
  - docs/agent_tasks/form_only_v1_acquisition_foundation_20260718.md
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
  - scripts/build_form_only_v1_packet.py
  - tests/test_build_form_only_v1_packet.py
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/STATE.md
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/DECISIONS.md
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/VALIDATION.md
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/CODE_REVIEW.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/RUN_OUTCOME.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/DECISION_ENTRY.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/status.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/validation.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/guard-preflight.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/commands.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/build-a.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/build-b.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/compile.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/ruff.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/focused-pytest.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/full-parent-pytest.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/full-head-pytest.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/suite-delta.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/coverage.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/coverage.txt
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/forbidden-linkability-scan.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/tamper-probes.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/ambiguity-probes.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/determinism.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/hash-counts.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/diff-check.log
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/diff-check.json
  - reports/agent_jobs/PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3/pr-body.md
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3
proof_question: Does one normal descendant of rejected PR 51 head 7ca2d553 close every reproduced independent-acceptance blocker by separating trainer and sealed validation trust domains, independently binding sidecar semantics, rejecting ambiguous source discovery, isolating non-authoritative diagnostics, and failing closed on malformed declarations and artifacts without opening outcomes or widening scope?
hypothesis_id: pr51_form_only_v1_contract_repair_v3
program_track: offline_development
entry_state: exact_head_7ca2d553_independently_rejected_on_trust_domain_semantic_binding_ambiguity_diagnostic_coupling_fail_closed_and_disclosure_blockers
target_transition: repaired_pr51_form_only_v1_contract_ready_for_new_independent_exact_head_review_v3
exit_predicate: Every blocker in PR51_FORM_ONLY_V1_REPAIR_INDEPENDENT_ACCEPTANCE_V1 is mapped to a passing unit integration or adversarial fixture; trainer artifacts contain only race-scoped race-plus-box row identity and have zero cross-race or development-to-OOT identity intersections; sealed validation artifacts are separately hashed and excluded from trainer manifests; diagnostic bytes and paths cannot change canonical trainer outputs; source ambiguity and malformed declarations fail closed; two builds are byte-identical separately by trust domain; candidate included and OOT counts are source-revalidated with deltas explained; compile Ruff focused tests coverage forbidden/linkability scans one-bit tampering ambiguity probes diff review and parent/head suite delta complete with no hidden new failure error or timeout; docs descriptor and PR body are accurate; no raw or large artifacts are committed; and PR 51 remains draft open and unmerged after one normal descendant push.
source_class: exact_remote_pr51_head_7ca2d553_plus_independent_acceptance_review_validation_evidence_and_hash_bound_pre_race_inputs_only
dataset_version: pr51_head_7ca2d553_independent_acceptance_v1_contract_repair_v3_20260719
evidence_hash: sha256:8c3bb81d8cdaad14572a4d7935f06eda2a577670420b72f922a98b6f2a728283
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while remote PR 51 remains exactly 7ca2d55393c2bc5273b3c90c93ff49fbc06658c8 until publication, the independent acceptance REVIEW VALIDATION and evidence retain hashes 398f4129 f40c827e and 8c3bb81d, Jul 11 through Aug 9 outcomes remain unopened, no active claim owns these exact paths, and the work remains acquisition-only. Stop HEAD_CHANGED on remote drift and stop BLOCKED on any unresolved trust-domain, semantic-binding, ambiguity, diagnostic-isolation, malformed-input, reproducibility, suite-delta, or scope invariant.
docs_impact: DOCS_UPDATED
docs_checked:
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
  - docs/agent_tasks/form_only_v1_acquisition_foundation_20260718.md
  - docs/agent_tasks/pr51_form_only_acquisition_contract_repair_20260718.md
docs_changed:
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
  - docs/agent_tasks/form_only_v1_acquisition_foundation_20260718.md
docs_followup: A new exact-head independent review remains required; model market outcome runtime activation merge and PR 46 through 48 work remain separately unauthorized.
reason: The repair changes trust domains, artifact shape, validation semantics, source selection, diagnostic isolation, and fail-closed contracts, so tracked documentation and the reproducibility descriptor must change with the builder.
---

# PR51 FORM_ONLY_V1 contract repair V3

Repair only the blockers reproduced in
`PR51_FORM_ONLY_V1_REPAIR_INDEPENDENT_ACCEPTANCE_V1`. Prior V2 completion is
prerequisite evidence, not V3 completion.

Trainer-visible artifacts must be identity-minimal and independent of the
sealed validation-only and non-authoritative diagnostic domains. Publication
is authorized only after all listed validations pass. The only GitHub writes
allowed are one non-force push of one normal descendant commit to
`codex/form-only-v1-acquisition-20260718` and an accurate PR 51 body update.
Leave PR 51 draft and unmerged.
