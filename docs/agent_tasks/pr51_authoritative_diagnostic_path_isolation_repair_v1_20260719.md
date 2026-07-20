---
job_id: PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1
title: Repair PR 51 authoritative diagnostic and trusted-path isolation
lane: Provenance
supporting_lanes:
  - Data Engineering
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-19 /goal explicitly authorizes a new bounded repair from exact rejected PR 51 head 16c60fd5bdee27c26476410bba666d38c9022f03, using a fresh worktree and card, local validation, one normal descendant commit, a non-force push to the existing draft PR branch, and an accurate PR-body update.
allow_unapproved_safe_extension: false
allow_audit_code_changes: true
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Read only the immutable pre-race acquisition evidence and outcome-unopened Jul 11 through Aug 9 freeze already bound by V3. Do not open outcomes, read or write databases, touch runtime or services, fit or evaluate models, create market cohorts, claim edge, mutate PRs 46 through 48, activate, merge, or mark ready.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/pr51_authoritative_diagnostic_path_isolation_repair_v1_20260719.md
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
  - scripts/build_form_only_v1_packet.py
  - tests/test_build_form_only_v1_packet.py
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/README.md
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/STATE.md
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/DECISIONS.md
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/VALIDATION.md
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/commands.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/red-repro.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/isolation-proof.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/domain-inventories.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/no-follow-root-proof.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/exploit-to-test-matrix.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/build-a.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/build-b.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/build-without-diagnostics.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/diagnostic-tests.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/hash-counts.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/attacker-scan.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/semantic-rebinding.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/ambiguity-fixtures.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/compile.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/ruff.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/focused-pytest.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/coverage.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/coverage.txt
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/parent-suite.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/head-suite.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/junit-parent.xml
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/junit-head.xml
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/suite-delta.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/diff-check.log
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/diff-check.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/CODE_REVIEW.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/pr-body.md
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/remote-pr.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/github-checks.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/final-refs.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/RUN_OUTCOME.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/DECISION_ENTRY.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/status.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/validation.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/guard-preflight.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/guard-final.json
  - reports/agent_jobs/PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1/release-receipt.json
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR51_AUTHORITATIVE_DIAGNOSTIC_PATH_ISOLATION_REPAIR_V1
proof_question: Does one normal descendant of rejected PR 51 head 16c60fd make authoritative trainer construction operationally independent from diagnostics, enforce exact declared physical sets in all four domains, and reject every ancestor-link, alternate-root, traversal, alias and path-swap access through descriptor-relative no-follow opens before returning verified bytes?
hypothesis_id: pr51_authoritative_diagnostic_path_isolation_repair_v1
program_track: offline_development
entry_state: pr51_exact_head_16c60fd_rejected_on_diagnostic_dependency_exact_domain_sets_ancestor_alias_and_disclosure
target_transition: repaired_pr51_authoritative_diagnostic_and_path_isolation_ready_for_fresh_full_independent_acceptance
exit_predicate: Authoritative build and load complete without enumerating opening hashing validating or requiring diagnostic inputs or outputs; diagnostics are a separate optional read-only phase whose failure cannot invalidate the authoritative packet; all physical 10/2/6/3 files are safely declared and hash-bound without self-reference and every domain rejects unexpected files dotfiles links directories duplicates missing files and role length hash or type changes; trusted-anchor descriptor-relative no-follow opens reject symlinked ancestors packet domains files traversal aliases path swaps and default discovery before bytes return; preserved counts hashes semantic binding non-linkability ambiguity duplicate and zero-aware gates pass; compile Ruff focused coverage attacker fixtures two authoritative builds with and without diagnostics optional diagnostic tests diff review and parent/head suite delta are complete; no raw or large data is committed; and PR 51 remains draft open and unmerged after one normal descendant push and accurate body update.
source_class: exact_remote_pr51_head_16c60fd_plus_PR51_V3_TRAINER_SURFACE_REPAIR_INDEPENDENT_ACCEPTANCE_V1_review_d7369a12_plus_hash_bound_pre_race_inputs_only
dataset_version: pr51_head_16c60fd_authoritative_diagnostic_path_isolation_repair_v1_20260719
evidence_hash: sha256:d7369a12bf665c2d6f4271562651fde698d48e9ae37a46a1d6c5abc2ab2843f3
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while remote PR 51 head remains exactly 16c60fd5bdee27c26476410bba666d38c9022f03 until publication, the named independent-acceptance review remains source-identical, Jul 11 through Aug 9 outcomes remain unopened, generated packets and hostile fixtures stay under fresh temporary directories, no active claim owns the exact mutation paths, and work remains acquisition-only. Stop HEAD_CHANGED on remote drift and stop BLOCKED on any unresolved diagnostic dependency, exact-set, no-follow path, linkability, trust-root, reproducibility, suite, disclosure, or scope invariant.
docs_impact: DOCS_UPDATED
docs_checked:
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
docs_changed:
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
docs_followup: A new full exact-head independent acceptance remains required; model, market, outcome, runtime, activation, merge and PR 46 through 48 work remain separately unauthorized.
reason: The repair changes build phase boundaries, exact domain inventories, trusted path-opening semantics, reproducibility hashes and PR disclosure, so the tracked contract docs and descriptor must change with the builder.
---

# PR51 authoritative diagnostic/path isolation repair V1

Repair only the four blocker families reproduced by
`PR51_V3_TRAINER_SURFACE_REPAIR_INDEPENDENT_ACCEPTANCE_V1` at exact head
`16c60fd5bdee27c26476410bba666d38c9022f03`.

The only GitHub writes allowed are one non-force push of one normal descendant
commit to `codex/form-only-v1-acquisition-20260718` and an accurate PR 51 body
update after every local gate passes. Leave PR 51 draft and unmerged.
