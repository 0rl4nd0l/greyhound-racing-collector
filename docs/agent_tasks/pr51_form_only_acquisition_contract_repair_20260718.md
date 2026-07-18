---
job_id: pr51_form_only_acquisition_contract_repair_20260718
title: Repair PR 51 FORM_ONLY_V1 acquisition contract
lane: Provenance
supporting_lanes:
  - Data Engineering
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-18 /goal explicitly authorizes one narrow acquisition-contract repair from rejected PR 51 head 00319beb, local validation, one normal descendant commit, and a non-force push to the existing draft PR branch.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Read only the immutable pre-race evidence and outcome-unopened Jul 11-Aug 9 input freeze named by the rejected exact-head packet. Do not open outcomes, read or write databases, touch runtime or services, fit or evaluate models, create market cohorts, or mutate PRs 46 through 48.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/pr51_form_only_acquisition_contract_repair_20260718.md
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
  - scripts/build_form_only_v1_packet.py
  - tests/test_build_form_only_v1_packet.py
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/STATE.md
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/DECISIONS.md
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/VALIDATION.md
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/CODE_REVIEW.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/status.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/validation.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/guard-preflight.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/commands.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/build-a.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/build-b.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/compile.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/ruff.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/focused-pytest.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/full-pytest.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/coverage.json
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/coverage.txt
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/leakage-scan.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/tamper-probes.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/determinism.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/hash-verification.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/diff-check.log
  - reports/agent_jobs/pr51_form_only_acquisition_contract_repair_20260718/diff-check.json
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr51_form_only_acquisition_contract_repair_20260718
proof_question: Does one normal descendant of rejected PR 51 head 00319beb enforce race-scoped unlinkable identity, exact runner and scratch reconciliation, source-derived OOT policy, complete hash-bound provenance, conflict-safe genuine reconciliation, canonical deduplication and provable ordering, and durable deterministic reproduction without opening outcomes or widening beyond acquisition?
hypothesis_id: pr51_form_only_acquisition_contract_repair_v1
program_track: offline_development
entry_state: exact_head_00319beb_rejected_with_five_reproduced_acquisition_contract_defect_families
target_transition: repaired_hash_bound_form_only_v1_acquisition_packet_ready_for_new_independent_exact_head_review
exit_predicate: Every rejected finding is fixed and mapped to passing synthetic and current-input evidence; current outcome-unopened OOT inputs are revalidated without forced counts; builder statement coverage is at least 80 percent with untested safety branches reported; compile, Ruff, focused and full tests, leakage scans, tamper probes, adversarial fixtures, two clean builds, hash checks, and full diff review pass; no raw or large artifacts are committed; and PR 51 remains draft, open, and unmerged after one normal descendant push.
source_class: rejected_exact_head_00319beb_plus_hash_bound_review_packet_and_immutable_pre_race_inputs_only
dataset_version: pr51_00319beb_acquisition_contract_repair_20260718
evidence_hash: sha256:4027ab66bbf73670557fe04fb680270b9cb4c8e98714133242c3c804a4c6cd8c
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while PR 51 remains at exact head 00319beb before this lane publishes, the immutable review packet retains its recorded digests, no active claim owns these exact paths, Jul 11-Aug 9 outcomes remain unopened, and all work stays acquisition-only. Stop before publication on any unproven identity, roster, time, provenance, reconciliation, ordering, reproducibility, coverage, or scope invariant.
docs_impact: DOCS_UPDATED
docs_checked:
  - docs/form_only_v1_acquisition.md
  - docs/agent_tasks/form_only_v1_acquisition_foundation_20260718.md
docs_changed:
  - docs/form_only_v1_acquisition.md
  - docs/form_only_v1_reproducibility.json
docs_followup: Independent exact-head review is required after a green pushed repair; model, market, outcome, runtime, and activation work remain separately unauthorized.
reason: The repair changes acquisition identity, provenance, ordering, reconciliation, OOT binding, and reproducibility contracts, so tracked contract documentation and a small immutable reproduction descriptor must change with the builder.
---

# PR 51 FORM_ONLY_V1 acquisition-contract repair

Repair only the acquisition defects reproduced in
`pr51_form_only_acquisition_exact_head_review_20260718`. Do not repeat
discovery, relax a reproduced defect, force legacy counts or hashes, or cross
the outcome, model, market, database, runtime, service, activation, merge, or
PR 46 through 48 boundaries.

The repair must fail closed unless it proves race-scoped unlinkable runner
identifiers, exact card/sidecar/label roster reconciliation with immutable
scratch evidence, source-derived OOT time and window eligibility, complete
hash-bound trusted inputs, conflict-safe genuine reconciliation predicates,
normalized deduplication, and verified total ordering for recent history.

Publication is authorized only after the full validation contract passes. The
only GitHub write allowed is one non-force push of one normal descendant commit
to `codex/form-only-v1-acquisition-20260718`; leave PR 51 draft and unmerged.
