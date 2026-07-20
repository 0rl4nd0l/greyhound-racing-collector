---
job_id: RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3
title: Reconstruct the useful PR 47 live-race handoff on current master
lane: Provenance
supporting_lanes:
  - Evaluation
  - Testing
  - Repo Hygiene
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-20 /goal RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3
  explicitly authorizes reconstruction from exact live master
  98e363dd9cc9950ac5d05f4d533df3f5e06f138e, one normal successor commit,
  a new pushed branch, and one new draft PR targeting master, with CI wait.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Source code, focused fixtures, tracked frozen artifacts, outcome-free pre-jump
  evidence, and optional isolated temporary live-proof outputs only. No database
  writes, target outcomes, PR 51 FORM_ONLY_V1 trainer/control/sealed domains,
  production history, runtime state, services, timers, deployment, activation,
  promotion, EV, betting, model fitting, artifact mutation, or history migration.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3
proof_question: >-
  Can only PR 47's still-useful exact live-race identity, grade, sealed packet,
  quarantine, and read-only manual scoring behavior be reconstructed on exact
  current master while using master record V3 and effective-state V2, preserving
  every frozen byte, excluding obsolete scorer/writer ancestry and PR 51 data,
  and publishing one independently reviewable draft successor PR?
hypothesis_id: pr47_live_race_handoff_semantic_reconstruction_v3_20260720
program_track: offline_development
entry_state: >-
  master 98e363dd contains merged PR 46 scorer/writer V3 and merged PR 51
  FORM_ONLY_V1; old draft PR 47 remains at 0ae5937c on base c1dfd464 and is
  non-mergeable because it carries stale ancestry plus useful handoff behavior.
target_transition: >-
  one clean draft successor PR on exact master carries only the source-proven
  live-race handoff and read-only CLI behavior and leaves old PR 47 unchanged.
exit_predicate: >-
  The old-base-to-PR47 diff is semantically classified; exact TheDogs meeting-card
  grade transport binds canonical URLs, date, venue, race number, normalized
  grade, proof key, runner identity and source hashes; unsafe ambiguity,
  laundering, mismatch, post-jump and outcome inputs fail closed; one named race
  resolves to one sealed packet or one deterministic quarantine reason; the
  stdout-only CLI uses master's record V3, effective-state V2 and portable
  numerical contract in canonical runner order; focused and inherited PR 46
  tests, collector/resource tests, Ruff, compile, diff and task guards pass;
  frozen model, manifest and fit-population bytes equal master; optional live
  proof is isolated or reported DATA_MISSING; exactly one normal commit is pushed
  to one new branch; one new draft PR targets master; live CI is terminal green;
  PRs 47, 48, 53 and 54 remain unchanged and the successor remains unmerged.
source_class: >-
  exact_master_98e363dd_plus_exact_old_pr47_base_c1dfd464_to_head_0ae5937c_diff_plus_fixture_only_outcome_free_live_race_contract
dataset_version: greyhound_live_race_handoff_v3_master_98e363dd_20260720
evidence_hash: sha256:8a4be90db5cd1a111307daca01f21edfd4b246ed37fe4962163f59ef164fbbf0
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master remains exactly
  98e363dd9cc9950ac5d05f4d533df3f5e06f138e until publication, old PR 47 remains
  exactly 0ae5937cde87131c714fb7383c58ce13e3cfbc06 on base
  c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, PRs 46 and 51 remain merged, the
  clean task worktree has no unrelated dirt, no non-stale overlapping claim
  exists, frozen bytes remain unchanged, and the work needs no outcome, PR 51,
  persistence, runtime, service, deployment, activation, or migration access.
  Abort on material drift, disallowed-path need, source-proven blocker, test or
  review failure, or inability to publish exactly one normal commit.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/manual_market_form_residual_prediction.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed:
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: >-
  Deployment, activation, production persistence, migration and PR 51 integration
  remain separately gated and are not follow-ups authorized by this task.
reason: >-
  The successor restores an operator command, sealed input contract, and exact
  grade transport that are not documented on current master.
task_tier: large
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: >-
  The change couples live race identity, temporal and outcome leakage exclusion,
  hash-bound artifacts, canonical scoring output, exact-head publication and
  adversarial fail-closed validation across several existing seams.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation. The repository instructions make deep or multi-agent review
  opt-in, and this coupled reconstruction stays in one clean worktree.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/reconstruct_greyhound_live_race_handoff_v3_20260720.md
  - scripts/predict_market_form_residual.py
  - scripts/refresh_prejump_upcoming.py
  - upcoming_race_browser.py
  - utils/csv_metadata.py
  - tests/test_predict_market_form_residual.py
  - tests/test_csv_download_hardening.py
  - tests/test_prejump_prediction_loop.py
  - tests/test_upcoming_race_time_mapping.py
  - docs/manual_live_market_form_residual_prediction.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/README.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/STATE.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/DECISIONS.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/VALIDATION.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/CODE_REVIEW.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/SEMANTIC_INVENTORY.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/HASH_PROOF.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/LIVE_PROOF.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/RUN_OUTCOME.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/DECISION_ENTRY.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/status.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/validation.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/guard-preflight.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/guard-final.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/diff-check.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/pr-body.md
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/remote-pr.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/github-checks.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/final-refs.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/WAIT_RESULT.json
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/WAIT_RESULT.json.log
  - reports/agent_jobs/RECONSTRUCT_GREYHOUND_LIVE_RACE_HANDOFF_V3/release-receipt.json
---

# Reconstruct Greyhound live-race handoff V3

Port only the useful behavior from old draft PR 47 onto exact current master.
Treat the old diff as evidence, not ancestry: do not copy its reports, generated
fit packet, frozen scorer/writer implementation, service changes, or unrelated
history. Use master's frozen residual implementation unchanged.

The successor must bind one exact outcome-free TheDogs/Sportsbet feature packet
to one named upcoming race, fail closed with deterministic quarantine on missing
evidence, and score read-only to canonical stdout. It must preserve PR 51
FORM_ONLY_V1 isolation and remain a draft, unmerged PR for independent review.
