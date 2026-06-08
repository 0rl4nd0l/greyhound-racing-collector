# EV Readiness Summary Fix - 2026-06-02

## Session

- Agent: Codex
- Branch: feature/dog-odds-snapshot-readiness
- Worktree: /home/l4nd0/greyhound_racing_collector
- Lane: Evaluation / reporting
- Execution mode: safe extension, no live writes
- Approved commit scope: scripts/prejump_prediction_loop.py, tests/test_prejump_prediction_loop.py, tests/conftest.py, tests/test_v4_contract_builder.py, this report
- Forbidden surfaces not staged or modified by this closeout: model files, model registry promotion state, production labels, snapshot JSON files, artifacts/prediction_snapshots/manifest.jsonl, unrelated Tenn/extraction files

## EV Reporting Root Cause

The post-execution operator/live-odds packet was reading EV readiness from the dry-run persist readiness report after an approved persist run. For the latest approved June 2 second batch, the dry-run report correctly showed pre-persist live odds were not available:

- dry-run report: EV_NOT_READY 5, priced_ev_runner_count 0, odds_exclusion_counts {"missing_live_odds": 32}
- path: artifacts/full_evidence_orchestration_20260525/prejump_approved_persist_odds_after_np_fix_20260602T141219AEST/dry_run_capture_report.json

The authoritative post-persist report for the same batch showed persisted EV readiness:

- persist report: EV_READY 5, EV_NOT_READY 0, priced_ev_runner_count 32, odds_exclusion_counts {}
- path: artifacts/full_evidence_orchestration_20260525/prejump_approved_persist_odds_after_np_fix_20260602T141219AEST/persist_capture_report.json

This was a reporting-source mismatch, not an EV failure.

## EV Patch

The loop now uses the authoritative persisted capture report for post-execution EV reporting when an approved persist step ran or when that report exists. A capture report is accepted as authoritative only when all of these are true:

- status=SUCCESS
- dry_run=false
- persist_requested=true
- persist_approved=true

The loop uses explicit top-level ev_readiness_counts when present. If top-level fields are absent, fallback values are computed only from persisted capture-level EV statuses, priced EV runner counts, and odds exclusion counts. Dry-run reports are explicitly rejected as authoritative persisted EV evidence, so missing dry-run-only fields are not classified as persisted EV_NOT_READY.

Diagnostic fields added to loop output:

- ev_summary_source
- ev_ready_count
- ev_not_ready_count
- priced_ev_runner_count
- odds_exclusion_count
- odds_exclusion_counts
- authoritative_capture_report_path
- ev_summary_consistency_check
- ev_summary_failure_reason

After recheck against the authoritative report:

- ev_summary_source=authoritative_persist_capture_report
- ev_ready_count=5
- ev_not_ready_count=0
- priced_ev_runner_count=32
- odds_exclusion_count=0
- ev_summary_consistency_check=MATCH

The dry-run report is rejected with ev_summary_source=NOT_AUTHORITATIVE_CAPTURE_REPORT and ev_summary_consistency_check=REJECTED_NON_PERSISTED_REPORT.

No EV gating semantics were loosened. EV remains null unless odds provenance passes. Final runner-set verification and target metadata verification remain mandatory.

## GREYHOUND_DB_PATH Blocker

The previously blocked broad selector failed at:

- tests/test_v4_contract_builder.py::test_build_and_save_contract_contains_expected_metadata
- failure: RuntimeError: Database preflight failed: missing required table(s) dog_race_data, race_metadata
- contaminated DB path: /tmp/pytest-of-l4nd0/pytest-277/test_persist_requires_explicit0/greyhound.sqlite

The same V4 contract test passed in isolation. A minimal ordered pair reproduced the failure:

- tests/test_capture_target_metadata.py::test_persist_requires_explicit_approval
- then tests/test_v4_contract_builder.py::test_build_and_save_contract_contains_expected_metadata

Root cause: scripts/capture_prediction_snapshot.py::_configure_safe_runtime sets GREYHOUND_DB_PATH in process-global os.environ for capture execution. The capture test invoked that runtime helper with a temporary DB path, and the environment value leaked into the later V4 contract test in the same pytest process.

Fix: tests/conftest.py now has an autouse fixture that snapshots and restores database-routing environment variables around each test. tests/test_v4_contract_builder.py adds a regression pair proving a GREYHOUND_DB_PATH mutation in one test is restored before V4 contract tests run.

This is a test-isolation fix. Production capture behavior was not changed.

## Untracked File Audit

scripts/prejump_prediction_loop.py and tests/test_prejump_prediction_loop.py were pre-existing untracked files in this checkout. Staging them blindly would add whole files, not a narrow tracked-file diff.

Audit results:

- scripts/prejump_prediction_loop.py: 5510 lines, 224180 bytes
- tests/test_prejump_prediction_loop.py: 5054 lines, 180220 bytes
- full addition size: 10564 inserted lines
- git diff --check for both full additions: clean
- classification: active intended operator-loop source and matching regression suite from prior greyhound work, not scratch/generated artifacts

Reasons the full additions are safe to stage for this task after review:

- the script has a guarded CLI entry point and explicit approval gates for live persist, live odds capture, result label write, and promotion
- tests import and exercise the script directly
- existing project scripts/tests reference the pre-jump loop path, plan schema, and post-execution packet fields
- validation passed with these files present
- no unrelated Tenn/extraction files are included

## Validation

Passed after the EV fallback tightening and test-isolation fix:

- python3 -m py_compile scripts/prejump_prediction_loop.py scripts/refresh_prejump_upcoming.py scripts/capture_prediction_snapshot.py scripts/ingest_results_for_date.py scripts/build_label_write_preflight_packet.py scripts/build_label_write_rehearsal_packet.py utils/csv_metadata.py utils/runner_completeness.py utils/date_parsing.py tests/conftest.py tests/test_v4_contract_builder.py
- .venv/bin/python -m pytest -q tests/test_prejump_prediction_loop.py --maxfail=1: 71 passed
- .venv/bin/python -m pytest -q tests -k 'prejump or ev_readiness or odds or capture_report' --maxfail=1: 110 passed, 756 deselected, 1 warning
- .venv/bin/python -m pytest -q tests/test_form_guide_delimiter_normalization.py tests/test_csv_download_hardening.py tests/test_runner_completeness.py tests/test_capture_target_metadata.py --maxfail=1: 44 passed
- .venv/bin/python -m pytest -q tests -k 'prejump or snapshot or metadata or leakage or runner_set or odds or ev' --maxfail=1 -vv: 241 passed, 625 deselected, 7 warnings
- git diff --check: clean
- sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;': ok
- guarded June 1 live label row count: 0
- APPROVE_RESULT_LABEL_WRITE: not set

Endpoint health:

- ss -ltnp 'sport = :5002': no listener
- /api/health: connection refused
- /api/model_health: connection refused

Process checks:

- no active prejump_prediction_loop, capture_prediction_snapshot, or ingest_results_for_date process after validation

Code review:

- the review found one fallback aggregation issue in the first EV helper version; it is fixed and covered by test_authoritative_persist_capture_report_fallback_aggregates_capture_fields
- no remaining critical, warning, or suggestion findings after the fallback patch and validation rerun

## June 2 Snapshot And Result Status

Persisted 2026-06-02 snapshots considered: 10.

All 10 are result-free, final_runner_set_status=verified, target_metadata_status=verified, target_distance non-null, target_grade non-null, required prediction fields present, and group-normalised within persisted rounding tolerance.

Full-corpus EV readiness is 9 EV_READY and 1 EV_NOT_READY. The latest approved second batch remains 5/5 EV_READY with 32/32 priced EV runners and no odds exclusions.

June 2 top-pick distribution: 10/10 top picks from box 1.

Result dry-run was not rerun in this closeout because the prior dry-run already showed clean_for_label_write=false and the known blockers were result-evidence issues, not EV summary issues. Prior dry-run status remains:

- dry_run=true
- persisted snapshots considered=10
- ingested=9
- pending=1
- participant_mismatches=0
- clean label candidates=7
- label-write blocked=3
- EV-evaluable runners=71
- positive EV count=33
- clean_for_label_write=false
- pending: Race 11 - AP_K - 2026-06-02 because TheDogs returned 403 and Sportsbet R11 result was not found
- blockers include incomplete official positions for Race 2 - HOR and partial Sportsbet fallback for Race 2 - LADBROKES-Q1-LAKESIDE

Because result dry-run evidence is not clean, no label readiness, preflight, rehearsal, production label write, or live result-ingestion write was performed.

## No-Mutation Confirmation

Confirmed for this task:

- no labels written
- no live result ingestion writes
- no result dry-run write mode
- no APPROVE_RESULT_LABEL_WRITE
- no snapshot writes or rewrites
- no manifest entries appended by this reporting patch
- artifacts/prediction_snapshots/manifest.jsonl not staged
- no --capture-live-odds
- no --persist
- no --allow-unverified-runner-set
- no fake odds
- no fake EV
- no retrain
- no model promotion
- no betting
- no push

## Accuracy Caveat

Infrastructure progress is real. Current champion accuracy is not acceptable and is not promotion-ready.

Known metrics remain:

- historical clean official packet: 105 races, top1 0.1619, top2 0.3333, top3 0.4857, mean winner rank 3.914, Brier 0.1230, log loss 1.9595, calibration slope 0.395
- latest rolling clean official packet: 29 races, top1 0.1379, top2 0.3103, top3 0.4828, mean winner rank 3.9655
- historical diagnostic top-pick box distribution: {"1": 110}
- June 2 persisted corpus top-pick distribution: 10/10 box 1

EV mechanics do not imply an EV edge. No betting is justified. No promotion is justified.

## Next Safe Task

With the EV summary fix and broad selector clean, the next safe task is a report-only isolated challenger study on a clean official holdout comparing champion, no-box, reduced-box-band, history-only, market-implied, blend, and calibrated variants. Promotion must remain blocked until a challenger beats the champion on clean official metrics and reduces the box-1 collapse without hiding failures.
