# Target Metadata Snapshot Fix Report

Date: 2026-05-27

## Files changed

- `scripts/capture_prediction_snapshot.py`
- `utils/csv_metadata.py`
- `tests/test_capture_target_metadata.py`
- `artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/dry_run_capture_report.json`
- `artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/report.md`

## Metadata source rules

Future live snapshot persistence now requires verified target metadata before write:

- sidecar path is `<race_csv>.metadata.json`
- `target_distance` and `target_grade` must both normalize to non-null values
- `target_distance_source` and `target_grade_source` must be canonical sidecar sources: `canonical_pre_race_page` or `sidecar_target_metadata`
- top-level sidecar `metadata_is_leakage_safe` must be `true`
- `race_time_mapping_status` must be `exact_url_match`
- `race_time_source` must be `canonical_race_url`
- canonical TheDogs URL race number must match the captured race number

If any rule fails, capture marks `target_metadata_status` as `missing`, `unsafe`, or `mismatch`, sets `target_metadata_verified=false` in snapshot readiness, and skips persistence.

Final runner-set verification is also unconditional for persistence: the deprecated `--allow-unverified-runner-set` flag no longer permits writes when `final_runner_set_status` is not `verified`.

## Dry-run outcome

Command:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir artifacts/full_evidence_orchestration_20260525/final_runner_verified_live_batch/upcoming_races \
  --snapshot-dir artifacts/prediction_snapshots \
  --output artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/dry_run_capture_report.json
```

Result:

- status: `SUCCESS`
- dry_run: `true`
- persist_requested: `false`
- capture_count: `10`
- target_metadata_counts: `{"verified": 10}`
- final_runner_set_counts: `{"verified": 2, "mismatch": 8}`
- snapshot readiness: `2 READY`, `8 NOT_READY`
- persistence_status_counts: `{"dry_run": 10}`
- persisted_with_top_level_metadata_count: `0`
- priced_ev_total: `0`
- max probability-sum absolute error: `0.0003999999999997339`

## Skipped reasons

- Metadata: no skips in the dry-run; all 10 attempted captures had verified canonical sidecar target metadata.
- Runner set: 8 captures were `NOT_READY` due to final runner-set mismatch.
- Lifecycle: candidate scan saw `13 upcoming_not_jumped`, `4 jumped_pending_results`, and `3 resulted`; dry-run attempted the first 10 live targets only.
- Odds provenance: no live odds capture was requested; EV remained null for all runners.

## Leakage checks

- `assert_no_result_fields` passed for built snapshots.
- MLSystemV4 temporal integrity validation logged passed during dry-run prediction attempts.
- No result ingestion was run.
- No labels were written.
- Calibration drift is reported as `not_evaluated_no_result_ingestion` because this fix is label-free.
- Data integrity check: SQLite `PRAGMA quick_check` returned `ok`.
- Endpoint health: port `5002` was not listening, so `/api/health` and `/api/model_health` were not reachable.

## No snapshot rewrites

- Existing snapshot JSON files were not modified: before/after mtime diff over 71 `artifacts/prediction_snapshots/**/*.json` files was empty.
- Dry-run assertions confirmed zero `persisted` captures.
- `artifacts/prediction_snapshots/manifest.jsonl` was already modified before this task and was not appended by the dry-run.
- No BAL/MEADOWS frozen snapshot JSON was rewritten.

## Next live-capture command after approval

Run only after confirming the local upcoming batch is still pre-jump or after refreshing it:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir artifacts/full_evidence_orchestration_20260525/final_runner_verified_live_batch/upcoming_races \
  --snapshot-dir artifacts/prediction_snapshots \
  --persist \
  --output artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/persist_capture_report.json
```

Do not add `--allow-unverified-runner-set`. Do not add `--capture-live-odds` unless odds capture is separately approved.
