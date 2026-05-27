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

## Final closeout addendum

Code fix commit: `bc528837dfb63d3c6e622a27b1c3d8819bc21bdf`

Exact files staged and committed in `bc528837dfb6`:

- `scripts/capture_prediction_snapshot.py`
- `utils/csv_metadata.py`
- `tests/test_capture_target_metadata.py`
- `artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/report.md`

Final verification commands and results:

- `git diff --check`: passed.
- `.venv/bin/python -m py_compile scripts/capture_prediction_snapshot.py utils/csv_metadata.py utils/date_parsing.py`: passed.
- `.venv/bin/python -m pytest -q tests/test_capture_target_metadata.py --maxfail=1`: `7 passed in 14.72s`.
- `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set' --maxfail=1`: `89 passed, 607 deselected, 2 warnings in 166.29s`.
- `sqlite3 greyhound_racing_data_writable.db 'PRAGMA quick_check;'`: `ok`.
- Dry-run capture to `target_metadata_snapshot_fix/dry_run_capture_report.json`: `SUCCESS`, `dry_run=true`, `persist_requested=false`, `candidate_files=20`, `capture_count=8`, `persisted_with_top_level_metadata_count=0`.
- Dry-run lifecycle counts: `{"jumped_pending_results": 9, "resulted": 3, "upcoming_not_jumped": 8}`.
- Dry-run final runner-set counts: `{"mismatch": 7, "verified": 1}`.
- Dry-run target metadata counts: `{"missing": 1, "verified": 7}`.
- Dry-run regular checks: model version `V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033`; calibration drift `not_evaluated_no_result_ingestion`; data integrity `quick_check=ok`; temporal leakage `assert_no_result_fields=passed`; endpoint health `not_running_or_unreachable`.
- `jq` dry-run non-persistence checks: both passed.
- Snapshot JSON mtime before/after diff: empty.
- `ss -ltnp 'sport = :5002'`: no listener; `/api/health` and `/api/model_health` returned connection refused.

Preservation confirmations:

- `artifacts/prediction_snapshots/manifest.jsonl` was not staged or committed. It remained outside this task's staged set, and its mtime stayed `2026-05-27 17:15:32.362613917 +1000` across the final dry-run.
- `artifacts/full_evidence_orchestration_20260525/target_metadata_snapshot_fix/dry_run_capture_report.json` was treated as ignored local validation evidence and was not force-added.
- No existing `artifacts/prediction_snapshots/**/*.json` snapshot file was rewritten.
- No result ingestion, label writes, retrain, promotion, push, betting, fake odds, or fake EV occurred.
- `--allow-unverified-runner-set` was not used.
- `--capture-live-odds` was not used.
- Final runner-set verification remains mandatory for persistence.
- Future persisted snapshots require verified canonical target metadata and non-null top-level `target_distance` and `target_grade`.
- Missing, unsafe, or race-mismatched target metadata fails closed.
- Dry-run remains non-mutating for snapshot persistence.

Next gated live-capture command, not run in this closeout:

Preconditions:

- Repo state has been reviewed and staged/unstaged dirt is understood.
- Upcoming race files are freshly refreshed through the existing canonical TheDogs upcoming/form-guide path, or the local batch is explicitly confirmed still pre-jump.
- Dry-run report shows READY races with `final_runner_set_status=verified`, `target_metadata_status=verified`, non-null top-level `target_distance`, non-null top-level `target_grade`, canonical target metadata sources, exact URL-backed race time, result-free snapshots, and no use of `--allow-unverified-runner-set`.

Dry-run only:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir <fresh_upcoming_races_dir> \
  --snapshot-dir artifacts/prediction_snapshots \
  --output artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_live_batch/dry_run_capture_report.json
```

Persist only after the fresh dry-run proves READY verified races and after explicit approval:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir <fresh_upcoming_races_dir> \
  --snapshot-dir artifacts/prediction_snapshots \
  --persist \
  --output artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_live_batch/capture_report.json
```
