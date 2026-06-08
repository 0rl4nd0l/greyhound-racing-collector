# Post Target Metadata Fix Delimiter Contract Report

## Decision

Status: OK for delimiter/normalisation fix; dry-run only completed.

No persist, result ingestion, label write, retrain, model promotion, push, betting, fake odds, fake EV, or snapshot rewrite was performed.

## Root Cause

The live TheDogs expert-form export path returned canonical TheDogs export CSVs with comma delimiters. The project accepted-upcoming contract expects UTF-8 pipe-delimited form-guide CSVs. The previous gate correctly refused comma CSVs, so the accepted upcoming directory stayed empty even though the raw exports were valid source data.

Observed in this run: all 20 raw TheDogs exports had `original_delimiter=","`; 18 passed strict verification and were converted to `normalized_delimiter="|"`.

## Fix

The existing active refresh path remains:

`upcoming_race_browser.UpcomingRaceBrowser.get_upcoming_races + download_race_csv`

No new refresh script was added. Since no existing active pipe-output path was found, explicit audited normalisation was added for verified canonical TheDogs exports only.

Changed files:

- `upcoming_race_browser.py`
- `expert_form_csv_scraper.py`
- `utils/csv_metadata.py`
- `utils/runner_completeness.py`
- `scripts/validate_upcoming_races.py`
- `docs/FORM_GUIDE_SPEC.md`
- `tests/test_form_guide_delimiter_normalization.py`
- `tests/test_csv_download_hardening.py`

## Accepted CSV Contract

Accepted capture inputs under `upcoming_races/` are UTF-8 pipe-delimited (`|`) canonical form-guide CSVs only. Raw comma-delimited TheDogs exports are preserved under `upcoming_races/raw_exports/` and are never passed directly to capture.

Normalisation requires all of these gates to pass first:

- exact TheDogs expert-form schema
- dog-name blocks with box-prefixed primary rows and continuation rows suitable for forward-fill
- historical DATE rows strictly before the target race date
- complete runner set from the source CSV
- canonical sidecar target metadata with non-null `target_distance` and `target_grade`
- leakage-safe canonical pre-race metadata source
- URL-backed race time with exact race-number match, including nested `race_info` fields

Rejected candidates are kept out of the accepted top-level CSV set and written to quarantine with a reason.

## Artifact Layout

- Accepted pipe CSVs: `artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_delimiter_contract/upcoming_races/*.csv`
- Accepted sidecars: `*.csv.metadata.json`
- Raw source exports: `upcoming_races/raw_exports/*.csv`
- Quarantine: `upcoming_races/quarantine/*.csv`
- Reports: `refresh_report.json`, `normalization_report.json`, `accepted_races.tsv`, `rejected_races.tsv`, `dry_run_capture_report.json`

Sidecars record `original_delimiter`, `normalized_delimiter`, `normalization_source`, `normalization_status`, `normalization_timestamp`, `raw_export_path`, `accepted_csv_path`, `form_guide_spec_version`, and `normalization_verification`.

## Refresh Outcome

- total_races_found: 203
- selected_count: 20
- raw_export_count: 20
- accepted_upcoming_csv_count: 18
- normalized_count: 18
- format_rejected_count: 0
- quarantine_count: 2
- delimiter_counts: `{',': 20}`
- runner_set_counts: `{'COMPLETE': 20}`
- target_metadata_counts: `{'missing': 2, 'verified': 18}`
- lifecycle_counts: `{'future_outside_preferred_window': 154, 'missing_or_unparsed_race_time': 7, 'past_or_started': 7, 'preferred_20_160_min': 35, 'selected': 20}`

Rejected/quarantined:
- Race 5 - TAREE - 2026-05-27.csv: target_metadata_not_verified:missing_target_grade (runner_set=COMPLETE, metadata=missing)
- Race 2 - CANN - 2026-05-27.csv: target_metadata_not_verified:missing_target_grade (runner_set=COMPLETE, metadata=missing)

## Dry-Run Outcome

- status: SUCCESS
- dry_run: True
- persist_requested: False
- candidate_files: 18
- capture_count: 3
- READY races: 1
- NOT_READY races: 2
- final_runner_set_counts: `{'mismatch': 2, 'verified': 1}`
- target_metadata_counts: `{'verified': 3}`
- lifecycle_counts: `{'resulted': 15, 'upcoming_not_jumped': 3}`
- metadata_verified_count: 3
- metadata_missing_count: 0
- metadata_unsafe_count: 0
- metadata_mismatch_count: 0
- persisted_with_top_level_metadata_count: 0

Probability/model checks:
- Race 10 - BAL - 2026-05-27: READY, final_runner_set=verified, probability_sum=1.0001, model=V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033
- Race 11 - BAL - 2026-05-27: NOT_READY, final_runner_set=mismatch, probability_sum=0.9999999999999999, model=V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033
- Race 3 - CANN - 2026-05-27: NOT_READY, final_runner_set=mismatch, probability_sum=0.9999000000000001, model=V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033

READY race present, but not persisted in this task.

## Regular Checks

- endpoint health: port 5002 not listening; `/api/health` and `/api/model_health` returned connection refused
- SQLite quick_check: ok
- model_version: `['V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033']`
- calibration drift: `{'reason': 'capture fix is label-free and does not ingest results', 'status': 'not_evaluated_no_result_ingestion'}`
- data integrity: `{'quick_check': 'ok', 'status': 'ok'}`
- temporal leakage: `{'guard': 'assert_no_result_fields', 'status': 'passed'}`
- odds_capture_requested: False

## Validation Commands

Passed:

- `git diff --check`
- `python3 -m py_compile upcoming_race_browser.py form_guide_csv_scraper.py expert_form_csv_scraper.py comprehensive_form_data_collector.py utils/csv_metadata.py utils/date_parsing.py utils/runner_completeness.py scripts/validate_upcoming_races.py scripts/capture_prediction_snapshot.py tests/test_form_guide_delimiter_normalization.py tests/test_csv_download_hardening.py`
- `.venv/bin/python -m pytest -q tests/test_form_guide_delimiter_normalization.py tests/test_csv_download_hardening.py tests/test_runner_completeness.py tests/test_capture_target_metadata.py --maxfail=1` -> 29 passed
- `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set' --maxfail=1` -> 91 passed, 613 deselected, 2 pytest return-value warnings
- `.venv/bin/python scripts/validate_upcoming_races.py --dir artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_delimiter_contract/upcoming_races` -> validated 18 files
- delimiter check -> accepted_csv_count 18, bad_delimiter_count 0
- SQLite `PRAGMA quick_check` -> ok
- dry-run jq assertion -> true
- snapshot JSON mtime diff -> clean
- manifest diff -> clean

Residual failure recorded:

- `.venv/bin/python -m pytest -q tests -k 'form_guide or delimiter or upcoming or csv_metadata or target_metadata' --maxfail=1` failed at `tests/api/test_flask_api.py::test_all_upcoming_races_prediction`: expected `data["total_races"] >= 2`, got `0`. This is the synthetic Flask API fixture path, not the live TheDogs normalisation path exercised by this task.

## No-Persist Confirmation

- `--persist` was not used.
- `--allow-unverified-runner-set` was not used.
- `--capture-live-odds` was not used.
- dry-run report has `dry_run=true` and `persist_requested=false`.
- zero captures have `persistence.status="persisted"`.
- snapshot JSON mtimes are unchanged from preflight.
- `artifacts/prediction_snapshots/manifest.jsonl` is unchanged from preflight.
- existing frozen snapshots were not rewritten.
- no result ingestion, labels, retrain, promotion, push, betting, fake odds, or fake EV occurred.

## Next Gated Persist Command

Only run after explicit approval:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_delimiter_contract/upcoming_races \
  --snapshot-dir artifacts/prediction_snapshots \
  --persist \
  --output artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_delimiter_contract/capture_report.json
```

## Commit Closeout Addendum - 2026-05-27

This closeout supersedes the direct persist command above. The READY race in
the dry-run evidence was from the 2026-05-27 batch and may no longer be
pre-jump. Do not persist from this stale directory. The next gated task must
start with a fresh post-normalisation dry-run.

Files changed for the delimiter-normalisation patch:

- `docs/FORM_GUIDE_SPEC.md`
- `upcoming_race_browser.py`
- `expert_form_csv_scraper.py`
- `scripts/validate_upcoming_races.py`
- `utils/csv_metadata.py`
- `utils/runner_completeness.py`
- `tests/test_csv_download_hardening.py`
- `tests/test_form_guide_delimiter_normalization.py`
- this `report.md`

Exact root cause: the canonical TheDogs expert-form export endpoint emits
comma-delimited CSV, while the accepted project form-guide contract requires
UTF-8 pipe-delimited CSV in the accepted upcoming-races directory.

Exact fix: the active refresh path now preserves raw TheDogs comma exports
under `raw_exports/` and normalises only strictly verified canonical TheDogs
exports into accepted pipe-delimited CSVs. Capture continues to enumerate only
accepted top-level pipe CSVs; raw comma exports are kept as source evidence and
are not capture inputs.

Normalisation remains fail-closed before accepted CSV write when any of these
checks fail: target distance missing, target grade missing,
`metadata_is_leakage_safe` not true, race-time mapping not
`exact_url_match`, race-time source not `canonical_race_url`, canonical URL
race number mismatch, incomplete runner set, target-date or future history row,
schema/header mismatch, or row column-count drift. BOM handling is limited to
header/cell normalisation during schema validation and accepted pipe output
generation; schema drift is still rejected.

Closeout evidence:

- raw_export_count: 20
- accepted_upcoming_csv_count: 18
- normalized_count: 18
- quarantine_count: 2
- quarantines: Race 5 - TAREE and Race 2 - CANN, both
  `target_metadata_not_verified:missing_target_grade`
- all accepted sidecars include `original_delimiter=","`,
  `normalized_delimiter="|"`, `normalization_source="canonical_thedogs_export"`,
  `normalization_status="verified"`, `normalization_timestamp`,
  `raw_export_path`, `accepted_csv_path`, and `normalization_verification`
- dry_run: true
- persist_requested: false
- persisted capture count: 0
- dry-run READY count: 1
- dry-run NOT_READY runner-set mismatch count: 2
- `--allow-unverified-runner-set`: not used
- `--capture-live-odds`: not used
- snapshot JSON mtime comparison: clean
- manifest comparison: clean
- BOM-prefixed real TheDogs export regression: verified normalisation emits
  validator-accepted pipe CSV beginning with `Dog Name|Sex|PLC|BOX`

Validation rerun during commit closeout:

- `git diff --check` passed
- `python3 -m py_compile upcoming_race_browser.py form_guide_csv_scraper.py expert_form_csv_scraper.py comprehensive_form_data_collector.py utils/csv_metadata.py utils/date_parsing.py utils/runner_completeness.py scripts/validate_upcoming_races.py scripts/capture_prediction_snapshot.py` passed
- `.venv/bin/python -m pytest -q tests/test_form_guide_delimiter_normalization.py tests/test_csv_download_hardening.py tests/test_runner_completeness.py tests/test_capture_target_metadata.py --maxfail=1` passed: 30 passed
- `.venv/bin/python -m pytest -q tests -k 'snapshot or metadata or leakage or runner_set' --maxfail=1` passed: 91 passed, 614 deselected, 2 warnings
- `.venv/bin/python scripts/validate_upcoming_races.py --dir artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_delimiter_contract/upcoming_races` passed: 18 files validated
- accepted delimiter audit passed: 18 accepted CSVs, 0 bad delimiters
- normalisation JSON assertions passed: 18 accepted, 20 raw, 18 normalised, 2 quarantined
- dry-run capture JSON assertions passed: dry-run true, persist false, 0 persisted, READY race metadata and final runner set verified
- SQLite `PRAGMA quick_check` passed: ok
- snapshot JSON mtime diff passed: no rewrites
- manifest diff passed: unchanged from preflight copy

Residual test failure status:

- `.venv/bin/python -m pytest -q tests/api/test_flask_api.py::test_all_upcoming_races_prediction --maxfail=1 -vv` still fails with `total_races == 0`.
- `.venv/bin/python -m pytest -q tests -k 'form_guide or delimiter or upcoming or csv_metadata or target_metadata' --maxfail=1` fails at the same test only.
- This appears unrelated to the delimiter-normalisation patch because `app.py`, `tests/api/test_flask_api.py`, and `utils/race_lifecycle.py` are not modified in this patch, the failure occurs before prediction or CSV normalisation, and the fixture creates `test_race_1.csv` / `test_race_2.csv` without filename date, venue, race number, sidecar metadata, or future race-time evidence. The current Flask route enumerates live upcoming files through the live lifecycle filter, so those synthetic fixture filenames are filtered out and `total_races` is 0.
- I did not repair this fixture in this patch because doing so would be outside the approved delimiter-normalisation diff and would not change the live TheDogs normalisation contract.

Endpoint health during closeout:

- port 5002: no listener
- `/api/health`: connection refused
- `/api/model_health`: connection refused

No-persist and no-mutation confirmation:

- no persist
- no result ingestion
- no label writes
- no retrain
- no model promotion
- no push
- no betting
- no fake odds
- no fake EV
- no snapshot rewrite
- no manifest append

Next gated task, not executed here:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir <fresh_post_normalization_upcoming_races_dir> \
  --snapshot-dir artifacts/prediction_snapshots \
  --output artifacts/full_evidence_orchestration_20260525/post_delimiter_contract_verified_live_batch/dry_run_capture_report.json
```

Persist remains approval-gated and must only follow a fresh dry-run that proves
READY verified pre-jump races:

```bash
.venv/bin/python scripts/capture_prediction_snapshot.py \
  --db greyhound_racing_data_writable.db \
  --upcoming-dir <fresh_post_normalization_upcoming_races_dir> \
  --snapshot-dir artifacts/prediction_snapshots \
  --persist \
  --output artifacts/full_evidence_orchestration_20260525/post_delimiter_contract_verified_live_batch/capture_report.json
```
