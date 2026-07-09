# Source-Backed Cumulative Accuracy Odds Closeout

## Scope

- Worktree: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610`
- Baseline checkout preserved: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector`
- Mode: bounded source-backed report/code lane after the point-in-time runtime snapshot; no live timer chasing.
- No DB, label, odds, registry, production pointer, model replacement, TGR, EV, betting, snapshot, or manifest write was performed by this closeout task.

## Artifacts Produced

- Rebuilt source-aware cumulative unified inputs:
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_001`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_002`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_003`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_004`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_005`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_006`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_007`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_008`
  - `unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_009`
- Cumulative backlog status:
  - `shadow_autopilot_v1_20260614T125217+1000_source_backed_cumulative_backlog_status/backlog_unified_evidence_datasets_status.json`
- Cumulative rolling comparison:
  - `rolling_model_comparison_20260614T125217+1000_source_backed_cumulative/rolling_model_comparison_report.json`
- Promotion-distance diagnostic:
  - `promotion_distance_report_20260614T125217+1000_source_backed_cumulative/promotion_distance_report.json`
- High-accuracy packet:
  - `high_accuracy_refinement_packet_20260614T125217+1000_source_backed_cumulative/high_accuracy_refinement_packet.json`
  - `high_accuracy_refinement_packet_20260614T125217+1000_source_backed_cumulative/promotion_pr_gate.json`

## Evidence Counts

- Rebuilt source-aware cumulative deduped eligible races: `112`.
- Rolling sample: `112` races, `801` runner rows, review floor `100`, races needed `0`.
- Backlog aggregate: `9` datasets, `801` eligible runner rows.
- Official-result coverage source: `deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids`.
- Official-result requested races: `134`.
- Official-result races with rows: `112`.
- Official-result missing races: `22`.
- Official-result missing exclusions: `193`.

## Gate Results

- Rolling comparison: `ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`.
- Sample floor: passed.
- Best rank candidate: `market_only_implied`.
- Best non-market candidate: `stage2_market_blend_50`.
- Best non-market minus market:
  - Top1: `0.0`
  - Top3: `-0.044642857142857095`
  - Mean winner rank: `0.2142857142857144`
  - Brier: `0.011892784247862775`
  - Logloss: `0.05580569938761393`
  - Box1 top-pick share: `0.008928571428571425`
- Promotion distance: `PROMOTION_DISTANCE_BLOCKED`.
- Promotion blockers:
  - `no_candidate_passed_rank_first_accuracy_gate`
  - `best_non_market_top1_margin_below_target`
  - `predeclared_residual_trigger_count_below_directional_floor`
  - `predeclared_residual_top1_not_above_market`
- High-accuracy final status: `BLOCKED_KEEP_BASELINE`.
- Promotion PR gate: `BLOCKED`.
- Protected paths unchanged: `true`.

## Key Commands Run

```bash
python3 scripts/build_unified_evidence_dataset.py --shadow-run-dir artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_20260611T205925+1000_daemon_autopilot --db /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db --output-dir artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_009 --official-result-runners-jsonl artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_20260612T193211+1000_daemon/backlog_unified_evidence_inputs/official_result_runners_backlog_033.jsonl

python3 scripts/build_rolling_model_comparison_packet.py --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_001/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_002/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_003/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_004/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_005/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_006/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_007/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_008/unified_evidence_dataset_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T125217+1000_source_backed_cumulative_009/unified_evidence_dataset_report.json --output-dir artifacts/full_evidence_orchestration_20260525/rolling_model_comparison_20260614T125217+1000_source_backed_cumulative

python3 scripts/build_promotion_distance_report.py --rolling-report artifacts/full_evidence_orchestration_20260525/rolling_model_comparison_20260614T125217+1000_source_backed_cumulative/rolling_model_comparison_report.json --pre-race-gated-report artifacts/full_evidence_orchestration_20260525/pre_race_gated_challenger_20260614T110200+1000_daemon_autopilot/pre_race_gated_challenger_report.json --high-accuracy-gate artifacts/full_evidence_orchestration_20260525/high_accuracy_refinement_packet_20260614T124343+1000_source_backed_above_floor_backlog_005/promotion_pr_gate.json --output-dir artifacts/full_evidence_orchestration_20260525/promotion_distance_report_20260614T125217+1000_source_backed_cumulative

python3 scripts/build_high_accuracy_refinement_packet.py --stage2-predictions artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_20260613T193731+1000_daemon_autopilot/stage2_shadow_predictions.jsonl --odds-gate-report artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_20260613T193731+1000_daemon_autopilot/odds_research_gate_report.json --odds-augmented-report artifacts/full_evidence_orchestration_20260525/rolling_model_comparison_20260614T125217+1000_source_backed_cumulative/rolling_model_comparison_report.json --unified-evidence-report artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_20260614T124343+1000_coverage_source_refresh_backlog_005/unified_evidence_dataset_report.json --backlog-unified-evidence-status artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_20260614T125217+1000_source_backed_cumulative_backlog_status/backlog_unified_evidence_datasets_status.json --promotion-distance-report artifacts/full_evidence_orchestration_20260525/promotion_distance_report_20260614T125217+1000_source_backed_cumulative/promotion_distance_report.json --reserve-substitution-preflight artifacts/full_evidence_orchestration_20260525/official_result_reserve_substitution_preflight_20260614T110200+1000_daemon_autopilot/official_result_reserve_substitution_preflight.json --timing-aligned-rerun-plan artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_20260614T110200+1000_daemon/timing_aligned_prediction_rerun_plan.json --timing-aligned-rerun-execution-status artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_20260614T110200+1000_daemon/timing_aligned_prediction_rerun_execution_status.json --output-dir artifacts/full_evidence_orchestration_20260525/high_accuracy_refinement_packet_20260614T125217+1000_source_backed_cumulative

python3 -m py_compile scripts/build_unified_evidence_dataset.py scripts/shadow_autopilot_v1.py scripts/build_high_accuracy_refinement_packet.py scripts/build_rolling_model_comparison_packet.py scripts/build_promotion_distance_report.py

git diff --check -- scripts/build_unified_evidence_dataset.py scripts/shadow_autopilot_v1.py scripts/build_high_accuracy_refinement_packet.py scripts/build_rolling_model_comparison_packet.py scripts/build_promotion_distance_report.py tests/test_build_unified_evidence_dataset.py tests/test_shadow_autopilot_v1.py tests/test_build_high_accuracy_refinement_packet.py tests/test_build_rolling_model_comparison_packet.py tests/test_build_promotion_distance_report.py

uv run --with pytest pytest -q tests/test_build_unified_evidence_dataset.py tests/test_shadow_autopilot_v1.py tests/test_build_high_accuracy_refinement_packet.py tests/test_build_rolling_model_comparison_packet.py tests/test_build_promotion_distance_report.py -q

timeout 120s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_autonomous_live_odds_capture.py -q -k 'fixed_window or existing_capture or stale or superset'

timeout 180s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_shadow_autopilot_daemon.py -q -k 'defer_decision or lock or odds_capture_timer or write_odds_capture_service_files or service_and_timer_define'

timeout 240s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_build_unified_evidence_dataset.py tests/test_shadow_autopilot_v1.py tests/test_build_high_accuracy_refinement_packet.py tests/test_build_rolling_model_comparison_packet.py tests/test_build_promotion_distance_report.py -q
```

## Verification

- `py_compile`: passed.
- `git diff --check`: passed.
- `python3 -m pytest`: blocked because `/usr/bin/python3` has no `pytest`.
- `pytest`: blocked because no `pytest` executable is on `PATH`.
- `uv run --with pytest pytest ...`: blocked at test collection because `tests/conftest.py` imports `flask`, and the isolated uv environment did not include Flask. The repo `.venv` also lacks `flask`, `pandas`, `sklearn`, and `pytest`.
- `timeout 120s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_autonomous_live_odds_capture.py -q -k 'fixed_window or existing_capture or stale or superset'`: passed, `6` selected tests.
- `timeout 180s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_shadow_autopilot_daemon.py -q -k 'defer_decision or lock or odds_capture_timer or write_odds_capture_service_files or service_and_timer_define'`: passed, `27` selected tests.
- `timeout 240s uv run --with-requirements requirements/requirements.lock pytest -q tests/test_build_unified_evidence_dataset.py tests/test_shadow_autopilot_v1.py tests/test_build_high_accuracy_refinement_packet.py tests/test_build_rolling_model_comparison_packet.py tests/test_build_promotion_distance_report.py -q`: passed.

## Remaining Blockers

- This lane now has a source-backed cumulative rolling packet above the 100-race review floor.
- It is not promotion-ready. Market-only remains the best rank candidate, and the best non-market challenger does not beat the market benchmark on the rank-first criteria.
- Official-result coverage is still incomplete across requested source-backed cumulative races: 22 requested race IDs have no official-result rows and were excluded rather than backfilled.
- Predeclared residual evidence is still underpowered: 2 triggered races, 8 more needed for the directional-read floor.
