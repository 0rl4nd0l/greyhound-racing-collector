# Greyhound Daemon Orchestrator Decision Packet

Generated: 2026-06-16T17:22:33+10:00

## Decision

`DATA_MISSING`

The patched deterministic reserve-remap policy is report-only ready for a controlled backlog rejoin subset, but the full daemon readiness objective is still blocked. Current-cycle official result rows are absent, cumulative pending remains nonzero, and strict replay still leaves unresolved unsafe joins.

Sub-decision: `REPORT_ONLY_BACKLOG_REJOIN_READY_FOR_RESERVE_SUBSET`

## Current State

- Primary checkout: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector`
- Primary HEAD: `554bea2875ad41f55b2f100cd0f26ea358f83d0b`
- Runtime checkout: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610`
- Runtime HEAD: `2bbd35743fbb00373d132a25687c4d2f158aabe4`
- Latest completed full packet: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_20260616T170209+1000/`
- Newer full attempt: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_20260616T171700+1000/`
- Newer full attempt status: incomplete, lock-held, `DAEMON_WAITING_FOR_ODDS_CAPTURE_LOCK`

## Counts

Latest completed join:

- Safe joined races: `0`
- Pending races: `15`
- Unsafe races: `0`
- Pending reason: `official_result_rows_not_present=15`

Cumulative aggregate before replay:

- Safe joined races: `442`
- Safe runner rows: `3117`
- Pending races: `82`
- Unsafe races: `70`
- Pending reasons: `official_result_rows_not_present=66`, `no_race_url_available_for_lookup=16`

Main-agent dry replay of the 70 unsafe rows under current patched identity policy:

- Safe after replay: `502`
- Pending after replay: `82`
- Unsafe after replay: `10`
- Delta: `+60 safe`, `0 pending`, `-60 unsafe`
- Accepted reserve remaps: `66`
- Rejected reserve remaps: `0`
- Remaining unsafe reasons: `winner_count_not_exactly_one=6`, `extra_official_non_scratch_boxes_outside_prediction_set=5`, `dog_name_mismatch_after_exact_badge_stripping=1`

## Official Result Availability

Cumulative pending classification from the report-only audit:

- `32` races have extracted official rows but remain pending in aggregate joins.
- `18` races now have parseable page rows but are absent from the latest official-result capture artifact.
- `16` races have pages available but no official finish rows yet.
- `16` races have no race URL available for lookup.

The latest 15 current-cycle pending races were classified at artifact time as `race_not_jumped:upcoming_not_jumped`.

## Feature Gate

`same_distance_same_grade_best_time` and `same_distance_same_grade_avg_time` remain quarantined.

- Train coverage for both: `0/751`
- Holdout coverage for both: `10/192`
- Current same-distance provenance audit: strict prior-race-only path passes.
- Activation blockers: all-missing in train, train rows below minimum, train pct below minimum, train unique values below minimum, train/holdout ratio unstable, inactive by train-all-missing policy, missing shadow metric comparison.

Minimum activation conditions remain:

- Train present rows `>=30`
- Train present pct `>=5%`
- Train unique values `>=5`
- Stable train/holdout ratio
- Fresh candidate metric comparison
- Strict history-before-race provenance with no target-race rows, no result fields, no market or odds features

## No-Mutation Statement

No training, promotion, model registry mutation, label write, snapshot mutation, manifest mutation, production pointer write, betting action, EV action, or daemon trigger was performed by this orchestrator closeout.

The already enabled append-only live odds capture lane belongs to the running service state and was not manually triggered here.

## Verification

- Primary tracked status: clean before packet creation except preserved untracked artifacts.
- Runtime tracked status: clean; large pre-existing untracked artifact surface preserved.
- Services point at runtime checkout.
- `uv run --with-requirements requirements/requirements.lock python -m py_compile scripts/ingest_results_for_date.py scripts/join_forward_shadow_results.py scripts/shadow_autopilot_v1.py scripts/shadow_autopilot_daemon.py`: pass.
- `uv run --with-requirements requirements/requirements.lock pytest -q tests/test_results_ingest_official_first.py::test_result_validation_rejects_official_boxes_outside_local_participants tests/test_join_forward_shadow_results.py::test_classify_join_rejects_fuzzy_name_mismatch tests/test_shadow_autopilot_daemon.py::test_feature_activation_gate_status_from_autopilot_packet tests/test_shadow_autopilot_v1.py::test_summary_surfaces_feature_activation_gate_status`: pass, `4 passed`.
- `git diff --check`: pass.
- Main-agent dry replay matched subagent replay: 70 unsafe inputs, 60 safe after replay, 10 unsafe after replay, 66 reserve remaps accepted, 0 rejected.

## Next Safe Action

Run a controlled report-only backlog rejoin packet using the patched deterministic reserve-remap policy, and separately refresh official-result capture/report-only availability for the 18 parseable-but-not-captured pending races. Keep same-distance/same-grade timing features quarantined until train coverage and metric-comparison gates pass.
