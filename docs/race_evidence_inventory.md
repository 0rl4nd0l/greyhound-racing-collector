# Race Evidence Inventory

Use this when checking what races the system already has. Do not infer coverage
from one shadow run or one audit packet.

## Canonical Stores

- Shadow predictions and feature rows:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_*`
- Official-result capture artifacts:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_*`
- Append-only official-result evidence DB tables:
  `autonomous_official_result_evidence_races`
  `autonomous_official_result_evidence_runners`
- Strict pre-jump odds DB table:
  `live_odds`
- Main DB path used by the runtime:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db`

## Installed Runtime Truth

Do not infer the active code checkout from this document, a Git branch, or an
older report. The installed user units are authoritative:

```bash
systemctl --user cat shadow-autopilot.service
systemctl --user cat shadow-autopilot-odds-capture.service
systemctl --user show shadow-autopilot.service -p WorkingDirectory -p ActiveState -p SubState -p Result
systemctl --user show shadow-autopilot-odds-capture.service -p WorkingDirectory -p ActiveState -p SubState -p Result
```

The generated `ops/systemd/*.service` files and installed files under
`~/.config/systemd/user/` must be byte-identical before enabling timers. Record
the deployed commit, unit hashes, DB/model/evidence paths and rollback evidence
in the current runtime-reconciliation deployment manifest.

An odds window is complete only when the same capture group contains validated
dog-level WIN and PLACE rows for the complete active runner set. Historical
WIN-only groups are incomplete and may be recaptured append-only; they must not
be reported as complete. PLACE is appended first with `topN=3`, then WIN. A
failed PLACE append prevents the WIN append for that attempt.

## Recent PR Boundaries

Last verified on 2026-07-01 against PR #33 merge commit `cb869022`.

Recent source patches changed evidence wiring, not production readiness:

- PR #28 keeps strict pre-jump evidence from accepting DB `live_odds` rows at
  or after jump time.
- PR #32 lets automatic historical source selection include approved append
  unified-evidence roots and resolve artifact-root-relative `dataset_jsonl`
  paths.
- PR #33 lets report builders and shadow-autopilot wrappers use a retained
  `--evidence-root` while keeping artifact-prefix output guards in force.

These patches do not train, promote, deploy, restart services, mutate the live
DB, update model registry pointers, approve EV actions, or approve betting.
Promotion and high-accuracy decisions still have to come from the current
matching evidence root's rolling comparison, promotion-distance report, and
high-accuracy packet.
Raw DB counts, older successful packets, and daemon tick watching are not
substitutes for those gates.

## Shadow Autopilot Evidence Root

Use `--evidence-root` when an already approved report-only workflow needs to
read or write retained shadow-autopilot artifacts outside the source checkout,
for example:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`

The evidence root is an artifact parent, not a runtime pointer. Passing it to a
script must not be treated as permission to capture more official results, start
or restart the daemon, write the live DB, train, promote, or perform betting
actions.

Output guards allow report output only under either:

- the repo-local default `artifacts/full_evidence_orchestration_20260525/...`
- the supplied `--evidence-root`

The child output directory must still use the expected artifact prefix, such as
`shadow_autopilot_v1_`, `daily_race_ingest_shadow_`,
`shadow_odds_snapshot_`, `autonomous_official_result_capture_`,
`unified_evidence_dataset_`, `rolling_model_comparison_`,
`promotion_distance_report_`, `high_accuracy_refinement_packet_`,
`forward_shadow_result_join_`, `forward_shadow_result_aggregate_`,
`forward_shadow_status_`, or `shadow_feature_activation_gate_`. An absolute
retained root does not permit arbitrary output paths.

Common CLIs in the current evidence-root chain include:

- `scripts/shadow_autopilot_v1.py --evidence-root`
- `scripts/shadow_autopilot_daemon.py run-once --evidence-root`
- `scripts/shadow_autopilot_daemon.py run-odds-capture-only --evidence-root`
- `scripts/shadow_autopilot_daemon.py write-service-files --evidence-root`
- `scripts/shadow_autopilot_daemon.py write-odds-capture-service-files --evidence-root`
- `scripts/daily_race_ingest_shadow_orchestrator.py --output-parent`
- `scripts/collect_shadow_odds_snapshots.py --evidence-root`
- `scripts/autonomous_live_odds_capture.py --evidence-root`
- `scripts/autonomous_official_result_capture.py --evidence-root`
- `scripts/build_unified_evidence_dataset.py --evidence-root`
- `scripts/build_rolling_model_comparison_packet.py --evidence-root`
- `scripts/build_promotion_distance_report.py --evidence-root`
- `scripts/build_high_accuracy_refinement_packet.py --evidence-root`
- `scripts/build_pre_race_gated_challenger_packet.py --evidence-root`
- `scripts/join_forward_shadow_results.py --evidence-root`
- `scripts/aggregate_forward_shadow_results.py --evidence-root`
- `scripts/forward_shadow_status_report.py --evidence-root`
- `scripts/shadow_feature_activation_gate.py --evidence-root`
- `scripts/run_shadow_non_tgr_rf_evaluation.py live --evidence-root`

A source commit on `master` does not change the installed user service or the
active runtime checkout. Re-verify the systemd unit path before reasoning about
live daemon behavior.

## Daily And Autopilot Odds Status Fields

PR #33 adds sample-scoped odds diagnostics to `DAILY_STATUS.json` and
`DAILY_STATUS.md`:

- `prediction_sample_odds_coverage_status`
- `prediction_sample_odds_coverage_blocker`
- `prediction_sample_odds_expected_races`
- `prediction_sample_odds_complete_prejump_races`
- `prediction_sample_odds_missing_prejump_races`
- `prediction_sample_odds_coverage_report`
- `autonomous_live_odds_capture_scope_status`
- `autonomous_live_odds_capture_scope_gap_races`

The shadow-autopilot command return payload and `verification_results.txt`
surface the compact summary fields:

- `prediction_sample_odds_coverage_status`
- `prediction_sample_odds_missing_prejump_races`
- `autonomous_live_odds_capture_scope_status`
- `autonomous_live_odds_capture_scope_gap_races`

The full daemon reads the autopilot cycle's `DAILY_STATUS.json` and carries
status forward into daemon artifacts with `autopilot_cycle_*` fields and daemon
runtime state with `last_*` fields. Treat these as pass-through status pointers
to the autopilot cycle, not as a separate readiness gate or permission to mutate
runtime state.

Interpret these against the current prediction sample only. A complete raw
`live_odds` table, a large historical evidence root, or a later odds-only
daemon packet does not by itself prove that the current prediction sample has
complete strict pre-jump odds.

Status values are intentionally gate-shaped:

- `PASS_COMPLETE_PREJUMP_ODDS` means the prediction sample has complete valid
  pre-jump odds.
- `BLOCKED_MISSING_PREJUMP_ODDS` means at least one prediction-sample race is
  missing complete valid pre-jump odds.
- `DATA_MISSING_NO_PREDICTION_SAMPLE` means there was no scored prediction
  sample to evaluate.
- `PASS_AUTONOMOUS_ODDS_SCOPE_COVERS_SAMPLE` means autonomous odds capture
  readiness covers the prediction sample.
- `PARTIAL_AUTONOMOUS_ODDS_SCOPE` means autonomous odds capture readiness covers
  some, but not all, of the prediction sample.
- `BLOCKED_NO_AUTONOMOUS_ODDS_SCOPE` means autonomous odds capture readiness
  covers none of the prediction sample.

## Backlog Append

`scripts/append_official_result_evidence_backlog.py` accepts either exact
official-result capture directories or the parent evidence root. When the parent
root is passed, it recursively discovers child capture directories containing
both:

- `official_result_races.jsonl`
- `official_result_runners.jsonl`

Report-only inventory:

```bash
python3 scripts/append_official_result_evidence_backlog.py \
  --artifact-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --db /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db \
  --output-dir artifacts/full_evidence_orchestration_20260525/official_result_evidence_append_backlog_<timestamp>_report_only
```

Approved append-only DB ingest:

```bash
python3 scripts/append_official_result_evidence_backlog.py \
  --artifact-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --db /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db \
  --output-dir artifacts/full_evidence_orchestration_20260525/official_result_evidence_append_backlog_<timestamp>_execute \
  --execute-db-ingest \
  --require-lock-free \
  --lock-path /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock
```

This writes only the append-only official-result evidence tables. It does not
write canonical labels, train models, promote models, update registries, emit EV,
or place bets.

## Quick Counts

```bash
sqlite3 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db "
SELECT 'official_result_evidence_races', COUNT(DISTINCT race_id)
FROM autonomous_official_result_evidence_runners;
SELECT 'live_odds_races', COUNT(DISTINCT race_id)
FROM live_odds
WHERE race_id IS NOT NULL AND TRIM(race_id) != '';
"
```

To prove evaluation readiness, build a unified evidence dataset and check rows
with all three sources: shadow prediction, official-result evidence, and strict
pre-jump odds.

Current cross-store inventory:

```bash
python3 scripts/build_race_evidence_inventory_packet.py \
  --artifact-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --db /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db \
  --output-dir artifacts/full_evidence_orchestration_20260525/race_evidence_inventory_<timestamp>_report_only
```

This packet is report-only. It tells future agents exactly which race IDs exist
in shadow predictions, official-result artifacts, official-result evidence DB
tables, and strict pre-jump odds, plus the next action for each race.

It also writes `race_evidence_scorecard.csv` for races that are complete across
shadow prediction, official-result evidence DB, and strict pre-jump odds. Use
that scorecard to compare model Top1/Top3, winner rank, and logloss against the
market before proposing training, promotion, EV, or betting changes.

The scorecard metrics include both broad skip reasons and action-level gap
counts:

- `skipped_race_reason_counts` keeps stable broad buckets, such as
  `official_result_incomplete_for_shadow_boxes`.
- `skipped_race_action_counts` maps skipped races to the row-level
  `recommended_next_action`, so the denominator loss is split into actions such
  as `capture_official_result`,
  `repair_official_result_runner_set_or_identity_join`, or
  `collect_future_strict_prejump_odds`.
- `official_result_gap_action_counts` and `strict_odds_gap_action_counts`
  isolate the result and odds coverage classes that should drive the next
  bounded repair.

## Weather And Track Metadata

Weather and track condition are collected into each refreshed upcoming CSV
sidecar. They are accepted only when the sidecar proves source-safe pre-jump
metadata.

- Weather source: `open_meteo_forecast_api` via `utils/prejump_weather.py`.
- Track-condition source: `sportsbet_pre_race_page` via
  `utils/prejump_sportsbet.py`.
- The venue map in `utils/prejump_weather.py` is also the timezone map used by
  Sportsbet track matching and expert-form metadata.

The live daemon Python/runtime environment is:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python`

Use a fresh sibling worktree from `origin/master` for merge and review work.
A master-only patch will not affect the systemd daemon until it is deployed
through the service path. Always use the installed-unit commands above to find
the current runtime checkout.

Q/The Q aliases are mapped to the Purga/Ipswich venue family:

- `QOT`, `Q-STRAIGHT`, `LADBROKES-Q-STRAIGHT`
- `Q1-LAKESIDE`, `LADBROKES-Q1-LAKESIDE`
- `Q2-PARKLANDS`, `LADBROKES-Q2-PARKLANDS`
- `THE-Q`

Latest Q1 validation artifact:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/q_venue_metadata_alias_validation_20260619T162736+1000_collector_probe_report_only/`

That probe used an already collected Q1 sidecar and confirmed:

- `venue_weather_location("LADBROKES-Q1-LAKESIDE")` resolves to
  `Australia/Brisbane`.
- Open-Meteo returned source-safe weather.
- Sportsbet matched `Q1 Lakeside` and returned source-safe
  `track_condition=Good`.
