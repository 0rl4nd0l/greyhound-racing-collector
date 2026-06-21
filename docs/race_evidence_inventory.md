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

## Weather And Track Metadata

Weather and track condition are collected into each refreshed upcoming CSV
sidecar. They are accepted only when the sidecar proves source-safe pre-jump
metadata.

- Weather source: `open_meteo_forecast_api` via `utils/prejump_weather.py`.
- Track-condition source: `sportsbet_pre_race_page` via
  `utils/prejump_sportsbet.py`.
- The venue map in `utils/prejump_weather.py` is also the timezone map used by
  Sportsbet track matching and expert-form metadata.

The live daemon code checkout is:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-release-clean-20260621`

The live daemon Python/runtime environment is:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python`

Use a fresh sibling worktree from `origin/master` for merge and review work.
A master-only patch will not affect the systemd daemon until it is deployed
through the service path.

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
