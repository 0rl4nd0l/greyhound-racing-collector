# AGENTS.md - Greyhound Runtime Evidence Guide

Before claiming races, odds, results, weather, track condition, or prediction
coverage, build or read the race evidence inventory. Do not infer coverage from
one shadow run, one result packet, or one daemon artifact.

## Canonical Runtime Paths

- Runtime checkout used by systemd:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610`
- Master merge checkout:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-master-merge-autonomous-accuracy-odds-v1-20260618`
- Runtime evidence root:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`
- Runtime DB:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db`

Master-only patches do not affect the live daemon until they are applied to the
runtime checkout or deployed through the service path.

## Race Inventory First

Use `docs/race_evidence_inventory.md` and
`scripts/build_race_evidence_inventory_packet.py` to answer:

- which races have shadow predictions
- which races have official-result artifacts
- which races have appended official-result evidence in the DB
- which races have strict pre-jump Sportsbet odds
- which races are ready for model-vs-market evaluation
- which races still need result capture, odds collection, or runner-set repair

The inventory packet is report-only. It does not train, promote, label, bet,
rewrite snapshots, or mutate registries.

## Live Daemon Safety

The live daemon and odds-capture daemon coordinate through:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock`

Do not delete or bypass that lock while its PID is alive. Odds capture uses
browser scraping and may hold the lock for several minutes.

Weather and track condition must come from source-safe pre-jump sidecars:

- Weather: Open-Meteo via `utils/prejump_weather.py`
- Track condition: Sportsbet pre-race page via `utils/prejump_sportsbet.py`
- Expert form: TheDogs expert-form collection metadata

Placeholder/default values are not evidence.

## Promotion Boundary

The current post-backlog evidence shows the model trails the market on complete
races. Do not train, promote, emit EV, or place bets unless a fresh report-only
evaluation proves model quality is better than the market on the declared slice.
