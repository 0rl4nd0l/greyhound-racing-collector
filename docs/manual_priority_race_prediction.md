# Manual priority race prediction

`scripts/run_priority_race_prediction.py` is a standalone, outcome-free
operator command for one exact pre-jump race. It does not extend or activate
the shadow daemon. The default is plan-only: it reads the upcoming schedule,
resolves one race, prints canonical JSON, and performs no refresh, odds
capture, feature write, prediction write, service change, or timer change.

Plan by stable race identity:

```bash
python scripts/run_priority_race_prediction.py \
  --race-id 'Race 7 - SAN - 2026-07-18'
```

Plan by an exact named-race query:

```bash
python scripts/run_priority_race_prediction.py --race 'Sandown race 7'
```

Collection is deliberately double gated. Both flags are required:

```bash
python scripts/run_priority_race_prediction.py \
  --race-id 'Race 7 - SAN - 2026-07-18' \
  --execute-collection \
  --allow-auto-scrape-odds \
  --db /absolute/path/to/greyhound_racing_data.db
```

Execution uses the existing shared daemon lock and may wait for it only for the
bounded `--max-wait-seconds` interval (0 to 600 seconds). It refreshes only the
resolved race, admits only strict WIN and PLACE captures in the existing fixed
T-60, T-30, T-10, or T-2 windows, validates the exact active runner/box set,
and uses the reviewed append-only/idempotent collector. It creates a fresh
hash-bound feature packet in a temporary directory, scores it before jump, and
deletes that packet when the command exits. Only the canonical JSON prediction
is printed; the full/half prediction is never persisted.

Successful predictions report `PREDICTION_READY`. Normal non-success statuses
are `WAITING_FOR_DAEMON_LOCK`, `WAITING_FOR_CAPTURE_WINDOW`,
`BLOCKED_RACE_NOT_FOUND`, `BLOCKED_RACE_AMBIGUOUS`,
`BLOCKED_RACE_ALREADY_JUMPED`, `BLOCKED_EXACT_METADATA`,
`BLOCKED_RUNNER_IDENTITY`, `BLOCKED_ODDS_CAPTURE`,
`BLOCKED_FEATURE_SEAL`, and `BLOCKED_MANUAL_PREDICTION`.

An already complete capture is never appended again. The command reports the
idempotency gate through `BLOCKED_ODDS_CAPTURE` with
`idempotent_existing_capture: true`; it does not reconstruct a prediction from
mutable database rows without the original accepted capture artifact.

This command never reads target outcomes or ingests results. It does not train,
tune, replace model artifacts, alter thresholds, assign a cohort cutoff,
promote, bet, deploy, merge, or modify services and timers. PR #48 remains a
legacy runtime surface and is not read from or mutated by this command. Its
activation or retirement requires a separate owner-approved lane.
