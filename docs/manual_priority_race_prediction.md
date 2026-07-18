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

Execution first looks for a finalized autonomous capture for the exact race and
the currently due T-60, T-30, T-10, or T-2 window. It checks again while the
shared writer lock is busy, but only for the bounded `--max-wait-seconds`
interval (0 to 600 seconds). Discovery waits for the producer's finalized
`AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED` marker before parsing the paired report,
so an in-progress report write is never mistaken for a bad receipt. A reusable
capture must have one exact target plan, one original `APPENDED`/`PASS` attempt,
its capture-time form and sidecar, and
complete matching WIN and PLACE rows in one query-only SQLite snapshot. The
report, plan, form, and sidecar are read once, hash-sealed into a private
temporary directory, and checked against exact race, venue, runner, provenance,
append, and fixed-window timestamps.

The reuse path never acquires or releases the writer lock, refreshes a race,
scrapes Sportsbet, or writes the database. It reports `capture_reused: true`,
`inserted_live_odds_rows: 0`, the fixed window, source hashes, exact DB row
count/hash, and `consistency_claim: HASH_SEALED_DB_BOUND_AT_USE_TIME`.
This proves that the finalized report and append-only database rows matched at
use time; it is not cryptographic authentication against a coordinated earlier
rewrite of both local surfaces.

For a guaranteed read-only proof, add `--require-autonomous-handoff`. This mode
polls only for an already-finalized exact receipt and cannot fall back to the
direct writer-lock, refresh, or capture path:

```bash
python scripts/run_priority_race_prediction.py \
  --race 'Sandown race 7' \
  --execute-collection \
  --allow-auto-scrape-odds \
  --require-autonomous-handoff \
  --max-wait-seconds 30
```

If no reusable receipt exists, execution retains the original direct path: it
acquires the existing shared lock, refreshes only the resolved race, admits only
strict WIN and PLACE captures in the fixed windows, validates the exact active
runner/box set, and uses the reviewed append-only/idempotent collector. Both
paths create a fresh hash-bound feature packet in a temporary directory, score
before jump, and delete the packet when the command exits. Only canonical JSON
is printed; the full/half prediction is never persisted by the command.

By default the receipt search checks this checkout's evidence root and the
retained autonomous runtime evidence root. Override or add bounded roots with a
repeatable option:

```bash
python scripts/run_priority_race_prediction.py \
  --race 'Sandown race 7' \
  --execute-collection \
  --allow-auto-scrape-odds \
  --capture-evidence-root /absolute/path/to/evidence-root
```

Successful predictions report `PREDICTION_READY`. Normal non-success statuses
are `WAITING_FOR_DAEMON_LOCK`, `WAITING_FOR_CAPTURE_WINDOW`,
`BLOCKED_RACE_NOT_FOUND`, `BLOCKED_RACE_AMBIGUOUS`,
`BLOCKED_RACE_ALREADY_JUMPED`, `BLOCKED_EXACT_METADATA`,
`BLOCKED_RUNNER_IDENTITY`, `BLOCKED_ODDS_CAPTURE`,
`BLOCKED_FEATURE_SEAL`, and `BLOCKED_MANUAL_PREDICTION`.

An already complete capture is never appended again. A skip-only report is not
a receipt; reuse must find the original finalized `APPENDED` attempt and bind it
to the exact database group. If the original artifact is absent, the command
reports the existing idempotency block rather than reconstructing a prediction
from database rows alone.

This command never reads target outcomes or ingests results. It does not train,
tune, replace model artifacts, alter thresholds, assign a cohort cutoff,
promote, bet, deploy, merge, or modify services and timers. PR #48 remains a
read-only legacy code/worktree/service/timer surface. The command may consume a
finalized autonomous capture produced in the retained evidence root, but never
uses PR #48's progress JSONL or persisted shadow-prediction path and never
modifies that runtime. Activation or retirement requires a separate
owner-approved lane.
