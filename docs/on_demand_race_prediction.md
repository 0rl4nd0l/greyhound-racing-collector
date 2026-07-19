# On-demand race prediction

`scripts/predict_race_now.py` attempts one research-only prediction immediately
for one exact named upcoming race. It does not wait for a timer, persist a
production prediction, append the production database, emit betting output, or
change a collector, service, timer, model pointer, or registry.

```bash
uv run scripts/predict_race_now.py \
  --race "gunnedah r5" \
  --model latest-research \
  --config configs/prediction/manual-default.json \
  --odds-source auto
```

The race query must resolve to exactly one upcoming race with an exact jump
timestamp, and scoring must finish before that timestamp. Missing or ambiguous
identity and post-jump requests fail closed.

## Odds source and collector concurrency

`--odds-source auto` first asks the reviewed receipt reader from local dependency
`021889ab` for one finalized, current-window autonomous receipt. That reader
binds the report, plan, form, sidecar, runner set, source URL, timestamps, and
matching WIN/PLACE rows through query-only SQLite access.

If no reusable receipt exists, the command tries the existing collector lock.
It never deletes, steals, cancels, or treats a stale lock as permission to
bypass its owner. While the lock is busy, the bounded config wait can notice a
receipt completed by the collector. If the wait expires, the command returns
`BUSY`. With the lock legitimately acquired, it refreshes only the exact race,
fetches and validates current Sportsbet WIN and PLACE markets, and writes the
accepted capture only into the private run bundle. The production database is
never passed to an append path.

Use `receipt` to prohibit direct capture or `capture` to require immediate
isolated capture. Direct capture is an explicit network operation performed by
the command; it still respects the shared lock.

## Feature cutoff

For the residual model, the source database is opened with SQLite URI
`mode=ro` and `PRAGMA query_only=ON`. Before feature materialization, the command
creates a private SQLite database containing only uniquely identified history
whose `race_date` is strictly before the target jump date. The exact target
race, every same-day row, every future row, and ambiguous dates are excluded.
Relevant runner history without a unique dated race identity is a hard blocker.
The reviewed feature builder then reads only this sealed database. Its
target-row and post-outcome provenance counters must remain zero.

This deliberately excludes same-day history when the source schema cannot
prove a precise pre-jump timestamp. It trades some feature coverage for a
fail-closed temporal boundary.

## Models and canonical config

Supported selectors are finite:

- `market-only`, `market-only-implied`, and `market_only_implied` resolve to
  `market_only_v1` and the `market_only_implied` baseline.
- `latest-research`, `market-form-residual-v1`, and
  `market_form_residual_v1` resolve to the frozen
  `market_form_residual_v1` artifact.

Every alias resolves before config validation. The config file must already be
canonical JSON (sorted compact keys plus a final newline), must match the
resolved model's checked-in schema, and must contain no unsupported fields.
`latest-research` records the exact model, manifest, schema, and config SHA-256
values. Config selects only the frozen `full_strength` or `half_strength`
variant; the command never rewrites coefficients.

The checked-in examples are:

- `configs/prediction/manual-default.json`
- `configs/prediction/market-only.json`

## Output and replay

Stdout is one canonical JSON object. A successful object has
`status: PREDICTION_READY`, `research_only: true`,
`production_persisted: false`, the exact race/model/config/runner identities,
probabilities, and an absolute private bundle path. Normal failures use one
stable blocker in `blockers`; the CLI exits 2.

The bundle contains the request, canonical config, model schema, copied frozen
artifacts when applicable, receipt/capture provenance, source form and sidecar,
sealed history database and audit, sealed features, result, and a hash manifest.
Replay is offline and rejects changed, missing, or added file bytes:

```bash
uv run scripts/predict_race_now.py \
  --replay-bundle /absolute/path/to/prediction_...
```

Market-only replay recomputes and compares probabilities. Residual replay
verifies the complete sealed input identity, reruns the frozen artifact scorer
at the recorded score timestamp, and compares the selected prediction exactly.

Running a live prediction still requires the owner to name the race and
explicitly authorize that execution. Building, testing, or reviewing the
command does not provide live-execution authority.
