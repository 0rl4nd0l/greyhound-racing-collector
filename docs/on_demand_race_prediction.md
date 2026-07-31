# On-demand race prediction

`scripts/predict_race_now.py` attempts one research-only prediction immediately
for one exact named upcoming race. It does not wait for a timer, persist a
production prediction, append the production database, emit betting output, or
change a collector, service, timer, model pointer, or registry.

```bash
uv run --no-project scripts/predict_race_now.py \
  --race "gunnedah r5" \
  --model latest-research \
  --config configs/prediction/manual-default.json \
  --odds-source auto
```

The selector may be an exact named race, canonical race ID, or canonical
TheDogs race URL. Named selectors are resolved only against today's bounded
index (maximum 160 races, 12-second default deadline), and scoring must finish
before the exact jump timestamp. Missing or ambiguous identity, discovery
overrun, and post-jump requests fail closed.

## Odds source and collector concurrency

Race resolution reads one collector-owned
`shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json`
packet beneath the first `--capture-evidence-root` (or the explicit
`--current-race-index`). A successful scheduled odds-refresh cycle atomically
replaces this fixed packet with at most 32 selected races and seals the source
refresh-report path and SHA-256. Publication occurs immediately after that
bounded refresh completes, before the slower multi-race odds-capture batch, so
a later batch timeout cannot leave discovery dependent on an older index. The
predictor validates the packet schema,
canonical bytes, source path and hash, exact TheDogs identities, timezone-aware
jump timestamps, uniqueness, row bound, and configured 1,200-second maximum
source age. It does not browse, scan race/evidence directories, publish a timer
request, or wait for a timer boundary during discovery. Missing, stale, changed,
or invalid packets fail closed before a browser starts.

`--odds-source auto` first performs a direct race-keyed lookup for a current
sealed receipt. The append-only exact-receipt index binds the request, exact
race identity, runner set, source file paths and hashes, capture attempt, append
report, and pre-jump timestamp. Lookup is confined to that race's directory and
rejects unsafe paths, changed bytes, malformed records, or more than 32 indexed
receipts. `receipt` mode is reuse-only.

Successful scheduled appends also publish
`collector-exact-capture-receipt-v1` records immediately from inside the same
validated collector capture loop. Each record binds the scheduled collector run,
exact TheDogs URL/date/race number, canonical race-ID aliases, runner-set hash,
append timestamp, capture attempt and append report, and immutable form,
sidecar, and per-attempt source-report bytes. The predictor checks the bounded
directory for the resolved exact race before requesting a new acquisition.
This allows a current scheduled capture to be reused while the scheduled batch
still owns the canonical lock; it does not create another browser, wait for a
timer boundary, accept an unsealed database row, or weaken `BUSY`.

If no reusable receipt exists, `auto` or `capture` atomically publishes one
research-only, one-attempt request under
`artifacts/full_evidence_orchestration_20260525/manual_prediction_collector_requests_v1`.
The immutable request binds its UUID, exact TheDogs race identity and URL,
venue, race number, date, jump, creation and expiry times, requested output
schemas and statuses, and the expected runner set when available. Only one
unexpired unanswered manual request is supported.

The predictor then synchronously invokes
`scripts/shadow_autopilot_daemon.py capture-one`. This collector entry point is
the only acquisition authority: it claims the request, checks the computed
pre-jump margin, attempts the canonical daemon lock once without stealing or
waiting, refreshes only the resolved TheDogs URL, builds one canonical
fixed-window plan item, and calls the scheduled collector's existing validated
Sportsbet fetch and append-only `live_odds` persistence code. The background
timer remains unchanged and is never used as interactive transport.

Contention returns `BUSY` immediately with the existing owner run ID, PID,
hostname, start time, output directory, and phase where available. No service
or timer control is attempted. The collector writes one attempt marker before
capture and exactly one terminal response. Existing V1 protocol statuses
remain:

- `RECEIPT_READY`
- `REQUEST_EXPIRED`
- `RACE_NOT_FOUND`
- `CAPTURE_WINDOW_CLOSED`
- `IDENTITY_MISMATCH`
- `CAPTURE_FAILED`

The synchronous result additionally distinguishes `BUSY`, `CANCELLED`, and
`INSUFFICIENT_PREJUMP_MARGIN`; the preserved V1 protocol records those
conditions as `CAPTURE_FAILED` with a deterministic reason.

`RECEIPT_READY` points to a sealed protocol receipt that binds the request and
claim hashes, exact race, runner identities and runner-set hash, capture and
response timestamps, source evidence hashes, canonical capture attempt and
append report hashes, and the exact refreshed form/sidecar bytes. The predictor
receives the terminal result synchronously, consumes it atomically once,
rediscovers that race-keyed receipt, verifies it against the sealed response,
and then scores exactly once in its isolated research bundle. A receipt sealed
before cancellation remains reusable. Cancellation terminates and reaps the
collector process group so its browser cannot be orphaned; cancellation before
sealing returns `CANCELLED` and terminalizes the protocol request.

The checked-in latency budget is enforced as six explicit components:
discovery 12 seconds, lock 1, capture 60, validation 8, scoring 30, and safety
15. A fresh capture therefore requires more than 114 seconds after resolution
(126 seconds including discovery), while receipt reuse requires more than 53
seconds. Discovery is one bounded local packet-and-source validation within its
12-second wall-clock budget; index freshness is independently capped at 1,200
seconds. Failure occurs before the Sportsbet browser starts when the applicable
margin is not available. The collector recomputes the remaining margin after
lock acquisition and again after exact TheDogs refresh, immediately before the
Sportsbet fetch.

The exact TheDogs meeting slug remains authoritative during named-race
selection. Murray Bridge and Murray Bridge Straight therefore remain distinct
meeting identities even where downstream compatibility data uses `MURR` for
both; a shared-code query that matches both is ambiguous and fails closed.

Use `receipt` to prohibit request publication. `capture` uses the same
collector-owned synchronous path as `auto` and still reuses an exact valid
receipt first. The predictor contains no browser, capture, database append, or
lock implementation.

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

List the complete finite catalog, including resolved model/config/artifact
hashes, without selecting a race:

```bash
uv run --no-project scripts/predict_race_now.py --list-configs
```

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
uv run --no-project scripts/predict_race_now.py \
  --replay-bundle /absolute/path/to/prediction_...
```

Market-only replay recomputes and compares probabilities. Residual replay
verifies the complete sealed input identity, reruns the frozen artifact scorer
at the recorded score timestamp, and compares the selected prediction exactly.

Running a live prediction still requires the owner to name the race and
explicitly authorize that execution. Building, testing, or reviewing the
command does not provide live-execution authority.
