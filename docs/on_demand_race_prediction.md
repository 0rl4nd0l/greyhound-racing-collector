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

The race query must resolve to exactly one upcoming race with an exact jump
timestamp, and scoring must finish before that timestamp. Missing or ambiguous
identity and post-jump requests fail closed.

## Odds source and collector concurrency

`--odds-source auto` first asks master's PR #56 scorer to discover and validate
one finalized, current sealed pre-jump packet. The packet is accepted only when
its race identity, date, race number, distance, grade, jump aliases, runner set,
form/sidecar provenance, capture report, record-V3 checksum, and
effective-state-V2 hash agree. The command then copies only the verified packet
bytes into its private bundle; it does not rescrape or cross a database writer.

If the discovered packet predates the exact target-grade proof contract or its
implementation hashes do not match the current source, `auto` records that
candidate and continues to the collector-request path. This exception is
limited to the known pre-current packet fields and implementation seal:
ambiguous, conflicting, malformed, post-jump, or otherwise invalid evidence
remains terminal. `receipt` mode is always reuse-only.

If no reusable receipt exists and a fixed capture window is due, `auto`
atomically publishes one research-only, one-attempt request under
`artifacts/full_evidence_orchestration_20260525/manual_prediction_collector_requests_v1`.
The immutable request binds its UUID, exact TheDogs race identity and URL,
venue, race number, date, jump, creation and expiry times, requested output
schemas and statuses, and the expected runner set when available. Only one
unexpired unanswered manual request is supported.

The scheduled collector remains the sole browser and capture-lock authority.
At the start of a collector cycle, after its daemon parent has acquired the
existing shared lock, the daemon passes that exact resolved lock path to its
direct autopilot child. The child validates the non-symlink regular file,
parent PID, hostname, and run ID before it may atomically claim one request.
Direct compatibility invocations without an explicit path continue to infer
the lock beneath the evidence root. It never checks or claims during an active
capture. The normal refresh and fixed-window plan put
the requested exact race first, but the request cannot bypass race, URL, jump,
runner, time-window, Sportsbet validation, or append-only persistence checks.
The collector writes one attempt marker before capture and exactly one terminal
response:

- `RECEIPT_READY`
- `REQUEST_EXPIRED`
- `RACE_NOT_FOUND`
- `CAPTURE_WINDOW_CLOSED`
- `IDENTITY_MISMATCH`
- `CAPTURE_FAILED`

`RECEIPT_READY` points to a sealed protocol receipt that binds the request and
claim hashes, exact race, runner identities and runner-set hash, capture and
response timestamps, source evidence hashes, record and effective-state hashes,
and the verified master-packet handoff. The predictor waits on a monotonic
finite deadline (600 seconds in checked-in configs, with a 900-second schema
maximum), consumes
the terminal response atomically once, rediscovers the exact packet, verifies
it against the sealed response, and then enters the unchanged receipt
validation and scoring path. Timeout, duplicate claim/attempt/response/consume,
replay, malformed or unknown records, identity disagreement, expiry, post-jump
state, and hash drift fail closed.

The exact TheDogs meeting slug remains authoritative during named-race
selection. Murray Bridge and Murray Bridge Straight therefore remain distinct
meeting identities even where downstream compatibility data uses `MURR` for
both; a shared-code query that matches both is ambiguous and fails closed.

Use `receipt` to prohibit request publication. `capture` is retained only as an
explicit fail-closed selector and returns `CAPTURE_AUTHORITY_FORBIDDEN`; the
manual predictor never starts a second capture process or acquires the shared
collector lock.

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
