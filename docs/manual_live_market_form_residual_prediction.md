# Manual pre-jump frozen residual prediction

`scripts/predict_market_form_residual.py` turns already captured, verified
pre-jump evidence for one race into one outcome-free ranking. It is an offline
operator command over the frozen `market_form_residual_v1` artifact. It does
not fetch a race, open a database, write a file, append a shadow record, change
a service, activate a model, or place a bet.

## Required evidence

- A TheDogs form CSV with its canonical adjacent
  `<form CSV>.metadata.json` sidecar.
- A `shadow_feature_rows.json` packet with its adjacent `shadow_manifest.json`
  and `implementation_file_manifest.json`. The implementation manifest must
  identify the reviewed PR #45 feature generator at head `aa35fa70fc49`.
- An autonomous Sportsbet capture report, single-attempt JSON, or attempts
  JSONL containing exactly one `APPENDED` / `PASS` attempt for the race.
- A race ID in the canonical form `Race <number> - <venue code> - YYYY-MM-DD`.

The command binds the race ID, race number, venue, date, runner boxes and
normalized names across both sources. A verified scratched runner may remain
in the complete form guide but must be declared scratched by the accepted
capture and is not scored.

## Explicit invocation

```bash
python scripts/predict_market_form_residual.py \
  --race-id 'Race 2 - SAN - 2026-07-16' \
  --form-csv '/evidence/daily_run/eligible_inputs/source_0002/Race 2 - SAN - 2026-07-16.csv' \
  --feature-rows '/evidence/daily_run/shadow_score_live/shadow_feature_rows.json' \
  --capture '/evidence/autonomous_live_odds_capture_report.json'
```

The sidecar is resolved only at the adjacent canonical path. `--sidecar` may
be supplied for clarity, but a non-adjacent path is rejected.
The two feature manifests are resolved only beside `--feature-rows`.
`--feature-manifest` and `--implementation-manifest` may be supplied for
clarity, but non-adjacent paths are rejected. This version is deliberately an
explicit-input scorer: it does not discover or generate a missing feature
packet from a race name alone.

## Output and failure contract

Success is one canonical JSON object on stdout. It includes the frozen model
and manifest hashes, complete runner-set hash, raw and selected-source hashes,
capture/freeze/jump timestamps, normalized market/half/full probabilities, and
the ranking. `activation`, `persisted`, and `outcomes_present` are all `false`.

Validation errors return exit code 2 and one canonical
`BLOCKED_MANUAL_PREDICTION` JSON object on stderr. The command fails closed on,
among other things:

- a non-TheDogs sidecar or non-Sportsbet capture URL;
- unsafe metadata, incomplete runners, box/name drift, or race-ID drift;
- missing, extra, duplicate, ambiguously captured, or invalid-odds runners;
- undeclared or inconsistent scratches;
- naive, future, post-jump, or incorrectly ordered source timestamps;
- a current execution time at or after jump;
- any outcome-like field in the sidecar or capture artifact;
- target-day form rows or a frozen artifact schema/hash mismatch;
- feature packets not generated at the reviewed PR #45 head, or packet,
  manifest, source-path, byte-count or SHA-256 drift;
- probabilities that do not meet the frozen scorer contract.

The canonical model and manifest paths are fixed in the command and cannot be
overridden. Tests may inject a historical score timestamp to replay a sealed
fixture, but the CLI deliberately has no backdating option.

## Separate activation boundary

This command consumes evidence that another approved capture path has already
materialized. Race-only discovery, missing-feature generation, network
fetching, scheduler or collector integration, database access, shadow
persistence, deployment and model activation remain separately gated. Any
later activation must descend from the PR #45 resource-isolation changes as
well as the frozen-model lineage.
