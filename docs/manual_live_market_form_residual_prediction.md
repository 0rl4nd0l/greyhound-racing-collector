# Manual pre-jump frozen residual prediction

`scripts/predict_market_form_residual.py` turns already captured, verified
pre-jump evidence for one race into one outcome-free ranking. It is an offline
operator command over the frozen `market_form_residual_v1` artifact. It does
not fetch a race, open a database, change a service, activate a model, or place
a bet. By default it writes nothing. The explicit `--append-shadow-output`
mode may append only the same canonical outcome-free record to a `.jsonl` file
whose parent already exists.

## Required evidence

- A TheDogs form CSV with its canonical adjacent
  `<form CSV>.metadata.json` sidecar.
- A `shadow_feature_rows.json` packet with its adjacent `shadow_manifest.json`
  and `implementation_file_manifest.json`. The implementation manifest must
  either match the exact sealed legacy PR #45 packet (feature rows
  `0ebecaf9...` and implementation manifest `9822a77a...`), or bind the exact
  current generator, local feature dependencies, and test file by SHA-256.
  Merely declaring the legacy branch/head is not sufficient.
- An autonomous Sportsbet capture report, single-attempt JSON, or attempts
  JSONL containing exactly one `APPENDED` / `PASS` attempt for the race.
- A race ID in the canonical form `Race <number> - <venue code> - YYYY-MM-DD`.

The command binds the race ID, race number, venue, date, runner boxes and
normalized names across both sources. A verified scratched runner may remain
in the complete form guide but must be declared scratched by the accepted
capture and is not scored.

## Race-first invocation

From the canonical checkout, the normal operator command is:

```bash
python scripts/predict_market_form_residual.py --race 'sandown r6'
```

This searches only the default outcome-free evidence root. An external or
additional evidence root can be named explicitly and repeated:

```bash
python scripts/predict_market_form_residual.py \
  --race 'sandown r6' \
  --evidence-root '/evidence/full_evidence_orchestration_20260525'
```

The query must contain a venue and race number. Venue matching accepts the
canonical venue code, the verified TheDogs URL venue, and a close spelling
only when the race resolves unambiguously. Discovery selects the nearest
current/future race date, the latest exact feature packet for that race, and
the latest strict accepted capture. Equal latest candidates fail closed.

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
clarity, but non-adjacent paths are rejected.

The scorer never generates features. In a full approved autopilot cycle, a
successful strict capture whose race was omitted by the smaller primary
refresh is now handed to the existing daily feature generator as one
supplemental form input. A fresh cutoff is taken immediately before feature
generation, so a race that jumped while odds were being captured is excluded.
The primary refresh limit and resource-isolation settings are unchanged.

The odds-only daemon consumes that exact handoff immediately after capture,
while it still owns the shared lock. For each newly appended capture it runs
the existing feature generator through its read-only SQLite path, then invokes
the frozen residual scorer with explicit artifact paths and:

```bash
--append-shadow-output \
  /evidence/full_evidence_orchestration_20260525/market_form_residual_shadow_predictions_v1.jsonl
```

The append is idempotent for an exact replay and rejects a conflicting record.
Feature or scoring failure is reported as an early-residual blocker without
changing the odds-capture result. The lock is released only after this bounded
stage finishes or fails closed. Timer frequency and fixed capture windows do
not change.

## Output and failure contract

Success is one canonical JSON object on stdout. It includes the frozen model
and manifest hashes, complete runner-set hash, raw and selected-source hashes,
capture/freeze/jump timestamps, normalized market/half/full probabilities, and
the ranking. `activation` and `outcomes_present` are always `false`. In the
default mode `persisted` is `false`. With `--append-shadow-output`, `persisted`
is `true` and `persistence_status` is `APPENDED` or `EXACT_REPLAY`.

Validation errors return exit code 2 and one canonical
`BLOCKED_MANUAL_PREDICTION` JSON object on stderr. The command fails closed on,
among other things:

- a non-TheDogs sidecar or non-Sportsbet capture URL;
- unsafe metadata, incomplete runners, box/name drift, or race-ID drift;
- missing, extra, duplicate, ambiguously captured, or invalid-odds runners;
- undeclared or inconsistent scratches;
- naive, future, post-jump, or incorrectly ordered source timestamps; feature
  generation and odds capture may occur in either order, but each independent
  timeline must remain complete and pre-jump;
- a current execution time at or after jump;
- any outcome-like field in the sidecar or capture artifact;
- target-day form rows or a frozen artifact schema/hash mismatch;
- feature packets that neither match the exact sealed legacy PR #45 packet nor
  every declared current generator/dependency-file SHA-256, or packet,
  manifest, source-path, byte-count or SHA-256 drift;
- probabilities that do not meet the frozen scorer contract.

The canonical model and manifest paths are fixed in the command and cannot be
overridden. Tests may inject a historical score timestamp to replay a sealed
fixture, but the CLI deliberately has no backdating option.

## Runtime boundary

This command still consumes only already materialized evidence. The daemon,
not the scorer, owns capture and feature generation. The feature generator's
database access remains SQLite `mode=ro`; the scorer has no database or network
path. The append-only JSONL is shadow evidence, not a production prediction
pointer. Results, labels, outcomes, cohort assignment, betting, promotion,
production model activation and merge remain outside this runtime path. The
installed runtime must retain PR #45 resource isolation, the frozen-model
lineage, and the reviewed PR #47 handoff repair.
