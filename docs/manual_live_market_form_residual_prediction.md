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
  either match the exact sealed legacy PR #45 packet (feature rows
  `0ebecaf9...` and implementation manifest `9822a77a...`), or bind the exact
  current generator, local feature dependencies, and test file by SHA-256.
  Merely declaring the legacy branch/head is not sufficient.
- An autonomous Sportsbet capture report, single-attempt JSON, or attempts
  JSONL containing exactly one `APPENDED` / `PASS` attempt for the race.
- A race ID in the canonical form `Race <number> - <venue code> - YYYY-MM-DD`.
  Venue codes may contain uppercase letters, digits, underscores, and internal
  single hyphens, for example `LADBROKES-Q1-LAKESIDE`.

The command binds the race ID, race number, venue, date, runner boxes and
normalized names across both sources. A verified scratched runner may remain
in the complete form guide but must be declared scratched by the accepted
capture and is not scored.

## Race-first invocation

From the canonical checkout, the normal operator command is:

```bash
uv run --offline scripts/predict_market_form_residual.py --race 'sandown r6'
```

The script declares its frozen NumPy dependency inline, so this command does
not depend on a checkout-local virtual environment or a `python` shell alias.
`--offline` prohibits package-registry access; the command fails before scoring
if the declared Python and NumPy dependency are not already in the uv cache.

By default this searches the checkout-local outcome-free evidence root plus the
sealed packet directories named by finalized outcome-free status indexes in
the existing retained system evidence root. Only indexes from the preceding
36 hours are eligible. Unsafe, outcome-bearing, future-dated or path-escaping
indexes fail closed. This avoids a recursive search of the large retained root
and does not read the production database. The index is only a bounded search
hint: the scorer still validates and hashes the selected sidecar, feature
packet, implementation manifest and odds capture, and those emitted hashes are
the prediction's provenance authority. An external or additional evidence root
can still be named explicitly and repeated:

```bash
uv run --offline scripts/predict_market_form_residual.py \
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
uv run --offline scripts/predict_market_form_residual.py \
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

Target grade comparison uses an immutable finite table derived from the
non-conflicting generator maps, their canonical values and the bounded exact
labels already admitted by preprocessing and observed in sealed packets. This
makes source-proven labels such as `Restricted` / `Restricted Win` and
`5th Grade` / `Grade 5` equivalent. Exact ordinal-composite labels remain
distinct from `Mixed` labels. Bare `M` is rejected because the generator maps
disagree on its meaning. The scorer preserves identity-bearing punctuation,
rejects non-string or unknown grades and does not treat genuinely different
known grades as equal. Syntactically grade-like but undeclared values such as
`Grade 999` and `NG14` also fail closed.

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

## Separate activation boundary

This command consumes evidence that an approved capture and feature path has
already materialized. This change does not deploy or activate the repaired
handoff. Network fetching, direct database access, service/unit/timer changes,
shadow persistence, deployment and model activation remain separately gated.
Any later activation must descend from the PR #45 resource-isolation changes,
the final reviewed PR #46 effective-state repair, and the reviewed handoff and
input-contract repair.
