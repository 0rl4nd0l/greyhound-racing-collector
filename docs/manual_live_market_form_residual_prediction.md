# Manual pre-jump frozen residual prediction

`scripts/predict_market_form_residual.py` turns already captured, verified
pre-jump evidence for one race into one outcome-free ranking. It is an offline
operator command over the frozen `market_form_residual_v1` artifact. It does
not fetch a race, open a database, write a file, append a shadow record, change
a service, activate a model, or place a bet.

## Required evidence

- A TheDogs form CSV with its canonical adjacent
  `<form CSV>.metadata.json` sidecar.
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
  --form-csv '/evidence/odds_capture_refreshed_upcoming/Race 2 - SAN - 2026-07-16.csv' \
  --capture '/evidence/autonomous_live_odds_capture_report.json'
```

The sidecar is resolved only at the adjacent canonical path. `--sidecar` may
be supplied for clarity, but a non-adjacent path is rejected.

When an evidence tree contains one semantic form/sidecar pair and one semantic
accepted capture, discovery is also available:

```bash
python scripts/predict_market_form_residual.py \
  --race-id 'Race 2 - SAN - 2026-07-16' \
  --evidence-root '/evidence/full_evidence_orchestration_20260525'
```

Byte-identical duplicate copies are resolved by a stable path ordering.
Semantically different matching forms or captures fail as ambiguous; the
command never chooses between them using outcomes or filesystem modification
times.

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
- probabilities that do not meet the frozen scorer contract.

The canonical model and manifest paths are fixed in the command and cannot be
overridden. Tests may inject a historical score timestamp to replay a sealed
fixture, but the CLI deliberately has no backdating option.

## Separate activation boundary

This command consumes evidence that another approved capture path has already
materialized. Automatic exact-event discovery and network fetching, scheduler
or collector integration, database access, shadow persistence, deployment and
model activation remain separately gated. Any later activation must descend
from the PR #45 resource-isolation changes as well as the frozen-model lineage.
