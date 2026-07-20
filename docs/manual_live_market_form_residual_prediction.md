# Manual live residual prediction handoff

This handoff scores one named upcoming race with the frozen market/form residual
model already on `master`. It is outcome-free and read-only: it does not append
shadow history, write a database, activate a service, migrate v1/v2 records,
emit EV, or place a bet.

## Evidence contract

Scoring requires one sealed packet whose form CSV, adjacent sidecar, feature
rows, feature manifest, implementation manifest, and pre-jump odds capture all
agree on the same race and canonical runner set. The sidecar grade is admitted
only when its exact TheDogs meeting-card proof agrees on:

- canonical race and meeting-card URLs;
- date, venue, race number, normalized grade, and grade proof key;
- canonical final runner boxes and names; and
- the hashes of the meeting card, CSV, packet artifacts, and capture artifact.

Cached grades, race-name substrings, multi-race card scopes, conflicting grades,
URL or runner mismatches, post-jump material, outcome-shaped fields, and PR #51
`FORM_ONLY_V1` trainer/control/sealed paths fail closed. Missing evidence is not
filled from another race; race-first discovery returns a stable
`race_feature_packet_quarantined:<reason>` error when an upstream quarantine
report proves why the named race has no scoreable packet.

## Run it

Use race-first discovery when the sealed packet is present beneath an
outcome-free evidence root:

```bash
uv run --script scripts/predict_market_form_residual.py \
  --race "sandown r6" \
  --evidence-root /path/to/outcome-free/evidence
```

Or pass the six exact artifacts explicitly:

```bash
uv run --script scripts/predict_market_form_residual.py \
  --race-id "Race 6 - SAN - 2026-07-20" \
  --form-csv /path/Race\ 6\ -\ SAN\ -\ 2026-07-20.csv \
  --feature-rows /path/shadow_feature_rows.json \
  --capture /path/autonomous_live_odds_capture_report.json
```

The adjacent sidecar, feature manifest, and implementation manifest are inferred
from their canonical names unless supplied explicitly. Model and manifest paths
cannot be overridden.

Successful stdout is one canonical JSON object. It identifies the frozen model
and manifest, record schema `market_form_residual_shadow_record_v3`, effective
state schema `market_form_residual_effective_state_v2`, portable numerical
canonicalization contract, record/effective-state checksums, verified input
hashes, grade proof, and canonical box-ordered runners.

`persistence_status` is `NOT_REQUESTED_READ_ONLY`. If a separate authorized
caller later crosses the master writer boundary, it must handle all three
possible statuses: `APPENDED`, `EXACT_REPLAY`, and `COMMIT_STATE_UNKNOWN`.
