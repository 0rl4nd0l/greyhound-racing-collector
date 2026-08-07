# Manual scoring readiness index

`race_collection.manual_scoring_readiness.publish_manual_scoring_readiness_index`
publishes the separate `manual_prediction_scoring_readiness_index.json` beside
the collector runtime state. It is a pre-capture, research-only index. It does
not replace or alter `manual_prediction_current_race_index.json` and it does
not capture odds, write the database, start a browser, or invoke scoring.

The source refresh report is read as canonical JSON and retained while its
selected races are validated. A bad race is recorded in `exclusions` with its
source identity and stable `reason_code`; a source, schema, model/config,
identity, path, or publication fault rejects the whole new packet and leaves
the prior readiness bytes untouched.

Eligible races carry their exact race identity, ordered active runner set,
source CSV/sidecar hashes, and pre-jump timing. Odds are intentionally absent
and marked `PENDING_GHU_051`; GHU-051 is the authorized next one-race capture
step. GHU-052/GHU-053/PR #125 remain authoritative after capture and must build
the complete canonical scoring input, including strict odds and timestamps,
before any research prediction can be produced.
