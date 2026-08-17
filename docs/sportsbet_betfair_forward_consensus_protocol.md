# Sportsbet + Betfair forward consensus protocol

Terminal preparation state: `FORWARD_CONSENSUS_TEST_FROZEN_NOT_SCORED`.

The prospective interval is 2026-08-18 through 2026-09-30 inclusive. Its
model is permanently fixed at 95% normalized Betfair
`BEST_AVAIL_BACK_AT_SCHEDULED_OFF` probability plus 5% normalized corrected
Sportsbet WIN probability. Betfair-only was marginally better on development
validation, but the selected rule must not be changed or rescreened.

## Two-phase evidence boundary

Population sealing and scoring are separate commands in
`scripts/evaluate_frozen_sportsbet_betfair_forward.py`.

1. `seal-population` may run only after both official Betfair monthly files are
   available and a label-blind corrected Sportsbet completeness receipt covers
   the whole fixed interval. A separate closed-schema, label-blind Betfair source
   manifest receipt must bind the exact August and September filenames, official
   HTTPS URLs, byte sizes, and SHA-256 values. It uses only date, frozen venue
   alias, race number, exact scheduled clock, complete box/TAB agreement, native
   Betfair market and selection IDs, corrected Sportsbet WIN probability, and
   Betfair scheduled-off back price. Scheduled clocks may be second precision or
   end in `.000`; other fractional values fail closed. It records every
   predictor-only exclusion and emits no metric.
2. `score` may run once only after 2026-09-30, after the sealed population is
   immutable, and under separate authority to read an approved result projection.
   Before results are opened, an external reviewer must provide the closed-schema
   population approval receipt bound to the exact `population_manifest.json`
   SHA-256. The command creates a durable exclusive consumed marker before opening
   the result projection and writes only the sibling evaluation filename derived
   from that approved manifest SHA-256;
   a failed attempt remains consumed and cannot be retried without new authority.
   Results must exactly equal the sealed race set; no missing or extra result race
   is accepted.

Both phases fail closed on duplicate rows, partial or mismatched runner sets,
reserve numbers, ambiguous markets, unknown aliases, invalid prices, changed
frozen artifacts, reused outputs, or result-population drift. Names are only a
post-identity corroboration check and are never an identity key.

The source and population receipts are evidence inputs, not self-approval. Their
truth and external approval must be established and durably recorded outside the
evaluator before use. The evaluator validates their closed schemas and hashes but
does not confer approval itself.

## Forbidden evidence

BSP, `ACTUAL_OFF_TIME`, WIN result fields in the Betfair predictor file,
matched-price aggregates, in-play values, names as identity, reserves,
replacement heuristics, post-jump predictors, and any new model or weight are
forbidden. The official CSV can contain those columns, but the population
sealer retains indexes only for the explicit scheduled-off predictor and
identity whitelist.

The scheduled-off field supports only the literal first-party claim that it is
the best available back quote at scheduled off. It does not prove a minimum
available size, executable liquidity, virtual-price treatment, an official jump
timestamp, or profitability.

## Frozen analysis

The primary statistic is paired mean multiclass race log loss, consensus minus
Sportsbet. Secondary metrics are paired Brier, fixed ten-bin top-label
calibration, top-1/top-2/top-3 accuracy, and winner rank. Meeting-date clusters
are `race_date|sportsbet_venue`; the percentile bootstrap uses 10,000 replicates
and seed 20260817. Prospective confirmation requires both delta log loss below
zero and the cluster-bootstrap 95% upper bound below zero.

No interim analysis, results-driven stopping, population repair after labels,
ROI/EV analysis, refit, promotion, deployment, or automated prediction is part
of this protocol.
