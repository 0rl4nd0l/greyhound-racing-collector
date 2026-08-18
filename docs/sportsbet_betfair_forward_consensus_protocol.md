# Sportsbet + Betfair forward consensus protocol

Terminal preparation state:
`BLOCKED_NO_OUTCOME_FREE_BETFAIR_SCHEDULED_OFF_SOURCE`.

The replacement prospective interval is 2026-08-20 through 2026-09-30
inclusive. Its
model is permanently fixed at 95% normalized Betfair
`BEST_AVAIL_BACK_AT_SCHEDULED_OFF` probability plus 5% normalized corrected
Sportsbet WIN probability. Betfair-only was marginally better on development
validation, but the selected rule must not be changed or rescreened.

The predecessor interval beginning 2026-08-18 is
`COMPROMISED_FOR_PRISTINE_CONFIRMATION`. This status is recorded only by the
replacement manifest. The predecessor freeze directory remains immutable and
may be secondary forward evidence, but it cannot support an untouched-forward-
outcomes claim.

At the replacement freeze time, 2026-08-18T20:27:06+10:00, the 2026-08-20
start remained future, zero replacement population rows existed, and zero
replacement outcomes had been inspected. October begins 2026-10-01 and is not
part of this cohort.

## Two-phase evidence boundary

Population sealing and scoring are separate commands in
`scripts/evaluate_frozen_sportsbet_betfair_forward.py`.

The only currently verified first-party Betfair monthly surface contains result
fields. Reading or projecting that surface would inspect outcome-bearing rows,
so it is forbidden to this replacement protocol. The evaluator deliberately
has no command that converts a raw monthly response. Collection cannot begin
until an independently outcome-free source supplies exactly the nine frozen
identity/predictor columns plus a truthful, externally reviewed provenance
receipt. No such source or collector was available at freeze.

1. `seal-population` may run only after both Betfair scheduled-off predictor
   projections are available and a label-blind corrected Sportsbet completeness receipt covers
   the whole fixed interval. A separate closed-schema, label-blind Betfair source
   manifest receipt must bind the exact August and September filenames, official
   HTTPS URLs, byte sizes, and SHA-256 values. It uses only date, frozen venue
   alias, race number, exact scheduled clock, complete box/TAB agreement, native
   Betfair market and selection IDs, corrected Sportsbet WIN probability, and
   Betfair scheduled-off back price. Scheduled clocks may be second precision or
   end in `.000`; other fractional values fail closed. It records every
   predictor-only exclusion and emits no metric. Each supplied Betfair CSV must
   have exactly the nine frozen predictor/identity columns. A raw monthly CSV,
   any result/BSP/actual-off column, or any unknown column fails before the first
   data row is parsed. Raw result-bearing monthly files are never admissible to
   this evaluator and must stay outside normal diagnostic/reporting surfaces.
2. `score` may run once only after 2026-09-30, after the sealed population is
   immutable, and under separate authority to read an approved result projection.
   Before results are opened, an external reviewer must provide the closed-schema
   population approval receipt bound to the exact `population_manifest.json`
   SHA-256. The Melbourne calendar gate, external receipt, manifest validation,
   and durable exclusive consumed marker are enforced by the single result-
   opening score operation. Generic JSONL parsing rejects the result schema
   before opening its path. The command writes only the sibling evaluation filename derived
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
forbidden. A predecessor official CSV may contain those columns, but such a file
is not admissible to the replacement sealer. The replacement accepts only a
predictor-only closed-schema projection containing the explicit scheduled-off
predictor and identity whitelist.

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
of this protocol. Scoring remains impossible until after 2026-09-30 Melbourne
time and a separate external authorization receipt is supplied.

The frozen candidate and future window are therefore prepared, but the honest
terminal verdict is blocked rather than ready: a pristine collection path for
the required Betfair scheduled-off predictor does not presently exist.
