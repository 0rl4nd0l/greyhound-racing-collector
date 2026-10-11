# Retained live prediction benchmark: outcome-blind protocol

Frozen before performance inspection on 11 October 2026. Scope is the user's retained, nonreserved operational and development forecasts and matching official results. No fitting, acquisition, production changes or protected evaluation access.

## Population and selection

Read allocation/access metadata first. Exclude reserved race identities and windows, frozen prospective evaluations and 169 withheld historical targets before opening their outcomes. Operational labels do not override reservations. Keep one exclusion per discovered forecast and separate source-level census/accounting.

Primary model is the installed R3 `market_form_residual_v1`, identified by exact artifact/configuration hashes, never a newly replayed forecast. Other genuinely contemporaneously sealed frozen versions are separate secondary comparisons. Deferred snapshots and retrospective reconstructions cannot enter the live comparison.

Require original verified sealed forecast, complete original active field, complete contemporaneous fixed WIN odds, retained input identities, exact race/runner identity and authorised official result. Odds must have been captured by the forecast information cutoff; completion and durable forecast verification must precede the recorded jump. Acquisition timestamps measure capture age, not provider quote age when the latter is absent. Missing provider publication time is explicitly reported. Do not replace prices with SP, later quotes or outcome-reconstructed fields.

For native one-terminal-per-race operations use that terminal's verified forecast. If a source allows refreshes, primary is its last successfully sealed pre-jump forecast (latest generated time, then sealed time, then lexical prediction ID), selected without outcome access; retain the earlier horizons separately. Do not replace an outcome-ineligible selected forecast with an earlier better-joining forecast.

Require exact unchanged active runner identities and boxes at result closure. Scratches/substitutions or unresolved fields remain explicit exclusions; do not renormalize/remove original runners. DNF/FELL/DISQ remain losing original runners. Follow native quarantine rules for ambiguous outcomes; confirmed dead heats are diagnostic only, with fractional winner targets if the native source verifies them. Cancelled/no-result races do not receive artificial scores.

## Scores and uncertainty

Market q is inverse decimal WIN odds normalized across the complete sealed field. Preserve odds and sum of inverse odds (overround factor), plus excess above one. Race log loss is negative log winner probability; zero probability gives infinite loss rather than a post-hoc floor. Race Brier is sum of squared probability-minus-target residuals. Average races equally. Top accuracy divides credit among exactly tied maxima. Differences are always model minus market; negative loss differences favour model.

Report every race, all unchanged/differing top sets, both gain/loss directions, all dates and versions. Calibration uses ten fixed equal-width runner probability bins, counts, mean forecast and observed frequency; race-level weighting and runner-weighted ECE are labelled. No new calibration is fitted. Date tables and leave-one-date-out sensitivity accompany paired race differences. Whole-source-date bootstrap (10,000 draws, seed 20261011) is descriptive and only reported for at least two dates; one-date uncertainty is unidentified. Few dates and repeated dog appearances prevent claims of independent confirmation. No exploratory subgroup becomes a deployed rule.

Report usable-forecast conditional performance separately from operating-scope opportunities, failures, exclusions and unresolved attempts. Discovery alone is not a service promise. Preserve historical coverage denominators and snapshot times; never combine overlapping ledgers by addition.

## Reproduction and checks

Retain hash-bound references, allocation decisions, selected forecasts, runner dataset, excluded records, per-race scores, date sensitivity and original errors/derived corrections. Test rejected identity, field, probability, chronology, result and allocation cases. Independently recompute metrics and audit joins after integration. A zero strict population must still have a machine-readable empty scorecard with null metrics and exact reconciled causes; useful authorised diagnostic comparisons remain separately labelled.
