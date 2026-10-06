# Historical sectional candidate, October 6

This separately versioned exploratory construction leaves the strict historical
first-sectional calculation unchanged. It accepts the user's working meaning of
`1 SEC` as an individual dog's first sectional, measured in seconds. It does not
claim provider certification or that standardized times prove physical equivalence
between different sectional endpoints.

## Pure interface and responsibility

`build_sectional_candidate(packet)` in
`race_collection/sectional_speed_candidate.py` is a pure function. The packet has:

- `target`: `race_id`, source racing `date`, timezone-aware `cutoff`.
- `roster`: ordered `runner_id`, authenticated stable `identity_id` (or null), and
  `identity_available_at` (or null). Optional adapter metadata is preserved.
- `observations`: authenticated runner identity, event identity/proxy, date,
  availability, literal and canonical track, distance, sectional cell,
  conflict fingerprint, source bindings, and optional layout/era or alias proof.

The adapter establishes identity, alias evidence, input allocation and timestamps.
Availability must be the latest dependency needed to establish an observation and
its identity or mapping, not merely the historical race date. A stable dog ID
learned on a later card cannot retrospectively authorize an earlier join.
The module does not fetch, fit, open labels or grant authority. Shared malformed
schema or contradictory roster identities fail closed with fixed error codes.

Source racing dates must use the calendar timezone represented by the target
cutoff. Date-only histories require a strictly earlier source racing date. Evidence
and identity availability must be strictly before the cutoff. Same-day history is
excluded because its ordering cannot be established from date-only cells. The
source date may precede the cutoff's local date for a meeting continuing after
midnight.

## One fixed construction

All settings below were chosen before this candidate's evaluation outcomes.
There is no support-threshold search, alternative spread fallback or lookback
selection based on results.

1. Filter observations to those historical dates and dependencies available at the
   target cutoff **before** reconciling conflicts. A future contradictory copy
   cannot contaminate the earlier state.
2. Group by stable runner identity and prior date. Identical event/context/cell/
   fingerprint copies count once, preserving all source bindings. Different
   copies exclude that runner-date. The same runner-event represented on
   different dates is also excluded. Date/context event proxies are explicitly
   labelled; they are not asserted to be native source race IDs.
3. Context is canonical track, distance, explicit layout and explicit era.
   Unknown layout/era stays separate from known values. Unknown contexts assume
   a stable sectional definition within their literal key. Alias merging requires
   an adapter-supplied retained evidence binding; there is no fuzzy matching.
4. For each target runner and historical context, exclude **all** observations of
   that runner. For every other distinct dog, first take its median sectional in
   that context. The context centre is the median of these per-dog medians.
   This gives each dog equal benchmark weight irrespective of how often its
   histories are repeated or how many distinct retained performances it has.
5. Require at least **five other established runner identities**. The spread is
   **1.4826 times the median absolute deviation** of those per-dog medians.
   Zero or non-finite spread makes that context unsupported. No epsilon,
   IQR fallback or guessed cross-context benchmark is used.
6. For each usable runner history, compute:

   ```text
   z = clip((context_centre - sectional) / context_spread, -3, +3)
   ```

   Positive means relatively faster within that historical context. Keep the
   **latest five supported independent observations at most**, then take their
   median `z`. Different historical contexts may contribute, under an explicitly
   exploratory transfer assumption.
7. If `n` observations remain, the estimate is:

   ```text
   speed_estimate = median(z) * n / (n + 3)
   ```

   One and two observations contribute at weights 1/4 and 2/5. Five contribute
   at 5/8. These are fixed conservative shrinkage weights, not fitted reliability
   probabilities. An unsupported runner receives estimate zero and an explicit
   missing-support status. A supported runner may also have a genuine zero
   estimate; support and numerical value are separate.

Benchmarks update at each target cutoff from all authorized, authenticated,
available prior observations. This rule uses no target labels and does not require
an observation's benchmark to have existed at that historical event's own date:
it must exist when the target forecast is made. Later evaluation-period history
can enter a later prediction only through this fixed availability rule. The
training/evaluation allocation and coefficient selection are owned by the separate
experiment protocol and evaluator.

## Missing fields and downstream probabilities

Whole-field support is not required. Individual estimates are retained regardless
of other runners' support. Downstream scoring must apply the same baseline to
all runners, multiply supported runners by the prespecified speed adjustment,
and normalize the entire field. Unsupported runners have zero **direct** speed
adjustment, but their final probabilities can still change after normalization.
When nobody has supported speed information, downstream scoring must return the
original baseline exactly. This module returns features, not forecast probabilities.

## Private reproducibility and accounting

Output contains each selected original sectional, identity/event proxy, dates,
availability, original source bindings, benchmark identity and clipped residual.
The benchmark catalogue retains every contributing other runner, its median and
its deduplicated original observations, plus centre, MAD, spread and support.
Thus an independent verifier can reconstruct a sample directly from the retained
source cells rather than trusting the feature output's arithmetic.

Population exclusions distinguish date, availability, missing sectional, invalid
sectional, duplicate and conflicting observation failures. Per-runner exclusions
distinguish sparse benchmark support and unusable spread. All roster entries
receive dispositions. The parent empirical adapter/report retains original strict
coverage and accounts for source allocation, unresolved identity and aliases.

The full output contains private historical cells and stays private. Coverage and
failure counts alone do not establish forecasting improvement. Grade, going,
remaining-time features, training variations and live integration are outside this
construction.

## Focused verification

The pure fabricated tests exercise exact independent arithmetic, one/two-history
shrinkage, latest-five selection, cross-context scaling, equal per-dog benchmark
weight, own-history exclusion, repeated-card deduplication, conflicting values or
fingerprints, event/date contradictions, future evidence and identity mapping,
strict prior dates, zero spread, layout separation, explicit aliases, missingness,
clipping, deterministic output and shared schema failure. Real source-cell
recomputation is a separate root-owned execution and independent review step.
