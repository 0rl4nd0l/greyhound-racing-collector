# History depth and confidence: executed results

**Decision: retain the existing method.** Two authorized experiments completed
on the same 177 chronological evaluation races / 1,251 runners / 15 dates. Modestly
richer as-of history worsened the full residual's average scores. Support-based
shrinkage slightly improved fixed half-strength but remained worse than full;
uncertainty includes no difference. Neither result justifies a new candidate
or an operational change. No January wait or new data allocation was needed
to execute this development work.

## Does production already have more useful history?

**Sometimes; it is not universally limited to five starts.** Three exact retained
September 29 forecast bundles contain 22 matched runners: 96 card starts plus 22
scoped DB starts produce 118 merged starts, 116 with known finishes. Six runners
gain history; their five-start cards become 6–12 accepted starts. All 352 original
production feature values replay exactly. Added history changes 40 values,
holding production definitions fixed; changing research/production definitions
on the cards alone changes 60 values. These are separate contrasts, not additive
counts or proof that all differences reflect depth.

| Quantity | Card only | Production merged |
|---|---:|---:|
| Accepted starts / known finishes |96 /96|118 /116|
| Same literal venue starts / known finishes |22 /22|27 /27|
| Exact-distance starts / known finishes |26 /26|30 /30|
| ±50m-distance starts / known finishes |78 /78|95 /95|
| Production-grade-token starts / known finishes |22 /22|24 /24|

No duplicate removal, observed coarse conflict or 20-start truncation occurred
in this operational sample. Extra rows did not resolve any runner's absent
venue/distance/grade context. Production divides win/top-three rates by all
starts and returns zero for empty subsets; canonical research uses known
finishes, missing empty rates, exact distance and different grade aliases.
The [matched history report](history_support_20260929_history.md) supplies each
race denominator, source contributions, conflicts and source identities.

Those retained seals prove input availability for the September cutoffs,
**not** June/July. No current mixed DB was searched, and no operational target
result was opened. Protected histories stayed machine-only feature inputs;
only aggregate diagnostic counts were emitted within the existing scope.

## Can richer information be reproduced fairly in development?

**Yes, through earlier admitted cards, with a narrower interpretation than
production DB replay.** Capture-time-qualified union of the 331 authorized cards
adds 678 distinct starts for 360 runners across 159 races. All 331 races / 2,360
runners qualify, with zero conflict exclusions. On the existing evaluation
subset, 129/177 races and 311/1,251 runners gain 627 starts. Histories grow to at
most 10 starts; no complete-career claim is possible.

Only cards captured strictly before each target card's cutoff are eligible.
Identical observations are deduplicated; any same-day disagreement excludes
the complete target race. Formulas, context definitions, cap 20, missing-value
rules, features, L2=1, residual cap .35 and chronological boundaries are fixed.
Canonical runner-token linkage is inherited and cannot rule out cross-card
homonyms without native IDs. This is an exploratory earlier-card reconstruction,
not a reconstructed historical production DB or fresh confirmation.

It changes 1,939 feature values, principally longer-history and context
statistics; recent-three/five values remain unchanged. Missing venue/distance/
grade rates shrink only 940→936, 523→514, 746→737 across all development runners.
Additional depth therefore resolves little of the original missing-context gap.

## Executed comparisons

Lower scores are better. Each race is weighted equally; Brier sums its runner
errors. The existing full and half forecasts are retained #193 anchors, not a
new claim about the live production model. New short-history fits reproduce
the full anchor with **zero probability error**.

| Method | Evaluation races | Log loss | Brier |
|---|---:|---:|---:|
| Normalized same-time WIN market |177|1.412894|.682928|
| Existing full / newly reproduced short residual |177|1.397606|.677388|
| Existing fixed-half residual |177|1.403559|.679370|
| Prespecified support-adaptive residual |177|1.401073|.678414|
| Controlled richer-history residual |177|1.400323|.678301|

**Adaptive support:** one transparent rule, two lambda choices, six earlier-OOF
validation evaluations; zero new fits. Every period selects lambda .25. Mean
race strength is .6406. Compared with half, ΔLL=−.002487 and ΔBrier=−.000956;
simultaneous paired date-cluster95% intervals are[−.005704,+.000730] and
[−.002620,+.000709]. Adaptive is worse than full in all three periods.
Low-support races worsen against half; higher-support races account for most
gain. Five races contribute56% of net LL gain over half. Every leave-one-date-out
mean remains slightly better than half, but that does not establish an edge.
Because average strength exceeds .5, the small gain cannot be attributed solely
to useful adaptation. [Rule, prior-ledger novelty, periods and influence](history_support_20260929_adaptive.md).

**History depth:** exactly six new fits, short/richer ×three chronological
periods; no tuning. Richer−short ΔLL=+.002716, pointwise date-cluster95%
[−.000592,+.007000]; ΔBrier=+.000913,[−.000648,+.002651]. Richer helps73 races
and harms104 on LL. It improves the small 16-race middle period but worsens
the 86-race first and 75-race final periods. Removing any date leaves deterioration
(LL+.001134 to+.003342); removing the five worst races still leaves+.000985.
The129 enriched evaluation races also worsen on average, so dilution by
unchanged-history fields does not explain the conclusion.
[Controlled experiment and complete fit receipts](history_support_20260929_depth.md).

All new results remain exploratory: these dates have already been repeatedly
inspected and used for development. Date bootstrap bands do not correct the
entire research history. Same-time prices retain #193's retrospective
availability-selection limitation. No betting-return or causal winner claim
is made. No post-result thresholds, subgroups, extra strengths or new history
variants were added.

## Deliverables and handoff

- Independent operational replay and matched context/definition counts;
  earlier-card reconstruction with full coverage/exclusion records.
- Executed support experiment with retained OOF inputs, every lambda trial,
  selections, unchanged preprocessing/models, predictions and metrics.
- Six new depth models with complete preprocessing, coefficients, exact
  training/evaluation membership and inputs, feature contract, environment,
  source identities, attempt/failure ledgers and whole-field forecasts.
- Targeted tests and independent timing/leakage/fairness review; commands in
  the linked reports. Frozen protocol/source versions and intermediate attempts
  are preserved, including the resolved reconstruction-source pin mismatch.

**One next action: retain the existing method.** Neither tested change merits
promotion or a new prospective candidate on this evidence. This does not prove
that richer, better-linked history can never help; it rejects the case for
advancing these particular variants now. Production's definition differences
are measured and documented; changing them would be a separate model decision.

Operational owner: no deployment, acquisition, frontend, scheduler, service,
provider-control or October allocation/model/result changes are requested.
Continue live reliability work independently. Research branch:
`research/history-support-20260929`, separate worktree
`/home/l4nd0/greyhound-history-support-20260929`, based on #198
`fc979e370cc518a14e029c37678a172ed7528ea6`. Draft PR only; no merge or promotion.
