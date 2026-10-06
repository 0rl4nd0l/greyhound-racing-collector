# Prespecified private sectional comparison

This module executes one small retrospective probability experiment. It does
not acquire data or confer result authority. Root authenticates the development
membership, label allocation, original probabilities, independently reproduced
baseline, and feature bindings before constructing the packet. All returned
records, outcomes, probabilities, and metric tables are private output.

The pure interface is `frozen_protocol(...)` followed by
`evaluate_experiment(records, protocol)`. Root must write and hash the protocol
before reading evaluation labels and bind it to the fresh execution receipt.
The function only checks reference syntax; it deliberately does not reopen
files independently of the root-owned retained-data adapter.

## Authorised population and chronological design

The current root-selected predictive population is the **177 development races
from June 24 through July 8**, subject to the authenticated membership and label
checks. The earlier 154 races in the 331-race foundation remain baseline warmup
and historical-input evidence; their labels are not required by this candidate
evaluation. The October 82-card construction remains separate feature evidence
and does not automatically allocate those labels to fitting or evaluation.

Root freezes these explicit date arrays:

- Training: June 24–30, 2026.
- Validation: July 1–2, 2026.
- Evaluation: July 3–8, 2026.

Dates with no member races are retained in the prespecified ranges. Each
required partition must contain at least one eligible labelled race; otherwise
the evaluator returns a complete blocker/accounting record and does not fit.
The interface supports other explicitly frozen, nonoverlapping chronological
date arrays; it does not infer a favourable split from outcomes. Its default
October arrays exist for fabricated compatibility tests, not October label
authority.

All of this development population has previously been inspected. The final
period is a chronological **retrospective evaluation**, not a fresh holdout or
strong independent confirmation. Repeated dogs and reused historical runs also
do not become independent experimental replications.

## Baselines and one candidate

All three methods use exactly the same eligible races and complete runner
fields:

1. **Normalized market:** reciprocal decimal odds divided by their field sum.
   This is recomputed and checked against the stored market vector within
   `1e-12` absolute error.
2. **Existing development baseline method:** root reproduces the 626 original
   runner forecasts from the first 86 races against the retained fixed base16
   model, with tolerance `1e-12`. That same model, trained before June 24, then
   generates baseline probabilities for all 177 later races. The remaining 91
   races are newly scored with the fixed earlier model, not independently
   reproduced against their original later refits. The execution summary states
   both counts. This is the historical development method, not a claim to replay
   the currently installed live model. The evaluator separately checks supplied
   baseline-vector consistency; it never labels all 177 as independent original
   replay. The baseline is not retrained for this experiment.
3. **Same baseline plus sectional feature:** multiply each baseline probability
   by `exp(beta * estimate)`, then normalize across the entire field.

The separate feature module supplies estimates with positive values meaning
faster historical sectionals relative to the relevant past-only context
benchmark. Its fixed preprocessing includes clipping to ±3 and support
shrinkage `n / (n + 3)`. No outcome-dependent transformation is fitted here.

Unsupported runners must have estimate zero and receive exactly zero **direct**
logit adjustment. They may still receive a changed final probability when a
supported rival changes the field normalization. When the whole field is
unsupported, or beta is zero, the output is a literal copy of the stored
baseline vector; missingness is never interpreted as slowness. Partly supported
fields remain in fitting and the principal evaluation.

## Fixed finite fitting and validation

The only coefficient candidates are `[0, 0.1, 0.25, 0.5, 1]`.

Training chooses the coefficient with the smallest **mean race log loss**.
Exact ties choose the smaller coefficient. Validation compares only that
training winner with zero; it retains the nonzero coefficient only if its
validation mean log loss is **strictly lower** than the unchanged baseline.
Otherwise the final coefficient is zero. There is no validation grid search,
refit, alternative feature selection, or post-evaluation retry.

The module records all five training trials, exact training membership, the
single validation comparison, the chosen coefficient, and every final forecast.
If validation rejects the trained adjustment, the complete experiment still
finishes with the neutral candidate rather than searching for a better result.

Earlier evaluation-period historical observations may enter later predictions
only under the feature module's fixed **past-only unlabelled update rule**:
the observation and evidence availability must precede each target cutoff. No
target labels are used to select or update context benchmarks.

## Metrics, uncertainty, and sensitivity

Primary log loss is `-sum(y_i * log(p_i))`, averaged with equal race weight.
Multiclass Brier score is `sum((p_i - y_i)^2)`, also averaged across races,
without dividing by runner count. Independently verified dead heats use equal
winner mass. Known nonfinishers can retain valid win-target status under the
existing verified-result rules; unknown or quarantined labels remain absent.

Paired changes are candidate minus comparator, so negative means lower loss.
Reports compare speed versus baseline, speed versus market, and baseline versus
market. They include each racing date, the principal evaluation population, a
clearly secondary supported-race-only population, and pooled descriptive
results whose training/validation roles remain visible.

The fixed uncertainty procedure resamples whole racing dates, with replacement,
2,000 times using seed `20261006`. Each resample retains every race in its
selected date clusters; intervals are the empirical 2.5th/97.5th percentiles of
the paired mean changes. Fewer than five dates yields **not estimable**, not an
artificial interval derived from treating individual races as independent.
Six evaluation dates still provide weak cluster uncertainty: the report flags
any evaluation with fewer than ten dates and makes no confirmatory claim.

Leave-one-date-out tables remove each **evaluation date from the evaluation
population only**, use the already selected coefficient and never refit.
Training or validation races do not enter this sensitivity analysis. The pooled
three-partition table is separately labelled descriptive. No ranking, track
subgroup, or additional hypothesis is selected after looking at outcomes.

## Packet contract and retained accounting

Each record contains:

```text
race_id, race_date, cutoff, runner_ids,
market_odds, stored_market_probabilities,
stored_baseline_probabilities, reproduced_baseline_probabilities,
speed_estimates, speed_supported, label_status, outcome,
source_bindings = {forecast, reproduction, features, label}
```

Vectors all share one ordered, unique runner roster. References contain an
absolute path and SHA256; root must prove their relation to that roster before
calling the pure module. `FULL_ORDER_WIN_ELIGIBLE` and
`KNOWN_NONFINISH_WIN_ELIGIBLE` are the only eligible label statuses. All other
statuses require `outcome = null` and `source_bindings.label = null`, preserving
the exclusion without opening its target. Every manifest member remains in the
returned record set, and every eligible race remains in the principal
comparison irrespective of feature coverage.

The returned object contains the protocol/input hashes, population and split
counts, label-status exclusions, supported runner counts, exact-fallback counts,
baseline reproduction error, fit records, prediction records, metric tables,
uncertainty limitations and source references. It recommends retaining the
candidate as exploratory; root's final evidence-based decision may reject the
variant or freeze it for genuinely fresh prospective testing. It never grants
production promotion.

Root supplies a durable private progress sink for empirical execution. It writes
numbered records before and after each consumed coefficient trial, at training
selection and validation, and before and after each final prediction. A sink
failure stops the experiment. Label decoding reports each actual exposed row to
the orchestrator, so an input failure cannot label already opened races as
unattempted. Completed predictions and consumed trial values remain preserved if
a later report or oracle fails.

The complete private result is written with provisional status before the final
probability oracle. Only after every oracle check passes is the final private
result written; the successful summary is published last. A failure instead
records the exact stage, exposed/remaining population, consumed trials,
prediction count and independently verified races in `FAILED.json`.

## Focused validation

`tests/test_sectional_speed_evaluation.py` uses fabricated inputs only. Tests
cover hand-worked log loss/Brier values, unsupported normalization and exact
fallback, finite training selection, strict validation rejection, unchanged
selection when evaluation outcomes change, quarantine accounting, native
baseline mismatch, market mismatch, membership/identity failures, sparse date
uncertainty, deterministic six-date bootstrap, chronological split rejection,
and measured tight floating-point reproduction tolerance.

No real labels, predictions, or metric reports were opened by the evaluation
implementation agent. Root owns the actual empirical execution.
