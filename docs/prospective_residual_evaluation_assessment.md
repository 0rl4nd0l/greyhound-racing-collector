# Separate residual evaluation proposal: planning assessment

Status: **UNAPPLIED_PROPOSAL_ASSESSMENT**. This reviews the September 16
`PROSPECTIVE_EVALUATION_PROPOSAL.md` and `READINESS_REPORT.md` retained in
`/home/l4nd0/greyhound-prospective-readiness-20260916/`. It neither changes that
proposal nor authorizes collection, population reservation, predictions, labels
or scoring. No target outcomes or additional records were accessed.

## Precision and power under the stated assumptions

With 1,000 races, 20 per date and hypothetical within-date correlation 0.05,
the equal-cluster planning design effect is `1 + (20 - 1) * 0.05 = 1.95`.
The corresponding planning effective sample size is `1000 / 1.95 = 512.82`.
Using normal approximations, a 95% interval half-width is
`1.96 * sigma / sqrt(512.82)`; an 80% marginal-power effect scale for a
one-sided 2.5% test is approximately `2.80 * sigma / sqrt(512.82)`.

| Hypothetical paired SD, sigma | 95% CI half-width | Approximate 80% marginal-power effect scale |
| --- | --- | --- |
| 0.10 | 0.00866 | 0.01236 |
| 0.20 | 0.01731 | 0.02473 |
| 0.30 | 0.02597 | 0.03709 |

For paired log loss these values are nats per race. The same arithmetic applies
to Brier differences only with that endpoint's own SD, in Brier units. None of
the SDs or correlations is measured here. These calculations are planning
approximations, not promised coverage or power for the proposed date bootstrap.
Actual cluster sizes, repeat runners, cross-date dependence and finite cluster
count can change the uncertainty materially. Twenty thousand resamples do not
create additional independent dates.

The proposal requires both losses' upper 97.5-percentile bounds below zero.
An 80% marginal-power calculation for each endpoint does not establish 80%
power for that joint decision: if endpoint successes were independent it would
be 64%; with two 80% marginal powers the general joint-probability bounds are
60% to 80%. Their actual dependence is unknown. Requiring both endpoints is an
intersection decision; no independent-test assumption is justified here.

At most 20 races per date requires at least 50 contributing dates. Accrual can
span up to 180 calendar dates under the proposed deadline. Fifty consecutive
dates cover roughly seven weeks, providing few independent weekly blocks;
date count is not proof of independence. The proposal's week sensitivity should
remain descriptive. A scientifically worthwhile minimum improvement for each
loss has not yet been chosen independently of outcomes. Consequently 1,000 is
a finite operational target with transparent assumed precision, not a
demonstrated adequately powered sample size.

## Accrual and complete-result feasibility

To reach 1,000 within 180 calendar dates requires an average of at least
`1000 / 180 = 5.56` successful, eligible prediction seals per calendar date,
after the daily cap. If `R_d` eligible opportunities and a seal-success fraction
`q_d` occur on date d, the planning requirement is approximately
`sum(min(20, R_d * q_d)) >= 1000`; variable yields and the cap make an average
rate alone insufficient. At the fastest possible rate, all 50 contributing
dates must reach 20 successful seals.

Illustrative uncapped thresholds, assuming a constant number of opportunities
every calendar date, are 92.6% success at six opportunities/day, 55.6% at ten,
27.8% at twenty, or 13.9% at forty. These are arithmetic scenarios, not observed
collector success rates. Days without opportunities require higher throughput
on contributing dates.

The fixed audit had five races on one date, four venue-date meetings and ten
WIN receipts; five receipts met the proposed window, one per race. Full history
and replay were unassessed, so zero demonstrated qualifications does not mean
an observed zero success probability. The separate future-race sample contained
zero races. Neither sample measures daily opportunity flow, full-input retention
success, pre-cutoff latency, date coverage or result-closure reliability.

The proposed rule that any unresolvable sealed member prevents the primary
analysis is also a material feasibility constraint. As an illustration only,
independent permanent closure failure of 0.1% per race would leave probability
`0.999^1000 = 36.8%` of complete closure. An 80% chance of all 1,000 closing would
require per-race permanent failure below approximately 0.0223% under that
independence assumption. Correlated outages invalidate this simplified model.
No such failure rate was measured. This does not recommend changing the rule
after outcomes; it identifies a design choice to resolve before adoption.

## Decision before adoption

Do not recommend activating this statistical proposal yet. First establish a
future population disjoint from reserved studies and separately approved
machine-only history handling. An input-only operational acceptance must show
real retained-source lineage, isolated feature replay, time/space cost and
pre-cutoff completion. A later, separately authorized readiness exercise needs
predeclared denominators for opportunities, input failures, successful seals,
contributing dates and finite closure failures, without interim model metrics.
One successful input-only race proves a delivery path, not daily accrual or
statistical adequacy.

Before adopting any evaluation design, specify the smallest useful paired
improvement, obtain or explicitly accept defensible variance/dependence
assumptions, assess joint endpoint power, establish sustainable after-cap
accrual and decide whether all-or-nothing result closure is operationally
credible. Any separately approved design revision must precede cohort
activation; this assessment makes none. Input retention integration can proceed
through its own review and acceptance gates independently of this proposal.
