# Greyhound Predictor — Predictive Research Roadmap

## Status and authority

This document preserves the programme's forward modelling strategy so promising
research directions are not forgotten, repeatedly rediscovered, or incorrectly
discarded after one negative experiment.

It records hypotheses and priorities. It is **not evidence that any listed
approach works** and does not authorize fitting, scoring, promotion, deployment,
cohort access, or betting. Canonical repository evidence, frozen experiment
artifacts, current data, and tests remain authoritative for claims. If this
roadmap conflicts with those sources, those sources win.

Current ownership, execution state, blockers, and next safe actions are tracked
in the canonical [active-workstream ledger](ACTIVE_WORKSTREAMS.md). This roadmap
must not be used as an active queue.

The objective is not to find the most complicated model. It is to identify
**trustworthy incremental information beyond the market**, convert that
information into calibrated probability adjustments, and eventually determine
whether those adjustments have economic value at actually available prices.

## 1. Supported evidence position

### Corrected Sportsbet baseline

The corrected Sportsbet WIN market remains the fundamental historical baseline.
A major historical WIN/PLACE provenance problem was repaired, substantially
changing runner probabilities. All modelling conclusions must use the corrected
market surface.

### Historical Betfair forward candidate

Official Betfair scheduled-off data has produced the clearest incremental signal
discovered so far. Historical development and validation selected this frozen
candidate:

Frozen candidate: **95% normalized Betfair scheduled-off + 5% corrected
Sportsbet WIN**.

The original candidate was frozen for an untouched forward evaluation covering
**2026-08-18 through 2026-09-30**, but later outcome-bearing diagnostic exposure
made it `COMPROMISED_FOR_PRISTINE_CONFIRMATION`. Its replacement is
`BLOCKED_NO_OUTCOME_FREE_BETFAIR_SCHEDULED_OFF_SOURCE`; it has zero replacement
population and outcomes. The original freeze remains immutable historical
development evidence. The exact authority and evidence boundaries are defined
by the
[Sportsbet + Betfair forward consensus protocol](sportsbet_betfair_forward_consensus_protocol.md).

### Frozen overround successor

The Sportsbet overround-allocation hypothesis remains alive. Its forward
successor has been engineered and merged, but its confirmatory population begins
no earlier than **2026-10-01**. It must remain separate from the Betfair forward
cohort. The exact boundary is defined by the
[forward overround successor protocol](forward_overround_successor_protocol.md).

### Form and speed evidence

Broad historical form and speed modelling has repeatedly failed to improve on
the market. Most recently, a market-offset residual model using source-bound
prior speed, pace, recency, form, track/distance experience, box context, and
field-relative features produced:

- worse overall log loss than Sportsbet;
- worse results for non-favourites; and
- poor economic diagnostics among its own supposed positive-EV selections.

Verdict: **`NO_INCREMENTAL_FORM_SPEED_SIGNAL`**.

This means the existing representation failed. It does **not** prove that all
racing-performance information is useless.

### Exotic and finish-order evidence

Plackett–Luce, pairwise, and nonlinear finish-distribution models did not
reliably improve trifecta or First Four predictive coverage over market ranking
at equal combination budgets.

Verdict: **`NO_INCREMENTAL_EXOTIC_PREDICTION_SIGNAL`**.

Park this family until either a better underlying probability signal is proven
or trustworthy historical exotic dividends become available.

## 2. Working predictive thesis

The likely successful architecture is **market intelligence → racing
intelligence → small residual correction**, rather than **historical dog data →
independent winner prediction**.

The market should generally provide the starting probability. The research task
is to identify circumstances where the market is systematically wrong.

### Layer 1 — Market intelligence

Potential sources include:

- Sportsbet;
- Betfair;
- TAB and additional bookmakers;
- exchange markets;
- market trajectories over time;
- cross-market disagreement;
- overround structure; and
- liquidity and spread information if legitimately available.

### Layer 2 — Racing intelligence

Potential sources include:

- opponent-adjusted ability;
- adjusted speed figures;
- early pace and sectionals;
- performance trend;
- same-track/distance evidence;
- box suitability;
- field interaction and likely congestion;
- race grade and context; and
- uncertainty and consistency.

### Layer 3 — Residual and value layer

Ask:

> Given what the market already believes, does independent racing or market
> evidence justify changing the probability?

For example:

- Sportsbet or market-consensus probability: **17%**;
- independent racing-evidence adjustment: **+3 percentage points**;
- estimated fair probability: **20%**;
- fair odds: **$5.00**; and
- available odds: **$6.00**.

The runner still loses approximately 80% of the time. That does not inherently
make the prediction bad. The relevant question is whether the probability
estimate is systematically better than the price implies.

## 3. Priority research directions

Everything in this section is a working hypothesis, not an established signal.

### Priority A — Multi-market disagreement

This is currently the strongest direction. Betfair has demonstrated materially
different and historically stronger probabilities than Sportsbet.

Future features should include:

- Sportsbet probability minus Betfair probability;
- market-rank disagreement;
- favourite disagreement;
- Sportsbet versus Betfair overround;
- probability dispersion;
- difference in probability concentration;
- eventual TAB disagreement; and
- eventual three-or-more-market consensus.

Key question:

> When independent markets disagree, which market is more likely to be correct,
> and under what conditions?

Do not disturb the frozen 95/5 forward candidate while developing
later-generation hypotheses.

### Priority B — Betfair temporal market information

Scheduled-off Betfair data has been promising enough that earlier Betfair
information is now high priority.

Potential observations:

- T-120;
- T-60;
- T-30;
- T-10; and
- T-2.

Potential features:

- price and probability change;
- acceleration of movement;
- rank and favourite changes;
- Sportsbet–Betfair convergence or divergence;
- which market moves first;
- late shortening or drifting; and
- movement relative to field redistribution.

A particularly valuable target is:

> Predict the later market before predicting the race.

For example, can racing information at T-60 predict which $6 runner will shorten
to $4.80 by T-2? This provides a potentially much larger training target than
winners alone and tests whether racing features identify information the market
eventually recognizes.

### Priority C — Dynamic latent ability ratings

Raw recent form is a crude representation of ability. Develop continuously
updated runner ratings using only prior races.

Candidate latent dimensions:

- overall racing ability;
- adjusted speed ability;
- early-pace ability;
- finishing or closing ability; and
- uncertainty around current ability.

Potential methods:

- Elo-style ratings;
- TrueSkill-style models;
- Bayesian or state-space ratings; and
- recency-weighted latent performance models.

A second place against an elite field may represent stronger evidence than
winning a weak race. Ratings should therefore account for **opponent quality**,
not just finishing position.

### Priority D — Proper speed figures

Do not equate raw times directly across meetings or contexts. Develop
horse-racing-style adjusted speed figures.

Potential adjustments:

- track and distance;
- meeting or day speed;
- contemporaneous races on the same card;
- track or weather state where trustworthy;
- race class and context; and
- sectional structure.

Target output should resemble:

> Runner A is +0.42 standard deviations above today's field on recent adjusted
> track-distance performance.

rather than:

> Runner A ran 29.80 last start.

Field-relative features may include adjusted-speed rank, runner versus field
median, runner versus second-fastest runner, best-of-last-N adjusted performance,
median-of-last-N, trend, and variance.

### Priority E — Fast non-favourite mechanism

Before completely abandoning traditional speed intuition, directly test the
transparent observation:

> A runner can be a non-favourite despite clearly superior exact-track/distance
> speed evidence.

Do this without fitting another broad model. Predeclare groups such as:

- speed rank 1 / market rank 1;
- speed rank 1 / market rank 2;
- speed rank 1 / market rank at least 3;
- large speed-margin leader; and
- speed leader also favoured more strongly by Betfair than Sportsbet.

Evaluate whether these groups win more often than the **sum of Sportsbet
probabilities predicts**. If this transparent mechanism fails, current
historical speed should be parked as a probability-adjustment source.

### Priority F — Early pace × box × field interaction

Greyhound races are not independent runner contests. A dog's expected
performance depends on the rest of the field.

Potential features:

- early-pace rating and rank;
- number of fast starters inside and outside;
- likely first-turn congestion;
- vacant boxes;
- rail-seeking, neutral, or wide-running behaviour;
- squeeze risk;
- clear-lead probability;
- competing pace pressure; and
- box × early-pace interaction.

The key unit is the **whole race configuration**, not just individual dog
features. A runner with only the fourth-best overall speed may have the highest
win probability if it is the only strong beginner and is likely to obtain an
uncontested lead.

### Priority G — Sectional-shape modelling

Final time can hide different performance types.

Potential runner archetypes:

- explosive beginner / fading finisher;
- balanced runner;
- slow beginner / strong closer;
- highly variable beginner; and
- consistent leader.

Potential features:

- first-sectional rank and consistency;
- early-to-final speed ratio;
- estimated closing speed;
- acceleration or deceleration;
- probability of leading early; and
- interaction with today's competing early pace.

Use only genuine pre-target prior sectionals.

### Priority H — Performance uncertainty and ceiling

Point-estimate ability may miss useful long-shot structure. Estimate both
**expected performance** and **performance variance**.

Potential features:

- speed variance;
- first-sectional variance;
- best-versus-median gap;
- frequency of elite historical performances;
- probability of reproducing top-quartile performance;
- consistency; and
- recent change in variance.

A $6 runner may have lower median ability than the favourite while having a
higher probability of producing the single elite run required to win. This
hypothesis remains largely untested.

### Priority I — Adjustment gating

Do not force a racing-data correction in every race. Eventually develop a gate
that answers:

> Is this a race where independent evidence is strong enough to justify moving
> away from the market?

Possible gate inputs include history depth, feature reliability,
Betfair–Sportsbet disagreement, ability uncertainty, track/distance evidence
quality, pace configuration, and reserve or scratch complexity.

Examples:

**Market agreement + sparse history** → adjustment approximately zero

**Strong market disagreement + reliable speed evidence + favourable pace
interaction** → permit a larger residual correction

This may be safer than perturbing every race.

## 4. Additional data acquisition

### Betfair one-minute history

High priority. Investigate the free Basic historical stream for usable
greyhound coverage.

Desired output:

- Betfair T-120/T-60/T-30/T-10/T-2 last-traded probabilities;
- immutable source provenance;
- exact race and selection IDs; and
- deterministic timestamp extraction.

Do not treat delayed API data as equivalent to genuine historical point-in-time
observations.

### TAB and additional bookmakers

Pursue permission or API access where feasible. The goal is to build an
independent bookmaker consensus rather than relying on one source.

### Exchange liquidity

If legitimately accessible later, potential inputs include back price, lay
price, spread, available liquidity, traded volume, imbalance, and movement.
These may contain useful gates or signals, but no such value is established.

### Weather and track

This is a secondary priority. Use only timestamped authoritative observations.
Weather should not be assumed useful merely because it is available.

## 5. Experiment sequence

Unless new evidence changes priorities:

1. **Fast non-favourite mechanism audit**
   - transparent;
   - no fitted model; and
   - directly tests the user-observed phenomenon.
2. **Betfair temporal-data acquisition**
   - determine whether one-minute history can be reconstructed reliably.
3. **Opponent-adjusted latent ability / speed rating**
   - simple model first; and
   - no giant feature search.
4. **Early-pace × box × race-interaction experiment**.
5. **Sectional-shape / runner-archetype experiment**.
6. **Performance uncertainty / ceiling experiment**.
7. Only after individual signals demonstrate incremental information:
   - combine them into a residual model;
   - add an adjustment gate; and
   - evaluate market-relative probability improvement.

Do not jump directly to a large ensemble containing every speculative feature.

## 6. Evaluation principles

Every new candidate must declare before evaluation:

- hypothesis;
- baseline;
- eligible population;
- training and validation dates;
- frozen features;
- model and configuration;
- leakage controls;
- primary metric;
- uncertainty method; and
- strongest permitted claim.

The primary predictive metric remains **paired multiclass race log loss versus
the relevant market baseline**.

Secondary metrics include:

- Brier score;
- calibration and ECE;
- top-1 accuracy;
- winner rank; and
- temporal or fold stability.

Economic diagnostics may include fixed one-unit returns, break-even probability,
ROI, bootstrap confidence intervals, and drawdown. However:

> Economic diagnostics cannot rescue a model that fails its predeclared
> predictive hypothesis.

Do not optimize betting thresholds after viewing outcomes.

## 7. Sample-size discipline

Near-market effects are expected to be small. Use approximately:

- about 500 races for debugging and exploration;
- about 2,500 for useful screening;
- about 7,500 for serious candidate development;
- 10,000 or more for stronger confirmation around 0.01 log-loss effects; and
- 25,000 or more for very small effects around 0.005.

Avoid repeatedly declaring winners from approximately 200-race windows.

## 8. Low-priority repeated work

Unless genuinely new evidence appears, avoid repeatedly running:

- generic RF, ExtraTrees, or XGBoost variants;
- broad form-feature soups;
- repeated calibration methods;
- arbitrary odds-band searches;
- post-hoc profitable subsets;
- exotic-ticket optimization without payout evidence; and
- models trained to relearn the market from scratch.

Complexity is not itself new information.

## 9. Role of form and speed in the live product

Even if historical form and speed never earn a probability adjustment, they
should remain visible to the operator as research context.

Useful explanatory fields may include:

- speed rank;
- adjusted track-distance performance;
- best or median recent run;
- early-pace rank;
- trend;
- same-track/distance history;
- market rank;
- Betfair rank; and
- Sportsbet–Betfair disagreement.

The system can distinguish **core probability evidence** from **operator
research context** without pretending every displayed feature is predictive.

## 10. Intended long-term predictor

The intended architecture is:

**Market anchor** → Betfair + Sportsbet + additional bookmakers

**Market trajectory** → movement, disagreement, convergence, overround

**Latent racing ability** → opponent-adjusted speed and pace

**Race configuration** → box, early pace, congestion, and interactions

**Uncertainty** → reliability and performance ceiling

**Gated residual correction** → adjust only when independent evidence is strong

**Final output** → runner probabilities, fair odds, market odds, probability
difference, confidence, and explanation

Illustrative output only:

> **Runner 6**
>
> - Market consensus: 18.2%
> - Racing adjustment: +2.1 percentage points
> - Final probability: 20.3%
> - Fair odds: $4.93
> - Available Sportsbet: $6.00
> - Evidence: strong adjusted track-distance ability, best early pace, favourable
>   race configuration
> - Confidence: medium

This is the intended research destination. It is not an established market
edge.

## 11. Parked ideas and revisit conditions

### Form and speed

Revisit only with materially better adjusted speed representation, latent
opponent-adjusted ratings, race interactions, or genuinely new history
coverage. Do not simply retune the failed residual model.

### Exotics

Revisit only if a better underlying probability or ranking signal is
prospectively proven, or trustworthy historical exotic dividends become
available.

### Overround

Leave frozen until its October forward experiment.

### Betfair consensus

Leave frozen through the September forward-test window.

## 12. Core strategic rule

The programme should continuously ask:

> **What new information are we adding that the market may not already know or
> correctly price?**

If the answer is merely "another model trained on the same information," the
experiment should have low priority.

The strongest current path is **independent markets + temporal market
information + better latent racing ability + race interactions**, with the
market remaining the anchor until evidence demonstrates otherwise.
