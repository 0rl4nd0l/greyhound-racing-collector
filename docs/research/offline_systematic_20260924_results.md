# Systematic offline prediction search — 24 September 2026

**Conclusion: collect new qualified observations; retain the current production configuration.** A finite, nested chronological search completed **277 fits, 219 validation candidate evaluations and 204 selection-rule evaluations**. The validation-selected linear procedure slightly worsened later log loss in all three evaluation periods. The simplest fixed residual anchors remained stronger than the more elaborate search. Their apparent gains are exploratory: simultaneous uncertainty intervals include no improvement, and all evaluation dates had already influenced earlier research. No confirmed advantage, dependable vulnerable-favourite rule or durable outsider edge was found. This does not prove that useful improvements do not exist.

**Foundation correction and exposure accounting.** The previous 332-race qualification claim is superseded. TAREE R2 on June 13 has 08:19 in acquisition/market metadata but 07:59 in its hash-bound pre-race sidecar. The source clock implies only 52m35s card lead, below the asserted T−60 requirement. The race and its seven runners are excluded; the isolated loader now requires manifest capture/jump times to equal source-sidecar times and tests the demonstrated discrepancy. Original records, old predictions and old report bytes are preserved. Those old numerical results remain a historical run with a defective input-qualification claim, not the qualified baseline for this continuation.

The corrected population is **331 races / 2,360 runners** from June 10–July 8, with **177 later evaluation races / 1,251 runners** unchanged. All model variants use those same later races. New refits use all earlier dates after nested validation choices freeze; therefore changes from commit e09c16e reflect both the timing exclusion and different final training sizes, and are not attributed solely to the repair.

The earlier 58-reserved-race / 393-runner incident remains [unchanged](offline_20260924_access_incident.json). The original probe decoded whole rows but only date/capture-lag fields fed its calculations; no demonstrated label-value path reaches the prepared model inputs or decisions. However, its mixed-date timing aggregates **did inform the context for choosing the original T−2 cutoff**. Thus the earlier study was not independent of all protected information at the design stage. No model input identity overlaps the incident/reservation union, and every accepted history precedes both its target and the reserved dates. Exclusion does not reverse access or restore pristine status. This continuation reads reservation/incident metadata first, verifies input hashes, rejects unauthorized identities/dates before decoding development rows, and makes no untouched-holdout claim. See the [independent foundation audit](offline_systematic_foundation_audit.md).

**Available data and remaining limits.** Existing manifests provide no additional *qualified, untouched* later population. Earlier May 27–June 9 cards have no matching corrected timed-market surface. Forty-nine August 1–2 race identities exist in the corrected market inventory, but no qualified pre-race-card manifest was established for them and their later histories could incorporate reserved July outcomes; their outcomes were not opened. July 15–18 remains excluded under explicit reservations/closed-study authority, not released by literal alias nonmatches. The closed 114-race reconstruction was not restarted. This is a bounded inventory conclusion, not proof that every historical source is unusable.

The retained price is corrected Sportsbet fixed WIN evidence, normalized within complete fields. Target snapshots are 2–10 minutes before the agreed scheduled jump: a T−2 decision question with quote age up to eight minutes. Upstream storage selected the latest pre-jump quote, so this subset’s inclusion can depend on later-than-T−2 observations. It is a **retrospective availability-selected probability benchmark**, not a replay of what a live T−2 strategy could cover. Scheduled off is not an independently observed actual jump. Quote persistence/acceptance, deductions and scratch timing are unqualified; simulated returns are therefore not calculated. Largest-payout-removal analysis is inapplicable; selected-win/date removal is reported as predictive sensitivity instead.

Race fields, box/name-token identity, single-winner labels, finish-position agreement, complete raw-card rosters, matching source timestamps and source hashes were checked. Histories are strictly before target and source-capture dates, not future-updated lifetime aggregates. Median imputation, missing indicators, scaling and venue-category vocabularies fit training only. Predictions normalize within race; noncontiguous race groups, invalid labels and non-normalized probabilities fail. No target SP, target time, target margin or result-derived population selection enters new features.

Legacy form venue aliases merge some Q1/Q2/straight, Richmond and Murray Bridge layouts. The original 16-feature anchor retains these legacy context semantics, labelled as measurement ambiguity. New layout-sensitive times/sectionals are unavailable for those ambiguous groups, not inferred across layouts. Raw first-sectional measurements are only compared at the same qualified venue and distance; cross-era timing definitions remain a limitation. PIR is not used as early speed: earlier audit found single-digit PIR equalled finish position. Exact normalized grade changes are tested as category changes, with no invented class ordering. The snapshot is not a full schedule census: absent/abandoned/unresolved-result races are outside the prequalified source universe. [Exclusions](offline_systematic_evidence/exclusions.json), [dataset assessment](offline_systematic_evidence/dataset_assessment.json) and [feature timing/missingness audit](offline_systematic_evidence/feature_audit.json) are retained.

**Search design and stopping.** [STAGE_PLAN.json](offline_systematic_evidence/STAGE_PLAN.json) was written before new model performance. [SEARCH_PROTOCOL.json](offline_systematic_evidence/SEARCH_PROTOCOL.json) froze the exact recipes, splits, candidate settings and rule gates before fitting. The 23 group recipes covered single additions/removals and three mechanism combinations: draw geometry and distance/track interactions; recent/long form and fixed recency weighting; margin/finish improvement; same-layout times; sectionals; nearby faster-section rivals; grade change; layoff/experience; field-relative form; uncertainty; market concentration/entropy/overround; and selected interactions. No brute-force subset sweep was used.

The best three inner-validation group recipes survived to a coarse penalty/strength search: L2 {0.1,0.3,1,3}, residual strength {0.25,0.5,1}, cap 0.35. Four market powers {0.75,1,1.25,1.5} competed with them. Six fixed nonlinear configurations covered histogram boosting, Extra Trees and shallow market-residual boosting. Ensemble weights {0,.25,.5,1} used component predictions made without fitting those validation rows. The final winner was chosen by earlier OOF log loss, never by later scores or return.

Budgets were finite: per outer iteration, at most 24 group recipes, top-three tuning, six nonlinear configurations; CPU ceilings 180/300/120 seconds for those stages and 90 seconds for final refits, then 60 seconds for diagnostics. Actual entire model-search execution was **13.28 seconds wall / 12.24 CPU seconds**, single-threaded with `nice 10`; peak RSS **248.0 MiB**. Host load was checked before fitting (1.63/1.43/1.03), with ample available RAM. The hypotheses were exhausted within budget. No recipes, tuning ranges or population rules were expanded after later metrics appeared.

**Nested chronological structure.** All periods are 2026; earliest training date is June 10. Calendar dates and complete races remain together. Earlier outer dates can enter later training as time advances; no outer-score feedback branch changes the search. Each outer selection/rule freeze is in the ledger, and all outer predictions were sealed before aggregate outer metrics were computed. All of these outer races are previously inspected development evidence.

| Later evaluation | Inner training end → validation | Final refit through | Final training / evaluation races |
|---|---|---|---|
| June 24–30 | June 13 → June 14–17; June 17 → June 18–21 | June 21 | 154 / 86 |
| July 1–2 | June 17 → June 18–21; June 21 → June 24–30 | June 30 | 240 / 16 |
| July 3–9 (observed through July 8) | June 17 → June 18–21; June 21 → June 24–30; June 30 → July 1–2 | July 2 | 256 / 75 |

The final choices changed across periods: track×box at L2 .3/full strength, then margin improvement at L2 .3/full strength, then market power 1.5 without form. Tree selection chose a depth-two residual booster in period one and depth-one residual boosters later. Ensemble tree weights were 0, 0 and .25. This is an evaluated **selection procedure**, not a claim that one fixed fitted model produced every prediction. [Frozen selections](offline_systematic_evidence/frozen_selections.json) retain every choice and inner score.

**Executed later comparison.** Lower log loss/Brier is better. Brier sums squared runner errors within each race before averaging. Top-choice ties share credit. Intervals resample 15 calendar-date blocks 3,000 times, seed 20260924. The simultaneous bands use the maximum standardized bootstrap deviation across the seven reported candidate comparators. They address that reporting family only—not all prior historical experimentation. Nested selection reduces current tuning optimism but cannot manufacture confirmation from already-inspected data.

| Procedure / fixed anchor | Log loss | Brier | Top choice | ΔLL vs market | Simultaneous 95% interval |
|---|---:|---:|---:|---:|---|
| market | 1.412894 | 0.682928 | 45.20% | +0.000000 | — |
| uniform | 1.944654 | 0.855260 | 14.47% | +0.531761 | — |
| refit_base16 | 1.397606 | 0.677388 | 44.07% | -0.015287 | [-0.03510, +0.00453] |
| refit_half | 1.403559 | 0.679370 | 45.76% | -0.009334 | [-0.01925, +0.00058] |
| refit_box | 1.396280 | 0.676556 | 44.07% | -0.016614 | [-0.03553, +0.00230] |
| selected_linear | 1.415566 | 0.691528 | 44.92% | +0.002673 | [-0.02921, +0.03455] |
| selected_nonlinear | 1.408841 | 0.681927 | 46.33% | -0.004053 | [-0.00919, +0.00109] |
| selected_market_calibration | 1.404636 | 0.685737 | 45.20% | -0.008258 | [-0.04320, +0.02668] |
| selected_ensemble | 1.412317 | 0.688619 | 45.20% | -0.000577 | [-0.02526, +0.02410] |

Unadjusted date intervals for box and half-strength exclude zero, but both simultaneous intervals include it. The broader linear selection procedure is worse in every outer period; the ensemble barely changes pooled log loss and worsens Brier. Nonlinear residual boosting modestly improves all three periods but provides less average gain than the simple residual anchors. Hist/Extra Trees did not survive validation. No candidate establishes a confirmed edge.

| Procedure | June 24–30 ΔLL | July 1–2 ΔLL | July 3–8 ΔLL | Negative dates / 15 |
|---|---:|---:|---:|---:|
| refit_base16 | -0.010966 | -0.035493 | -0.015931 | 11 |
| refit_half | -0.007293 | -0.018762 | -0.009664 | 11 |
| refit_box | -0.011635 | -0.032677 | -0.018897 | 11 |
| selected_linear | +0.002420 | +0.008559 | +0.001707 | 4 |
| selected_nonlinear | -0.004370 | -0.009255 | -0.002579 | 11 |
| selected_ensemble | +0.002420 | +0.008559 | -0.005962 | 4 |

Removing any one date leaves box ΔLL between −0.02077 and −0.01239, half between −0.01140 and −0.00714, and selected nonlinear between −0.00507 and −0.00307. The adaptive linear procedure changes sign under date removal. Venue results are heterogeneous: among 16 venues with at least five evaluated races, box improves 10, half nine and nonlinear nine. For example, box regresses at Geelong/Taree/Dubbo despite pooled improvement; venue samples of five or six races cannot support reliable specialist models. [Date/interval results](offline_systematic_evidence/summary.json), [period results](offline_systematic_evidence/period_metrics.json) and [venue results](offline_systematic_evidence/venue_metrics.json) contain all values.

Calibration is not rescued by average proper-score gains. Market runner ECE is .02108; box .02600, half .02470, selected nonlinear .02006 and selected linear .02893. Ten fixed equal-width bins and every bin denominator are reported. The small nonlinear ECE reduction is descriptive, not evidence of calibrated individual true chances. The power-calibration procedure improves pooled log loss mainly through the small 16-race middle period and slightly worsens the last period. Broad odds-band actual/market/model frequencies are [fully retained](offline_systematic_evidence/odds_band_calibration.json).

**Which features helped?** These are **inner OOF screening differences against the base16 anchor**, not independently confirmed incremental later gains. Inner populations overlap across successive outer folds; do not treat three columns as three independent replications. Every one of the 23 recipes is preserved in the ledger. Margin improvement was consistently attractive in validation but did not produce a successful later adaptive choice; selecting it in period two worsened later log loss. Times, sectionals, pace pressure and the tested mechanism interactions consistently failed this screen. That is evidence against these particular constructions at this sample size, not evidence that physical early speed or times can never help.

| Recipe | First inner ΔLL | Second inner ΔLL | Third inner ΔLL |
|---|---:|---:|---:|
| base16 | +0.000000 | +0.000000 | +0.000000 |
| base_box | +0.001944 | -0.000352 | +0.000002 |
| add_draw | +0.003828 | -0.000716 | -0.000282 |
| add_draw_distance | +0.002843 | -0.001076 | -0.000913 |
| add_draw_track | -0.009345 | +0.002999 | +0.001470 |
| add_recency | +0.000977 | -0.000542 | +0.000302 |
| add_margin_improvement | -0.007366 | -0.011081 | -0.006855 |
| add_times | +0.011468 | +0.007110 | +0.006755 |
| add_sectionals | +0.008106 | +0.001876 | +0.001559 |
| add_pace_pressure | +0.007644 | +0.000433 | +0.000279 |
| add_grade_change | -0.005342 | +0.001099 | -0.000008 |
| add_experience | +0.001961 | -0.000612 | +0.001772 |
| add_field_form | +0.003034 | -0.001768 | -0.002257 |
| add_uncertainty | +0.000677 | -0.000338 | +0.001029 |
| add_market_shape | +0.001355 | -0.001031 | -0.004170 |
| add_mechanism_interactions | +0.006346 | +0.007658 | +0.007837 |
| remove_recent | -0.000337 | +0.003789 | +0.003653 |
| remove_long | -0.001646 | +0.000559 | +0.000631 |
| remove_context | +0.013977 | -0.002137 | -0.000238 |
| recent_only | +0.010378 | -0.000752 | +0.001652 |
| draw_times | +0.012755 | +0.006323 | +0.006400 |
| section_pressure | +0.010830 | +0.002562 | +0.002225 |
| margin_recency | -0.006663 | -0.011356 | -0.006763 |

Recent/long/context removal results are mixed, rather than establishing one dominant group. The existing fixed box extension remains a small later-period lead, while more flexible track×box and draw-distance constructions do not provide stable validation support. Market structure improves later inner folds but power calibration is unstable later. Sparse comparable time/section measurements and layout exclusions limit power. [Feature audit](offline_systematic_evidence/feature_audit.json) records every missing value count and source timing.

**Vulnerable favourites.** Broad frequencies are unchanged: unique favourites won 77/171 (45.0%), with six tied-favourite races treated separately. Threshold/condition selection used earlier OOF predictions, requiring at least 20 selected runners **and** 20 races across five dates and positive binary-Brier improvement over the same-odds market. No qualifying rule means abstention. Twenty favourite rules and 48 outsider rules were examined per outer period, including comparable losing runners; winners never determined the evaluation population.

The chosen favourite rules selected **16/177 races**, with **7 wins and 9 losses** (43.75%), versus market mean **40.28%** and model **33.69%**. There were 161 abstention races. This is the opposite of convincing favourite overestimation. Period counts/wins were 14/6, 2/1 and 0/0; no rule qualified for the final period. Model binary Brier on selected favourites worsened by +0.004983. Actual-minus-market date-block interval is [−.3021,+.3178], and leave-one-date-out estimates range from −.0465 to +.1338. Median quoted odds were 1.95. **No useful vulnerable-favourite rule established.**

**Underestimated outsiders.** Chosen rules required improving recent margins, odds ≥10 then ≥15, and model excess probability ≥.005. The final fold had no qualifying rule. Across all later races they selected **35 runners in 32/177 races**, with **3 wins / 32 losses** (8.57%), market mean **4.61%**, model **5.83%**, median odds 20 and range 10–35; 145 races had no selection. Period counts/wins were 29/3, 6/0 and 0/0. These three wins occurred on only two dates. Removing June 26 leaves one win among 32 selections and reverses actual-minus-market from +.0396 to **−.0137**. The date-block interval [−.0487,+.2188] is wide and crosses zero. Nearby odds/edge thresholds do not establish stability. **Promising-looking small-sample excess is inconclusive, not an outsider edge.**

For context all odds≥10 runners across the same 177 races number 528, with 10 wins (1.89%) versus mean normalized market probability 3.88%; a long-priced winner is not by itself evidence of underestimation. Every selected and unselected runner, loss and abstention is in [outer_predictions.csv](offline_systematic_evidence/outer_predictions.csv). [Selection results](offline_systematic_evidence/selection_results.json), [threshold sensitivity](offline_systematic_evidence/selection_sensitivity.json) and [date/win/nearby-odds influence](offline_systematic_evidence/selection_influence.json) retain full denominators. Post-fit sensitivities neither refit models nor choose new rules.

**Shortlist and decision.** At most two candidates merit carrying forward as fixed exploratory comparisons: (1) the 16-feature capped market residual with a simple box term, L2=1 and full strength; (2) the unchanged 16-feature method at half residual strength. Both improve average scores across the three periods, but neither has simultaneous evidence of an advantage or consistently better calibration. These are methods to refit/freeze before genuinely new observations, not claims that old production coefficients were tested independently. The adaptive linear search, ensemble and vulnerable-favourite/outsider rules are not advanced. The small nonlinear gain is retained in evidence but does not justify adding a third candidate given its weaker proper-score gain and complexity.

The evidence-based stopping conditions are met: validation gains failed later for the selected complex linear recipes; selected outsider gains depend on one influential date; only 15 test dates and sparse field signals leave material uncertainty; no qualified untouched population exists. Continuing to combine these features would mostly search known outcomes. The next useful observation is a complete, prospectively retained pre-decision field/quote/form record with verified scratch and result closure, under separate future authority. No provider requests, new collection, service changes, protocol amendments, betting or promotion occurred here.

**Artifacts, latency and reproduction.** The [1,000-entry experiment ledger](offline_systematic_evidence/experiment_ledger.jsonl) records fit starts/completions, all validation trials, stage budgets, eliminations, selection rules, freezes and source identities. Its SHA-256 chain was independently verified; existing entries were not edited. The [preparation ledger](offline_systematic_evidence/preparation_ledger.jsonl) includes the corrected feature-construction KeyError before any model fitting; no failed candidate disappeared. A [diagnostic continuation journal](offline_systematic_evidence/diagnostic_ledger.jsonl) binds later sensitivity reporting to the completed model ledger. Raw-card sources, prepared feature rows, all inner OOF arrays, final coefficient/preprocessing receipts and research-only tree artifacts remain in `/home/l4nd0/greyhound-offline-systematic-output-20260924/`. Source snapshots and pinned hashes allow exact replay without changing the original experiment.

Prediction timing, excluding feature preparation and fitting, was roughly 1.0–7.7 ms for selected linear batches of 16–86 races and 1.2–8.7 ms for nonlinear batches of 16–86 races (about .06–.10 ms/race in batch). This is offline cached-feature inference latency, not end-to-end provider or live service latency. Full environment: Python 3.11.15, numpy1.26.4, scipy1.16.1, scikit-learn1.7.1; one BLAS/OpenMP thread. [Execution identities/cost](offline_systematic_evidence/execution.json) are retained. Inputs require the pinned local source files; no download/replacement fallback is performed.

```bash
cd /home/l4nd0/greyhound-offline-prediction-20260924
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
# Use a NEW sibling output root; prior ledgers and runs must be preserved.
"$RESEARCH_PY" scripts/offline_prediction_research.py prepare --out /new/root/foundation
"$RESEARCH_PY" -m scripts.offline_systematic_features --prepared /new/root/foundation --out /new/root/features
nice -n 10 "$RESEARCH_PY" -m scripts.offline_systematic_search --features /new/root/features --out /new/root/search_v1
"$RESEARCH_PY" -m scripts.offline_systematic_selection_diagnostics --out /new/root/search_v1
"$RESEARCH_PY" tests/test_offline_prediction_research.py
"$RESEARCH_PY" tests/test_offline_systematic_research.py
python3 tests/test_offline_form_packet.py
```

Validation: 20 focused synthetic tests (six prior scoring seams, five source/timing seams, nine new systematic seams), complete source reconstruction, the executed finite search and independent source/method review. Remaining uncertainty is explicitly scientific and source-related, not hidden by a new split. The report’s recommendation is **COLLECT_MORE_QUALIFIED_DATA / RETAIN_BASELINE**, with two exploratory candidate methods and no production change.
