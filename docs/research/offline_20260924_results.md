# Offline greyhound development results — 24 September 2026

Executed 13 model variants on **177 chronological evaluation races / 1,251 runners**. The strongest average result was a small residual model with box position: log loss 1.397706 versus market 1.412894. Its primary date-block 95% interval includes zero. Half-strength residuals narrowly improved the market under that unadjusted interval. These are exploratory leads, not confirmed predictive or executable advantages. Form-only models failed to beat the market; outsider selections did not establish underestimation. No model was promoted.

**Boundary incident.** An initial capture-lag inventory, before the separate form-only reservation manifest was identified, decoded whole canonical JSON rows dated before July 17. That transiently decoded label fields for **58 reserved July 15–16 races / 393 runners**. The probe emitted only date/capture-lag aggregates; no target label values were displayed or used for fitting, scoring or selection. This exceeded the requested access restriction. The main comparison and all its histories are disjoint from those records. Original studies and records were not modified, and no protected-population performance was computed. This session cannot claim zero protected-outcome access. The exact identity-only incident record is [preserved here](/home/l4nd0/greyhound-offline-research-output-20260924/protected_projection_incident.json); it must accompany any later review of that reservation’s exposure history. No protocol or eligibility status was amended.

**Population and timing.** The located corrected Sportsbet surface contains 1,153 races / 8,234 runners spanning June 10–August 2, 2026. The old price column was affected by WIN/PLACE mislabelling; this experiment pins and verifies the corrected matrix and paired-column provenance sidecar. Original contaminated metrics, late TheDogs OPEN/LOW/HIGH prices, and the closed 114-race recovery are not baselines here.

The raw-card development inventory contains 917 races / 6,456 runners, May 27–July 9, across 42 source venue labels. Hash-verified cards were recorded at least 60 minutes before scheduled jump. Reconstruction retains 868 races / 6,119 runners; 49 fail exact complete card/sidecar/target-roster agreement. The old materialized feature file differs from the current descriptor, so it was not used for feature values. Explicit metre suffixes such as `400m` are parsed in this isolated loader; production parsing is unchanged.

Before comparisons, inclusion was fixed to June 10–July 9, complete exact race/box/name-token fields, one winner, valid finish positions, matching form/market scheduled times, verified corrected fixed WIN prices and a retained atomic snapshot **2–10 minutes before scheduled jump**. This answers a **T−2 decision using a quote no more than eight minutes old**. It is a retrospective retained-snapshot availability subset, not proof that every scheduled race was serviceable at T−2; no later price or starting price was substituted.

The intersection is **332 races / 2,367 runners, June 10–July 8, across 36 venues**. Lead-time quartiles are 4.64 / 6.54 / 8.02 minutes (range 2.03–9.19). Of the 332 races, 328 carry official-race-page label provenance and four published-history provenance. **All 177 evaluation races carry official label provenance**; the four other races only enter earlier development periods. Label values come from the pinned corrected canonical matrix; provenance classes are joined from exact race metadata, not a fresh re-extraction or adjudication of raw results. The source universe is already conditioned on result availability: full-schedule missing-result/abandonment rates cannot be recovered from these packets.

Race identity is exact original race ID plus box, corroborated by normalized dog token in hash-bound card and runner metadata. No fuzzy venue/name joining is used. Duplicate box/race rows fail; raw histories are deduplicated and deterministically ordered by the existing canonical parser. Histories are strictly earlier calendar dates, capped at 20 starts; “career” here means that retained history, not future-updated lifetime statistics. Same-day starts are excluded. No inferred scratch/reserve removal, target-race correction or dead-heat relabelling is performed. Non-single-winner, malformed finish or incomplete fields are excluded. Counts describe the prequalified source population and cannot establish that all original dead heats/abandoned races were retained upstream.

Exact exclusions and missingness are in [dataset_assessment.json](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3/dataset_assessment.json) and [exclusions.json](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3/exclusions.json). Sequential exclusions partition the 1,153 races: 291 outside this analysis/reserved; 230 outside the odds window only; 141 outside that window and missing qualified form; 159 missing qualified form only; 332 included. In retained runner features, same-venue win rate is missing 40.0%, same-distance 22.1%, same-grade 31.5%; no matching history yields a missing rate, not an invented win rate. Training medians and explicit missing indicators handle it.

Protection is based on manifests, not dates alone: closed 114 identities, separate 88-race form-only reservation, 1,076-race August frozen odds manifest, and the Sportsbet/Betfair predecessor/replacement and October successor declarations. Their exact identity union is **1,265** (overlaps preserved through source manifests); the earliest reserved target is July 15. All research targets are at most July 9 and all history dates precede their targets, preventing indirect protected outcomes in features or aggregates. Identity records and source hashes are in [protected_records.json](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3/protected_records.json). This enforced modelling boundary does not erase the initial probe incident above.

**Frozen predictor inspection.** Its actual scorer and model bytes match the installed R3 release package identified by the on-disk user-service unit. The model SHA256 is `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`; scorer SHA256 is `50039cbc46f48d2f2d0dda8973e75dc73055872ff74b082821535060cc36b7f6`. This is package-byte verification, not a live service health claim. It fitted 678 races / 4,752 runners through July 9; testing those frozen coefficients here would be in-sample. The executed baseline retrains its capped conditional-softmax method using only earlier dates, with rebuilt as-of features and corrected market prices. See [full source/provenance note](offline_20260924_model_provenance.md).

Its 16 features cover history depth/recency, recent finish/win/place/margin, retained-history win/place/finish, and venue/distance/grade history. Preprocessing adds 16 missing indicators, training-only median/scale fitting and within-race centering. The market offset remains fixed; residuals use `0.35*tanh(Xβ/0.35)`. The faithful objective is mean race log loss plus `0.5||β||²`. The independent form-only regularized objective uses `0.5||β||²/N_train_races`; this is a separate benchmark, not a claim of identical regularization scaling. Full and half residual predictions share one fit.

**Exact chronological comparisons.** Each date stays wholly within its phase. Training begins June 10 and expands. No training/validation/evaluation race overlap within a fold. Earlier evaluation dates may enter later training, as required for expanding walk-forward prediction. The final block was not used for this run’s tuning, but is historically inspected development data—not a pristine holdout.

| Block | Training through | Validation | Evaluation | Train / validation / test races |
|---|---|---|---|---|
| period1 | 2026-06-17 | 2026-06-18–2026-06-21 | 2026-06-24–2026-06-30 | 114 / 41 / 86 |
| period2 | 2026-06-24 | 2026-06-25–2026-06-30 | 2026-07-01–2026-07-02 | 162 / 79 / 16 |
| final_later_block | 2026-06-30 | 2026-07-01–2026-07-02 | 2026-07-03–2026-07-09 | 241 / 16 / 75 |

Observed final-block races end July 8; no eligible July 9 race exists. The middle validation/test blocks are small. The tree uses 60 iterations, seven leaves, minimum leaf 30, L2 10 and learning rate .05, without early stopping. Only validation selects market power in [0.5,1.75], form temperature in [0.5,2], and tree blend in {0,.1,.25,.5}. All feature variants and selection thresholds were specified before comparative performance. The last validation fit hit the upper calibration bounds; they were not enlarged. The final primary fit/evaluation took **9.46 seconds**, with one BLAS/OpenMP thread and `nice 10`; preparation took 2.22 seconds. Initial host load was 0.45/0.67/0.79, with about 27 GB available memory. CPU-seconds and peak RSS were not measured.

**Paired results.** Lower log loss and Brier are better. Brier is the within-race sum of squared runner errors, averaged across races. Top-choice accuracy splits credit across tied top probabilities. Intervals are 2,000 paired calendar-date block bootstraps over 15 evaluation dates, seed 20260924. They are unadjusted for 13 model comparisons and prior historical experimentation.

| Variant | Log loss | Brier | Top choice | Δ log loss vs market | 95% date-block interval |
|---|---:|---:|---:|---:|---|
| uniform | 1.944654 | 0.855260 | 14.47% | +0.531761 | [+0.449679, +0.621322] |
| market | 1.412894 | 0.682928 | 45.20% | +0.000000 | [+0.000000, +0.000000] |
| market_power_calibrated | 1.404411 | 0.686885 | 45.20% | -0.008482 | [-0.053448, +0.041389] |
| form_regularized | 1.982550 | 0.878063 | 23.73% | +0.569656 | [+0.443377, +0.709732] |
| form_boosted | 1.841991 | 0.823747 | 25.99% | +0.429097 | [+0.354160, +0.497553] |
| market_plus_boosted | 1.412894 | 0.682928 | 45.20% | +0.000000 | [+0.000000, +0.000000] |
| residual_frozen_method | 1.400139 | 0.678978 | 44.63% | -0.012755 | [-0.030015, +0.003095] |
| residual_half | 1.404845 | 0.680157 | 45.20% | -0.008049 | [-0.016655, -0.000137] |
| residual_without_recent | 1.402288 | 0.680020 | 44.63% | -0.010606 | [-0.027561, +0.005807] |
| residual_without_context | 1.405567 | 0.679040 | 44.63% | -0.007327 | [-0.017936, +0.004497] |
| residual_recent_only | 1.406281 | 0.679537 | 44.63% | -0.006613 | [-0.015487, +0.003073] |
| residual_ewma | 1.402055 | 0.679776 | 44.63% | -0.010839 | [-0.027662, +0.005090] |
| residual_with_box | 1.397706 | 0.677435 | 44.63% | -0.015188 | [-0.031467, +0.000183] |

The validation-selected tree blend was zero in all three periods: that comparison exactly reproduces market predictions, not a tree improvement. Form-only regularization was worse even than uniform; the shallow tree beat uniform but was far worse than the market. Faster recent weighting (EWMA, fixed half-life three starts) did not improve the full residual: paired Δ=+0.001916, interval [+0.000137,+0.003647]. Removing recent information worsened loss by +0.002149, with an interval crossing zero. Removing venue/distance/grade context worsened by +0.005428, also uncertain. Box position added a small increment over the full residual: −0.002433 [−0.004830,−0.000142], exploratory and unadjusted.

| Variant | June 24–30 ΔLL | July 1–2 ΔLL | July 3–8 ΔLL |
|---|---:|---:|---:|
| market_power_calibrated | -0.022775 | -0.145084 | +0.037048 |
| residual_frozen_method | -0.005040 | -0.037644 | -0.016291 |
| residual_half | -0.004360 | -0.020037 | -0.009722 |
| residual_ewma | -0.002705 | -0.037169 | -0.014549 |
| residual_with_box | -0.006360 | -0.036172 | -0.020834 |

Full and half residuals improve the mean score in every block and on 10/15 dates; box improves on 12/15. Removing any single date leaves full residual ΔLL negative (−0.01734 to −0.00851), so no single date accounts for the average gain. A post-fit three-consecutive-observed-date bootstrap sensitivity also stays negative for these variants; its narrower intervals do not supersede the primary interval. Fifteen date blocks, repeated dogs across days, overlapping training histories, and extensive earlier candidate inspection limit inference. Full and box primary intervals still include zero; the half-strength bound is only narrowly negative.

Calibration is separate from proper-score improvement. Ten fixed equal-width runner bins give ECE 0.02108 for market, 0.02886 full residual, 0.02460 half and 0.02727 box: lower log loss did **not** establish better calibration. Uniform has ECE zero because pooled within-race averages match by construction, yet has poor predictive scores; ECE alone is not a model-selection criterion. Validation-only market-power calibration improved pooled log loss slightly, worsened Brier and ECE, and regressed on the final period. Full bin counts and predicted/observed rates are retained in [results.json](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3/results.json).

**Favourite and outsider findings.** There are 171 unique favourites and six tied-favourite races among all 177 evaluated races. Unique favourites won **77/171 (45.0%)**, versus average normalized market chance 39.8%; the date-bootstrap actual-minus-market interval crosses zero. Winning quoted odds have median 2.70, quartiles 1.95/4.80, 90th percentile 7.20 and maximum 23.00. Ten of 177 winners had quoted odds ≥10. These prices are retained decision-time quotes, not SP.

The predeclared primary vulnerable-favourite rule is full residual probability at least five percentage points below the market. It selected **10/177 races**, with **1 win / 9 losses**, mean market probability 37.5%, model 31.5%, median odds 2.10. Selection counts/wins by period are 8/1, 1/0 and 1/0. Its apparent weakness is concentrated in the first period. An illustrative independent-binomial Wilson interval for 1/10 is about 1.8%–40.4%, encompassing the mean market chance; date-bootstrap estimates with only one observed win are also unstable. This is a hypothesis lead, not reliable identification of true chances.

The primary outsider rule is decimal WIN odds ≥10 and model minus market probability ≥.02. It selected **one runner in one of 177 races**, lost, and made no selection in 176 races. At the weaker predeclared +.005 sensitivity threshold, it selected 65 runners across 55 races, with **3 wins / 62 losses** (4.62%) against mean market 5.46% and model 6.42%; median odds 14.0. Its actual-minus-market date-block interval crosses zero. All ≥10 runners—not just selected winners—number **528**, with **10 wins / 518 losses** (1.89%) versus market mean 3.88%. The data does not support underestimated outsiders as a general claim.

Every sensitivity below is preserved; none was selected for maximum historical return. Selection frequency uses all 177 races, including no-selection races. All runner losses remain in both training and evaluation.

| Rule | Threshold | Runners selected | Races selected / 177 | No-selection races | Wins / selections | Mean market / model chance |
|---|---|---:|---:|---:|---:|---:|
| vulnerable_favourite | −0.005 | 70 | 70 / 177 | 107 | 30 / 70 | 37.33% / 34.51% |
| vulnerable_favourite | −0.010 | 60 | 60 / 177 | 117 | 23 / 60 | 36.70% / 33.53% |
| vulnerable_favourite | −0.020 | 42 | 42 / 177 | 135 | 15 / 42 | 36.86% / 32.99% |
| vulnerable_favourite | −0.050 | 10 | 10 / 177 | 167 | 1 / 10 | 37.49% / 31.47% |
| vulnerable_favourite | −0.100 | 0 | 0 / 177 | 177 | 0 / 0 | — |
| outsider | odds≥8, +0.005 | 96 | 75 / 177 | 102 | 5 / 96 | 6.49% / 7.57% |
| outsider | odds≥8, +0.010 | 42 | 37 / 177 | 140 | 2 / 42 | 7.12% / 8.66% |
| outsider | odds≥8, +0.020 | 5 | 5 / 177 | 172 | 0 / 5 | 8.51% / 10.73% |
| outsider | odds≥8, +0.050 | 0 | 0 / 177 | 177 | 0 / 0 | — |
| outsider | odds≥10, +0.005 | 65 | 55 / 177 | 122 | 3 / 65 | 5.46% / 6.42% |
| outsider | odds≥10, +0.010 | 22 | 21 / 177 | 156 | 0 / 22 | 5.71% / 7.15% |
| outsider | odds≥10, +0.020 | 1 | 1 / 177 | 176 | 0 / 1 | 7.63% / 9.64% |
| outsider | odds≥10, +0.050 | 0 | 0 / 177 | 177 | 0 / 0 | — |
| outsider | odds≥15, +0.005 | 26 | 23 / 177 | 154 | 2 / 26 | 3.90% / 4.74% |
| outsider | odds≥15, +0.010 | 7 | 7 / 177 | 170 | 0 / 7 | 4.34% / 5.65% |
| outsider | odds≥15, +0.020 | 0 | 0 / 177 | 177 | 0 / 0 | — |
| outsider | odds≥15, +0.050 | 0 | 0 / 177 | 177 | 0 / 0 | — |

Full odds quantiles and all period denominators are in [favourite_outsider.json](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3/favourite_outsider.json). The investigation compares losing as well as winning outsiders and tests form incrementally over a fixed odds offset. Individual “true chances” are not observable from one race; statements about mispricing are group-level hypotheses with uncertainty.

No returns were calculated. Quote persistence/acceptance at T−2, scratch timing and deductions are unverified; normalized probabilities are not executable prices. Fixed-quote predictive scores do not establish a practical betting advantage.

**What the remaining source fields can support.** A separate outcome-free audit of the exact 332 races found 11,487 repeated prior-start observations. All 6,920 single-digit PIR entries equal finishing position, so interpreting them as first-call early speed would manufacture information. Only 1,102/2,367 runners have any multi-digit PIR; semantics need source clarification. Matching venue/distance sectional history exists for 878 runners, at least three starts for 358; only seven full fields have ≥3 for every runner. Matching venue/distance times exist for 1,246 runners (639 with ≥3). Margins and finishes have ≥3 observations for 2,293 runners. Counts and source limitations are in the [provenance note](offline_20260924_model_provenance.md) and [signal audit](/home/l4nd0/greyhound-offline-research-output-20260924/signal_coverage.json). Grade comparisons use exact normalized categories; no invented cross-jurisdiction grade ordering or class-change scale was used.

Prioritized follow-up experiments: (1) carry the full/half residual and a single box extension as fixed candidates into a separately authorized future confirmation population, preserving all current reservations and this exposure record; (2) one offline margin/finish-improvement extension on the broad supported history, evaluated against matched losers and the same odds offset; (3) investigate venue/distance-comparable time and sectional definitions before a small pace extension. Another broad hyperparameter sweep is not justified.

**Reproduction and account.** Worktree: `/home/l4nd0/greyhound-offline-prediction-20260924`, branch `research/offline-prediction-20260924`, base `1570dbadd1aca25deb249c41cf288101275b0b2b`. Frozen inputs, rebuilt matrix, individual OOF probabilities, coefficients/preprocessing, validation choices, all 13 variants, all rule sensitivities and environment identities are in [baseline_v3](/home/l4nd0/greyhound-offline-research-output-20260924/baseline_v3). `runner_executed.py` preserves the exact evaluated source hash; the current runner adds a split-equality assertion and explicit objective documentation without changing models. `baseline_v1` and `baseline_v2` are retained preparatory outputs superseded before fitting when metre-suffix parsing was repaired; they produced no model scores. No unsuccessful variant was deleted.

Runtime used Python 3.11.15, numpy 1.26.4, scipy 1.16.1 and scikit-learn 1.7.1. Use the pinned interpreter recorded in results.json and a new output directory; prepare fails if the directory exists and evaluation fails if results already exist. The source files live outside Git and are required by their pinned hashes. No network, provider request, canonical DB read during model construction, service operation, production configuration write, betting, protocol amendment or promotion is part of this runner. Local unit-file/model-byte reads established the installed model identity.

```bash
cd /home/l4nd0/greyhound-offline-prediction-20260924
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
"$RESEARCH_PY" scripts/offline_prediction_research.py prepare --out /path/to/new-output
nice -n 10 "$RESEARCH_PY" scripts/offline_prediction_research.py evaluate --out /path/to/new-output
"$RESEARCH_PY" -m scripts.offline_prediction_diagnostics --out /path/to/new-output
"$RESEARCH_PY" tests/test_offline_prediction_research.py
python3 tests/test_offline_form_packet.py
```

Ten focused synthetic tests cover protected target exclusion before card access, strict history timing, source mutation, metre parsing, normalization, train-only preprocessing, within-race permutation invariance, the exact capped/ridge objective, disjoint date splits and identity projection without decoding labels. The full form rebuild and one complete model run passed. Results remain **EXPLORATORY_CANDIDATE_EVIDENCE**, with **NO_OUTSIDER_EDGE_ESTABLISHED** and **NO_PRODUCTION_CHANGE**.
