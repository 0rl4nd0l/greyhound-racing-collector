# Why the residual helped and hurt — retained development forecasts, 29 September 2026

**Verdict: small, uncertain average probability-score gains; meaningful individual harm; insufficient qualified movement data.** The existing 16-input residual method improves mean log loss by **0.01529** and Brier by **0.00554**, but its top-ranked accuracy falls from **45.20% to 44.07%**. Both adjusted intervals include deterioration. This explains forecasts already retained in [PR #193](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/193), not a new independent model test or a profitability claim. Production and the October prospective study are unchanged.

The [plan](market_explanation_20260929_plan.md) preceded these comparisons. There were **zero new fits, zero candidate/selection-rule trials, zero provider requests, zero database opens and zero protected target decodes**. All 177 chronological evaluation races / 1,251 runners / 15 dates (June 24–July 8) are included. The other 154 of 331 admitted development races were training-only and have no retained outer forecasts; they are not treated as missing evaluation records. There are six tied-favourite races; fractional top-rank tie credit is retained. Every original exclusion is preserved in [source_exclusions.json](market_explanation_20260929_evidence_verified/source_exclusions.json), including Taree R2 June 13's timing discrepancy. Missing base inputs are imputed using the saved earlier-training medians and missing indicators, not omitted races.

The exact original reservation union and 58-race incident manifest were checked before decoding retained records; all rows underwent identity-only admission and complete-roster checks. No mixed database/history search occurred. The earlier protected exposure is not reversed by exclusion, and the inherited cutoff context had protected timing-metadata influence, as documented in the [foundation audit](offline_systematic_foundation_audit.md). The original corrected source selects latest-prejump quotes and then filters to T−2–T−10: inclusion is retrospective availability-selected, not complete-schedule T−2 execution. Prices, scratch timing, deductions and accepted execution do not support returns.

## Overall scores

Improvement is **market loss minus model loss**, so positive is better (opposite the Δ sign in #193). Race Brier is the sum of squared runner probability errors, then averaged over races. All rows use identical races and normalized corrected WIN inverse odds.

| Forecast | Races / dates | Log loss | Brier | Top accuracy | LL improvement | Simultaneous 95% | Brier improvement | Simultaneous 95% |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Market | 177 / 15 | 1.412894 | .682928 | 45.20% | — | — | — | — |
| refit_base16 | 177 / 15 | 1.397606 | 0.677388 | 44.07% | +0.015287 | [-0.01475, +0.04532] | +0.005541 | [-0.01019, +0.02127] |
| refit_half | 177 / 15 | 1.403559 | 0.679370 | 45.76% | +0.009334 | [-0.00570, +0.02437] | +0.003559 | [-0.00429, +0.01141] |
| refit_box | 177 / 15 | 1.396280 | 0.676556 | 44.07% | +0.016614 | [-0.01203, +0.04526] | +0.006372 | [-0.00834, +0.02108] |

Uncertainty uses 3,000 paired calendar-date bootstrap draws, seed 20260929, with common date weights. The 95% simultaneous bands use the maximum standardized deviation across **128 estimable proper-score contrasts** (three overall models and the subgroups spanning at least two dates, each with LL/Brier; all 78 subgroup rows remain reported). These are a broader reporting family than #193's seven-model family, so intervals differ. They do not correct the entire history of experimentation. All pointwise intervals, bootstrap nonempty counts and leave-one-date-out ranges are in [table.json](market_explanation_20260929_evidence_verified/table.json). Groups with fewer than 20 races or five dates are marked sparse; one-date intervals are explicitly not estimable and are excluded from the simultaneous family. No interaction or between-subgroup significance test was performed.

## Where gains and harms occurred

Base16 improves log loss on **102 races**, worsens it on **75**; Brier improves on **93**, worsens on **84**. Log-loss improvement sums to **+8.8342** across positive races and **−6.1284** across negative races, net **+2.7058**. The best five contribute **+1.1690**, or **43.2% of the net** (13.2% of gross positive improvement). Excluding those five leaves **+0.00893** mean improvement; excluding the five worst leaves **+0.02364**. These are retrospective influence descriptions, not race-selection strategies.

Eleven of 15 dates improve. The strongest mean gain is July 5, **+0.08120 across 11 races**; strongest mean harm July 4, **−0.04716 across 11 races**. June 26, June 27 and July 7 also deteriorate. Removing any one date leaves mean LL improvement **+0.01092 to +0.01943** and Brier **+0.00347 to +0.00828**. Gains are therefore not wholly attributable to one date or one outsider, although sample uncertainty remains large. [Every date](market_explanation_20260929_evidence_verified/date_losses.json), [every race](market_explanation_20260929_evidence_verified/race_losses.json), and [best/worst influence](market_explanation_20260929_evidence_verified/influence.json) are retained.

Seven-runner fields have the clearest within-group association: **44 races / 14 dates**, LL gain **+0.04269**, simultaneous interval **[+0.01580,+0.06958]**, Brier gain **+0.01623**, interval **[+0.00437,+0.02809]**. Eight-runner fields show almost no average LL gain (**+0.00035; 80 races / 15 dates**). This is a newly described association, not proof that vacancies cause improvement or that a seven-runner-only rule works. Roster omissions identify unoccupied boxes but do not identify their scratch/vacancy cause. The 18 complete-input races also look better, but that subgroup is sparse and overlaps other conditions.

Strong favourites (probability ≥.6) regress on average, but there are only **10 races / 8 dates**, and leave-one-date-out changes sign. Low model disagreement regresses slightly (**22 races / 11 dates**, TV<.02), with intervals crossing zero. Venue averages range widely: Taree **−.08256, 5 races/3 dates**, Geelong **−.07765, 6/3**, versus Capalaba **+.05337, 22/5**. Sparse venue findings do not justify specialist models. Distance, literal grade, history/recency, competitiveness and favourite box are fully reported below; no attractive subgroup was silently promoted.

## What mechanically changed the probabilities

For runner i, using the saved earlier-training preprocessing and coefficients:

`x_i = race_center((concat(impute(raw_i, train_median), missing_i) − train_mean) / train_scale)`

`z_i = sum_j x_ij * beta_j; r_i = strength * 0.35 * tanh(z_i / 0.35)`

`log(p_model_i / p_market_i) = r_i − log(sum_k p_market_k * exp(r_k))`.

Thus the winner's log-probability change equals the race's LL improvement exactly. A positive raw residual need not increase final probability if market-weighted normalization is larger. The smooth tanh cap shrinks extremes; it is not a hard probability clip. Half strength halves the capped residual before normalization, so it does not simply halve probability differences. Reconstruction agrees with every retained base16/half probability to **2.22e−16**. [All runner contributions](market_explanation_20260929_evidence_verified/runner_contributions.jsonl) preserve raw inputs, all 32 coefficient products, cap effect, normalizer and final log change; [three fold receipts](market_explanation_20260929_evidence_verified/base16_receipts.json) preserve medians, scales and coefficients.

Largest average absolute contributions before the cap (all 1,251 runners):

| Input | Mean absolute contribution | Coefficient across chronological fits |
|---|---:|---|
| recent_avg_margin_5 | 0.02505 | +0.03697, +0.04352, +0.04252 |
| win_rate_same_venue | 0.02490 | +0.04635, +0.05403, +0.04716 |
| starts_same_distance | 0.02254 | +0.04475, +0.02594, +0.02572 |
| starts_same_venue | 0.01989 | +0.05033, +0.03785, +0.04004 |
| missing::win_rate_same_distance | 0.01744 | -0.03581, -0.01814, -0.02347 |
| recent_win_rate_5 | 0.01673 | +0.02718, +0.02809, +0.02989 |
| career_win_rate | 0.01673 | +0.02718, +0.02809, +0.02989 |
| recent_finish_best_5 | 0.01660 | -0.03443, -0.02260, -0.01789 |

Coefficients explain the computation conditional on other correlated inputs; they do not measure causal sporting effects. The positive recent-margin coefficient does **not** mean being beaten farther causes wins. Missing-indicator contributions are nonzero for 1,033 runners because race centering also changes the contribution of nonmissing runners relative to their field. “Career” here means available retained card history, not a verified whole career. Recent and career win/place rates are identical for all 1,251 evaluated runners; separate coefficient entries therefore are not independent sources of evidence. Same-distance context is not necessarily same-track distance; literal grades are not a universal class ordering. Grade spellings remain separate. These definitions follow [offline_form_packet.py](../../scripts/offline_form_packet.py) and [canonical feature construction](../../scripts/build_form_only_v1_packet.py).

The base16 method has **no direct box term**; favourite-box associations cannot be called a box mechanism. #193 retained box predictions but omitted the fitted box coefficients from its receipts. Exact box-term decomposition is therefore unavailable without a new refit, which this analysis deliberately did not substitute for retained evidence. This limitation does not affect scoring its saved forecasts. No PIR, sectional or early-speed interpretation was introduced; [PR #195](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/195)'s semantic stop remains intact.

## Balanced forecast explanations

Examples were selected retrospectively as largest help, largest harm, and median positive/negative LL changes. They explain outcomes and must not become winner-defined deployment rules. Full fields, losers and all terms are in [examples.json](market_explanation_20260929_evidence_verified/examples.json).

| Selection / race | Winner | Market → model | LL improvement | Brier improvement | Winner linear sum → capped residual; normalizer |
|---|---|---:|---:|---:|---|
| largest_help: Race 6 - WAR - 2026-06-29 | NOTORIOUSMILO (box 4) | 0.1423 → 0.1844 | +0.25900 | +0.10332 | +0.25119 → +0.21541; -0.04359 |
| largest_harm: Race 11 - MURR - 2026-06-26 | PANIPUCCI (box 8) | 0.0819 → 0.0607 | -0.29978 | -0.07661 | -0.27267 → -0.22826; +0.07152 |
| median_positive: Race 5 - MURR - 2026-06-26 | HELDFORRANSOM (box 4) | 0.5008 → 0.5454 | +0.08518 | +0.05202 | +0.16584 → +0.15445; +0.06927 |
| median_negative: Race 4 - GAWL - 2026-06-30 | BONMATI (box 4) | 0.2244 → 0.2121 | -0.05660 | -0.02625 | -0.05538 → -0.05493; +0.00168 |

Warrnambool R6: Notorious Milo's same-venue win-rate contribution (+.08908) and best recent finish (+.04718) helped lift its probability. The losing favourite Remi Joy fell .3641→.3059, driven partly by missing distance-context (−.05673) and fewer venue starts (−.04321). Neither forecast ranked the winner first: probability quality improved without winner accuracy changing.

Murray Bridge R11: Pani Pucci was penalized by missing same-grade win rate (−.07780), fewer same-grade starts (−.06827) and best recent finish (−.04246); margin (+.05294) partly offset this. Normalization (+.07152) reduced its probability further. The losing favourite Terry Keeping rose .6274→.6573. The same context/missingness machinery can help one field and hurt another.

Murray Bridge R5: winning favourite Held For Ransom gained from same-grade starts (+.03953), win rate (+.03503) and relative missingness (+.03112), partly offset by margin (−.03594); it remained top-ranked. Gawler R4: Bonmati's recent and career win-rate terms each contributed −.02291 and best recent finish −.02264. The losing favourite Tommy Fire rose .3342→.3420; neither forecast chose Bonmati. These moderate examples counterbalance the extremes.

## Complete fixed subgroup table

Base16 only. Each factor partitions the complete population; factors overlap one another and are not independent replications. Values show market→model losses and fractional accuracy, and simultaneous LL/Brier improvement bands. `*` flags <20 races or <5 dates. Literal source grade is descriptive, without cross-jurisdiction equivalence. Full numerical precision, pointwise bands and each subgroup's date-removal range are in the machine-readable table.

| Factor: group | Races/dates | LL market→model | Brier market→model | Accuracy market→model | LL improvement [sim95] | Brier improvement [sim95] |
|---|---:|---:|---:|---:|---|---|
| favourite_probability: .4-.6 | 67/15 | 1.2438→1.2188 | 0.6168→0.6105 | 0.552→0.552 | +0.0249 [-0.0123,+0.0621] | +0.0063 [-0.0134,+0.0261] |
| favourite_probability: <.4 | 100/15 | 1.5736→1.5620 | 0.7479→0.7421 | 0.360→0.340 | +0.0116 [-0.0268,+0.0500] | +0.0058 [-0.0174,+0.0289] |
| favourite_probability: >=.6* | 10/8 | 0.9390→0.9511 | 0.4761→0.4781 | 0.700→0.700 | -0.0121 [-0.1822,+0.1580] | -0.0020 [-0.0843,+0.0802] |
| entropy: <.8 | 72/15 | 1.2594→1.2473 | 0.6130→0.6120 | 0.542→0.528 | +0.0121 [-0.0375,+0.0616] | +0.0010 [-0.0252,+0.0273] |
| entropy: >=.8 | 105/15 | 1.5181→1.5007 | 0.7309→0.7222 | 0.390→0.381 | +0.0175 [-0.0135,+0.0484] | +0.0086 [-0.0087,+0.0259] |
| field_size: 7 | 44/14 | 1.3388→1.2961 | 0.6573→0.6411 | 0.489→0.523 | +0.0427 [+0.0158,+0.0696] | +0.0162 [+0.0044,+0.0281] |
| field_size: 8 | 80/15 | 1.5784→1.5780 | 0.7235→0.7225 | 0.412→0.400 | +0.0004 [-0.0381,+0.0388] | +0.0010 [-0.0192,+0.0212] |
| field_size: <=6 | 53/13 | 1.2246→1.2096 | 0.6430→0.6395 | 0.481→0.434 | +0.0151 [-0.0369,+0.0670] | +0.0035 [-0.0245,+0.0315] |
| minimum_history: <5 | 37/11 | 1.4249→1.4154 | 0.6658→0.6629 | 0.514→0.486 | +0.0094 [-0.0154,+0.0343] | +0.0028 [-0.0109,+0.0165] |
| minimum_history: >=5 | 140/15 | 1.4097→1.3929 | 0.6875→0.6812 | 0.436→0.429 | +0.0168 [-0.0220,+0.0557] | +0.0063 [-0.0135,+0.0260] |
| maximum_recency: <=21 | 92/15 | 1.4706→1.4490 | 0.7074→0.7020 | 0.413→0.402 | +0.0217 [-0.0219,+0.0653] | +0.0054 [-0.0151,+0.0258] |
| maximum_recency: >21 | 85/15 | 1.3504→1.3420 | 0.6564→0.6507 | 0.494→0.482 | +0.0084 [-0.0336,+0.0503] | +0.0057 [-0.0120,+0.0234] |
| missing_base_input: False* | 18/11 | 1.2440→1.1804 | 0.5959→0.5698 | 0.472→0.500 | +0.0636 [+0.0154,+0.1118] | +0.0261 [-0.0042,+0.0565] |
| missing_base_input: True | 159/15 | 1.4320→1.4222 | 0.6928→0.6896 | 0.450→0.434 | +0.0098 [-0.0196,+0.0393] | +0.0032 [-0.0124,+0.0189] |
| disagreement_TV: <.02 | 22/11 | 1.2738→1.2779 | 0.6379→0.6399 | 0.500→0.500 | -0.0041 [-0.0225,+0.0143] | -0.0019 [-0.0134,+0.0095] |
| disagreement_TV: >=.02 | 155/15 | 1.4326→1.4146 | 0.6893→0.6827 | 0.445→0.432 | +0.0180 [-0.0160,+0.0521] | +0.0066 [-0.0111,+0.0243] |
| favourite_box: 1-2 | 66/15 | 1.3108→1.2981 | 0.6360→0.6266 | 0.561→0.561 | +0.0127 [-0.0429,+0.0683] | +0.0093 [-0.0174,+0.0361] |
| favourite_box: 3-6 | 56/14 | 1.5882→1.5733 | 0.7517→0.7497 | 0.375→0.339 | +0.0149 [-0.0305,+0.0603] | +0.0020 [-0.0208,+0.0248] |
| favourite_box: 7-8 | 49/15 | 1.3734→1.3560 | 0.6732→0.6695 | 0.388→0.388 | +0.0174 [-0.0300,+0.0648] | +0.0037 [-0.0214,+0.0289] |
| favourite_box: tied* | 6/5 | 1.2229→1.1929 | 0.6371→0.6256 | 0.500→0.500 | +0.0300 [-0.1916,+0.2515] | +0.0115 [-0.1111,+0.1341] |
| venue: AP K* | 8/1 | 1.4911→1.4942 | 0.7661→0.7742 | 0.375→0.375 | -0.0030 [not estimable] | -0.0081 [not estimable] |
| venue: BAL* | 3/1 | 1.1715→1.2142 | 0.5737→0.6024 | 0.667→0.667 | -0.0427 [not estimable] | -0.0287 [not estimable] |
| venue: BEN* | 4/2 | 2.1725→2.1709 | 0.9459→0.9508 | 0.250→0.000 | +0.0015 [-0.1401,+0.1432] | -0.0048 [-0.0671,+0.0574] |
| venue: BH* | 5/1 | 1.6519→1.6573 | 0.8573→0.8609 | 0.000→0.200 | -0.0055 [not estimable] | -0.0036 [not estimable] |
| venue: BULLI* | 1/1 | 2.0802→2.0823 | 0.9256→0.9286 | 0.000→0.000 | -0.0021 [not estimable] | -0.0030 [not estimable] |
| venue: CAPA | 22/5 | 1.2739→1.2205 | 0.6278→0.6113 | 0.523→0.545 | +0.0534 [-0.0198,+0.1266] | +0.0165 [-0.0158,+0.0488] |
| venue: CASO* | 4/2 | 1.1140→1.0808 | 0.5281→0.5082 | 0.750→0.750 | +0.0332 [-0.0196,+0.0859] | +0.0199 [-0.0120,+0.0517] |
| venue: DUBBO* | 9/3 | 1.6763→1.7011 | 0.7789→0.8030 | 0.333→0.333 | -0.0247 [-0.0882,+0.0387] | -0.0240 [-0.0718,+0.0237] |
| venue: GAWL* | 4/1 | 1.3773→1.3450 | 0.7183→0.7015 | 0.250→0.250 | +0.0322 [not estimable] | +0.0168 [not estimable] |
| venue: GEE* | 6/3 | 1.3318→1.4094 | 0.6733→0.7216 | 0.417→0.333 | -0.0776 [-0.2493,+0.0940] | -0.0483 [-0.1390,+0.0424] |
| venue: GOUL* | 4/1 | 1.2925→1.1918 | 0.6617→0.6349 | 0.250→0.250 | +0.1008 [not estimable] | +0.0268 [not estimable] |
| venue: GRDN* | 3/1 | 1.3487→1.4260 | 0.6577→0.6798 | 0.667→0.667 | -0.0773 [not estimable] | -0.0221 [not estimable] |
| venue: HEA* | 7/3 | 1.3715→1.3168 | 0.6717→0.6376 | 0.429→0.571 | +0.0547 [-0.0785,+0.1879] | +0.0341 [-0.0334,+0.1017] |
| venue: HOBT* | 8/2 | 1.4441→1.4693 | 0.6527→0.6535 | 0.625→0.500 | -0.0251 [-0.0351,-0.0151] | -0.0008 [-0.0144,+0.0127] |
| venue: HOR* | 6/2 | 1.2719→1.3043 | 0.6271→0.6544 | 0.417→0.333 | -0.0324 [-0.2475,+0.1827] | -0.0273 [-0.1533,+0.0988] |
| venue: MEA* | 2/2 | 1.7842→1.7878 | 0.7842→0.8063 | 0.500→0.500 | -0.0037 [-0.3286,+0.3212] | -0.0221 [-0.1480,+0.1039] |
| venue: MT_G* | 7/2 | 1.5856→1.5197 | 0.7946→0.7660 | 0.286→0.143 | +0.0659 [-0.0202,+0.1521] | +0.0286 [-0.0140,+0.0712] |
| venue: MURR* | 10/4 | 1.2812→1.2829 | 0.6676→0.6593 | 0.450→0.400 | -0.0017 [-0.0920,+0.0886] | +0.0084 [-0.0332,+0.0499] |
| venue: NOR* | 4/1 | 1.3736→1.3640 | 0.6676→0.6688 | 0.250→0.250 | +0.0095 [not estimable] | -0.0012 [not estimable] |
| venue: QOT* | 12/7 | 1.2973→1.2611 | 0.5828→0.5700 | 0.667→0.667 | +0.0362 [-0.0167,+0.0890] | +0.0129 [-0.0092,+0.0349] |
| venue: RICH* | 7/2 | 1.3768→1.3422 | 0.6587→0.6363 | 0.571→0.429 | +0.0347 [-0.0634,+0.1327] | +0.0224 [-0.0287,+0.0735] |
| venue: ROCK* | 1/1 | 0.5755→0.5700 | 0.2223→0.2187 | 1.000→1.000 | +0.0055 [not estimable] | +0.0036 [not estimable] |
| venue: SAL* | 8/3 | 1.4337→1.3877 | 0.6830→0.6617 | 0.250→0.375 | +0.0459 [-0.1808,+0.2727] | +0.0213 [-0.1038,+0.1464] |
| venue: TAREE* | 5/3 | 1.9245→2.0071 | 0.9428→0.9734 | 0.200→0.200 | -0.0826 [-0.3005,+0.1354] | -0.0307 [-0.1182,+0.0569] |
| venue: TEM* | 1/1 | 2.9658→3.0838 | 1.1961→1.2077 | 0.000→0.000 | -0.1179 [not estimable] | -0.0116 [not estimable] |
| venue: TRA* | 8/2 | 1.3088→1.2124 | 0.6364→0.5929 | 0.562→0.625 | +0.0965 [+0.0701,+0.1228] | +0.0436 [+0.0374,+0.0498] |
| venue: WAR* | 14/2 | 1.3514→1.3519 | 0.6327→0.6295 | 0.536→0.500 | -0.0005 [-0.0218,+0.0208] | +0.0032 [+0.0020,+0.0044] |
| venue: WPK* | 1/1 | 1.1120→0.9997 | 0.5681→0.5026 | 1.000→1.000 | +0.1123 [not estimable] | +0.0656 [not estimable] |
| venue: WRGL* | 3/2 | 1.0755→1.0405 | 0.5613→0.5464 | 0.667→0.667 | +0.0350 [-0.1859,+0.2559] | +0.0148 [-0.1162,+0.1458] |
| distance: <=400 | 126/15 | 1.3964→1.3823 | 0.6774→0.6708 | 0.464→0.460 | +0.0141 [-0.0255,+0.0537] | +0.0066 [-0.0131,+0.0262] |
| distance: >400 | 51/14 | 1.4535→1.4353 | 0.6967→0.6937 | 0.422→0.392 | +0.0182 [-0.0309,+0.0674] | +0.0030 [-0.0253,+0.0313] |
| grade: 2nd/3rd Grade* | 1/1 | 1.7766→1.9010 | 0.8233→0.8592 | 0.000→0.000 | -0.1244 [not estimable] | -0.0359 [not estimable] |
| grade: 3rd/4th Grade* | 2/2 | 1.1718→1.1450 | 0.6040→0.6019 | 1.000→0.500 | +0.0268 [-0.1026,+0.1562] | +0.0021 [-0.0317,+0.0359] |
| grade: 4th Grade* | 5/5 | 1.2278→1.1977 | 0.6176→0.6084 | 0.600→0.600 | +0.0301 [-0.0829,+0.1432] | +0.0091 [-0.0531,+0.0714] |
| grade: 4th/5th Grade* | 8/6 | 1.9415→1.8724 | 0.9311→0.9032 | 0.000→0.000 | +0.0691 [-0.0691,+0.2073] | +0.0279 [-0.0315,+0.0873] |
| grade: 5th Grade | 39/14 | 1.5157→1.4883 | 0.7247→0.7118 | 0.436→0.410 | +0.0274 [-0.0146,+0.0694] | +0.0129 [-0.0062,+0.0319] |
| grade: 5th/6th Grade* | 2/1 | 1.5546→1.5406 | 0.7694→0.7643 | 0.000→0.000 | +0.0140 [not estimable] | +0.0051 [not estimable] |
| grade: 6th Grade* | 8/7 | 1.0392→1.0896 | 0.5311→0.5474 | 0.625→0.625 | -0.0504 [-0.1482,+0.0473] | -0.0163 [-0.0637,+0.0311] |
| grade: Free For All* | 5/4 | 1.3600→1.4098 | 0.6837→0.7114 | 0.400→0.400 | -0.0499 [-0.2897,+0.1899] | -0.0277 [-0.1397,+0.0842] |
| grade: Grade 5 | 29/10 | 1.4395→1.4063 | 0.7010→0.6889 | 0.362→0.379 | +0.0332 [-0.0649,+0.1313] | +0.0121 [-0.0431,+0.0673] |
| grade: Grade 6* | 2/2 | 1.8255→1.7319 | 0.8510→0.7922 | 0.500→0.500 | +0.0936 [-0.1907,+0.3780] | +0.0588 [-0.0835,+0.2011] |
| grade: Grade 7* | 6/6 | 1.0911→1.1041 | 0.5004→0.5114 | 0.667→0.667 | -0.0130 [-0.1394,+0.1133] | -0.0110 [-0.0866,+0.0645] |
| grade: Group 2* | 1/1 | 1.1449→1.4132 | 0.6359→0.7929 | 0.000→0.000 | -0.2684 [not estimable] | -0.1571 [not estimable] |
| grade: Invitation* | 1/1 | 1.8344→1.8386 | 0.9162→0.9173 | 0.000→0.000 | -0.0042 [not estimable] | -0.0011 [not estimable] |
| grade: M2/M3* | 2/2 | 1.3205→1.3447 | 0.7110→0.7514 | 0.000→0.000 | -0.0242 [-0.4505,+0.4020] | -0.0403 [-0.3197,+0.2390] |
| grade: M3* | 3/3 | 1.3829→1.3707 | 0.7007→0.6894 | 0.667→0.667 | +0.0123 [-0.1979,+0.2224] | +0.0113 [-0.1060,+0.1286] |
| grade: M5* | 4/3 | 1.8201→1.7418 | 0.7787→0.7532 | 0.500→0.500 | +0.0783 [-0.2039,+0.3606] | +0.0255 [-0.0578,+0.1087] |
| grade: Maiden* | 16/7 | 1.3850→1.3760 | 0.6593→0.6599 | 0.500→0.438 | +0.0090 [-0.0300,+0.0480] | -0.0006 [-0.0192,+0.0179] |
| grade: Mixed* | 4/4 | 1.3180→1.2810 | 0.5950→0.5838 | 0.500→0.750 | +0.0370 [-0.1591,+0.2331] | +0.0112 [-0.0869,+0.1093] |
| grade: N/P* | 7/3 | 1.3481→1.3556 | 0.7083→0.7139 | 0.429→0.429 | -0.0075 [-0.1206,+0.1057] | -0.0056 [-0.0703,+0.0592] |
| grade: NG1-4* | 6/5 | 1.4001→1.3767 | 0.7165→0.7078 | 0.333→0.333 | +0.0234 [-0.2321,+0.2789] | +0.0087 [-0.1184,+0.1357] |
| grade: Open* | 3/3 | 0.7197→0.6762 | 0.3354→0.3106 | 1.000→1.000 | +0.0435 [-0.0432,+0.1302] | +0.0248 [-0.0157,+0.0652] |
| grade: Other* | 8/3 | 1.4040→1.3814 | 0.7094→0.7058 | 0.375→0.375 | +0.0227 [-0.0154,+0.0607] | +0.0036 [-0.0017,+0.0089] |
| grade: P5* | 3/2 | 0.9635→0.9221 | 0.5140→0.5039 | 0.667→0.667 | +0.0414 [-0.0788,+0.1616] | +0.0102 [-0.1323,+0.1526] |
| grade: PM* | 1/1 | 0.9887→1.0074 | 0.4688→0.4791 | 1.000→1.000 | -0.0187 [not estimable] | -0.0103 [not estimable] |
| grade: R/W* | 4/3 | 1.4724→1.4621 | 0.5850→0.5517 | 0.750→0.750 | +0.0103 [-0.1732,+0.1938] | +0.0333 [-0.0039,+0.0704] |
| grade: Restricted* | 6/4 | 1.4797→1.5384 | 0.6873→0.7084 | 0.583→0.500 | -0.0587 [-0.2687,+0.1514] | -0.0211 [-0.1183,+0.0760] |
| grade: Special Event* | 1/1 | 0.6558→0.6142 | 0.2777→0.2524 | 1.000→1.000 | +0.0416 [not estimable] | +0.0253 [not estimable] |

## Market movement and handoff

The independent [movement audit](market_movement_20260929_results.md) finds **331 races / 27 dates with one corrected WIN snapshot**, and an older surface intersecting **291 races / 26 dates** with two timing-compatible rows. **Zero races have two qualified corrected WIN snapshots**: exact source-row binding verifies only the later point, earlier rows lack runner identity and raw WIN classification evidence, and independent availability/scratch histories are absent. The older surface was already invalidated by the WIN/PLACE provenance audit. Timing alone does not repair it. No movement features, performance comparison, model fit or regularization choice was attempted.

The audit supplies the smallest retention specification: exact race/runner/market/source IDs; separate observed, available and scheduled-jump timestamps; atomic complete WIN fields with status and scratches; two pre-cutoff snapshots with retained raw classification evidence and margins. T−10/T−4 observations and a T−2 decision are a practical owner-reviewed cadence, not an authorization to change collection. Normalize inverse WIN odds separately at each observation and reject field changes; raw odds movement alone confounds bookmaker margin and scratches. The operational owner owns any future integration and access. No changes to production, frozen coefficients, prospective population, predictions, results or October study allocation are requested or made.

At most two follow-up hypotheses:

1. **Missingness-driven adjustment is a diagnostic risk.** Missing context indicators materially move forecasts and all-complete fields look better, but only 18 races qualify. Unlike #193's broad feature-removal search, first verify semantic completeness and exact indicator contributions on separately authorized future records; require enough dates/fields and a frozen earlier-data rule before any later performance claim. This task does not launch a missingness selection strategy or alter the October study.
2. **Normalized pre-cutoff movement adds information beyond the same-time market.** Require independently available paired WIN snapshots with unchanged active fields, verified scratches and enough chronological date blocks before fitting the one small experiment in the movement protocol. This differs from #193's static-market recipes and #195's unverifiable early-speed proxies. Current records cannot test it.

## Reproduction, trials and review

Use the new isolated worktree `/home/l4nd0/greyhound-market-explanation-20260929`, branch `research/market-explanation-20260929`, based on #193 commit `956d8289b8be062e9fe5ccdf554b9a9462391c20`. Original dirty worktrees and original research ledgers were preserved. Source hashes are [frozen inputs](market_explanation_20260929_inputs.json) and [all checked inputs](market_explanation_20260929_evidence_verified/input_hashes.json). The script fails closed on changed pins/membership and creates a fresh output directory; it does not overwrite existing results or refit.

```bash
cd /home/l4nd0/greyhound-market-explanation-20260929
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
"$RESEARCH_PY" -m unittest tests.test_explain_market_residual tests.test_audit_retained_market_movement -v
"$RESEARCH_PY" -m scripts.explain_market_residual --out /tmp/market-explanation-NEW
"$RESEARCH_PY" -m scripts.audit_retained_market_movement --output /tmp/market-movement-NEW
```

No original model training rerun is required. The local pinned original sources remain necessary for complete provenance replay; portable generated loss/contribution/receipt files support independent scoring/mechanism inspection. The [complete trial ledger](market_explanation_20260929_complete_trial_ledger.jsonl) includes preparation failures, all diagnostic runs and movement audit/test attempts. The original #193 1,000-event search ledger remains unchanged. Targeted synthetic tests exercise access-before-decode, complete rosters, tied-rank versus proper scores, exact cap/normalization decomposition, clustered weighting, movement timing eligibility. See the [review record](market_explanation_20260929_review.md) for independent source/spec validation. Draft review only; no promotion or dependable edge claim.

The first explanation output is preserved under `market_explanation_20260929_evidence/`; the final `_evidence_verified/` run suppresses unestimable one-date intervals after independent review. It changes no forecasts, populations, fits or point estimates. The earlier movement run pins the scope-helper source as it existed then; the later change only affects explanation uncertainty reporting.
