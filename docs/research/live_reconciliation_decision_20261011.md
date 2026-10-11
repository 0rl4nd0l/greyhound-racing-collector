# Retained live forecast benchmark — PR #231

**The installed July policy retained the market baseline; it shows no improvement over that same baseline. The genuinely sealed shadow forecasts remain inconclusive.** The fixed 65-attempt checkpoint supports 51 decision-time comparisons on three dates. Full-strength shadow log loss and Brier both worsen slightly, and winner accuracy is unchanged. No model, calibration, forecast or selector was fitted or replaced.

The fixed hold checkpoint contains **65 consumed attempts: 64 retained response bodies and one uncertain transport timeout**. The timeout occurred at 15:33:51 UTC on 11 October (02:33:51 AEDT on 12 October); no response body was retained and the request must not be retried. The collector still owns its process lock, but the source and July result lane are held. Existing authority covers the remaining 145 distinct unattempted identities; continuation requires the sole operational owner's reviewed hold reconciliation. See [execution status](live_result_acquisition_status_20261011.md).

| Disposition at the fixed 65-attempt checkpoint | Races |
|---|---:|
| Evaluable without repair | 0 |
| Recovered and fully evaluable | 51 |
| Retained response incomplete or incompatible | 13 |
| Requires a specific official-result acquisition | 146 |
| Restricted within this exact nonreserved population | 0 |
| **Total original forecasts** | **210** |

The 13 unqualified retained responses comprise 12 races with 33 unresolved terminal cells and one confirmed original-field scratch (Capalaba, July 19, race 5). Six incomplete races remain separate weak diagnostics. The 146 absent results comprise 145 never-attempted races and the consumed Sale race 8 timeout; they are different operational dispositions. All nine prior diagnostic races occur among the acquired races: three now qualify and six remain diagnostic. This is a complete census, not a performance-selected sample. The separate 65 reserved records and 169 withheld historical races remain excluded.

The retained-evidence investigation found nine exact joins in 261 observations across 65 July capture directories. Losing competition-rank ties explained 81 database rejections and recovered two joins. However, the old result writer copied forecast names instead of retaining independent official runner names. Earlier private closures did not repair that gap: 18 databases and 1,275 metadata files yielded 44 closure identities and zero overlap with this population. Absence in those named archives is not proof that a race was never captured anywhere. [Original reconciliation and exact evidence](live_result_reconciliation_20261011.md).

The offline repairs preserve every original rejection. They correct a false challenge classification on the first response, accept exact source titles witnessed by retained pre-race identities, and extract dog names from their unique official profile anchors. The last repair separates an `NBT` timing badge from the dog name; it does not strip arbitrary suffixes or force a fuzzy match. The unchanged title policy rejects the unsupported sponsored display `Ladbrokes Gardens`. A separately reviewed source-structure adapter recovers those four pages through unique active race, RESULTS-tab and meeting links matching the exact approved routes, plus the title date/race. It adds no venue alias; missing, duplicate, malformed or contradictory declarations fail. Its criteria were frozen before corrected scoring. This is an after-inspection parser repair, not prospective preregistration. Unknown `F`, `T`, `BD` and bare `-` cells remain unmapped. Full losing finishing order is unnecessary for WIN, but independent runner identity and explicit disposition are needed to distinguish starters from scratches and substitutions.

An additional [pre-race field repair](live_pre_race_field_clarification_20261011.md) recovers eight races using exact sealed evidence of inactive runners and reserve mappings. It preserves every source row and requires source annotations for reserve moves. HEA July 19 race 11 gains identity continuity but remains terminal-incomplete. Capalaba race 5 retains its genuine original-runner scratch and stays excluded; no probabilities are renormalized. Criteria were frozen before corrected scoring, following inspection of parser failures.

A separate [non-trial navigation repair](live_nontrial_navigation_clarification_20261011.md) recovers Rockhampton race 4 through its exact pre-race URL and source-declared `trial=false` race/RESULTS links. It requires matching title/date/race, active meeting and a unique result table; no sponsored venue alias or arbitrary query normalization is introduced. The original refusal remains retained.

The qualified population contains **51 races, 357 runners, July 17–19**, with one sealed model artifact `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d` and its full/half shadow variants. The native production policy was `KEEP_MARKET_BASELINE`. Its retained baseline equals the contemporaneous normalized WIN market, so its paired differences are zero by construction. Shadow forecasts below were actually sealed before jump; they are not offline reconstructions or production replacements.

| Same 51 races and original fields | Log loss | Brier | Top accuracy |
|---|---:|---:|---:|
| Market / installed policy baseline | 1.433382 | 0.693949 | 21/51 |
| Full-strength shadow | 1.436326 | 0.699424 | 21/51 |
| Half-strength shadow | 1.433032 | 0.695617 | 21/51 |
| Full minus market | +0.002944 | +0.005475 | 0 |
| Half minus market | −0.000350 | +0.001668 | 0 |

Lower probability losses are better. Market probabilities use each reciprocal decimal WIN quote divided by the sum of reciprocal quotes over the complete original field. No starting prices, later quotes or outcome-reconstructed fields are substituted. Original odds and overrounds are retained; overround ranges from 1.214493 to 1.404335. Prediction lead time is 15.19–56.20 minutes (median 51.09); capture age is 123.97–353.71 seconds (median 154.18). Provider quote-publication age is unknown.

All 51 top selections are unchanged, including 30 losses. Winner probability improves on 26 races and worsens on 25.

| Date | Races | Full minus market log loss | Full minus market Brier |
|---|---:|---:|---:|
| July 17 | 2 | −0.025711 | −0.018357 |
| July 18 | 22 | −0.013235 | +0.000826 |
| July 19 | 27 | +0.018249 | +0.011029 |

Dropping July 19 reverses the pooled log-loss direction; improvement is not consistent across dates or metrics. The frozen date bootstrap reports descriptive log-loss endpoints [−0.025711, +0.018249] and Brier [−0.018357, +0.011029]. **Three dates and repeated dogs do not support a reliable confidence claim.** No selected race or date was removed. Runner-weighted calibration ECE is 0.032916 full, 0.029181 half and 0.014437 market; sparse bins are descriptive, not established reliability.

Six weak diagnostics (48 runners, July 17–18) stay separate: full-shadow log-loss difference +0.031265 and Brier +0.023284, with 4/6 top accuracy for both. These do not increase strict comparison coverage. Prior one-, two-, three-, four-, eleven-, fifteen-, forty- and fifty-race snapshots and their original misclassifications remain retained as history.

The selectors remain byte-identical to PR #231: 79 earlier development forecasts on July 17–19 define thresholds applied to 131 later forecasts on July 20–22. High winning probability, trustworthy probability estimates and attractive prices are different questions. History is an evidence-quality proxy; `p × odds − 1` measures price disagreement, not a demonstrated edge. These analyses remain exploratory under the original protocol.

| Selector: selected races out of 131 | Nominal 10% | 25% | 50% | Full |
|---|---:|---:|---:|---:|
| Model confidence | 9 | 19 | 42 | 131 |
| Relevant history | 21 | 35 | 57 | 131 |
| Price disagreement | 8 | 32 | 59 | 131 |

Every selected group has a market-confidence control at exactly matched coverage and a same-race model/market comparison. **All groups still have zero evaluable later outcomes**, so their missing counts equal their selected counts. All passes remain recorded; no thresholds or membership were changed after new results. All 131 later evaluation races are within the 145 unattempted identities; the other 14 are earlier development races. Outcome availability is concentrated on the first three development dates. The original all-210 unchanged-field bounds are now explicitly inapplicable because of the confirmed Capalaba scratch. Keeping all 210 dispositions, a separate conditional analysis of the remaining 209 races uses 51 known and 158 unresolved outcomes. Its full-shadow log-loss-difference bounds are [−0.100515, +0.133002] and Brier bounds [−0.038628, +0.041538]. These are identification bounds, not confidence intervals: they condition on eventual proof of unchanged original fields and a single winner in unresolved races. The unresolved operational races are unrelated to the separately withheld historical population. Missing fields are not thereby qualified; no selector or control was reranked after exclusion.

Forecast coverage is **210/431 = 48.72% of recorded attempted races**, from 1,292 attempts. There are 212 appended forecast records on 210 unique races and 1,080 blocked attempts; 425 blocked attempts belong to races that also produced a forecast. Thus 221 unique races produced no forecast:

| Exclusive cause | Unique races | Interpretation |
|---|---:|---|
| Equivalent grade spellings rejected | 126 | Largest demonstrated software defect |
| Hyphenated venue grammar rejected | 49 | Parser limitation; downstream success not established |
| Unsafe or incomplete metadata | 45 | Valid exclusion: 34 mapping gaps, 11 also decode failures |
| Admission after jump | 1 | Valid late exclusion |

The offline grade repair reproduces 369 rejected attempts. Of these, 316 attempts on 122 races also pass retained input and chronology checks after the finite-vocabulary correction. This is an input-readiness ceiling, not newly produced live forecasts. Tests preserve genuine grade disagreements, ambiguous metadata, hash failures and isolation between races. Earlier installed source already contained grade and venue fixes; the audit does not propose redundant deployment. [Failure accounting and repair validation](live_forecast_failure_diagnosis_20261011.md).

The 51 qualified races represent **24.29% of retained forecasts and 11.83% of attempted races**. Pre-attempt downtime and unrecorded opportunities cannot be inferred from that denominator. Unavailable forecasts are not zero-probability predictions. No profitability claim is supported.

The highest-value next action is **finish the already approved exact-population result reconciliation**, while preserving the actual source hold and consumed timeout. There are 145 unattempted identities within the existing allowance. The sole operational owner must reconcile the hold before any permitted continuation. The Sale race 8 request remains consumed; any fresh request for that identity requires an explicit additional exception, not a reset of its accounting. Retained incomplete responses remain separate repair investigations. No new model search, calibration fit or selector tuning is warranted.

The current machine-readable dataset, all 210 dispositions, original odds, corrections, scorecards, calibration bins, date sensitivity, every selector/pass and conditional bounds are under `acquisition/observed-continuation-06/corrected-v7/` in `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence`. Exact exclusions and references are in `integration/integration-dispositions.json`; strict runner data are in `completed/strict-dataset.json`. The [hash index](live_reconciliation_evidence_20261011.json) also preserves earlier snapshots, authority, consumed receipts, source/title/name policies and review artifacts.

**416 focused tests pass**, with the tested implementation hashes unchanged. Independent review checks membership, identity, field alignment, chronology, result joins and calculations; a separate root execution reproduces all version metrics and byte-identical frozen selector outputs. Tests reject malformed profile links, misleading name badges, genuine name changes, unknown terminal codes, changed policy hashes and unrelated-race failure propagation. Reproduce this fixed snapshot into a new directory:

```bash
bash /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/acquisition/observed-continuation-06/corrected-v7/reproduce.sh /tmp/july-benchmark-checkpoint06
```

The wrapper verifies the command manifests and executes integration, the frozen scorer, the independent checker and the separate 209-race sensitivity utility. Original all-210 bounds remain retained but explicitly inapplicable. Independent arithmetic also checks every possible unresolved winner and all 24 selector/control groups. Raw evidence remains local; no provider requests or runtime changes are made by reproduction.
