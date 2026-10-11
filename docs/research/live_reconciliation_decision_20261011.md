# Retained live forecast benchmark — PR #231

**The installed July policy retained the market baseline; it shows no improvement over that same baseline. The genuinely sealed shadow forecasts remain inconclusive.** The fixed 23-response checkpoint supports 15 decision-time comparisons on two dates. Full-strength shadow log loss and Brier both worsen slightly, and winner accuracy is unchanged. No model, calibration, forecast or selector was fitted or replaced.

The snapshot was frozen at 13:01:04 UTC on 11 October (00:01 AEDT on 12 October). Acquisition is incomplete: the scheduled handoff stopped the collector and failed after 23 consumed requests. Its consumed intent, original receipts and source history remain preserved; the sole operational owner is reconciling the failure. See [execution status](live_result_acquisition_status_20261011.md).

| Current disposition of the original population | Races |
|---|---:|
| Evaluable without repair | 0 |
| Recovered and fully evaluable | 15 |
| Retained response remains incomplete or unqualified | 8 |
| Requires the approved specific official-result acquisition | 187 |
| Restricted within this exact nonreserved population | 0 |
| **Total original forecasts** | **210** |

The eight unqualified responses comprise six prior weak diagnostics and two newly acquired records, all with unresolved runner dispositions. All nine prior diagnostic races occur among the 23 acquired races: three now qualify and six remain diagnostic. This is a complete census, not a performance-selected sample. The separate 65 reserved records and 169 withheld historical races remain excluded.

The retained-evidence investigation found nine exact joins in 261 observations across 65 July capture directories. Losing competition-rank ties explained 81 database rejections and recovered two joins. However, the old result writer copied forecast names instead of retaining independent official runner names. Earlier private closures did not repair that gap: 18 databases and 1,275 metadata files yielded 44 closure identities and zero overlap with this population. Absence in those named archives is not proof that a race was never captured anywhere. [Original reconciliation and exact evidence](live_result_reconciliation_20261011.md).

The offline repairs preserve every original rejection. They correct a false challenge classification on the first response, accept exact source titles witnessed by retained pre-race identities, and extract dog names from their unique official profile anchors. The last repair separates an `NBT` timing badge from the dog name; it does not strip arbitrary suffixes or force a fuzzy match. The unchanged title policy rejects the unsupported sponsored display `Ladbrokes Gardens`. A separately reviewed source-structure adapter recovers those four pages through unique active race, RESULTS-tab and meeting links matching the exact approved routes, plus the title date/race. It adds no venue alias; missing, duplicate, malformed or contradictory declarations fail. Its criteria were frozen before corrected scoring. This is an after-inspection parser repair, not prospective preregistration. Unknown `F`, `T`, `BD` and bare `-` cells remain unmapped. Full losing finishing order is unnecessary for WIN, but independent runner identity and explicit disposition are needed to distinguish starters from scratches and substitutions. Original fields, unused reserves, corrections and failures remain recorded; no confirmed changed fields were found in this checkpoint.

The qualified population contains **15 races, 113 runners, July 17–18**, with one sealed model artifact `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d` and its full/half shadow variants. The native production policy was `KEEP_MARKET_BASELINE`. Its retained baseline equals the contemporaneous normalized WIN market, so its paired differences are zero by construction. Shadow forecasts below were actually sealed before jump; they are not offline reconstructions or production replacements.

| Same 15 races and original fields | Log loss | Brier | Top accuracy |
|---|---:|---:|---:|
| Market / installed policy baseline | 1.584058 | 0.738586 | 5/15 |
| Full-strength shadow | 1.586805 | 0.746333 | 5/15 |
| Half-strength shadow | 1.583802 | 0.741640 | 5/15 |
| Full minus market | +0.002747 | +0.007747 | 0 |
| Half minus market | −0.000256 | +0.003053 | 0 |

Lower probability losses are better. Market probabilities use each reciprocal decimal WIN quote divided by the sum of reciprocal quotes over the complete original field. No starting prices, later quotes or outcome-reconstructed fields are substituted. Original odds and overrounds are retained; overround ranges from 1.288470 to 1.349919. Prediction lead time is 23.62–52.18 minutes (median 42.28); capture age is 127.46–353.71 seconds (median 200.17). Provider quote-publication age is unknown.

All 15 top selections are unchanged, including ten losses. Winner probability improves on seven races and worsens on eight. July 17 has two races: full-shadow differences are −0.025711 log loss and −0.018357 Brier. July 18 has thirteen: +0.007125 and +0.011763. Dropping one date versus the other reverses both loss directions. The frozen date bootstrap reports descriptive log-loss endpoints [−0.025711, +0.007125] and Brier [−0.018357, +0.011763]. **Two dates and repeated dogs do not support a reliable confidence claim.** No selected race or date was removed. Runner-weighted calibration ECE is 0.068725 full, 0.049354 half and 0.029065 market; sparse bins are descriptive, not established reliability.

Six weak diagnostics (48 runners, the same two dates) stay separate: full-shadow log-loss difference +0.031265 and Brier +0.023284, with 4/6 top accuracy for both. These do not increase strict comparison coverage. Prior one-, two-, three-, four- and eleven-race snapshots and their original misclassifications remain retained as history.

The selectors remain byte-identical to PR #231: 79 earlier development forecasts on July 17–19 define thresholds applied to 131 later forecasts on July 20–22. High winning probability, trustworthy probability estimates and attractive prices are different questions. History is an evidence-quality proxy; `p × odds − 1` measures price disagreement, not a demonstrated edge. These analyses remain exploratory under the original protocol.

| Selector: selected races out of 131 | Nominal 10% | 25% | 50% | Full |
|---|---:|---:|---:|---:|
| Model confidence | 9 | 19 | 42 | 131 |
| Relevant history | 21 | 35 | 57 | 131 |
| Price disagreement | 8 | 32 | 59 | 131 |

Every selected group has a market-confidence control at exactly matched coverage and a same-race model/market comparison. **All groups still have zero evaluable later outcomes**, so their missing counts equal their selected counts. All passes remain recorded; no thresholds or membership were changed after new results. Outcome availability is concentrated on the first two dates. Conditional full-population log-loss-difference bounds are [−0.124286, +0.169475], and Brier bounds [−0.050428, +0.052824], using the 15 known outcomes. These are identification bounds, not confidence intervals: they condition on eventual proof of unchanged original fields and a single winner for every race. Missing fields are not thereby qualified.

Forecast coverage is **210/431 = 48.72% of recorded attempted races**, from 1,292 attempts. There are 212 appended forecast records on 210 unique races and 1,080 blocked attempts; 425 blocked attempts belong to races that also produced a forecast. Thus 221 unique races produced no forecast:

| Exclusive cause | Unique races | Interpretation |
|---|---:|---|
| Equivalent grade spellings rejected | 126 | Largest demonstrated software defect |
| Hyphenated venue grammar rejected | 49 | Parser limitation; downstream success not established |
| Unsafe or incomplete metadata | 45 | Valid exclusion: 34 mapping gaps, 11 also decode failures |
| Admission after jump | 1 | Valid late exclusion |

The offline grade repair reproduces 369 rejected attempts. Of these, 316 attempts on 122 races also pass retained input and chronology checks after the finite-vocabulary correction. This is an input-readiness ceiling, not newly produced live forecasts. Tests preserve genuine grade disagreements, ambiguous metadata, hash failures and isolation between races. Earlier installed source already contained grade and venue fixes; the audit does not propose redundant deployment. [Failure accounting and repair validation](live_forecast_failure_diagnosis_20261011.md).

The 15 qualified races represent **7.14% of retained forecasts and 3.48% of attempted races**. Pre-attempt downtime and unrecorded opportunities cannot be inferred from that denominator. Unavailable forecasts are not zero-probability predictions. No profitability claim is supported.

The highest-value next action is **finish the already approved exact-population result reconciliation**, including recovery of the interrupted handoff by the sole operational owner. There are 187 unattempted identities within the existing finite allowance; 23 requests are consumed and must not be repeated. Retained incomplete responses remain separate repair investigations. No new model search, calibration fit or selector tuning is warranted.

The current machine-readable dataset, all 210 dispositions, original odds, corrections, scorecards, calibration bins, date sensitivity, every selector/pass and conditional bounds are under `acquisition/observed-continuation-04/corrected-v5/` in `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence`. Exact exclusions and references are in `integration/integration-dispositions.json`; strict runner data are in `completed/strict-dataset.json`. The [hash index](live_reconciliation_evidence_20261011.json) also preserves earlier snapshots, authority, consumed receipts, source/title/name policies and review artifacts.

**281 focused tests pass**, with the tested implementation hashes unchanged. Independent review checks membership, identity, field alignment, chronology, result joins and calculations; a separate root execution reproduces all version metrics and byte-identical frozen selector outputs. Tests reject malformed profile links, misleading name badges, genuine name changes, unknown terminal codes, changed policy hashes and unrelated-race failure propagation. Reproduce this fixed snapshot into a new directory:

```bash
bash /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/acquisition/observed-continuation-04/corrected-v5/reproduce.sh /tmp/july-benchmark-checkpoint04
```

The wrapper verifies the command manifest and executes integration, the frozen scorer and the independent checker. Raw evidence remains local; no provider requests or runtime changes are made by reproduction.
