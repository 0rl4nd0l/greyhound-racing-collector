# Independent benchmark review, 11 October 2026

The strict decision-time scorecard has **zero eligible races**. The seven retained winner joins are useful diagnostics, but every original forecast has eight runners and every result projection has fewer than eight. Missing result rows cannot be treated as confirmed nonfinishers or retrospectively removed. These diagnostics do not establish whether the installed system improved on its contemporaneous market.

## Population, identities and provenance

An independent pass verified the original journal SHA-256 and all 277 line hashes, the 212 admitted / 65 scientifically excluded record partition, all six protection-reference hashes and inclusion of every original closed-study identity. Only admitted prediction lines were JSON-decoded; protected outcomes were not accessed. The 212 admitted records cover 210 distinct races on six dates. Exact conservative venue aliases were used for exclusion membership only, not for fuzzy result joins.

The independent source check compares each integrated diagnostic row with the admitted original line: race and prediction IDs, exact model hash and full/half variant, every runner ID and box, unchanged probabilities and original WIN odds, original score and jump times. It verifies the retained artifact hashes and official result subset names, boxes and unique winner. No original probability is recalculated by model replay or renormalized after an outcome.

A separate outcome-free sample of July 19 Healesville race 4 confirmed all six input hashes, complete seven-runner WIN capture and exact original odds. The native append implementation flushes and fsyncs before returning, and the daemon's successful score completion precedes jump. This led to an important review correction: a missing artifact named “seal” alone is not sufficient to dismiss equivalent original durable-append evidence. The alignment adapter binds the original journal, daemon prediction and saved command output rather than creating a new live forecast.

For the seven result joins, independent review found all eight original runners remained in the forecasts. The three seven-position results therefore did not establish complete field closure. The earlier four-position and later seven-position Traralgon record may be reconciled only when the official source and overlapping identities/order/winner agree; both originals remain referenced. Result-row `captured_at` is a batch-generation timestamp, so retained per-race attempt completion supplies the more accurate closure bound when available.

The residual model bytes match the currently installed artifact, but their July historical role was **live shadow**, while the production decision remained `KEEP_MARKET_BASELINE`. Model identity does not establish that the residual forecasts were production-selected then.

## Independently reproduced arithmetic

`scripts/check_live_benchmark_independently.py` imports neither benchmark nor selector code. It independently computes normalized inverse-odds probabilities, original model losses, summed multiclass Brier, tied-top credit, paired differences, ten-bin calibration, date summaries, leave-one-date-out sensitivity and date-cluster bootstrap intervals. The checker also validates the strict/diagnostic classification, membership and protocol/data hashes. Zero-probability winners produce infinite log loss; no outcome-based floor is introduced.

On the seven diagnostic races (56 original runners, July 17–18), the full residual has log loss **1.413074774549** versus market **1.401722214476**, difference **+0.011352560072**. Its Brier score is **0.635905573695** versus **0.619411167083**, difference **+0.016494406612**. The half residual has log-loss difference **+0.004096197097** and Brier difference **+0.007662280350**. Both model variants and market have top-selection accuracy **4/7**. Positive loss differences favour market. These are arithmetic statements about the explicitly incomplete-field diagnostic population, not evidence of production superiority or inferiority.

## Selector review

The primary selector population remains the 210 original nonreserved forecasts. Its first three dates contain 79 development races; the later three dates contain 131 evaluation forecasts. Independent reconstruction verified all 12 fixed rule/coverage combinations, thresholds, selected IDs, exactly matched market-confidence IDs and selected/pass dispositions. It also verified all 210 original forecast probability/odds fields and the hash-bound minimum same-distance-start values used to operationalize relevant history.

The primary strict and diagnostic selector evaluations apply those full-population cohorts unchanged. Neither has an authorised retained outcome join among the 131 later forecasts, so every primary selector performance comparison is empty and its missing-outcome denominator is retained. The seven-result diagnostic subset uses a different five-development/two-later split; its 24 full/half comparisons were independently recomputed, but they are explicitly **result-availability-conditioned exploratory diagnostics**, not the primary selector test. No selective prediction improvement is established.

## Operational denominator

Independent regrouping of 1,602 attempt records reproduces 548 distinct attempted races, of which 431 are nonreserved. Exactly 210 nonreserved races have at least one appended forecast, matching the admitted journal identities; 221 have none. The 1,080 failed nonreserved attempts reconcile by explicit failure code. This supports **210/431 = 48.72%** original-forecast availability among retained attempted races. It does not account for races with no captured attempt, so it is not a calendar-wide coverage claim. Diagnostic result joins cover **7/210 = 3.33%** of forecasted nonreserved races and **7/431 = 1.62%** of the retained attempted population; strict evaluable coverage is zero.

## Validation and reproducibility

Meaningful failure tests reject duplicate runner/box identities, nonfinite or unnormalized probabilities, invalid odds, ambiguous winners, allocation violations, replay masquerading as a live prediction, changed fields, naive or late timestamps, and corrupt accounting. Partial result fields require explicit diagnostic mode; even diagnostic mode rejects changed fields. **24 focused tests passed**; they run without importing the production application:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python -m pytest -q -o addopts= --noconftest tests/test_check_live_benchmark_independently.py
```

Review evidence is retained under `/mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/review/`: independent membership, source sample, coverage and strict/diagnostic metric reports. Each metric report binds the dataset, frozen protocol and source scorecard hashes. The final reproduction command is recorded alongside the main benchmark report.

The appropriate conclusion is **inconclusive for the installed model at decision time**. The current narrow diagnostic point estimates favour market. Neither the seven-race sample, its two dates, nor the missing outcome-field closure supports selector advancement, model promotion or profitability claims.
