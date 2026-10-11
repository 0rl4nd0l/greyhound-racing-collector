# Independent continuation review

This review uses the exact PR #231 population and retained allocation boundary. It reads only admitted forecast bodies and exact allowlisted results, without provider access, model fitting or live changes. Numerical verification imports the independent arithmetic checker; it never imports the benchmark scorer.

## Population, chronology and frozen selections

The initial independent replay authenticates all 277 original journal line hashes while decoding only the 212 authorised records. Deterministic primary selection gives 210 races. All original full-strength probabilities, decimal WIN odds, runner identities, model hashes and forecast timestamps agree with the original journal. Capture precedes calculation, durable append precedes jump, and the two repeated horizons remain outside primary selection.

Independent attempt-row grouping confirms 1,292 nonreserved attempts: 212 appended records and 1,080 blocked attempts. These represent 431 distinct races: 210 with at least one append and 221 without. The unsuccessful races divide exclusively into 126 target-grade failures, 49 invalid-venue failures, 45 unsafe-feature-metadata failures and one timestamp-order failure. A blocked duplicate after a successful forecast does not create another unsuccessful race.

The continuation checker requires exact equality of the original forecast population, frozen selector plan and every selection/pass disposition. It independently reconstructs the 79-race July 17–19 development split and 131-race July 20–22 evaluation split, all 12 selector comparisons and matched market-confidence race sets. Newly available outcomes cannot change any selector threshold, replace an unavailable selected race or erase a pass.

## Necessary qualification and a source-identity defect

WIN log loss, Brier score and top-selection accuracy require a confirmed winner and unchanged original field, not a unique finite finishing rank for every losing starter. Explicit DNF/FELL/DISQ can establish a losing starter without inventing a placing. An absent runner with unknown terminal status remains unresolved because scratch/substitution ambiguity changes the market contract.

Healesville July 17 R7 and R8 retain eight finishing boxes with a single winning box. Their lower placings respectively contain ties for seventh and fifth. Rejecting these solely because every placing must equal its enumeration is an overly strict WIN requirement. The original protocol's phrase “confirmed dead heats” is ambiguous in isolation. The continuation must explicitly clarify lower-place ties before scoring, preserve the original protocol bytes and continue excluding winning ties from the strict single-winner scorecard.

However, matching the native result JSONL dog names against the forecast is insufficient authentication. `scripts/ingest_results_for_date.py:write_result` builds both winner name and every result runner name from `candidate.participants`. Its successful summary discards `SourceResult.dog_names_by_box`; `result_validation_error` checks box membership and competition ranking, without matching official names. The apparently unchanged names can therefore be circular. All eight finishing boxes and scratched reserves 9/10 do not independently establish the unchanged source runner identities. The lower-place tie repair must not silently promote either race without an independent retained official field witness. This is a separate limitation from missing finishing positions and applies to the older diagnostic joins too.

## Offline forecast-failure repair

The independent checker verifies all referenced source hashes and re-derives every attempt and race disposition. The grade correction uses the existing finite vocabulary and modifies only a derived context comparison spelling; original feature bytes and the complete native feature validator remain unchanged. All 369 grade-failure attempts across 126 unique races reproduce their original rejection and pass the repaired feature validation. Only 316 recorded attempts across 122 unique races have valid feature-to-score pre-jump chronology. These are validation recoveries, not newly recovered live forecasts or a measured coverage increase. The historical 210/431 forecast coverage remains unchanged. Six targeted repair tests pass, including rejection isolation and preservation of the metadata safety gate.

## Completed independent checks

The final integrated independent checker passes. Its exact evidence is `greyhound-live-reconciliation-20261011-evidence/review/integrated-independent-check.json`. It verifies the complete 210-record reconciliation, all 2,251 allowlisted native JSONL rows against their original line bodies, retained archive hashes and every projected result column. Current dispositions are zero evaluable without repair, zero fully evaluable after repair, nine still incomplete, 201 requiring acquisition and zero restricted. The two recovered archive joins are included within the nine incomplete records, not added as another population.

A producer rerun initially appended the same two correction events again because tuple/list serialization broke equality. Review detected the four-event/two-recovery discrepancy. The correction now uses canonical JSON event identity, preserves the exact four-entry erroneous trail in a hash-bound history file, annotates the two duplicate appends and publishes the two unique current events. The independent checker verifies that deduplication neither edits nor loses either real correction.

All nine diagnostic races retain both original full/half variants, producing 18 race/model records, not 18 races. The complete original eight-runner probability fields remain unchanged. Their labels are official winning boxes mapped to the original forecast roster; independent official dog identity is not established. They cannot enter the strict benchmark.

| Diagnostic on nine races | Original full-strength | Contemporaneous market | Paired difference |
|---|---:|---:|---:|
| Log loss | 1.409208930630 | 1.406092796089 | +0.003116134541 |
| Brier | 0.642702128639 | 0.633952497530 | +0.008749631108 |
| Top-selection credit | 5/9 | 5/9 | 0 |

Independent arithmetic reproduces per-race scores, market normalization and overround, calibration bins/ECE, date tables, leave-one-date-out results and seeded date-cluster intervals. With only two dates, the log-loss interval [−0.070410872779,+0.024123850918] is descriptive. The diagnostic does not establish model advantage. The strict population remains zero after actual archive recovery and qualification review.

The frozen plan SHA-256 remains `775ff69107a93cbabd803f5d24d614fe59d8468f2372493cae58c37329f5ffee`; original selection/pass artifact SHA-256 remains `f77316e5fad866d741137a5d230fa91d20ff7012ebe9f8bebd36fe06bb06fdbd`. Both newly joined races belong to earlier development dates. Every one of the 131 later forecasts still lacks a result join, so all frozen selected-group performance estimates remain unavailable. The 79/131 split, thresholds, selected identities, matched market-confidence controls and every pass are unchanged.

Independent brute-force enumeration of every possible winning original runner reproduces the missing-outcome bounds for all 210 forecasts and all 24 selected/control groups in both strict and diagnostic analyses. These are conditional identification bounds, not confidence intervals: they assume unchanged fields and, when diagnostic labels are used, correctness of the winning-box-to-original-runner mappings. They do not qualify missing source identities or eliminate missing-outcome selection bias.

Five continuation-review tests pass, including a new complete-box/circular-name case that is rejected from strict scoring while remaining explicitly diagnostic. Combined focused review, arithmetic and failure-repair checks pass 35 tests plus four subtests. Live services, fitted artifacts, original forecasts, scientific allocations and provider request accounting remain unchanged by these offline checks.

```bash
python3 scripts/check_live_reconciliation.py \
  --baseline-root /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --completed /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/completed \
  --failure-root /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/failures/v2 \
  --result-root /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results \
  --output /tmp/live-reconciliation-independent-check.json
```
