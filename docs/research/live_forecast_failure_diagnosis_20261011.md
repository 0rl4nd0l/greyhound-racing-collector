# Retained live forecast failure diagnosis — 11 October 2026

The largest demonstrated repairable cause was **grade spelling disagreement at the validator boundary: 126 unique races**, not a model rejection. Retained features say `Grade 5`, for example, while the same pre-race source says `5th Grade`. The July scorer compared their raw tokens. A narrowly scoped offline adapter proves equality using the existing finite grade vocabulary, then reruns the complete native feature validator. It changes no feature bytes, fitted model, original forecast, live process, or historical attempt.

**122 of those races had at least one retained attempt that also passes form binding, exact identity, complete contemporaneous market alignment and source/feature/score chronology.** This is an input-readiness ceiling for a historical counterfactual, not 122 recovered forecasts or a promised increase in throughput. Four grade-rejected races had no timely attempt. No prediction was generated during this diagnosis.

## Counting units

| Unit | Count |
|---|---:|
| Nonreserved unique races with native scoring attempts | 431 |
| Native scoring-attempt records for those races | 1,292 |
| APPENDED attempt records | 212 |
| BLOCKED attempt records | 1,080 |
| Unique races with a retained forecast | 210 |
| Unique races without any appended forecast | 221 |
| Conditional forecast coverage | 48.7239% |

The two extra APPENDED records are additional retained horizons on existing races. Feature subprocesses, scorer invocations, race identities and model outputs are separate units. The original 431−210=221 subtraction is valid at the unique-race level. A race with a successful forecast remains successful even when later attempts failed. In particular, all 346 duplicate-conflict failures and 79 of 80 timestamp failures belong to already-forecast races; they are not additional unsuccessful races.

The denominator is the retained early-residual attempt population, not every calendar race, a promise to forecast every discovered race, or a complete census of pre-attempt downtime. Within the 221 failures there are no unexplained terminal causes: each race has one consistent recorded blocker across its attempts. This does not establish absence of upstream downtime outside the attempt population.

## Complete unsuccessful-race accounting and repair priority

| Exclusive first recorded blocker | Unique races | Attempts | Finding and next step |
|---|---:|---:|---|
| `feature_row_target_grade_mismatch` | 126 | 369 | Proven spelling defect; all 369 native failures reproduce, all 369 pass the complete native feature validator after finite-vocabulary comparison correction. 122 races / 316 attempts also pass remaining retained input and timing checks. Highest verified repair benefit. |
| `target_venue_invalid` | 49 | 145 | Native sidecar grammar rejects exact hyphenated venue labels present in both the sidecar and feature rows. Current repository code already has a wider venue grammar. No venue alias collapse or retroactive feature changes were performed. Downstream eligibility after that fix is not claimed. |
| `feature_row_metadata_not_safe` | 45 | 140 | The safety rejection is valid for the retained incomplete weather/track evidence. All 45 show missing venue/weather mapping; 11 also have at least one source JSON decoding failure. Repair collection/configuration and retain safe evidence before changing eligibility. Do not set the safety flag to true. |
| `source_timestamp_order_invalid` | 1 | 1 | Ballarat R9, 18 July: jump 19:40; features generated 19:44:48.705692; scoring finished 19:44:49.087290 (+10:00). Valid late-admission exclusion, not a recoverable live forecast. |
| **Total** | **221** | **655** | The other 425 blocked attempts belong to successful races. |

The four grade spellings are `5th Grade`→`Grade 5` (243 attempts), `6th Grade`→`Grade 6` (65), `Restricted`→`Restricted Win` (40), and `4th Grade`→`Grade 4` (21). These are existing vocabulary equivalences, not inferred grades or new training semantics. Ambiguous `M`, unknown values, genuinely different grades and missing grades remain blocked.

Venue rejections affect Lakeside 21, Q Straight 11, Parklands 10 and Murray Bridge Straight 7. These course identities remain distinct in the native evidence. Metadata rejections affect Goulburn 8, Meadows 10, Maitland 9, Launceston 8 and Wagga 10. Their safety flags are preserved. The four grade-defect races with no timely retained attempt are 18 July Wentworth Park R3/R4, Gardens R4 and Gunnedah R9.

## Repair implementation and evidence

`diagnose_retained_forecast_failures.py` reads the access boundary, membership and coverage manifests before opening race artifacts. It rejects protected keys and windows, verifies the original stage/stderr hashes, checks that the 210 APPENDED race identities equal the authorised forecast membership, and preserves every original attempt. It opens only the allowlisted unsuccessful races' pre-race feature and market artifacts; it never opens outcome records.

The score command recorded in the native stage receipts names:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-early-residual-shadow-activation-v1-20260716/scripts/predict_market_form_residual.py`

This source file is unmodified in that retained checkout; its last source change is commit `21472cba` on 16 July. The reproduction uses that module, not the current repository's changed validator. Each original feature implementation manifest's declared source hashes passes the native source-identity checks; every file actually read is hashed in the output references. The finite grade vocabulary comes from PR #231's existing `scripts/predict_market_form_residual.py`.

The adapter copies only the in-memory comparison context, verifies every runner's grade maps to the same known canonical grade as the source, and invokes the original full feature validator with unchanged sealed bytes and manifests. The original hash, provenance, race, roster, distance, history-cutoff and no-outcome checks all run. Form binding, complete market runner alignment and original recorded chronology are separately verified. Original sidecars, features, failed attempts and consumed attempts remain untouched. Read-only inspection of the current installed source checkout `/mnt/tenn-nvme2/tenn/greyhound-reserve-exclusion-20261010`, clean scorer at commit `55f728a3`, confirms that it already has finite canonical-grade comparison (lines 1830–1834) and hyphenated-venue grammar (line 316). The two historical defects are therefore **not established current live defects, and do not justify deploying the same fix again**. This is a tested offline application to the old validator. The 122-race ceiling describes missed historical input readiness only; current sustained coverage needs its own operational evidence.

Known race-level input failures produce explicit BLOCKED dispositions and processing continues to the next independent race. Unexpected programming errors are not hidden as input exclusions. No evidence shows that the historical July race loop itself aborted unrelated races; the repair's isolation test prevents introducing that failure into the offline utility.

All five generated JSON artifacts reproduce byte-for-byte in a second independent execution (`failures/reproduction-hashes.json`). Six regression tests pass, including the actual retained native validator rejecting the source/feature spelling pair, the adapter accepting it, hash tampering still failing, genuine grade disagreement and unsafe metadata still failing, and one rejected race not aborting or erasing the next race's disposition. Independent review recomputed the original 431/210/221 accounting and checked the source references; see `live_reconciliation_independent_review_20261011.md` for the final independent checks.

## Reproduce

From the reconciliation worktree, use the retained environment for the native scorer's NumPy dependency:

```bash
OPENBLAS_NUM_THREADS=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python scripts/diagnose_retained_forecast_failures.py \
  --access-boundary /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/access-boundary.json \
  --membership /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/residual-membership.json \
  --coverage /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/july-operational-coverage.json \
  --output-dir /tmp/greyhound-failure-reproduction-new
OPENBLAS_NUM_THREADS=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python -m unittest discover -s tests -p test_diagnose_retained_forecast_failures.py
```

The output directory must be new; previous corrections cannot be overwritten. Final retained outputs are under `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/failures/v2/`:

- `race-accounting.json`: all 431 races, exclusive and overlapping causes, append counts and attempt indexes.
- `attempt-accounting.json`: every 1,292 original attempt, exact source references, timing and repair checks.
- `derived-grade-corrections.json`: 369 explicit derived comparison corrections with the unchanged original reason and `new_forecast_generated: false`.
- `failure-summary.json`: reconciled counts and the 122-race input-readiness ceiling.
- `source-references.json`: source byte sizes and SHA-256 hashes.

The earlier `v1` outputs remain retained. `v2` adds market/form chronology checks and the unique-race timing ceiling; it neither replaces nor erases original failures. The original benchmark population stays at 210 retained forecasts. None of the 221 historical failures is converted into a live prediction for performance scoring.
