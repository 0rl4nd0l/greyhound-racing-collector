# Gate Contract Worker Result

Generated: 2026-06-23

## Scope

Worker: `gate-contract-worker`

Edited only:
- `scripts/build_promotion_distance_report.py`
- `scripts/build_high_accuracy_refinement_packet.py`
- `tests/test_build_promotion_distance_report.py`
- `tests/test_build_high_accuracy_refinement_packet.py`
- this worker result artifact

No DB mutation, live runtime/service mutation, training, promotion, EV/betting,
snapshot rewrite, registry mutation, or gate weakening was performed.

## Result

Promotion-distance gate diagnostics now distinguish:
- `DATA_MISSING`: true missing audit inputs, such as absent candidate metrics or
  missing primary/market baselines.
- `SOURCE_NOT_READY`: evidence exists, but the rolling source is not evaluable
  for the gate yet, such as collecting status, non-unified sample, unmet sample
  floor, or fewer than 100 rolling races.
- `POLICY_FAILED`: the source is evaluable, but the declared
  `dual_baseline_market_rank_primary_safety` policy has no passing candidate.

The high-accuracy PR gate now carries promotion-distance blockers and the
promotion-distance gate-contract diagnostic summary even when no candidate
passed the rank-first gate. This keeps `promotion_pr_allowed=False` conservative
while making the cause actionable.

## Diagnostics Added

`promotion_distance_report.json` now includes these fields under
`gate_contract_candidate`:
- `audit_classification`
- `policy_evaluation_status`
- `data_missing_reasons`
- `source_not_ready_reasons`
- `policy_failure_reasons`
- `candidate_policy_blocker_counts`
- `candidate_gate_matrix_row_count`
- `candidate_metrics_key_count`
- rolling status/scope/floor/sample fields used for classification

`high_accuracy_refinement_packet.json` now preserves the promotion-distance
`gate_contract_candidate` summary and includes compact diagnostics under
`promotion_pr_gate.promotion_distance_gate_contract`.

## Tests Run

Command:

```bash
uv run --with-requirements requirements/requirements.lock --with pytest pytest tests/test_build_promotion_distance_report.py tests/test_build_high_accuracy_refinement_packet.py -q
```

Result:

```text
22 passed in 14.95s
```

Additional check:

```bash
git diff --check -- scripts/build_promotion_distance_report.py scripts/build_high_accuracy_refinement_packet.py tests/test_build_promotion_distance_report.py tests/test_build_high_accuracy_refinement_packet.py
```

Result: passed with no whitespace errors.

## Remaining Blockers

- Existing generated artifacts still show their old ambiguous diagnostics until
  promotion-distance and high-accuracy packets are rerun.
- Promotion remains blocked. This worker did not restore the narrowed rolling
  source set, increase the rolling denominator to 100+ races, train a candidate,
  promote a model, or change the dual-baseline market-rank safety policy.
- Current known blockers remain actionable rather than cleared:
  `no_candidate_passed_rank_first_accuracy_gate`, high-accuracy selected
  candidate missing when no stage passes, rolling sample below review floor, and
  `promotion_pr_allowed=False`.
