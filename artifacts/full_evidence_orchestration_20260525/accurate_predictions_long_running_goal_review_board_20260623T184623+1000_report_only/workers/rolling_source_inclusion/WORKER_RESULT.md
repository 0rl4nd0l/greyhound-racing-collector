# Rolling Source Inclusion Worker Result

Generated: 2026-06-23

## Scope

- Worker: `rolling-source-inclusion-worker`
- Task card: `artifacts/full_evidence_orchestration_20260525/accurate_predictions_long_running_goal_review_board_20260623T184623+1000_report_only/NEXT_GOAL.md`
- Allowed code scope used:
  - `scripts/build_rolling_model_comparison_packet.py`
  - `tests/test_build_rolling_model_comparison_packet.py`
- No DB mutation, live runtime/service mutation, training, promotion, EV/betting, snapshot rewrite, registry mutation, or gate weakening was performed.

## Root Cause

The rolling comparison script only evaluated the explicit `--unified-evidence-report`
paths handed to it. The latest rolling packet therefore stayed narrowed to the
29 current/backlog source reports already present in its invocation.

Older retained unified-evidence reports were also effectively undiscoverable from
the live runtime checkout because many historical reports store `dataset_jsonl`
as a repo-relative `artifacts/...` path from the evidence-producing checkout.
The rolling script resolved those paths against the live runtime checkout, not
against the retained evidence root's repo, so broad historical reports appeared
dataset-missing even though their datasets still existed under the retained root.

## Fix

- Added retained-root-aware dataset path resolution for historical
  `dataset_jsonl` values.
- Added in-script cumulative historical unified-evidence discovery from
  `--evidence-root`, capped at the existing historical limit of 500 automatic
  reports.
- Preserved source safety filters:
  - final status must be `UNIFIED_EVIDENCE_DATASET_BUILT`
  - `unified_evidence_eligible_rows` must be positive
  - dataset JSONL must exist after retained-root resolution
  - manual, probe, validation, odds-only, and lock-wait wrapper names are ignored
- Preserved existing race-level gates in `collect_race_groups`; expanded reports
  still must pass official-result availability, prediction availability,
  full unified-evidence eligibility, winner-count, and race-id dedupe checks.
- Preserved existing explicit input order semantics so later explicit reports
  still win race-id dedupe.
- Added `source_discovery` counts to the rolling report and summary.

## Files Changed

- `scripts/build_rolling_model_comparison_packet.py`
- `tests/test_build_rolling_model_comparison_packet.py`
- `artifacts/full_evidence_orchestration_20260525/accurate_predictions_long_running_goal_review_board_20260623T184623+1000_report_only/workers/rolling_source_inclusion/WORKER_RESULT.md`

Pre-existing dirty edits in `scripts/build_rolling_model_comparison_packet.py`
for retained output-dir guards were preserved and not reverted.

## Tests Run

```bash
uv run --with pytest --with flask --with flask-compress --with flask-cors --with pyyaml --with matplotlib --with seaborn --with scikit-learn --with pandas --with requests --with joblib pytest tests/test_build_rolling_model_comparison_packet.py -q
```

Result: `9 passed in 0.25s`

```bash
python3 -m py_compile scripts/build_rolling_model_comparison_packet.py tests/test_build_rolling_model_comparison_packet.py
```

Result: pass

```bash
git diff --check -- scripts/build_rolling_model_comparison_packet.py tests/test_build_rolling_model_comparison_packet.py
```

Result: pass

## Report-Only Validation

Command class: rebuilt rolling comparison from the latest narrowed
`rolling_model_comparison_20260623T181701+1000_daemon_rejoin` source list using
the retained evidence root:

```bash
python3 scripts/build_rolling_model_comparison_packet.py \
  --evidence-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --output-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/rolling_model_comparison_20260623Trolling_source_inclusion_worker_validation \
  --unified-evidence-report <29 latest source reports>
```

Output:

- Final status: `ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`
- Explicit source reports: `29`
- Historical reports discovered: `500`
- Effective source reports: `529`
- Sample race count: `158`
- Minimum races for review: `100`
- Sample floor met: `true`
- Races needed for review: `0`
- Blockers: `[]`
- Output dir: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/rolling_model_comparison_20260623Trolling_source_inclusion_worker_validation`

## Remaining Blockers

No blocker remains for this worker's assigned source-inclusion scope. Promotion
is still not implied or allowed; downstream gate-contract and model-quality
workers must evaluate the restored denominator under their own scopes.
