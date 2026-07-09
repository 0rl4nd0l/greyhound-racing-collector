# Accurate Predictions Remediation Closeout

Generated: 2026-06-23T19:05:00+10:00

## Outcome

The long-running goal was split into three parallel worker lanes and integrated.

The highest-value blocker was remediated: the latest rolling source set was
narrowed to 29 source reports / 35 races, while historical retained evidence had
100+ race packets. The implemented rolling-source fix now discovers retained
historical automatic unified-evidence reports from the evidence root and keeps
the eligibility gates intact.

## Accepted Worker Changes

- Rolling source inclusion:
  - `scripts/build_rolling_model_comparison_packet.py`
  - `tests/test_build_rolling_model_comparison_packet.py`
- Evidence-gap prioritization:
  - `scripts/build_unified_evidence_dataset.py`
  - `tests/test_build_unified_evidence_dataset.py`
- Gate-contract diagnostics:
  - `scripts/build_promotion_distance_report.py`
  - `scripts/build_high_accuracy_refinement_packet.py`
  - `tests/test_build_promotion_distance_report.py`
  - `tests/test_build_high_accuracy_refinement_packet.py`
- Shared retained-evidence output-dir guard dependency:
  - `utils/report_output_dir_guard.py`

## Validation

- Focused test suite: `47 passed`.
- Diff whitespace check: passed.
- Py compile: passed.
- Rolling report-only validation:
  - `ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`
  - `158` races
  - `1110` runner rows
  - `529` effective source reports
  - blockers: `[]`
- Promotion-distance report-only validation:
  - `PROMOTION_DISTANCE_REVIEW_READY`
  - gate-contract status: `PASS`
  - selected gate-contract candidate: `stage2_market_blend_80`
- Final high-accuracy report-only validation:
  - `READY_FOR_PROMOTION_PR_DRAFT`
  - promotion PR gate: `READY_FOR_PR_DRAFT`
  - selected stage: `odds_augmented_model_research`
  - selected candidate: `stage2_market_blend_80`
  - `promotion_pr_allowed: true`

## Boundaries Preserved

No DB mutation, live runtime/service mutation, training, production promotion,
EV, betting, snapshot rewrite, registry mutation, or gate weakening was
performed. Protected paths were unchanged in the generated high-accuracy packet.

## Remaining Owner Boundary

The generated evidence says the candidate is ready for a promotion PR draft, not
that production has been promoted. Any actual promotion remains owner-reviewed
and PR-boundary controlled.

## Docs Impact

- `docs_impact`: `DOCS_FOLLOWUP`
- `docs_checked`: `AGENTS.md`, board artifacts, worker result artifacts, changed
  report builders and tests
- `docs_changed`: `AGENTS.md`
- `docs_followup`: document canonical schemas for `source_discovery`,
  `race_gap_prioritization`, and gate-contract diagnostic fields if these report
  shapes become permanent
- `reason`: behavior and report schemas changed; operator guidance was updated,
  durable schema docs are still a follow-up
