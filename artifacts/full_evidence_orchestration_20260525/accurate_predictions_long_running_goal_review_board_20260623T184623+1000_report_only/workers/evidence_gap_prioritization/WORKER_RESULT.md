# Evidence Gap Prioritization Worker Result

Generated: 2026-06-23

## Scope

Worker: `evidence-gap-prioritization-worker`

Allowed write scope used:

- `scripts/build_unified_evidence_dataset.py`
- `tests/test_build_unified_evidence_dataset.py`
- This worker result artifact

`scripts/forward_shadow_status_report.py` was already dirty before this worker and was not edited by this worker.

Hard boundaries preserved:

- No DB mutation.
- No live runtime or service mutation.
- No training.
- No promotion.
- No EV or betting output.
- No snapshot, registry, manifest, or gate-policy mutation.

## Guard

- Repo: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621`
- Branch: `codex/runtime-master-live-20260621`
- HEAD: `c96363fbcd708bba78ecfb69e4bc4dacb183d867`
- Upstream/base: `origin/codex/runtime-master-live-20260621`
- Guard runner: `/home/l4nd0/.agents/skills/tenn-git-guard/scripts/tenn_git_guard.py`
- Guard support: `PASS`
- Registry status: `PASS`
- Ledger status: `DATA_MISSING`
- Duplicate work classification: `DATA_MISSING_FALLBACK_CHECKED`
- Final guard decision: `warning`

Existing dirty files were treated as in-scope context and were not reverted.

## Implementation

Added race-level source-aware gap reporting to `unified_evidence_dataset_report.json` under:

- `race_gap_prioritization`

The new report object records:

- `raw_db_count_basis: false`
- source basis, using `join_eligibility_packet_accepted_race_ids` when available and falling back to dataset race IDs
- source and dataset race counts
- source-set missing race count
- sample-blocking gap count
- primary gap class counts
- all gap class counts
- top gap race IDs and compact top gap rows

Gap classes now separate:

- `source_set_missing`
- `identity_mismatch`
- `official_result_missing`
- `strict_prejump_odds_missing`
- `stage2_missing`
- `other_gate`

Also added per-race rejected identity context in joined-shadow official-result audits via `rejected_race_ids_by_reason`, so unsafe identity joins can be classified as `identity_mismatch` instead of being flattened into generic official-result absence.

`SUMMARY.md` now surfaces the race gap source basis, raw DB count basis, class counts, and top gap race IDs.

## Root Cause Improved

The reporting problem was that official-result and strict pre-jump odds gaps could be read as raw/global DB shortages, even when the rolling lineage blocker was the current source set. The new report classifies gaps from the current source/dataset race set and explicitly marks `raw_db_count_basis` as false.

This does not repair rolling source discovery. It makes the remaining gap classes actionable once rolling source inclusion is restored, and it prevents source-set misses from being mixed with official-result or strict-odds collection gaps.

## Tests Run

Passed:

- `python3 -m py_compile scripts/build_unified_evidence_dataset.py tests/test_build_unified_evidence_dataset.py`
- `uv run --isolated --with pytest pytest --noconftest tests/test_build_unified_evidence_dataset.py -q`
  - Result: `16 passed`
- `uv run --isolated --with pytest --with requests pytest --noconftest tests/test_build_rolling_model_comparison_packet.py tests/test_build_promotion_distance_report.py -q`
  - Result: `13 passed`
- `uv run --isolated --with pytest pytest --noconftest tests/test_build_high_accuracy_refinement_packet.py -q`
  - Result: `18 passed`
- `git diff --check -- scripts/build_unified_evidence_dataset.py tests/test_build_unified_evidence_dataset.py`

Validation environment notes:

- `python3 -m pytest ...` could not run because system Python has no `pytest`.
- `uv run --with pytest pytest ...` with repo conftest hit unrelated app dependency imports (`flask`, then `flask_compress`).
- Focused script tests were run with `--noconftest` to avoid unrelated Flask app startup.

## Code Review

Review result: no critical findings, warnings, or suggestions found in the worker-owned diff after focused validation.

Review checks covered:

- additive schema shape
- no gate weakening
- no DB/runtime/registry write path
- identity mismatch classification does not accept rejected rows
- source-set race IDs are normalized
- focused downstream consumers still pass

## Docs Impact

- `docs_impact`: `DOCS_FOLLOWUP`
- `docs_checked`: board artifacts, `scripts/build_unified_evidence_dataset.py`, downstream rolling/promotion/high-accuracy tests
- `docs_changed`: none
- `docs_followup`: document `race_gap_prioritization` schema and summary fields in the durable evidence/report schema docs if this report shape becomes canonical
- `reason`: report artifact shape changed, but durable docs are outside this worker write scope

## Remaining Blockers

- Rolling source inclusion is still the upstream blocker for the current lineage: the reviewed audit showed 999 safe joined races not requested by the latest rolling source set.
- This worker did not edit rolling source discovery, promotion-distance, high-accuracy, runtime, service, DB, or registry code.
- Full repo pytest with `tests/conftest.py` was not run because it requires unrelated Flask app dependencies in this environment.
