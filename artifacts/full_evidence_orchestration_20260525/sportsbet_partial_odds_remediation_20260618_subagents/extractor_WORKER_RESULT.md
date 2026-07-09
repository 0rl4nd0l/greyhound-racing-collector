# Worker Result: Sportsbet Extractor Lane

## Scope

- Parent task id: `sportsbet-partial-odds-remediation-20260618`
- Lane: Sportsbet extractor lane
- Worktree: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-master-merge-autonomous-accuracy-odds-v1-20260618`
- Branch: `master`
- HEAD: `4b7f5941f98cdadf5ed11c492f8d3b792de42665`
- Upstream: `origin/master`
- Selected base: `origin/master`
- Merge-base: `4b7f5941f98cdadf5ed11c492f8d3b792de42665`
- Task card: `DATA_MISSING`; direct user-approved bounded remediation.
- Ledger status: `DATA_MISSING`
  - `scripts/agent_job_registry.py` unavailable in this greyhound checkout.
  - `docs/agent_registry/task_ledger/LEDGER.jsonl` unavailable.
  - Common-dir `tenn-agent-registry/task-ledger.jsonl` unavailable.

## Task Status

`DONE_WITH_RISK`

The extractor fix and focused regression test are complete and validated. Risk remains only from missing Tenn task-card/ledger tooling in this checkout and no live Sportsbet scrape by design.

## Changes Made

- `sportsbet_odds_integrator.py`
  - Removed `candidate_cards[:8]` truncation in `extract_odds_strategy_runner_cards`.
  - Processes all Sportsbet candidate runner containers.
  - Dedupe now happens after parsing runner text and deriving a runner identity:
    - explicit box plus normalized dog name when runner text provides box metadata;
    - normalized dog name only when the box source is list-position fallback.
  - Preserves existing fail-closed behavior for concatenated multi-runner blocks.
  - Added expected-count-aware debug trigger when 8 or more explicit runner identities are discovered but fewer odds rows are extracted.

- `tests/test_sportsbet_odds_safety.py`
  - Added fake Selenium primitives and a focused regression for 16 interleaved candidate cards.
  - The first 8 cards cover only dogs 1-4 as duplicate containers; the full candidate list covers dogs 1-8.
  - Expected result is 8 WIN runners with explicit box numbers and first-seen WIN odds, not 4.

- `artifacts/full_evidence_orchestration_20260525/sportsbet_partial_odds_remediation_20260618_subagents/extractor_WORKER_RESULT.md`
  - Mandatory worker result file.

## Touched Files

- `sportsbet_odds_integrator.py`
- `tests/test_sportsbet_odds_safety.py`
- `artifacts/full_evidence_orchestration_20260525/sportsbet_partial_odds_remediation_20260618_subagents/extractor_WORKER_RESULT.md`

## Validation

- `python3 -m py_compile sportsbet_odds_integrator.py tests/test_sportsbet_odds_safety.py`
  - Exit status: 0

- `uv run --with-requirements requirements/requirements.lock pytest tests/test_sportsbet_odds_safety.py -q`
  - Exit status: 0
  - Result: `19 passed in 13.45s`
  - Note: `uv` warned that no `requires-python` value was found and defaulted to `>=3.11`.

## Git Preflight Evidence

- `pwd`: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-master-merge-autonomous-accuracy-odds-v1-20260618`
- `git branch --show-current`: `master`
- `git rev-parse HEAD`: `4b7f5941f98cdadf5ed11c492f8d3b792de42665`
- `git remote -v`: `origin https://github.com/0rl4nd0l/greyhound-racing-collector.git`
- `git rev-parse --abbrev-ref --symbolic-full-name @{u}`: `origin/master`
- Allowed files were clean before mutation.

## Existing Dirt Observed

Preflight showed unrelated untracked artifacts under:

- `artifacts/full_evidence_orchestration_20260525/high_accuracy_refinement_packet_20260618T*/`
- `artifacts/full_evidence_orchestration_20260525/prediction_accuracy_system_audit_20260618T143833+1000_post_merge_live_evidence_report_only/`
- `artifacts/full_evidence_orchestration_20260525/promotion_distance_report_20260618T*/`
- `artifacts/full_evidence_orchestration_20260525/promotion_gate_contract_audit_20260618T*/`

Final status also showed unrelated tracked changes and another worker artifact:

- `scripts/autonomous_live_odds_capture.py`
- `tests/test_autonomous_live_odds_capture.py`
- `artifacts/full_evidence_orchestration_20260525/sportsbet_partial_odds_remediation_20260618_subagents/capture_gate_WORKER_RESULT.md`

These were not touched by this worker.

## Risks And Boundaries

- No live Sportsbet/network scrape was run.
- No DB writes were run.
- No GitHub mutation, push, merge, rebase, reset, clean, stash, or branch deletion was performed.
- The debug trigger reuses the existing `_save_debug_info` mechanism; Worker B/main integration can tune artifact routing if they want debug output outside `debug_screenshots/`.

## Recommended Action

Main integration should review the three-file worker diff and, if accepted, run any broader odds integration suite it requires before committing or merging.
