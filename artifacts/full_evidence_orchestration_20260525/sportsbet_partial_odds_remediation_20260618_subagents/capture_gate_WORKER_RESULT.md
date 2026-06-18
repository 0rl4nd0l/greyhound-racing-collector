# Capture Gate Worker Result

## Worker Metadata

- Parent task id: `sportsbet-partial-odds-remediation-20260618`
- Worker: `Worker B`
- Lane: autonomous capture validation and reporting lane
- Worktree: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-master-merge-autonomous-accuracy-odds-v1-20260618`
- Branch: `master`
- HEAD: `4b7f5941f98cdadf5ed11c492f8d3b792de42665`
- Task card: `DATA_MISSING`; direct user-approved bounded remediation
- Ledger status: `DATA_MISSING`; Tenn-specific task-card / agent ledger tooling was not found in this greyhound checkout during bounded inspection.

## Task Status

`DONE_WITH_RISK`

Implementation and focused validation completed inside the assigned capture gate lane. Risk remains because ledger/task-card tooling is unavailable in this checkout and unrelated outside-lane tracked dirt appeared after preflight.

## Preflight

Command:

```bash
pwd && git branch --show-current && git rev-parse HEAD && git status --short --untracked-files=all
```

Result:

```text
/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-master-merge-autonomous-accuracy-odds-v1-20260618
master
4b7f5941f98cdadf5ed11c492f8d3b792de42665
```

Preflight status showed unrelated untracked report artifacts under `artifacts/full_evidence_orchestration_20260525/...`. No tracked dirt in the allowed code/test files was present at preflight.

## Files Changed

- `scripts/autonomous_live_odds_capture.py`
- `tests/test_autonomous_live_odds_capture.py`
- `artifacts/full_evidence_orchestration_20260525/sportsbet_partial_odds_remediation_20260618_subagents/capture_gate_WORKER_RESULT.md`

## Implementation Summary

- Added fail-closed validation diagnostics for partial same-race Sportsbet WIN captures.
- When active expected runners are complete in the PLACE market but WIN rows are partial, validation now emits:
  - `failure_root_cause: sportsbet_win_market_partial_but_place_complete`
  - `failure_detail` with active expected count, accepted WIN rows, missing active count, extra count, fetch `win_count`, and fetch `place_count`
  - a reason string carrying those counts for operator-facing summaries
- Blocked attempt summaries now include active expected runner count plus validation failure root cause/detail.
- Extras remain distinct as `sportsbet_unexpected_runner_identity_mismatch`; they do not get collapsed into the partial WIN-market diagnosis.
- Capture still fails closed. No PLACE odds are treated as WIN odds, no synthetic odds are created, and append is not attempted after failed validation.

## Tests And Checks

Command:

```bash
python3 -m py_compile scripts/autonomous_live_odds_capture.py tests/test_autonomous_live_odds_capture.py
```

Result: exit 0.

Command:

```bash
uv run --with-requirements requirements/requirements.lock pytest tests/test_autonomous_live_odds_capture.py -q
```

Result:

```text
28 passed in 15.26s
```

The `uv` command printed:

```text
warning: No `requires-python` value found in the workspace. Defaulting to `>=3.11`.
```

## Safety Boundaries

- No live Sportsbet/network scrape run.
- No DB writes run.
- No label, registry, training, promotion, snapshot, or pointer writes.
- No push, merge, rebase, clean, delete, stash, or GitHub mutation.
- No edits made outside the exact allowed files by this worker.

## Current Dirt / Collision Notes

Final `git status --short --untracked-files=all` shows this worker's allowed-file changes plus unrelated outside-lane tracked dirt:

```text
 M scripts/autonomous_live_odds_capture.py
 M sportsbet_odds_integrator.py
 M tests/test_autonomous_live_odds_capture.py
 M tests/test_sportsbet_odds_safety.py
?? artifacts/full_evidence_orchestration_20260525/sportsbet_partial_odds_remediation_20260618_subagents/capture_gate_WORKER_RESULT.md
```

There are also unrelated untracked report artifacts under `artifacts/full_evidence_orchestration_20260525/...` from other lanes. The outside-lane tracked files `sportsbet_odds_integrator.py` and `tests/test_sportsbet_odds_safety.py` were not edited by this worker and were left untouched.

## Risks / DATA_MISSING

- `DATA_MISSING`: no task-card validator or branch-independent agent ledger tooling was available in this greyhound checkout.
- Outside-lane tracked dirt appeared after preflight, so final integration should preserve or reconcile that extractor-lane work separately.
- This fix only improves diagnostics/gating classification. It does not repair the Sportsbet extractor/rendering source of partial WIN rows.

## Recommended Action

Have the parent orchestrator review this capture-gate diff together with the separate Sportsbet extractor lane. Keep this lane's change if the extractor fix can still produce partial WIN outputs in edge cases, because the capture report will now identify `sportsbet_win_market_partial_but_place_complete` instead of implying scratch or odds absence.
