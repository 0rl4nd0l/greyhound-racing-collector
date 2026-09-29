# Independent review, attempts and validation

Fixed point: #198 `fc979e370cc518a14e029c37678a172ed7528ea6`.
Implementation commits: `be940a55` (adaptive) and `8852fd6b` (operational
history and controlled depth). Review command: `git diff fc979e37...HEAD`;
generated artifacts inspected through summaries, manifests and saved replay
checks rather than an indiscriminate full-data dump.

## Standards

The independent Standards reviewer checked applicable AGENTS guidance,
CONTEXT.md and pyproject.toml, plus the code-review smell baseline. No hard
documented-standard breach or refactoring requirement was found. One optional
focused-validation finding was resolved: the history test command now disables
automatic plugins, global conftest/coverage options and cache output. The task's direct user specification supplied the
review contract; no unrelated issue-tracker setup was performed.

## Spec: timing, leakage and fairness

The independent reviewer found no scientific timing/leakage/whole-field defect
in either experiment. The review confirmed earlier-only OOF selection, strict
prior-card capture admission, unchanged canonical definitions and model settings,
identical whole-field populations, training-only preprocessing, original
full/half replay and six explicitly new depth fits. Paired date clustering,
period effects, harmful adjustments and influence support the cautious decision.
Operational protected inputs were not reopened by the reviewer.

Two reproducibility findings were repaired:

1. A metadata-only reconstruction edit temporarily changed the source path
   pinned by the already-frozen depth protocol. The original exact
   `6ca4fc8f78912e1bb1f2c6b2acbcf89af113978876fdd151ce31db981ac252b5`
   source was restored and archived; the successor was preserved separately,
   with byte-identical paired features. Metadata counting moved to a separate
   helper. No protocol amendment or refitting was needed. All frozen input
   pins were subsequently checked successfully.
2. A history test command named an absent bare `python` alias / an interpreter
   lacking pytest. The report now specifies the actual existing research
   interpreter for execution and the verified R3 interpreter for pytest.
   No dependencies were installed.

The reviewer independently confirmed both fixes and publication readiness.
A nonblocking wording suggestion was also applied: the 678 added history rows
are counted across runner-target snapshots, not 678 globally independent events.

Standards: zero hard findings; one optional validation-documentation finding resolved. Spec: two reproducibility findings resolved,
zero unresolved scientific findings. This is research review, not production
activation or evidence of a predictive advantage.

## Executions and preserved attempts

| Attempt | Outcome / retained record |
|---|---|
| Initial adaptive interface test | Failed before module implementation; then passed. Synthetic only. |
| Adaptive attempt1 | Completed six prespecified validation trials, four evaluation procedures, zero fits. `history_support_20260929_adaptive_attempt1/trial_ledger.jsonl`. |
| Adaptive final evidence replay | Same choices and metrics; explicit predictions for both lambdas added, nonfinite support handling clarified. No new candidate. `history_support_20260929_adaptive_evidence/trial_ledger.jsonl`. |
| Depth preparation command | Missing bare Python alias, exit 127 before protocol or data decoding; preserved in frozen protocol. |
| Actual depth experiment attempt1 | All six fits completed; no failed fit and no subsequent refits. Local attempt tree and its portable manifest are retained. |
| Metadata reconstruction versions | Original/v2/v3 receipts preserved; final executable reconstruction restored to fitted source. A separate support-only helper reproduces duplicate/cap totals. |
| Review pytest attempts | First failed before tests because pytest was unavailable in research interpreter. A separate default-config review invocation loaded global app fixtures/coverage and was terminated during collection; no results from it are claimed. The isolated rerun disabled plugins/conftest/coverage defaults and passed all 16 synthetic tests. |
| Final validation | Ten unittest tests and six pytest tests passed; source pins pass; six saved-model replays differ by at most 2.22e−16 without additional fits. |

These are retrospective descriptions of tool attempts, not invented execution
timestamps. Scientific ledgers retain their own exact events and source hashes.
No failed/consumed operational attempt was changed, and no result-driven search
extension occurred.

## Final focused commands

```bash
cd /home/l4nd0/greyhound-history-support-20260929
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
TEST_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python
"$RESEARCH_PY" -m unittest tests.test_history_support_shrinkage tests.test_history_depth_comparison -v
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$TEST_PY" -m pytest --noconftest -q -o addopts= -p no:cacheprovider tests/test_retained_history_depth.py
```

All commands operate on synthetic fixtures or already-admitted development
artifacts as specified. No providers, runtime/service changes, production model
updates, protected target-result reads, or October amendments were performed.
