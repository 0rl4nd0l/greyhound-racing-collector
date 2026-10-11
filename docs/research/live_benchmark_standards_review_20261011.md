# Standards review, live benchmark, 11 October 2026

Fixed base: `779761165637b709227d965f6c9be7e80706d23f`. Review covered the task-only new `score_live_benchmark.py`, `select_live_benchmark.py`, `run_retained_live_benchmark.py`, `project_live_benchmark_results.py` and their focused tests. The comparison is the new-file working-tree delta; these files were untracked at review, so a three-dot tracked diff alone would omit the entire change. Root performs Spec review separately.

Applicable standards: user-supplied project guidance (preserve existing work; offline-only authority), repository `AGENTS.md` (inventory before coverage claims; smallest correct change; read-only DB/evidence; no unnecessary abstractions), and the code-review skill's heuristic smell baseline. No unsupported V2 activation or fresh external issue lookup is required for this explicitly authorized task.

**Two concrete correctness findings were repaired and rechecked:**

1. `project_live_benchmark_results.py` correctly applies SQL `WHERE race_id=?` before decoding allowed outcomes, with `mode=ro` and `query_only`. The repaired source starts `BEGIN` before the race/runner queries, preserving one read snapshot under concurrent writes. This was a consistency issue, not broad reserved-outcome access.
2. `score_live_benchmark.py` computes timing median as `sorted(values)[len(values)//2]`, which is the upper middle observation for even samples. The repaired source uses `statistics.median`, averaging the two middle values.

**One scientific-interface concern was repaired:** primary selector evaluation now reuses the result-independent forecast census via `score_forecast_selection`; missing results preserve original choices and explicit missing counts. Separately retained result-conditioned diagnostics carry `EXPLORATORY_RESULT_AVAILABILITY_CONDITIONED_SUBSET_NOT_PRIMARY_SELECTOR_TEST`. Freezing does not inspect winner values, passes remain recorded, missing quality is explicit, and later exposure is labelled exploratory.

No hard documented style breach requires changes. Possible duplicated selector selection logic is a heuristic maintainability smell; no refactor is requested during this focused audit. Source-level boundaries are narrow and local; no network, fit, provider, service or production mutation appears in these utilities.

Validation: the installed project Python ran **23 focused tests, all passing**. They cover hand-calculated log loss/Brier/tie credit, infinite zero-winner loss, allocation/replay/timestamp/field rejection, exact SQL exclusion of reserved and suffix identities, unchanged DB bytes, outcome-blind thresholds, missing-quality passes edited-plan rejection, and invariant primary choices when all later outcomes are missing. System Python lacked pytest; no dependency was installed.

Review disposition: requested fixes rechecked; no outstanding blocking Standards finding. These tests substantiate mechanisms, not a nonempty strict real-world comparison.
