# Live Daemon Data Availability Repair

Final status: `PARTIAL_REPORT_ONLY_DAEMON_DATA_AVAILABILITY_REPAIR_READY_KEEP_FEATURES_QUARANTINED`

This packet covers the current live daemon data-availability issues found by the delegated read-only agents. The code repair is scoped to deterministic TheDogs promoted-reserve result parsing/joining and report-only visibility for same-distance same-grade feature blockers.

## What Changed

- TheDogs result rows for promoted reserves can now be remapped from rug `9`/`10` back to the frozen pre-jump participant box only when the source row includes `(from box N)` and the cleaned dog name exactly matches the frozen participant.
- Rejected reserve remaps now carry explicit reasons such as `promoted_reserve_name_mismatch` and `duplicate_promoted_reserve_target_box`.
- Forward-shadow result joins now use the same remap before strict exact box/name validation.
- Autopilot and daemon summaries now surface feature data availability, blocker counts, same-distance history status, and live same-distance feature rows.

## What Did Not Change

- No training, promotion, registry mutation, production pointer update, DB write, label write, snapshot or manifest mutation, odds/EV fabrication, or betting action was performed.
- `same_distance_same_grade_best_time` and `same_distance_same_grade_avg_time` remain quarantined.
- Result and market leakage boundaries are unchanged.

## Remaining Gaps

- Same-distance same-grade train coverage remains `0/751` for both watched features.
- Current live same-distance provenance is only `6/122` rows, useful as collection evidence but not enough to activate features.
- A fresh report-only candidate metric comparison is still missing/stale against the `442` safe joined race aggregate.
- The runtime variant with autonomous official-result artifacts still needs a no-DB same-cycle artifact-to-join flow.
- Odds identity normalization remains a separate data availability issue.
