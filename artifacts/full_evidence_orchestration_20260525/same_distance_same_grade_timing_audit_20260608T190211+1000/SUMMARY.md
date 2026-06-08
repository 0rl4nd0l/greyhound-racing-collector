# Same-Distance/Same-Grade Timing Feature Audit

Final status: `AUDIT_COMPLETE_NEEDS_MORE_SOURCE_DATA`

## Scope

- Active shadow repaired matrix: `943` rows / `132` races.
- Shadow train split: `751` rows / `107` races.
- Shadow holdout split: `192` rows / `25` races.
- Feature schema: `78` features; `tgr_*` columns: `[]`.
- DB state: quick_check `ok`, official pool `214` races / `1493` dog rows.

## Affected Features

- `same_distance_same_grade_best_time`: train present `0/751`, holdout present `10/192`.
- `same_distance_same_grade_avg_time`: train present `0/751`, holdout present `10/192`.

## Root Cause

The active shadow train split has no safely recoverable rows for the two affected timing features.

- Missing safe target distance: `708` rows.
- Missing safe target grade: `708` rows.
- Safe target metadata present but no strict-canonical prior same-distance/same-grade valid time: `43` rows.
- Recoverable safe rows: `0`.

## Verdict

`NEEDS_MORE_SOURCE_DATA`

Keep `quarantine_feature`. Do not train, promote, mutate registry, enable TGR, write labels, write DB rows, or rewrite snapshots/manifests.
