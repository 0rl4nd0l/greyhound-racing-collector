## Summary

- transplant only PR #53's eight-file on-demand predictor delta onto exact PR #56 master
- consume master's race identity, grade, jump-time, runner/provenance, record-V3, and effective-state-V2 contracts
- preserve two finite model/config modes, verified receipt reuse, isolated fixed-window capture, canonical hash-sealed bundles, and deterministic replay
- contain all runtime writes to the isolated bundle and keep production/database/model/betting surfaces unchanged

## Validation

- `35 passed` focused
- `560 passed, 1 skipped, 4 deselected` relevant regressions; the four separated cases are reproduced master/environment cases
- Ruff check and format pass
- Python 3.11 compile pass
- receipt-only Bendigo R5 proof failed closed as `DATA_MISSING` on missing target-grade context; no capture or database write

## Boundaries

Draft review only. This PR does not authorize merge, deploy, activation, production persistence, EV, or betting, and does not mutate PRs #47, #48, #52, #53, or #54.
