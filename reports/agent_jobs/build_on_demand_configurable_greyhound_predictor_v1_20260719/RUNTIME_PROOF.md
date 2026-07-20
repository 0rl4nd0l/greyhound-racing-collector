# Runtime proof

Result: `DONE_WITH_RISK`.

The fixture CLI exercised the full orchestration boundary from named race
resolution through receipt selection, history sealing, feature scoring, model
and config identity, canonical stdout, bundle manifest, and replay. Immediate
capture was exercised with an isolated SQLite fixture and exact WIN/PLACE rows.
No fixture changed its source database bytes.

No live runtime proof was performed. The owner did not name and authorize a live
race, so the command did not contact live schedule/odds sources, acquire the real
collector lock, inspect a production database, or interact with any service or
timer. This is the intentional remaining runtime risk.

## Runtime Functionality Proof

- result: PARTIAL
- intended output: one canonical research-only prediction plus one isolated,
  reproducible run bundle for an exact pre-jump named race
- live output location: DATA_MISSING because no live race execution was
  authorized; fixture bundles existed only under pytest temporary directories
- pre-run max timestamp or count: production database and runtime artifacts were
  not queried; fixture source database hash was recorded before each write probe
- post-run max timestamp or count: production database and runtime artifacts were
  not queried or changed; fixture source database hash and row counts were exact
- rows/files inserted or updated after run start: zero production rows and zero
  runtime files; only isolated pytest temporary bundle files were created
- readiness/gate status: fixture command WORKING, live execution NOT_AUTHORIZED,
  merge BLOCKED_DO_NOT_MERGE by PR #46
- exact command/query used: `uv run scripts/predict_race_now.py --help`; focused
  fixture execution used `main()` with the exact race/model/config/odds selectors
- remaining blocker: PR #46 staged-descriptor leak blocks merge, and live
  functionality remains DATA_MISSING until the owner names and authorizes a race
