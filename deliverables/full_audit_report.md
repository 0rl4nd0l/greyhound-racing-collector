# Full Audit Report (executive summary)

Date: 2025-09-15
Branch: fix/sportsbet-odds-stability

1) System health and readiness
- Contract: V4 feature contract strictly matches code (make contract-validate: success)
- Schema: make schema-tests PASS; make schema-monitor shows no drift (stub). DBs contain dogs, dog_race_data, race_metadata, live_odds.
- App smoke (TESTING=1, no scraping): Passed; ML V4 loaded; health endpoints 200.
- Quick backend bundle: 52 tests passed; indicates stable API and DB ops.

2) Prediction pipeline checks (V4)
- /api/predict_v4 across multiple CSVs succeeded under:
  - optimizer OFF / TGR OFF
  - optimizer ON / TGR OFF
  - optimizer OFF / TGR ON
- All returned 200 OK. Optimizer integrated and produced optimized predictions where enabled. TGR toggled, but features were zero for sampled races (needs enrichment). Logs showed temporal integrity validations passed.
- Saved matrix: reports/endpoint_io/predict_v4_matrix.json

3) FastAPI /health refinement
- Now probes a safe table (default race_metadata) and degrades gracefully if missing; reports db_probe_table in components. Verified 200 OK response.

4) DB schema parity
- Writable/stage DBs include live_odds.topN; analytics DBs don’t. Code handles both (fallback paths are in place). Optional standardization deferred.

5) CSV validation (sample)
- Ran validator on 10 upcoming CSVs inside project venv. Reports were saved next to CSVs; summary written to reports/validation/upcoming_validation_summary.json.
  - Examples: 0–66 issues across sampled files (records parsed varied). Action: re-run validator in CI gate and automate archiving of invalid files per policy.

6) Codebase reconnaissance
- Route map saved to reports/code_map.json. Flask app.py is primary; FastAPI is auxiliary. Blueprints include analytics_api.py and model_training_api.py.

7) Notable warnings and follow-ups
- Logger: one EnhancedLogger.info(...) call used extra= in FastAPI /health path (harmless). We can standardize to avoid extra kwarg.
- TGR: repeated warnings of all-zero TGR features; likely missing enrichment; enable TGR_FEATURES_ENABLED and/or backfill to exercise TGR.

8) Recommendations (next actions)
- Broader pytest sweep (exclude E2E/load initially) to solidify the branch.
- Commit WIP changes (V4 inference/optimizer, Sportsbet odds stabilization, FastAPI /health) in logical chunks.
- Re-enable Playwright API-only E2E in CI once backend suite is green; iterate to green.
- Optional: standardize live_odds schema across DBs to include topN; not urgent due to graceful handling.
- Address validator issues in upcoming_races_temp by moving offending files to archive or correcting headers/fields per FORM_GUIDE_SPEC.
- Consider adding a CI step to run the CSV validator on a small sample to maintain input quality.

Appendix (artifacts created)
- reports/endpoint_io/predict_v4_matrix.json
- reports/validation/upcoming_validation_summary.json
- reports/code_map.json
