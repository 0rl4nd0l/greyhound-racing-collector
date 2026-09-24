# Evidence for the systematic offline search

See [the research report](../offline_systematic_20260924_results.md) for the population, exposure incident, splits, results and reproduction commands.

This directory contains the complete append-only model-search ledger and compact review artifacts from `/home/l4nd0/greyhound-offline-systematic-output-20260924/`. `outer_predictions.csv` contains every eligible later runner, its outcome, all compared predictions, selection flags and abstentions. It permits rescoring the common evaluation population without accessing original mixed-population sources. No protected outcome is included.

`SEARCH_PROTOCOL.json` fixes inputs, runner hashes, recipes and nested splits. `STAGE_PLAN.json` records the finite search budget. `frozen_selections.json` records earlier-validation decisions before later scores. `LEDGER_VERIFICATION.json` records independent verification of all 1,000 chained entries. Preparation failures and post-fit diagnostic work have separate journals; the completed experiment ledger is unchanged.

`SOURCE_IDENTITIES.json` pins all seven executed Python sources and records an independent CSV rescore of all nine log-loss/Brier comparisons. `protected_records.json` retains the explicit reservation union used by preparation, containing identities and protection reasons only. See the foundation audit for alias limitations and the additional chronological guard.

The ledger refers to hashed local inner prediction arrays, fitted-model receipts and source snapshots in the output directory. Those local artifacts, raw source cards and fitted tree binaries are not bundled here. Full refitting requires the pinned original inputs and environment documented in the report. All later evaluation races were previously inspected development data; this package is not a fresh holdout.
