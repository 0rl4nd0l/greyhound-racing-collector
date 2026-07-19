# On-demand configurable predictor V1

Result: `DONE_WITH_RISK / PASS_WITH_DEPENDENCY_BLOCK`.

Commit `76f17dbeec78a43e5493a8049ff84c47a13d3e8f` adds the research-only
`scripts/predict_race_now.py` command, model adapters, canonical configs,
isolated bundle/replay support, documentation, and focused tests. No live race
was requested or run. No service, timer, production database, shadow history,
existing pull request, or production registry was mutated.

PR #46 exact head `624dde3067edda1bd045573e8bec5c9d749c6836` is an ancestor but remains
`BLOCKED_DO_NOT_MERGE`: its unused `append_shadow_record` path has the known
staged-descriptor leak. The new command imports only PR #47's exact
`score_from_artifacts` function and never calls a shadow writer.

Draft publication: PR #53,
https://github.com/0rl4nd0l/greyhound-racing-collector/pull/53.
