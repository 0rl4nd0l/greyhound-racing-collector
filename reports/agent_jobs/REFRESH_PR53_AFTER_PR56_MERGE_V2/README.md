# PR #53 successor refresh

This run transplanted the eight-file on-demand predictor implementation from
draft PR #53 onto exact master `585052ba7271f3a7e357dd5b69aec7f661591938`.
It consumes master's PR #56 identity, grade, jump-time, runner, provenance,
record-V3, and effective-state-V2 contracts. No old PR, service, timer, daemon,
production database, model pointer, betting surface, or production output was
mutated.

Immediate repo-root command:

```bash
uv run --no-project scripts/predict_race_now.py --race "<venue> r<number>" --model latest-research --config configs/prediction/manual-default.json --odds-source auto
```

Use `uv run --no-project scripts/predict_race_now.py --list-configs` to list the
two finite checked-in modes and their resolved immutable hashes.
