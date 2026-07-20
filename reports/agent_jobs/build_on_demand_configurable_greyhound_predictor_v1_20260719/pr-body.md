## Summary

Adds one research-only command for an immediate named pre-jump prediction with
finite model/config selection, verified receipt reuse or isolated capture,
query-only sealed history, canonical bundles, and deterministic replay.

## Hard dependency block

This PR must remain draft and must not merge. Exact PR #46 head
`624dde3067edda1bd045573e8bec5c9d749c6836` is an ancestor and is
`BLOCKED_DO_NOT_MERGE`: its unused `append_shadow_record` atomic-publication path
leaks the staged descriptor if `os.fchmod` or `os.fdopen` fails after `os.open`.
This change does not modify or call that writer. It reuses only scoring:
`score_from_artifacts` is exact to PR #47 and `score_race` is exact to PR #46.

## Validation

- 27 focused tests passed
- 371 combined relevant tests passed
- Ruff check and format passed
- `py_compile` passed
- exact `uv run scripts/predict_race_now.py --help` passed
- independent review: `PASS_WITH_DEPENDENCY_BLOCK`

No live prediction ran. No production DB, shadow output, service, timer, daemon,
registry, betting, model coefficient, or existing PR was mutated.
