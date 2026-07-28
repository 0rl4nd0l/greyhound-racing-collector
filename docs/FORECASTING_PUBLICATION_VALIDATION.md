# Forecasting publication validation

## Reconstruction provenance

The forecasting programme was recovered from experiment
`a9b16fd906838d1a5b0a7215b9401d2f31b28eae` (tree
`91bd7a8a5cc426c1f87f3a56cae2f3655eadf9cb`) and reconstructed onto canonical
base `02f59c147a7702a13eee154161a4be950aeb0e60` without the experiment's
sandbox-control deletions or `.gitignore` change. Product reconstruction
`e54bc58599d8e536ab918b26991ecbe97c023fe6` and normalized source
`e9e49816a33f751b680e3473f44b0c1cbc6f9775` remain in this branch's ancestry.
Later repair commits must remain descendants of that clean export and must not
replace its canonical controls, migrations, schemas, or post-base master work.

## Exact-head evidence contract

Historical candidate logs are not release evidence for a later commit. Every
pull request to `master` receives the stable
`Forecasting acceptance / tests-race-collection` GitHub Actions check. A
lightweight classifier compares the exact pull-request base and head, validates
its checked-in fixture cases, and selects the broadest applicable tier:

- `full_forecasting` for forecasting core and shared contracts, feature schemas,
  model/training/source-admission surfaces, common dependencies, test
  infrastructure, workflow/classifier changes, forecasting contract docs, or
  any unknown path;
- `manual_prediction` for the on-demand predictor, its two checked-in configs,
  operator docs, residual scorer, and directly associated tests; or
- `non_forecasting` only for explicitly allowlisted unrelated docs and UI
  surfaces.

Renames and copies classify both old and new paths. Deletions retain their
deleted path. Empty, malformed, unmatched, mixed, or overlapping changes select
the broader tier, with uncertainty selecting `full_forecasting`.

The stable gate checks out the pull-request head explicitly. The full tier runs
the complete `tests/race_collection` suite. The manual tier runs:

```text
uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/test_predict_race_now.py tests/test_predict_market_form_residual.py
```

Both relevant tiers upload:

- `forecasting-suite.log`, the complete combined test output; and
- `forecasting-ci-attestation.json`, containing the exact expected head, checked
  out commit, tree, selected tier and command, result, Python/platform/uv
  environment, and log SHA-256.

The unrelated tier returns successfully without dependency setup after stating
that no forecasting contracts changed. The stable job fails if classification
does not produce a trusted tier, checkout does not equal the pull-request head,
the selected suite exits nonzero, evidence cannot be generated, or the evidence
artifact cannot be uploaded. Because a tracked file cannot contain the identity
of the commit that contains itself, the CI attestation and the reviewer-local
post-commit report are the authoritative exact-head evidence surfaces.

## Required local acceptance

Run from a clean checkout of the exact pull-request head:

```text
uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection
```

The retained report must record the command, exit status, exact commit and tree,
environment, complete log path, and log SHA-256. It must also record focused
adversarial tests, populated migration and current-master overlap regressions,
schema parsing/validation, changed-file compile/lint/format, `git diff --check`,
and Git integrity.

## Repaired release contracts

Startup authenticates the resolved release root as the exact Git worktree for
the declared existing commit. The checked-out commit/tree, clean tracked and
untracked state, checked-in `race_collection/service.py`, executable bytes and
executable mode must match that immutable release. The executable cannot be a
symlink. This completes before runtime-adapter import or lease acquisition.

Adaptive odds use `adaptive-odds-timing-v1`: every immutable attempt stores
`scheduled_due_at` separately from `attempted_at`. Attempts may occur from zero
through five seconds after due, inclusive. Missing, early, duplicate,
noncanonical, wrong-policy, post-cutoff, or later attempts fail closed.
The complete Racing-Day odds cohort is parsed, domain-validated, and
artifact-authenticated before race state or odds authority is mutated.
Reconciliation advances cadence from due time rather than observed latency.

Official results use `official-result-timing-v1`. The ordered timeline is
`published <= observed <= attempted <= trusted_command`; the interval from
publication through attempt may be at most five minutes, inclusive. Missing,
reversed, future, wrong-policy, and excessive-latency inputs fail closed before
result authority is mutated. The complete Racing-Day result cohort is
source-authenticated and validated before any member enters result collection.

## Boundary

This validation is synthetic and repository-local. It does not start services,
access production data, mutate a production database, deploy, merge, place a
bet, or run a live forecast. Intended live runtime proof remains
`DATA_MISSING`.
