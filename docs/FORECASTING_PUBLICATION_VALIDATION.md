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
its checked-in fixture cases, and selects one of the eight risk tiers documented
in [forecasting_ci_tiers.md](forecasting_ci_tiers.md). CI contract, manual
prediction, official-result, forward-corpus, Operator UI, and isolated
forecasting-core changes receive focused validation. Shared or high-risk paths,
unknown paths, destructive changes, and combinations of two or more focused
product tiers fail closed to `full_forecasting`. Renames and copies classify
both old and new paths; empty or malformed change sets and uncertain rules also
fail closed.

The stable gate checks out the pull-request head explicitly. The full tier runs
the complete `tests/race_collection` suite plus directly associated top-level
manual, official-result, and Operator UI tests. It runs on scheduled and manual
dispatches, for PRs labeled `ci:full-forecasting`, and whenever risk
classification escalates. CI-only routing changes select `ci_contract`, which
validates classifier and workflow contracts plus one named smoke from each
focused tier without invoking the complete race-collection suite. If CI routing
paths accompany exactly one focused product tier, that product tier stays
primary with `ci_contract_changed=true`; the stable job runs the CI-contract
command first and then the product suite. CI-contract changes are not
transparent to two product tiers, a full-risk path, or any uncertain change.

Every tier except `non_forecasting` uploads:

- `forecasting-suite.log`, the complete combined test output, plus one command
  file and log per executed validation; and
- `forecasting-ci-attestation.json`, containing the exact expected head, checked
  out commit, tree, selected primary tier, `ci_contract_changed`, ordered
  commands, per-command result and log SHA-256, combined log SHA-256, and the
  Python/platform/uv environment.

The unrelated tier returns successfully without dependency setup after stating
that no forecasting contracts changed. The stable job fails if classification
does not produce a trusted tier, checkout does not equal the pull-request head,
the selected suite exits nonzero, evidence cannot be generated, or the evidence
artifact cannot be uploaded. Because a tracked file cannot contain the identity
of the commit that contains itself, the CI attestation and the reviewer-local
post-commit report are the authoritative exact-head evidence surfaces.

## Required acceptance

Run the selected command from [forecasting_ci_tiers.md](forecasting_ci_tiers.md)
in a clean checkout of the exact pull-request head. Scheduled, explicitly
dispatched, forced, and classifier-escalated validation runs:

```text
uv run --no-project --with-requirements requirements/all.in --with PyYAML python scripts/ci/run_full_forecasting.py
```

The retained report must record the command, exit status, exact commit and tree,
selected tier, environment, complete log path, and log SHA-256. Changed-file
fatal lint, Python compilation, workflow syntax validation, classifier tests,
and `git diff --check` remain required for CI routing changes.

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
