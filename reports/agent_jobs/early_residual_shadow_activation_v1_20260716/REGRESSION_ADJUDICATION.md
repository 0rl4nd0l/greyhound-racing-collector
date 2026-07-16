# Regression adjudication

## Root cause

External one-shot predictions competed for the same continuously handed-off
collector lock and often began too late. Moving scoring into the legitimate
odds-only lock owner removes that scheduling race.

The first live attempt then exposed a narrower generator defect: the full
service had an exact Stage-2 feature-model pin, but the odds-only CLI and unit
did not carry it. Automatic lookup in the odds evidence root returned no model,
so the new stage correctly failed closed.

## Repair

- Added `--shadow-model` to the odds-only run and service-generation commands.
- Threaded the explicit path through the generated unit and early plan.
- Retained evidence-root confinement for fallback discovery while allowing the
  operator-configured external model used by the existing full service.
- Added regression coverage for parsing, generation, external-path planning,
  and execution-before-lock-release.

## Non-regressions

- No timer calendar or capture offset changed.
- No model, coefficient, preprocessing, feature order, strength, seed,
  threshold, or normalization changed.
- No result, label, outcome, cohort, production prediction, registry, betting,
  promotion, or merge path was added.
- PR #45 resource isolation remains active and in ancestry.
- PR #47 remains untouched at its exact reviewed head.
- Append persistence remains outcome-free, canonical, idempotent, and
  fail-closed.

## Adjudication

The configuration defect is fixed and installed. The read-only exact-packet
plan now returns `READY` with zero blockers. End-to-end live append evidence is
`PARTIAL`, solely because no eligible race occurred after installation; it may
not be upgraded until the next genuine pre-jump capture produces the record.
