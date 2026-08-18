# Validation

## Passing

- 17 focused and adversarial unittests pass.
- `py_compile` passes for the evaluator and focused test file.
- The result-bearing Betfair header is rejected before a data row or full-file
  hash is read.
- Wrong-path Sportsbet parsing fails before the result path is opened.
- Generic result parser and authorization-bypass symbols are absent.
- Premature scoring fails before result access or consumed-marker creation.
- Authorized scoring is single-use and consumes before result opening.
- Population sealing and scoring replay deterministically in the synthetic
  label-blind fixture.
- Replacement candidate bytes exactly match PR #137.
- Replacement JSON parses, `SHA256SUMS` verifies, and protocol/evaluator
  bindings match.
- PR #137 artifact directory has no diff from `origin/master`.
- `git diff --check` passes.
- Independent Standards and Spec reviews report no actionable findings.

## Claims boundary

- Synthetic tests prove only the guard, sealing, replay, and one-shot seams.
- They do not establish a pristine live collection source or prospective model
  evidence.
- No collector is activated because the only verified Betfair monthly surface
  is result-bearing.
