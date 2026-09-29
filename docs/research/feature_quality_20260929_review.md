# Independent review and validation

Fixed point `bec84e8074dacf9c32330f040a67dc46919f3d88` (#197), initial
implementation `530c496d`. Diff: `git diff bec84e80...HEAD`. Separate agents
reviewed Standards and Spec using the code-review skill. Standards came from
the applicable AGENTS guidance, CONTEXT.md and pyproject.toml; the user's direct
task and `feature_quality_20260929_plan.md` supplied the spec. No issue-tracker
installation was needed for this directly supplied task.

## Standards

No documented-standard violations. One optional maintainability finding: a
positional `zip` mapped legacy names to development names, so future reordering
could silently mislabel inputs. Replaced it with an explicit keyed mapping and
an exact order check. This was a future-drift precaution, not an observed
mislabelled development value.

## Spec

One P2 finding: inherited integer conversion could silently truncate `PLC=1.9`
into a win or `DIST=400.9` into a 400m match; zero distance could yield a false
"no matches" status. A focused regression reproduced all four cases (including
nonfinite distance) before the repair. V2 now validates numeric prior-date raw
finishes/distances before canonical conversion. Non-numeric unavailable fields
remain missing. The original/frozen parser is unchanged.

The Spec reviewer independently rechecked zero, fractional, negative and
nonfinite fixtures after repair and verified expected values on a valid record.
No actionable Spec findings remain. Replaying all admitted cards after the fix
still gives zero changed numerical values or rejected development runners.

Summary: Standards 0 hard violations / 1 optional finding resolved; Spec 1
correctness finding resolved. These reviews make no production/live claims.

## Validation evidence

- Initial feature-interface test failed before the new module existed, then
  passed after implementing unknown-versus-zero context behavior.
- Fit-receipt test failed before its module existed; its first implementation
  exposed a list/array mismatch at legacy validation, fixed by supplying the
  required numeric array. Synthetic saved-model replay then passed.
- Final focused command in the results report: **15 tests passed**. It covers
  synthetic fit replay/failure retention and identity/roster gating as well as
  feature meaning. No development model fits were performed.
- Initial audit remains locally under `feature_quality_20260929_evidence/`.
  The pre-review `_evidence_verified/` artifacts remain committed at the first
  implementation. `_evidence_reviewed/` is final; the new guards leave summary
  counts and runner numeric values unchanged.
- Final fresh replay at `/tmp/feature-quality-reviewed-replay-yecow1y_/evidence`
  matched all five final artifacts byte-for-byte. Reproduction requires a new
  output directory and the existing pinned local authorized inputs.
- All 27 baseline source/control/model/unit files still matched their hashes
  after review. All 37,760 legacy feature values matched; saved base16/half
  forecast reconstruction error remained at most `2.220446049250313e-16`.
- `git diff --check` passed. No broad tests, provider requests, database opens,
  protected records, installed service writes or October amendments occurred.
