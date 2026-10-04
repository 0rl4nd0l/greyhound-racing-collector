# Retained baseline evaluation (default off)

This adapter evaluates the four **original** stored forecasts on the same race
set. It does not regenerate forecasts, fit models, modify original scientific
membership, consume the original terminal evaluation claim, or acquire data.
It is descriptive retrospective analysis, not prospective confirmation.

Preparation and execution are distinct public seams:

1. `freeze_membership(protocol_ref, journal, through_date, output, now)` copies
   an immutable hash-chain-verified observer journal, selects every member by
   original racing date through the cutoff, and retains all journal event counts
   and original denominator references. It opens admission metadata only. No
   result status is used for membership. Repeating a destination fails.
2. `run_baseline(membership_ref, authority_ref, execute=False)` returns
   `DEFAULT_OFF` without opening paths or writing anything. Execution requires
   the separate schema below, then creates a one-time claim beside the new
   membership manifest. Changing authority/output cannot erase that claim.
3. `join_forecasts` verifies all four stored records against the original
   admission, same request roster/native-field hash, input identities, model
   artifact hashes and pre-cutoff timestamps. Models/features are never replayed.
4. `win_target` and `summarize` produce common-race win targets and race-weighted
   mean log loss / multiclass Brier sum. Explicit verified `FELL`, `DNF`, or
   `DISQ` can be win-label eligible with complete starter identity and one unique
   winner; no numbered non-finisher placement is invented. A dash is unknown.
   Complete full-order dead heats use equal mass over explicit co-winners.

The proposal prepared on October 4 freezes 82 October 1–3 observer members,
with all 17 original consumed attempts preserved. Metadata describes 73
full-order candidates, one known non-finisher and eight quarantines; these
are not yet an executable model/result join or a performance claim.

## Required execution artifacts

All references are exact absolute `{path, sha256}` pairs. The root-issued
`private_retained_baseline_authority_v1` must contain:

- `status=AUTHORIZED_ONE_SHOT_PRIVATE_BASELINE`, nonempty `authority_reference`
  and `evaluation_id`, `performance_evaluation=true`.
- Exact `membership`, `policy=RETROSPECTIVE_COMMON_FOUR_WIN_LOGLOSS_BRIER_V1`,
  and `implementation_files` equal to `implementation_pins()` from the final
  reviewed candidate.
- Aware `issued_at`, `expires_at`, `result_cutoff`; cutoff is no later than issue,
  issue is no later than execution, execution is before expiry.
- `provider_requests=0`, `result_requests=0`, and false `training`, `promotion`,
  `human_outcome_access`, `public_performance_outputs`.
- New isolated `output_root`, exact `closure_manifest`, and positive finite
  `limits`: `max_races` equal to the frozen denominator, `max_files`, `max_bytes`,
  `max_wall_seconds`. Resource limits are bound from measured manifest workload;
  this document issues no allocation or arbitrary campaign ceiling.

The protected closure manifest has schema `sealed_baseline_closure_manifest_v1`,
status `SEALED_INDEPENDENTLY_VERIFIED`, exact `membership`, `verification_receipt`
and exactly one record per frozen race, including unresolved races. States:

- `CLOSED`: `evidence` is a sealed official-result SQLite snapshot; `bytes` is
  pinned. The existing `ComparisonResultSource` checks field, identity, official
  source, timing and complete ranking. Live WAL/SHM/journal snapshots are refused.
- `CLOSED_NON_FINISH`: `evidence` is the original independently verified
  `comparison_known_nonfinish_result_v1` receipt, retaining body/request/response
  references. The adapter rechecks its content hash, timing, native starter
  identity, explicit terminal status and unique winner, plus exact retained HTTP
  evidence. It does not reclassify an existing quarantine.
- `QUARANTINED`, `PENDING`, `UNRESOLVED`: preserved exclusions; no result source
  for that race is opened, and the race stays in the denominator.

The separately pinned verification receipt has schema
`baseline_closure_verification_v1`, `independent_identity_verification=true`,
exact `membership`, `result_cutoff`, and SHA256 of canonical `records`. Root
must reconcile this mapping to original closure records before issuing it;
the adapter does not issue or self-certify the receipt. Building this exact
historical snapshot mapping and obtaining evaluation authority remain activation
requirements. Existing expired closure permissions are not revived.

A missing native race-entry ID or identity ambiguity remains unresolved even
if a prior broader metadata summary says CLOSED. Do not invent a bridge from
a dog-profile ID to a race-entry ID. The original producing implementation and
feature manifests remain part of each member's evidence.

## Invocation and privacy

```
python -B -m scripts.evaluate_retained_baseline
python -B -m scripts.evaluate_retained_baseline --freeze-membership \
  --protocol ABS --protocol-sha256 SHA --journal ABS \
  --through-date 2026-10-03 --out NEW_ABS
python -B -m scripts.evaluate_retained_baseline --execute \
  --membership ABS --membership-sha256 SHA --authority ABS --authority-sha256 SHA
```

The first command performs no reads/writes. The second prepares metadata; it
cannot authorize evaluation. The third is for root only after review/authority.
Run in a finite process with networking denied and original evidence read-only.

All metric values go exclusively to `private_metrics.json` (0600 in a 0700
directory). CLI/status outputs contain only completion, counts, references and
hashes. Failures retain the one-time claim and a generic failure status, never
raw protected exception content. No original claim, journal, model, source
ledger, service or provider allowance changes.

Controlled production full-versus-half derivation remains a separate task.
The existing `residual_half` candidate has different preprocessing/history/
coefficients and is not an isolated strength comparison. No such derivation
is implemented here.

## Documented comparison-row enrichment

`future_comparison.py` writes `comparison/inputs.json` runners by adding
`win_odds` to each canonical sealed roster row. The request retains the four
canonical keys only: `box_number`, `display_name`, `identity`,
`source_native_runner_id`. The baseline requires **exactly** those four keys
plus `win_odds` in every comparison row; odds must be a finite numeric decimal
strictly greater than one, excluding booleans. It removes only that known
extra key before invoking the unchanged canonical roster and sealed field-hash
validators, and comparing the exact ordered request roster. Unknown additions,
missing odds, changed native ID/name/identity/box, duplicate or reordered
participants remain rejected. No stored probabilities or forecasts are rebuilt.
The fabricated end-to-end reader fixture now uses the native producer's actual
five-key comparison/four-key request shape.

## One failure-linked corrected execution

The first execution and its `evaluation_claim.json` remain consumed. The
specific pre-metric `baseline_request_field_changed` implementation failure can
support **one** corrected attempt only, with a freshly issued ordinary
`private_retained_baseline_authority_v1`. It adds:

```json
{
  "corrected_attempt": {
    "schema_version": "baseline_corrected_attempt_v1",
    "predecessor_claim": {"path": "ABS/evaluation_claim.json", "sha256": "SHA"},
    "predecessor_authority": {"path": "ABS/authority.json", "sha256": "SHA"},
    "predecessor_status": {"path": "OLD_OUTPUT/status.json", "sha256": "SHA"},
    "failure_diagnosis": {"path": "ABS/root-failure-diagnosis.json", "sha256": "SHA"}
  }
}
```

The exact membership path/hash, closure manifest, policy, historical cutoff,
permissions and finite limits must be unchanged. New issue time must follow the
original claim; a new evaluation ID and disjoint output directory are required.
Only the baseline adapter implementation pin may change; all other pinned reader,
model and original-input dependencies remain fixed. Model/input references are
also retained through the identical membership and bundle seals.

The predecessor claim must be the original fixed filename in the **same**
membership directory, pin the predecessor authority and its output, and have no
correction linkage. The terminal status must be exactly
`FAILED_PRESERVED_CLAIM` for that membership at the predecessor output's
`status.json`. `private_metrics.json` must not exist, even as a dangling symlink.
No claim copying, new membership freeze, successful predecessor, changed closure,
chained correction, output overlap or implicit ordinary retry is accepted.

The root-issued diagnosis schema is `baseline_failed_execution_diagnosis_v1`,
status `AUTHENTICATED_IMPLEMENTATION_FAILURE_BEFORE_RESULT_OR_METRIC_READS`,
with failure code `baseline_request_field_changed`, `result_reads=0`,
`metric_calculation=false`, `metric_artifact_exists=false`. Its `original_claim`,
`original_authority`, `original_terminal_status`, membership, closure manifest,
policy, result cutoff and private output root must match the exact predecessor;
`created_at` is between the original claim and new authority issue. This uses the
existing root metadata receipt without reading private diagnostic values.

After these checks and another deadline check, exclusive creation of
`evaluation_claim.corrected-01.json` in the original membership directory
consumes the only correction, storing all linkage. It remains consumed on failure
or ambiguous termination. The original claim, authority and output are never
rewritten. Root should mount the original claim read-only within the otherwise
writable claim directory. No correction is executable until root issues the new
exact implementation authority; the original authority stays frozen.
