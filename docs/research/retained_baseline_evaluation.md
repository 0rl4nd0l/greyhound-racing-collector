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

## One separately versioned label-provenance evaluation

A successful partial corrected evaluation consumes corrected-01. It cannot be
retried with changed labels. An optional, explicit `label_provenance_successor`
branch permits one new version while preserving the exact original membership
reference (the deployed proposal has 82 members), forecasts, policy and result
cutoff. It creates only `evaluation_claim.label-provenance-v1.json` beside that
same membership. The original and corrected claims and their outputs remain
untouched; a copied membership directory, chained label version, simultaneous
correction mode, or already existing label claim is rejected.

The new authority retains the ordinary schema/expiry/privacy/finite-limit
requirements. It pins a changed reviewed adapter with unchanged other baseline
implementation dependencies and adds `label_provenance_successor` with exactly:

- `schema_version=baseline_label_provenance_successor_v1`.
- `original_claim`, `original_authority`, `original_status`.
- `corrected_claim`, `corrected_authority`, `corrected_status`, `corrected_metrics`.
- `label_provenance`, described below.

Both predecessor claim/authority/status linkages and timestamps are checked.
The original status must be `FAILED_PRESERVED_CLAIM`; the corrected status must
be `PRIVATE_BASELINE_COMPLETE` and point to the pinned private metric artifact.
That artifact is hashed opaquely, never decoded by authorization. A fresh output
must be disjoint from both old outputs. Failed execution consumes this new fixed
claim; authority/output reissue cannot retry it.

`label_provenance` has `schema_version=baseline_label_provenance_v1` plus exact
file references named `original_closure_manifest`, `revalidation_authority`,
`revalidation_claim`, `revalidation_status`, `proposed_manifest`, `proof_bindings`,
`verification_authority`, `verification_claim`, `verification_status`,
`verification_source`, `verification_helper`, and `independent_review`. A source
reference means a pinned source-identity receipt, not a directory. All are
hash-checked; opaque proofs, helpers and source references are not interpreted
as result values. The verifier owns the independent source-identity replay.

The separately sealed closure uses `sealed_baseline_closure_manifest_v1` and a
`baseline_closure_verification_v1` receipt with the identical `label_provenance`
object. Its complete ordered record list must preserve the original denominator.
The revalidation status must be `RETAINED_IDENTITY_REVALIDATION_COMPLETE`; the
verification terminal status must be `RETAINED_IDENTITY_VERIFICATION_COMPLETE`
and bind `membership`, `records_sha256` and `result_cutoff`. The verifier writes
that terminal status last, so an incomplete publication is unusable.

The proposal must retain `status=REQUIRES_INDEPENDENT_VERIFICATION`,
`evaluation_authority=false`, original denominator and every original CLOSED
candidate (73 in this deployment). For each `IDENTITY_REVALIDATED` proposal entry,
only the original record's `evidence` and `bytes` may be replaced with the exact
new database reference and size. A failed entry must become exactly
`{race_id, state:QUARANTINED, prior_state_preserved:CLOSED, reason:<proposal status>}`.
Original known-nonfinish and quarantine records must stay byte-equivalent as
JSON values. Neither an existing CLOSED label nor a proposed repair alone grants
eligibility. The unchanged per-race reader still checks native identity,
complete result semantics and unique/equal-mass winners before metrics.

This code adds no authority or completed repair evidence. The original 82
membership, 73 candidate repairs, one known non-finisher and eight quarantines
remain the operational scope; focused fixtures use the same three state classes
without opening those protected artifacts. Pre-claim metadata/proof validation
uses fixed reference fields and the existing bounded two-MiB file reader;
post-claim artifact operations retain the existing authority budget. Root must
include that fixed bootstrap workload when calculating the finite process and
resource limits. All actual historical evaluation remains a separate root action.
