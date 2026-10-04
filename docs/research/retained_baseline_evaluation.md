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
