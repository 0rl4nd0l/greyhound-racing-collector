---
job_id: GHU_053_SEALED_EVIDENCE_RESEARCH_SCORING_V1_20260805
title: Implement GHU-053 sealed-evidence feature building and research scoring
base: 492500684fb017b29c3af9748b00e9af8505b457
base_tree: c16f7acff89fd0e85afe28719f3319eb54154bdd
production_data_access: false
browser_or_network: false
canonical_db_or_history: false
phase7: excluded
---

# GHU-053 refreshed implementation pack

This ticket starts only from the verified GHU-052 merge `492500684f…` and
tree `c16f7acf…`. GHU-052 remains the sole sealed-evidence reader. The
prediction lane is research-only, non-canonical, and excluded from Phase 7.

## Verified reuse surface

- `src/predictor/manual_independent_capture_sealer.py::verify_manual_evidence_bundle`
  validates the final hash-named GHU-052 directory, manifest, bundle, ordered
  runners, normalized odds, raw-source hash, timing, cleanup, and safety
  closure. GHU-053 consumes its returned `SealedManualEvidence` only.
- `src/predictor/market_form_residual.py::load_frozen_model` and
  `::score_race` are the frozen hash-pinned loader/scorer. They are pure after
  model loading and reject non-finite values, runner disagreement, post-jump
  timestamps, and outcome-shaped inputs. GHU-053 does not call its history
  writer or any autonomous/canonical reader.
- `src/predictor/manual_independent_capture.py` and the four GHU-052 schemas
  remain the authority and path contract. No GHU-050/051/052 rule is widened.

## GHU-053 input contract

The scorer receives an in-memory canonical JSON document using
`embedded-form.schema.json`; it does not receive a form path. Its hash is
supplied by the caller and bound into the prediction. The document carries
the exact GHU-052 race identity, source timestamp, sealed cutoff, target
distance/grade, exact ordered runners, and prior form rows. Each prior row has
an explicit event timestamp strictly before the sealed cutoff and a prior
race date strictly before the target race date. `prior_finish` is historical
form data used by the approved feature definitions; target-race result fields,
result sources, canonical history, and outcome artifacts are not admitted.

The adapter computes the approved 16 nullable features in the frozen model
order: prior count, recency, recent finish/margin summaries, career rates,
venue/distance/grade counts and win rates. Missing historical form is
represented by nullable feature values for that runner; an absent or malformed
form document fails closed. The feature hash covers the canonical form hash,
cutoff, target context, adapter version/identity, and ordered feature rows.

## Output and authority

`research-prediction.schema.json` and
`research-prediction-manifest.schema.json` define a deterministic two-member
read-only bundle. Publication writes a fsynced staging directory and renames
it atomically beneath the caller-owned isolated output root. The final
directory is named by a deterministic bundle hash; identical inputs replay
the exact bytes. Ranking uses full probability descending and box ascending
as the explicit tie-breaker. The output has finite market/half/full
probabilities and no EV, staking, betting, result, or Phase 7 fields.

The output binds the race, ordered runner set, odds and cutoff, GHU-052 bundle
and manifest hashes, embedded-form hash, feature hash, model/manifest/effective
state hashes, config hash, and the refreshed implementation/schema identities.
The read-only verifier rejects unsafe paths, symlinks, partial/extra members,
tampering, identity drift, non-canonical JSON, non-finite values, and ranking
or probability disagreement.

## Stop conditions and claims

Stop on ambiguous form provenance, any history/result/canonical/Phase-7 access,
late rows, model/config drift, non-determinism, or any need to weaken GHU-050,
GHU-051, or GHU-052. Acceptance supports only a fixture-proven sealed-
evidence research prediction bundle. It does not support live readiness,
model quality, market edge, EV, deployment, training, promotion, or betting.

GHU-054 remains the next prerequisite and is not started by this ticket: a
separately reviewed CLI/API/UI adapter may call the accepted GHU-053 scorer,
but it must not add capture, discovery, network, runtime, or canonical
authority.
