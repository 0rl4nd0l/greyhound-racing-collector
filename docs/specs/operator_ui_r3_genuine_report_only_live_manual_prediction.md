---
title: Operator UI R3 genuine report-only live manual prediction
status: ready-for-agent
---

## Problem Statement

The operator cannot yet produce and trust one genuine report-only live manual prediction through Operator UI R3. The currently running service is bound to a rejected source candidate that omits later accepted R3 safety work, while the authoritative repository lineage and durable deployment generator exist on a different commit line. Earlier diagnosis also conflated a legacy Flask scikit-learn model warning with the R3 frozen JSON model contract. A successful-looking response is not sufficient: the result must come from the authoritative R3 job lifecycle, consume fresh provenance-bound pre-jump evidence without prediction-time fetching, preserve the canonical racing database, and leave an immutable audit and prediction bundle.

## Solution

Replace the disqualified active R3 runtime with a candidate generated from a clean isolated checkout of the exact locally verified authoritative `origin/master` commit and tree. Validate source identity, fixed artifacts, deployment generation, R3 safety, and deterministic frozen-model behavior under the pinned runtime before installation. Generate a fresh enabled-package live-authority observation from contemporaneous local evidence, verify the package, then—under the granted narrow runtime authority—replace, reload, and restart only the Operator UI R3 service.

After the accepted service is healthy, wait for a fresh collector-owned current-race index and a matching fresh sealed receipt. Submit exactly one manual job with the latest-research selector resolving to the frozen market-form residual model, the checked-in manual-default configuration, and receipt-only odds. Permit only the required R3 operations-store and immutable-bundle writes. If any gate or the live job fails, stop and report the exact evidence without retry, capture, fallback, EV, or betting.

## User Stories

1. As a Greyhound operator, I want the deployed R3 service to identify its exact source commit and tree, so that I know which code produced a prediction.
2. As a Greyhound operator, I want R3 deployments generated from the authoritative master lineage, so that accepted safety work is not omitted.
3. As a Greyhound operator, I want installed snapshots treated as deployment evidence rather than source, so that obsolete runtime copies do not become a competing source of truth.
4. As a Greyhound operator, I want rejected candidates retained as evidence but excluded from prediction acceptance, so that process health cannot rehabilitate invalid lineage.
5. As a maintainer, I want deployment generation to use a clean isolated checkout, so that unrelated feature work and untracked files remain untouched.
6. As a maintainer, I want the durable repository-owned generator restored to the deployment path, so that installed units are never hand-edited as the durable fix.
7. As a reviewer, I want exact fixed artifact hashes bound into the candidate, so that configuration, predictor, schema, manifest, and model substitution fail closed.
8. As a reviewer, I want a fresh live-authority observation for each enabled candidate, so that stale service and evidence observations cannot be reused.
9. As a reviewer, I want the live-authority observation to record the real observation time, so that its provenance is auditable.
10. As a reviewer, I want candidate generation to reject missing, contradictory, unsafe, or changing authority inputs, so that partial packages cannot masquerade as valid deployments.
11. As a maintainer, I want the generated unit verified before installation, so that invalid systemd configuration never reaches activation.
12. As a maintainer, I want all generated package outputs and hashes reviewed, so that the installed unit matches the accepted candidate.
13. As a Greyhound operator, I want only the R3 service replaced and restarted, so that the autonomous collector and unrelated services remain undisturbed.
14. As a Greyhound operator, I want the existing rejected runtime left running until its replacement is validated, so that validation does not create unnecessary downtime.
15. As a model reviewer, I want R3 compatibility judged against its own frozen model contract, so that irrelevant legacy Flask dependencies do not block or falsely approve it.
16. As a model reviewer, I want deterministic fixed-fixture and portability replay under Python 3.11.15 and NumPy 1.26.4, so that inference compatibility is demonstrated rather than inferred from warning-free loading.
17. As a Greyhound operator, I want the latest-research selector to resolve to the expected frozen market-form residual model, so that registry selection is explicit and verified.
18. As a Greyhound operator, I want a fresh collector-owned v2 current-race index, so that the selected race is genuinely upcoming.
19. As a Greyhound operator, I want the index race identity, jump time, and runner-set hash to match the prediction input, so that stale or changed races fail closed.
20. As a Greyhound operator, I want a fresh verified sealed receipt matching the selected race, so that prediction consumes genuine pre-jump market evidence.
21. As a Greyhound operator, I want the prediction forced to receipt-only odds, so that it cannot initiate external fetching.
22. As a Greyhound operator, I want missing or stale receipt evidence to block prediction, so that automatic capture cannot silently substitute for readiness.
23. As a data custodian, I want the canonical racing database opened read-only, so that report-only prediction cannot mutate runtime history.
24. As a data custodian, I want prediction to construct only a cutoff-filtered sealed history database inside its immutable bundle, so that feature evidence is reproducible and leakage-safe.
25. As an auditor, I want the manual job lifecycle persisted in the separate operations store, so that the genuine job is distinguishable from a direct script or legacy endpoint response.
26. As an auditor, I want an append-only job audit trail and immutable prediction bundle, so that the result can be independently verified.
27. As an auditor, I want the final bundle to bind model, configuration, receipt, protocol, source, runner, history, feature, and prediction identities, so that a success claim has complete provenance.
28. As a Greyhound operator, I want exactly one preflight-qualified live attempt, so that acceptance work does not become an uncontrolled retry loop.
29. As a Greyhound operator, I want a blocked or failed attempt reported exactly as observed, so that pipeline failure cannot masquerade as a genuine prediction.
30. As a safety reviewer, I want no fallback model, EV calculation, or betting action, so that the acceptance run remains strictly report-only.
31. As a safety reviewer, I want no training, model promotion, external fetch, or canonical database mutation, so that prediction proof does not broaden operational authority.
32. As a repository owner, I want every existing tracked and untracked file preserved, so that deployment work does not absorb or destroy unrelated work.

## Implementation Decisions

- The locally verified `origin/master` commit `d343af94a57af80327dd41f18433f7466f86ca0d` and tree `a04197ba455de2549f9c76adcd474d1feb520bd1` are the authoritative R3 source baseline for this work.
- The installed `e4f36999` snapshot is historical deployment evidence. The active `7b614325` candidate is rejected because it is based directly on that snapshot with legacy Flask changes and omits the later R3 lineage.
- Legacy Flask hardening commits remain on their separate lineage and are not grafted onto the R3 replacement.
- Candidate generation uses a clean, isolated deployment source. The divergent feature worktree and all existing untracked files remain untouched.
- The repository-owned finite deployment generator is the only acceptable source of the binding, environment, unit, and rollback outputs. Generated or installed files are not hand-edited.
- The enabled candidate binds a newly constructed, contemporaneous live-authority observation. It explicitly names all required reports, refresh evidence, inventory/raw packets, installed units, and actual full/odds service states.
- R3 runtime compatibility is Python 3.11.15, NumPy 1.26.4, and deterministic fixed-fixture replay for the hash-bound JSON model. scikit-learn and SciPy versions used by other interfaces are not R3 compatibility criteria.
- Candidate installation is permitted only after all agreed pre-activation gates pass. Installation replaces only the Operator UI R3 unit; daemon reload and restart apply only to that service.
- The autonomous odds collector is observed but never stopped, restarted, edited, or otherwise controlled by this work.
- A Genuine Report-Only Live Manual Prediction is an R3 manual job, not a legacy Flask response or direct synthetic invocation.
- Race admission requires a fresh verified collector v2 current-race index. No index is edited, synthesized, copied, or bypassed.
- Prediction requires a matching fresh sealed receipt and forces the receipt odds source. Automatic and synchronous capture are prohibited.
- The canonical racing database is read-only input. A filtered sealed-history database may be created only inside the new prediction bundle.
- Writes are limited to the separate R3 operations store: required job-state transitions, append-only audit evidence, and immutable bundle publication.
- The single job uses the latest-research selector resolving to `market_form_residual_v1` and the checked-in manual-default configuration.
- One preflight-qualified attempt is authorized. Failure or blocking terminates the effort without automatic retry or fallback.
- The implementation must preserve the endpoint hardening already present on the separate feature lineage, but that hardening is neither deployed with nor used to qualify R3.

## Testing Decisions

- Tests assert externally visible acceptance and rejection behavior rather than private implementation structure.
- The highest pre-activation seam is the existing repository-owned deployment generator. Its tests cover exact clean identity, fixed authority, enabled live evidence, transactional output, safe paths, and fail-closed rejection.
- Existing Operator UI R3 end-to-end safety tests cover the composed job, audit, worker, and bundle behavior without introducing a new production seam.
- Existing frozen-model fixed-fixture and portability tests prove deterministic scoring under the pinned runtime and verify the model record remains pinned.
- Candidate validation records the exact source commit/tree, runtime versions, fixed artifact hashes, test commands and results, four generated output hashes, and systemd verification result.
- Post-activation smoke verification checks the installed unit and process identity, loopback listener, authenticated R3 health/read behavior, and continued availability of unrelated services without triggering a prediction.
- Live acceptance occurs at the existing R3 manual-job boundary. Preflight verifies one fresh current-index race and its matching receipt; the resulting job, audit records, and immutable bundle prove success or explain failure.
- A valid success must contain finite, structurally valid runner predictions and a fully verified sealed bundle. Empty, malformed, exceptional, or downstream-failed results remain failures.
- Unchanged narrow checks are not repeated. Any failed gate stops progression to the next operational stage.

## Out of Scope

- External data fetching initiated by the prediction workflow.
- Starting, stopping, restarting, or editing the autonomous odds collector or any unrelated service.
- Mutation of the canonical racing database or collector evidence.
- Training, refitting, evaluating for promotion, promoting, or changing a production model pointer.
- EV calculations, staking, bet construction, or betting actions.
- Deployment of the legacy Flask endpoint hardening commits as part of R3.
- Retrying a blocked or failed live manual job.
- Pushing, merging, publishing code, creating commits, or modifying remote issue trackers without separate authority.
- Deleting rejected packages, old snapshots, operations evidence, or existing tracked/untracked files.
- Public exposure or port forwarding of the Operator UI service.

## Further Notes

- During discovery, the R3 unit changed from failed to active without action by this workflow. Its active state is evidence only and does not qualify the rejected source lineage.
- Fresh sealed receipts were observed while the collector was running, but the then-observed current-race index was stale. Receipt freshness alone is not readiness; the collector-owned index must also be freshly published and matching.
- The execution environment that produced this spec cannot write outside its worktree or access the remote issue tracker. Deployment and tracker publication require an appropriately authorized session.
- This spec is labeled `ready-for-agent` locally so it can feed `/to-tickets` once the tracker boundary is available.
