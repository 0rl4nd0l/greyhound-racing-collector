# Controlled model comparison protocol — 3 October 2026

**Decision: prepare one paired comparison of the frozen production model at full strength (1.0) versus half strength (0.5). No refitting, parameter sweep or box change belongs in this first contrast.** This isolates a scoring-policy change on identical inputs; it does not establish a causal effect in racing or a performance improvement.

Status: **DESIGN_ONLY / ACTIVATION_OFF**. This note grants no provider, result, training, evaluation, scheduler or promotion authority. The [improvement preparation plan](/mnt/tenn-nvme2/tenn/greyhound-model-improvement-preparation-20261003/improvement-plan.PROPOSED.json) remains disabled. The original four-way scientific population, schedule, model registry, source pins and `CANARY_NOT_VERIFIED` disposition remain unchanged. Source inspected at `a6f861ea9277d215f51bd61d7d0b361506bd3b02`; no input records or outcome artifacts were decoded, and no scores were computed. Structural feasibility is established by code, not by a current count of qualified races.

## Why this is the first contrast

The production scorer already derives full and half predictions from one frozen base. Both use the same 16 features, fitted medians, means, scales, 32 coefficients including missingness indicators, within-race centering and residual cap 0.35. It computes a single capped adjustment and changes only its multiplier in the final softmax. The design therefore needs no fitted challenger artifact. [Feature order](../../src/predictor/market_form_residual.py:38), [one-base derivation contract](../../src/predictor/market_form_residual.py:145), [shared transformations and adjustment](../../src/predictor/market_form_residual.py:753), [full and half scoring](../../src/predictor/market_form_residual.py:796).

For each runner, retain the same normalized inverse-WIN-odds probability `m` and capped residual `r`; compare `softmax(log(m) + r)` with `softmax(log(m) + 0.5*r)`. Use the existing deterministic numerical canonicalization. The two alternatives must share every input and artifact binding; the variant identifier and strength are the only differences. [Numerical contract](../../src/predictor/market_form_residual.py:153), [market and variants](../../src/predictor/market_form_residual.py:793).

The existing `residual_half` candidate is **not** this production-half comparator. It uses canonical card-only history and its own fitted coefficients/preprocessing. Existing production merges earlier DB and embedded-card history; candidates use canonical history capped at 20, with different deduplication, context and missingness rules. `residual_box` versus `residual_half` also changes both box inclusion and strength. Preserve those frozen methods; do not relabel their contrast as an isolated strength test. [Candidate recipes](../../src/predictor/comparison_candidates.py:11), [candidate fitting](../../scripts/freeze_future_comparison.py:68), [production history](../../scripts/run_shadow_non_tgr_rf_evaluation.py:2429), [canonical history](../../scripts/build_form_only_v1_packet.py:503).

## Exact comparison contract

| Item | Required decision |
|---|---|
| Arms | Frozen production full `1.0` and the same frozen production base at `0.5`; one paired contrast only. |
| Fits / search | Zero fits, zero tuning trials, zero validation-selected strengths. No alternate cap or temperature. |
| Shared inputs | Exact same complete decision-time field, native dog/box identities, quote receipt, retained form, sidecar, sealed history, feature rows and preprocessing. |
| History | Replay the already sealed production feature contract. No augmentation, later repairs, deeper-history substitution or canonical-parser replacement. |
| Model bindings | One model SHA256, manifest SHA256, effective-state SHA256, feature order, fitted preprocessing and coefficient bytes for both arms. Pin the source revision and runtime too. |
| Market / timing | Same exact WIN receipt, captured 120–600 seconds before the verified jump; both arms sealed by jump minus 120 seconds. No later quote or result-dependent timestamp correction. |
| Numerical behavior | Existing canonical scalar scorer and normalization; positive finite probabilities summing to one, exact roster equality and deterministic replay. |
| Analysis direction | Half minus full for each prespecified loss; negative means lower loss. Market may remain a frozen descriptive reference, not another selectable challenger or primary contrast. |

The quote interval and pre-jump gate mirror the current comparison interface; they do not retrospectively qualify a new forecast. [Admission gate](../../src/predictor/future_comparison.py:150), [quote and shared-input verification](../../src/predictor/future_comparison.py:199).

## Membership, temporal boundary and retained evidence

Before any new scoring or outcome access, issue a separate immutable development allocation with an exact prospective calendar interval, race inclusion rule, source dates, authoritative reservation-manifest hashes and closure deadline. The rule must be independent of outcomes and must not select successful forecasts. Resolve all scientific, protected-history, engineering and weekend-pilot reservations before assigning membership; any conflict is an exclusion or HOLD, not borrowed allocation. Engineering collection authority alone does not create a research population. Freeze the allocation before its first eligible race; append each race's admission identity before its decision cutoff. No backdating or post-result enrolment.

Use previously retained qualified bundles only for separately authorized technical readiness/replay on exact reservation-cleared membership. Such replay is retrospective development evidence and cannot be presented as newly prospective confirmation. A prospective forecast requires its own timely admission and both durable pre-jump seals. The new comparison must not read protected outcomes or reuse an inspected cohort as fresh confirmation. The current design does not authorize even that retrospective scoring step.

An outcome-blind readiness manifest must enumerate each required role, hash, byte count, schema, capture/seal timestamp, qualification status and rejection reason, without exposing payloads:

- Race/source identity, official venue/date/race identity, verified jump and native dog-to-box roster, including explicit reserve substitutions and scratches; field hash and full roster agreement.
- Retained input manifest and completion/acceptance receipt; normalized form and sidecar; exact WIN odds receipt and capture artifact; collection-to-prediction chain and independent verifier receipt.
- Sealed earlier-history DB and history seal, target exclusion/cutoff basis, production feature rows and feature manifest, implementation manifest, frozen model/manifest/configuration and effective-state hashes.
- New allocation/admission hashes, two named arm receipts, sealed probabilities and shared-input digest, failure state and decision deadline. A result-ready queue identity is metadata only; results remain under separate authority.

The retained-input verifier binds model/config, exact receipt, form, sealed DB and feature rows together. History sealing excludes the target date and later rows. Admission must also check reserved/denied history intervals before materializing historical values. [Retained role bindings](../../src/predictor/retained_inputs.py:180), [history seal verification](../../src/predictor/on_demand.py:586), [pre-decode history gates](../../src/predictor/future_comparison.py:234).

## Denominators and terminal analysis

Keep an append-only ledger of all discovered opportunities in the fixed interval, reservation/eligibility exclusions, unattempted eligible races, consumed attempts, shared-input failures, one-arm failures, paired seals, independent verification, queued results, verified closures and unresolved quarantines. Reasons and counts must reconcile; no successful-only sample. Do not retry a consumed race or substitute a later race to repair a failed pair. Continue unrelated races only where there is no shared source, integrity or ownership problem.

Prespecify race-weighted paired log loss as the primary metric and multiclass Brier sum as a required supporting criterion, using one terminal analysis after the fixed endpoint and closure deadline. No weekly held-out scores or repeated significance checks. Freeze a date-block uncertainty method, whole-week sensitivity check, confidence level, practical improvement thresholds and decision rule before outcomes. These numerical/statistical settings and sample-size justification are **unresolved activation requirements**, not values to choose after seeing performance. Existing 40-date/12-week references are not a power guarantee for this population.

Report missing predictions and unresolved results against the full admitted denominator. Complete-case estimates, if authorized, are explicitly conditional/descriptive when closure remains unresolved. Do not silently assign zero loss, impute winners, treat a known non-finisher as an invented numeric placing, or discard troublesome identities. A prospective sensitivity/missingness policy must specify which terminal result statuses establish the target label, and which permit valid bounds; unresolved field identity blocks a broad claim. Numerical log-loss bounds require an explicit probability-bound contract; do not invent one or clip retrospectively. Exact minimum coverage and allowed missingness are further before-outcome decisions.

## Rejection, stop and rollback

Reject an individual pair for incomplete or conflicting roster/identity, invalid history dates/reservations, missing roles or changed hashes, stale/out-of-window quotes, late seal, invalid probabilities or failed replay. Preserve both consumed attempt and failure. A corrupted shared artifact, changed frozen source/config, unknown process ownership, unavailable allowance or provider denial is a shared HOLD/stop, following the existing owner and source controls.

The prospective run is ineligible for a confirmatory claim if its allocation or terminal analysis was chosen after outcomes, a held-out cohort informed tuning, required seals are late/missing, population accounting cannot reconcile, or its frozen missingness/coverage conditions fail. Do not weaken the protocol, restart an exhausted cohort, or rename retrospective records to repair it.

Deployment rollback is not needed for a design-only note. If later authorized, run the alternate arm in an isolated research namespace with no production pointer change. On failure, disable new comparison admission, drain owned work, preserve seals/counters/quarantines, and leave the frozen production/scientific routes intact. Any production replacement requires separate review of a concrete versioned artifact and rollback package; a favorable comparison alone cannot promote it.

## Separate later question: box features

Box inclusion needs matched fits: same authorized training races and chronological folds, raw history construction, 16 baseline features, target definition, optimizer/regularization, missingness/preprocessing recipe, strength and cap. One arm omits box and the other adds the verified native box. Fitted preprocessing and coefficients will necessarily differ where the feature space changes; those differences must be logged rather than described as a fixed-coefficient experiment. Freeze both artifacts before admitting the shared prospective evaluation cohort. No fits are authorized here.

The existing box feature is a single `float(box)` with one linear coefficient before residual saturation. It is ordinal, not separate effects for boxes or track layouts. Replacing that scalar with categorical encoding is a **different** matched-fit contrast, requiring a prespecified vocabulary, reference/identifiability rule, missing/unknown handling and per-box training support. Do not combine box inclusion, encoding, venue interactions and strength in one claimed ablation. [Literal box](../../src/predictor/comparison_candidates.py:94), [scalar scoring](../../src/predictor/comparison_candidates.py:119).

Before any such fit, freeze the population/training rights, temporal folds, training cutoff, leakage firewall, feature contract, finite attempted-fit ceiling, compute limits and rejected-fit policy. Test folds stay untouched during fitting/selection; preprocessing is fitted on training data only. Any later confirmation uses new prospectively allocated races. This is an association/predictive comparison, not proof that changing a dog's starting box causes the estimated probability change.

## Remaining decisions before activation

1. Verify exact installed-release live acceptance and private identity-verified result closure; neither is asserted by this source-only audit.
2. Freeze separate, reservation-cleared membership and inspect allowlisted retained metadata to count currently qualified paired inputs. No current cohort size, box coverage or history support is asserted here.
3. Bind the exact production artifact/effective-state, feature, implementation and runtime hashes from the authorized registry and verified retained manifests. Preserve existing scientific pins; issue new research references separately.
4. Freeze calendar endpoint, closure authority/deadline and finite budgets, statistical settings, practical thresholds, sample justification and missingness policy. Do not extend source or result authority implicitly.
5. Implement and independently test the isolated paired-admission/sealing/queue path only under subsequent execution authority, then issue its explicit activation receipt. Production replacement remains separately reviewed.

The first contrast is deliberately finite: two variants, one unchanged frozen base, one prospective allocation, one terminal paired analysis, no refit and no broad search. Box, history-depth, form-only, boosted and pace investigations are not extra arms of this protocol.

## Metadata readiness audit — 3 October, 16:27 AEST

The [immutable readiness manifest](controlled_model_readiness_20261003/manifest.json) fixed exactly the 15 October 2 live07 engineering admissions and 15 completion-bound bundles before metadata projection: 233 allowlisted metadata/source files, 1,888,309 bytes per pass, at most three passes, a 16 MiB pass ceiling and a 17:06 AEST deadline. This is retrospective technical readiness, not research enrolment. The [report](controlled_model_readiness_20261003/readiness-report.json) verifies metadata bindings for **15/15**: declared roles, checked metadata hashes, retained roster/hash-reference equality, original implementation pins, shared production model and timely pre-jump publication. Publication lead ranges from 516.17 to 571.63 seconds. The same frozen model manifest explicitly supports full strength 1.0 and half strength 0.5 from one base; no new paired scores or seals were created.

The producing source is `b0534293fb48bc1204bb02b5bbf677d6c95ca9ab`. Its feature-implementation hashes match all 15 bundles. The present research worktree differs in `utils/csv_metadata.py` and `utils/race_lifecycle.py`; those differences are preserved, and current code was not substituted. The [initial audit draft](controlled_model_readiness_20261003/initial-hash-interpretation-failure.json) is retained: it mistakenly applied the receipt-roster hash formula to a sealed race-plus-roster hash, and digested a partial source-identity projection. The corrected report compares original pinned hash references and source-identity bytes. It does not change evidence or widen the frozen field allowlist.

**Remaining unknowns:** native-ID-inclusive roster-hash reconstruction was not possible under that allowlist; official reserve identity was not independently rechecked. Opaque form/history/features and retained archive contents were not decoded or rehashed; their declared bindings, presence and sizes were checked. No probabilities, result bodies, coefficients or preprocessing values were evaluated, and no model replay was performed. The separate October 3 admission root was absent at 16:27:43 AEST, giving a point-in-time filename count of zero; this neither changes the old cohort nor predicts future admissions. Fitting, scoring, evaluation and activation remain off.
