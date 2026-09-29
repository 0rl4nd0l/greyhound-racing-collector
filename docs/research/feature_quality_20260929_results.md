# Retained history quality: decision and offline implementation

**Prioritize better-defined, source-qualified history before another model search.**
All 331 admitted development races / 2,360 runners were replayed from their
qualified cards. A separate `retained_card_form_v2` development interface now
states what each feature actually observes, preserves its denominator, and
distinguishes unavailable context from observed zero matches. This is an
information-quality improvement, not evidence of improved predictive accuracy.
There were **zero new development fits**, zero provider requests, zero database
opens and zero protected target/history decodes. The frozen October comparison,
production models, installed services and acquisition policies are unchanged.

## Three findings, ranked

1. **Short source history and route differences dominate interpretation.** Every
   admitted runner has only 1–5 retained starts; 2,219/2,360 have exactly five.
   Recent-five and “career” win/top-three rates coincide for all 2,360 runners
   because both formulas receive the same records. The 20-start parser cap never
   fires. This is incomplete retained coverage and misleading naming, not a
   copied-column defect or evidence of complete careers. Production uses merged
   DB/card rows without this cap, ±50m distance and different rate denominators;
   canonical candidates use card-only exact-distance histories. Those method
   differences prevent treating this audit as production-input parity. Evidence:
   direct admitted-card replay plus inspected production/candidate source.
2. **Missingness mostly means absent matching retained experience.** There are
   missing base inputs in 306/331 development races and 159/177 evaluated races.
   All audited context fields and finishes are present, so missing venue,
   distance and grade win rates arise from no matching retained starts, not
   parsing failure. After race centering, missingness coefficients contribute
   to 1,033/1,251 evaluated runners, including some with complete raw inputs.
   This is a source-coverage limitation; indicators are part of the intended
   model. Separately, legacy unknown-target context counts become zero: a
   confirmed missing-value implementation defect exercised by synthetic tests,
   affecting **zero** current admitted values. V2 returns null for unavailable
   target context, incomplete retained context, or absent retained history.
3. **A useful second WIN observation requires real acquisition.** The installed
   policy consumes the race across windows. DOM readiness polls span at most
   five seconds and stop at the first complete field. WIN/PLACE pairs represent
   two markets at one time. Existing persisted quotes are append-only; there is
   no qualified earlier WIN stream being overwritten. Evidence: installed
   source and approved configuration, not a new live capture. A second
   observation needs a new operational policy, budget and population decision.

The [16-feature inventory](feature_quality_20260929_inventory.md) traces formulas,
units, truncation/deduplication, all source routes, missing/zero meanings and
preprocessing. Two further cautions matter: 1,608 **repeated historical
observations** record a winner with positive `MGN`, so this cannot safely be
called mean distance beaten; grade-label equality lacks jurisdictional ability
equivalence and venue aliases can pool layouts. No sign conversion, grade
equivalence, missing histories or layout corrections were invented. A possible
production DB/card deduplication limitation is documented but its incidence
cannot be quantified without additional qualified merged inputs.

## What changed and how much

| Quantity | Full development | Existing chronological evaluation |
|---|---:|---:|
| Races / runners | 331 / 2,360 | 177 / 1,251 |
| Retained history lengths 1 / 2 / 3 / 4 / 5 | 31 / 43 / 25 / 42 / 2,219 | 16 / 23 / 14 / 22 / 1,176 |
| Races / runners with any missing base value | 306 / 1,422 | 159 / 710 |
| Missing same-venue rate: races / runners | 243 / 940 | 125 / 444 |
| Missing exact-distance rate: races / runners | 240 / 523 | 127 / 268 |
| Missing grade-label rate: races / runners | 217 / 746 | 110 / 383 |
| Numerically changed races / runners / values | **0 / 0 / 0** | **0 / 0 / 0** |
| Renamed values (13 of 16 column names) | 30,680 | 16,263 |
| Runner records with explicit quality/denominators | 2,360 | 1,251 |

The two populations overlap; do not add their counts. All 37,760 original
development feature values reproduce exactly. The source cards had no history
rejections, missing finishes or invalid finishes. No numerical repair to those
values was justified. The improvement is honest semantics and explicit quality
metadata on every runner, plus tested correction of the unavailable-context
case for future development inputs. `retained_top3_rate` deliberately avoids
claiming bookmaker paid-place terms. `recent_recorded_margin_mean_5` deliberately
retains the original measurement without claiming verified units or sign.

V2 retains both recent and retained-rate columns. With identical standardized
columns and a fixed combined coefficient, equally splitting the coefficients
halves the ridge penalty compared with one column. Duplication changes effective
regularization, but it does not mechanically double signal or predictions.
Removing a duplicate would be a separately recorded model-method change.

Implementation is confined to new development files:

- [Feature record](../../scripts/development_form_quality.py): canonical history
  reuse, 16 explicit names, denominators/status, date/cap/rejection metadata,
  unknown career coverage, strict missing-context handling and invalid-finish/
  nonfinite-margin rejection, including raw ordinal/distance validation before
  lossy integer conversion. No production imports this interface.
- [Gated audit](../../scripts/audit_development_form_quality.py): reservation and
  incident manifests first, identity-only whole-file admission, permitted-source
  resolution before card opens, exact rosters and timing, original-value replay,
  named changes and saved-model replay. Hashes bind evidence but never stand in
  for completeness checks.
- [Diagnostic fit writer](../../scripts/development_form_fit.py): one fixed
  L2=1 variant through the existing chronological framework, complete 32
  coefficients and preprocessing, exact training rows/membership/cutoff,
  feature contract, caller's admission evidence, interpreter/package/source
  identities and append-only start/failure/completion ledger. It rejects reused
  output directories. Tiny synthetic fixtures validate replay; no development
  retraining was needed. Any future caller must supply independently authorized
  input admission. New fits are explicitly `new_diagnostic_fit_not_original_artifact`.
  Missing historical box coefficients were not reconstructed or relabelled.

Final [summary](feature_quality_20260929_evidence_reviewed/summary.json),
[all runner records](feature_quality_20260929_evidence_reviewed/runner_quality.jsonl),
[contract](feature_quality_20260929_evidence_reviewed/feature_contract.json), and
[input hashes](feature_quality_20260929_evidence_reviewed/input_hashes.json) are
retained. Base16 and half saved forecasts replay within **2.22e−16** for all
1,251 evaluated runners per model. The [27-file preservation check](feature_quality_20260929_preservation.json)
found unchanged installed/frozen code, units, controls, registry and model bytes.
This establishes source/artifact preservation and saved-development replay;
installed original forecasts were not replayed because that would require
additional original inputs. No live acceptance or predictive advantage is claimed.

## Recommended next data improvement

On a newly authorized, independently allocated future population, first retain
an outcome-blind per-runner history provenance/coverage packet: explicit source
event identity, availability cutoff, DB/card origin, deduplication disposition,
source grade/jurisdiction, venue/layout and raw margin definition. Record the
number of accepted starts and known finishes in each context, and whether
complete-career coverage is actually attested. This directly addresses the most
common existing information gap and the production/candidate mismatch; another
fit on the same five starts cannot supply the missing information.

The narrow later question is: **does longer, source-qualified history improve
the residual adjustment beyond the same-time market, compared on identical
eligible races?** Freeze one history-definition contrast, training chronology,
quality gate and complete receipts before future labels. First measure coverage
and route reconciliation; do not turn missing-context status into an outcome-
selected race filter. Keep grade/margin semantics unresolved until source evidence
supports them. Use this reviewed development cohort only for diagnostic design;
its 277 prior fits and later inspections mean it supplies no fresh confirmation.

No suitable unreserved future population was established. The existing programme
is allocated through **21 January 2027, 12:00 Melbourne time**; even later dates
are not automatically unreserved. The exact decision needed is an explicit new
population manifest specifying dates/venues/identities, acquisition and history/
label authority, chronology and evaluation rights, reconciled against current
and deferred reservations. No October race is excluded, moved or repurposed.

## Operational-owner handoff

**No deployment or October acquisition change is requested.** The
[concrete odds proposal](feature_quality_20260929_odds_handoff.md) specifies
T−10/T−4 observations durably available by T−2, two separately consumed slots in
the existing supervisor, one extra guarded browser acquisition (not one HTTP
request), lock release between observations, coverage priority, explicit missed
slots, and a 512 KiB/pair planning allowance. Cost and lock duration remain
unmeasured assumptions requiring owner validation. No new scheduler or parallel
pipeline was implemented. Saving metadata around the single current acquisition
could improve auditability, but cannot create the missing second price point.

## Reproduction and validation

```bash
cd /home/l4nd0/greyhound-feature-quality-20260929
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
"$RESEARCH_PY" -m unittest tests.test_development_form_quality tests.test_explain_market_residual -v
"$RESEARCH_PY" -m scripts.audit_development_form_quality --out /tmp/feature-quality-NEW
```

Fifteen focused tests passed after independent review. They exercise
unknown versus zero context, partial history, known-value denominators, positive
winner margin preservation, date/dedup/cap limits, >5-start differentiation,
invalid values, access gates, chronological failure retention and saved-fit
replay. The earlier explanation seams cover whole-field identity admission and
missing-indicator/cap/normalization reconstruction. No broad suite or historical
search was repeated. All five final artifacts reproduce byte-for-byte. See the
[independent review](feature_quality_20260929_review.md) for findings and repairs. The initial local audit output remains under
`feature_quality_20260929_evidence/`; the published `_evidence_verified/` output
adds the explicit post-admission byte check and recency's one-record denominator.
The final `_evidence_reviewed/` run additionally validates raw numeric history
before integer conversion; the demonstrated fractional-finish/distance and zero-
distance defects affect zero admitted records. All prior outputs remain retained.
None of these runs fits models or changes admitted feature values.

Worktree base: `bec84e80` (#197); branch `research/feature-quality-20260929`.
Original dirty worktrees, research artifacts, installed services and frozen
comparison files were preserved. Draft review only; no merge or deployment.
