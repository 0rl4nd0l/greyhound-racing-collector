# Retained speed measurement contract — 6 October 2026

The retained TheDogs pages establish field labels and a pre-race document/runner
binding. They do **not** establish comparable physical speed measurements. The
existing `UNQUALIFIED` disposition stands. This contract specifies the evidence
needed to test the proposed three-comparable-prior-starts hypothesis; it neither
changes that threshold nor repeats the presence census.

## Primary evidence checked

The exact source bodies, URLs, receipt hashes and target jump are pinned in
[the retained scope](retained_speed_history_scope_20261005.json). On October 6,
offline verification rechecked the body/receipt hashes, exact successful HTTP
response bindings, capture-before-jump ordering and parent entry/profile/route
binding using the existing [retained adapter](../../race_collection/retained_speed_history.py).
Only sort controls and identity/structure attributes were inspected; timing and
outcome cells were not decoded. No new website access was performed.

The expert parent's `<select><option>` controls map the following labels to
source-owned keys. Its [retained body](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/thedogs-live-form-inspection-20261004/expert.html)
has SHA256 `be3a92645bf6264286d0ab241a0db2d69e1d359f64b2bae60c9f0879bc4de0ea`;
its [receipt](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/thedogs-live-form-inspection-20261004/0-receipt.json)
has SHA256 `342b97aa4e38ec587836647f558b32d3cc662a8bce252ae526bfffbb9c4c7e55`.
The original `FAILED_PRESERVED` receipt records a local post-retention parsing
error alongside the retained HTTP 200 response. It remains unchanged.

| Field | What the first-party control establishes | What remains unestablished |
|---|---|---|
| `TIME` | `finish_time` | Whose finish clock; measured versus derived; start/finish triggers; units/precision; exceptional values; layout/era scope. The label alone does not authenticate an individual elapsed time. |
| `WIN` | `race_finish_time` | Whether and how the winning runner/race reference is timed; clock agreement with `TIME`; units/precision; event and revision binding. Do not subtract it from `TIME` without that contract. |
| `BON` | `best_of_night_time` | Eligible races and grouping; measured versus computed reference; when the reference is first published, revised and finalized; units/precision and layout/distance scope. It is not a source-defined “bonus”. |
| `1 SEC` | `first_sectional_time` | Runner versus leader clock; units/precision; start trigger and physical sectional endpoint; layout/distance/equipment era; exceptional/missing codes and publication/revisions. It is not yet an authenticated runner's early split. |

The [normal detail](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/thedogs-runner-inspection-20261004/runner.html)
and [expert detail](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/thedogs-runner-inspection-20261004/runner_expert.html)
are individually hash-bound in the scope. The normal detail has a profile
attribute agreeing with the parent. The expert detail has no native entry or
profile attribute; its response is bound through the exact parent loader URL and
receipt. A normal last-win marker and show-more control are present. These facts
prevent treating current entry identity as lifetime identity, repeated presentation
as another start, or one response as complete lifetime history. The existing
[four-surface audit](thedogs_four_surface_data_audit_20261004.md) records that raw
track namespaces differ and exact historical event URLs/date observations are
not a verified historical event-entry-profile mapping.

PIR's source class is `runner-form__in-running-places`; this supplies neither call
locations nor a codebook. PIR, final placement and total race time must not stand
in for an early sectional. `Av 1 SEC` and best-first-split summaries additionally
need their selection window, included observations and missing-value policy;
they cannot establish three distinct comparable starts. These limits follow from
the same retained surfaces and [adapter outputs](retained_speed_history_qualification_20261005.md).

## Measurement and availability requirements

Each proposed measurement must bind to a versioned first-party definition and
the exact historical source revision. An evidence hash proves byte identity,
not the truth of a caller's interpretation. An independent source review must
establish all of the following before any measurement is qualified:

1. **Physical quantity:** provider/export version, exact field, clock subject,
   units/precision, measured versus derived status and any derivation, start
   trigger and physical endpoint. For the existing early-speed hypothesis the
   quantity must be a runner's elapsed first-sectional time. A documented leader
   split would be a different hypothesis and must not be relabelled.
2. **Comparability:** canonical track/layout identity, race distance, sectional
   endpoint and clock arrangement, applicable layout/equipment era and its
   effective interval. Raw track text, equal distance numbers or a display-name
   match are insufficient. Cross-venue normalization needs separate evidence.
3. **Native identity:** an unambiguous historical event, historical entry and
   lifetime profile binding in the provider's namespace, with source evidence
   tying the observed field to that participation. Preserve the observation
   provenance while counting each historical participation once. Conflicting
   observations require explicit resolution; do not average or fuzzy-join them.
4. **Availability:** distinguish event occurrence, publication of the **exact
   revision used**, capture and seal times. Require
   `occurred <= revision_published <= captured <= sealed < prediction_cutoff`,
   with `prediction_cutoff <= target_jump`. A pre-race capture shows an upper
   bound on availability; it is not a recovered original publication timestamp.
   Later revisions cannot backfill an earlier frozen prediction.
5. **Observation quality:** provider missing/special/status codes, incomplete
   run rules, precision and timing method. Missing observations remain missing.
   A numeric-looking string is not proof of measurement quality.
6. **Prespecified field coverage:** at least three distinct qualified comparable
   prior starts for every admitted runner in the frozen target roster. Account
   for all runners and exclusions before outcomes are inspected. One runner's
   successful check cannot establish a complete field or predictive benefit.

For `BON`, the exact revision used must already be available before the target
prediction cutoff. A meeting reference finalized after its underlying historical
race could be available for a later target, but cannot be used in a prediction
whose cutoff predates that publication. This contract does not assume either
availability or finalization merely from the label.

## Executable metadata consistency check

[check_sectional_metadata](../../race_collection/speed_measurement_contract.py)
is a pure, offline check at one deliberately narrow interface: one target runner,
one declared TheDogs runner-sectional definition, and three to one hundred
declared prior run observations. It has no file/network access, acquisition,
model integration or command-line activation path. No numeric measurements or
outcomes are accepted in its strict schema. Its public-interface tests use only
[fabricated metadata](../../tests/test_speed_measurement_contract.py).
The checker requires the proposed representation of a runner's elapsed seconds;
this is a candidate contract, not evidence that the source uses seconds. A
different documented quantity/unit needs an explicitly reviewed contract or
conversion, never an implicit relabelling.

The packet contains `definition`, `target` and `observations`. The definition
declares the physical meaning, evidence digest/locator, layout/distance/era and
effective interval. Each observation declares a native event/entry/profile
binding, definition/identity/body/receipt digests, the same scope and four
timestamps. `published_at` means publication of the exact body revision, not
first publication of a mutable record. Timestamp comparisons use offsets.

The check rejects unknown required semantics, unsupported quantities, mixed
layout/distance/era, ambiguous identity basis, duplicate events or entries,
insufficient starts, missing evidence references, invalid effective intervals,
late or inconsistent availability, unknown/extra fields and asserted
`qualified` flags. Errors contain fixed categories rather than input values.
The hundred-observation ceiling is a finite metadata-processing bound, not a
changed feature-selection window. No start-selection or averaging rule is added.

**A successful check returns `METADATA_CONSISTENT_EVIDENCE_UNREVIEWED`,
`measurement_qualified: false` and `feature_use_authorized: false`.** It does not
open or hash the declared evidence, authenticate identity strings, interpret
quoted definitions, establish numeric quality or independently check source
claims. Free-text declaration completeness is only syntactic. Even completely
fabricated, internally consistent declarations cannot qualify measurements.
There is no callable promotion switch. Actual evidence verification, full-field
admission and any model experiment remain separate work.

Validation: focused tests exercised this public interface through successive
failing/passing cases. They run with pytest's repository conftests and plugin
autoload disabled, so collection cannot import the app or initialize databases:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  -B -m pytest --noconftest -c /dev/null -p no:cacheprovider -q \
  tests/test_speed_measurement_contract.py
```

## Exact evidence request for the coordinator

The established retained route was the same-session public HTTPS pre-race card
to its observed expert-form link and observed runner-detail loader URLs, with
exact HTTP receipts. This proves that route's prior use, not a documentation
endpoint's accessibility. No working TheDogs dictionary/help route is retained.
Root should locate a provider-owned documentation link through its established
workflow before selecting a new probe; existing holds and request limits remain.

Request the following finite evidence package, without target results:

| Evidence to obtain | Question it must answer | What it unlocks |
|---|---|---|
| Versioned TheDogs field/export dictionary applicable to the retained page/export version | For each of `TIME`, `WIN`, `BON`, `1 SEC`: clock subject, unit/precision, measured/derived rule, start/end trigger, missing/special/status codes | Physical interpretation; keeps elapsed, winner-reference and meeting-reference clocks separate |
| Provider layout/sectional configuration and effective-date mapping | Which track/layout, race distance, split call, equipment/clock arrangement and era produced each field? How are normal/expert/export track namespaces mapped? | Prespecified comparable-start membership |
| Historical event-entry-profile schema plus a provider-owned retained fixture | Which native identifiers bind a historical observation to the actual dog participation? How are repeated last-win rows and revisions identified? | Distinct-start counting and conflict handling without name joins |
| Publication/revision specification plus exact revision provenance | When was the specific historical observation/reference revision available; can `BON` change during the meeting; how are revisions and summary windows timestamped? | As-of eligibility without assuming capture time equals first publication |
| Summary/PIR dictionary, only if those measurements are separately pursued | What population creates average/best sectional summaries, and what physical calls and special codes does PIR encode? | A separate assessment; does not substitute for the current runner-sectional requirement |

The retained [GRV/Isolynx documentation lead](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/speed-primary-documentation-followup-20261004.json)
describes a different source and does not supply any missing TheDogs fact. It
requires its own access/schema/measurement qualification. No documentation was
re-fetched, no provider contacted, and no new access method inferred here.

Once those facts are independently reviewed, pin the source definitions and
identity mappings, verify the exact retained observation/receipt chain, and
produce a values-free projection of three comparable starts per target runner
on a frozen membership. The current audit remains unqualified until that work
succeeds. This change supplies no data, model or runtime authority.
