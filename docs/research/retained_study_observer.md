# Retained study observer

`run_retained_study_observer.py --config ABSOLUTE_PATH --config-sha256 SHA`
runs one finite observation. Root may schedule this command; it starts no
collector, renews no source allowance, fetches nothing and computes no scores.
The readiness amendment and exact protocol must already be authorized. The
readiness gate runs before the observer creates its separate journal.

The `retained_study_protocol_v1` protocol binds its actual issue/effective time,
original study plan/end, separate state root, historical plan references,
frozen model/manifest/registry hashes, prior scientific capture count, prior
scientific member identities, total/new member caps, and finite scan limits
(`max_files`, `max_bytes`). Root must reconcile the original ledger before
issuing the intended 17 original attempts plus 983 new references. These are
membership limits, not permission for additional provider requests.

`historical_plans` includes all authorized runs, including failed sessions.
`persistent_source` binds `standing_authority`, `runtime_root` and
`first_racing_date`; authentic daily allocations and comparison plans under
that root extend discovery through the protocol endpoint. No result status is
consulted. `opportunity_evidence` pins the historical denominator audits;
`independent_verifier_evidence` preserves existing verification audit references.
Native opportunities and dispatches lacking admission are retained separately.

Every new member requires original admission/completion timing, four native
seals, exact race/job/request identity, fixed model pins and opaque SHA256
verification of **every** bundle-manifest file, including capture, retained
input archive, history DB, features and forecasts. No archive is unpacked, no
history/result database is opened, and no forecast values are decoded. Original
source implementation and membership remain unchanged. This proves retained
integrity using the existing native seal chain, not a new numerical replay or
result closure. All members are explicitly reference/retrospective membership;
actual observer selection time is retained, including future observations.

The hash-chained journal is locked and append-only. Same admission is idempotent;
other same-race admissions and reused job/prediction identities are excluded.
Earlier exclusions/pending evidence remain when a later completion qualifies.
Any changed already-selected file holds the observer before additional
admission. Existing-member files are rehashed each cycle; downstream use must
also verify the recorded immutable references. Corrupt/partial journals are
never repaired/reset automatically.

Budget exhaustion is incomplete work, not acceptance. Exhaustion within a
candidate records pending qualification; exhausting the preliminary scan or
existing-member integrity check exits HOLD. Either preserves earlier records.
The next observation can finish only if its finite allowance fits the retained
workload. Initial 50-race full-file check used about 74 MB; root must size limits
for the complete 82-race history, current/future growth, repeat verification and
explicit headroom. No successful scan implies continuous producer health:
producer HOLD is reported separately, and historical readiness never establishes
current source availability.

Tests cover the public observation seam, original artifact immutability,
tampered input/history/forecast, late/missing completion, symlink/path escape,
prior membership and endpoint, gate-before-write, unchanged-authority restart,
corrupt journal, finite budget, native opportunity/failed-dispatch retention,
authenticated persistent day discovery and truthful producer HOLD. Synthetic
fixtures supply only external gate/plan-loader seams; production persistent
authority validation is exercised for future-day discovery.

The optional `development_reservations` configuration reference applies the
September 30 approved first-six development allocation before **new** study
membership on October 10–11. This is an operational reservation-routing
correction under existing authority. It does not alter the study protocol,
existing membership, comparison schedule, source allowances or result access.

The referenced immutable JSON has these fields (all references are exact
`{"path": absolute_path, "sha256": raw_file_sha256}` objects):

```json
{
  "schema_version": "retained_study_development_reservations_v1",
  "status": "AUTHORIZED_ORIGINAL_DEVELOPMENT_RESERVATIONS",
  "allocation": {"path": "APPROVED_ALLOCATION", "sha256": "SHA256"},
  "exclusive_amendment": {"path": "APPROVED_EXCLUSIVE_AMENDMENT", "sha256": "SHA256"},
  "plan": {"path": "FROZEN_SPEED_PLAN", "sha256": "SHA256"},
  "state_root": "ABSOLUTE_SPEED_COORDINATOR_STATE_ROOT",
  "dates": ["2026-10-10", "2026-10-11"],
  "predecessor_observer_config": {"path": "INSTALLED_ORIGINAL_OBSERVER_CONFIG", "sha256": "SHA256"}
}
```

The allocation, exclusive amendment and approval must bind the original
`development-single-snapshot-20261003-v1` authority, four original dates and
`first_six_1310_1420_melbourne_before_WIN_qualification_v1` rule. The separately
frozen speed plan must name the same allocation and amendment and retain the
exact two-date, six-per-date population policy. The deployment owner verifies
these references against the approved original files before pinning the
successor configuration.

For each date, the observer reads the coordinator's `date-accounting.json`.
Absent accounting holds potential 13:10–14:20 candidates as
`DEVELOPMENT_SELECTION_PENDING`; merely passing the deadline never releases
an unknown selection. A `POPULATION_FROZEN` record must bind `population.json`
and its canonical content digest. The observer independently recomputes the
first six from the entire observed census and checks the protected-membership
snapshot against a hash-verified prefix of the original observer journal. It
also verifies `original-population.json` and its `.completion.json` against
the same census, allocation, fresh index timestamps and 12:50 completion.
Selected keys remain reserved even after a failed worker or changed race time;
all nonselected candidates retain their ordinary study qualification rules.

A durable `INDEX_MISSING`, `INDEX_STALE`, `INDEX_INCOMPLETE`,
`FREEZE_INTERRUPTED` or `SOURCE_OR_AUTHORITY_UNAVAILABLE` record with a reason
and no population digest consumes the date without selecting keys. Normal
study consideration can then continue. The date disposition and earlier
race-specific pending records remain in the journal. An orphan population
alone never establishes selection or success. Previously observed terminal
accounting/census references cannot subsequently change or disappear.

Installation recipe for the sole runtime owner:

1. Keep the original configuration, protocol, amendment, journal and all native
   artifacts. Write the reservation document to a new immutable path and pin
   the original observer configuration in `predecessor_observer_config`.
2. Create a successor configuration changing only `source_commit`,
   `study_amendment`, and the new `development_reservations` reference.
   Preserve `retained_study_protocol` and every other configuration field.
3. Produce the source compatibility receipt against the existing integrity
   witness's producing commit and the actual clean successor HEAD. Its
   `changed_paths` must equal `git diff --name-only PRODUCING SUCCESSOR`; all
   model, feature and frozen-membership change flags remain false. The original
   readiness verifier still enforces the installed commit and clean worktree.
4. Regenerate the operational successor amendment with the same original user
   authorization, predecessor schedule, integrity/closure evidence and scope.
   Bind its `target_config_sha256` with `live_freshness_contract.digest` over
   the successor configuration excluding `study_amendment`, then pin the new
   amendment's raw file hash in the configuration. Use truthful receipt times
   and the complete historical-slot inventory required by the unchanged
   readiness validator; never overwrite or backdate an old receipt.
5. Validate readiness from the actual successor source and pinned configuration
   before coordinating the observer service cutover. The first observation
   retains the old journal `IDENTITY` and appends
   `DEVELOPMENT_RESERVATION_BINDING`; it never replaces existing records.
   Future observations reject removal or replacement of that binding.

The helper reads only existing local metadata and hashes. It performs no
provider call, result decode, score calculation or producer-state write.

An explicitly authorized earlier schedule can use
`retained_study_development_reservations_v2`, with status
`AUTHORIZED_RESCHEDULED_DEVELOPMENT_RESERVATIONS`. It retains the reference
fields above and adds `predecessor_reservations` and
`reschedule_authorization`. Its `predecessor_observer_config` is the currently
installed reservation-enabled configuration. The source/amendment/reservation
reference remain the only permitted observer configuration changes.

The dated `development_reschedule_authorization_v1` receipt binds the original
allocation, predecessor reservation, exact `selection_windows`, six-per-date
and twelve-total limits, and preservation of existing members and consumed
allowances. The new `development_allocation_v2` and
`development_reservation_amendment_v2` bind that approval and their original
predecessors. The original allocation identity remains
`development-single-snapshot-20261003-v1`; rescheduling never creates a new
cumulative allowance. All frozen experiment fields outside schedule, authority,
closure times and the inventory population policy must match the prior plan.

Each of the two windows is `{local_date, freeze_at, jump_start, jump_end}` with
aware timestamps on the same Melbourne date. The plan and user receipt must
precede the declared freeze minute; the jump window begins at least twenty
minutes after its start. The v2 plan's embedded schedule must match its pinned
`prospective_speed_schedule_amendment_v1` document. Both the original receipt
and final population must finish within the declared freeze minute.

V2 population selection verifies the entire pinned
`daily_race_inventory_v1`, its producer preparation and native output path,
and a maximum age of 1,800 seconds at freeze. Every discovery row must appear
in the original census, including missing jump times. Native venue codes keep
hyphens exactly. The v1 parser and original 12:50 rule retain their semantics.
The first six in the declared window are determined before qualification;
existing protected members are removed from that six without replacement.

Supersession requires the original operation to remain unconsumed. Its state
may contain only the two worker lock files and empty `attempts`, `closure`,
and `result-controller` directories. Any population, accounting record,
claim, result or evaluation artifact holds the transition. The observer
verifies the original journal binding, appends
`DEVELOPMENT_RESERVATION_SUPERSESSION` with disposition
`SUPERSEDED_UNCONSUMED`, and preserves every earlier byte and member. Restoring
the superseded configuration or changing the adopted successor is rejected.
The old reservation scope then ceases to withhold new study members; ordinary
study qualification continues to apply. The new unknown selections remain
held until an authenticated freeze or terminal date disposition is available.
