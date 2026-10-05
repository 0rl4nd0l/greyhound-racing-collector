# Bounded retained history qualification

The new adapter makes the previously identified one-card, one-runner audit executable. It is default off, reads exactly four pinned pre-race bodies and their four HTTP receipts, and emits only counts, identity-proof hashes and qualification categories. It does not acquire pagination, target results or database rows, compute features, or evaluate predictions.

## Exact scope and operation

[Scope manifest](retained_speed_history_scope_20261005.json) pins the original Temora card, expert card and the same selected runner's two detail routes. These are **not** the historical 82-race research membership and do not establish coverage for that population. The scope is derived from the retained October 4 selection and request receipts; no selection depends on outcomes. Original receipts, including the expert card's local `TypeError` after successful body retention, remain unchanged.

[Adapter](../../race_collection/retained_speed_history.py) validates body/receipt hashes, exact URLs, successful HTTP status, bounded size, pre-jump capture ordering and source retry guidance. It independently links the standard parent entry/profile to its exact lazy URL, the expert parent's same entry/profile to its exact loader URL, and the normal detail's profile back to that parent. Expert detail has no required embedded native ID: the verified parent route and exact HTTP receipt bind that response; a contradictory detail ID rejects the audit.

Historical event URLs and timestamps are treated as opaque source observations. They are not a proven native historical runner/race namespace. Raw track labels and distance text remain separate across surfaces; the adapter compares equality but performs no mapping, fuzzy matching or inferred unit conversion. Event timestamps must precede the target jump. Capture time bounds when the response was observed; it does not establish the original publication or revision time of an individual historical observation.

The normal detail contains more than one `runner-form` table. Selection requires the unique table containing the full requested history schema; summary tables and nonrectangular control rows remain outside the event denominator. An exact event/date duplicate is suppressed only when track, distance and all inspected field text agree and at least one row has the source `runner-form__last-win` class. A conflicting or unmarked duplicate excludes that event from qualified structural counts, retaining a counted failure category. Show-more controls and unknown history extent always leave completeness unverified.

## Qualification matrix

| Item | Retained source evidence | Executable check / remaining requirement |
|---|---|---|
| Parent entry and profile | Standard DOM identifiers; expert preceding block and loader | Exact same-source parent/route/receipt checks; no profile-name matching |
| Historical event | Date-cell timestamp and event hyperlink | Exact opaque URL/date overlap; native historical event and runner namespace still unknown |
| Track and distance | Raw labels in both history tables | Equality counts only; verified track aliases, physical layout and units still needed |
| TIME | Expert sort key `finish_time` | Field presence; runner clock, unit, origin and measured/derived status still needed |
| WIN | Expert sort key `race_finish_time` | Field presence; exact race clock/reference definition still needed |
| BON | Expert sort key `best_of_night_time` | Field presence; meeting/layout grouping and publication/update timing still needed |
| 1 SEC | Expert sort key `first_sectional_time` | Missingness; clock owner, start trigger, call position, units, layout/distance/era and special codes still unknown |
| PIR | Cell class `runner-form__in-running-places` | Presence only; call-position/code dictionary needed; never used as a timing proxy |
| History depth | Source show-more controls | Rendered sample only; pagination completeness unverified |
| Pre-target knowledge | All four captures precede target jump | No per-history-row original publication/revision timestamp |

The existing four-surface audit reports six normal rows representing five unique observations, five matching expert observations, differing raw track labels and missing `1 SEC` in every rendered row. Those are **prior retained-audit expectations**, not a claim that the new executable has independently completed its real-body run. Root will retain the new machine audit separately after review. Source sort keys correct earlier local BON/PIR interpretations but do not authenticate physical speed semantics. All usable early-speed counts remain zero until those definitions and provenance are resolved.

## Validation and execution

Fabricated tests exercise the real retained table shapes, parent/route binding, hash/size/denial/time failures, contradictory identities, duplicate suppression/conflicts, missing/future events, default-off behavior and no-clobber output. Test outputs never contain historical values. A real-body audit is separate and requires root execution under kernel network denial with production/source mounted read-only and only a new audit output directory writable.

```bash
python -B -m scripts.audit_retained_speed_history
# DEFAULT_OFF; no manifest/body reads

python -B -m scripts.audit_retained_speed_history --execute \
  --manifest /absolute/source/docs/research/retained_speed_history_scope_20261005.json \
  --manifest-sha256 09d4ede34bfe3ad229f3a0539eebfc048060e35e03445b41c9211efe1fd5abfb \
  --output /new/audit/qualification.json
```

Use `bwrap --unshare-net --ro-bind / / --bind /new/audit /new/audit`, source on `PYTHONPATH`, the existing Python 3.11 environment and an external 60-second timeout. The bound is one manifest, four receipts (128 KiB each) and four bodies (4 MiB each), at most 100 history-table rows per detail, and zero provider/result requests. Unknown schema or identity failures reject the audit; the CLI exposes only `AUDIT_REJECTED`, never a raw exception containing input values. Output creation is exclusive. This is an observation audit, not a new authority or enrollment receipt.

## Primary evidence and next request

- [Four-surface audit](thedogs_four_surface_data_audit_20261004.md) identifies all original selection, body, receipt and presence projection paths. The manifest pins the actual bodies and receipts individually.
- The retained expert parent supplies sort-key definitions; the two actual lazy detail bodies supply headers, class names, row structure and pagination controls. External scripts or result endpoints are not read by this adapter.
- [Prior source-contract work](speed_source_feasibility_20261004.md) separates first-party labels from legacy local aliases and unverified storage columns.

No new provider lookup is needed to run this audit. To progress to a speed feature, root needs an official versioned TheDogs field dictionary explaining sectional clock ownership/calls/units and PIR encoding, plus history event/runner identity and publication semantics. The established access route is the retained same-session public HTTPS pre-race card → expert card → observed lazy URLs. That route does not itself establish access permission or availability for a documentation endpoint. Root must identify the documented help/dictionary route before any separately accounted request; another timing field or a different provider's definition cannot fill this gap.
