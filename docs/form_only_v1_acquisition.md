# FORM_ONLY_V1 acquisition contract

## Status and boundary

`FORM_ONLY_V1` is an acquisition and feature-lineage contract. It does not fit,
calibrate, rank, evaluate, blend, promote, or activate a model. Market-only
implied probability remains the safe baseline and the residual challenger
remains `NO_CREDIBLE_CHALLENGER`.

The development population ends on 2026-07-09. The separate 2026-07-11 through
2026-08-09 packet is `OUTCOME_UNOPENED_OUT_OF_TIME`; it is not called
prospective. No result source is consulted for that packet.

## Frozen sources and precedence

The candidate union is the set union of:

1. frozen Tier-A official-race-page races in
   `historical_win_tier_a_race_provenance_v1.json`; and
2. races marked `used_for_training=1` whose rows are in
   `thedogs_training_rows_v1.csv`.

Race and dog identity use `race_id`, box, and an uppercase alphanumeric dog-name
token. Identity is a join key only and is never a feature.

For races in both sources, the raw form CSVs must be byte-identical. An eligible
Tier-A raw card has precedence over an eligible published-history raw card. If
only the published-history card meets the availability rule, that card is used.
Label provenance remains separate:

- `OFFICIAL_RACE_PAGE_TIER_A` is retained only when the frozen Tier-A
  provenance declares it.
- `THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A` is never promoted to Tier A.
- no label value is copied into either input packet.

Card availability is exact: `capture_timestamp <= jump_timestamp - 60 minutes`.
Tier-A jump time comes from its frozen race provenance. Published-history jump
time comes from frozen `race_timestamp_utc`, which is authoritative over the
older sidecar display time. Naive card capture timestamps are interpreted in
`Australia/Melbourne`.

## Canonical prior-history semantics

Features are rebuilt from the selected raw pre-race card, never selected from a
legacy builder output or from a mutable database.

- Each non-empty `Dog Name` row starts that runner's history block; blank-name
  continuation rows remain in that block.
- Rows are accepted only when `history_date < target_date`.
- Exact duplicate rows are removed using date, track, distance, grade, finish,
  box, and margin.
- Accepted rows are sorted newest first and capped at 20. Recent windows use the
  newest 3 or 5; `career_*` means all available accepted rows within that
  explicit 20-start evidence cap, not an assertion of complete lifetime starts.
- Recency is integer calendar days from target date to the newest accepted row.
- Finish is numeric `PLC`; win is finish 1 and place is finish 1-3. Margin uses
  numeric `MGN` only. Missing numeric values are excluded from their aggregate.
- Same-distance means exact integer metres. Same-venue uses the embedded,
  versioned venue-alias table. Same-grade uses the embedded grade aliases below.
- Missing history, recency, finish, margin, and each same-context slice have
  explicit flags. Missing values are blank, never silently zero-filled.

Grade normalization removes `Tier 3 -` and `Bottom Up -` prefixes, maps ordinal
and numeric grades to `GRADE_n`, maps `M`/`Maiden`, `FFA`/`Free For All`,
`INV`/`Invitation`, restricted-win variants, mixed-grade variants, and common
short codes explicitly. Unmapped non-empty grades are preserved as a stable
uppercase alphanumeric token; missing grades map to `__MISSING__`.

## V1 feature scope

CORE is limited to starts, recency, recent/capped-career finish-win-place-margin,
same venue/distance/grade history, and missingness.

CONTEXT is limited to box, target venue, distance, grade, and field size. A later
fit must derive categorical vocabularies from training folds only, pool values
seen fewer than 10 times to `__RARE__`, and map unseen values to `__UNKNOWN__`.
This acquisition step does not create a vocabulary.

DEFERRED includes speed, times, sectionals, opponent strength, prize money,
weather, dog identity, trainer identity, and high-dimensional interactions.

The builder ignores `SP`, `OPEN`, `LOW`, `HIGH`, odds, target result fields,
post-jump corrections, and target/post-target history. Their presence in a raw
legacy export does not authorize parsing or transport into the packet.

## Out-of-time source rule

Only leakage-safe, complete pre-race card sidecars and their hash-matching CSVs
are considered. Paths marked as result, replay, reconstruction, repair,
backfill, or official-result material are rejected before content is read. For
each race, the freshest valid card at or before T-60 is selected. No label or
outcome source is opened.

## Market separation

Market timing is not an input eligibility rule. T-60 market pairing remains
separate `DATA_MISSING`. The owner-supplied T-60/T-30/T-10/T-2 counts are
preserved as carry-forward expectations for a later evaluation, but this lane
does not claim they are independently frozen: no immutable market cohort
manifest with matching pairing semantics was available. Market timing and
labels require separate authorization and never gate this odds-free packet.
