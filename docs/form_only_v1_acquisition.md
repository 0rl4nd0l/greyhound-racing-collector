# FORM_ONLY_V1 acquisition contract

## Status and boundary

`FORM_ONLY_V1` is an acquisition and feature-lineage contract. It does not fit,
calibrate, rank, evaluate, blend, promote, or activate a model. Model and market
dispositions are deliberately quarantined from this trusted path; this packet
makes no predictive-performance or edge claim.

The development population ends on 2026-07-09. The separate 2026-07-11 through
2026-08-09 packet is `OUTCOME_UNOPENED_OUT_OF_TIME`; it is not called
prospective. No result source is consulted for that packet.

## Frozen sources and precedence

The candidate union is the set union of:

1. frozen Tier-A official-race-page races in
   `historical_win_tier_a_race_provenance_v1.json`; and
2. races marked `used_for_training=1` whose rows are in
   `thedogs_training_rows_v1.csv`.

Trainer-visible `row_id` values derive only from `race_id` plus box. The trainer
is given only the `trainer/` directory, whose ten regular files must exactly
equal the ten declarations in `control_plane/trainer_input_manifest.json`.
Dog names, tokens, digests, source-runner IDs, alignment mappings, source paths,
and URLs exist only in the separately hashed `sealed_validation/` bundle and
are absent from the trainer allowlist. Reusing a dog name in another race
creates no trainer-visible join key; development/out-of-time row-ID
intersections are required to be zero. Hashing a dog token is not treated as
anonymization.

For races in both sources, the raw form CSVs must be byte-identical. An eligible
Tier-A raw card has precedence over an eligible published-history raw card. If
only the published-history card meets the availability rule, that card is used.
Within the winning precedence, the freshest eligible capture wins. Equal-time
winners fail closed unless card and sidecar bytes prove one identical canonical
source identity; pathnames never decide conflicting source content.
Label provenance remains separate:

- `OFFICIAL_RACE_PAGE_TIER_A` is retained only when the frozen Tier-A
  provenance declares it.
- `THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A` is never promoted to Tier A.
- no label value is copied into either input packet.

Every selected card roster must exactly equal its COMPLETE sidecar's canonical
box/name multiset. A label roster may omit a sidecar participant only when the
hash-bound published-history input independently binds the listed and active
counts plus scratch/reserve state; every such participant receives an explicit
opaque exclusion row and evidence digest. Missing, extra, swapped, duplicate,
or colliding identities otherwise fail closed.

Card availability is exact: `capture_timestamp <= jump_timestamp - 60 minutes`.
Tier-A jump time comes from its frozen race provenance. Published-history jump
time comes from frozen `race_timestamp_utc`, which is authoritative over the
older sidecar display time. Naive card capture timestamps are interpreted in
`Australia/Melbourne`.

Every sidecar is also validated semantically, independently from its supplied
hash: filename/race ID, date, venue, race number, canonical source URL, jump and
capture evidence, card path, and canonical roster must agree with the bound
card/acquisition evidence. Rebinding inventory hashes cannot make a wrong-race
sidecar valid. Missing byte lengths, duplicate declarations, malformed schema
or types, and incomplete or duplicate freeze roles fail closed.

## Trust domains

The build has four trust domains:

- `trainer/` is `TRAINER_VISIBLE_AUTHORITATIVE`. Its five
  `MODEL_INPUT_DATA` files and five explicitly role-tagged trainer-safe metadata
  files are the complete readable set. It contains no manifest, signature,
  symlink, directory, dotfile, undeclared file, or alternate-path alias.
- `control_plane/` contains `trainer_input_manifest.json` and
  `artifact-manifest.sha256`. A launcher validates this domain, the exact
  trainer directory set, every declaration, file type, byte length, digest,
  and single-link/no-follow path before returning any bytes to a trainer. These
  two files are never trainer-readable inputs.
- `sealed_validation/` contains source inventories and dog/card alignment used
  only to validate semantics and provenance. It is not a trainer input.
- `non_authoritative_diagnostic/` contains the legacy overlap reconciliation.
  It cannot affect canonical features, eligibility, expected authoritative
  counts, or the trainer manifest; pre/post diagnostic trainer hashes must be
  byte-identical.

The runtime-real read entrypoint is
`load_verified_trainer_inputs(packet_root, reproducibility_contract)`. It opens
the packet, control, trainer, and declared files with no-follow semantics;
requires regular single-link files; compares actual and declared names before
returning data; rejects path separators, traversal, absolute or dot paths; and
verifies the control-plane bytes against the tracked reproducibility descriptor.
Unexpected regular files, a thirteenth file, dotfiles, directories, symlinks,
hard links, renames, duplicate declarations, missing files, type changes,
length/hash changes, and packet/root aliases therefore fail closed.

The non-self-referential trust chain is: reviewed Git commit -> tracked
`form_only_v1_reproducibility_v3` control hashes -> the two control-plane files
-> the signature's hashes of exactly the ten trainer files. The trainer input
manifest hashes the signature, the signature hashes only trainer files, and no
manifest or signature hashes itself.

## Canonical prior-history semantics

Features are rebuilt from the selected raw pre-race card, never selected from a
legacy builder output or from a mutable database.

- Each non-empty `Dog Name` row starts that runner's history block; blank-name
  continuation rows remain in that block.
- Rows are accepted only when `history_date < target_date`.
- Rows are normalized before deduplication using canonical date, venue,
  distance, grade, finish, box, margin, and verified start identity. Distinct
  same-day starts with otherwise identical results remain distinct.
- Accepted rows are sorted newest first and capped at 20. Multiple distinct
  same-day rows require one unique verified timestamp or race-ordinal key;
  input order is never used and unprovable ordering fails closed. Recent windows use the
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

Equal freshest capture times fail closed when their bytes differ. Byte-identical
aliases collapse to one canonical source identity; discovering the same live
path twice is an error rather than a silent skip.

Frozen mode binds only three construction inputs: the OOT source inventory,
exclusion inventory, and OOT manifest. The builder verifies their bytes and
digests, then rederives each selected path, filename/race identity, sidecar
identity, timezone-aware capture and jump timestamps, Jul 11-Aug 9 membership,
T-60 eligibility, and card/sidecar roster. Manifest counts are checked against
the rederived population; neither 88/617 nor an older artifact hash is hardcoded
in source. Capture declarations must carry an explicit offset. Canonical race
times must be exact URL matches and are interpreted in the source display zone
`Australia/Melbourne`, which is emitted in the rebuilt manifest.

## Durable reproduction

The tracked [reproducibility descriptor](form_only_v1_reproducibility.json)
binds four development construction inputs, 3,231 authoritative card, sidecar,
and label records, 38 separately non-authoritative shadow sources, and the
three-file OOT freeze at
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-form-only-v1-acquisition-20260718/reports/agent_jobs/form_only_v1_acquisition_foundation_20260718`,
the expected current counts, and 19 artifact hashes across four domains. The OOT
freeze aggregate is `30671f48...e7cc`; trainer, control-plane,
sealed-validation, and diagnostic aggregates are respectively
`97967ab3...4e31`, `1712d3d6...5462`, `4ce9d105...26ed`, and
`e5eaf492...995f`. Reviewers on the evidence host can reproduce without raw or
large Git commits using:

```bash
python3 scripts/build_form_only_v1_packet.py \
  --eligibility-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-historical-win-eligibility-20260715/reports/agent_jobs/historical_win_eligibility_reconciliation_20260715 \
  --training-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-training-first-data-evidence-20260713/historical_training_dataset_v1 \
  --out-of-time-freeze-dir /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-form-only-v1-acquisition-20260718/reports/agent_jobs/form_only_v1_acquisition_foundation_20260718 \
  --reproducibility-contract docs/form_only_v1_reproducibility.json \
  --output-dir /tmp/form-only-v1-review
```

Any one-bit change in a bound input for its trust domain or any expected
count/hash causes a non-zero exit. The descriptor intentionally carries no raw cards,
labels, outcomes, databases, models, or large generated packet files.

The independent-review diagnostic baseline was 504/530 unexplained rows across
73/73 overlap races. Parsing numeric zero as a real value reclassifies 245
zero-history rows and recomputes the diagnostic to 259/530 unexplained across
71/73 races. This delta changes no authoritative count or trainer byte. The
focused adversarial suite contains 84 tests, including exact-set and filesystem
alias attacks; no "zero unexplained" claim is made.

## Market separation

Market timing is not an input eligibility rule. T-60 market pairing remains
separate `DATA_MISSING`. The owner-supplied T-60/T-30/T-10/T-2 counts are
preserved as carry-forward expectations for a later evaluation, but this lane
does not claim they are independently frozen: no immutable market cohort
manifest with matching pairing semantics was available. Market timing and
labels require separate authorization and never gate this odds-free packet.
