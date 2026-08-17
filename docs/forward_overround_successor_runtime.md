# Forward overround successor runtime

Status: `READY_FOR_ACTIVATION_AUTHORIZATION`. The implementation is prepared,
but collection is not active. The cohort directory, `ACTIVATION.json`, and
installed units are deliberately absent; the timer remains disabled and
inactive. This package does not authorize their creation or activation.

## Frozen boundary

The runtime accepts only the unchanged successor protocol at SHA-256
`4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be`.
It verifies the protocol's frozen development assets on every invocation:

- `final_model.json`:
  `c81b4b3047b7840ba31269504e0c5ceb6c54d742a82a4e01cca52b11fdaa471e`
- `preprocessing.json`:
  `ad85722337d80360e1745f75fe57ff6b3fbd1e80deac57af318c898372b01998`
- `protocol.json`:
  `2b20704e41574d5557eb1d6381bc314212b382a195ca5bffea95b832d4a5fb4a`
- `scorer_contract.json`:
  `c119feea4f67baad73dc9e23ff7f98d755a34054c32a0a98dea4f44cdafa2576`

V2 remains terminal `BLOCKED_FORWARD_EVIDENCE`, consumed with nine sealed
predictions, six approved results, and no metrics. Its files and known outcomes
are not input to this runtime or finalizer.

## Authority and activation boundary

The checked-in service cannot create a cohort. Both the cohort directory and a
separately reviewed `ACTIVATION.json` must already exist before systemd will
start it. The runtime also refuses to create the root or activation receipt.

A future activation review must bind all of these exact identities:

- successor protocol and semantic scorer contract;
- frozen model, preprocessing, development protocol, and scorer assets;
- collector/sealer code, finalizer code, and installed service-unit bytes;
- an activation time no earlier than `2026-09-01T00:00:00+10:00`;
- explicit `collection_authorized` and `scheduler_authorized` decisions; and
- an initial reviewed admission with no predecessor.

The implementation does not contain a command that writes this receipt.
Installation, daemon reload, timer enablement, or service start requires a new
owner authorization and deployment task. A repository commit alone does not
change the installed runtime.

## Evidence intake

After separate activation, the service reads immutable JSON packets deposited
under `candidate_inbox/`, `result_inbox/`, and, only during a runtime-admission
pause, `admission_inbox/`. It makes no network request and never writes the
canonical database. The upstream capture path remains outside this unit and
must supply source receipts with exact provenance.

A candidate packet uses schema
`forward_overround_successor_candidate_v1` and contains:

- exact Sportsbet race ID, aware capture and jump timestamps;
- `source: sportsbet`, `market_type: win`, and
  `raw_paired_column_win_proof: true`;
- the source receipt SHA-256 and complete active-runner count; and
- one row per active runner with native box number, exact dog name, corrected
  decimal WIN odds, and raw source-row SHA-256.

Before any write, the sealer checks activation, the T-33 inclusive to T-10
exclusive capture window, the 2026-09-01 boundary, exact runner identities,
complete field size, corrected Sportsbet WIN provenance, and the active runtime
admission. An invalid or ambiguous candidate is a nonmember rejection. It
cannot be repaired retrospectively into the cohort.

An official-result packet uses schema
`forward_overround_successor_official_result_v1` and contains:

- the immutable member and race IDs;
- `source: thedogs` and `official: true`;
- an aware post-jump capture timestamp and source receipt SHA-256; and
- the exact sealed runner set, one finish position per runner, and exactly one
  winner whose native box agrees with `winner_box`.

Result evidence may close only an existing immutable member. Conflicting race,
runner, winner, timestamp, or receipt evidence terminates the cohort with no
metrics.

## Append-only state and admission

`EVENTS.jsonl` is a hash-chained append-only journal. Prediction and result
receipts are write-once files whose hashes are replayed on every invocation.
`STATUS.json` is derived operational status only and exposes counts, admission
state, provenance integrity, and result closure. It never exposes candidate or
baseline loss, paired deltas, calibration, ranks, or an interim verdict.

Unadmitted collector-code or service-unit drift before a prediction seal moves
the runtime to `ADMISSION_PAUSED` without membership. Resume requires a later,
reviewed, hash-chained admission that binds the observed code and unit, retains
the exact finalizer and semantic contract, and names the active admission hash
as predecessor. The runtime cannot self-approve that receipt.

Protocol, model, preprocessing, scorer-contract, finalizer, sealed prediction,
sealed result, member identity, timing, runner-set, or winner drift is fatal.
Fatal evidence produces `BLOCKED_FORWARD_EVIDENCE`, `metrics: null`, and an
immutable consumption receipt. A fatal sentinel cannot be removed or
re-admitted.

## Deterministic finalization

At exactly 1,000 prediction receipts, collection closes permanently and waits
for those same 1,000 approved results. No member can be removed, replaced, or
added. When result closure is complete, the journal accepts one
`FINALIZE_REQUESTED` event and freezes the exact ordered receipt manifest.

The finalizer recomputes the proportional Sportsbet WIN baseline and unchanged
linear overround transform from sealed odds. It then performs one paired
analysis on the identical 1,000 races:

- mean multiclass race log loss as primary metric;
- 20,000 paired race bootstrap replicates with seed `20260817`;
- 20,000 Australia/Melbourne race-date cluster replicates with seed
  `20260818`;
- five equal-count chronological blocks; and
- the frozen secondary Brier, calibration/ECE, accuracy, and winner-rank
  summaries.

The metrics receipt must bind the frozen member manifest. `FINAL_REPORT.json`
and `CONSUMED.json` are write once. ROI, returns, staking, EV, and betting are
not computed.

## Prepared unit

The repository contains `forward-overround-successor.service` and
`forward-overround-successor.timer` only as deployment inputs. The service is
network-denied and write-restricted to the future cohort root. The timer is
non-persistent so it cannot backfill missed collection windows. Neither file is
installed by this change, and neither unit is enabled or active.

