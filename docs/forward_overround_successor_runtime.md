# Forward overround successor runtime

Status: `READY_FOR_INDEPENDENT_REVIEW`. The implementation is prepared, but
collection is not active. The cohort directory, `ACTIVATION.json`, and installed
units are deliberately absent; the timer remains disabled and inactive. This
package does not authorize their creation or activation, and independent review
is still required before any later activation review.

## Frozen boundary

The runtime accepts only the current successor protocol at SHA-256
`55f553232c1b63e979f09ee8c605116b18c44e140cb0eb2e1bd5ddb06667837c`.
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
- collector/sealer code, state-machine code, finalizer code, and installed
  service-unit bytes;
- an activation time no earlier than `2026-10-01T00:00:00+10:00`;
- explicit `collection_authorized` and `scheduler_authorized` decisions; and
- an initial reviewed admission with no predecessor.

The implementation does not contain a command that writes this receipt.
Installation, daemon reload, timer enablement, or service start requires a new
owner authorization and deployment task. A repository commit alone does not
change the installed runtime.

The October boundary is also a hard non-overlap control for the current-master
Sportsbet/Betfair confirmatory cohort, whose frozen eligibility window is
2026-08-18 through 2026-09-30 inclusive. Neither activation nor race admission
can occur inside that earlier cohort's window.

The first authorization event binds the exact activation-receipt SHA-256. Every
later runtime invocation and status replay revalidates those bytes before
accepting sealed evidence. Missing or changed activation evidence fails closed
with no metrics.

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

For each candidate, the sealer first reads and hashes the immutable file bytes,
then samples the runtime clock immediately before admission. Before any write,
it checks activation, the T-33 inclusive to T-10
exclusive capture window, the 2026-10-01 boundary, exact runner identities,
complete field size, corrected Sportsbet WIN provenance, and the active runtime
admission. The runtime clock observation is sealed into both the prediction
receipt and journal event, and admission requires
`captured_at <= observed_at < jump_at`; a packet cannot use a claimed source
timestamp to create a prediction after jump. An invalid or ambiguous candidate
is a nonmember rejection. Its race ID is durably tombstoned: it cannot be
repaired retrospectively or enter through a different filename or packet.

Each candidate inbox file is identified by its filename and exact content
SHA-256. Malformed JSON, a non-object payload, or missing identity/provenance
fields produces an immutable rejection containing that stable identity rather
than being skipped. Replaying the unchanged file is an exact no-op even when
the timer's current observation time advances. Changing the bytes under a
previously rejected filename is a fatal evidence-identity conflict. Changing
the candidate identity for an already rejected race ID is also fatal; the first
rejection remains the permanent exclusion record.

An official-result packet uses schema
`forward_overround_successor_official_result_v1` and contains:

- the immutable member and race IDs;
- `source: thedogs` and `official: true`;
- an aware post-jump capture timestamp and source receipt SHA-256; and
- the exact sealed runner set, one finish position per runner, and exactly one
  winner whose native box agrees with `winner_box`.

The runtime is strictly two phase. Until exactly 1,000 prediction receipts have
sealed and membership has moved to `RESULT_CLOSURE`, any result-inbox presence
is fatal contamination and candidate sealing does not proceed. The state
machine independently rejects result events or pending-result events before
the same fixed-N boundary. Only after that boundary does the runtime read and
admit result content. Its observation is sealed into the result receipt and
event, and timing must satisfy `jump_at < captured_at <= observed_at`, so a
pre-staged future result cannot be accepted.

Finish positions must be integers and form the complete unique sequence from
one through the accepted runner count. Missing, duplicate, boolean, string,
zero, negative, or out-of-range positions fail closed before result admission;
the finalizer independently rechecks the sealed order before scoring.

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
the exact state machine, finalizer, and semantic contract, and names the active
admission hash as predecessor. The runtime cannot self-approve that receipt.
State-machine or finalizer drift is always fatal. Collector or unit drift after
membership freezes is also fatal because re-admission cannot alter result or
finalization semantics.

Protocol, model, preprocessing, scorer-contract, finalizer, sealed prediction,
sealed result, member identity, timing, runner-set, or winner drift is fatal.
Fatal evidence produces `BLOCKED_FORWARD_EVIDENCE`, `metrics: null`, and an
immutable consumption receipt. Write-once evidence is fully written and
fsynced under a same-directory temporary name, atomically linked into the final
create-if-absent path, and followed by a parent-directory fsync. A final path is
never exposed while its bytes are incomplete. Existing final artifacts count
only after JSON schema, content hash, referenced-artifact hashes, and journal
state bindings all validate. A fatal sentinel cannot be removed or
re-admitted. Event-induced terminal states use the same deterministic
`FINAL_REPORT.json` and `CONSUMED.json` sealing path; a restart after either
terminal write boundary completes the missing receipt without scoring.

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

`FINALIZE_REQUESTED` is the durable scorer-start precommit. Only the process
that appends it may begin scorer computation. A restart reuses a complete,
valid write-once metrics receipt without running the scorer again. If the
process exits after the start precommit but before complete metrics publication,
restart seals deterministic no-metrics terminal evidence; it does not attempt
the scorer a second time. This is an at-most-once execution guarantee, not a
claim that a process crash can make computation exactly once. Successful
finalization still produces exactly one committed score. Later crashes while a
complete metrics receipt, report, consumed receipt, or status is being
published resume idempotently to unchanged terminal bytes.

`CONSUMED.json` is the publication commit marker only when its schema and all
referenced final-report, metrics, member-manifest, and optional sentinel hashes
validate. If a crash leaves
`METRICS.json` or `FINAL_REPORT.json` before that marker and a later evidence or
hash conflict makes the run fatal, or if a partial/corrupt final artifact is
found, those uncommitted score artifacts are
removed before the deterministic no-metrics terminal is sealed. A committed
score event is never replayed, and the terminal receipt retains the single
scorer-invocation count without exposing metrics.

## Prepared unit

The repository contains `forward-overround-successor.service` and
`forward-overround-successor.timer` only as deployment inputs. The service is
network-denied and write-restricted to the future cohort root. The timer is
non-persistent so it cannot backfill missed collection windows. Neither file is
installed by this change, and neither unit is enabled or active.
