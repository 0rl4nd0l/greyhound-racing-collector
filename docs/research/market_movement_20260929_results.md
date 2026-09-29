# Market movement: insufficient qualified retained data

The retained evidence cannot support the prespecified T−2 incremental-movement experiment. **291 of 331 admitted development races have two timestamped complete box sets, but zero have two qualified WIN snapshots with independent decision-time availability and scratching evidence.** This is an evidence failure, not a result that price movement has no predictive value. Zero movement fits, performance comparisons or return calculations were run.

| Qualification stage | Races / 331 | Runners | Dates / 27 |
|---|---:|---:|---:|
| Exact admitted development population, June 10–July 8 | 331 | 2,360 | 27 |
| One canonical complete corrected-WIN snapshot | 331 | 2,360 | 27 |
| Older fixed-window extract present | 291 | 2,059 | 26 |
| Two complete box sets in that extract | 291 | 2,059 | 26 |
| Timing-only pair meeting fixed windows in the retained union | 291 | 2,059 | 26 |
| Two complete snapshots individually bound to corrected WIN | 0 | 0 | 0 |
| All movement requirements satisfied | 0 | 0 | 0 |

These counts are regenerated in [qualification.json](market_movement_20260929_evidence_verified/qualification.json); all 331 race-specific inclusions and overlapping exclusion reasons are in [race_qualification.json](market_movement_20260929_evidence_verified/race_qualification.json). The denominator is the manifest-admitted development population, not the larger historical extract. Labels and histories were unnecessary and were not decoded by this audit. All source file hashes and scope checks are retained. A complete box set here is a numeric roster match; it does **not** prove the same active named runners or absence of scratchings at both observations.

The [protocol](market_movement_20260929_protocol.md) fixed scheduled T−2 before source-row inspection. The late point is the latest eligible snapshot available by cutoff, maximum age eight minutes; the early point is the earliest eligible snapshot within T−30 and at least two minutes earlier. Timing-only inventory applies those time bounds to capture timestamps and does not relabel capture time as availability time. Scheduled jump is not independently observed actual off.

The exact earlier-source extract contains 4,118 rows in the permitted population. Exact source-row IDs link 2,059 late rows to the canonical corrected-WIN sidecar. The other 2,059 early rows carry a legacy `market_type=win` label but no runner name/native identity, raw paired WIN/PLACE evidence, availability timestamp, active/suspended status or scratching history. Existing box/source URL/capture information cannot replace those missing fields. The canonical late sidecar proves the named runner and corrected WIN price for its own observation only. Both sources lack independent availability and explicit scratch/market-status evidence for all 331 races. Forty races have no intersecting earlier extract. No incomplete box sets were found among the intersecting extract snapshots; this does not establish unchanged active fields.

This audit deliberately stops before database reconstruction: recovering earlier raw WIN evidence alone would not repair independent availability or scratching provenance. It is a finite audit of the two retained surfaces linked by the existing research, not an assertion that every local archive has been exhaustively searched. No provider requests, database opens, new collection, services, production configuration or October-study records were involved.

What is new relative to prior work is this exact protected-scope intersection, T−2 qualification and explicit separation of timestamp/box coverage from usable market information. [PR #193's report](offline_systematic_20260924_results.md) already identifies the canonical matrix's latest-prejump selection followed by a 2–10-minute filter: inclusion can depend on later observations and is a retrospective availability-selected benchmark, not a live T−2 coverage replay. The older [fixed-window implementation](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/scripts/run_sportsbet_market_structure_experiment.py) extracted T−30/T−10 rows and previously tested log/probability/rank movement; those recipes are not claimed as new. Its [protocol](/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/prejump_market_structure_experiment_20260815_report_only/protocol.json) identifies the source extract and its hash. The subsequent [WIN provenance audit](/home/l4nd0/greyhound/artifacts/sportsbet_win_market_surface_audit_20260815_report_only/REPORT.md) required rebuilding that surface because its legacy WIN labels included PLACE/misparsed prices. No historical movement performance from that invalidated surface is reused. [PR #195's audit](/home/l4nd0/greyhound-early-speed-neighbours-20260928/scripts/audit_early_speed_neighbours.py) supplies the identity-before-decode discipline; it ran no early-speed fit and does not qualify market movement.

## Minimal retention specification and operational handoff

This is a written design handoff, **not a collection instruction or study amendment**. The operational owner would need separately allocated records outside existing protected populations and permission for acquisition. Production and the frozen October comparison remain unchanged.

1. Retain a manifest of every scheduled eligible target and every attempted/missed observation, independent of later outcome or collection success. Bind the canonical race key, provider-native race/market IDs, scheduled-jump version and timezone, acquisition source, price-origin bookmaker, WIN/fixed-price classification and parser version. Preserve source response bytes/hash and receipt hash; never overwrite earlier observations.
2. For every snapshot retain the full provider runner roster with stable native runner ID and mapped canonical identity, box/reserve status, active/scratched/suspended flag and WIN price validity. Retain scratching/change events with source effective time when supplied and collector observation time, and market status at both points. Exclude changed or unproved fields rather than silently dropping runners and renormalizing.
3. Retain timezone-aware request start, response receipt/observation time, durable availability/commit time, source quote time when actually supplied, scheduled decision cutoff, and monotonic sequence/clock provenance. Never backfill a missing availability time from file mtime or scheduled collection slot. All model inputs must be demonstrably available by the fixed cutoff.
4. The minimum useful cadence is one complete snapshot around T−10 and another around T−4, both durably available before T−2. Retain actual times and all failures: slot names are not timing proof. Optional denser snapshots require separate owner budgeting and are unnecessary to specify here. The unchanged selection rule uses the latest eligible late point and earliest eligible earlier point; require at least two minutes separation, early inside T−30, late no older than eight minutes at cutoff. Missed/late points mean exclusion, not retries inferred from this document.
5. Compute normalized probabilities separately within each unchanged active field, retain both overrounds, and inspect log probability movement and elapsed-time-normalized movement. Margin shifts alone must not create runner information. Before accessing future labels, freeze eligible races, a chronological evaluation schedule and a sample/date adequacy criterion against a stated minimum meaningful proper-score improvement. Aggregate collection health first; do not promise statistical power from these 291 unqualified historical pairs.

No live integration or generic new framework is needed to establish the present evidence gap. The small offline auditor directly verifies the retained population, timestamp windows, source-ID cross-binding and exclusions.

## Reproduction and attempts

Run from the isolated worktree using the existing research interpreter; use a fresh output directory because prior audit outputs are never overwritten:

```bash
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$RESEARCH_PY" -m scripts.audit_retained_market_movement --output /tmp/market-movement-reproduction-unique
PYTHONDONTWRITEBYTECODE=1 "$RESEARCH_PY" -m unittest tests.test_audit_retained_market_movement -v
```

Eight synthetic targeted tests pass: protected malformed payload skipped before decode; unadmitted runner rejected before decode; admitted identity permitted; conflicting identities fail closed; timezone required; decision cutoff ignores later quotes; age/spacing boundaries; duplicate/missing boxes rejected. No broad suites or package installs ran.

The [final ledger](market_movement_20260929_evidence_verified/trial_ledger.jsonl) retains zero-fit completion and tool/setup failures: unavailable generic `python`, missing incident-manifest path in the new worktree (repaired to the pinned original), and absent pytest in the pinned interpreter (used standard-library unittest). The [first audit](market_movement_20260929_evidence/trial_ledger.jsonl) is preserved. The second audit added executed-script/protocol hashes; qualification counts were unchanged. All attempted analysis variants: one qualification protocol, zero movement performance variants. No uncertainty interval or leave-one-date-out performance sensitivity exists for zero qualified pairs; fabricating one would imply observations the audit did not establish.
