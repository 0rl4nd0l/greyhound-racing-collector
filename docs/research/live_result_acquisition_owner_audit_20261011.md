# July result acquisition ownership audit — 11 October 2026

**The approved 210-request result grant does not currently have an installed owner-admitted execution route.** The live collector owns the shared campaign mutex for its entire lifetime. Its installed code has no result-purpose delegation interface. An empty source-operation slot or a temporarily absent collector worker lock does not grant another process the campaign owner role.

This was a read-only audit. No provider request, source operation, lock acquisition, service change, allocation change or control-file mutation was attempted. Existing result, study and engineering consumption remains untouched.

## Actual installed owner

Systemd reports `greyhound-persistent-collector.service` active/running, PID **2848247**, started 10 October at 19:06:19 AEDT. Its working directory is `/mnt/tenn-nvme2/tenn/greyhound-reserve-exclusion-20261010`, with source commit `55f728a394d4e10728f2ef8f033bd7f98d6bd384`. The actual command invokes `scripts.run_persistent_collector` with `/mnt/tenn-nvme2/tenn/greyhound-persistent-reserve-exclusion-controls-20261010/collector.json` and configuration hash `377a586701986da00df17025b82c71234b50d5a50d968614c0e47dcbc60eac37`.

The kernel lock table independently proves ownership:

```text
FLOCK ADVISORY WRITE 2848247 103:01:27727813 0 EOF
```

`/proc/2848247/fd/5` points to `/home/l4nd0/greyhound-collector-campaign-20260923/owner.lock`, inode 27727813. No conflict probe or new descriptor lock attempt was needed. The installed `race_collection/persistent_collector.py` opens and exclusively locks this file at lines 550–551 before its long-running loop, preserving ownership even during `WAITING_FOR_RACE`.

At the inspected health snapshot, `2026-10-11T05:24:20.496869+00:00`, the collector was `HOLD`, had an `odds` child, and reported `result_access: false`. This is a snapshot rather than an enduring claim; process health and active children may change independently of this audit.

## Source and result interfaces

The retained source state `/home/l4nd0/.local/state/greyhound/sportsbet-access.json` was `OPEN` with no active operation. Its current diagnostic authority is the collector's 11 October persistent daily allocation, hash `87b0f24830ca514614ee0099235865b5d4a7295b11ec6201bd2cd26144b75ad9`. That allocation is engineering-only and permits zero result requests.

Calling the installed `SportsbetAccess.check_admission()` without impersonating the collector's allocation, a documented read-only preflight, returns:

```text
SportsbetAccessBlocked sportsbet_persistent_owner_required
```

The code verifies the persistent lease and requires the genuine owner allocation tag before source admission (`utils/sportsbet_access.py`, `_check_diagnostic`). Copying its environment tag would not create a result-purpose grant and would incorrectly use engineering source accounting. The new user grant must have its own durable result consumption; it must not replace or reopen this allocation.

The existing `src/predictor/comparison_result_runtime.py::collector_guard` independently opens the same campaign `owner.lock` and requests `LOCK_EX | LOCK_NB` before acquiring the collector lock or entering transport. It is therefore not a concurrent delegation mechanism: the kernel-proven current owner blocks it before a GET. Existing comparison result plans also bind their own populations and budgets, so they cannot be repurposed for the July population.

The installed collector entrypoint and implementation contain no `owner_fd`, result-owner, fair-result or delegate hook. The normal daily owner handles full capture, odds capture, inventory and predictions, with `result_access: false`; no existing result-purpose child slot was found.

## A prepared route exists, but is not installed or in this grant

The previously retained fair-comparison audit accurately documents the same lifetime ownership constraint:

`/mnt/tenn-nvme2/tenn/greyhound-fair-comparison-20261011/docs/research/fair_result_source_authority_20261011.md`, lines 54–67.

Its separately prepared optional coordinator can pass the genuine locked owner descriptor to a bounded child after an idle tick, fresh schedule and lead-in checks, no collector children or active prediction, current allocation validation and a conflict probe. However, the patch and policy are **pending runtime**, bind a different fair-comparison programme, and are absent from the inspected installed release. Root has contacted the actual operational-owner thread and reports that owner is preparing a separately approved earlier handoff today. That new coordination is pending; its new timer or delegation admission must not be assumed to cover the July grant. The separately inspected fair-programme approval and budgets cannot be borrowed for the 210 July requests.

## Smallest remaining action

The result purpose, population and finite GET allowance are now approved. The remaining blocker is owner coordination implementation, not missing user permission to inspect July outcomes. The current operational owner needs an explicitly authorised, narrowly scoped integration that admits the new result grant through its genuine owner descriptor while preserving its engineering budgets and all existing source holds. It must yield between requests and refuse when a collector child, prediction, source operation or protected race lead-in is active.

At the time of this inspection, that integration was a runtime/source change outside the approved scope. The user has since approved safe recovery and activation of the reviewed July interface; see [the later execution update](live_result_acquisition_status_20261011.md). There was no verified execution route through the previously installed interfaces. Operational-owner thread `01a11a4d-08df-7c71-b9c1-d092513b8b2b` now owns the approved recovery and activation. The new approval does not permit an unchanged GET loop, borrowing an old plan, impersonating the engineering owner, or stealing a lock. Reverify the actual installed interface and its exact July-grant binding before acquisition.

The contemporaneous machine-readable snapshot, including the approved grant hash, kernel owner proof, current service command and source projection, is retained at `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/acquisition/owner-admission-inspection.json`. It explicitly distinguishes approved result purpose from pending owner admission and records zero requests.
