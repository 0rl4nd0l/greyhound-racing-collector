# Operator UI R3 Ticket 02 candidate validation

Status: `PASS`

The repository-owned generator produced an enabled, unactivated R3 candidate
from authoritative commit `d343af94a57af80327dd41f18433f7466f86ca0d`
and tree `a04197ba455de2549f9c76adcd474d1feb520bd1`.
Per the generator's repository contract, the four transactional outputs span
two fixed roots: the binding is written under the authoritative source's
`var/operator_ui/generated/` directory, while the environment, unit, and
rollback files are written into the new empty candidate package directory.

The fresh `operator_ui_live_authority_v1` observation was captured at
`2026-08-10T20:28:33+10:00`. It binds the installed collector units and actual
full/odds service states, the current full report/state, the latest completed
generator-admissible odds report and its same-run refresh evidence, the
2026-08-10 report-only race inventory and raw packets, and a freshly generated
model catalog and deployment manifest. No older candidate authority was reused.

## Candidate

- Source: `/home/l4nd0/operator-ui-r3-ticket01-authoritative-HCOLll/source`
- Package: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-ticket02-candidate-20260810-xoXmLv`
- Operations: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-ticket02-operations-20260810-IX5u7P`
- Authority: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-ticket02-authority-20260810-CPDtYE/live_authority.json`
- Bind: `127.0.0.1:5055`

The unit mounts the authoritative source, collector evidence, producer root,
and canonical racing database read-only. Only the new Ticket 02 operations root
is writable.

## Output hashes

- Source binding `var/operator_ui/generated/repository-v1.binding.json`: `6e8d1966d1fe6a18faad40cf02babf4e0bdf0a81db915d776b50040dc99b8e46`
- Candidate package `operator-ui-r3.env`: `37b8abe312b839d8ecb4b74b194fcb408d05583145c140a3512e2f274e5c8b86`
- Candidate package `greyhound-operator-ui-r3.service`: `14635fc14f244d93b3de1c956aef7c0c973d9fa4ea66fe1ee95edec48d34d655`
- Candidate package `ROLLBACK.md`: `39e434f6a9665a446b3fb5160c02cc7b0ed1c737c9764de36f5de6715d96646d`

`systemd-analyze verify` exited 0. It reported only unrelated host warnings
about `netplan-ovs-cleanup.service` and the host's `snapd.service` syntax. The
user-manager form could not initialize in this session.

Ticket 01's unchanged focused seam results were not repeated: 112 deployment
generator tests, 2 R3 end-to-end safety tests, 114 frozen-model fixture and
portability tests, and the focused selector-resolution test all passed under
Python 3.11.15 and NumPy 1.26.4. See the Ticket 01 evidence packet for the exact
commands and repository-wide collection limitation.

No installed unit or service was changed. The rejected active R3 runtime
remained active with PID 1692390 throughout final validation.
