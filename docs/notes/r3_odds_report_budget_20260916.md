# R3 retained odds-report budget repair — 16 September 2026

## Approved scope and fixed review base

The owner approved a focused offline compatibility repair after PR #181's
production package was rejected. Base:
`2e5a2040024d32fd00b747fac1a5191b14f45113`.
No merge, deployment, restart, collector change, outcome access, activation,
model/feature/protocol change or historical-record mutation is part of this repair.

Required behavior: preserve complete retained pre-jump provenance through package
generation, startup and the configured evidence reader under justified finite
budgets; preserve fail-closed identity, hash, path, shape and disclosure rules.
Review against this base and this specification, plus the owner's conversation.
This is an engineering resource contract, not a research protocol amendment.

## Retained failure and supporting observations

Original release record:
`/home/l4nd0/greyhound-r3-181-rollout-20260916.jABWAw/ROLLOUT_BLOCKED.md`.

The 20260916T132209+1000 odds refresh is 596,627 bytes, SHA-256
`3730a78a4b1e320dfd3ba9d23e16f0eefb37a59d7ae7246056a495fc57ded657`.
Package generation rejected it above the old 524,288-byte budget, before publishing
a binding or installing anything. Its containing odds report is 626,883 bytes,
also above that budget. Both roles need the compatibility correction.

An outcome-blind file-size census of available refresh reports dated September
15–16 found, at observation time:

| File-date group | Files | Median bytes | Maximum bytes | Above 512 KiB | Above 1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| September 15 | 183 | 443,585 | 1,960,709 | 51 | 13 |
| September 16, partial day | 73 | 16,870 | 1,676,256 | 13 | 4 |

These are **report-file** denominators, not race coverage or result delivery.
Full and odds-only refresh reports are included. Files are not independent
observations and the partial day is not a representative distribution.
A separate same-date stat-only observation found 1,659 odds-only reports,
maximum 652,342 bytes (41 above 512 KiB), and 73 full-run reports, maximum
351,585 bytes (none above 512 KiB). Discovery and counts were not outcome-selected.

Only the failed refresh/containing report and the size-selected largest refresh
were structurally inspected. No embedded HTTP body was decoded or displayed.
The failed refresh has 2,092 JSON value nodes, root-zero depth 9 and a largest
string of 64,140 bytes; its containing report has 2,712 nodes and depth 10.
The largest refresh has 7,637 nodes, depth 9 and largest string 67,144 bytes.
Large strings are retained native-identity HTTP `body_base64` evidence, not an
unbounded log-padding assumption. The existing reader's 4 KiB scalar budget
would reject these even if the file-byte gate alone were enlarged.

## Smallest compatible finite contract

Only `odds_refresh` and `odds_report` receive:

- total retained JSON bytes: **2 MiB (2,097,152)**;
- per JSON string/key at runtime: **128 KiB (131,072)**.

2 MiB is the smallest power-of-two envelope covering the observed 1,960,709-byte
refresh. 128 KiB covers observed 64,140/67,144-byte retained HTTP strings with
bounded headroom. This is an explicit operating cap, not a claim that future
collector reports, or every embedding of a near-cap refresh, will fit. A containing
report still has its own 2 MiB total limit; exceeding either remains failure.
Do not dynamically grow these budgets, truncate raw reports, choose older/smaller
reports, or synthesize a replacement provenance packet to pass a release gate.

All other source file limits remain 256/512 KiB. Other runtime string limits
remain 4 KiB. Existing runtime depth 12 and item count 100,000 remain unchanged.
Package, startup retained reads and configured reader use the same source-byte map;
the configured reader applies the new per-source string map. Package/startup
byte acceptance is NOT a declaration of semantic validity: strict shape,
serialization, item/depth/string, hash and lifecycle checks still run at evidence
consumption. No outcome or receipt admission rule is relaxed.

No reader implementation, no-follow/owner policy, hash binding, canonical
serialization, immutable source identity or timeout is changed. Raw HTTP bodies
remain opaque and absent from the collector response.

## Evidence-driven validation plan and recorded distinctions

Test seams agreed in the approved repair: existing package generator, real
repository startup composition, public collector observation and evidence reader.

- Two package regressions fail before the production change on the retained
  596,627-byte size, one for each source role.
- The packaged public collector regression then fails with byte-only repair:
  its 64,140-byte HTTP strings are rejected. The per-source string correction
  makes it pass. This proves why only changing file size was insufficient.
- Exact 512 KiB and 2 MiB package/startup acceptance; one-byte-over package
  rejection before any generated output, with unchanged small-source budgets.
- Runtime near-cap / exact-string-boundary success; one-byte-over string,
  excessive depth/items, source growth and post-startup tamper rejection.
- A valid-byte tamper is `DIVERGENT`, not malformed `INTEGRITY_FAILED`.
  An initial negative test expected the latter; its assertion was corrected,
  not production behavior. The failed run is retained.
- Near-cap startup + public collector reading is measured within pytest with a
  64 MiB traced-allocation regression ceiling. This is a synthetic-case resource
  check, not total process RSS, a worst-case adversarial proof or a live latency
  guarantee. No time limit is increased or new timing guarantee claimed.
- Default journal-off environment, no job-store creation and no raw HTTP-body
  disclosure are checked in the packaged path.

The first affected-suite run had 568 passes and four failures in unchanged
fixed-reader byte-limit fixtures. All four failed at bootstrap.py's
foreign-owned group-writable ancestor guard, before their byte assertion.
Measured host facts: effective UID 1000; /tmp UID 0 mode 1777; pytest descendants
UID 1000 mode 0700. This is a proven invalid fixture location under the existing
policy, not a directory-mtime race or a hypothesis based on a passing rerun.
Only those four fixtures now use private directories below the resolved test
repository, independent of CWD. Production ownership checks remain unchanged.

Retain all red, corrected-fixture and final-run results. Do not count synthetic
acceptance as production deployment or prospective prediction-to-result proof.
The exact-head final validation/review record lives under
`/home/l4nd0/greyhound-r3-report-budget-20260916.QmcHqM/`.
No original failure, attempt, activation or protected scientific record is edited.

## Release boundary

The installed release remains #180 / 1570dbad, not merged #181 or this repair.
Publication, reviewed exact-head merge and actual merged-source deployment require
their staged authority. A new production candidate must receive fresh authority
and pass its actual package/startup gate; no stale authority is reused.

September stays protected and unchanged. October's separate conditional one-job
approval is inactive and unscheduled. The repair does not create an activation,
schedule, prediction or result-access opportunity, nor demonstrate predictive value.
