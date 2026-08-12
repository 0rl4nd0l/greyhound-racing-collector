# Operator UI R3 post-merge deployment handoff

Handoff state: `MERGED_DEPLOYMENT_PENDING`.

PR [#131](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/131)
was squash-merged into `master` at
`98281c22d1ab9fcd28df0119e32ef5fa37be5836`, tree
`29cf2be115600a73787f2993ab8907b45c4acf6b`, on
`2026-08-12T16:29:13+10:00`.

The merged source changes Operator UI R3 job admission from `auto` to the
sealed `receipt` odds source. The connected UI offers only `receipt`, the
backend rejects `auto`, and accepted jobs persist `odds_source=receipt`.
Tickets 01–04, the R3 specification, and ADRs 0009–0017 are preserved in the
merged history.

## Validation and review record

- The focused Operator UI suite passed locally: `50 passed`.
- Targeted receipt-only TDD and cleanup checks passed.
- `git diff --check` passed.
- Final Standards review: no findings.
- Final Spec review: no findings.
- Full forecasting CI did not complete. The owner explicitly authorized an
  admin bypass; the remaining runs were cancelled and that bypass is recorded
  in the squash-merge message. Do not represent the merge as full-suite green.

## Runtime boundary

No deployment, service restart, runtime database mutation, collector control,
external fetch, prediction submission, EV calculation, or betting action was
performed by the merge workflow. The previously deployed R3 runtime remains
historical Ticket 03 state and must not be assumed to contain the receipt-only
change until a new candidate is installed and verified.

Ticket 04 remains terminal `DATA_MISSING`. Do not retry that attempt or reuse
its one-attempt authority.

## Next authorized work

1. Create a clean, isolated checkout of exact merge commit `98281c22`.
2. Use the repository-owned deployment generator to create a new, uninstalled
   R3 candidate. Do not hand-edit generated or installed files.
3. Bind the candidate to the exact source commit/tree, pinned Python runtime,
   model, manifest, schema, manual-default configuration, and a fresh
   contemporaneous live-authority observation.
4. Run the focused deployment-generator, R3 safety, and frozen-model
   compatibility checks; record exact commands, exits, versions, and hashes.
5. Run `systemd-analyze verify` and review the environment, unit, binding,
   manifest, live authority, writable paths, and rollback package.
6. Stop before installation if any identity, test, authority, or unit gate
   fails. Preserve exact fail-closed evidence.

Installation is a separate operational gate. After candidate acceptance,
replace and restart only `greyhound-operator-ui-r3.service`, then verify the
installed identity, loopback listener, authentication, capability, overview,
system read-only endpoints, and receipt-only submission contract without
submitting a prediction.

A new Ticket 04 attempt requires fresh explicit authority plus one valid
collector-owned current-race index and one matching fresh sealed receipt. It is
limited to one report-only receipt job with no capture, fetch, fallback, retry,
alternate race, EV, or betting.
