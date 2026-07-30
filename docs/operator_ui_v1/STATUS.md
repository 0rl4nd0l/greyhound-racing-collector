# Greyhound Operator UI V1 status

Observed 2026-07-30 before this uncommitted `GHU-002` candidate:

| Field | State |
|---|---|
| Repository | `0rl4nd0l/greyhound-racing-collector` |
| Branch | Canonical integration branch `agent/operator-ui-programme-20260730`; this uncommitted candidate was produced on an isolated Codex X worktree branch |
| HEAD / tree | `aa4b45a004b1f897fc5d4cd06b0a741be6cd2446` / `b34eeefdff0ebf74cc4f38d99652bea848488abc` |
| Baseline cleanliness | clean |
| Upstream base | `origin/master` `51a5287dfc28c8d059df2768534498c4b6321230`, merged by `6f4fba42c45c73702efb017a21cbd284b44c1d04` |
| Current release / ticket | R0 / `GHU-002` active, review required |
| Counts | 2 accepted tickets, 1 active, 2 deferred, 26 planned (plus accepted audit milestone `GHU-000A`) |
| Next dependency-ready action | After `GHU-002` independent acceptance and exact parent integration, the `GHU-010` bounded candidate must atomically record exact `GHU-002` commit/tree/reviewer/parent-decision evidence, then `GHU-010`'s legal `ready -> active` transition before product work and its `active -> review` plus focused validation before freezing the same product-and-ledger delta |
| Validation | Candidate documentation-only focused validation passed; independent review pending; no broad forecasting suite |

`GHU-000` was accepted at original base
`9be52ecd589615b4ebd6212bd9595be761520b89`. `GHU-000A` accepted source-delta
audit run `20260730T092421Z-6f4fba42c4-baf67b`, session
`019fb257-217f-7d10-8409-b2a06a6bd20b`.

The accepted `GHU-001` contract is commit
`aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, parent
`6f4fba42c45c73702efb017a21cbd284b44c1d04`; file SHA-256
`b2b4af016b24dafa4f121f0415c0d948a4c0699c73586a235c6547fc3002512b`;
reviewed base-to-result diff SHA-256
`22df513812a6460fbae2f247a36bfca4f37536438d5eed6eabebda47b2774b30`;
implementer session `019fb267-ff74-71e3-bb91-2ed8d55be316`; independent reviewer
session `019fb26b-ae76-7411-91ae-b54d5514a137`.

Supported now: the accepted repository/UI/runtime authority inventory and the
exact source/evidence/authority contract. Unsupported now: an implemented UI,
UI operations store, live dashboard/API, deployed identity, runtime health,
manual UI prediction, runtime prediction proof, corpus/model readiness,
market edge, training, promotion, EV, staking, betting, or public availability.

No push, PR, merge, deployment, runtime mutation, or live prediction has
occurred in this programme yet. Publication, merge, deployment, and runtime
proof are all `NOT_OCCURRED`. `GHU-002` remains active/review-required until an
independent review and the exact parent integration commit occur. There is no
genuine programme stop currently evidenced.

This is intentionally the pre-integration snapshot: it does not preclaim
`GHU-002` acceptance or change `GHU-010` from `planned`. Parent integration of
an accepted exact delta is atomic with its ledger updates; rejected candidates
do not mutate the integrated ledger.
