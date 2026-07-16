# State

- State before: `pr45_merged_pr46_blocked_on_source_proven_effective_state_identity_mutation`
- State after: `pr46_effective_state_repair_ready_for_independent_merge_review`
- Outcome: `ADVANCED`
- Owner terminal state: `PR46_REPAIR_READY_FOR_INDEPENDENT_MERGE_REVIEW`
- Authorized old PR head: `106fbc09c6d9e4943365c2c1034b09575031ec2e`
- Post-PR45 master: `c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa`
- Repair publication: one normal descendant commit to the existing draft PR #46
- Model SHA-256: `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`
- Manifest SHA-256: `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`
- Effective-state SHA-256 on the fixed fixture: `97da118363975ae63183a81b7d7773b7c6b7aff8377239703288a7c0f4bea95f`
- Fit, artifact, database, outcome, runtime, service, timer, activation, deployment, promotion, or merge mutation: none
- Production state: `KEEP_BASELINE / market-only implied probability`

The exploit was reproduced before the repair: predictions changed while the
model hash, manifest hash, and record key remained unchanged, and the writer
accepted the mutated score. After the repair, direct nested mutation fails,
forced array mutation fails effective-state verification, mutation paired with
a forged effective-state key fails the encapsulated-state comparison, and a
mutated record fails writer-side canonical rescoring.

The live task ledger was unavailable, so preflight used
`DATA_MISSING_FALLBACK_CHECKED`; the shared claim registry and decision ledger
remained available. The installed runtime was read only. It now points to the
external worktree `greyhound-early-residual-shadow-activation-v1-20260716` at
`f776bfd142b1e8acd3befca330eee36f490402ed`; its timers were active/waiting and
latest service results were successful. That runtime still uses the original
writer signature and was not altered by this repair.
