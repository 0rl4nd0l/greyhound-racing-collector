# PR #46 effective-state integrity repair

Final status: `PR46_REPAIR_READY_FOR_INDEPENDENT_MERGE_REVIEW`

The source-proven defect allowed post-load mutation of nested model state to
change predictions while cached artifact hashes and the shadow record key
remained unchanged. The repair deep-freezes the loaded contract, stores
score-affecting NumPy state in non-aliasing read-only buffers, verifies a
canonical effective-state digest before scoring and append acceptance, and
makes the append writer rescore from source inputs before reconstructing the
record identity and bytes it will write.

The canonical model SHA-256 remains
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`.
The manifest SHA-256 remains
`8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`.
No fit, artifact, candidate, feature, weight, database, outcome, cohort,
runtime, service, timer, deployment, activation, promotion, or merge mutation
occurred.

## Runtime Functionality Proof

- Intended output: a repository-only repair on the existing draft PR #46; no live output was authorized.
- Live output location: none created or changed by this task.
- Pre-run max timestamp or count: `DATA_MISSING`; production database access was forbidden.
- Post-run max timestamp or count: `DATA_MISSING`; production database access was forbidden.
- Rows/files inserted or updated after run start: zero production rows and zero runtime files by this task.
- Readiness/gate status: `PR46_REPAIR_READY_FOR_INDEPENDENT_MERGE_REVIEW`; production remains `KEEP_BASELINE / market-only implied probability`.
- Exact command/query used: focused and resource-lock pytest suites, Ruff, format check, `py_compile`, artifact `sha256sum`, source-proven mutation probes, and clean post-PR45 integration simulation; no database query was executed.
- Result: `PARTIAL`.
- Remaining blocker: the separately activated runtime branch still calls the old two-argument writer API and must be adapted only in a future owner-approved deployment task after this repair is independently reviewed and merged.

The installed timers were read only and were active/waiting with their latest
services successful. Their external runtime worktree was not changed.
