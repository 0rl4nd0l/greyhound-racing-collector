# PR review

- Draft PR: #48, `Run frozen residual shadow scoring before odds lock release`
- URL: https://github.com/0rl4nd0l/greyhound-racing-collector/pull/48
- Head: `cbbe78a2103da18a381263a9a2874ce02f243fbf`
- Base: `codex/manual-live-residual-prediction-v1-20260716`
- State: open draft, clean, mergeable
- Current checks: two `hardening` checks successful
- Merge: not performed

PR #47 remains open, draft, clean, and mergeable at exact head
`097002a7561e9895dccfb593d709c4c4063b78c4`; all five of its checks are
successful. Its branch was not mutated by this task.

PR #45 is merged at exact head
`aa35fa70fc49199acde09f5561b521ddb00d45aa`; all five checks were successful.
That head is an ancestor of PR #47 and PR #48. The installed full service still
has CPUWeight 20, IOWeight 20, and idle I/O scheduling; the odds-only service
still has Nice 10 and best-effort I/O scheduling.

Canonical activation prerequisite: merge PR #47 first, retarget/rebase draft
PR #48 onto the resulting `master` without dropping PR #45 ancestry, rerun the
full required checks, obtain owner review, and only then consider merging PR
#48. Production model replacement, promotion, and betting remain separate and
unauthorized.
