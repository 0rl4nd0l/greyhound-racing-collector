# Validation

## Contract And Preflight

- V2 task-card validation: passed; scope fingerprint
  `ae428870f9dcc54ad6cc7cbf55b9cd5cf3cd253703774e7db336e4c3a1d453e3`.
- Shared Greyhound decision ledger: explicitly initialized empty, then
  validated.
- Active V2 claim: passed in the shared Git common-dir registry.
- Portable card-only preflight: `ALLOW_NEW_SCOPE`; decision-ledger and registry
  `PASS`; absent legacy task ledger reported as a warning.
- Negative Stop-hook proof: correctly blocked while `RUN_OUTCOME.json` was
  absent.
- Tenn cross-repository enforcement PR #507 merged as
  `ac5a56c142ee1a9781ae54ad0e8857ba5510f7d1`; the stable canonical and
  installed physical guard copies match SHA-256
  `bba72ec16c726265609208e8c1e9281b11189dec289661a21669437aed1f0da4`.
- Installed host guard suite: 40 passed.
- No-environment portable preflight selected Greyhound `origin/master` at
  `224dc2dddace3c50d131ada2e078a980090a7b3a`, the stable Tenn control plane,
  and the shared Greyhound decision ledger; semantic status was
  `ALLOW_NEW_SCOPE`.

## Seed And Focused Tests

- Four real artifact hashes and required semantic predicates: passed.
- Generated JSONL: exactly four entries; Tenn decision-ledger validation
  passed.
- Deterministic regeneration with `--check`: passed.
- Focused pytest with repo application conftest disabled and PyYAML supplied:
  11 passed in 1.56 seconds through the installed host guard, with all Tenn
  selector environment variables unset.
- Portable integration: exact floor scope returned `REUSED_COMPLETE`, denied
  report writes, and created no report; changed dataset returned
  `ALLOW_CHANGED_EVIDENCE` with substantive work admitted. The fixture uses a
  bare remote named `greyhound-source`, remote default `master`, and a
  self-published tracking topic; the test ran against the corrected Tenn guard
  through the synced installed host skill, without an override.
- Sportsbet composite tests: changing either evaluation or overlap evidence
  changed both evidence hash and scope fingerprint.
- Bridge exact-key and single-read hash/parse regressions: passed.
- Ruff over builder and focused tests: passed.
- `py_compile`: passed.
- `.codex/hooks.json`: valid JSON and not ignored.
- Task-card `check-diff`: passed with no disallowed files.

## Publication And Post-Merge Pending

- GitHub checks and merge ancestry.
- Post-merge append of exactly four seed decisions and exact 663-race reuse
  proof.
- Review of the first five non-trivial post-pilot runs before broader adoption.
