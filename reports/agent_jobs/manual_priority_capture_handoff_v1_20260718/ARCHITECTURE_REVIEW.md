# Architecture review

## ARCHITECTURE REVIEW

### Change: finalized autonomous capture to manual priority reader handoff

The host `architecture-check` skill expects a `.cursor/rules/` corpus for
embedding, vector-store, RAG, and backend-service invariants. This repository
does not contain that corpus, and this change touches none of those systems.
Those skill-specific checks are therefore `DATA_MISSING`, not silently
approved. The applicable Greyhound boundaries were checked from `AGENTS.md`,
`ARCHITECTURE.md`, issue #50, and the exact source contracts.

| Rule file or source | Section | Status | Explanation |
|---|---|---|---|
| `.cursor/rules/00_mandatory_index.md` | mandatory index | `DATA_MISSING` | File is absent in this repository. |
| `.cursor/rules/backend_architecture.md` | backend services | `DATA_MISSING_NOT_APPLICABLE` | No backend service or API boundary changes. |
| `.cursor/rules/embedding_rules.md` | embeddings | `DATA_MISSING_NOT_APPLICABLE` | No embeddings. |
| `.cursor/rules/vector_store_invariants.md` | vector store | `DATA_MISSING_NOT_APPLICABLE` | No vector store. |
| `.cursor/rules/failure_policy.md` | failure policy | `DATA_MISSING` | Repo-local fail-closed contracts substituted below. |
| `AGENTS.md` | live daemon safety | `COMPLIANT` | Reuse path never acquires, releases, deletes, steals, or bypasses the writer lock. |
| `AGENTS.md` | live odds guardrails | `COMPLIANT_WITH_TEST_GATE` | Final report plus exact DB evidence is required; both WIN and PLACE are compared. |
| issue #50 | standalone tracer bullet | `COMPLIANT` | Change stays inside the existing standalone CLI and remains plan-only by default. |
| issue #50 | runtime boundary | `COMPLIANT` | PR #48, services, timers, outcomes, models, deployment, and GitHub remain untouched. |

### Invariants

- Discover through exact target-bearing plans, never broad unrelated report
  parsing.
- Require one finalized report and one exact currently due fixed window.
- Bind plan, report, form, sidecar, append metadata, WIN rows, PLACE rows, and
  SQLite provenance in one read-only consistency decision.
- Read selected source files once and score staged private bytes.
- Keep direct capture as the fail-closed fallback when no receipt is available.
- Claim consistency at use time only; do not claim historical authentication.

### Summary

The architecture-specific host corpus is unavailable, so its verdict is
`DATA_MISSING`. Against the actual Greyhound architecture and issue contract,
the selected three-file extension is `COMPLIANT_WITH_PERMANENT_TEST_GATES`.
No migration, daemon edit, or runtime activation is authorized.

