# Seed Evidence

The builder independently re-hashed and parsed four recorded artifacts. The
source worktrees are dirty and the artifacts are ignored, so content hashes and
validated semantics—not Git tracking, mtime, or conversational claims—are the
authority.

| Stable decision | Serialized evidence SHA-256 | Validated semantics |
| --- | --- | --- |
| TheDogs historical-source floor | `f878b67628e8f462f3aa7578b9a65e011a11a0b738479f6e18c2e22dffb786dd` | 663 complete races, required floor 300, report-only source class |
| Aggregate challenger | `e935c9c65dafd1355cedf24fd0aaf646b800bc92d41a682d6d4ea2c35c1d5da6` | 663-race split, nine models, no qualifier, `KEEP_BASELINE` |
| Strict Sportsbet same-floor snapshot | `sha256:5e4c1f0d3bba8fe5c9bf7368f9269005db639e9bc9cdd9b0578d357988f6d910` | deterministic composite of the evaluation and strict-overlap artifacts; recorded snapshot remains `DATA_MISSING_FLOOR`; source class stays separate |
| Historical identity bridge | `bd5557dc66a2c29e50e674551e84ec0bdac5baaf91e412e61a05f9b2a6a67efd` | `REPORT_ONLY_BRIDGE_READY`, `COPY_REPAIR_BLOCKED`, no writes |

The strict Sportsbet composite binds both recorded inputs in manifest order:

- Evaluation: `sha256:e935c9c65dafd1355cedf24fd0aaf646b800bc92d41a682d6d4ea2c35c1d5da6` at `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-csv-history-feature-repair-scaffold-20260709/reports/agent_jobs/thedogs_published_market_large_csv_history_challenger_20260709/evaluation_results.json`
- Strict overlap: `sha256:870067e6f4024647162265ebcf850850a855c3314e6c5c1e008627d0624f3b85` at `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-csv-history-feature-repair-scaffold-20260709/reports/agent_jobs/thedogs_published_market_large_csv_history_challenger_20260709/strict_sportsbet_overlap.csv`

The prospective overlap count is checked while building the historical seed
but is deliberately absent from the serialized decision identity and claim.
The seed contains no timer state, current capture count, current prospective
count, service state, or deployment claim.

The generated manifest contains exactly four entries and passes the merged
Tenn decision-ledger validator. It is not appended to the shared Greyhound
ledger until the reviewed pilot is merged.
