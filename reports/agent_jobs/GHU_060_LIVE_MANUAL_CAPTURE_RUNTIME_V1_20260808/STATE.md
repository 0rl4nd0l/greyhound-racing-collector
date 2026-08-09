status: VALIDATING
base: 0a67f95ea06effa04609faabe0103fe2e69ff94e
pr_head_before_fix: 8bcb9cd574549871b5f6de71edd4a62e4a2a0cd7
live_attempt: false
deployment: false
merge: false
canonical_mutation: false
phase7_mutation: false

task_worktree: /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-ghu-060-network-allowlist-20260809
task_branch: codex/ghu-060-network-allowlist-20260809
registry_claim: GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808
duplicate_work: DATA_MISSING_FALLBACK_CHECKED; no matching active GHU-060 job before claim
docs_impact: DOCS_UPDATED
docs_checked: docs/manual_independent_capture_v1.md
docs_changed: docs/manual_independent_capture_v1.md
docs_followup: none
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: no delegated implementation

Runtime Functionality Proof:
- intended output: controlled fixture proof that the live child network policy
  permits only the exact race document and reviewed static assets.
- live output location: none; live browser/source execution is forbidden by the
  task card.
- pre-run max timestamp or count: not applicable; no live capture or runtime
  data source was read.
- post-run max timestamp or count: not applicable; no live capture or runtime
  data source was read.
- rows/files inserted or updated after run start: no runtime/data files; only
  allowlisted source, test, documentation, and report artifacts changed.
- readiness/gate status: controlled-fixture validation ready; live-source proof
  intentionally not claimed.
- exact command/query used: `uv run --no-project --with-requirements
  requirements/all.in --with jsonschema python -m pytest -q --noconftest
  <manual_prediction suite>` plus the focused policy test command.
- result: PARTIAL
- remaining blocker: real browser asset necessity and live-source success remain
  unobserved by explicit task boundary; no safe odds API endpoint was admitted.
