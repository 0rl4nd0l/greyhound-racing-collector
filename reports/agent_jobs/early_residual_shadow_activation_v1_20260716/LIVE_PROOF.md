# Live proof

State: WAITING_ON_TIMER

## Runtime Functionality Proof

- Intended output: one normalized outcome-free frozen residual shadow record appended immediately after each successful strict pre-jump capture and before lock release.
- Live output location: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/market_form_residual_shadow_predictions_v1.jsonl`.
- Pre-run max timestamp or count: 0 records; the JSONL did not exist before installation.
- Post-run max timestamp or count: 0 records; the JSONL remains absent because no eligible capture window occurred after the repaired unit was installed.
- Rows/files inserted or updated after run start: 0 residual shadow records; one generated user service file was installed and systemd metadata was reloaded.
- Readiness/gate status: code, tests, committed worktree, generated unit, installed unit, model pin, and read-only exact-plan replay are ready; the first real append remains pending.
- Exact command/query used: `systemctl --user cat shadow-autopilot-odds-capture.service`; `sha256sum` over generated and installed units; `test -e "$OUT" && wc -l "$OUT" || echo 0`; and a read-only call to `early_residual_shadow_prediction_plan` against the exact 23:08 PR #47 handoff.
- Result: `PARTIAL`.
- Remaining blocker: no eligible post-repair capture exists tonight; the runtime state names 2026-07-17T09:55:00+10:00 as the next meaningful capture target.

## Observed sequence

- 23:08 AEST: strict capture appended 12 odds rows for
  `Race 12 - MAND - 2026-07-16`; the early plan failed closed with the sole
  blocker `feature_model_missing` and read no outcomes.
- 23:21:53 AEST: repaired odds-only service installed at SHA-256
  `8d798ce374495486c839c42a6687685c2acab8027d4fd73190bcf5b7bd50380e`;
  timer bytes remained unchanged; `daemon-reload` only.
- 23:22 AEST: the normal timer invoked the repaired unit successfully and it
  exited `ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW` because the next feed race is
  on 2026-07-17.
- Read-only exact-plan replay: `READY`, zero blockers, one sealed race,
  configured feature-model path present, `outcomes_read=false`, and
  `activation=false`. Scoring was not executed after jump.
