# Greyhound Odds Capture Daemon Packets V1 - 2026-06-10

Status: DONE_WITH_RISK

Worktree:
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-odds-capture-daemon-packets-v1-20260610`

Branch:
`codex/odds-capture-daemon-packets-v1-20260610`

## Objective

Add a daemon/autopilot approval-packet lane for live odds capture while preserving the default no-write daemon behavior.

## Implemented

- Added `build_live_odds_capture_approval_packet(...)` in `scripts/shadow_autopilot_v1.py`.
- The packet reads `prejump_metadata_report.json`, requires verified pre-jump runner metadata, and plans fixed capture windows at T-60, T-30, T-10, and T-2.
- The packet records required provenance fields for source-bound Sportsbet WIN odds capture:
  - canonical race identity
  - Sportsbet source URL
  - Sportsbet source race identity
  - scrape timestamp
  - market type
  - dog-level WIN odds
  - Sportsbet box source
  - runner name/box match status
- The default packet is an approval artifact only:
  - `can_capture_live_odds_now=false`
  - no DB write
  - no betting action
  - no EV action
  - no model training
  - no label write
  - no production pointer mutation
- Wired the autopilot output to write `live_odds_capture_approval_packet.json`.
- Wired the daemon output to read and re-emit `live_odds_capture_approval_packet.json`.
- Added daemon dashboard, daily status, summary, verification, required output, and runtime-state fields for the live odds capture packet.
- Added report-only old-odds audit fields to daemon odds coverage:
  - stale rows
  - missing source URL rows
  - race-id mismatch rows
  - dog name/box conflict rows
  - ambiguous strict identity rows
- Added regression tests for:
  - fixed-window packet planning for verified pre-jump races
  - fail-closed packet status when no verified races exist
  - daemon summary of the autopilot packet
  - final summary packet display
  - read-only old odds audit fields
  - approved live odds capture passing `append_only=True`

## Protected Actions Avoided

- No protected DB write was performed.
- No live odds scrape was performed.
- No labels were written.
- No snapshots or snapshot manifests were rewritten.
- No model registry, production pointer, or champion artifact was mutated.
- No TGR change was made.
- No odds were used for model scoring.
- No betting advice, stake, EV action, or betting action was emitted.

## Validation

Syntax:

```bash
python3 -m py_compile scripts/shadow_autopilot_v1.py scripts/shadow_autopilot_daemon.py scripts/capture_prediction_snapshot.py
```

Exit status: 0

Focused regression set:

```bash
uv run --with pytest --with flask --with flask-compress --with flask-cors --with matplotlib --with pandas --with seaborn --with scikit-learn --with joblib --with numpy --with pyyaml --with tqdm --with requests --with beautifulsoup4 --with lxml python -m pytest tests/test_shadow_autopilot_v1.py::test_live_odds_capture_approval_packet_plans_fixed_windows_for_verified_races tests/test_shadow_autopilot_v1.py::test_live_odds_capture_approval_packet_fails_closed_without_verified_races tests/test_shadow_autopilot_daemon.py::test_live_odds_capture_packet_from_autopilot_surfaces_approval_artifact tests/test_shadow_autopilot_daemon.py::test_final_summary_includes_live_odds_capture_packet tests/test_shadow_autopilot_daemon.py::test_read_only_odds_coverage_report_does_not_mutate_db tests/test_capture_target_metadata.py::test_live_odds_capture_requires_explicit_approval tests/test_capture_target_metadata.py::test_approved_live_odds_capture_uses_append_only_loader -q
```

Exit status: 0

Result: `7 passed in 0.45s`

Touched-module regression:

```bash
uv run --with pytest --with flask --with flask-compress --with flask-cors --with matplotlib --with pandas --with seaborn --with scikit-learn --with joblib --with numpy --with pyyaml --with tqdm --with requests --with beautifulsoup4 --with lxml python -m pytest tests/test_shadow_autopilot_v1.py tests/test_shadow_autopilot_daemon.py tests/test_capture_target_metadata.py -q
```

Exit status: 0

Result: `78 passed, 6 skipped in 0.67s`

## Residual Risk

- This change prepares and surfaces an approval packet; it does not run a live daemon cycle against production runtime in this worktree.
- The live daemon currently running from the dirty checkout will not pick up this code until it is intentionally moved/merged/deployed.
- Actual odds capture still requires a separate explicit approved run with `--capture-live-odds --approve-live-odds-capture` or `APPROVE_LIVE_ODDS_CAPTURE=true`.
- Old odds rows are now audited in reports, but not repaired or deleted.

## Next Best Steps

1. Review this worktree diff.
2. Run a daemon `run-once` dry run from this worktree against a non-protected or copied DB if runtime-level evidence is needed.
3. If accepted, merge/deploy this lane to the live daemon checkout.
4. Wait for a verified pre-jump race packet showing `AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS`.
5. Approve one bounded append-only odds capture for those verified races only.
