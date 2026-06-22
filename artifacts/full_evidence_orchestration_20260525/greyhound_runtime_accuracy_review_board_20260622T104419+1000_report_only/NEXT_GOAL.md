# Next Goal

Work in:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-master-live-20260621`

Selected bounded goal:

Extend the race evidence inventory packet so scorecard metrics include
actionable gap reason counts tied to each row's `recommended_next_action`.

Implementation scope:

- Update `scripts/build_race_evidence_inventory_packet.py`.
- Update `tests/test_build_race_evidence_inventory_packet.py`.
- Update `docs/race_evidence_inventory.md` for the artifact shape change.
- Run focused tests and a fresh report-only inventory packet.

Validation commands:

```bash
python3 -m py_compile scripts/build_race_evidence_inventory_packet.py tests/test_build_race_evidence_inventory_packet.py
uv run --with pytest python -m pytest --noconftest tests/test_build_race_evidence_inventory_packet.py -q
python3 scripts/build_race_evidence_inventory_packet.py \
  --artifact-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --db /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db \
  --output-dir artifacts/full_evidence_orchestration_20260525/race_evidence_inventory_20260622T_report_only
```

Closeout checks:

- Confirm the packet remains report-only with no DB, daemon, registry, training,
  promotion, EV, betting, label, odds, official-result, snapshot, or manifest
  mutation.
- Show the new actionable gap counts from the fresh report.
- Re-check odds-capture service state and `live_odds` latest timestamp if any
  runtime-adjacent file was touched. The selected scope should not touch runtime
  service files.

Do not:

- train, promote, emit EV, place bets, or update model pointers
- mutate the DB or registries
- rewrite snapshots or manifests
- edit installed systemd units
- push the local branch without owner approval
