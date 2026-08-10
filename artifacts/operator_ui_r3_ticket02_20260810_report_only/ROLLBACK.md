# Operator UI R3 rollback

Regenerate this package without `--enable`, stop/disable only
`greyhound-operator-ui-r3.service`, and verify the existing UI remains available.
Do not delete `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-ticket02-operations-20260810-IX5u7P`: it contains Operator UI audit/job evidence. Do not
delete `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-manual-collector-live-deploy-20260730-9be52ecd` or `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`: prediction and collector evidence is retained.
Rollback changes the feature gate only; it does not edit installed/generated files
by hand and does not alter the canonical database `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db`.
