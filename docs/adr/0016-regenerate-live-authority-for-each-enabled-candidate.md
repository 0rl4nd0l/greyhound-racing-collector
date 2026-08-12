# Regenerate live authority for each enabled candidate

Each enabled R3 candidate receives a fresh `operator_ui_live_authority_v1` observation built deterministically from contemporaneous read-only local evidence and actual service state. It records the real observation time and exact reports, refresh file, inventory/raw packets, installed units, and full/odds systemd states. Authority from a prior or rejected package is not reused, and producing the observation performs no external fetch.
