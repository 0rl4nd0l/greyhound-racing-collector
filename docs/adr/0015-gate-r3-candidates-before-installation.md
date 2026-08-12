# Gate R3 candidates before installation

Before installation, a replacement R3 candidate must prove its exact clean source commit/tree and fixed artifact hashes; pass the deployment-generator and R3 end-to-end safety tests; pass frozen-model fixed-fixture and portability replay under Python 3.11.15 and NumPy 1.26.4; generate into a new empty directory whose four outputs and hashes are reviewed; and pass `systemd-analyze verify`. Any failed gate rejects the candidate before service installation, daemon reload, or restart.
