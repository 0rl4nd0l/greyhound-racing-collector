# Read canonical history without mutating it

The report-only R3 prediction may read the canonical runtime database and may write a cutoff-filtered `sealed_history.db` inside its new immutable operations bundle. It must not mutate the canonical database. This matches the predictor's SQLite read-only/query-only connection and the generated service's read-only systemd mount for the canonical database.

A genuine R3 job may also write its required lifecycle transitions to `jobs.sqlite3`, append its audit trail to `audit.sqlite3`, and publish its immutable bundle in the separate operations root. Those operational-evidence writes do not grant write access to canonical racing history.
