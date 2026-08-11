# Operator UI R3 Ticket 03 activation report

Terminal status: `PASS`

Ticket 03 reverified and installed the accepted Ticket 02 unit, reloaded the
user manager, and restarted only `greyhound-operator-ui-r3.service`. The
installed service runs the accepted source and pinned Python and listens only
on `127.0.0.1:5055`.

The initial smoke stopped fail-closed after its Python cookie jar withheld the
service's `Secure` session cookie over plain loopback HTTP, causing the login
POST to return `400 NON_OPERATIONAL/CSRF_REJECTED`. A fresh continuation
reverified every accepted identity and diagnosed that client-side cookie
handling from the deployed security contract and tests before making one
corrected request. The corrected smoke returned the cookie issued by the same
loopback service explicitly, authenticated successfully, and received `200`
from the connected sentinel, R3 capability, overview, and system read-only
endpoints. No prediction or capture endpoint was called.

The final state at `2026-08-11T16:05:41+10:00` is:

- R3: `active/running`, PID `2406787`, installed unit SHA-256
  `14635fc14f244d93b3de1c956aef7c0c973d9fa4ea66fe1ee95edec48d34d655`.
- Full collector: `failed/failed`; installed unit unchanged at SHA-256
  `4f606f21c1215006d43acf3df8db9e1390f70226b03de118030efb63940d9630`.
- Odds collector: `activating/start`, PID `1378085`; installed unit unchanged
  at SHA-256
  `f85170b90735dfe98a7f6e13355be466c1811b7b9d7f393e25dda2a8e97682d7`.

See `evidence.json` for commands, exits, identities, response hashes, and
preserved boundaries.
