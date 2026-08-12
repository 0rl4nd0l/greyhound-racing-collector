# Ticket 04 terminal handoff

Stop state: `DATA_MISSING`.

The single permitted preflight failed closed with
`CURRENT_INDEX_REPORT_INVALID`. No race or matching sealed receipt was
admitted, no job was submitted, and no prediction bundle was produced. The
deployed API is additionally incompatible with Ticket 04's receipt-only input:
it accepts `auto`, not `receipt`.

Do not retry this Ticket 04 attempt, select another race, capture evidence, or
change the deployed contract without separate explicit authority.
