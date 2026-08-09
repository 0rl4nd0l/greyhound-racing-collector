Independent review scope: corrective network-policy diff from PR #128 head
`8bcb9cd574549871b5f6de71edd4a62e4a2a0cd7`.

Review conclusion: no critical or warning findings. The pure policy has no
same-origin blanket allow, admits no data-bearing request family, and keeps
unknown/unclassified requests fail closed. Controlled fixtures prove the
adapter and unchanged parent/sealer path; real source success remains outside
this ticket.
