# Independent adversarial review

Two independent read-only reviews examined the implementation, producer,
scorer, feature builder, task card, and tests.

Resolved findings:

- stage and hash the plan as well as report/form/sidecar;
- reject post-race and loosely prefixed venue URLs and bind TheDogs date/race;
- support trusted venue-code/full-name mappings such as WPK;
- validate fetch as well as append time in the fixed window;
- enforce one runner per box and one box per runner;
- validate explicit fetch provenance;
- reject tokenized/camelCase outcome keys while allowing only the benign
  `fetch_result` wrapper;
- wait for the producer final marker before parsing a report;
- fail closed past a newer finalized truncated target plan;
- compare odds losslessly rather than at 12 significant digits;
- recheck the due window before both sealing and scoring;
- add a receipt-only mode that cannot acquire the lock.

Remaining blocker is contractual, not an unresolved code defect: the optional
live score would invoke a historical feature query not authorized by this
card. Verdict: implementation review PASS; live prediction
`BLOCKED_TASK_CONTRACT`.

