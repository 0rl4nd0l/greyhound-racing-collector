# Independent review and integration decision

Draft PR: https://github.com/0rl4nd0l/greyhound-racing-collector/pull/183

The authorized original commit is
`d889399416648bd7d7d3877489c870255ee27076`; review starts at its parent
`5013ff039fda418f47a59373041d7bc7c4124f07` and includes the integration
corrections in this PR. Two independent agent reviews covered standards and
the user's implementation/authority requirements. Follow-up review was limited
to concrete findings and their corrections. No real histories or target labels
were accessed by the reviewers or the implementation validation.

## Standards

Final verdict: PASS for the reviewed scope, zero outstanding findings.

The most severe integration finding was a P1 default-history schema regression:
an added return field would have broken the existing exact-field verifier even
when retention was disabled. The extra field was removed; retention-specific
scope remains in its own manifest. Existing history-seal fixtures passed.

A P2 finding allowed replay of a completed child packet without the parent's
cutoff acceptance. Scheduled replay now requires RETAINED parent status, matching
manifest/configuration hashes and acceptance before cutoff. Fixtures reject
missing, rejected and mismatched parent terminals. Earlier findings concerning
completion timing and loss of safe failure codes were also corrected.

The reviewer confirmed source-only that both focused findings are resolved.
No additional documented-standards violation was found. Full-database temporary
copying remains an explicit access/cost limitation, not an unstated exemption.

## Spec

Final verdict: no remaining material offline implementation finding.

The original utility lacked a collector call site and isolated feature replay.
The optional configuration now follows both existing daemon paths into the
successful WIN append/exact-receipt callback; no new scheduler or prediction
pipeline exists. Retention snapshots the history used by its archived generator,
retains source bindings and availability times, and preserves all 16 model
features and nulls in a private file.

The follow-up review confirmed parent-terminal binding, execution of the retained
worker rather than checkout code, and the full-versus-runner-projected feature
comparison. Actual source completeness, source lineage, latency/storage cost and
authorized real replay remain explicitly unproven live-acceptance requirements.
The separate evaluation assessment does not amend or activate a protocol.

## Targeted validation

The existing isolated Python 3.11 environment ran synthetic fixtures only:

- Retention, scheduled receipt integration, callback success/failure and existing
  history seals: 27 passed, 113 deselected.
- Existing daemon/service/autopilot option plumbing and default compatibility:
  11 passed, 262 deselected.
- The retention subset was repeated after adding an explicit feature-file fsync:
  9 passed (already included in the 27 unique tests above).
- `git diff --check` passed.

The 27-test command selects `prospective_input_retention`,
`scheduled_input_retention`, `appends_after_exact_sportsbet_validation`, and
`history_seal` across the corresponding retention/capture/predictor test modules.
The 11-test command selects `input_retention`,
`run_once_exception_writes_terminal_daemon_report`,
`odds_capture_only_autopilot_command`, `autonomous_live_odds_capture_command`,
`full_service_generator_emits_forward_opt_in`, and
`existing_forward_corpus_service_command` across both autopilot test modules.
Both use `pytest -q -o addopts='' --disable-warnings --maxfail=1`.

These are fixture assertions, not measured live success rates. No production
service or collector was restarted, and no prediction or research activation
occurred. Real filesystem inspection was restricted to database file sizes and
journal-sibling existence; no database contents were opened.

## Decision

The integrated, default-off PR is ready for owner review and a separate merge
decision. Do not enable recurring retention on fixture evidence alone. The next
live acceptance must use the existing collector and an exact separately approved
future race, source inventory, whole-database machine-only snapshot/history scope,
cutoff and resource limits described in `prospective_input_retention_preparation.md`.
One successful race would prove the input delivery path only, not multi-race
readiness or statistical feasibility. Merge, deployment and that bounded access
remain distinct authorizations.

Standards: zero outstanding findings; Spec: zero outstanding offline findings,
with real-data acceptance and statistical adoption explicitly pending.
