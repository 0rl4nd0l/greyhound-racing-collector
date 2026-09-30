# Approved development pilot result worker

`python scripts/development_pilot_results.py inspect --config PATH --config-sha256 SHA`
reads only pinned authority and queue identity/status metadata. An empty or
not-yet-due queue returns `NO_WORK`, does not initialize a queue, and never opens
pre-result inputs, target outcomes or a provider connection. `cycle` processes
at most one due race under the same controls. Root alone installs its separate
minute timer; this module does not activate itself or change study services.

The shared `development_pilot_runtime_v1` configuration requires the existing
capture fields (`status`, `authority_reference`, `allocation`, `source_root`,
`source_commit`, `python`, `state_root`, `campaign_root`, `source_state`,
`lock_path`, `study_schedule`, `pilot_campaign_authority`) and:

- `result_closure_at`: `2026-10-25T12:00:00+11:00`;
- `max_result_operations`: 72;
- `max_result_transport_requests`: 720;
- `max_result_checks_per_race`: 3.

The real runtime verifies its Git source, interpreter, private state and
approved campaign/allocation binding. Its separate campaign profile maps the
720 transport allowance to `max_result_logical_requests`, never study counters.
The adapter currently needs one exact-URL request per check, so 72 is its
attainable maximum; 720 remains the approved ceiling, not a retry allowance.

Capture publishes private immutable `ready/SHA256(race_id).json` only after a
verified pre-result seal. Schema `development_pilot_capture_ready_v1` carries
`race_id`, `race_key`, `jump_at`, `access_path`, `access_sha256`, `example_dir`,
`pre_result_sha256`, `completion_sha256`, `job_id`, `prediction_entry`,
`published_at`, and `allocation_sha256`. The packet must be beneath
`state_root/sessions`; nomination alone cannot bypass allocation membership,
pre-jump completion or retained-input replay. No study job/result store is
inventoried or joined.

Checks are eligible at jump +30 minutes, noon on the next Melbourne calendar
day (including daylight-saving transitions), and in the final October 25
11:30–12:00 Melbourne closure window. The last window allows up to 24 one-per-
minute checks. Each requires at least 45 seconds remaining before hard noon
and has a 25-second absolute transport deadline. No transport occurs at/after
noon. Late restarts use the latest eligible milestone only; missed milestones
are retained, never replayed as a burst. Actual timer throughput and provider
completion remain unproven.

The worker takes its own queue flock, the shared campaign owner flock and the
existing collector lock without stealing. It yields during study preparation/
forecast windows and to fresh study result-health metadata showing due/running
retention. Missing/stale study health fails closed. Shared provider STOP and
campaign source holds are checked again inside ownership. HTTP 401/403/429,
retry guidance, or recognized challenge content persists STOP before any next
request. Header denials are held before reading a possibly failing body.

Each operation is consumed before transport. A durable request reservation
precedes the separate authoritative campaign charge; charge acknowledgement
and possible transport start have separate immutable markers. Inspection counts
reserved requests as consumed and exposes unconfirmed reservations, including
a crash after charge but before acknowledgement. It never replenishes them.
A crash preserves started checks. HTML is bounded to 1 MiB,
retained privately with its hash and actual response-availability timestamp.
There are no redirects, transport retries, browser fallback, alternate sources,
index discovery or general label DB writes. Only existing TheDogs parsing and
strict exact-field validation are used: observed names cannot be replaced by
expected names, dead heats/duplicates/partial fields are ambiguous, and native
runner IDs are left absent when not observed.

Verified rows flow through the existing restricted per-race `join_result` and
standalone replay seam. Missing and ambiguous checks persist; terminal closure
produces a durable non-trainable example. The existing result schema also
retains an explicitly authorized exact-race `VOID` disposition. This parser
cannot establish race-level void from an unfamiliar page, so it never infers
void from missing positions. Unknown terminal pages remain missing/ambiguous.
An interrupted published result recovers via the same admitted replay/join,
with no new request. Invalid nominations and failed joins receive explicit
non-trainable terminal queue records; inspection reports their rejected count,
and a corrupt first race cannot indefinitely starve later due races. Private artifact files are mode 0600 and roots 0700.

All implementation demonstrations use labeled synthetic capture/HTTP fixtures.
No new official result or provider acquisition is required for validation.
