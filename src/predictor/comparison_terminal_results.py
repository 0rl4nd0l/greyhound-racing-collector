"""Private, identity-verified terminal results that cannot supply a full order.

The caller owns authority, immutable job/bundle verification and membership in
its charged request ledger. This decoder acquires nothing and writes nothing.
Its record must never be inserted as ordinary full-order result evidence.
"""
from datetime import datetime
import hashlib
from pathlib import Path

from scripts import ingest_results_for_date as ingest
from scripts.autonomous_official_result_capture import comparison_runner_identity_error
from scripts.reconcile_comparison_result_identity import checked
from src.operator_ui.job_store import canonical, utc_text
from src.predictor.comparison_runner_identity import frozen_result_participants
from src.predictor.comparison_result_runtime import MAX_BODY
from utils.csv_metadata import canonical_thedogs_race_identity

NONFINISH_STATUSES = frozenset({'FELL', 'DNF', 'DISQ'})


def _require(condition):
    if not condition:
        raise ValueError('COMPARISON_TERMINAL_RESULT_NOT_VERIFIED')


def _time(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    utc_text(value)
    return value


def known_nonfinish_evidence(job, bundle, body, url, captured, now, *, deadline,
                             prediction_bundles, source_evidence):
    """Return known non-finish evidence only with complete frozen-field coverage.

    source_evidence binds the exact native body, request and response files.
    Their association with the job's latest charged request is the caller's
    responsibility, as is the dated permission to read protected results.
    """
    import json
    deadline.check_deadline()
    _require(set(source_evidence) == {'body', 'request', 'response'})
    retained = deadline.call(checked, source_evidence['body'], maximum=MAX_BODY)
    _require(isinstance(body, bytes) and body == retained)
    request = deadline.call(json.loads, deadline.call(checked, source_evidence['request']))
    response = deadline.call(json.loads, deadline.call(checked, source_evidence['response']))
    race = bundle.result['race']
    captured, now = _time(captured), _time(now)
    _require(url in {race['url'], race['url']+'?trial=false'}
             and canonical_thedogs_race_identity(url) is not None)
    _require(request['url'] == response['final_url'] == url
             and _time(request['at']) == _time(response['observed_at']) == captured)
    _require(type(response['status']) is int and response['status'] == 200
             and response['host'] == 'www.thedogs.com.au'
             and set(response['retry_headers']) <= {'date'}
             and response['content_type'].lower().startswith('text/html')
             and response['sha256'] == hashlib.sha256(body).hexdigest()
             and type(response['bytes']) is int and response['bytes'] == len(body))
    _require(max(_time(job.input.jump_timestamp), _time(bundle.result['generated_at'])) < captured <= now)
    participants = deadline.call(frozen_result_participants, job, bundle, prediction_bundles)
    expected = {r['box_number']: r for r in participants}
    _require(bool(expected) and len(expected) == len(participants))
    candidate = ingest.RaceCandidate(job.input.race_id, race['venue'], race['race_number'],
        race['race_date'], None, job.input.jump_timestamp, None, Path('/unused-sealed-r3-inputs'),
        participants, 'JUMPED_AWAITING_RESULT', participant_source='verified_r3_prediction',
        csv_participants=participants, canonical_thedogs_url=race['url'])
    text = deadline.call(body.decode, 'utf-8', errors='strict')
    title = deadline.call(ingest.title_from_html, text)
    rendered = deadline.call(ingest.rendered_text_from_html, text)
    _require(not deadline.call(ingest.response_is_forbidden, 200, title, rendered))
    selected = deadline.call(ingest.TheDogsResultFetcher(None)._result_from_html, candidate, url, text)
    _require(selected is not None and selected.source == 'thedogs_official' and selected.status == 'resulted')
    _require(deadline.call(comparison_runner_identity_error, candidate, selected) is None)
    positions = selected.positions_by_box
    terminals = {box: value for box, value in (selected.terminal_status_by_box or {}).items()
                 if box in expected}
    _require(bool(terminals) and set(terminals.values()) <= NONFINISH_STATUSES)
    _require(not set(positions).intersection(terminals)
             and set(positions) | set(terminals) == set(expected))
    _require(all(type(p) is int and p > 0 for p in positions.values())
             and deadline.call(ingest.finish_positions_follow_competition_ranking, positions.values()))
    _require(deadline.call(ingest.result_validation_error, candidate, selected)
             in {None, 'duplicate_first_place_results'})
    record = {
        'schema_version': 'comparison_known_nonfinish_result_v1',
        'state': 'RESULT_KNOWN_NON_FINISH', 'result_known': True,
        'identity_verified': True, 'full_order_eligible': False,
        'full_order_exclusion': 'OFFICIAL_NONFINISH_HAS_NO_NUMBERED_POSITION',
        'job_id': job.job_id, 'race_id': job.input.race_id,
        'source': selected.source, 'source_url': url, 'captured_at': captured.isoformat(),
        'source_evidence': source_evidence,
        'runner_results': [dict(expected[box], finish_position=positions.get(box),
                                terminal_status=terminals.get(box)) for box in sorted(expected)],
        'reserve_box_remappings': selected.reserve_box_remappings or [],
        'ignored_terminal_status_rows': selected.ignored_terminal_status_rows or [],
        'outcomes_released': False,
    }
    record['evidence_sha256'] = hashlib.sha256(canonical(record)).hexdigest()
    deadline.check_deadline()
    return record
