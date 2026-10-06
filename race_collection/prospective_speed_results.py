"""Selected native forecasts reuse the original finite development result ledger.

No discovery or collection is performed. The original ready/results directory,
milestones, durable request reservations, campaign charges and denial handling
remain authoritative. This worker waits for source ownership; it never pauses
the producer or changes scientific readiness.
"""
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from race_collection import development_pilot_results as legacy
from race_collection import prospective_speed_closure as closure
from race_collection import prospective_speed_evaluation_io as evaluation
from race_collection import prospective_speed_plan as planning
from race_collection import prospective_speed_runtime as runtime
from race_collection.development_source_authority import load_development_authority
from race_collection.retained_card_timing_coverage import Reader, instant


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_config(reference):
    config = Reader().json(reference)
    plan = Reader().json(config['plan'])
    planning._validate_plan(plan)
    activation = Reader().json(config['activation'])
    cfg = Reader().json(config['legacy_runtime'])
    profile = load_development_authority(cfg['pilot_campaign_authority'])
    allocation = Reader().json(cfg['allocation'])
    _require(config.get('schema_version') == 'prospective_speed_result_bridge_v1'
        and config.get('status') == 'AUTHORIZED_SELECTED_NATIVE_DEVELOPMENT_RESULTS'
        and config['legacy_runtime'] == plan['authority']['result_runtime']
        and cfg.get('status') == 'AUTHORIZED' and allocation.get('status') == 'AUTHORIZED'
        and allocation['allocation_id'] == plan['allocation_id']
        and cfg['allocation'] == plan['authority']['allocation'] == profile['allocation']
        and cfg['state_root'] == profile['state_root']
        and cfg['authority_reference'] == profile['authority_reference'] == allocation['authority_reference']
        and cfg['max_result_operations'] == profile['max_result_operations'] == 72
        and cfg['max_result_transport_requests'] == profile['max_result_logical_requests'] == 720
        and cfg['max_result_checks_per_race'] == 3
        and cfg['result_closure_at'] == profile['result_closure_at'] == plan['result_requests_stop_at'],
        'ORIGINAL_DEVELOPMENT_RESULT_AUTHORITY_CHANGED')
    _require(activation.get('status') == 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
        and activation.get('plan_sha256') == config['plan']['sha256']
        and activation.get('development_precedence_verified') is True
        and activation.get('competing_pilot_disabled') is True
        and activation.get('legacy_result_worker_disabled') is True
        and activation.get('development_readiness_substitution_verified') is True
        and activation.get('result_retention_routing_verified') is True
        and activation.get('additional_source_requests') == 0
        and activation.get('additional_result_requests') == 0, 'RESULT_ROUTING_NOT_ACTIVATED')
    for reference in activation['verified_control_files']:
        Reader().read(reference)
    for path in (config['coordinator_state_root'], cfg['state_root'], cfg['campaign_root'], cfg['lock_path']):
        _require(Path(path).is_absolute() and Path(path).resolve() == Path(path), 'RESULT_PATH_UNSAFE')
    _require(Path(config['coordinator_state_root']) != Path(cfg['state_root']), 'RESULT_LEDGER_COLLISION')
    return config, plan, cfg


def _admitted(config, plan, ready):
    """Replay selected membership and the exact speed/native seal, never a target."""
    _require(ready.get('prospective_speed_role') == 'SELECTED_NATIVE_DEVELOPMENT_RESULT'
        and ready['plan'] == config['plan'] and ready['allocation_sha256'] == plan['authority']['allocation']['sha256'],
        'NATIVE_RESULT_NOMINATION_CHANGED')
    population = Reader().json(ready['population'])
    planning._validate_population(plan, population)
    day = population['local_date']
    account = Reader().json(runtime.reference(Path(config['coordinator_state_root'])/day/'date-accounting.json'))
    _require(day in planning.DATES and ready['race_id'] in population['selected_race_ids']
        and account['status'] == 'POPULATION_FROZEN' and account['population'] == ready['population']
        and account['population_sha256'] == planning._digest(population)
        and ready['race_key'] == legacy.race_key(ready['race_id']), 'NATIVE_RESULT_NOT_SELECTED')
    reader = evaluation._Checked()
    completion = reader.json(ready['forecast_completion'])
    seal = reader.json(completion['seal'])
    terminal = reader.json(seal['terminal'])
    payload = reader.output_json(terminal['payload'])
    shell = {'plan': config['plan'], 'forecast_root': str(Path(config['coordinator_state_root'])/'attempts'),
        'population_by_date': {day: ready['population']}}
    entry = {'race_id': ready['race_id'], 'race_date': day,
        'label_status': 'UNREAD_RESULT_NOMINATION',
        'forecast_completion': ready['forecast_completion'], 'runner_ids': [r['runner_id'] for r in payload['predictions']]}
    evaluation._forecast(reader, entry, shell, plan, population)
    context = closure._native_context(reader, payload)
    _require(ready['pre_result_sha256'] == terminal['payload']['sha256']
        and ready['job_id'] == context['job'].job_id
        and instant(ready['jump_at']) == instant(payload['jump_at']), 'NATIVE_READY_BINDING_CHANGED')
    request = context['job'].input
    packet = {'synthetic': False, 'race': context['bundle'].result['race'],
        'times': {'scheduled_jump_at': request.jump_timestamp,
                  'prediction_at': context['bundle'].result['generated_at']},
        'runners': [{'box_number': r['box'], 'display_name': r['name'],
            'source_native_runner_id': r['source_native_runner_id']} for r in request.ordered_runners]}
    return packet, context


def nominate(config, plan, cfg):
    """Only exact selected disposition keys are inspected; no successful-directory census."""
    root = Path(cfg['state_root'])
    ready_root = legacy.private_output(root/'ready')
    count = 0
    for day in planning.DATES:
        day_root = Path(config['coordinator_state_root'])/day
        account_path = day_root/'date-accounting.json'
        if not account_path.exists():
            continue
        account = Reader().json(runtime.reference(account_path))
        if account['status'] != 'POPULATION_FROZEN':
            continue
        population_ref = account['population']
        population = Reader().json(population_ref)
        planning._validate_population(plan, population)
        _require(population['local_date'] == day and planning._digest(population) == account['population_sha256'],
                 'RESULT_POPULATION_CHANGED')
        for race_id in population['selected_race_ids']:
            key = hashlib.sha256(race_id.encode()).hexdigest()
            disposition_path = day_root/'jobs'/(key+'.disposition.json')
            if not disposition_path.exists():
                continue
            disposition = Reader().json(runtime.reference(disposition_path))
            if disposition['status'] != 'SEALED_PREJUMP':
                continue
            _require(disposition['race_id'] == race_id, 'RESULT_DISPOSITION_CHANGED')
            completion_ref = disposition['completion']
            path = ready_root/(key+'.json')
            if path.exists():
                # Repeated outcome-free scheduler ticks authenticate the small
                # immutable nomination only. Full payload/native-chain replay
                # remains mandatory immediately before each result operation.
                ready = Reader().json(runtime.reference(path))
                expected_jump = next(row['jump_at'] for row in population['dispositions']
                                     if row['race_id'] == race_id)
                _require(ready.get('schema_version') == 'development_pilot_capture_ready_v1'
                    and ready.get('prospective_speed_role') == 'SELECTED_NATIVE_DEVELOPMENT_RESULT'
                    and ready.get('race_id') == race_id and ready.get('race_key') == legacy.race_key(race_id)
                    and ready.get('allocation_sha256') == cfg['allocation']['sha256']
                    and ready.get('plan') == config['plan'] and ready.get('population') == population_ref
                    and ready.get('forecast_completion') == completion_ref
                    and instant(ready['jump_at']) == instant(expected_jump), 'EXISTING_RESULT_NOMINATION_CHANGED')
                continue
            completion = Reader().json(completion_ref)
            seal = Reader().json(completion['seal'])
            terminal = Reader().json(seal['terminal'])
            payload = runtime.read_output(terminal['payload'])
            context = closure._native_context(evaluation._Checked(), payload)
            ready = {'schema_version': 'development_pilot_capture_ready_v1',
                'prospective_speed_role': 'SELECTED_NATIVE_DEVELOPMENT_RESULT',
                'race_id': race_id, 'race_key': legacy.race_key(race_id),
                'jump_at': payload['jump_at'], 'job_id': context['job'].job_id,
                'allocation_sha256': cfg['allocation']['sha256'],
                'pre_result_sha256': terminal['payload']['sha256'],
                'plan': config['plan'], 'population': population_ref,
                'forecast_completion': completion_ref}
            _admitted(config, plan, ready)
            runtime.put_new(path, ready)
            count += 1
    return count


@contextmanager
def ownership(cfg, output, now):
    """The approved development health replacement leaves native source locks intact."""
    from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
    from utils.sportsbet_access import SportsbetAccess
    legacy._shared_stop(cfg)
    _require(SportsbetAccess(cfg['source_state']).read().get('active') is None, 'RESULT_SOURCE_BUSY')
    _require(not Path(cfg['lock_path']).exists(), 'RESULT_COLLECTOR_BUSY')
    lock = None
    with open(Path(cfg['campaign_root'])/'owner.lock', 'a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            lock = acquire_collector_lock_no_steal(Path(cfg['lock_path']),
                run_id='prospective-development-result-'+output.name, output_dir=output,
                phase='development_pilot_result', acquisition_policy='approved_development_pilot_v1')
            legacy._shared_stop(cfg)
            _require(SportsbetAccess(cfg['source_state']).read().get('active') is None, 'RESULT_SOURCE_BUSY')
            yield
        finally:
            if lock is not None:
                release_owned_collector_lock(lock)


class _EnvelopeSession:
    """Retain actual transport headers needed by independent native parsing."""
    def __init__(self, session, attempt):
        self.session, self.attempt = session, attempt

    def get(self, url, **kwargs):
        response = self.session.get(url, **kwargs)
        headers = {k.lower(): v for k, v in response.headers.items()}
        if response.status_code in {401, 403, 429} or set(headers).intersection({
                'retry-after', 'ratelimit-reset', 'x-ratelimit-reset'}):
            # The original fetch must persist its global STOP before optional
            # envelope IO can fail. Denied responses are never parsed here.
            return response
        names = {'content-type', 'content-encoding', 'retry-after', 'ratelimit-reset', 'x-ratelimit-reset'}
        try:
            runtime.put_new(self.attempt/'http-envelope.json', {'url': response.url,
                'status_code': response.status_code, 'headers': {k: v for k, v in headers.items() if k in names},
                'observed_at': runtime.utc_now().isoformat()})
        except BaseException:
            response.close()
            raise
        return response


def validate_retained_response(context, references, now):
    """Native identity/terminal parsing from one legacy transport's exact bytes.

    Legacy transport records distinct request and response timestamps. Preserve
    and validate both instead of fabricating a simultaneous capture envelope.
    """
    from scripts import ingest_results_for_date as ingest
    from scripts.autonomous_official_result_capture import comparison_runner_identity_error
    from src.predictor.comparison_runner_identity import frozen_result_participants
    from src.predictor.comparison_terminal_results import NONFINISH_STATUSES
    from src.predictor.comparison_results import ComparisonResultSource
    _require(set(references) == {'body', 'request', 'response', 'http_envelope'}, 'RETAINED_RESULT_ROLES_CHANGED')
    reader = Reader()
    body = reader.read(references['body'])
    request, response, envelope = [reader.json(references[k]) for k in ('request', 'response', 'http_envelope')]
    race, job, bundle = context['bundle'].result['race'], context['job'], context['bundle']
    url = race['url']+'?trial=false'
    captured = instant(response['observed_at'])
    headers = envelope['headers']
    _require(request['race_id'] == race['race_id']
        and request['url'] == response['url'] == envelope['url'] == url
        and response['status_code'] == envelope['status_code'] == 200
        and instant(job.input.jump_timestamp) < instant(request['at']) <= instant(envelope['observed_at']) <= captured <= now
        and len(body) <= legacy.MAX_BODY and response['bytes'] == len(body)
        and response['sha256'] == hashlib.sha256(body).hexdigest()
        and headers.get('content-encoding', 'identity') in ('', 'identity')
        and 'text/html' in headers.get('content-type', '').lower()
        and not set(headers).intersection({'retry-after', 'ratelimit-reset', 'x-ratelimit-reset'}),
        'RETAINED_RESULT_ENVELOPE_CHANGED')
    text = body.decode('utf-8', errors='strict')
    _require(not ingest.response_is_forbidden(200, ingest.title_from_html(text), ingest.rendered_text_from_html(text)),
             'RETAINED_RESULT_SOURCE_DENIED')
    participants = frozen_result_participants(job, bundle, context['prediction_bundles'])
    expected = {row['box_number']: row for row in participants}
    _require(bool(expected) and len(expected) == len(participants), 'RETAINED_FIELD_AMBIGUOUS')
    candidate = ingest.RaceCandidate(job.input.race_id, race['venue'], race['race_number'], race['race_date'],
        None, job.input.jump_timestamp, None, Path('/unused-retained-native-source'), participants,
        'JUMPED_AWAITING_RESULT', participant_source='verified_r3_prediction', csv_participants=participants,
        canonical_thedogs_url=race['url'])
    parsed = ingest.TheDogsResultFetcher(None)._result_from_html(candidate, url, text)
    _require(parsed is not None and parsed.source == 'thedogs_official' and parsed.status == 'resulted'
        and comparison_runner_identity_error(candidate, parsed) is None
        and ingest.result_validation_error(candidate, parsed) in {None, 'duplicate_first_place_results'},
        'RETAINED_OFFICIAL_IDENTITY_CHANGED')
    positions = parsed.positions_by_box
    terminals = {box: value for box, value in (parsed.terminal_status_by_box or {}).items() if box in expected}
    _require(set(positions).isdisjoint(terminals) and set(positions) | set(terminals) == set(expected)
        and set(terminals.values()) <= NONFINISH_STATUSES
        and ingest.finish_positions_follow_competition_ranking(positions.values())
        and any(value == 1 for value in positions.values()), 'RETAINED_OFFICIAL_TERMINAL_INCOMPLETE')
    if terminals:
        return {'schema_version': 'prospective_development_known_nonfinish_v1',
            'race_id': job.input.race_id, 'job_id': job.job_id, 'captured_at': response['observed_at'],
            'identity_verified': True, 'full_order_eligible': False,
            'runner_results': [dict(expected[box], finish_position=positions.get(box),
                terminal_status=terminals.get(box)) for box in sorted(expected)],
            'reserve_box_remappings': parsed.reserve_box_remappings or []}, 'KNOWN_NONFINISH'
    common = {k: race[k] for k in ('race_id', 'race_date', 'race_number', 'venue')} | {
        'source': 'thedogs_official', 'source_url': url, 'captured_at': response['observed_at']}
    names = parsed.dog_names_by_box or {}
    rows = [{**common, 'box_number': box, 'dog_name': names.get(box, ''),
        'source_native_runner_id': expected[box].get('source_native_runner_id'),
        'finish_position': position, 'is_winner': position == 1} for box, position in positions.items()]
    order = sorted(positions, key=lambda box: (positions[box], box))
    winner = order[0]
    race_row = {**common, 'status': 'resulted', 'start_datetime': job.input.jump_timestamp,
        'winner_box': winner, 'winner_name': names.get(winner, ''), 'position_count': len(rows),
        'participant_count': len(expected), 'box_order': order}
    ComparisonResultSource._validate(job, bundle, [race_row], rows, now)
    return {'race_rows': [race_row], 'runner_rows': rows}, 'OFFICIAL'


def _complete(directory, ready, evidence, disposition, reason, observed, source_evidence=None):
    record = {'schema_version': 'development_official_result_v1', 'synthetic': False,
        'prospective_speed_role': 'SELECTED_NATIVE_DEVELOPMENT_RESULT',
        'race_id': ready['race_id'], 'race_key': ready['race_key'], 'disposition': disposition,
        'observed_at': observed.isoformat(), 'official_source': 'thedogs_official',
        'source_evidence_sha256': legacy.digest(legacy.canonical(evidence)) if evidence else None,
        'official_evidence': evidence, 'reason': reason, 'forecast_completion': ready['forecast_completion'],
        'source_evidence': source_evidence}
    path = directory/'official-result.json'
    if path.exists():
        _require(Reader().json(runtime.reference(path)) == record, 'RESULT_COMPLETION_CHANGED')
        result_ref = runtime.reference(path)
    else:
        result_ref = runtime.put_new(path, record)
    return runtime.put_new(directory/'complete.json', {'race_id': ready['race_id'],
        'pre_result_sha256': ready['pre_result_sha256'], 'result': result_ref,
        'status': ('KNOWN_NONFINISH_RETAINED' if disposition == 'KNOWN_NONFINISH' else 'OFFICIAL_RESULT_RETAINED') if evidence else (
            'QUARANTINED_UNVERIFIED_OFFICIAL_RESULT' if disposition == 'AMBIGUOUS' else 'UNRESOLVED_AT_CLOSURE'),
        'reason': reason,
        'target_join_completed': False, 'outcomes_released': False})


def _run_due(config, plan, cfg, root, now):
    _, due, _, operations, transports = legacy._inventory(cfg, root, now)
    selected = [(at, ready, stage) for at, ready, stage in due
        if ready.get('prospective_speed_role') == 'SELECTED_NATIVE_DEVELOPMENT_RESULT'
        and ready.get('plan') == config['plan']]
    if not selected:
        return {'status': 'NO_SELECTED_RESULT_DUE', 'operations_consumed': operations,
                'transport_requests_consumed': transports}
    _, ready, stage = selected[0]
    packet, context = _admitted(config, plan, ready)
    directory = legacy.private_output(root/'results'/legacy.digest(ready['race_id'].encode()))
    existing = directory/'official-result.json'
    if existing.exists():
        value = Reader().json(runtime.reference(existing))
        _complete(directory, ready, value['official_evidence'], value['disposition'], value['reason'],
                  instant(value['observed_at']), value['source_evidence'])
        return {'status': 'RETAINED_COMPLETION_RECOVERED'}
    if stage is None or now >= instant(cfg['result_closure_at']) or operations >= 72 or transports >= 720:
        disposition, reason = 'MISSING', 'CHECKS_EXHAUSTED_OR_INTERRUPTED'
        for prior in reversed(range(3)):
            metadata = directory/f'attempt-{prior}'/'finished.json'
            if metadata.exists():
                value = Reader().json(runtime.reference(metadata))
                if value['disposition'] == 'AMBIGUOUS':
                    disposition, reason = 'AMBIGUOUS', value['reason']
                break
        _complete(directory, ready, None, disposition, reason, now)
        return {'status': 'SELECTED_RESULT_UNRESOLVED'}
    if (root/'results'/'source-stop.json').exists():
        return {'status': 'SOURCE_STOP'}
    if instant(cfg['result_closure_at'])-now < timedelta(seconds=45):
        return {'status': 'FINAL_TRANSPORT_WINDOW_CLOSED'}
    allocation = Reader().json(cfg['allocation'])
    stored = sum(p.stat().st_size for p in root.parent.rglob('*') if p.is_file() and not p.is_symlink())
    _require(shutil.disk_usage(root).free >= 2*2**30
        and stored+2*legacy.MAX_BODY <= allocation.get('max_storage_bytes', 10*2**30), 'RESULT_DISK_PRESSURE')
    attempt = legacy.private_output(directory/f'attempt-{stage}')
    from race_collection.freshness_campaign import Campaign
    import requests
    try:
        with ownership(cfg, attempt, now):
            now = runtime.utc_now()
            if instant(cfg['result_closure_at'])-now < timedelta(seconds=45):
                return {'status': 'FINAL_TRANSPORT_WINDOW_CLOSED'}
            # Authoritative inventory is rechecked under both original worker
            # and campaign locks; every prior original attempt remains charged.
            _, _, _, operations, transports = legacy._inventory(cfg, root, now)
            _require(operations < 72 and transports < 720, 'RESULT_CUMULATIVE_BUDGET')
            campaign = Campaign(Path(cfg['campaign_root']), development_authority=cfg['pilot_campaign_authority'])
            with requests.Session() as session:
                now = runtime.utc_now()
                if instant(cfg['result_closure_at'])-now < timedelta(seconds=45):
                    return {'status': 'FINAL_TRANSPORT_WINDOW_CLOSED'}
                runtime.put_new(attempt/'started.json', {'race_id': ready['race_id'],
                    'at': now.isoformat(), 'stage': stage, 'pre_result_sha256': ready['pre_result_sha256'],
                    'missed_prior_stages': [n for n in range(stage) if not (directory/f'attempt-{n}'/'started.json').exists()]})
                try:
                    from race_collection.prospective_speed_deadline import bounded
                    with bounded(25):
                        evidence, disposition, reason, observed = legacy._fetch(cfg, packet, attempt,
                            _EnvelopeSession(session, attempt), campaign, now)
                except Exception as error:
                    evidence, disposition, reason, observed = None, 'MISSING', 'RESULT_CHECK_INTERRUPTED_'+type(error).__name__, runtime.utc_now()
                source_evidence = None
                if disposition in {'OFFICIAL', 'AMBIGUOUS'}:
                    source_evidence = {key: runtime.reference(attempt/name) for key, name in (
                        ('body', 'response.html'), ('request', 'request.json'), ('response', 'response.json'),
                        ('http_envelope', 'http-envelope.json'))}
                    try:
                        evidence, disposition = validate_retained_response(context, source_evidence, observed)
                        reason = 'EXACT_NATIVE_OFFICIAL_FIELD'
                    except (ValueError, TypeError, KeyError):
                        evidence, disposition, reason = None, 'AMBIGUOUS', 'NATIVE_OFFICIAL_IDENTITY_OR_TERMINAL_UNVERIFIED'
                runtime.put_new(attempt/'finished.json', {'disposition': disposition, 'reason': reason,
                                                        'at': observed.isoformat()})
                if disposition in {'OFFICIAL', 'KNOWN_NONFINISH'} or stage == 2:
                    _complete(directory, ready, evidence, disposition, reason, observed, source_evidence)
                return {'status': 'SELECTED_RESULT_CHECK_RETAINED', 'operations_consumed': operations+1}
    except (BlockingIOError, FileExistsError):
        return {'status': 'WAITING_FOR_EXISTING_SOURCE_OWNER'}
    except ValueError as error:
        if str(error) in {'RESULT_COLLECTOR_BUSY', 'RESULT_SOURCE_BUSY', 'RESULT_SHARED_SOURCE_HOLD'}:
            return {'status': 'WAITING_FOR_EXISTING_SOURCE_OWNER', 'reason': str(error)}
        raise


def run_cycle(config_reference):
    config, plan, cfg = load_config(config_reference)
    root = Path(cfg['state_root'])
    legacy.private_output(root)
    results = legacy.private_output(root/'results')
    # Same lock as the original development result worker, not a second budget owner.
    with runtime.exclusive(results):
        with runtime.exclusive(Path(config['coordinator_state_root'])):
            nominate(config, plan, cfg)
        return _run_due(config, plan, cfg, root, runtime.utc_now())


def inspect_queue(config_reference):
    """Outcome-free scheduling metadata for the root's quiet-gap coordinator."""
    config, plan, cfg = load_config(config_reference)
    now = runtime.utc_now()
    rows, due, completed, operations, transports = legacy._inventory(cfg, Path(cfg['state_root']), now)
    selected = [item for item in due if item[1].get('prospective_speed_role') == 'SELECTED_NATIVE_DEVELOPMENT_RESULT'
                and item[1].get('plan') == config['plan']]
    transport_due = [item for item in selected if item[2] is not None]
    return {'status': 'SELECTED_RESULT_DUE' if selected else 'NO_SELECTED_RESULT_DUE',
        'ready_total': len(rows), 'selected_due': len(selected), 'transport_due': len(transport_due),
        'completed_total': completed, 'operations_consumed': operations,
        'transport_requests_consumed': transports, 'result_requests_stop_at': cfg['result_closure_at'],
        'transport_permitted_by_time_and_budget': now+timedelta(seconds=45) < instant(cfg['result_closure_at'])
            and operations < 72 and transports < 720}


def prepare_queue(config_reference):
    """Nominate source-only work safely before root decides whether a quiet gap is needed."""
    config, plan, cfg = load_config(config_reference)
    root = Path(cfg['state_root'])
    legacy.private_output(root)
    results = legacy.private_output(root/'results')
    with runtime.exclusive(results):
        with runtime.exclusive(Path(config['coordinator_state_root'])):
            nominated = nominate(config, plan, cfg)
        return {**inspect_queue(config_reference), 'nominated': nominated}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    print(run_cycle({'path': args.config, 'sha256': args.config_sha256})['status'])


if __name__ == '__main__':
    main()
