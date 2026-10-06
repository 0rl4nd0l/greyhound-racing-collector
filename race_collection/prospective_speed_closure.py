"""Selected-only retained closure and automatic one-look prospective evaluation.

Run at the fixed analysis time with a previously approved configuration. No
provider, collector, broad result inventory or mixed-population export exists
here. Only a selected, sealed race's exact queue key and official rows are read.
The outer durable claim consumes the look before any result source is opened.
"""
from contextlib import closing
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

from race_collection import prospective_speed_evaluation_io as evaluation
from race_collection import prospective_speed_plan as planning
from race_collection import prospective_speed_runtime as runtime
from race_collection.retained_card_timing_coverage import Reader, instant
from src.predictor.comparison_results import ComparisonResultSource


class ClosureRejected(ValueError):
    """Safe failure reason; never include target values."""


def _require(condition, reason):
    if not condition:
        raise ClosureRejected(reason)


def _pin(value):
    return hashlib.sha256(runtime.encoded(value)).hexdigest()


def _private_read(path):
    return Reader().json(runtime.reference(path))


def _local(path, root):
    path, root = Path(path), Path(root)
    _require(path.is_absolute() and path.resolve() == path and root.is_absolute()
        and root.resolve() == root and path.is_relative_to(root), 'RETENTION_PATH_OUTSIDE_APPROVED_ROOT')
    return path


def _configuration(config_reference):
    config = Reader().json(config_reference)
    plan = Reader().json(config['plan'])
    planning._validate_plan(plan)
    activation = Reader().json(config['activation'])
    _require(config.get('schema_version') == 'prospective_speed_closure_configuration_v1'
        and config.get('status') == 'AUTHORIZED_PROSPECTIVE_CLOSURE_CONSUMER'
        and config.get('authority_reference')
        and activation.get('status') == 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
        and activation.get('plan_sha256') == config['plan']['sha256']
        and activation.get('development_precedence_verified') is True
        and activation.get('result_retention_routing_verified') is True
        and activation.get('automatic_result_projection_authorized') is True
        and activation.get('retention_by_date') == config['retention_by_date']
        and activation.get('additional_source_requests') == 0
        and activation.get('additional_result_requests') == 0, 'AUTOMATIC_CLOSURE_AUTHORITY_REQUIRED')
    _require(set(config['retention_by_date']) == set(plan['dates']), 'RETENTION_DATES_CHANGED')
    for source in config['retention_by_date'].values():
        if source.get('kind') == 'DEVELOPMENT_SELECTED_NATIVE':
            from race_collection.prospective_speed_results import load_config
            _require(set(source) == {'kind', 'root', 'authority', 'bridge_config'}, 'RETENTION_SOURCE_INVALID')
            bridge, bridge_plan, legacy = load_config(source['bridge_config'])
            _require(bridge['plan'] == config['plan'] and bridge_plan == plan
                and source['root'] == legacy['state_root']
                and source['authority'] == bridge['legacy_runtime'], 'DEVELOPMENT_RETENTION_ROUTING_CHANGED')
            continue
        _require(set(source) == {'root', 'queue_database', 'result_database', 'authority'},
                 'RETENTION_SOURCE_INVALID')
        _local(source['queue_database'], source['root'])
        _local(source['result_database'], source['root'])
        Reader().read(source['authority'])  # Exact pre-approved retention control, never outcomes.
    state = Path(config['coordinator_state_root'])
    _require(state.is_absolute() and state.resolve() == state
        and Path(config['forecast_root']) == state/'attempts', 'COORDINATOR_ATTEMPTS_ROOT_CHANGED')
    output = Path(config['closure_root'])
    _require(output.is_absolute() and output.resolve() == output and output != state
        and not output.is_relative_to(state/'attempts'), 'CLOSURE_OUTPUT_PATH_UNSAFE')
    return config, plan


def _census(config, plan, output):
    """Use exact two date files and selected job keys; never scan forecast successes."""
    state = Path(config['coordinator_state_root'])
    populations, accounts, selected = [], [], {}
    for day in plan['dates']:
        day_root = state / day
        path = day_root/'date-accounting.json'
        if not path.exists():
            accounts.append({'local_date': day, 'status': 'FREEZE_INTERRUPTED',
                'reason': 'NO_TERMINAL_DATE_RECORD_AT_FIXED_EVALUATION', 'population_sha256': None})
            continue
        account = _private_read(path)
        _require(account['local_date'] == day, 'DATE_ACCOUNTING_IDENTITY_CHANGED')
        accounts.append(account)
        if account['status'] != 'POPULATION_FROZEN':
            continue
        population_ref = account['population']
        _local(population_ref['path'], state)
        population = Reader().json(population_ref)
        planning._validate_population(plan, population)
        _require(planning._digest(population) == account['population_sha256'], 'POPULATION_BINDING_CHANGED')
        populations.append(population_ref)
        for race_id in population['selected_race_ids']:
            _require(race_id not in selected, 'SELECTED_MEMBERSHIP_DUPLICATE')
            key = hashlib.sha256(race_id.encode()).hexdigest()
            disposition_path = day_root/'jobs'/(key+'.disposition.json')
            disposition = (_private_read(disposition_path) if disposition_path.exists()
                else {'race_id': race_id, 'status': 'UNATTEMPTED_AT_HORIZON'})
            _require(disposition['race_id'] == race_id, 'DISPOSITION_IDENTITY_CHANGED')
            status = disposition['status']
            if status == 'ATTEMPT_ALREADY_CONSUMED':
                status = disposition['original_status']
            attempt = Path(config['forecast_root'])/key
            completion = disposition.get('completion')
            if status in {'SEALED_PREJUMP', 'LATE_SPEED_SEAL'} and completion is None:
                completion = runtime.reference(attempt/'completion.json')
            terminal = disposition.get('terminal')
            if status == 'LATE_SPEED_SEAL' and terminal is None:
                terminal = runtime.reference(attempt/'terminal.json')
            entry = {'race_id': race_id, 'race_date': day, 'forecast_status': status,
                'forecast_completion': completion, 'forecast_terminal': terminal,
                'runner_ids': [], 'label_status': 'MISSING_AT_DEADLINE' if status == 'SEALED_PREJUMP' else 'UNREAD_FORECAST_FAILURE',
                'target': None, 'target_role': None, 'identity_proof': None,
                'closure_evidence': None, 'failure_evidence': disposition.get('evidence')}
            if status == 'INTERRUPTED_PARTIAL_CLAIM':
                source_job = day_root/'jobs'/(key+'.job.json')
                if source_job.exists():
                    entry['source_job'] = runtime.reference(source_job)
            selected[race_id] = (entry, population)
    accounts_ref = runtime.put_new(output/'date-accounting.json', accounts)
    shell = {'plan': config['plan'], 'allocation': plan['authority']['allocation'],
        'populations': populations, 'date_accounting': accounts_ref,
        'forecast_root': config['forecast_root']}
    reader = evaluation._Checked()
    _, _, members, by_date = evaluation._membership(reader, shell, plan)
    shell['population_by_date'] = by_date
    _require(set(members) == set(selected), 'CLOSURE_MEMBERSHIP_CHANGED')
    prepared = []
    for race_id, (entry, population) in sorted(selected.items()):
        if entry['forecast_status'] == 'SEALED_PREJUMP':
            completion = reader.json(entry['forecast_completion'])
            seal = reader.json(completion['seal'])
            terminal = reader.json(seal['terminal'])
            payload = reader.output_json(terminal['payload'])
            entry['runner_ids'] = [r['runner_id'] for r in payload['predictions']]
            evaluation._forecast(reader, entry, shell, plan, population)
            context = _native_context(reader, payload)
        else:
            evaluation._failed_forecast(reader, entry, shell)
            context = None
        prepared.append((entry, context))
    runtime.put_new(output/'membership-before-results.json', {
        'populations': populations, 'date_accounting': accounts_ref,
        'entries': [entry for entry, _ in prepared], 'result_sources_opened': 0})
    return shell, prepared


def _native_context(reader, payload):
    """Rebuild expected official identity from source-bound native forecast files."""
    member = payload['source_member']
    manifest = reader.json({k: member['bundle_manifest'][k] for k in ('path', 'sha256')})
    base = Path(member['bundle_manifest']['path']).parent
    def file(name):
        return reader.json({'path': str(base/name), 'sha256': manifest['files'][name]['sha256']})
    request, result = file('request.json'), file('result.json')
    _require(request['race_id'] == result['race']['race_id'] == payload['race_id']
        and request['job_id'] == result['job_id'] == manifest['job_id']
        and instant(request['jump_timestamp']) == instant(payload['jump_at'])
        and {r['source_native_runner_id'] for r in request['runners']}
            == {r['runner_id'] for r in payload['predictions']}
        and len(request['runners']) == len(payload['predictions'])
        and {r['source_native_runner_id']: r['box_number'] for r in request['runners']}
            == {r['runner_id']: r['box_number'] for r in payload['predictions']},
        'NATIVE_RESULT_CONTEXT_CHANGED')
    native = SimpleNamespace(job_id=request['job_id'], input=SimpleNamespace(
        race_id=payload['race_id'], jump_timestamp=payload['jump_at'],
        ordered_runners=[{'box': r['box_number'], 'name': r['display_name'],
            'source_native_runner_id': r['source_native_runner_id']} for r in request['runners']]))
    bundle = SimpleNamespace(directory=base.name, manifest=manifest, result=result)
    return {'job': native, 'bundle': bundle, 'prediction_bundles': base.parent,
        'boxes': [r['box_number'] for r in payload['predictions']],
        'jump_at': payload['jump_at']}


def _queue_record(source, job):
    path = _local(source['queue_database'], source['root'])
    if not path.exists():
        return None
    _require(not any(Path(str(path)+suffix).exists() or Path(str(path)+suffix).is_symlink()
        for suffix in ('-wal', '-shm', '-journal')),
             'RESULT_QUEUE_NOT_QUIESCENT')
    before = path.stat()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True, timeout=1)) as db:
        db.execute('PRAGMA query_only=ON')
        deadline = time.monotonic()+1.
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        rows = db.execute('SELECT job,jump,state FROM jobs WHERE race=? LIMIT 2',
                          (job.input.race_id,)).fetchall()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _require(identity(before) == identity(after), 'RESULT_QUEUE_CHANGED')
    if not rows:
        return None
    _require(len(rows) == 1 and rows[0][0] == job.job_id
        and instant(rows[0][1]) == instant(job.input.jump_timestamp), 'RESULT_QUEUE_IDENTITY_CHANGED')
    return {'job': rows[0][0], 'jump_at': rows[0][1], 'state': rows[0][2],
        'source_database': str(path), 'selected_race_id': job.input.race_id,
        'source_identity': list(identity(after))}


def _nonfinish(source, context, now, cutoff):
    """Revalidate exact recorded source bytes, not just a previously asserted flag."""
    from scripts.reconcile_comparison_result_identity import RetainedDeadline
    from src.predictor.comparison_terminal_results import known_nonfinish_evidence
    job, bundle = context['job'], context['bundle']
    _require(Path(job.job_id).name == job.job_id, 'RESULT_JOB_PATH_INVALID')
    path = _local(Path(source['root'])/'terminal-results'/(job.job_id+'.json'), source['root'])
    record = _private_read(path)
    _require(record['job_id'] == job.job_id and record['race_id'] == job.input.race_id
        and instant(record['captured_at']) <= cutoff, 'NONFINISH_RECORD_IDENTITY_CHANGED')
    refs = record['source_evidence']
    _require(set(refs) == {'body', 'request', 'response'}, 'NONFINISH_SOURCE_ROLES_INVALID')
    for ref in refs.values():
        _local(ref['path'], Path(source['root'])/'attempts')
    body = Reader().read(refs['body'])
    # Fresh local execution bound for already-authorized retained processing;
    # this is not an extension of the expired provider-request deadline.
    deadline = RetainedDeadline({'expires_at': (now+timedelta(minutes=10)).isoformat()}, None)
    reconstructed = known_nonfinish_evidence(job, bundle, body, record['source_url'],
        instant(record['captured_at']), cutoff, deadline=deadline,
        prediction_bundles=context['prediction_bundles'], source_evidence=refs)
    _require(reconstructed == record, 'NONFINISH_RECONSTRUCTION_CHANGED')
    winners = {r['box_number'] for r in reconstructed['runner_results'] if r['finish_position'] == 1}
    _require(bool(winners), 'NONFINISH_NO_VERIFIED_WINNER')
    return winners, record['captured_at'], {'retained_terminal': runtime.reference(path), 'record': record}


def _closure(source, context, now, cutoff):
    if source.get('kind') == 'DEVELOPMENT_SELECTED_NATIVE':
        return _development_closure(source, context, cutoff)
    queue = _queue_record(source, context['job'])
    if queue is None:
        return 'MISSING_RESULT_QUEUE', None
    if queue['state'] == 'CLOSED_NON_FINISH':
        winners, observed, evidence = _nonfinish(source, context, now, cutoff)
        return 'KNOWN_NONFINISH_WIN_ELIGIBLE', {'winners': winners, 'observed_at': observed,
            'evidence': {'queue': queue, **evidence}}
    if queue['state'] != 'CLOSED':
        return ('QUARANTINED_RETAINED_RESULT' if queue['state'] == 'QUARANTINED'
                else 'MISSING_AT_DEADLINE'), None
    database = _local(source['result_database'], source['root'])
    value = ComparisonResultSource(database).read(context['job'], context['bundle'], now=cutoff)
    if value['state'] != 'RESULT_AVAILABLE':
        return ('QUARANTINED_RESULT_IDENTITY' if value['state'] == 'RESULT_REJECTED'
                else 'MISSING_RETAINED_OFFICIAL_RESULT'), None
    winners = {r['box_number'] for r in value['evidence']['runner_rows'] if r['is_winner']}
    _require(bool(winners) and winners <= set(context['boxes']), 'RESULT_WINNER_OUTSIDE_FROZEN_FIELD')
    return 'FULL_ORDER_WIN_ELIGIBLE', {'winners': winners,
        'observed_at': value['evidence']['race_rows'][0]['captured_at'],
        'evidence': {'queue': queue, 'result_database': str(database),
            'selected_race_id': context['job'].input.race_id, 'validated_evidence': value}}


def _development_closure(source, context, cutoff):
    """Open only this selected native job's original development retention files."""
    from race_collection.prospective_speed_results import load_config, _admitted, validate_retained_response
    config, plan, cfg = load_config(source['bridge_config'])
    race_id = context['job'].input.race_id
    key = hashlib.sha256(race_id.encode()).hexdigest()
    ready_path = _local(Path(source['root'])/'ready'/(key+'.json'), source['root'])
    if not ready_path.exists():
        return 'MISSING_RESULT_QUEUE', None
    ready = _private_read(ready_path)
    _, admitted = _admitted(config, plan, ready)
    _require(ready['race_id'] == race_id and ready['job_id'] == context['job'].job_id
        and admitted['boxes'] == context['boxes'], 'DEVELOPMENT_RETAINED_JOB_CHANGED')
    root = Path(source['root'])/'results'/key
    completed = root/'complete.json'
    if not completed.exists():
        return 'MISSING_AT_DEADLINE', None
    metadata = _private_read(completed)
    _require(metadata['race_id'] == race_id and metadata['pre_result_sha256'] == ready['pre_result_sha256'],
             'DEVELOPMENT_RESULT_COMPLETION_CHANGED')
    if metadata['status'] not in {'OFFICIAL_RESULT_RETAINED', 'KNOWN_NONFINISH_RETAINED'}:
        return ('MISSING_AT_DEADLINE' if metadata['status'] == 'UNRESOLVED_AT_CLOSURE'
                else 'QUARANTINED_RETAINED_RESULT'), None
    result_ref = metadata['result']
    _require(Path(result_ref['path']) == root/'official-result.json', 'DEVELOPMENT_RESULT_ROLE_CHANGED')
    record = Reader().json(result_ref)
    _require(record['schema_version'] == 'development_official_result_v1'
        and record['prospective_speed_role'] == 'SELECTED_NATIVE_DEVELOPMENT_RESULT'
        and record['synthetic'] is False and record['race_id'] == race_id
        and record['race_key'] == ready['race_key'] and record['disposition'] in {'OFFICIAL', 'KNOWN_NONFINISH'}
        and record['official_source'] == 'thedogs_official'
        and record['forecast_completion'] == ready['forecast_completion']
        and instant(record['observed_at']) <= cutoff, 'DEVELOPMENT_OFFICIAL_IDENTITY_CHANGED')
    evidence = record['official_evidence']
    from race_collection.development_examples import canonical
    _require(hashlib.sha256(canonical(evidence)).hexdigest() == record['source_evidence_sha256'],
             'DEVELOPMENT_OFFICIAL_EVIDENCE_CHANGED')
    refs = record['source_evidence']
    parents = {Path(_local(ref['path'], root)).parent for ref in refs.values()}
    _require(len(parents) == 1, 'DEVELOPMENT_RESPONSE_ATTEMPT_CHANGED')
    attempt = next(iter(parents))
    _require(attempt.name in {'attempt-0', 'attempt-1', 'attempt-2'} and attempt.parent == root,
             'DEVELOPMENT_RESPONSE_ATTEMPT_CHANGED')
    started = _private_read(attempt/'started.json')
    charge = _private_read(attempt/'charge.json')
    _require(started['race_id'] == race_id and started['pre_result_sha256'] == ready['pre_result_sha256']
        and charge['status'] == 'CAMPAIGN_CHARGED', 'DEVELOPMENT_RESULT_CHARGE_CHANGED')
    reconstructed, disposition = validate_retained_response(context, refs, cutoff)
    _require(reconstructed == evidence and disposition == record['disposition'], 'DEVELOPMENT_RESULT_RECONSTRUCTION_CHANGED')
    if disposition == 'KNOWN_NONFINISH':
        _require(metadata['status'] == 'KNOWN_NONFINISH_RETAINED', 'DEVELOPMENT_RESULT_STATUS_CHANGED')
        winners = {row['box_number'] for row in evidence['runner_results'] if row['finish_position'] == 1}
        status = 'KNOWN_NONFINISH_WIN_ELIGIBLE'
    else:
        _require(metadata['status'] == 'OFFICIAL_RESULT_RETAINED', 'DEVELOPMENT_RESULT_STATUS_CHANGED')
        winners = {row['box_number'] for row in evidence['runner_rows'] if row['is_winner']}
        status = 'FULL_ORDER_WIN_ELIGIBLE'
    return status, {'winners': winners, 'observed_at': record['observed_at'],
        'evidence': {'ready': runtime.reference(ready_path), 'completion': runtime.reference(completed),
            'official_result': result_ref, 'validated_evidence': evidence}}


def _project(entry, context, value, output, config, plan):
    key = hashlib.sha256(entry['race_id'].encode()).hexdigest()
    directory = output/'selected-results'/key
    directory.mkdir(parents=True, mode=0o700)
    closure_ref = runtime.put_new(directory/'official-closure.private.json', value['evidence'])
    winners = value['winners']
    target_ref = runtime.put_new(directory/'win-target.private.json', {
        'schema_version': 'prospective_speed_verified_win_target_v1',
        'role': 'SELECTED_DEVELOPMENT_WIN_TARGET', 'race_id': entry['race_id'],
        'race_date': entry['race_date'], 'jump_at': context['jump_at'],
        'runner_ids': entry['runner_ids'],
        'outcome': [1/len(winners) if box in winners else 0. for box in context['boxes']],
        'label_status': entry['label_status'], 'official_observed_at': value['observed_at'],
        'closure_evidence': closure_ref, 'plan': config['plan'], 'allocation': plan['authority']['allocation']})
    proof_ref = runtime.put_new(directory/'identity-proof.json', {
        'schema_version': 'prospective_speed_result_identity_proof_v1',
        'status': 'VERIFIED_SELECTED_DEVELOPMENT_WIN_TARGET',
        'race_id': entry['race_id'], 'runner_ids': entry['runner_ids'],
        'target': target_ref, 'closure_evidence': closure_ref,
        'forecast_completion': entry['forecast_completion'],
        'allocation': plan['authority']['allocation'], 'plan': config['plan']})
    return {**entry, 'target': target_ref, 'target_role': 'SELECTED_DEVELOPMENT_WIN_TARGET',
        'identity_proof': proof_ref, 'closure_evidence': closure_ref}


def finalize(config_reference):
    """Fully automatic selected closure → manifest → job → single evaluation."""
    config, plan = _configuration(config_reference)
    now = runtime.utc_now()
    gate = planning.evaluation_gate(plan, now=now.isoformat(), collection_terminal=True, closure_terminal=True)
    if gate != 'EVALUATION_DUE':
        return {'status': gate, 'result_accesses_consumed': 0}
    root = Path(config['closure_root'])
    with runtime.exclusive(root):
        claim_path, terminal_path = root/'closure-claim.json', root/'closure-terminal.json'
        if claim_path.exists():
            claim_ref = runtime.reference(claim_path)
            try:
                old = Reader().json(claim_ref)
                _require(old['config'] == config_reference, 'CLOSURE_CLAIM_CONFIGURATION_CHANGED')
            except json.JSONDecodeError:
                old = None
            if not terminal_path.exists():
                runtime.put_new(terminal_path, {'status': 'INTERRUPTED_CLOSURE_LOOK_NO_RETRY',
                    'claim': claim_ref, 'at': now.isoformat()})
            return {'status': 'CLOSURE_LOOK_ALREADY_CONSUMED', 'terminal': runtime.reference(terminal_path),
                'result_accesses_consumed': 0}
        # Date/member/forecast integrity checks are outcome-free and precede claim.
        # A crash during outcome-free preparation does not consume a look.
        # Preserve that directory and use a new finite-numbered preparation.
        for number in range(1, 1001):
            preparation = root/f'prepared-{number:04d}'
            try:
                preparation.mkdir(mode=0o700)
                break
            except FileExistsError:
                continue
        else:
            raise ClosureRejected('PREFLIGHT_ATTEMPT_LIMIT_EXHAUSTED')
        from race_collection.prospective_speed_coordinator import close_elapsed_dates
        state = Path(config['coordinator_state_root'])
        with runtime.exclusive(state):
            close_elapsed_dates(state, plan, now.astimezone(planning.ZONE))
            shell, prepared = _census(config, plan, preparation)
        claim = runtime.put_new(claim_path, {'schema_version': 'prospective_speed_closure_claim_v1',
            'config': config_reference, 'plan': config['plan'], 'activation': config['activation'],
            'membership': runtime.reference(preparation/'membership-before-results.json'),
            'retention_by_date': config['retention_by_date'], 'at': runtime.utc_now().isoformat(),
            'evaluation_looks_consumed': 1})
        consumed, entries = 0, []
        try:
            cutoff = instant(plan['result_requests_stop_at'])
            for entry, context in prepared:
                if context is not None:
                    consumed += 1
                    runtime.put_new(root/f'retention-access-{consumed:03d}.json', {
                        'race_id': entry['race_id'], 'source': config['retention_by_date'][entry['race_date']],
                        'claim': claim, 'at': runtime.utc_now().isoformat()})
                    try:
                        label_status, value = _closure(config['retention_by_date'][entry['race_date']],
                            context, runtime.utc_now(), cutoff)
                        entry = {**entry, 'label_status': label_status}
                        if value is not None:
                            entry = _project(entry, context, value, root, config, plan)
                    except (ValueError, KeyError, OSError, TypeError, sqlite3.Error) as error:
                        entry = {**entry, 'label_status': 'QUARANTINED_OFFICIAL_VALIDATION_FAILURE'}
                        runtime.put_new(root/f'closure-failure-{consumed:03d}.json', {
                            'race_id': entry['race_id'], 'error_type': type(error).__name__,
                            'claim': claim, 'retained_status': entry['label_status']})
                entries.append(entry)
            members = sorted([{'race_id': e['race_id'], 'race_date': e['race_date']} for e in entries],
                             key=lambda e: e['race_id'])
            manifest_ref = runtime.put_new(root/'result-manifest.json', {
                'schema_version': 'prospective_speed_admitted_result_manifest_v1',
                'status': 'ROOT_VERIFIED_SELECTED_DEVELOPMENT_CLOSURE', 'plan': config['plan'],
                'allocation': plan['authority']['allocation'], 'members_sha256': _pin(members),
                'collection_terminal': True, 'closure_terminal': True, 'entries': entries})
            authority_ref = runtime.put_new(root/'evaluation-authority.json', {
                'schema_version': 'prospective_speed_evaluation_authority_v1',
                'status': 'AUTHORIZED_SELECTED_DEVELOPMENT_EVALUATION',
                'authority_reference': config['authority_reference'], 'plan': config['plan'],
                'allocation': plan['authority']['allocation'], 'result_manifest': manifest_ref,
                'members_sha256': _pin(members), 'additional_result_requests': 0,
                'underlying_approved_activation': config['activation'], 'closure_claim': claim})
            job_ref = runtime.put_new(root/'evaluation-job.json', {k: v for k, v in shell.items()
                if k != 'population_by_date'} | {'schema_version': 'prospective_speed_evaluation_job_v1',
                'result_manifest': manifest_ref, 'evaluation_authority': authority_ref})
            result = evaluation.run_evaluation(job_ref, root/'evaluation')
            terminal = {'status': result['status'], 'claim': claim, 'evaluation': result,
                'selected_races': len(entries), 'retention_accesses_consumed': consumed,
                'additional_provider_requests': 0, 'at': runtime.utc_now().isoformat()}
        except Exception as error:
            terminal = {'status': 'FAILED_CLOSURE_LOOK_NO_RETRY', 'claim': claim,
                'error_type': type(error).__name__, 'retention_accesses_consumed': consumed,
                'at': runtime.utc_now().isoformat()}
        terminal_ref = runtime.put_new(terminal_path, terminal)
        return {'status': terminal['status'], 'terminal': terminal_ref,
            'result_accesses_consumed': consumed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    result = finalize({'path': args.config, 'sha256': args.config_sha256})
    print(result['status'])


if __name__ == '__main__':
    main()
