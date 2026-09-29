"""Default-off, offline assembly of explicitly allocated development examples.

The collector and its frozen forecasts remain unchanged. This module consumes
one existing verified retained-input prediction, never discovers/acquires data,
and admits identities before reading prediction/history/result bytes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile

# Reviewed identity-only authorities. Real admission must include the complete
# pinned baseline; arbitrary caller-built empty registries cannot authorize it.
RESERVATION_PINS = {
    'historical_protection': '9fddce8fa70ea96c4d7a6c33ef61594ddeb47e254cd08bb87555b2d94fc4da88',
    'exclusive_comparison': 'b708fa973aa972b8cd248b4b4d3269fa7aa16402755ee0fb84da5212db6822d1',
    'approved_reservation_review': '4c1f859c8292fa76bb1a70809133f56fa534fc16c98d0371d9d36a5875b7030e',
}
PILOT_DATES = ['2026-10-03', '2026-10-04', '2026-10-10', '2026-10-11']
SELECTION_POLICY = 'first_six_1310_1420_melbourne_before_WIN_qualification_v1'


class DevelopmentRejected(ValueError):
    """Finite reason only; never include protected records in exceptions."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    try:
        result = datetime.fromisoformat(value)
        if result.utcoffset() is None:
            raise ValueError()
        return result
    except (TypeError, ValueError):
        raise DevelopmentRejected('INVALID_TIMESTAMP') from None


def checked(ref):
    """Explicit immutable local member; do not traverse symlinks or directories."""
    path = Path(ref['path'])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise DevelopmentRejected('INVALID_SOURCE_PATH')
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise DevelopmentRejected('INVALID_SOURCE_FILE')
    raw = path.read_bytes()
    if digest(raw) != ref['sha256']:
        raise DevelopmentRejected('SOURCE_HASH_CHANGED')
    return raw


def read(ref):
    return json.loads(checked(ref))


def put(path, value):
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.read_bytes() != raw:
            raise DevelopmentRejected('IMMUTABLE_RECORD_CHANGED')
        return digest(raw)
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(raw)


def private_output(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise DevelopmentRejected('OUTPUT_SYMLINK_FORBIDDEN')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    mode=output.stat()
    if mode.st_uid!=os.getuid() or stat.S_IMODE(mode.st_mode)&0o077:
        raise DevelopmentRejected('OUTPUT_NOT_PRIVATE')
    return output


def race_key(race_id):
    match = re.fullmatch(r'Race ([1-9][0-9]*) - ([A-Z0-9_]+) - (\d{4}-\d{2}-\d{2})', race_id)
    if not match:
        raise DevelopmentRejected('NONCANONICAL_RACE_ID')
    return f'{match[3]}|{match[2]}|{int(match[1])}'


def selected_population(races, local_date):
    from zoneinfo import ZoneInfo
    zone=ZoneInfo('Australia/Melbourne')
    intended=[]
    for row in races:
        jump=stamp(row['jump_at']).astimezone(zone)
        if jump.date().isoformat()==local_date and '13:10' <= jump.strftime('%H:%M') <= '14:20':
            if row['race_key'] != race_key(row['race_id']):
                raise DevelopmentRejected('POPULATION_IDENTITY_MISMATCH')
            intended.append(row)
    intended.sort(key=lambda row:(stamp(row['jump_at']),row['race_id']))
    if len({row['race_id'] for row in intended}) != len(intended):
        raise DevelopmentRejected('POPULATION_IDENTITY_DUPLICATE')
    return intended,[row['race_id'] for row in intended[:6]]


def freeze_population(index_path, evidence_root, allocation_path, allocation_sha256, output):
    """Read the existing verified index once; no discovery or provider call."""
    from zoneinfo import ZoneInfo
    allocation=read({'path':str(Path(allocation_path).absolute()),'sha256':allocation_sha256})
    if allocation.get('status')!='AUTHORIZED' or allocation.get('dates')!=PILOT_DATES:
        raise DevelopmentRejected('DEVELOPMENT_DISABLED')
    registry=read(allocation['reservation_registry'])
    if {r.get('kind'):r['sha256'] for r in registry['sources']}!=RESERVATION_PINS:
        raise DevelopmentRejected('RESERVATION_REVIEW_REQUIRED')
    for ref in registry['sources']:checked(ref)
    now=datetime.now(timezone.utc);local=now.astimezone(ZoneInfo('Australia/Melbourne'))
    if local.date().isoformat() not in PILOT_DATES or local.strftime('%H:%M')!='12:50':
        raise DevelopmentRejected('POPULATION_FREEZE_NOT_DUE')
    from race_collection.synchronous_manual_capture import bounded_current_race_index
    view=bounded_current_race_index(current_time=now,timeout_seconds=5,index_path=Path(index_path),
        evidence_root=Path(evidence_root),max_age_seconds=300,return_verified_view=True)
    rows=[{'race_id':r['race_id'],'race_key':race_key(r['race_id']),'jump_at':r['jump_datetime'],
        'runners':r['runners'],'source_native_race_id':r.get('source_native_race_id'),'url':r['race_url']}
        for r in view.races]
    intended,selected=selected_population(rows,local.date().isoformat())
    value={'schema_version':'development_population_freeze_v1','synthetic':False,
        'allocation_id':allocation['allocation_id'],'allocation_sha256':allocation_sha256,
        'selection_policy':SELECTION_POLICY,'local_date':local.date().isoformat(),
        'frozen_at':now.isoformat(),'source_observed_at':view.source_generated_at,
        'source_index':str(Path(index_path).absolute()),'source_packet_sha256':view.packet_sha256,
        'observed_races':rows,'intended':intended,'selected_race_ids':selected,
        'coverage_basis':'all entries in verified collector index; source metadata exclusions remain in original collector accounting'}
    output=Path(output).absolute()
    population_hash=put(output,value)
    completed=datetime.now(timezone.utc)
    if completed.astimezone(ZoneInfo('Australia/Melbourne')).strftime('%H:%M')!='12:50':
        put(Path(str(output)+'.failure.json'),{'status':'FREEZE_CROSSED_CUTOFF','population_sha256':population_hash})
        raise DevelopmentRejected('POPULATION_FREEZE_CROSSED_CUTOFF')
    put(Path(str(output)+'.completion.json'),{'status':'POPULATION_FROZEN',
        'population_sha256':population_hash,'completed_at':completed.isoformat()})
    if datetime.now(timezone.utc).astimezone(ZoneInfo('Australia/Melbourne')).strftime('%H:%M')!='12:50':
        put(Path(str(output)+'.failure.json'),{'status':'FREEZE_CROSSED_CUTOFF','population_sha256':population_hash})
        raise DevelopmentRejected('POPULATION_FREEZE_CROSSED_CUTOFF')
    return value


def admit(access_path, expected_sha256, race_id):
    access = read({'path': str(Path(access_path).absolute()), 'sha256': expected_sha256})
    if (access.get('schema_version') != 'development_access_v1'
            or access.get('enabled') is not True
            or access.get('status') not in {'SYNTHETIC_FIXTURE', 'AUTHORIZED_DEVELOPMENT'}):
        raise DevelopmentRejected('DEVELOPMENT_DISABLED')
    synthetic = access['status'] == 'SYNTHETIC_FIXTURE'
    allocation = read(access['allocation'])
    if (allocation.get('schema_version') != 'development_allocation_v1'
            or allocation.get('status') != ('SYNTHETIC_FIXTURE' if synthetic else 'AUTHORIZED')
            or not allocation.get('authority_reference')
            or allocation.get('mode') != 'single_snapshot'
            or allocation.get('operational_owner') != 'primary_orchestrator'):
        raise DevelopmentRejected('ALLOCATION_NOT_AUTHORIZED')
    registry = read(allocation['reservation_registry'])
    if registry.get('schema_version') != 'development_reservation_registry_v1':
        raise DevelopmentRejected('RESERVATION_REGISTRY_INVALID')
    if not synthetic:
        supplied = {ref.get('kind'): ref['sha256'] for ref in registry.get('sources', [])}
        if supplied != RESERVATION_PINS or len(registry['sources']) != len(RESERVATION_PINS):
            raise DevelopmentRejected('RESERVATION_REVIEW_REQUIRED')
        if (allocation.get('allocation_id') != 'development-single-snapshot-20261003-v1'
                or allocation.get('dates') != PILOT_DATES
                or allocation.get('max_capture_attempts') != 24
                or allocation.get('max_attempts_per_date') != 6
                or allocation.get('history_policy') != 'explicitly_authorized_strictly_earlier_machine_only_including_reserved'
                or not allocation.get('machine_history_authority_reference')
                or registry.get('owner_attestation') != 'primary_confirmed_no_additional_reservations'):
            raise DevelopmentRejected('ALLOCATION_OUTSIDE_REVIEWED_PILOT')
    # Read all reservation authorities before selecting any data-bearing path.
    reservations = [(ref, read(ref)) for ref in registry['sources']]
    exceptions = [read(ref) for ref in allocation.get('reservation_amendments', [])]
    members = access['members']
    if race_id not in members:
        raise DevelopmentRejected('RACE_NOT_ALLOCATED')
    member = members[race_id]
    key = race_key(race_id)
    jump = stamp(member['jump_at'])
    day = key[:10]
    from zoneinfo import ZoneInfo
    if day != jump.astimezone(ZoneInfo('Australia/Melbourne')).date().isoformat() or key != member['race_key']:
        raise DevelopmentRejected('RACE_IDENTITY_MISMATCH')
    if day not in allocation['dates'] or not stamp(allocation['starts_at']) <= jump < stamp(allocation['ends_at']):
        raise DevelopmentRejected('RACE_OUTSIDE_ALLOCATION')
    population=None
    if not synthetic:
        population=read(access['population'])
        population_completion=read(access['population_completion'])
        if (Path(access['population']['path']+'.failure.json').exists()
                or population_completion.get('status')!='POPULATION_FROZEN'
                or population_completion.get('population_sha256')!=access['population']['sha256']):
            raise DevelopmentRejected('POPULATION_FREEZE_INCOMPLETE')
        intended,selected=selected_population(population['observed_races'],day)
        frozen=stamp(population['frozen_at'])
        if (population.get('schema_version')!='development_population_freeze_v1'
                or population.get('synthetic') is not False
                or population.get('allocation_sha256')!=access['allocation']['sha256']
                or population.get('selection_policy')!=SELECTION_POLICY
                or population.get('local_date')!=day
                or frozen.astimezone(ZoneInfo('Australia/Melbourne')).date().isoformat()!=day
                or stamp(population_completion['completed_at']).astimezone(ZoneInfo('Australia/Melbourne')).strftime('%Y-%m-%dT%H:%M')!=day+'T12:50'
                or population.get('intended')!=intended or population.get('selected_race_ids')!=selected
                or race_id not in selected
                or frozen.astimezone(ZoneInfo('Australia/Melbourne')).strftime('%H:%M')!='12:50'
                or not 0 <= (frozen-stamp(population['source_observed_at'])).total_seconds() <= 300):
            raise DevelopmentRejected('POPULATION_NOT_FROZEN_BEFORE_QUALIFICATION')
        frozen_race=next(row for row in intended if row['race_id']==race_id)
        if stamp(frozen_race['jump_at'])!=jump:
            raise DevelopmentRejected('FROZEN_POPULATION_JUMP_CHANGED')
    for ref, reservation in reservations:
        kind = ref.get('kind')
        if kind == 'approved_reservation_review':
            if reservation.get('status') != 'RECHECKED_NO_NEW_RESERVATIONS' or reservation.get('deferrals_preserved') is not True:
                raise DevelopmentRejected('RESERVATION_REVIEW_INVALID')
            for path in reservation['checks']:
                if Path(path).exists() or Path(path).is_symlink():
                    raise DevelopmentRejected('SUCCESSOR_ACTIVATION_REQUIRES_NEW_REVIEW')
            continue
        if kind == 'historical_protection':
            if set(reservation) != {'additional_scope_cap', 'records', 'windows'}:
                raise DevelopmentRejected('RESERVATION_SCHEMA_INVALID')
        elif kind == 'exclusive_comparison':
            if reservation.get('status') != 'AUTHORIZED_EXCLUSIVE_ALLOCATION':
                raise DevelopmentRejected('RESERVATION_SCHEMA_INVALID')
        else:
            raise DevelopmentRejected('RESERVATION_SCHEMA_UNKNOWN')
        collision = key in reservation.get('records', [])
        # The original forward-successor window remains a deferred claim. A
        # separately approved amendment must reconcile it explicitly as well.
        collision |= any(window['start'] <= day and (window['end'] is None or day <= window['end'])
                         for window in reservation.get('windows', []))
        if reservation.get('status') == 'AUTHORIZED_EXCLUSIVE_ALLOCATION':
            collision |= stamp(reservation['starts_at']) <= jump < stamp(reservation['ends_at'])
        if collision and not any(
            amend.get('schema_version') == 'development_reservation_amendment_v1'
            and amend.get('status') == 'AUTHORIZED'
            and amend.get('authority_reference')
            and amend.get('prior_allocation_sha256') == ref['sha256']
            and day in amend.get('candidate_local_dates', [])
            and amend.get('selection_policy') == SELECTION_POLICY
            and race_id in population['selected_race_ids']
            and amend.get('development_allocation_id') == allocation['allocation_id']
            for amend in exceptions
        ):
            raise DevelopmentRejected('RESERVATION_COLLISION')
    accounting = read(access['opportunities'])
    if accounting.get('schema_version') != 'development_opportunities_v1' or accounting.get('complete') is not True:
        raise DevelopmentRejected('OPPORTUNITY_ACCOUNTING_INCOMPLETE')
    opportunities = accounting['opportunities']
    ids = [row['race_id'] for row in opportunities]
    if len(ids) != len(set(ids)) or any(race_key(row['race_id']) != row['race_key'] for row in opportunities):
        raise DevelopmentRejected('OPPORTUNITY_IDENTITY_INVALID')
    if population is not None:
        if set(ids)!={r['race_id'] for r in population['intended']} or accounting.get('population_sha256')!=access['population']['sha256']:
            raise DevelopmentRejected('OPPORTUNITY_MEMBERSHIP_MISMATCH')
        if any(r['attempt_consumed'] and r['race_id'] not in population['selected_race_ids'] for r in opportunities):
            raise DevelopmentRejected('UNSELECTED_ATTEMPT')
    attempts = [row for row in opportunities if row['attempt_consumed']]
    if len(attempts) > allocation['max_capture_attempts']:
        raise DevelopmentRejected('ATTEMPT_BUDGET_EXCEEDED')
    if any(sum(r['race_key'][:10] == day for r in attempts) > allocation['max_attempts_per_date']
           for day in allocation['dates']):
        raise DevelopmentRejected('DAILY_ATTEMPT_BUDGET_EXCEEDED')
    selected = [row for row in opportunities if row['race_id'] == race_id]
    if (len(selected) != 1 or selected[0]['disposition'] != 'VERIFIED_FORECAST'
            or selected[0]['attempt_consumed'] is not True or selected[0]['qualified'] is not True):
        raise DevelopmentRejected('FORECAST_OPPORTUNITY_NOT_QUALIFIED')
    for row in opportunities:
        if not row.get('reason') or not isinstance(row.get('attempt_consumed'), bool) or not isinstance(row.get('qualified'), bool):
            raise DevelopmentRejected('OPPORTUNITY_DISPOSITION_MISSING')
    if (member.get('history_access') != 'machine_only_pre_target'
            or member.get('target_label_access') != 'separate_result_authority'):
        raise DevelopmentRejected('INPUT_ACCESS_NOT_AUTHORIZED')
    return access, allocation, member, accounting


def _histories(contents, race_id, runners, denied_intervals):
    """Exact production merge decisions and separate v2 card quality meanings."""
    from scripts import run_shadow_non_tgr_rf_evaluation as production
    from scripts import build_form_only_v1_packet as form
    from scripts.development_form_quality import build_record, CONTRACT
    from src.predictor.comparison_candidates import projected_dates
    forms = [name for name in contents if name.startswith('source/') and name.endswith('.csv')]
    if len(forms) != 1:
        raise DevelopmentRejected('FORM_IDENTITY_AMBIGUOUS')
    raw = contents[forms[0]]
    target_date = datetime.fromisoformat(race_id.rsplit(' - ', 1)[1]).date()
    for day in projected_dates(raw):
        if day >= target_date or any(start <= day.isoformat() <= end for start, end in denied_intervals):
            raise DevelopmentRejected('HISTORY_DATE_ACCESS_DENIED')
    metadata = json.loads(contents[forms[0] + '.metadata.json'])
    blocks = form.parse_form_blocks_bytes(raw, source=race_id)
    delimiter = '|' if raw.partition(b'\n')[0].count(b'|') > raw.partition(b'\n')[0].count(b',') else ','
    card_rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter=delimiter))
    card = production.live_form_history_by_dog(card_rows, target_race_date=target_date.isoformat())
    # Materialize only the already admitted retained DB, never its original path.
    with tempfile.TemporaryDirectory(prefix='development-retained-') as tmp:
        db = Path(tmp) / 'history.db'
        db.write_bytes(contents['features/sealed_history.db'])
        conn = sqlite3.connect(db.as_uri() + '?mode=ro&immutable=1', uri=True)
        conn.row_factory = sqlite3.Row
        try:
            days = [row[0] for row in conn.execute('SELECT DISTINCT race_date FROM race_metadata')]
            if any(not day or day >= target_date.isoformat() or any(start <= day <= end for start, end in denied_intervals) for day in days):
                raise DevelopmentRejected('HISTORY_DATE_ACCESS_DENIED')
            histories = production.load_db_history(conn)
        finally:
            conn.close()
    venue, _, grade, _ = form.target_metadata({'metadata': metadata}, race_id)
    expected = sorted((row['box_number'], form.dog_token(row['display_name'])) for row in runners)
    if form.parse_card_target_roster_bytes(raw, source=race_id) != expected:
        raise DevelopmentRejected('HISTORY_ROSTER_MISMATCH')
    output = []
    for runner in runners:
        token = form.dog_token(runner['display_name'])
        key = production.clean_name(runner['display_name'])
        db_rows, embedded = histories.get(key, []), card.get(key, [])
        merged = production.merge_prior_history_rows(db_rows, embedded)
        selected_ids = {id(row): n for n, row in enumerate(merged)}
        contributions = []
        for source, rows in (('retained_database', db_rows), ('retained_card', embedded)):
            for index, row in enumerate(rows):
                contributions.append({'source': source, 'source_row': index, 'row': row,
                                      'disposition': 'accepted' if id(row) in selected_ids else 'duplicate',
                                      'merged_row': selected_ids.get(id(row))})
        record = build_record(blocks[token], target_date=target_date, venue=venue,
                              distance=metadata.get('target_distance'), grade=grade)
        output.append({'box_number': runner['box_number'], 'runner_identity': runner['identity'],
                       'source_contributions': contributions, 'production_merged_history': merged,
                       'production_dedup_key': ['race_date','venue.upper','distance_num','normalized_grade','time_num','finish_num'],
                       'production_complete_career': 'unknown', 'card_quality': record})
    return {'card_feature_contract': CONTRACT, 'runners': output,
            'production_semantics': 'unchanged production merged features; card quality is separate, not a replacement',
            'source_hashes': {name: digest(contents[name]) for name in (forms[0], forms[0]+'.metadata.json', 'features/sealed_history.db')}}


def seal(access_path, access_sha256, race_id, output, *, clock=None):
    """Write a durable pre-result packet. A repeated identical seal is a read."""
    access, allocation, member, accounting = admit(access_path, access_sha256, race_id)
    synthetic = access['status'] == 'SYNTHETIC_FIXTURE'
    if clock is not None and not synthetic:
        raise DevelopmentRejected('REAL_CLOCK_REQUIRED')
    clock = clock or (lambda: datetime.now(timezone.utc))
    output = private_output(output)
    identity = {'schema_version': 'development_attempt_v1', 'access_sha256': access_sha256,
                'race_id': race_id, 'synthetic': synthetic, 'allocation_id': allocation['allocation_id']}
    put(output / 'attempt.json', identity)
    if (output / 'failure.json').exists():
        raise DevelopmentRejected('INCOMPLETE_ATTEMPT_PRESERVED')
    if (output / 'completion.json').exists():
        completion = json.loads((output / 'completion.json').read_bytes())
        checked({'path': str(output / 'pre_result.json'), 'sha256': completion['pre_result_sha256']})
        return completion
    if (output / 'pre_result.json').exists():
        raise DevelopmentRejected('INCOMPLETE_ATTEMPT_PRESERVED')
    try:
        from src.predictor.on_demand import verify_indexed_prediction_bundle
        bundle_root = Path(member['bundle_root'])
        preflight = read({'path':str(bundle_root/member['entry']['directory']/'bundle_manifest.json'),
                          'sha256':member['entry']['manifest_sha256']})
        if any(name.startswith('comparison/') for name in preflight['files']):
            raise DevelopmentRejected('FROZEN_COMPARISON_INPUT_FORBIDDEN')
        if synthetic:
            request=read({'path':str(bundle_root/member['entry']['directory']/'request.json'),
                          'sha256':preflight['files']['request.json']['sha256']})
            receipt=read({'path':str(bundle_root/member['entry']['directory']/'protocol/collector_exact_receipt.json'),
                          'sha256':preflight['files']['protocol/collector_exact_receipt.json']['sha256']})
            if (request['race_id']!=race_id
                    or not all(r['display_name'].startswith('Synthetic ') for r in request['runners'])
                    or '/fabricated' not in receipt['sealed_handoff']['race']['url']):
                raise DevelopmentRejected('SYNTHETIC_DATA_REQUIRED')
        verified = verify_indexed_prediction_bundle(bundle_root, member['entry'])
        forecast = verified.result
        if forecast['race']['race_id'] != race_id or forecast['status'] != 'PREDICTION_READY':
            raise DevelopmentRejected('VERIFIED_FORECAST_IDENTITY_MISMATCH')
        if stamp(forecast['race']['jump_timestamp']) != stamp(member['jump_at']):
            raise DevelopmentRejected('JUMP_IDENTITY_MISMATCH')
        contents = {}
        for name, ref in verified.manifest['files'].items():
            if name.startswith('comparison/'):
                raise DevelopmentRejected('FROZEN_COMPARISON_INPUT_FORBIDDEN')
            contents[name] = checked({'path': str(bundle_root / verified.directory / name), 'sha256': ref['sha256']})
        if 'retained_inputs.zip' not in contents:
            raise DevelopmentRejected('RETAINED_INPUTS_REQUIRED')
        verification = read(member['verification'])
        if (verification['status'] != 'PREDICTION_READY' or verification['race_id'] != race_id
                or verification['job_id'] != forecast['job_id']):
            raise DevelopmentRejected('VERIFICATION_IDENTITY_MISMATCH')
        verification_raw=checked(member['verification'])
        with os.fdopen(os.open(output/'verification.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb') as handle:
            handle.write(verification_raw);handle.flush();os.fsync(handle.fileno())
        receipt = json.loads(contents['odds_receipt.json'])
        quoted = {row['identity']: row for row in receipt['markets']['win']}
        predictions = {row['identity']: row for row in forecast['prediction']['predictions']}
        runners = [{**row, 'win_odds': quoted[row['identity']]['odds_decimal'],
                    'model_win_probability': predictions[row['identity']]['probability']}
                   for row in verified.request['runners']]
        if synthetic and (not all(row['display_name'].startswith('Synthetic ') for row in runners)
                          or '/fabricated' not in forecast['race']['url']):
            raise DevelopmentRejected('SYNTHETIC_DATA_REQUIRED')
        odds = [row['win_odds'] for row in runners]
        model = [row['model_win_probability'] for row in runners]
        if (len(runners) < 2 or len({r['identity'] for r in runners}) != len(runners)
                or any(isinstance(v, bool) or not math.isfinite(v) or v <= 1 for v in odds)
                or any(isinstance(v, bool) or not math.isfinite(v) or not 0 < v < 1 for v in model)
                or not math.isclose(math.fsum(model), 1, abs_tol=1e-9)):
            raise DevelopmentRejected('INVALID_PROBABILITY_OR_FIELD')
        total = math.fsum(1/v for v in odds)
        market = [(1/v)/total for v in odds]
        history = _histories(contents, race_id, runners, allocation.get('denied_history_intervals', []))
        capture = json.loads(contents['source/capture.json'])
        request = verified.request
        import zipfile
        with zipfile.ZipFile(io.BytesIO(contents['retained_inputs.zip'])) as archive:
            retained = json.loads(archive.read('bundle/completion.json'))
        generated = stamp(forecast['generated_at'])
        verified_at = stamp(verification['completed_at'])
        observed = stamp(capture['source_attempt']['fetch_time'])
        retained_at = stamp(retained['inputs_sealed_at'])
        now = clock()
        if not synthetic:
            population=read(access['population'])
            selected=next(r for r in population['intended'] if r['race_id']==race_id)
            expected={(r['box'],r['identity'],r.get('source_native_runner_id')) for r in selected['runners']}
            actual={(r['box_number'],r['identity'],r.get('source_native_runner_id')) for r in runners}
            capture_native=capture['source_plan_item']['race_identity'].get('source_native_race_id')
            if (expected!=actual or stamp(population['frozen_at'])>observed
                    or selected['url']!=forecast['race']['url']
                    or selected.get('source_native_race_id')!=capture_native):
                raise DevelopmentRejected('POPULATION_FIELD_OR_CAPTURE_TIMING_CHANGED')
        if not observed <= retained_at <= generated <= verified_at <= now < stamp(member['jump_at']):
            raise DevelopmentRejected('PRE_RESULT_TIMING_INVALID')
        # Keep a standalone verified source package; no later replay depends on
        # mutable originals. The pre-result completion binds all source hashes.
        for name, raw in {**contents, 'bundle_manifest.json': canonical(verified.manifest)}.items():
            path = output / 'evidence' / verified.directory / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        directories = [p for p in (output/'evidence').rglob('*') if p.is_dir()]
        for directory in sorted([*directories,output/'evidence',output],key=lambda p:len(p.parts),reverse=True):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        packet = {**identity, 'schema_version': 'development_pre_result_v1',
            'label': 'SYNTHETIC' if synthetic else 'PROSPECTIVE_DEVELOPMENT',
            'race': forecast['race'], 'runners': [{**row, 'normalized_market_probability': p} for row, p in zip(runners, market)],
            'market': {'type': 'fixed_WIN', 'acquisition_source': 'sportsbet_rendered_page',
                       'price_origin_bookmaker': 'Sportsbet', 'overround': total,
                       'active_field_evidence': request['runners'],
                       'native_market_id': None, 'provider_quote_time': None,
                       'qualification': 'existing independently verified whole-field WIN receipt; absent native IDs remain absent'},
            'times': {'observation_at': observed.isoformat(), 'observation_basis': 'conservative_fetch_start',
                      'availability_upper_bound': retained_at.isoformat(), 'availability_basis': 'retained_inputs_post_fsync_completion',
                      'prediction_at': generated.isoformat(), 'verification_at': verified_at.isoformat(),
                      'index_observed_at': verification.get('index_observed_at'),
                      'index_age_at_dispatch_seconds': verification.get('index_age_at_dispatch_seconds'),
                      'scheduled_jump_at': member['jump_at'], 'actual_off_at': None,
                      'assembly_started_at': now.isoformat()},
            'production_features': json.loads(contents['features/sealed/shadow_feature_rows.json']),
            'history': history, 'model': forecast['model'],
            'provenance': {'bundle_entry': member['entry'], 'retained_input_manifest_sha256': request['retained_input_manifest_sha256'],
                           'exact_receipt_sha256': digest(contents['protocol/collector_exact_receipt.json']),
                           'members': verified.manifest['files'], 'allocation': access['allocation'],
                           'opportunities': access['opportunities'], 'verification': member['verification']},
            'accounting': {'intended': len(accounting['opportunities']),
                           'qualified': sum(r['qualified'] for r in accounting['opportunities']),
                           'attempts': sum(r['attempt_consumed'] for r in accounting['opportunities']),
                           'successful_forecasts': sum(r['disposition']=='VERIFIED_FORECAST' for r in accounting['opportunities'])}}
        packet_hash = put(output / 'pre_result.json', packet)
        completed = clock()
        if not now <= completed < stamp(member['jump_at']):
            raise DevelopmentRejected('SEAL_CROSSED_JUMP')
        completion = {'schema_version': 'development_pre_result_completion_v1',
                      'status': 'SEALED_PRE_RESULT', 'synthetic': synthetic,
                      'pre_result_sha256': packet_hash, 'sealed_at': completed.isoformat(),
                      'race_id': race_id, 'access_sha256': access_sha256}
        put(output / 'completion.json', completion)
        if clock() >= stamp(member['jump_at']):
            raise DevelopmentRejected('COMPLETION_CROSSED_JUMP')
        return completion
    except Exception as exc:
        reason = str(exc) if isinstance(exc, DevelopmentRejected) else 'ASSEMBLY_FAILED'
        put(output / 'failure.json', {'status': 'FAILED_PRESERVED', 'reason': reason, **identity})
        raise


def join_result(access_path, access_sha256, race_id, output, authority_path, authority_sha256):
    """Read exactly one result after identity, allocation and prior-seal checks."""
    access, allocation, member, _ = admit(access_path, access_sha256, race_id)
    output = private_output(output)
    completion = json.loads((output / 'completion.json').read_bytes())
    if (output / 'failure.json').exists():
        raise DevelopmentRejected('INCOMPLETE_ATTEMPT_PRESERVED')
    packet = read({'path': str(output / 'pre_result.json'), 'sha256': completion['pre_result_sha256']})
    if (completion['status'] != 'SEALED_PRE_RESULT' or completion['access_sha256'] != access_sha256
            or completion['race_id'] != race_id or packet['race']['race_id'] != race_id):
        raise DevelopmentRejected('PRE_RESULT_SEAL_IDENTITY_MISMATCH')
    verify_package(access_path,access_sha256,race_id,output)
    authority = read({'path': str(Path(authority_path).absolute()), 'sha256': authority_sha256})
    synthetic = access['status'] == 'SYNTHETIC_FIXTURE'
    actual_now=datetime.now(timezone.utc)
    if not synthetic and actual_now < stamp(member['jump_at']):
        raise DevelopmentRejected('RESULT_ACCESS_BEFORE_JUMP')
    if (authority.get('schema_version') != 'development_result_authority_v1'
            or authority.get('status') != ('SYNTHETIC_FIXTURE' if synthetic else 'AUTHORIZED')
            or authority.get('allocation_id') != allocation['allocation_id']
            or not authority.get('authority_reference') or race_id not in authority['members']):
        raise DevelopmentRejected('RESULT_ACCESS_NOT_AUTHORIZED')
    permit = authority['members'][race_id]
    if permit['race_key'] != member['race_key'] or permit['pre_result_sha256'] != completion['pre_result_sha256']:
        raise DevelopmentRejected('RESULT_PERMIT_IDENTITY_MISMATCH')
    # This is the first result-bearing read. Mixed files/broad DB joins have no interface.
    result = read(permit['result'])
    expected_result_keys={'schema_version','synthetic','race_id','race_key','disposition','observed_at',
                          'official_source','source_evidence_sha256','official_evidence','finishers','reason'}
    if set(result)!=expected_result_keys or (result.get('official_evidence') is not None
                                           and set(result['official_evidence'])!={'race_rows','runner_rows'}):
        raise DevelopmentRejected('RESULT_SCHEMA_NOT_SINGLE_RACE')
    if not synthetic and stamp(result['observed_at'])>actual_now:
        raise DevelopmentRejected('RESULT_OBSERVATION_IN_FUTURE')
    if (result.get('schema_version') != 'development_official_result_v1'
            or result.get('race_id') != race_id or result.get('race_key') != member['race_key']
            or result.get('synthetic') is not synthetic):
        raise DevelopmentRejected('RESULT_IDENTITY_MISMATCH')
    if not stamp(completion['sealed_at']) < stamp(member['jump_at']) <= stamp(result['observed_at']):
        raise DevelopmentRejected('RESULT_TIMING_INVALID')
    disposition = result['disposition']
    if disposition not in {'OFFICIAL', 'MISSING', 'AMBIGUOUS', 'VOID'}:
        raise DevelopmentRejected('INVALID_RESULT_DISPOSITION')
    if disposition == 'OFFICIAL':
        if not result.get('official_source') or not result.get('source_evidence_sha256') or not result.get('official_evidence'):
            raise DevelopmentRejected('OFFICIAL_RESULT_PROVENANCE_MISSING')
        evidence = result['official_evidence']
        if digest(canonical(evidence)) != result['source_evidence_sha256']:
            raise DevelopmentRejected('OFFICIAL_RESULT_EVIDENCE_CHANGED')
        from types import SimpleNamespace
        from src.operator_ui.journal_results import OfficialResultSource
        job = SimpleNamespace(input=SimpleNamespace(race_id=race_id,jump_timestamp=member['jump_at'],
            ordered_runners=[{'box':r['box_number'],'name':r['display_name'],
                             'source_native_runner_id':r.get('source_native_runner_id')} for r in packet['runners']]))
        bundle = SimpleNamespace(result={'race':packet['race'],'generated_at':packet['times']['prediction_at']})
        OfficialResultSource._validate(job,bundle,evidence['race_rows'],evidence['runner_rows'],
                                       stamp(result['observed_at']) if synthetic else actual_now)
        field = {r['identity'] for r in packet['runners']}
        places = result['finishers']
        if (set(places) != field or any(type(p) is not int or p < 1 or p > len(field) for p in places.values())
                or len(set(places.values())) != len(field) or sum(p == 1 for p in places.values()) != 1):
            raise DevelopmentRejected('OFFICIAL_RESULT_FIELD_AMBIGUOUS')
        by_box = {r['box_number']:r['finish_position'] for r in evidence['runner_rows']}
        if any(places[r['identity']] != by_box[r['box_number']] for r in packet['runners']):
            raise DevelopmentRejected('OFFICIAL_RESULT_LABEL_MISMATCH')
    elif result.get('finishers') or not result.get('reason'):
        raise DevelopmentRejected('RESULT_DISPOSITION_REASON_REQUIRED')
    example = {'schema_version': 'development_example_v1', 'synthetic': synthetic,
               'race_id': race_id, 'pre_result_sha256': completion['pre_result_sha256'],
               'result_authority_sha256': authority_sha256, 'result_sha256': permit['result']['sha256'],
               'disposition': disposition, 'trainable': disposition == 'OFFICIAL',
               'pre_result': packet, 'result': result}
    put(output / 'example.json', example)
    return {'status': 'ASSEMBLED', 'synthetic': synthetic, 'trainable': example['trainable'],
            'race_id': race_id, 'example_sha256': digest(canonical(example)), 'disposition': disposition}


def verify_package(access_path, access_sha256, race_id, output):
    """Replay retained production features and history from exported bytes only."""
    _, allocation, member, _ = admit(access_path, access_sha256, race_id)
    output = private_output(output)
    completion = json.loads((output/'completion.json').read_bytes())
    if (output/'failure.json').exists() or completion['race_id'] != race_id or completion['access_sha256'] != access_sha256:
        raise DevelopmentRejected('PRE_RESULT_SEAL_IDENTITY_MISMATCH')
    packet = read({'path':str(output/'pre_result.json'),'sha256':completion['pre_result_sha256']})
    from src.predictor.on_demand import verify_indexed_prediction_bundle
    verified = verify_indexed_prediction_bundle(output/'evidence', member['entry'])
    contents = {name:checked({'path':str(output/'evidence'/verified.directory/name),'sha256':entry['sha256']})
                for name,entry in verified.manifest['files'].items()}
    receipt=json.loads(contents['odds_receipt.json'])
    quotes={r['identity']:r for r in receipt['markets']['win']}
    predictions={r['identity']:r for r in verified.result['prediction']['predictions']}
    runners=[{**r,'win_odds':quotes[r['identity']]['odds_decimal'],
              'model_win_probability':predictions[r['identity']]['probability']} for r in verified.request['runners']]
    overround=math.fsum(1/r['win_odds'] for r in runners)
    for runner in runners:runner['normalized_market_probability']=(1/runner['win_odds'])/overround
    verification=read({'path':str(output/'verification.json'),'sha256':member['verification']['sha256']})
    if (packet['race']!=verified.result['race'] or packet['model']!=verified.result['model']
            or packet['runners']!=runners or packet['market']['overround']!=overround
            or packet['production_features']!=json.loads(contents['features/sealed/shadow_feature_rows.json'])
            or stamp(packet['times']['prediction_at'])!=stamp(verified.result['generated_at'])
            or stamp(packet['times']['verification_at'])!=stamp(verification['completed_at'])
            or stamp(packet['times']['scheduled_jump_at'])!=stamp(member['jump_at'])
            or stamp(packet['times']['observation_at'])!=stamp(json.loads(contents['source/capture.json'])['source_attempt']['fetch_time'])):
        raise DevelopmentRejected('EXAMPLE_SOURCE_REPLAY_MISMATCH')
    history = _histories(contents,race_id,packet['runners'],allocation.get('denied_history_intervals',[]))
    if canonical(history) != canonical(packet['history']):
        raise DevelopmentRejected('HISTORY_REPLAY_MISMATCH')
    import zipfile
    from race_collection.retained_feature_replay import replay_retained_inputs
    with tempfile.TemporaryDirectory(prefix='development-replay-') as tmp:
        with zipfile.ZipFile(io.BytesIO(contents['retained_inputs.zip'])) as archive:
            if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
                raise DevelopmentRejected('ARCHIVE_PATH_INVALID')
            archive.extractall(tmp)
        replay = replay_retained_inputs(Path(tmp)/'bundle')
    return {'status':'REPLAY_VERIFIED','race_id':race_id,'synthetic':packet['synthetic'],
            'retained_features_sha256':replay['feature_values_sha256'],
            'pre_result_sha256':completion['pre_result_sha256']}
