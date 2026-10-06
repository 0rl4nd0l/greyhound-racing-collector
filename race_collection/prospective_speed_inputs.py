"""Outcome-free, frozen speed forecasts from an already sealed native capture.

This adapter neither acquires sources nor publishes a prospective seal. Its caller
owns development allocation, bounded reads, pre-jump publication and immutable
history inventory. The retained production forecast is never an input feature.
"""
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from race_collection import sectional_development_inputs as baseline
from race_collection import sectional_speed_candidate as candidate
from race_collection import sectional_speed_evaluation as probabilities
from race_collection import speed_candidate_inputs as retained
from scripts import build_form_only_v1_packet as canonical
from scripts import offline_form_packet as form

require = retained.require
FROZEN_CANDIDATE_COMMIT = 'af55ae9322d08d0f255f767a14e385c794205dfe'
BASELINE_ARTIFACT_SHA256 = 'df6bd595e1905bd67a1da9e5becab10f8b5f3db95707090079421201ccf6861e'
ADJUSTMENT_STRENGTH = 0.1
FROZEN_SOURCE_SHA256 = {
    'race_collection/speed_candidate_inputs.py': '769422522123bae9d5580919b08d280ab98e1480075765708b799d791942c7a5',
    'race_collection/retained_card_timing_coverage.py': 'b8dde80795a3db78d348a96517a6e37cad2b31c4e3b4c1e5ad4bb28f34828904',
    'race_collection/retained_speed_features.py': 'd89a45a89cc613532bd42cf3be136bc2d5da11ce9aab26ee7b3884371e558956',
    'utils/runner_completeness.py': 'f004a44f5d043e700b399943479252c82f4fb85473c03322b1b06bbb3b8a9134',
    'utils/thedogs_runner_identity.py': '49b96bc04624c9ba18f1351a0f2b390dfeaf46546c5d29430a1bc2e27b727d3a',
    'race_collection/sectional_speed_candidate.py': 'b328a44feb7196817ab5b9e6bbc8aa91d3d7c9ff2189ed1624c62cf6f079cad6',
    'race_collection/sectional_development_inputs.py': '839bd928a569880210685300e331c388265414eee777bf6792e078fca07371fb',
    'race_collection/sectional_speed_evaluation.py': 'a3ab471ac6d965c664b8eacbb0c0f82db75a67a1d47215594f49cc07d31ffa39',
    'scripts/build_form_only_v1_packet.py': '11b56970de0e53975a444d0ad066f7ea566ffc717757b074d5eb0d5f8865f086',
    'scripts/offline_form_packet.py': '865f18598429a231ca515ec8cbf108312d6c2faa23e392608b398b98bbe28379',
}


def frozen_contract():
    """Verify imported frozen implementations; this performs only local reads."""
    root = Path(__file__).resolve().parents[1]
    require(all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                for name, digest in FROZEN_SOURCE_SHA256.items()), 'FROZEN_SOURCE_CHANGED')
    return {'candidate_commit': FROZEN_CANDIDATE_COMMIT,
        'baseline_artifact_sha256': BASELINE_ARTIFACT_SHA256,
        'baseline_artifact_key': 'base16', 'baseline_identity': 'JUNE_PERIOD1_DEVELOPMENT_BASE16',
        'production_model_used_as_baseline': False, 'adjustment_strength': ADJUSTMENT_STRENGTH,
        'sources': dict(FROZEN_SOURCE_SHA256), 'feature_names': list(form.FEATURES),
        'parameters': {'minimum_other_runners': 5, 'maximum_runner_observations': 5,
            'mad_scale': 1.4826, 'shrinkage_prior_count': 3, 'zscore_limit': 3.0},
        'working_assumptions': {'1 SEC': 'INDIVIDUAL_DOG_FIRST_SECTIONAL', 'TIME': 'INDIVIDUAL_DOG_TOTAL_TIME',
            'cross_context_transfer': 'EXPLORATORY_NOT_PROVEN_PHYSICAL_EQUIVALENCE'},
        'online_rule': 'ALL_AUTHENTICATED_HISTORY_AVAILABLE_BEFORE_EACH_ACTUAL_FORECAST_CUTOFF',
        'missing_history': 'ZERO_DIRECT_ADJUSTMENT_WITH_WHOLE_FIELD_NORMALIZATION',
        'no_supported_runners': 'EXACT_BASELINE_COPY'}


def _original_native_verification(reader, reference, root, admission_reference,
                                  completion_reference, expected_plan_sha256):
    """Use a package-bound original verifier without importing it into this process."""
    from race_collection.live_freshness_contract import verify_source_package
    package = reader.json(reference)
    source = Path(package['source_root'])
    require(source.is_absolute() and source.resolve() == source
        and source == Path(reference['path']).parent / 'source', 'NATIVE_VERIFIER_SOURCE_ROOT')
    require(package['frozen_comparison']['sha256'] == expected_plan_sha256
        and Path(package['prediction_root']) / 'bundles' == root,
        'NATIVE_VERIFIER_PACKAGE_BINDING')
    reader.read(package['frozen_comparison'])
    identity = verify_source_package(source, package['source_identity_sha256'])
    require(identity['commit'] == package['commit'], 'NATIVE_VERIFIER_SOURCE_COMMIT')
    python = Path(package['python'])
    require(python.is_absolute()
        and hashlib.sha256(python.read_bytes()).hexdigest() == package['python_sha256'],
        'NATIVE_VERIFIER_PYTHON_CHANGED')
    # Validate every source file again in the child before importing the original
    # native verifier. The expected identity digest comes from the bound package.
    code = '''import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[1]); digest=sys.argv[2]
identity=json.loads((source/'SOURCE_IDENTITY.json').read_bytes())
raw=(json.dumps(identity,sort_keys=True,separators=(',',':'))+'\\n').encode()
assert hashlib.sha256(raw).hexdigest()==digest, 'source identity changed'
for name,expected in identity['files'].items():
 p=(source/name).resolve()
 assert p.is_relative_to(source) and hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'source file changed'
from src.predictor.future_comparison import verify_comparison
verified=verify_comparison(Path(sys.argv[3]),Path(sys.argv[4]),expected_plan_sha256=sys.argv[5])
print(json.dumps({'source_identity_sha256':digest,'source_commit':identity['commit'],
 'admission_sha256':hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
 'completion_sha256':hashlib.sha256(Path(sys.argv[6]).read_bytes()).hexdigest(),
 'verified':{k:verified[k] for k in ('schema_version','eligible_common_race','completion')}},sort_keys=True))
'''
    command = ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
        '--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev', '--chdir', str(source),
        '--unsetenv', 'PYTHONHOME', '--setenv', 'PYTHONPATH', str(source),
        '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'PYTHONNOUSERSITE', '1',
        '--setenv', 'OPENBLAS_NUM_THREADS', '1', '--setenv', 'OMP_NUM_THREADS', '1',
        str(python), '-B', '-c', code, str(source), package['source_identity_sha256'],
        str(root), admission_reference['path'], expected_plan_sha256, completion_reference['path']]
    child = subprocess.run(command, capture_output=True, timeout=60, check=False)
    require(child.returncode == 0 and len(child.stdout) <= 2 * 1024**2,
        'ORIGINAL_NATIVE_VERIFIER_FAILED')
    proof = json.loads(child.stdout)
    require(proof['source_identity_sha256'] == package['source_identity_sha256']
        and proof['source_commit'] == package['commit']
        and proof['admission_sha256'] == admission_reference['sha256']
        and proof['completion_sha256'] == completion_reference['sha256'],
        'ORIGINAL_NATIVE_VERIFIER_PROOF_CHANGED')
    return proof['verified']


def member_from_native(reader, bundle_root, admission_reference,
                       completion_reference, *, expected_plan_sha256, allowed_source_roots,
                       verifier_source_reference=None):
    """Independently verify a native comparison and project its source graph.

    Allocation and resource isolation belong to the caller. This read-only call
    runs the existing native verifier; no caller-supplied success assertion can
    replace forecast/source replay or actual publication receipt verification.
    """
    reader = retained.SnapshotReader(reader)
    admission = reader.json(admission_reference)
    completion = reader.json(completion_reference)
    require(admission['plan_sha256'] == expected_plan_sha256, 'NATIVE_PLAN_OUTSIDE_ALLOCATION')
    root = Path(bundle_root)
    require(root.is_absolute() and root.resolve() == root, 'NATIVE_BUNDLE_ROOT_INVALID')
    if verifier_source_reference is None:
        from src.predictor.future_comparison import verify_comparison
        verified = verify_comparison(root, Path(admission_reference['path']),
            expected_plan_sha256=expected_plan_sha256)
    else:
        verified = _original_native_verification(reader, verifier_source_reference,
            root, admission_reference, completion_reference, expected_plan_sha256)
    require(verified['schema_version'] == 'verified_four_way_comparison_v1'
        and verified['eligible_common_race'] is True
        and verified['completion'] == completion
        and completion['admission_sha256'] == admission_reference['sha256'],
        'NATIVE_INDEPENDENT_VERIFICATION_MISMATCH')
    require(Path(completion_reference['path']) == Path(admission_reference['path']).with_name('completion.json')
        and completion['status'] == 'COMPLETE_BEFORE_CUTOFF'
        and all(completion[k] == admission[k] for k in ('race', 'runner_set_sha256',
            'plan_sha256', 'prediction_id', 'retained_input_manifest_sha256')),
        'NATIVE_COMPLETION_BINDING')
    directory = completion['bundle_entry']['directory']
    require(isinstance(directory, str) and Path(directory).name == directory
        and directory not in ('', '.', '..') and admission['bundle_directory'] == directory,
        'NATIVE_BUNDLE_DIRECTORY_INVALID')
    base = root / directory
    manifest_ref = {'path': str(base / 'bundle_manifest.json'),
        'sha256': completion['bundle_entry']['manifest_sha256']}
    manifest_bytes = reader.read(manifest_ref)
    manifest_ref['bytes'] = len(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    require(manifest['prediction_id'] == admission['prediction_id'], 'NATIVE_BUNDLE_PREDICTION_MISMATCH')

    def bundled(relative):
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'NATIVE_BUNDLE_FILE_INVALID')
        ref = {'path': str(base / relative), 'sha256': manifest['files'][relative]['sha256']}
        payload = reader.read(ref)
        return {**ref, 'bytes': len(payload)}

    forms = [name for name in manifest['files'] if name.startswith('source/') and name.endswith('.csv')]
    require(len(forms) == 1, 'NATIVE_FORM_AMBIGUOUS')
    form_ref = bundled(forms[0])
    sidecar_ref = bundled(forms[0] + '.metadata.json')
    sidecar = reader.json(sidecar_ref)
    allowed = tuple(Path(path) for path in allowed_source_roots)
    require(bool(allowed) and all(path.is_absolute() and path.resolve() == path for path in allowed),
        'NATIVE_ALLOWED_SOURCE_ROOT_INVALID')

    def source(path, digest, expected_size=None):
        path = Path(path)
        require(path.is_absolute() and any(path.is_relative_to(root) for root in allowed),
            'NATIVE_SOURCE_OUTSIDE_ALLOCATION')
        ref = {'path': str(path), 'sha256': digest}
        if expected_size is not None:
            ref['bytes'] = expected_size
        payload = reader.read(ref)
        return {**ref, 'bytes': len(payload)}

    original_card = Path(sidecar['accepted_csv_path'])
    require(original_card.is_absolute() and any(original_card.is_relative_to(root) for root in allowed),
        'NATIVE_SOURCE_OUTSIDE_ALLOCATION')
    primary = sidecar['primary_race_page_evidence']
    for key in ('raw_path', 'receipt_path'):
        require(not Path(primary[key]).is_absolute() and '..' not in Path(primary[key]).parts,
            'NATIVE_PRIMARY_PATH_INVALID')
    race = admission['race']
    member = {'race_id': race['race_id'], 'source_race_date': race['race_date'],
        'venue': race['venue'], 'jump_at': race['jump_timestamp'],
        'target_distance_raw': str(sidecar['target_distance']),
        'target_runner_slots': sidecar['runner_completeness_after_canonical_alignment']['runner_count'],
        'runner_set_sha256': admission['runner_set_sha256'], 'csv_header': retained.strict.HEADER,
        'bundle_manifest': manifest_ref, 'admission': admission_reference,
        'accepted_csv': form_ref, 'sidecar': sidecar_ref,
        'raw_export': source(sidecar['raw_export_path'], sidecar['raw_content_sha256'], sidecar['raw_content_length']),
        'primary_page': source(original_card.parent / primary['raw_path'], primary['body_sha256']),
        'primary_receipt': source(original_card.parent / primary['receipt_path'], primary['receipt_sha256']),
        'comparison_inputs': bundled('comparison/inputs.json'),
        'odds_receipt': bundled('odds_receipt.json'), 'request': bundled('request.json')}
    original = {'race_id': member['race_id'], 'jump_at': member['jump_at'],
        'runner_set_sha256': member['runner_set_sha256'], 'admission': admission_reference,
        'bundle_manifest': manifest_ref, 'completion': completion_reference,
        'original_admitted_at': admission['admitted_at'],
        'original_published_complete_at': completion['published_complete_at']}
    retained.coverage.verified_member(reader, member, original)
    return member, original


def base16_features(payload, sidecar, packet):
    """Reproduce original raw-card feature semantics, never live feature aliases.

    Historical PLC and MGN cells are permitted pre-race history. Target-date and
    later rows are excluded by the unchanged original accepted_history parser.
    """
    target = packet['target']
    day = date.fromisoformat(target['date'])
    source_card = target['source_card']
    captured = retained.coverage.instant(source_card['available_at'])
    cutoff = retained.coverage.instant(target['cutoff'])
    require(captured < cutoff, 'BASELINE_SOURCE_NOT_BEFORE_CUTOFF')
    blocks = canonical.parse_form_blocks_bytes(payload, source='sealed pre-race baseline card')
    venue, _, grade, _ = canonical.target_metadata({'metadata': sidecar}, target['race_id'])
    distance = form._metres(sidecar.get('target_distance') or sidecar.get('race_info', {}).get('distance'))
    require(distance is not None and distance == target['distance_m'], 'BASELINE_TARGET_DISTANCE')
    roster = source_card['roster']
    rows, history_audit = [], []
    for runner in roster:
        token = runner['block_token']
        require(token in blocks, 'BASELINE_HISTORY_BLOCK_MISSING')
        history, rejected = canonical.accepted_history(blocks[token], day)
        require(all(h['date'] <= captured.astimezone(cutoff.tzinfo).date() for h in history),
            'BASELINE_HISTORY_AFTER_CAPTURE')
        raw = canonical.feature_row(target['race_id'], day, venue, distance, grade,
            len(roster), runner['box_number'], token, history)
        values = {name: form._number(raw.get(form.ALIASES.get(name, name))) for name in form.FEATURES}
        finishes = [h['finish'] for h in history[:5] if h['finish'] is not None]
        values['recent_finish_best_5'] = float(min(finishes)) if finishes else None
        require(all(v is None or math.isfinite(v) for v in values.values()), 'BASELINE_FEATURE_NONFINITE')
        rows.append({'race_id': target['race_id'], 'race_date': target['date'],
            'runner_id': runner['runner_id'], 'box': runner['box_number'],
            'dog_token': token, 'features': values})
        history_audit.append({'runner_id': runner['runner_id'], 'box_number': runner['box_number'],
            'block_token': token, 'source': source_card['accepted_csv'],
            'accepted_history': [{**h, 'date': h['date'].isoformat()} for h in history],
            'rejections': dict(Counter(reason for reason, _ in rejected))})
    return rows, {'target_venue': venue, 'target_distance_m': distance, 'target_grade': grade,
        'history_contract': 'CANONICAL_DATE_BEFORE_TARGET_CAP20_ORIGINAL_BASE16_MAPPING',
        'source_available_at': captured.isoformat(), 'runners': history_audit}


def _bound_json(reader, member, role, relative):
    reference = member[role]
    base = Path(member['bundle_manifest']['path']).parent
    require(Path(reference['path']) == base / relative, 'PROSPECTIVE_ROLE_PATH')
    manifest = reader.json(member['bundle_manifest'])
    require(manifest['files'][relative]['sha256'] == reference['sha256'], 'PROSPECTIVE_ROLE_BINDING')
    return reader.json(reference)


def _market(reader, member, packet):
    """Bind the original atomic prices to the complete native runner field."""
    inputs = _bound_json(reader, member, 'comparison_inputs', 'comparison/inputs.json')
    receipt = _bound_json(reader, member, 'odds_receipt', 'odds_receipt.json')
    request = _bound_json(reader, member, 'request', 'request.json')
    require(inputs['form_sha256'] == member['accepted_csv']['sha256']
        and inputs['sidecar_sha256'] == member['sidecar']['sha256']
        and inputs['odds_receipt_sha256'] == member['odds_receipt']['sha256'], 'MARKET_SOURCE_BINDING')
    require(receipt['schema_version'] == 'on_demand_odds_receipt_v1'
        and receipt['race_id'] == member['race_id']
        and receipt['source_hashes']['source_form_sha256'] == inputs['form_sha256']
        and receipt['source_hashes']['source_sidecar_sha256'] == inputs['sidecar_sha256']
        and receipt['source_hashes']['source_report_sha256'] == inputs['capture_sha256']
        and receipt['captured_at'] == inputs['captured_at'], 'MARKET_RECEIPT_BINDING')
    require(request['race_id'] == member['race_id']
        and request['jump_timestamp'] == member['jump_at']
        and request['runner_set_sha256'] == member['runner_set_sha256'], 'MARKET_REQUEST_BINDING')
    capture = retained.coverage.instant(receipt['captured_at'])
    cutoff = retained.coverage.instant(packet['target']['cutoff'])
    jump = retained.coverage.instant(member['jump_at'])
    require(capture < cutoff < jump and 120 <= (jump-capture).total_seconds() <= 600,
        'MARKET_SNAPSHOT_OUTSIDE_FROZEN_WINDOW')
    roster = packet['target']['source_card']['roster']
    expected = {(r['box_number'], r['block_token'], r['runner_id']) for r in roster}
    observed = [(r['box_number'], r['identity'], r['source_native_runner_id']) for r in inputs['runners']]
    requested = [(r['box_number'], r['identity'], r['source_native_runner_id']) for r in request['runners']]
    require(len(observed) == len(requested) == len(expected)
        and set(observed) == set(requested) == expected, 'MARKET_NATIVE_ROSTER_MISMATCH')
    wins = receipt['markets']['win']
    require(len(wins) == len(roster)
        and {(r['box_number'], r['identity']) for r in wins} == {(b, t) for b, t, _ in expected},
        'MARKET_RECEIPT_ROSTER_MISMATCH')
    win_prices = {(r['box_number'], r['identity']): r['odds_decimal'] for r in wins}
    price_map = {(r['box_number'], r['identity']): r['win_odds'] for r in inputs['runners']}
    require(price_map == win_prices, 'MARKET_PRICE_MISMATCH')
    odds = [price_map[(r['box_number'], r['block_token'])] for r in roster]
    return odds, {'captured_at': receipt['captured_at'], 'source_url': receipt['source_url'],
        'odds_receipt': member['odds_receipt'], 'comparison_inputs': member['comparison_inputs'],
        'production_identity': request['model'], 'production_request': member['request']}


def forecast(reader, member, original, model_reference, *, forecast_at, prior_observations=()):
    """Return all three forecasts with full private calculation evidence.

    `member` is a dynamically allocated native member, not an 82-card manifest.
    It has the existing seven retained roles plus comparison_inputs, odds_receipt
    and request. `original` must come from the independently verified pre-jump
    publication ledger. Prior observations must be authenticated by the caller's
    frozen history inventory; future copies are filtered by the frozen candidate.
    The caller must seal this return value before jump, independently of timing
    arguments, or retain it only as a failed/replay experiment attempt.
    """
    contract = frozen_contract()
    require(model_reference['sha256'] == BASELINE_ARTIFACT_SHA256, 'FROZEN_BASELINE_ARTIFACT_CHANGED')
    reader = retained.SnapshotReader(reader)
    artifact = reader.json(model_reference)
    model = artifact['base16']
    require(model['prep']['names'] == list(form.FEATURES), 'FROZEN_BASELINE_FEATURE_NAMES')
    packet, observations, audit = retained.construct_member(reader, member, original)
    decision = retained.coverage.instant(packet['target']['cutoff'])
    actual = retained.coverage.instant(forecast_at).astimezone(decision.tzinfo)
    available = retained.coverage.instant(original['original_published_complete_at'])
    require(available < actual < retained.coverage.instant(member['jump_at']), 'FORECAST_NOT_AFTER_SEAL_AND_PREJUMP')
    packet['target']['cutoff'] = min(actual, decision).isoformat()
    packet['target']['forecast_at'] = actual.isoformat()
    packet['observations'] = list(prior_observations) + observations
    odds, market_audit = _market(reader, member, packet)
    baseline_started = time.perf_counter()
    rows, baseline_audit = base16_features(reader.read(member['accepted_csv']), reader.json(member['sidecar']), packet)
    market = probabilities.normalized_market(odds)
    for row, price, probability in zip(rows, odds, market):
        row.update({'odds': price, 'market': probability})
    base = baseline.reproduce_base16(rows, model)
    baseline_seconds = time.perf_counter() - baseline_started
    speed_started = time.perf_counter()
    speed = candidate.build_sectional_candidate(packet)
    speed_seconds = time.perf_counter() - speed_started
    require([r['runner_id'] for r in speed['runners']] == [r['runner_id'] for r in rows],
        'SPEED_BASELINE_ROSTER_ORDER')
    adjusted = probabilities.adjusted_probabilities(base,
        [r['speed_estimate'] for r in speed['runners']],
        [r['status'] == 'SUPPORTED' for r in speed['runners']], ADJUSTMENT_STRENGTH)
    reader.check()
    return {'schema_version': 'prospective_frozen_speed_forecast_v1',
        'race_id': member['race_id'], 'race_date': member['source_race_date'],
        'jump_at': member['jump_at'], 'forecast_at': actual.isoformat(),
        'information_cutoff': packet['target']['cutoff'], 'runner_set_sha256': member['runner_set_sha256'],
        'contract': contract, 'baseline_artifact': model_reference, 'source_member': member,
        'original_publication': original, 'market': market_audit, 'baseline_history': baseline_audit,
        'baseline_rows': rows, 'speed_packet': packet, 'speed_features': speed,
        'predictions': [{'runner_id': row['runner_id'], 'box_number': row['box'],
            'market': m, 'baseline': b, 'baseline_plus_speed': s}
            for row, m, b, s in zip(rows, market, base, adjusted)],
        'input_audit': audit, 'new_observations': observations,
        'calculation_cost': {'baseline_feature_and_score_seconds': baseline_seconds,
            'speed_feature_and_benchmark_seconds': speed_seconds},
        'provider_requests': 0, 'result_requests': 0, 'result_payloads_opened': 0}
