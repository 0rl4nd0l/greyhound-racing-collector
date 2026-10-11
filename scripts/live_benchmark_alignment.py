#!/usr/bin/env python3
"""Authenticate allowlisted original v1 shadow records without model replay or I/O writes."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

FROZEN_FEATURES = ('prior_start_count', 'days_since_last_start', 'recent_finish_mean_3', 'recent_finish_best_5', 'recent_win_rate_5', 'recent_place_rate_5', 'recent_avg_margin_5', 'career_win_rate', 'career_place_rate', 'career_avg_finish', 'starts_same_venue', 'win_rate_same_venue', 'starts_same_distance', 'win_rate_same_distance', 'same_grade_start_count', 'same_grade_win_rate')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def reference(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('naive_timestamp')
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def validate_original(record):
    if record['schema_version'] != 'market_form_residual_shadow_record_v1':
        raise ValueError('unsupported_original_schema')
    if record['outcomes_present'] or record['activation']:
        raise ValueError('original_record_contract')
    rows = record['predictions']
    ids = sorted(r['runner_id'] for r in rows)
    if len(ids) < 2 or len(set(ids)) != len(ids) or len({r['box_number'] for r in rows}) != len(ids):
        raise ValueError('duplicate_or_missing_runner')
    if digest(('\n'.join(ids) + '\n').encode()) != record['runner_set_sha256']:
        raise ValueError('runner_set_hash_mismatch')
    identity = {k: record[k] for k in ('race_id', 'runner_set_sha256', 'model_sha256', 'manifest_sha256')}
    if digest(canonical(identity) + b'\n') != record['record_key']:
        raise ValueError('original_record_key_mismatch')
    if any(not isinstance(r.get('strict_win_odds'), (int, float)) or not math.isfinite(r['strict_win_odds']) or r['strict_win_odds'] <= 1 for r in rows):
        raise ValueError('invalid_odds')
    total = sum(1 / r['strict_win_odds'] for r in rows)
    for r in rows:
        if r['race_id'] != record['race_id']:
            raise ValueError('runner_race_mismatch')
        for key in ('full_probability', 'half_probability', 'market_probability'):
            if not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
                raise ValueError('invalid_probability')
        if not math.isfinite(r['strict_win_odds']) or r['strict_win_odds'] <= 1:
            raise ValueError('invalid_odds')
        if not math.isclose(r['market_probability'], (1 / r['strict_win_odds']) / total, abs_tol=1e-12):
            raise ValueError('original_market_normalization_mismatch')
        if not max(stamp(r['feature_freeze_timestamp']), stamp(r['odds_capture_timestamp'])) <= stamp(record['score_timestamp']) < stamp(record['jump_timestamp']):
            raise ValueError('invalid_original_chronology')
    for key in ('full_probability', 'half_probability', 'market_probability'):
        if not math.isclose(sum(r[key] for r in rows), 1, abs_tol=1e-12):
            raise ValueError('probability_sum')


def bind_append(record, directories):
    """Find the native append report by bounded native daemon time, then exact key."""
    time = stamp(record['score_timestamp'])
    candidates = [(d, t) for d, t in directories if time - timedelta(minutes=30) <= t <= time]
    errors = []
    for directory, _ in candidates:
        path = directory / 'early_residual_shadow_status.json'
        if not path.exists():
            continue
        value = read(path)  # Outcome-free native execution report, never a result source.
        for item in value.get('races', []):
            pred = item.get('prediction') or {}
            if pred.get('record_key') != record['record_key']:
                continue
            try:
                if item['status'] not in ('APPENDED', 'EXACT_REPLAY') or not pred.get('persisted'):
                    raise ValueError('append_not_durable')
                for key in ('race_id', 'score_timestamp', 'jump_timestamp', 'runner_set_sha256', 'model_sha256', 'manifest_sha256'):
                    if pred[key] != record[key]:
                        raise ValueError('append_original_mismatch:' + key)
                displayed = {r['box']: r for r in pred['predictions']}
                if len(displayed) != len(record['predictions']):
                    raise ValueError('append_field_mismatch')
                for row in record['predictions']:
                    other = displayed[row['box_number']]
                    for source_key, target_key in [('dog_name', 'dog'), ('strict_win_odds', 'win_odds'), ('full_probability', 'full_probability'), ('half_probability', 'half_probability'), ('market_probability', 'market_probability')]:
                        if row[source_key] != other[target_key]:
                            raise ValueError('append_original_mismatch:' + source_key)
                step = item['score_step']
                if step['returncode'] != 0 or not stamp(record['score_timestamp']) <= stamp(step['finished_at']) < stamp(record['jump_timestamp']):
                    raise ValueError('append_after_jump_or_failed')
                command = step['command']
                paths = {k: Path(command[command.index(k) + 1]) for k in ('--form-csv', '--sidecar', '--feature-rows', '--feature-manifest', '--implementation-manifest', '--capture')}
                hashes = pred['input_hashes']
                mapping = {'--form-csv': 'form_csv_sha256', '--sidecar': 'sidecar_sha256', '--feature-rows': 'feature_rows_sha256', '--feature-manifest': 'feature_manifest_sha256', '--implementation-manifest': 'implementation_manifest_sha256', '--capture': 'capture_artifact_sha256'}
                refs = {'append_report': reference(path)}
                for flag, hash_key in mapping.items():
                    refs[flag[2:]] = reference(paths[flag])
                    if refs[flag[2:]]['sha256'] != hashes[hash_key]:
                        raise ValueError('retained_input_hash_changed:' + flag)
                stdout = Path(step['stdout_path'])
                if read(stdout) != pred:
                    raise ValueError('stdout_append_report_mismatch')
                refs['original_stdout'] = reference(stdout)
                return {'sealed_at': step['finished_at'], 'artifacts': refs, 'native_prediction': pred, 'source_paths': {k: str(v) for k, v in paths.items()}}, errors
            except (ValueError, KeyError, OSError) as exc:
                errors.append(str(exc))
    return None, errors or ['native_append_report_not_found']


def bind_win_field(record, proof):
    attempts = []
    for line in Path(proof['source_paths']['--capture']).read_bytes().splitlines():
        match = re.search(rb'"race_id"\s*:\s*"([^"]+)"', line)
        if match and match[1].decode() == record['race_id']:
            candidate = json.loads(line)
            if stamp(candidate['fetch_time']) == stamp(proof['native_prediction']['odds_capture_timestamp']):
                attempts.append(candidate)
    if len(attempts) != 1:
        raise ValueError('capture_attempt_missing_or_ambiguous')
    attempt = attempts[0]
    validation = attempt['validation']
    rows = validation['accepted_rows']
    win = validation['market_validations']['win']
    if validation['status'] != 'PASS' or win['market_type'] != 'win' or win['accepted_rows'] != rows:
        raise ValueError('not_complete_win_market')
    if len(rows) != validation['active_expected_runner_count'] or len(rows) != len(record['predictions']):
        raise ValueError('capture_active_field_mismatch')
    if win['missing_expected_runners'] or win['extra_unexpected_runners'] or win['duplicate_runner_keys']:
        raise ValueError('capture_invalid_runner_set')
    by_box = {r['box_number']: r for r in rows}
    if len(by_box) != len(rows):
        raise ValueError('capture_duplicate_box')
    for runner in record['predictions']:
        quote = by_box[runner['box_number']]
        if runner['runner_id'] != f"{record['race_id']}|box:{runner['box_number']}|dog:{quote['identity']}":
            raise ValueError('native_capture_identity_mismatch')
        if quote['odds_decimal'] != runner['strict_win_odds']:
            raise ValueError('original_quote_mismatch')
    if not stamp(attempt['fetch_time']) <= stamp(attempt['append_time']) <= stamp(record['score_timestamp']):
        raise ValueError('quote_availability_after_prediction')
    return {'status': 'EXACT_COMPLETE_WIN_FIELD', 'capture_at': attempt['fetch_time'], 'available_at': attempt['append_time'], 'provider_quote_publication_time': None, 'pre_race_scratched_runners': validation['scratched_expected_runners'], 'source_url': validation['source_url']}


def pre_race_quality(record, proof):
    features = read(proof['source_paths']['--feature-rows'])
    by_box = {r['box_number']: r for r in features if r['race_id'] == record['race_id']}
    original = {r['box_number']: r for r in record['predictions']}
    if set(by_box) != set(original):
        return {'status': 'UNAVAILABLE_FEATURE_FIELD_MISMATCH'}
    for box, row in by_box.items():
        if re.sub(r'[^A-Z0-9]', '', row['dog_name'].upper()) != original[box]['runner_id'].split('|dog:')[1]:
            return {'status': 'UNAVAILABLE_FEATURE_IDENTITY_MISMATCH'}
    result = {'status': 'SEALED_PRE_RACE_FEATURES', 'source': proof['artifacts']['feature-rows'], 'history_cutoff': 'race_date_less_than_target_race_date'}
    for key in ('prior_start_count', 'starts_same_distance', 'starts_same_venue', 'same_grade_start_count'):
        values = [row.get(key) for row in by_box.values()]
        result['minimum_' + key] = min(values) if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None
    result['frozen_feature_nonmissing_fraction'] = sum(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for row in by_box.values() for k in FROZEN_FEATURES) / (len(by_box) * len(FROZEN_FEATURES))
    return result


def join_result(record, projection):
    if not projection:
        return None, 'OFFICIAL_RESULT_NOT_RETAINED_IN_AUTHORIZED_PROJECTION'
    races = projection['races']
    if not races or any(r['status'] != 'resulted' or r['source'] != 'thedogs_official' for r in races):
        return None, 'AMBIGUOUS_OFFICIAL_RESULT'
    if len({r['source_url'] for r in races}) != 1:
        return None, 'AMBIGUOUS_OFFICIAL_RESULT'
    race_bodies = [json.loads(r['row_json']) for r in races]
    native = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d\d-\d\d)', record['race_id'])
    if not native or any(r.get('race_date') != native[3] or r.get('race_number') != int(native[1]) for r in race_bodies):
        return None, 'RESULT_NATIVE_DATE_OR_NUMBER_MISMATCH'
    longest = max(race_bodies, key=lambda r: len(r['box_order']))
    if any(r['box_order'] != longest['box_order'][:len(r['box_order'])] or (r['winner_box'], r['winner_name']) != (longest['winner_box'], longest['winner_name']) for r in race_bodies):
        return None, 'CONFLICTING_OFFICIAL_RESULTS'
    references = []
    for source_rows, filename in ((races, 'official_result_races.jsonl'), (projection['runners'], 'official_result_runners.jsonl')):
        for row in source_rows:
            body = json.loads(row['row_json'])
            if any(row[k] != body[k] for k in set(row) & set(body)):
                return None, 'RESULT_DATABASE_COLUMNS_DIFFER_FROM_RETAINED_ROW'
            path = Path(row['source_artifact_dir']) / filename
            matches = []
            for line in path.read_bytes().splitlines():
                m = re.search(rb'"race_id"\s*:\s*"([^"]+)"', line)
                if m and m[1].decode() == record['race_id']:
                    matches.append(json.loads(line))
            if json.loads(row['row_json']) not in matches:
                return None, 'RETAINED_RESULT_PROJECTION_MISMATCH'
            ref = reference(path)
            if ref not in references:
                references.append(ref)
    rows = projection['runners']
    by_box = {r['box_number']: r for r in rows}
    original = {r['box_number']: r for r in record['predictions']}
    if len(by_box) != len(rows) or not set(by_box).issubset(original):
        return None, 'CHANGED_OR_DUPLICATE_RESULT_FIELD'
    if len(longest['box_order']) != len(rows) or any(by_box.get(box, {}).get('finish_position') != rank for rank, box in enumerate(longest['box_order'], 1)):
        return None, 'RESULT_FINISH_ORDER_MISMATCH'
    for box, row in by_box.items():
        token = re.sub(r'[^A-Z0-9]', '', row['dog_name'].upper())
        if original[box]['runner_id'] != f"{record['race_id']}|box:{box}|dog:{token}":
            return None, 'RESULT_RUNNER_IDENTITY_MISMATCH'
        if row['source'] != 'thedogs_official' or row['source_url'] != races[0]['source_url']:
            return None, 'RESULT_SOURCE_MISMATCH'
    winners = [r for r in rows if r['is_winner'] == 1]
    if len(winners) != 1 or winners[0]['finish_position'] != 1:
        return None, 'AMBIGUOUS_OR_DEAD_HEAT_RESULT'
    exact = set(original) == set(by_box)
    completed = []
    for race in races:
        path = Path(race['source_artifact_dir']) / 'autonomous_official_result_capture_attempts.progress.jsonl'
        if path.exists():
            for line in path.read_bytes().splitlines():
                m = re.search(rb'"race_id"\s*:\s*"([^"]+)"', line)
                if m and m[1].decode() == record['race_id']:
                    attempt = json.loads(line)
                    if attempt['status'] == 'INGESTED_DRY_RUN':
                        completed.append(attempt['completed_at'])
            references.append(reference(path))
    correction = 'COMPATIBLE_PREFIX_RESULT_SNAPSHOTS_COALESCED' if len(races) > 1 else None
    return {'field_status': 'EXACT_UNCHANGED' if exact else 'RESULT_FIELD_PARTIAL', 'result_at': max(completed) if completed else max(r['captured_at'] for r in races), 'result_timestamp_semantics': 'native_result_attempt_completed_at' if completed else 'native_batch_generated_at', 'winner_box': winners[0]['box_number'], 'result_projection': projection, 'result_artifacts': references, 'derived_correction': correction}, None


def benchmark_row(item, joined, variant):
    record, proof = item['original'], item['append_proof']
    output = {'race_id': record['race_id'], 'date': record['race_id'][-10:], 'model_version': record['model_sha256'] + ':' + variant, 'model_sha256': record['model_sha256'], 'prediction_id': record['record_key'], 'forecast_type': 'ORIGINAL_SEALED_LIVE', 'historical_role': 'shadow', 'analysis_class': 'STRICT_DECISION_TIME' if joined['field_status'] == 'EXACT_UNCHANGED' else 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC', 'allocation_status': 'AUTHORISED_NONRESERVED', 'field_status': joined['field_status'], 'verified': True, 'quote_at': item['win_field']['capture_at'], 'cutoff_at': record['score_timestamp'], 'predicted_at': record['score_timestamp'], 'sealed_at': proof['sealed_at'], 'jump_at': record['jump_timestamp'], 'result_at': joined['result_at'], 'provider_quote_publication_time': None, 'membership': item['membership'], 'artifacts': proof['artifacts'], 'result_artifacts': joined['result_artifacts'], 'derived_correction': joined['derived_correction'], 'result_evidence': joined['result_projection'], 'runners': [{'runner_id': r['runner_id'], 'box': r['box_number'], 'dog_name': r['dog_name'], 'probability': r[variant + '_probability'], 'decimal_odds': r['strict_win_odds'], 'winner': int(r['box_number'] == joined['winner_box'])} for r in record['predictions']]}

    output['pre_race_quality'] = item.get('pre_race_quality')
    output['feature_freeze_at'] = proof['native_prediction']['feature_freeze_timestamp']
    output['result_timestamp_semantics'] = joined['result_timestamp_semantics']
    return output


def qualify_manual(census_path, output, result_projection_path):
    """Preserve original manual forecasts without inventing a completion witness."""
    census = read(census_path)
    result_projection = read(result_projection_path)
    if digest(Path(result_projection['membership_path']).read_bytes()) != result_projection['membership_sha256']:
        raise ValueError('manual_result_membership_changed')
    forecasts, dispositions = [], []
    for member in census['requests']:
        if member['access_disposition'] != 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL' or member.get('status') != 'PREDICTION_READY':
            continue
        try:
            root = Path(member['request']['path']).parent
            if member['race_id'] not in result_projection['missing_race_ids']:
                raise ValueError('manual_result_disposition_requires_new_join')
            for label in ('request', 'manifest', 'prediction_result'):
                if reference(member[label]['path']) != member[label]:
                    raise ValueError('manual_census_reference_changed')
            manifest = read(root / 'bundle_manifest.json')
            for filename, expected in manifest['files'].items():
                path = root / filename
                if path.resolve().parent != root.resolve() and not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('manual_bundle_path_escape')
                raw = path.read_bytes()  # Opaque hashes only for history database.
                if len(raw) != expected['bytes'] or digest(raw) != expected['sha256']:
                    raise ValueError('manual_bundle_hash_mismatch')
            result, receipt = read(root / 'result.json'), read(root / 'odds_receipt.json')
            capture = read(root / 'source/capture.json')['source_attempt']
            validation = capture['validation']
            original = result['prediction']['predictions']
            quotes = {r['box_number']: r for r in receipt['markets']['win']}
            accepted = {r['box_number']: r for r in validation['accepted_rows']}
            if result['race']['race_id'] != member['race_id'] or receipt['race_id'] != member['race_id']:
                raise ValueError('manual_native_race_mismatch')
            if validation['status'] != 'PASS' or len(original) != validation['active_expected_runner_count'] or len(original) != len(quotes) or set(quotes) != set(accepted):
                raise ValueError('manual_incomplete_win_field')
            if not stamp(receipt['captured_at']) <= stamp(result['score_timestamp']) < stamp(result['race']['jump_timestamp']):
                raise ValueError('manual_prediction_not_prejump')
            rows = []
            if any(not isinstance(row.get('probability'), (float, int)) or not math.isfinite(row['probability']) or not 0 <= row['probability'] <= 1 or not isinstance(row.get('win_odds'), (float, int)) or not math.isfinite(row['win_odds']) or row['win_odds'] <= 1 for row in original):
                raise ValueError('manual_invalid_probability_or_odds')
            implied_total = sum(1 / row['win_odds'] for row in original)
            for row in original:
                quote = quotes[row['box_number']]
                token = re.sub(r'[^A-Z0-9]', '', row['dog_name'].upper())
                if quote['identity'] != token or accepted[row['box_number']]['identity'] != token or row['win_odds'] != quote['odds_decimal'] or quote['odds_decimal'] != accepted[row['box_number']]['odds_decimal']:
                    raise ValueError('manual_win_identity_or_price_mismatch')
                if not math.isclose(row['market_probability'], (1 / row['win_odds']) / implied_total, abs_tol=1e-12):
                    raise ValueError('manual_original_market_normalization_mismatch')
                rows.append({'runner_id': f"{member['race_id']}|box:{row['box_number']}|dog:{token}", 'box': row['box_number'], 'dog_name': row['dog_name'], 'probability': row['probability'], 'decimal_odds': row['win_odds']})
            if not math.isclose(sum(r['probability'] for r in rows), 1, abs_tol=1e-12):
                raise ValueError('manual_probability_sum')
            forecasts.append({'race_id': member['race_id'], 'date': member['race_id'][-10:], 'prediction_id': root.name, 'model_version': result['model']['model_sha256'] + ':full', 'historical_role': 'manual_research', 'forecast_type': 'ORIGINAL_RETAINED_MANUAL', 'quote_at': receipt['captured_at'], 'predicted_at': result['score_timestamp'], 'sealed_at': None, 'jump_at': result['race']['jump_timestamp'], 'verified': False, 'bundle_bytes_verified': True, 'win_field_verified': True, 'pre_race_quality': {'status': 'NOT_PROJECTED_COMPLETION_UNVERIFIED'}, 'runners': rows, 'artifacts': {k: member[k] for k in ('request', 'manifest', 'prediction_result')}, 'qualification_dispositions': ['MISSING_DURABLE_PREJUMP_COMPLETION_WITNESS', 'OFFICIAL_RESULT_NOT_RETAINED_IN_AUTHORIZED_PROJECTION']})
            dispositions.append({'race_id': member['race_id'], 'status': 'ORIGINAL_MANUAL_FIELD_AND_BYTES_AUTHENTICATED', 'strict_exclusions': forecasts[-1]['qualification_dispositions']})
        except (KeyError, ValueError, OSError) as exc:
            dispositions.append({'race_id': member['race_id'], 'status': 'INVALID_MANUAL_EVIDENCE', 'reason': str(exc)})
    for name, data in [('manual-forecast-records', forecasts), ('manual-qualification-dispositions', dispositions)]:
        (output / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--membership', required=True, type=Path)
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--manual-census', type=Path)
    parser.add_argument('--manual-results', type=Path)
    args = parser.parse_args()
    membership = read(args.membership)
    source = Path(membership['source'])
    raw = source.read_bytes()
    if digest(raw) != membership['source_sha256']:
        raise ValueError('membership_source_changed')
    lines = raw.splitlines()
    directories = []
    for directory in args.evidence_root.iterdir():
        match = re.fullmatch(r'shadow_autopilot_daemonization_v1_(202607(?:17|18|19|20|21|22)T\d{6}\+1000)(?:_odds_capture)?', directory.name)
        if match:
            directories.append((directory, datetime.strptime(match[1], '%Y%m%dT%H%M%S%z')))
    admitted, excluded = [], []
    for member in membership['records']:
        if member['access_disposition'] != 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL':
            excluded.append({**member, 'reason': 'SCIENTIFIC_ALLOCATION', 'category': 'scientific_allocation'})
            continue
        line = lines[member['line'] - 1]
        if digest(line) != member['line_sha256']:
            raise ValueError('line_identity_changed')
        record = json.loads(line)
        if record['race_id'] != member['race_id'] or record['record_key'] != member['record_key']:
            raise ValueError('membership_record_identity_changed')
        try:
            validate_original(record)
        except (ValueError, KeyError) as exc:
            excluded.append({**member, 'reason': str(exc), 'category': 'invalid_evidence'})
            continue
        proof, errors = bind_append(record, directories)
        item = {'membership': member, 'original': record, 'append_proof': proof, 'append_errors': errors}
        if proof:
            try:
                item['win_field'] = bind_win_field(record, proof)
                item['pre_race_quality'] = pre_race_quality(record, proof)
            except (ValueError, KeyError) as exc:
                item['field_error'] = str(exc)
        admitted.append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'authenticated-originals.json').write_text(json.dumps(admitted, indent=2) + '\n')
    (args.output / 'original-exclusions.json').write_text(json.dumps(excluded, indent=2) + '\n')
    summary = {'membership': reference(args.membership), 'source': reference(source), 'decoded_authorized': len(admitted), 'excluded': len(excluded), 'append_verified': sum(bool(r['append_proof']) for r in admitted), 'append_errors': dict(Counter(e for r in admitted for e in r['append_errors']))}
    (args.output / 'alignment-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.results:
        results = read(args.results)
        if results['membership_sha256'] != digest(args.membership.read_bytes()):
            raise ValueError('result_membership_changed')
        result_index = {r['race_id']: r for r in results['records']}
        primary = {}
        for item in admitted:
            r = item['original']
            prior = primary.get(r['race_id'])
            key = (stamp(r['score_timestamp']), stamp(item['append_proof']['sealed_at']) if item['append_proof'] else stamp(r['score_timestamp']), r['record_key'])
            prior_key = (stamp(prior['original']['score_timestamp']), stamp(prior['append_proof']['sealed_at']) if prior['append_proof'] else stamp(prior['original']['score_timestamp']), prior['original']['record_key']) if prior else None
            if prior is None or key > prior_key:
                primary[r['race_id']] = item
        forecasts = []
        for item in primary.values():
            if not item['append_proof'] or item.get('field_error') or not item.get('win_field'):
                continue
            r = item['original']
            forecasts.append({'race_id': r['race_id'], 'date': r['race_id'][-10:], 'prediction_id': r['record_key'], 'pre_race_quality': item.get('pre_race_quality'), 'sealed_at': item['append_proof']['sealed_at'] if item['append_proof'] else None, 'predicted_at': r['score_timestamp'], 'quote_at': item['win_field']['capture_at'], 'jump_at': r['jump_timestamp'], 'model_version': r['model_sha256'] + ':full', 'runners': [{'runner_id': row['runner_id'], 'box': row['box_number'], 'probability': row['full_probability'], 'decimal_odds': row['strict_win_odds']} for row in r['predictions']]})
        (args.output / 'forecast-records.json').write_text(json.dumps(forecasts, indent=2) + '\n')
        strict, diagnostic, dispositions = [], [], []
        for item in admitted:
            r = item['original']
            reason = 'SECONDARY_EARLIER_HORIZON' if primary[r['race_id']] is not item else None
            if not reason and (not item['append_proof'] or item.get('field_error')):
                reason = item.get('field_error') or 'APPEND_NOT_VERIFIED'
            joined, join_error = join_result(r, result_index.get(r['race_id'])) if not reason else (None, None)
            reason = reason or join_error
            if not reason:
                target = strict if joined['field_status'] == 'EXACT_UNCHANGED' else diagnostic
                target.extend(benchmark_row(item, joined, v) for v in ('full', 'half'))
                reason = 'INCLUDED' if target is strict else 'PARTIAL_RESULT_FIELD_DIAGNOSTIC_ONLY'
            dispositions.append({**item['membership'], 'disposition': reason})
        for name, value in [('strict-records', strict), ('diagnostic-records', diagnostic), ('join-dispositions', dispositions)]:
            (args.output / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
        summary['join_dispositions'] = dict(Counter(r['disposition'] for r in dispositions))
        summary['official_results_projection'] = reference(args.results)
        (args.output / 'alignment-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.manual_census:
        if not args.manual_results:
            raise ValueError('manual_result_projection_required')
        qualify_manual(args.manual_census, args.output, args.manual_results)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
