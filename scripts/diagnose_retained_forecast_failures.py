#!/usr/bin/env python3
"""Outcome-free reconciliation and offline July feature-validator grade repair.

No model is loaded, no forecast generated, no provider queried, no source mutated.
Only coverage-manifest nonreserved attempts are selected. Original raw feature
bytes and the complete native validator remain authoritative; the adapter changes
only the derived comparison spelling after finite-vocabulary equality is proven.
"""
from __future__ import annotations

import argparse
import ast
import collections
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_grade_aliases(path):
    """Read the already reviewed finite vocabulary, without importing runtime code."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GRADE_ALIASES' for t in node.targets):
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != 'MappingProxyType':
                raise ValueError('grade_vocabulary_contract_changed')
            result = ast.literal_eval(node.value.args[0])
            if not isinstance(result, dict) or 'M' in result:
                raise ValueError('ambiguous_grade_alias')
            return result
    raise ValueError('grade_aliases_missing')


def corrected_feature_packet(native, kwargs, aliases):
    """Apply a spelling-only derived correction; all original checks run afterward."""
    context = dict(kwargs['context'])
    rows = json.loads(kwargs['feature_rows_raw'])
    selected = [r for r in rows if isinstance(r, dict) and r.get('race_id') == kwargs['race_id']]
    source_grade = context['target_grade']
    source_canonical = aliases.get(str(source_grade).strip().upper())
    if not selected or source_canonical is None or any(
        aliases.get(str(r.get('target_grade_safe') or '').strip().upper()) != source_canonical
        for r in selected
    ):
        raise native.ManualPredictionError('feature_row_target_grade_mismatch')
    context['target_grade'] = selected[0]['target_grade_safe']
    return native._feature_packet(**{**kwargs, 'context': context})


def validate_independently(items, validator, input_error):
    """Known evidence failures stay local; programming errors remain visible."""
    rows = []
    for item in items:
        try:
            rows.append({'status': 'VALIDATED', 'result': validator(item)})
        except input_error as exc:
            rows.append({'status': 'BLOCKED', 'reason': str(exc)})
    return rows


def load_native(path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('retained_july_forecast_native', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def run(args):
    # Membership/access files are decoded before any retained race artifacts.
    boundary = json.loads(args.access_boundary.read_bytes())
    membership = json.loads(args.membership.read_bytes())
    coverage = json.loads(args.coverage.read_bytes())
    protected = set(boundary['canonical_protected_identity_keys'])
    refs = {}
    expected_refs = {r['path']: r['sha256'] for r in coverage['sources']}

    def read(path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        digest = sha(raw)
        if expected is not None and expected != digest:
            raise ValueError('retained_source_hash_mismatch:' + str(path))
        refs[str(path)] = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
        return raw

    for path in (args.access_boundary, args.membership, args.coverage, args.grade_vocabulary):
        read(path)
    allow = {r['canonical_race_key']: r for r in coverage['races'] if not r['reserved']}
    for key in allow:
        day = key.split('|')[0]
        if any(w['start'] <= day and (w['end'] is None or day <= w['end']) for w in boundary['protected_windows']):
            raise ValueError('coverage_allowlist_intersects_reserved_window')
    if set(allow) & protected:
        raise ValueError('coverage_allowlist_intersects_reservation')
    member_keys = {r['canonical_race_key'] for r in membership['records'] if r['access_disposition'] == 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL'}
    successful = {k for k,r in allow.items() if r['any_append']}
    if successful != member_keys:
        raise ValueError('coverage_membership_forecast_population_mismatch')
    attempts = [a for a in coverage['attempts'] if a['canonical_race_key'] in allow]
    if any(a['reserved'] for a in attempts):
        raise ValueError('reserved_attempt_selected')
    aliases = load_grade_aliases(args.grade_vocabulary)
    grouped = collections.defaultdict(list)
    stage_cache = {}
    native_cache = {}
    audit_attempts = []
    corrections = []
    for number, a in enumerate(attempts):
        key = a['canonical_race_key']
        path = a['path']
        if path not in stage_cache:
            stage_cache[path] = json.loads(read(path, expected_refs[path]))
            if stage_cache[path].get('outcomes_read') is not False:
                raise ValueError('stage_not_outcome_free')
        matches = [r for r in stage_cache[path]['races'] if r['race_id'] == a['race_id']]
        if len(matches) != 1:
            raise ValueError('ambiguous_native_attempt')
        original = matches[0]
        if original['status'] != a['status']:
            raise ValueError('attempt_status_changed')
        cmd = original['score_step']['command']
        native_path = Path(cmd[1])
        if native_path not in native_cache:
            read(native_path)
            native_cache[native_path] = load_native(native_path)
        native = native_cache[native_path]
        reason = None
        failure_ref = a.get('failure_reference')
        if a['status'] == 'BLOCKED':
            if not failure_ref:
                raise ValueError('failure_reference_missing')
            reason = json.loads(read(failure_ref['path'], failure_ref['sha256']))['reason']
            if reason != a['failure_reason']:
                raise ValueError('failure_reason_changed')
        entry = dict(a, attempt_index=number, native_scorer=str(native_path), original_failure_reason=reason)
        # Successful races retain every failure/retry, but require no feature decoding.
        if key not in successful:
            p = lambda opt: Path(cmd[cmd.index(opt) + 1])
            sidecar = json.loads(read(p('--sidecar')))
            features_raw = read(p('--feature-rows'))
            features = json.loads(features_raw)
            selected = [r for r in features if r.get('race_id') == a['race_id']]
            if len(selected) != len(features) or not selected:
                raise ValueError('feature_packet_unexpected_membership')
            shadow = sidecar['prejump_shadow_metadata']
            entry['source_grade'] = shadow.get('grade')
            entry['feature_grades'] = sorted({str(r.get('target_grade_safe')) for r in selected})
            entry['source_venue'] = shadow.get('venue')
            entry['feature_venues'] = sorted({str(r.get('venue')) for r in selected})
            entry['metadata_rejections'] = sorted({str(v) for r in selected for field in ('target_metadata_rejected_sources','weather_track_metadata_rejected_sources') for v in (r.get(field) or [])})
            entry['feature_manifest'] = str(p('--feature-manifest'))
            if reason == 'feature_row_target_grade_mismatch':
                context = native._sidecar_context(sidecar)
                kwargs = {'context': context, 'race_id': a['race_id'], 'form_csv_path': p('--form-csv')}
                for flag, name in (('--feature-rows','feature_rows'),('--feature-manifest','feature_manifest'),('--implementation-manifest','implementation_manifest')):
                    target = p(flag)
                    raw = read(target)
                    kwargs[name + '_path'] = target
                    kwargs[name + '_sha'] = sha(raw)
                    kwargs[name if name == 'implementation_manifest' else name + '_raw'] = json.loads(raw) if name == 'implementation_manifest' else raw
                for relative in native.FEATURE_GENERATOR_FILES:
                    read(native.ROOT / relative)
                original_validation = validate_independently([kwargs], lambda kw: len(native._feature_packet(**kw)[0]), native.ManualPredictionError)[0]
                repaired_validation = validate_independently([kwargs], lambda kw: len(corrected_feature_packet(native, kw, aliases)[0]), native.ManualPredictionError)[0]
                entry['original_validation'] = original_validation
                entry['repaired_validation'] = repaired_validation
                if original_validation != {'status':'BLOCKED','reason':reason}:
                    raise ValueError('native_failure_not_reproduced')
                manifest = json.loads(kwargs['feature_manifest_raw'])
                timeline = [context['metadata_timestamp'], native._parse_timestamp(manifest['feature_freeze_timestamp'],'freeze'), native._parse_timestamp(manifest['generated_at'],'generated'), datetime.fromisoformat(original['score_step']['started_at']), datetime.fromisoformat(original['score_step']['finished_at']), context['jump_timestamp']]
                entry['feature_to_score_chronology_valid'] = all(x <= y for x,y in zip(timeline,timeline[1:])) and timeline[-2] < timeline[-1]
                form_raw = read(p('--form-csv'))
                def validate_capture(_):
                    native._validate_form_binding(sidecar, form_csv_path=p('--form-csv'), form_raw=form_raw, form_sha=sha(form_raw))
                    if a['race_id'] != context['expected_race_id']:
                        raise native.ManualPredictionError('race_id_sidecar_mismatch')
                    capture_raw = read(p('--capture'))
                    capture = native._select_capture_attempt(capture_raw, jsonl=p('--capture').suffix == '.jsonl', race_id=a['race_id'])
                    capture_rows, fetch, append = native._active_capture_rows(capture, context['runners'])
                    times = [context['metadata_timestamp'], fetch, append, timeline[-3], timeline[-2], timeline[-1]]
                    if not all(x <= y for x,y in zip(times,times[1:])) or not timeline[-2] < timeline[-1]:
                        raise native.ManualPredictionError('source_timestamp_order_invalid')
                    return len(capture_rows)
                entry['form_capture_chronology_validation'] = validate_independently([None], validate_capture, native.ManualPredictionError)[0]
                entry['all_retained_input_gates_pass'] = repaired_validation['status'] == 'VALIDATED' and entry['feature_to_score_chronology_valid'] and entry['form_capture_chronology_validation']['status'] == 'VALIDATED'
                if repaired_validation['status'] == 'VALIDATED':
                    corrections.append({'canonical_race_key':key, 'attempt_reference':path, 'original_reason':reason, 'original_source_grade':context['target_grade'], 'comparison_grade':selected[0]['target_grade_safe'], 'canonical_grade':aliases[str(context['target_grade']).strip().upper()], 'method':'finite_existing_grade_aliases_then_complete_native_feature_validation', 'original_records_modified':False, 'new_forecast_generated':False, 'corrected_validation':repaired_validation, 'feature_to_score_chronology_valid':entry['feature_to_score_chronology_valid'], 'all_retained_input_gates_pass':entry['all_retained_input_gates_pass']})
            elif reason == 'target_venue_invalid':
                entry['original_validation'] = validate_independently([sidecar], lambda s: native._sidecar_context(s)['target_venue'], native.ManualPredictionError)[0]
            elif reason == 'source_timestamp_order_invalid':
                manifest = json.loads(read(p('--feature-manifest')))
                entry['chronology'] = {'metadata':shadow.get('metadata_captured_at'), 'feature_freeze':manifest.get('feature_freeze_timestamp'), 'feature_generated':manifest.get('generated_at'), 'score_started':original['score_step']['started_at'], 'score_finished':original['score_step']['finished_at'], 'jump':native._sidecar_context(sidecar)['jump_timestamp'].isoformat()}
        audit_attempts.append(entry)
        grouped[key].append(entry)
    races = []
    for key, records in sorted(grouped.items()):
        reasons = sorted({r['original_failure_reason'] for r in records if r['original_failure_reason']})
        has_forecast = key in successful
        primary = 'FORECAST_RETAINED' if has_forecast else (reasons[0] if len(reasons)==1 else 'MULTIPLE_CAUSES_REQUIRES_REVIEW')
        races.append({'canonical_race_key':key, 'attempt_records':len(records), 'appended_attempts':sum(r['status']=='APPENDED' for r in records), 'blocked_attempts':sum(r['status']=='BLOCKED' for r in records), 'forecast_retained':has_forecast, 'primary_disposition':primary, 'overlapping_failure_reasons':reasons, 'offline_grade_validation_recovered':any(r.get('repaired_validation',{}).get('status')=='VALIDATED' for r in records), 'offline_grade_validation_chronology_recovered':any(r.get('repaired_validation',{}).get('status')=='VALIDATED' and r.get('feature_to_score_chronology_valid') for r in records), 'all_retained_input_gates_pass_after_grade_repair':any(r.get('all_retained_input_gates_pass') for r in records), 'evidence_references':sorted({r['path'] for r in records}), 'attempt_indexes':[r['attempt_index'] for r in records]})
    failed = [r for r in races if not r['forecast_retained']]
    summary = {'nonreserved_unique_attempted_races':len(races), 'nonreserved_attempt_records':len(attempts), 'appended_attempt_records':sum(r['status']=='APPENDED' for r in attempts), 'blocked_attempt_records':sum(r['status']=='BLOCKED' for r in attempts), 'unique_races_with_retained_forecast':len(successful), 'unique_races_without_forecast':len(failed), 'coverage_fraction':len(successful)/len(races), 'primary_failure_race_counts':dict(collections.Counter(r['primary_disposition'] for r in failed)), 'all_blocked_attempt_reason_counts':dict(collections.Counter(r['original_failure_reason'] for r in audit_attempts if r['status']=='BLOCKED')), 'offline_grade_validation_recovered_unique_races':sum(r['offline_grade_validation_recovered'] for r in failed), 'offline_grade_validation_recovered_attempts':len(corrections), 'recovered_validation_feature_chronology_valid_attempts':sum(r['feature_to_score_chronology_valid'] for r in corrections), 'offline_grade_recovered_chronology_valid_unique_races':sum(r['offline_grade_validation_chronology_recovered'] for r in failed), 'all_retained_input_gates_pass_after_grade_repair_unique_races':sum(r['all_retained_input_gates_pass_after_grade_repair'] for r in failed), 'all_retained_input_gates_pass_after_grade_repair_attempts':sum(r['all_retained_input_gates_pass'] for r in corrections), 'new_forecasts_generated':0, 'original_attempts_modified':0, 'unit_limitation':'Attempt records are residual scorer invocations, not model-output counts or calendar opportunity promises. Two additional appended records are secondary horizons on existing races.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {'failure-summary.json':summary, 'race-accounting.json':races, 'attempt-accounting.json':audit_attempts, 'derived-grade-corrections.json':corrections, 'source-references.json':list(refs.values())}
    for name,value in artifacts.items():
        (args.output_dir/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--access-boundary',type=Path,required=True)
    parser.add_argument('--membership',type=Path,required=True)
    parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--grade-vocabulary',type=Path,default=Path(__file__).with_name('predict_market_form_residual.py'))
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
