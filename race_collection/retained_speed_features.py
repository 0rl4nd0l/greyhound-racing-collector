"""Private historical features from the exact retained 82-member input graph.

The prior presence-only experiment and its authority remain unchanged. This
adapter reuses its authentication and parser, under a new feature authority.
"""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path

from race_collection import historical_speed_features as features
from race_collection import retained_card_timing_coverage as coverage

MANIFEST_PATH = '/home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/manifest.PROPOSED.json'
MANIFEST_SHA = '922456735d4117092fbec50333d68bbb8743008fc2c73b1fc8efa48529148223'
MEMBERSHIP_SHA = '782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342'
MEMBERS, RUNNER_SLOTS = 82, 583
POLICY = 'RECENT_THREE_DISTINCT_PRIOR_DATE_RAW_CONTEXT_FIRST_SECTIONAL_V1'
HEADER = ['Dog Name', 'Sex', 'PLC', 'BOX', 'WGT', 'DIST', 'DATE', 'TRACK', 'G',
          'TIME', 'WIN', 'BON', '1 SEC', 'MGN', 'W/2G', 'PIR', 'SP']
ASSUMPTIONS = {
    'measurement_meanings': 'USER_SUPPLIED_WORKING_MEANINGS_NOT_NEW_PROVIDER_CERTIFICATION',
    'first_sectional': 'TheDogs 1 SEC is the individual runner first sectional in seconds',
    'total_time': 'TheDogs TIME is the individual runner total time; unused by this construction',
    'target_track_key': 'LITERAL_RETAINED_ADMISSION_VENUE',
    'history_track_key': 'LITERAL_THEDOGS_CSV_TRACK_AFTER_WHITESPACE_TRIM',
    'track_match': 'EXACT_STRING_EQUALITY_NO_ALIAS_OR_FUZZY_JOIN',
    'distance_match': 'EXACT_POSITIVE_INTEGER_METRES_WITH_OPTIONAL_LITERAL_m_SUFFIX',
    'layout_and_clock': 'SAME_RAW_CONTEXT_STABILITY_ASSUMED_NOT_PROVIDER_CERTIFIED',
    'history_event_identity': 'DISTINCT_PRIOR_DATE_PROXY_NOT_NATIVE_EVENT_ID',
    'runner_identity': 'RACE_BOX_AND_EXISTING_CANONICAL_ROSTER_TOKEN_NOT_CROSS_CARD_JOIN',
    'availability': 'ORIGINAL_PUBLISHED_COMPLETE_TIME_IS_CONSERVATIVE_RETAINED_AVAILABILITY_UPPER_BOUND',
    'same_day_history': 'EXCLUDED_DATE_ONLY_SOURCE_CANNOT_ORDER_WITHIN_DAY',
    'gap': 'WITHIN_SUPPORTED_FIELD_MEDIAN_RELATIVE_SECONDS_NOT_PHYSICAL_SPEED_OR_PREDICTIVE_ADVANTAGE',
}
require = coverage.require


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def construct_member(reader, member, original):
    # Keep authentication and the existing native roster parser unchanged.
    coverage.verified_member(reader, member, original)
    payload = reader.read(member['accepted_csv'])
    sidecar = reader.json(member['sidecar'])
    admission = reader.json(member['admission'])
    receipt = reader.json(member['primary_receipt'])
    text = payload.decode('utf-8-sig', errors='strict')
    first = text.splitlines()[0]
    delimiter = '|' if first.count('|') > first.count(',') else ','
    header = next(csv.reader(io.StringIO(text), delimiter=delimiter))
    require(header == HEADER and member['csv_header'] == HEADER, 'UNSUPPORTED_RETAINED_COLUMNS')
    # This fixed source has no layout columns. Never silently discard a known
    # layout/clock declaration if the upstream evidence contract changes.
    require(not any(sidecar.get(key) for key in ('layout_id', 'layout_era', 'track_layout',
        'target_layout_id', 'target_layout_era', 'clock_convention', 'sectional_endpoint')),
        'SOURCE_CONTEXT_METADATA_REQUIRES_EXPLICIT_BINDING')
    cutoff = coverage.instant(admission['decision_at'])
    require(coverage.instant(original['original_admitted_at']) <= cutoff < coverage.instant(member['jump_at'])
        and admission['admitted_at'] == original['original_admitted_at']
        and coverage.instant(receipt['capture_timestamp']) < cutoff
        and coverage.instant(original['original_published_complete_at']) < cutoff,
        'FEATURE_CAPTURE_NOT_BEFORE_CUTOFF')
    projected = coverage.projected_card(payload)
    roster = coverage.parse_card_target_roster_bytes(projected, source='authenticated retained card')
    blocks = coverage.parse_form_blocks_bytes(projected, source='authenticated retained card')
    ids = {name: coverage.digest([member['race_id'], box, name]) for box, name in roster}
    packet = {'target': {'race_id': member['race_id'], 'date': member['source_race_date'],
            'source_track': member['venue'], 'distance_m': coverage.distance(member['target_distance_raw']),
            'cutoff': cutoff.isoformat()},
        'captured_at': original['original_published_complete_at'],
        'roster': [ids[name] for _, name in roster],
        'histories': {ids[name]: [{'date': row['DATE'].strip(), 'source_track': row['TRACK'].strip(),
            'distance_m': coverage.distance(row['DIST']), 'first_sectional': row['1 SEC'],
            'observation_fingerprint': coverage.digest({key: row[key].strip()
                for key in coverage.COLUMNS if key != 'Dog Name'})}
            for row in blocks[name]] for _, name in roster}}
    result = features.build_speed_features(packet)
    return {'race_id': member['race_id'], 'runner_set_sha256': member['runner_set_sha256'],
        'source': {role: member[role] for role in coverage.ROLES},
        'cutoff': cutoff.isoformat(), 'retained_available_by': packet['captured_at'],
        'context': packet['target'], 'features': result}


def _write(path, data):
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run(reference, output):
    require(reference == {'path': MANIFEST_PATH, 'sha256': MANIFEST_SHA}, 'FEATURE_MANIFEST_NOT_FIXED')
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve() == output.parent, 'OUTPUT_UNSAFE')
    output.mkdir(mode=0o700, exist_ok=False)
    reader = coverage.Reader()
    records, members = [], []
    active = None
    try:
        manifest = reader.json(reference)
        require(manifest['schema_version'] == 'retained_speed_breadth_metadata_manifest_v1'
            and manifest['selection'] == 'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS'
            and manifest['membership']['sha256'] == MEMBERSHIP_SHA, 'FEATURE_MANIFEST_INVALID')
        membership = reader.json(manifest['membership'])
        members = manifest['members']
        originals = {m['race_id']: m for m in membership['members']}
        require(len(members) == len(membership['members']) == len(originals) == MEMBERS
            and len({m['race_id'] for m in members}) == MEMBERS
            and set(originals) == {m['race_id'] for m in members}
            and sum(m['target_runner_slots'] for m in members) == RUNNER_SLOTS, 'FEATURE_DENOMINATOR_CHANGED')
        # Every consumed path is fixed by the authenticated manifest. Output
        # cannot alias any original, even if a root forgot a protected directory.
        inputs = [Path(manifest['membership']['path']), *[Path(m[role]['path']) for m in members for role in coverage.ROLES]]
        require(all(not (out == p or out.is_relative_to(p) or p.is_relative_to(out))
            for out in (output,) for p in inputs), 'FEATURE_INPUT_OUTPUT_OVERLAP')
        for member in members:
            reader.check()
            active = member['race_id']
            records.append(construct_member(reader, member, originals[active]))
            active = None
        reader.check()
        rows = [row for record in records for row in record['features']['runners']]
        require(len(rows) == RUNNER_SLOTS, 'FEATURE_ROSTER_CHANGED')
        dispositions = Counter(row['status'] for row in rows)
        exclusions = Counter()
        for row in rows:
            exclusions.update(row['exclusions'])
        supported = sum(record['features']['supported_runner_count'] for record in records)
        summary = {'schema_version': 'retained_speed_feature_summary_v1',
            'status': 'COMPLETE_PRIVATE_HISTORICAL_SPEED_FEATURES', 'policy': POLICY,
            'manifest': reference, 'cards': len(records), 'target_runner_slots': len(rows),
            'supported_runner_slots': supported,
            'full_field_supported_cards': sum(record['features']['full_field_supported'] for record in records),
            'runner_status_counts': dict(dispositions), 'row_exclusion_counts': dict(exclusions),
            'field_blocker_counts': dict(Counter(record['features']['field_blocker'] for record in records
                if record['features']['field_blocker'] is not None)),
            'records': [{'race_id': record['race_id'], 'runner_set_sha256': record['runner_set_sha256'],
                'runner_count': record['features']['runner_count'],
                'supported_runner_count': record['features']['supported_runner_count'],
                'full_field_supported': record['features']['full_field_supported'],
                'field_blocker': record['features']['field_blocker']} for record in records],
            'assumptions': ASSUMPTIONS, 'reads': reader.reads, 'read_bytes': reader.bytes,
            'provider_requests': 0, 'result_requests': 0, 'training': False,
            'performance_evaluation': False, 'model_changes': False}
        private = encoded({'schema_version': 'retained_speed_feature_values_v1', 'policy': POLICY,
            'manifest': reference, 'assumptions': ASSUMPTIONS, 'records': records})
        summary['private_features_sha256'] = hashlib.sha256(private).hexdigest()
        public = encoded(summary)
        require(len(private)+len(public)+65536 <= coverage.MAX_OUTPUT, 'FEATURE_OUTPUT_LIMIT')
        _write(output/'.features.pending.json', private)
        _write(output/'.summary.pending.json', public)
        reader.check()
        os.link(output/'.features.pending.json', output/'features.private.json')
        (output/'.features.pending.json').unlink()
        reader.check()
        # The directory was created exclusively by this invocation. Publish the
        # successful summary last, only after every member and private write.
        os.rename(output/'.summary.pending.json', output/'summary.json')
        return {k: summary[k] for k in ('status', 'cards', 'target_runner_slots', 'supported_runner_slots',
            'full_field_supported_cards', 'private_features_sha256', 'provider_requests', 'result_requests')}
    except Exception as exc:
        category = str(exc) if isinstance(exc, (coverage.CoverageRejected, features.FeatureRejected)) else 'UNCLASSIFIED_INPUT_FAILURE'
        attempted = len(records)+(active is not None)
        _write(output/'FAILED.json', encoded({'status': 'FAILED_NO_SUCCESSFUL_SPEED_FEATURES',
            'reason': category, 'denominator': MEMBERS, 'completed_race_ids': [r['race_id'] for r in records],
            'failed_race_id': active, 'unattempted_race_ids': [m['race_id'] for m in members[attempted:]],
            'members_not_yet_loaded': MEMBERS if not members else 0}))
        raise
