"""Pure consistency checks; supplied declarations never qualify measurements.

This module does not read evidence, timing values, results or the network. A
successful check means only that supplied metadata is internally consistent.
Independent review must establish the truth and source scope of every claim.
"""
from datetime import datetime
import re


_SCOPE = {'layout_id', 'layout_era', 'race_distance_m'}
_DEFINITION = _SCOPE | {
    'provider', 'source_field', 'source_key', 'version', 'evidence_sha256', 'evidence_locator',
    'clock_subject', 'unit', 'measurement_kind', 'measured_or_derived', 'clock_origin_id',
    'physical_endpoint_id', 'missing_codes_definition', 'identity_definition',
    'publication_revision_definition', 'valid_from', 'valid_until',
}
_TARGET = _SCOPE | {'event_id', 'runner_profile_id', 'prediction_cutoff', 'jump_at'}
_OBSERVATION = _SCOPE | {
    'event_id', 'runner_profile_id', 'entry_id', 'identity_basis',
    'definition_evidence_sha256', 'identity_evidence_sha256', 'body_sha256', 'receipt_sha256',
    'occurred_at', 'published_at', 'captured_at', 'sealed_at',
}


def _require(condition, category):
    if not condition:
        raise MetadataRejected(category)


def _shape(value, keys):
    _require(isinstance(value, dict) and set(value) == keys, 'SCHEMA_INVALID')


def _known(value):
    return (isinstance(value, str) and bool(value.strip())
            and value.strip().upper() not in {'UNKNOWN', 'UNQUALIFIED', 'N/A', 'NONE', 'NULL', '-'})


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def _time(value):
    _require(isinstance(value, str), 'TIMESTAMP_INVALID')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise MetadataRejected('TIMESTAMP_INVALID') from None
    _require(parsed.utcoffset() is not None, 'TIMESTAMP_INVALID')
    return parsed


class MetadataRejected(ValueError):
    """A fixed category without echoed source values."""


def check_sectional_metadata(packet):
    """Check one runner's declared comparable prior starts; never authorize use."""
    _shape(packet, {'schema_version', 'definition', 'target', 'observations'})
    _require(packet['schema_version'] == 'sectional_measurement_metadata_v1', 'SCHEMA_INVALID')
    _shape(packet['definition'], _DEFINITION)
    _shape(packet['target'], _TARGET)
    _require(isinstance(packet['observations'], list), 'SCHEMA_INVALID')
    _require(3 <= len(packet['observations']) <= 100, 'PRIOR_START_COUNT')
    for observation in packet['observations']:
        _shape(observation, _OBSERVATION)
    definition = packet['definition']
    fixed = {
        'provider': 'thedogs', 'source_field': '1 SEC',
        'source_key': 'first_sectional_time', 'clock_subject': 'runner',
        'unit': 'second', 'measurement_kind': 'elapsed',
    }
    _require(all(definition.get(key) == value for key, value in fixed.items())
             and definition.get('measured_or_derived') in ('measured', 'derived')
             and all(_known(definition.get(key)) for key in (
                 'version', 'evidence_locator', 'clock_origin_id', 'physical_endpoint_id',
                 'missing_codes_definition', 'identity_definition', 'publication_revision_definition'))
             and _sha(definition.get('evidence_sha256')), 'DEFINITION_UNRESOLVED')
    target = packet['target']
    observations = packet['observations']
    for item in [definition, target, *observations]:
        _require(_known(item.get('layout_id')) and _known(item.get('layout_era'))
                 and type(item.get('race_distance_m')) is int and item['race_distance_m'] > 0
                 and all(item.get(key) == definition.get(key)
                         for key in ('layout_id', 'layout_era', 'race_distance_m')),
                 'COMPARABILITY_MISMATCH')
    _require(_known(target.get('runner_profile_id')) and _known(target.get('event_id')),
             'IDENTITY_OR_EVIDENCE_BINDING')
    cutoff, jump = _time(target.get('prediction_cutoff')), _time(target.get('jump_at'))
    _require(cutoff <= jump, 'CUTOFF_AFTER_JUMP')
    valid_from, valid_until = _time(definition.get('valid_from')), _time(definition.get('valid_until'))
    _require(valid_from <= jump < valid_until, 'DEFINITION_OUTSIDE_ERA')
    events, entries = set(), set()
    for observation in observations:
        _require(observation.get('identity_basis') == 'native_event_entry_profile'
                 and _known(observation.get('event_id')) and _known(observation.get('entry_id'))
                 and observation.get('runner_profile_id') == target['runner_profile_id']
                 and observation['event_id'] != target['event_id']
                 and observation.get('definition_evidence_sha256') == definition['evidence_sha256']
                 and all(_sha(observation.get(key)) for key in (
                     'identity_evidence_sha256', 'body_sha256', 'receipt_sha256')),
                 'IDENTITY_OR_EVIDENCE_BINDING')
        _require(observation['event_id'] not in events and observation['entry_id'] not in entries,
                 'DUPLICATE_NATIVE_IDENTITY')
        events.add(observation['event_id'])
        entries.add(observation['entry_id'])
        occurred, published, captured, sealed = (
            _time(observation.get(key))
            for key in ('occurred_at', 'published_at', 'captured_at', 'sealed_at')
        )
        _require(occurred <= published <= captured <= sealed < cutoff, 'AVAILABILITY_ORDER')
        _require(valid_from <= occurred < valid_until, 'DEFINITION_OUTSIDE_ERA')
    return {
        'status': 'METADATA_CONSISTENT_EVIDENCE_UNREVIEWED',
        'measurement_qualified': False,
        'feature_use_authorized': False,
        'declared_distinct_prior_starts': len(packet['observations']),
    }
