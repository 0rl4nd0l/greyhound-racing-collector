"""Fabricated metadata only: no source timings, outcomes or model inputs."""
import pytest

from race_collection.speed_measurement_contract import (
    MetadataRejected,
    check_sectional_metadata,
)


def packet():
    return {
        'schema_version': 'sectional_measurement_metadata_v1',
        'definition': {
            'provider': 'thedogs', 'source_field': '1 SEC',
            'source_key': 'first_sectional_time', 'version': 'fictional-v1',
            'evidence_sha256': 'a' * 64, 'evidence_locator': 'fictional dictionary section 2',
            'clock_subject': 'runner', 'unit': 'second', 'measurement_kind': 'elapsed',
            'measured_or_derived': 'measured', 'clock_origin_id': 'fictional-start-beam',
            'physical_endpoint_id': 'fictional-split-beam',
            'missing_codes_definition': 'fictional missing code dictionary',
            'identity_definition': 'fictional native event-entry-profile binding',
            'publication_revision_definition': 'fictional immutable source revision rule',
            'layout_id': 'fictional-layout', 'layout_era': 'fictional-era-1',
            'race_distance_m': 500,
            'valid_from': '2020-01-01T00:00:00Z',
            'valid_until': '2027-01-01T00:00:00Z',
        },
        'target': {
            'event_id': 'fictional-target', 'runner_profile_id': 'fictional-dog',
            'prediction_cutoff': '2026-10-06T08:00:00Z',
            'jump_at': '2026-10-06T08:01:00Z',
            'layout_id': 'fictional-layout', 'layout_era': 'fictional-era-1',
            'race_distance_m': 500,
        },
        'observations': [
            {
                'event_id': f'fictional-event-{day}',
                'runner_profile_id': 'fictional-dog', 'entry_id': f'fictional-entry-{day}',
                'identity_basis': 'native_event_entry_profile',
                'definition_evidence_sha256': 'a' * 64,
                'identity_evidence_sha256': 'b' * 64,
                'body_sha256': 'c' * 64, 'receipt_sha256': 'd' * 64,
                'layout_id': 'fictional-layout', 'layout_era': 'fictional-era-1',
                'race_distance_m': 500,
                'occurred_at': f'2026-09-{day:02d}T08:00:00Z',
                'published_at': f'2026-09-{day:02d}T08:10:00Z',
                'captured_at': '2026-10-06T07:00:00Z',
                'sealed_at': '2026-10-06T07:01:00Z',
            }
            for day in (1, 8, 15)
        ],
    }


def test_consistent_declarations_do_not_qualify_measurements():
    assert check_sectional_metadata(packet()) == {
        'status': 'METADATA_CONSISTENT_EVIDENCE_UNREVIEWED',
        'measurement_qualified': False,
        'feature_use_authorized': False,
        'declared_distinct_prior_starts': 3,
    }


@pytest.mark.parametrize('field,value', [
    ('clock_subject', 'unknown'), ('clock_subject', 'leader'),
    ('unit', 'unknown'), ('measurement_kind', 'speed'),
    ('physical_endpoint_id', 'UNKNOWN'), ('clock_origin_id', ''),
    ('identity_definition', 'UNQUALIFIED'),
    ('publication_revision_definition', None),
    ('evidence_sha256', ''), ('source_field', 'PIR'),
])
def test_unknown_or_different_measurement_semantics_reject(field, value):
    supplied = packet()
    supplied['definition'][field] = value
    with pytest.raises(MetadataRejected, match='^DEFINITION_UNRESOLVED$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('section,field,value', [
    ('target', 'layout_id', 'another-layout'),
    ('target', 'layout_era', 'another-era'),
    ('target', 'race_distance_m', 450),
    ('observation', 'layout_id', 'another-layout'),
    ('observation', 'layout_era', 'another-era'),
    ('observation', 'race_distance_m', 450),
    ('definition', 'race_distance_m', True),
])
def test_mixed_or_invalid_layout_distance_reject(section, field, value):
    supplied = packet()
    item = supplied['observations'][0] if section == 'observation' else supplied[section]
    item[field] = value
    with pytest.raises(MetadataRejected, match='^COMPARABILITY_MISMATCH$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('field,value', [
    ('identity_basis', 'display_name_date'), ('event_id', ''),
    ('entry_id', 'UNKNOWN'), ('runner_profile_id', 'another-dog'),
    ('definition_evidence_sha256', 'e' * 64),
    ('identity_evidence_sha256', None), ('receipt_sha256', 'not-a-digest'),
    ('event_id', 'fictional-target'),
])
def test_ambiguous_or_mismatched_native_identity_reject(field, value):
    supplied = packet()
    supplied['observations'][0][field] = value
    with pytest.raises(MetadataRejected, match='^IDENTITY_OR_EVIDENCE_BINDING$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('duplicate', ['event_id', 'entry_id'])
def test_duplicate_run_observations_do_not_increase_start_count(duplicate):
    supplied = packet()
    supplied['observations'][1][duplicate] = supplied['observations'][0][duplicate]
    with pytest.raises(MetadataRejected, match='^DUPLICATE_NATIVE_IDENTITY$'):
        check_sectional_metadata(supplied)


def test_three_prior_starts_cannot_be_relaxed():
    supplied = packet()
    supplied['observations'].pop()
    with pytest.raises(MetadataRejected, match='^PRIOR_START_COUNT$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('section,field,value,reason', [
    ('observation', 'published_at', '2026-10-06T08:00:01Z', 'AVAILABILITY_ORDER'),
    ('observation', 'captured_at', '2026-10-06T08:00:01Z', 'AVAILABILITY_ORDER'),
    ('observation', 'sealed_at', '2026-10-06T08:00:00Z', 'AVAILABILITY_ORDER'),
    ('observation', 'occurred_at', '2026-10-06T09:00:00Z', 'AVAILABILITY_ORDER'),
    ('observation', 'published_at', '2026-08-31T08:00:00Z', 'AVAILABILITY_ORDER'),
    ('observation', 'captured_at', '2026-10-06T07:00:00', 'TIMESTAMP_INVALID'),
    ('observation', 'published_at', None, 'TIMESTAMP_INVALID'),
    ('target', 'prediction_cutoff', '2026-10-06T08:02:00Z', 'CUTOFF_AFTER_JUMP'),
    ('definition', 'valid_from', '2026-09-02T00:00:00Z', 'DEFINITION_OUTSIDE_ERA'),
    ('definition', 'valid_until', '2026-10-06T08:00:30Z', 'DEFINITION_OUTSIDE_ERA'),
])
def test_unknown_late_or_inconsistent_source_revision_timing_reject(section, field, value, reason):
    supplied = packet()
    item = supplied['observations'][0] if section == 'observation' else supplied[section]
    item[field] = value
    with pytest.raises(MetadataRejected, match=f'^{reason}$'):
        check_sectional_metadata(supplied)


def test_offsets_compare_as_instants():
    supplied = packet()
    supplied['observations'][0]['sealed_at'] = '2026-10-06T18:01:00+11:00'
    assert check_sectional_metadata(supplied)['measurement_qualified'] is False


@pytest.mark.parametrize('section', ['packet', 'definition', 'target', 'observation'])
def test_extra_values_or_self_asserted_qualification_are_rejected(section):
    supplied = packet()
    item = (supplied if section == 'packet' else supplied['observations'][0]
            if section == 'observation' else supplied[section])
    item['qualified'] = 'private-value-must-not-appear'
    with pytest.raises(MetadataRejected, match='^SCHEMA_INVALID$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('section', ['definition', 'target', 'observations'])
def test_missing_top_level_structure_has_a_fixed_rejection(section):
    supplied = packet()
    del supplied[section]
    with pytest.raises(MetadataRejected, match='^SCHEMA_INVALID$'):
        check_sectional_metadata(supplied)


@pytest.mark.parametrize('supplied', [None, [], {}, {'schema_version': 'unknown'}])
def test_invalid_payload_has_a_fixed_rejection(supplied):
    with pytest.raises(MetadataRejected, match='^SCHEMA_INVALID$'):
        check_sectional_metadata(supplied)
