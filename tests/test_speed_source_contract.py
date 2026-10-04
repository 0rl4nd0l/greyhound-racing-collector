"""Only code, fixed retained definition metadata and fabricated AST inputs."""
import copy
import json
from pathlib import Path

import pytest

from scripts.audit_speed_source_contract import (
    FIELDS, INPUTS, MAX_FILE_BYTES, audit, csv_mappings, definition_status,
    function, literal_get_keys, literal_keys_assigned, read_source,
)

ROOT = Path(__file__).resolve().parents[1]


def test_actual_retained_contract_exposes_adapter_gap_without_qualifying_values():
    result = audit(ROOT)
    assert result['status'] == 'SOURCE_SEMANTICS_NOT_QUALIFIED'
    assert result['fasttrack']['adapter_keys_without_parser_literal_assignment'] == ['comment', 'margin', 'pir', 'run_home', 'sp', 'split1']
    assert result['csv_mappings']['BON']['target_name'] == 'bonus'
    assert result['csv_mappings']['PIR']['aliases'] == ['PIR', 'Performance_Rating']
    assert 'best_first_split' in result['expert_track_distance_metadata_keys']
    assert 'best_first_split_date' not in result['expert_track_distance_metadata_keys']
    assert len(result['definitions']) == 7
    assert all(row['status'] == 'UNQUALIFIED' for row in result['definitions'])
    assert result['current_feed_numeric_coverage'] == 'NOT_MEASURED'
    assert not result['training_or_feature_use_authorized']
    assert sum(result['requests'].values()) == 0


def test_extractor_tracks_a_real_new_assignment_instead_of_hardcoding_gap():
    fn = function('def parse():\n dog_data["split1"] = value\n', 'parse')
    assert literal_keys_assigned(fn, 'dog_data') == ['split1']
    other = function('def parse():\n irrelevant["split1"] = value\n', 'parse')
    assert literal_keys_assigned(other, 'dog_data') == []


@pytest.mark.parametrize('source,extractor,variable', [
    ('def parse():\n dog_data[key] = value\n', literal_keys_assigned, 'dog_data'),
    ('def parse():\n dog_perf.get(key)\n', literal_get_keys, 'dog_perf'),
])
def test_unrecognized_dynamic_keys_fail_closed(source, extractor, variable):
    with pytest.raises(ValueError, match='dynamic_'):
        extractor(function(source, 'parse'), variable)


def test_claimed_definition_boolean_cannot_create_semantic_qualification():
    text = (ROOT / INPUTS['definitions']).read_text()
    data = json.loads(text)
    data['fields'][0]['source_definition_verified'] = True
    data['fields'][0]['qualification'] = 'QUALIFIED'
    data['fields'][0]['missing'] = []
    projected = definition_status(json.dumps(data))
    assert next(row for row in projected if row['field'] == data['fields'][0]['field'])['status'] == 'INDEPENDENT_SOURCE_DEFINITION_REVIEW_REQUIRED'


def test_missing_or_duplicate_definition_is_rejected():
    data = json.loads((ROOT / INPUTS['definitions']).read_text())
    data['fields'][-1] = copy.deepcopy(data['fields'][0])
    with pytest.raises(ValueError, match='definition_field_set_changed'):
        definition_status(json.dumps(data))


def test_mapping_change_requires_review():
    with pytest.raises(ValueError, match='timing_csv_mapping_changed'):
        csv_mappings('x = ColumnMapping(source_name="Split1", target_name="speed")')


def test_source_escape_symlink_and_oversize_rejected(tmp_path):
    (tmp_path / 'oversize.py').write_bytes(b' ' * (MAX_FILE_BYTES + 1))
    (tmp_path / 'link.py').symlink_to(ROOT / INPUTS['parser'])
    for name in ['oversize.py', 'link.py', '../outside.py']:
        with pytest.raises(ValueError):
            read_source(tmp_path, name)


def test_fixed_inputs_are_only_code_or_definition_metadata():
    assert len(INPUTS) == 5
    assert set(INPUTS.values()) == {
        'src/collectors/fasttrack_scraper.py', 'src/collectors/adapters/fasttrack_adapter.py',
        'csv_ingestion.py', 'utils/expert_form_metadata.py',
        'docs/research/prediction_source_definition_matrix_20261003.json',
    }
    assert len(FIELDS) == 7
