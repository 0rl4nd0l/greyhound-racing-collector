"""Fabricated stage failures preserve consumed work without a success summary."""
from copy import deepcopy
import json
import sys

import pytest

from scripts import run_sectional_speed_experiment as run
from tests.test_sectional_speed_evaluation import protocol, race, ref


def setup_execution(tmp_path, monkeypatch, *, record_failure=False):
    rows = [race(1), race(2), race(3)]
    frozen = protocol(rows)
    scope = {'protocol': ref('protocol'), 'raw_cell_checks': [ref('raw-cells')],
        'expected_race_ids': [r['race_id'] for r in rows], 'allowed_files': {},
        'limits': {'max_output_bytes': 16 * 1024 * 1024}}
    scope_path = tmp_path / 'scope.json'
    scope_path.write_text(json.dumps(scope))
    output_path = tmp_path / 'execution'
    class Reader:
        reads = 0
        bytes = 0

        def json(self, reference):
            self.reads += 1
            return deepcopy(frozen if reference == ref('protocol') else {'status': 'VERIFIED'})

        def check(self):
            pass

    monkeypatch.setattr(run, 'CheckedReader', lambda allowed, limits: Reader())

    def load_records(scope, reader, supplied_protocol, output, *, on_label_decoded, progress_sink):
        for row in rows:
            for _ in row['runner_ids']:
                on_label_decoded(row['race_id'])
            if record_failure:
                raise ValueError('FABRICATED_AFTER_FIRST_RACE_LABELS')
        progress_sink({'kind': 'LABEL_LOADING_COMPLETED', 'label_rows': 6, 'label_races': 3})
        proof = output.put('baseline-reproduction.json', {'status': 'FABRICATED_VALIDATED_BINDING'})
        progress_sink({'kind': 'BASELINE_REPRODUCTION_COMPLETED', 'proof': proof,
            'independent_original_races_checked': 1, 'independent_original_runner_forecasts_checked': 2,
            'fixed_baseline_generated_races': 3})
        return deepcopy(rows), proof

    monkeypatch.setattr(run, '_development_records', load_records)
    monkeypatch.setattr(sys, 'argv', ['run_sectional_speed_experiment', '--execute', '--scope', str(scope_path),
                                    '--stage', 'evaluation', '--output', str(output_path)])
    return rows, output_path


def test_final_oracle_failure_preserves_all_fit_and_prediction_records(tmp_path, monkeypatch):
    rows, output_path = setup_execution(tmp_path, monkeypatch)
    verified = []

    def fail_final(*args):
        if len(verified) == 2:
            raise ValueError('FABRICATED_FINAL_ORACLE_FAILURE')
        verified.append(True)

    monkeypatch.setattr(run.oracle, 'verify_adjustment', fail_final)
    with pytest.raises(ValueError, match='FABRICATED_FINAL_ORACLE_FAILURE'):
        run.main()
    assert not (output_path / 'summary.json').exists()
    assert not (output_path / 'evaluation.private.json').exists()
    provisional = json.loads((output_path / 'evaluation.provisional.private.json').read_text())
    assert provisional['status'] == 'COMPUTED_PENDING_PROBABILITY_ORACLE'
    assert len(provisional['fit_trials']) == 5
    assert len(provisional['records']) == 3
    failed = json.loads((output_path / 'FAILED.json').read_text())
    assert failed['status'] == 'FAILED_NO_SUCCESS'
    assert failed['completed'] == [rows[0]['race_id'], rows[1]['race_id']]
    assert failed['active'] == rows[2]['race_id']
    assert failed['unattempted'] == []
    assert failed['details']['label_rows_decoded'] == 6
    assert failed['details']['label_exposed_races'] == 3
    assert len(failed['details']['fit_trials_started']) == 5
    assert len(failed['details']['fit_trials_completed']) == 5
    assert failed['details']['prediction_completed_races'] == 3
    assert failed['details']['probability_oracle_checked_races'] == 2
    progress = [json.loads(path.read_text()) for path in sorted(output_path.glob('evaluation-progress-*.private.json'))]
    assert len([event for event in progress if event['kind'] == 'FIT_TRIAL_COMPLETED']) == 5
    predictions = [event['record'] for event in progress if event['kind'] == 'PREDICTION_COMPLETED']
    assert predictions == provisional['records']
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output_path.iterdir())


def test_input_failure_reports_only_actual_label_exposure_as_attempted(tmp_path, monkeypatch):
    rows, output_path = setup_execution(tmp_path, monkeypatch, record_failure=True)
    with pytest.raises(ValueError, match='FABRICATED_AFTER_FIRST_RACE_LABELS'):
        run.main()
    failed = json.loads((output_path / 'FAILED.json').read_text())
    assert failed['completed'] == []
    assert failed['active'] == rows[0]['race_id']
    assert failed['unattempted'] == [rows[1]['race_id'], rows[2]['race_id']]
    assert failed['details']['label_rows_decoded'] == 2
    assert failed['details']['label_exposed_races'] == 1
    assert failed['details']['fit_trials_started'] == []
    assert not (output_path / 'summary.json').exists()


def test_success_summary_is_last_and_distinguishes_original_replay_from_new_baseline_generation(tmp_path, monkeypatch):
    rows, output_path = setup_execution(tmp_path, monkeypatch)
    assert run.main() == 0
    summary = json.loads((output_path / 'summary.json').read_text())
    assert summary['status'] == 'COMPLETE_RETROSPECTIVE_EXPLORATORY_EVALUATION'
    assert summary['execution_accounting']['unattempted'] == []
    assert summary['execution_accounting']['completed'] == [r['race_id'] for r in rows]
    assert summary['baseline_verification_scope'] == {
        'independent_original_races_checked': 1,
        'independent_original_runner_forecasts_checked': 2,
        'fixed_baseline_generated_races': 3,
        'later_races_generated_with_same_fixed_model_not_independently_replayed': 2}
    assert summary['accounting']['supplied_baseline_vectors_checked_races'] == 3
    assert not (output_path / 'FAILED.json').exists()
    assert (output_path / 'evaluation.provisional.private.json').exists()
    assert (output_path / 'evaluation.private.json').exists()


def test_label_loader_reports_actual_decodes_even_when_later_validation_fails():
    lines, allocated = [], set()
    for number in range(1, 332):
        day = '2026-06-10' if number <= 154 else '2026-06-24'
        rid = f'Race {number} - TEST - {day}'
        if number > 154:
            allocated.add(rid)
        for box in range(1, 9 if number <= 43 else 8):
            label = 'DO_NOT_DECODE' if number <= 154 else 'null' if number == 331 else '0'
            lines.append('{"race_id":' + json.dumps(rid) + ',"race_date":' + json.dumps(day)
                + ',"box":' + str(box) + ',"dog_token":"DOG' + str(box)
                + '","odds":2.0,"y":' + label + '}')
    class Reader:
        def json(self, reference):
            return {'records': {'2026-07-15|RESERVED|1': 'protected'}}

        def read(self, reference):
            return '\n'.join(lines).encode()

    exposed = []
    with pytest.raises(ValueError, match='DEVELOPMENT_LABEL_OR_ODDS_INVALID'):
        run.development.evaluation_rows({'protected': ref('protected'), 'development': ref('development')},
            Reader(), allocated, on_label_decoded=exposed.append)
    assert len(exposed) == 177 * 7
    assert set(exposed) == allocated
