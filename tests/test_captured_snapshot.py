"""Exact native publications, with no providers or scoring."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_collection import synchronous_manual_capture as capture
from race_collection.captured_snapshot import classify_superseded_capture
from race_collection.manual_prediction_collector_request import runner_set_sha256
from src.predictor.on_demand import canonical_bytes, sha256_bytes
from tests.race_collection.test_synchronous_manual_capture import _runner_coverage


def snapshots(tmp_path, *, changed_runner=False):
    root = tmp_path / 'evidence'
    state = root / 'runtime/state.json'
    at = datetime.fromisoformat('2026-07-19T12:55:00+10:00')
    url = 'https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5'
    race = dict(date='2026-07-19', jump_datetime='2026-07-19T13:00:00+10:00',
        race_id='Race 5 - GUNN - 2026-07-19', race_number=5, race_time='13:00',
        race_id_aliases=['Race 5 - GUNN - 2026-07-19', 'Race 5 - GUNNEDAH - 2026-07-19'],
        source_native_race_id='15900', race_url=url, venue='GUNN')
    packets = []
    for number in range(2):
        phase = root / f'phase-{number}'
        coverage = _runner_coverage(phase, url, at)
        if changed_runner and number:
            sidecar = Path(coverage['races'][0]['sidecar_path'])
            payload = json.loads(sidecar.read_bytes())
            payload['prejump_shadow_metadata']['runner_box_name_list'][0]['source_native_runner_id'] = '159099'
            payload['runner_completeness_after_canonical_alignment']['participants'][0]['source_native_runner_id'] = '159099'
            sidecar.write_bytes(canonical_bytes(payload))
        source = dict(status='SUCCESS', generated_at=(at+timedelta(seconds=number)).isoformat(),
            sidecar_metadata_coverage=coverage, selected_count=1, selected_races=[race])
        report = phase/'refresh.json'
        report.write_bytes(canonical_bytes(source))
        publication = capture.publish_current_race_index(state_path=state, evidence_root=root,
            source_refresh_report_path=report, run_id=f'phase-{number}')
        assert publication['status'] == 'PUBLISHED', publication
        publication_path = phase/'current_race_index_publish.json'
        publication_path.write_bytes(canonical_bytes(publication))
        capture.publish_current_race_index_lifecycle(state_path=state, evidence_root=root,
            publication_report_path=publication_path, publication=publication)
        packets.append(json.loads(capture.current_race_index_path(state).read_bytes()))
    old = packets[0]['races'][0]
    files = old['runner_source']
    claim = {'item': {'race_id':race['race_id'], 'packet_sha256':sha256_bytes(canonical_bytes(packets[0])),
        'race_identity':{key:old[key] for key in ('race_id','race_url','jump_datetime','source_native_race_id','runner_set_sha256')},
        'input_files':{str(root/files['csv_path']):files['csv_sha256'], str(root/files['sidecar_path']):files['sidecar_sha256']},
        'capture_runner_set_sha256':runner_set_sha256([{'box_number':row['box'], 'dog_name':row['display_name'], 'identity':row['identity']} for row in old['runners']])}}
    current = at+timedelta(seconds=2)
    view = capture.bounded_current_race_index(current_time=current, timeout_seconds=5,
        index_path=capture.current_race_index_path(state), evidence_root=root,
        max_age_seconds=300, return_verified_view=True)
    return root, claim, view, current


@pytest.mark.parametrize('changed_runner', [False, True])
def test_only_verified_snapshots_can_be_race_exclusions(tmp_path, changed_runner):
    root, claim, view, current = snapshots(tmp_path, changed_runner=changed_runner)
    proof = classify_superseded_capture(claim, view, evidence_root=root, current_time=current)
    assert proof['code'] == ('CAPTURE_RUNNERS_CHANGED' if changed_runner else 'CAPTURE_SNAPSHOT_SUPERSEDED')
    assert proof['same_roster'] is not changed_runner
    assert proof['job_created'] is False and proof['comparison_admitted'] is False
    assert proof['captured_runner_set_sha256'] != proof['current_runner_set_sha256']


@pytest.mark.parametrize('mutation', ['publication', 'report', 'csv', 'symlink', 'claim_packet', 'claim_runner', 'claim_input', 'stale'])
def test_unproven_or_unsafe_supersession_stays_terminal(tmp_path, mutation):
    root, claim, view, current = snapshots(tmp_path)
    if mutation == 'publication':
        path = root/'phase-0/current_race_index_publish.json'
        data = json.loads(path.read_bytes()); data['status'] = 'REJECTED'
        path.write_bytes(canonical_bytes(data))
    elif mutation == 'report':
        path = root/'phase-0/refresh.json'
        path.write_bytes(path.read_bytes()+b' ')
    elif mutation in {'csv', 'symlink'}:
        path = Path(next(iter(claim['item']['input_files'])))
        if mutation == 'csv':
            path.write_bytes(path.read_bytes().replace(b'Alpha', b'Changed'))
        else:
            saved = path.with_suffix('.saved'); path.rename(saved); path.symlink_to(saved)
    elif mutation == 'claim_packet':
        claim['item']['packet_sha256'] = 'a'*64
    elif mutation == 'claim_runner':
        claim['item']['race_identity']['runner_set_sha256'] = 'a'*64
    elif mutation == 'claim_input':
        claim['item']['input_files'][next(iter(claim['item']['input_files']))] = 'a'*64
    else:
        current += timedelta(seconds=301)
    with pytest.raises((ValueError, capture.CaptureOneRejected)):
        classify_superseded_capture(claim, view, evidence_root=root, current_time=current)
