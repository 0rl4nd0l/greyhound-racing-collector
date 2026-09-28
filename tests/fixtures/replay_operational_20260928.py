"""Offline outcome-blind replay of the fixed, authorized September 28 evidence.

Outputs diagnostics only. Original timestamps identify historical evidence;
this does not create new pre-jump forecasts, admissions or collector attempts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from scripts.check_freshness_service import deny_network
deny_network()

def guard(event, args):
    if event == 'sqlite3.connect':
        raise RuntimeError('retained replay forbids database access')
sys.addaudithook(guard)

from bs4 import BeautifulSoup
from upcoming_race_browser import UpcomingRaceBrowser
from src.predictor.market_form_residual import load_frozen_model, score_race, _runner_set_sha256

def read(path):
    return json.loads(path.read_bytes())

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

def replay(root):
    operational = root.parent/'operational-predictions'
    bundles = [
        'prediction_20260928T192942461041+1000_0f76a87b55a0',
        'prediction_20260928T194401485536+1000_ba6c7a42b287',
        'prediction_20260928T195401605153+1000_6806e54dcf85',
    ]
    forecasts = []
    for name in bundles:
        bundle = operational/'bundles'/name
        manifest = read(bundle/'bundle_manifest.json')
        # Hash opaque sealed history without opening the database or rows.
        assert all(sha((bundle/path).read_bytes()) == value['sha256']
                   and (bundle/path).stat().st_size == value['bytes']
                   for path,value in manifest['files'].items())
        original = read(bundle/'result.json')  # Forecast artifact, not race results.
        request = read(bundle/'request.json')
        odds = read(bundle/'odds_receipt.json')
        freeze = read(bundle/'features/sealed/shadow_manifest.json')['feature_freeze_timestamp']
        with zipfile.ZipFile(bundle/'retained_inputs.zip') as archive:
            feature_raw = archive.read('bundle/feature_values.json')
            retained = json.loads(archive.read('bundle/manifest.json'))
            assert sha(feature_raw) == retained['feature_values_sha256']
            features = json.loads(feature_raw)
        model = load_frozen_model(bundle/'model/model.json', bundle/'model/manifest.json')
        assert model.model_sha256 == original['model']['artifact_sha256']
        feature_by_box = {row['box_number']: row for row in features}
        runners = []
        for quote in odds['markets']['win']:
            feature = feature_by_box[quote['box_number']]
            runners.append(dict(race_id=request['race_id'],
                runner_id=f"{request['race_id']}|box:{quote['box_number']}|dog:{quote['identity']}",
                box_number=quote['box_number'], dog_name=feature['dog_name'],
                strict_win_odds=quote['odds_decimal'], features=feature['features'],
                feature_source_sha256=sha(feature_raw),
                odds_source_sha256=sha((bundle/'odds_receipt.json').read_bytes()),
                feature_freeze_timestamp=freeze, odds_capture_timestamp=odds['captured_at']))
        ids = sorted(row['runner_id'] for row in runners)
        scored = score_race(model, runners, dict(expected_runner_ids=ids,
            runner_set_sha256=_runner_set_sha256(ids), race_id=request['race_id'],
            jump_timestamp=request['jump_timestamp'], score_timestamp=original['generated_at']))
        expected = {row['box_number']: row for row in original['prediction']['predictions']}
        ranked = sorted(scored['predictions'], key=lambda row: (-row['full_probability'],row['box_number']))
        maximum = max(abs(row['full_probability']-expected[row['box_number']]['probability']) for row in ranked)
        assert maximum <= 1e-12
        assert all(expected[row['box_number']]['rank'] == rank for rank,row in enumerate(ranked,1))
        forecasts.append(dict(race_id=request['race_id'], bundle=str(bundle),
            model_sha256=model.model_sha256, runners=len(ranked), maximum_probability_difference=maximum,
            ranks_identical=True, all_bundle_hashes_valid=True,
            original_prediction_at=original['generated_at']))

    exclusions = []
    browser = object.__new__(UpcomingRaceBrowser)
    browser.venue_map = {}
    for disposition in read(root/'race-dispositions.json'):
        if disposition['disposition'] != 'UNAVAILABLE_SAFE_METADATA_INCOMPLETE':
            continue
        first = Path(disposition['reports'][0])
        report = read(first)
        selected = next(row for row in report['selected_races'] if row['race_id']==disposition['race_id'])
        download = next(row for row in report['downloads'] if row['race_url']==selected['race_url'])
        normalization = download['result'].get('normalization', {})
        alignment = normalization.get('canonical_runner_alignment', {})
        coverage = next(row for row in report['sidecar_metadata_coverage']['races'] if row['race_url']==selected['race_url'])
        row = dict(race_id=disposition['race_id'], report=str(first), report_sha256=sha(first.read_bytes()),
            missing_fields=disposition['metadata_exclusions'], downloaded=bool(normalization),
            normalization_failure=normalization.get('normalization_failure_reason'),
            native_identity_status=alignment.get('native_identity_status'),
            native_identity_reasons=alignment.get('native_identity_reasons'),
            source_native_race_id=alignment.get('source_native_race_id'),
            weather_rejections=coverage.get('weather_track_rejected_reasons'),
            original_attempted=disposition['attempted'], fully_recovered_from_retained_evidence=False)
        if 'LCTN' in disposition['race_id']:
            for receipt_path in first.parent.glob('odds_capture_refreshed_upcoming/workers/*/source_evidence/primary_race_pages/*.json'):
                receipt = read(receipt_path)
                if receipt['race_discovery_key'] != disposition['race_id']:
                    continue
                assert stamp(receipt['request_end_utc']) < stamp(disposition['jump'])
                raw = (receipt_path.parent/Path(receipt['raw_path']).name).read_bytes()
                assert sha(raw) == receipt['body_sha256']
                # Restrict the parser to the one target header; no mixed page
                # text, runner history or result fields reach diagnostics.
                soup = BeautifulSoup(raw, 'html.parser')
                headers = soup.select('.race-header')
                assert len(headers)==1
                header = headers[0]
                number = header.select_one('.race-box__number').get_text(' ',strip=True)
                grade = header.select_one('.race-header__info__grade').get_text(' ',strip=True)
                assert number == 'R'+str(selected['race_number'])
                projection = BeautifulSoup('<div class="race-header"></div>', 'html.parser')
                for css in ('.race-box__number','.race-header__info__grade'):
                    projection.div.append(header.select_one(css))
                parsed = browser._extract_safe_target_metadata_from_page(projection, receipt['requested_url'],source_sha256=receipt['body_sha256'])
                assert parsed['metadata_is_leakage_safe'] is True
                row['grade_replay'] = dict(receipt=str(receipt_path), body_sha256=sha(raw),
                    retained_at=receipt['request_end_utc'], explicit_header_grade=grade,
                    recovered_grade=parsed['target_grade'], provenance_verified=True)
                break
            assert 'grade_replay' in row
        exclusions.append(row)
    assert len(exclusions)==5
    return dict(schema_version='operational_reliability_retained_replay_v1',
        replayed_at=datetime.now(timezone.utc).isoformat(), forecasts=forecasts, exclusions=exclusions,
        retained_complete_recoveries=0, retained_grade_recoveries=2,
        q_straight_r6='UNATTEMPTED_TARGET_REACHED_NOT_DATA_FAILURE',
        network_denied=True, databases_opened=False, new_predictions_published=False,
        limitation='No missing weather, track condition or rejected native identity evidence was fabricated. Synthetic acquisition coverage is reported separately.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = replay(args.evidence)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), forecasts_reproduced=len(value['forecasts']),
        retained_grade_recoveries=value['retained_grade_recoveries'],retained_complete_recoveries=0)))
