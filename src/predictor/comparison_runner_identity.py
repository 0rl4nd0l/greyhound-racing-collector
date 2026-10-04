"""Derive result identity only from the already verified pre-jump bundle."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

from scripts.capture_thedogs_market_history import (
    _stored_response, validate_primary_native_identity_evidence,
)
from src.predictor.comparison_result_runtime import check_active_deadline
from src.predictor.retained_inputs import _read
from utils.runner_completeness import extract_canonical_runner_set_from_html, normalise_runner_name
from utils.thedogs_runner_identity import extract_thedogs_profile_identity, TheDogsProfileIdentityMissing


def frozen_result_participants(job, bundle, prediction_bundles):
    """Preserve old exact-name matching; enrich only authenticated profile proof.

    Callers must first verify the producer's bundle and its binding to the job.
    A dog profile ID and a race-entry ID are different namespaces. The retained
    primary race page provides the same-row bridge between them.
    """
    check_active_deadline()
    participants = [dict(box_number=r['box'], dog_name=r['name'],
                         source_native_runner_id=r.get('source_native_runner_id'))
                    for r in job.input.ordered_runners]
    paths = [name for name in bundle.manifest['files']
             if name.startswith('source/') and name.endswith('.metadata.json')]
    if not paths:
        return participants
    if len(paths) != 1:
        raise ValueError('FROZEN_RUNNER_IDENTITY_METADATA_AMBIGUOUS')
    name = paths[0]
    raw = _read(Path(prediction_bundles), str(Path(bundle.directory) / name))
    entry = bundle.manifest['files'][name]
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('FROZEN_RUNNER_IDENTITY_METADATA_CHANGED')
    metadata = json.loads(raw)
    evidence = metadata.get('native_identity_evidence')
    if evidence is None:
        return participants
    race = bundle.result['race']
    source_url = metadata.get('race_url', race['url'])
    if source_url not in {race['url'], race['url'] + '?trial=false'}:
        raise ValueError('FROZEN_RUNNER_IDENTITY_SOURCE_MISMATCH')
    valid, _ = validate_primary_native_identity_evidence(
        evidence, expected_race_url=source_url,
        expected_native_race_id=metadata.get('source_native_race_id'),
        expected_active_runner_boxes=[(r['source_native_runner_id'], r['box_number']) for r in participants],
        metadata_captured_at=metadata.get('metadata_captured_at'))
    check_active_deadline()
    if not valid or datetime.fromisoformat(evidence['jump_timestamp']) != datetime.fromisoformat(job.input.jump_timestamp):
        raise ValueError('FROZEN_RUNNER_IDENTITY_PROVENANCE_INVALID')
    page = _stored_response(evidence['race_page_http'], field='race_page_http',
                            exact_url=source_url, content_type_prefix='text/html', require_body=True)
    markup = page.body.decode('utf-8', errors='strict')
    canonical = extract_canonical_runner_set_from_html(markup, source_url=race['url'],
        expected_race_number=race['race_number'], extraction_timestamp=page.request_end_utc.isoformat())
    active = canonical['final_runner_participants']
    frozen = {(r['box_number'], normalise_runner_name(r['dog_name']), r['source_native_runner_id']) for r in participants}
    parsed = {(r['box_number'], normalise_runner_name(r['dog_name']), r.get('source_native_runner_id')) for r in active}
    if canonical['canonical_runner_set_status'] != 'available' or parsed != frozen or len(active) != len(participants):
        raise ValueError('FROZEN_RUNNER_IDENTITY_FIELD_MISMATCH')
    rows_by_entry = {}
    for row in BeautifulSoup(markup, 'html.parser').select('tr.race-runner'):
        ids = {str(element.get('data-runner-id') or '').strip() for element in row.select('[data-runner-id]')}
        if len(ids) != 1:
            continue
        native = next(iter(ids))
        if native in rows_by_entry:
            raise ValueError('FROZEN_RUNNER_IDENTITY_ENTRY_DUPLICATE')
        rows_by_entry[native] = row
    by_box = {r['box_number']: r for r in active}
    enriched = []
    for participant in participants:
        check_active_deadline()
        row = rows_by_entry.get(participant['source_native_runner_id'])
        if row is None:
            raise ValueError('FROZEN_RUNNER_IDENTITY_ENTRY_MISSING')
        try:
            dog = extract_thedogs_profile_identity(row, require_profile_link=False)
        except TheDogsProfileIdentityMissing:
            # Legacy pages without a corroborated profile bridge remain subject
            # to the existing exact-name rule and cannot authorize a remapping.
            enriched.append(participant)
            continue
        member = {**participant, 'source_native_dog_id': dog}
        original = by_box[participant['box_number']].get('original_box_number')
        if original is not None:
            member['original_box_number'] = original
        # This projection is created only after the sealed metadata, transport
        # chain, canonical field and same-row entry/profile bridge pass above.
        member['native_identity_proof'] = {
            'schema_version': 'verified_prejump_runner_bridge_v1',
            'metadata_sha256': entry['sha256'],
            'native_evidence_sha256': evidence['evidence_sha256'],
            'race_page_body_sha256': evidence['race_page_http']['body_sha256'],
            'race_url': race['url'],
            **member,
        }
        enriched.append(member)
    profiles = [r['source_native_dog_id'] for r in enriched if r.get('source_native_dog_id') is not None]
    if len(set(profiles)) != len(profiles):
        raise ValueError('FROZEN_RUNNER_IDENTITY_DOG_DUPLICATE')
    check_active_deadline()
    return enriched
