from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from src.operator_ui.operational_inputs import validate_config, retained_binding


def test_pending_collector_directory_is_not_created(tmp_path):
    evidence = tmp_path / 'session/collector/evidence'
    retained = tmp_path / 'operational'
    retained.mkdir()
    value = {'evidence_root': str(evidence), 'retained_root': str(retained)}
    assert validate_config(value) == value
    assert not evidence.exists()


def test_config_rejects_symlink_and_unknown_keys(tmp_path):
    (tmp_path/'real').mkdir()
    (tmp_path/'alias').symlink_to(tmp_path/'real')
    with pytest.raises(ValueError):
        validate_config({'evidence_root':str(tmp_path/'alias/pending'), 'retained_root':str(tmp_path/'real')})
    with pytest.raises(ValueError):
        validate_config({'evidence_root':str(tmp_path), 'retained_root':str(tmp_path), 'ignore_identity':True})


def retained(tmp_path):
    from race_collection.prospective_input_retention import REQUIRED_ROLES
    race = {'race_id':'Race 1 - BULLI - 2026-09-29', 'jump_datetime':'2026-09-29T20:30:00+10:00'}
    now = datetime(2026,9,29,10,20,tzinfo=timezone.utc)
    root = tmp_path/'races'/hashlib.sha256(race['race_id'].encode()).hexdigest()/'retention'/'a'/'bundle'
    root.mkdir(parents=True)
    manifest = {'schema_version':'prospective_input_retention_v1','status':'INPUTS_PENDING_COMPLETION',
        'race_id':race['race_id'],'jump_at':race['jump_datetime'], 'prediction_cutoff':(now+timedelta(minutes=9)).isoformat(),
        'source_observed_at':now.isoformat(),'capture_started_at':now.isoformat(),'capture_completed_at':now.isoformat(),
        'files':{k:{} for k in REQUIRED_ROLES}}
    raw=json.dumps(manifest).encode();(root/'manifest.json').write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest()
    (root/'completion.json').write_text(json.dumps({'inputs_sealed_at':now.isoformat(),'status':'INPUTS_RETAINED_NOT_QUALIFIED','manifest_sha256':sha}))
    (root.parent/'terminal.json').write_text(json.dumps({'status':'RETAINED','manifest_sha256':sha}))
    return root,race,now,sha


def test_retained_binding_pins_complete_manifest_and_prejump_identity(tmp_path):
    root,race,now,sha=retained(tmp_path)
    assert retained_binding(tmp_path,race,now)=={'path':str(root),'manifest_sha256':sha}
    with pytest.raises(ValueError): retained_binding(tmp_path,race,now+timedelta(minutes=10))
    raw=json.loads((root/'manifest.json').read_bytes());raw['race_id']='wrong race'
    (root/'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): retained_binding(tmp_path,race,now)


def test_missing_incomplete_and_ambiguous_retention_fail_closed(tmp_path):
    root,race,now,sha=retained(tmp_path)
    (root.parent/'terminal.json').unlink()
    with pytest.raises(ValueError): retained_binding(tmp_path,race,now)
    (root.parent.parent/'b/bundle').mkdir(parents=True)
    with pytest.raises(ValueError): retained_binding(tmp_path,race,now)
    with pytest.raises(ValueError): retained_binding(tmp_path,dict(race,race_id='missing'),now)
